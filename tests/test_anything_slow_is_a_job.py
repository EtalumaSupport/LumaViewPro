# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Anything slow is a job, for every call alike.

A call runs on a thread of its own and its route waits what the client
says it will wait (``Prefer: wait``, 2 s by default, at most 60 s, the
wait applied named in ``Preference-Applied``). A call still running then
is ``202 Accepted`` with the job's address, and the job, read with its
own ``Prefer: wait``, ends ``completed`` with the result a 200 would have
carried or ``failed`` with the problem; reading it is 200 either way. A
member's own progress report is the job's ``progress``. A running call a
member hands back is a job too. A finished job is forgotten by ``DELETE``;
one still running is not. Past the live limit a call is refused before it
runs, and many clients waiting on jobs hold no thread a new call needs.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import rest.jobs
from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.settings_fixtures import complete_settings


@pytest.fixture
def live(tmp_path):
    folder = tmp_path / 'live'
    folder.mkdir()
    return folder


@pytest.fixture
def session(live):
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    yield s
    s.shutdown()


@pytest.fixture
def client(session):
    # One event loop for every request, as the server runs.
    with TestClient(build_app(session)) as client:
        yield client


@pytest.fixture
def held(session, monkeypatch):
    """``leds_off`` held until the test lets it go, then done as the scope does it."""
    release = threading.Event()
    calls = []
    real = session.scope.illumination.leds_off

    def leds_off():
        calls.append(1)
        assert release.wait(10)
        return real()

    monkeypatch.setattr(session.scope.illumination, 'leds_off', leds_off)
    release.calls = calls
    yield release
    release.set()


def test_a_call_past_the_clients_wait_is_a_job_that_ends_in_its_answer(client, held):
    accepted = client.post('/api/v1/scope/illumination/leds_off', headers={'Prefer': 'wait=0'})

    assert accepted.status_code == 202
    job = accepted.json()
    assert accepted.headers['Location'] == f'/api/v1/jobs/{job["id"]}'
    assert accepted.headers['Retry-After'] == str(rest.jobs.RETRY_AFTER_S)
    assert accepted.headers['Preference-Applied'] == 'wait=0'
    assert (job['member'], job['status'], job['ended']) == (
        'scope/illumination/leds_off',
        'running',
        None,
    )
    assert 'result' not in job

    held.set()
    ended = client.get(accepted.headers['Location'], headers={'Prefer': 'wait=10'})

    assert ended.status_code == 200
    assert ended.json()['status'] == 'completed'
    assert ended.json()['result'] is None
    assert ended.json()['ended'] is not None


@pytest.mark.parametrize('prefer', ['wait=3; note=x', 'wait="3"', 'respond-async, wait=3'])
def test_a_wait_is_read_in_every_form_rfc_7240_writes(client, prefer):
    answered = client.get('/api/v1/app_version', headers={'Prefer': prefer})

    assert answered.headers['Preference-Applied'] == 'wait=3'


def test_a_call_inside_the_wait_is_answered_and_the_wait_is_capped(client):
    answered = client.get('/api/v1/app_version', headers={'Prefer': 'wait=999'})

    assert answered.status_code == 200
    assert answered.headers['Preference-Applied'] == f'wait={int(rest.jobs.WAIT_CAP_S)}'
    unsaid = client.get('/api/v1/app_version')
    assert 'Preference-Applied' not in unsaid.headers


def test_a_job_that_fails_reads_200_with_its_problem(client, session, monkeypatch):
    release = threading.Event()

    def fails():
        assert release.wait(10)
        raise ValueError('bad shape (3,)')

    monkeypatch.setattr(session.scope.illumination, 'leds_off', fails)
    accepted = client.post('/api/v1/scope/illumination/leds_off', headers={'Prefer': 'wait=0'})
    release.set()

    ended = client.get(accepted.headers['Location'], headers={'Prefer': 'wait=10'})

    assert ended.status_code == 200
    job = ended.json()
    assert job['status'] == 'failed'
    assert (job['error']['status'], job['error']['title'], job['error']['detail']) == (
        500,
        'ValueError',
        'bad shape (3,)',
    )


def test_a_members_progress_is_its_jobs_progress(client, session, live, monkeypatch):
    # Held until it is a job: a zip that finished inside the zero wait would
    # be answered 200, with no job to read.
    release = threading.Event()
    real = session.make_logs_zip

    def zips_once_released(*args, **kwargs):
        assert release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(session, 'make_logs_zip', zips_once_released)
    accepted = client.post(
        '/api/v1/make_logs_zip', json={'output_dir': 'reports'}, headers={'Prefer': 'wait=0'}
    )
    assert accepted.status_code == 202
    release.set()

    job = client.get(accepted.headers['Location'], headers={'Prefer': 'wait=30'}).json()

    assert job['status'] == 'completed'
    assert job['progress']['percent'] == 100.0
    assert job['result']['path']['name'].startswith('reports/')
    assert (live / job['result']['path']['name']).is_file()


def test_a_running_call_a_member_hands_back_is_a_job(client):
    started = client.post('/api/v1/scope/motion/start_home')

    assert started.status_code == 200
    job = started.json()
    assert job['member'] == 'scope/motion/start_home'
    ended = client.get(f'/api/v1/jobs/{job["id"]}', headers={'Prefer': 'wait=30'}).json()
    assert ended['status'] == 'completed'
    assert client.get('/api/v1/jobs').json()[0]['id'] == job['id']


def test_a_finished_job_is_forgotten_and_a_running_one_is_not(client, held):
    accepted = client.post('/api/v1/scope/illumination/leds_off', headers={'Prefer': 'wait=0'})
    location = accepted.headers['Location']

    running = client.delete(location)
    assert (running.status_code, running.json()['reason']) == (409, 'job_running')

    held.set()
    client.get(location, headers={'Prefer': 'wait=10'})
    assert client.delete(location).status_code == 204
    assert client.get(location).status_code == 404


def test_a_call_past_the_live_limit_is_refused_before_it_runs(client, held, monkeypatch):
    monkeypatch.setattr(rest.jobs, 'LIVE_LIMIT', 0)

    refused = client.post('/api/v1/scope/illumination/leds_off')

    assert (refused.status_code, refused.json()['reason']) == (503, 'overloaded')
    assert refused.headers['Retry-After'] == str(rest.jobs.RETRY_AFTER_S)
    assert held.calls == []


def test_a_call_that_hands_out_a_job_is_refused_while_the_limit_of_jobs_runs(client, monkeypatch):
    first = client.post('/api/v1/scope/motion/start_home').json()
    monkeypatch.setattr(rest.jobs, 'LIVE_LIMIT', 1)

    refused = client.post('/api/v1/scope/motion/start_home')
    # A call that hands out no job is not refused for running jobs.
    answered = client.get('/api/v1/app_version')

    assert (refused.status_code, refused.json()['reason']) == (503, 'overloaded')
    assert answered.status_code == 200
    client.get(f'/api/v1/jobs/{first["id"]}', headers={'Prefer': 'wait=30'})


def test_a_finished_job_past_the_count_bound_is_let_go(client, held, monkeypatch):
    monkeypatch.setattr(rest.jobs, 'FINISHED_LIMIT', 1)

    def finished_job():
        # Held until it is a job, then let finish: a call that ended inside
        # the zero wait would be answered 200, with no job.
        held.clear()
        accepted = client.post('/api/v1/scope/illumination/leds_off', headers={'Prefer': 'wait=0'})
        held.set()
        client.get(accepted.headers['Location'], headers={'Prefer': 'wait=10'})
        return accepted

    oldest = finished_job()
    newer = finished_job()

    finished_job()

    assert client.get(oldest.headers['Location']).status_code == 404
    assert client.get(newer.headers['Location']).status_code == 200


def test_many_clients_waiting_on_a_job_hold_nothing_a_new_call_needs(client, held):
    accepted = client.post('/api/v1/scope/illumination/leds_off', headers={'Prefer': 'wait=0'})
    location = accepted.headers['Location']
    waiting = [
        threading.Thread(
            target=client.get, args=(location,), kwargs={'headers': {'Prefer': 'wait=10'}}
        )
        for _ in range(45)
    ]
    for t in waiting:
        t.start()
    time.sleep(0.2)

    start = time.monotonic()
    answered = client.get('/api/v1/app_version')
    took = time.monotonic() - start

    held.set()
    for t in waiting:
        t.join(10)
    assert answered.status_code == 200
    assert took < 0.1, f'{took:.3f} s behind 45 waiting clients'
