# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A retried request runs once: ``Idempotency-Key`` (``rest.idempotency``).

A POST carrying the header is reserved when it arrives; the same key again,
from any connection, runs nothing and is answered what the first request
was answered -- the same status and body, marked ``Idempotent-Replayed``,
waiting for that answer when the first is still running. The same key with
another member or body is refused ``idempotency_key_reused``; a body's
keys in another order are the same body. Keys are held apart from jobs,
for a day and up to a count, the oldest let go first; a request refused
before it ran holds no key.
"""

from __future__ import annotations

import json
import threading

import pytest
from fastapi.testclient import TestClient

import rest.idempotency
import rest.jobs
from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield s
    s.shutdown()


@pytest.fixture
def client(session):
    with TestClient(build_app(session)) as client:
        yield client


@pytest.fixture
def counted(session, monkeypatch):
    """``leds_off`` counting its calls, and held while ``release`` is clear."""
    release = threading.Event()
    release.set()
    calls = []
    real = session.scope.illumination.leds_off

    def leds_off():
        calls.append(1)
        assert release.wait(10)
        return real()

    # a stand-in by design: counts how often the member ran, which no
    # answer says.
    monkeypatch.setattr(session.scope.illumination, 'leds_off', leds_off)
    release.calls = calls
    return release


def _keyed(key: str) -> dict[str, str]:
    return {'Idempotency-Key': key}


def test_the_same_key_again_runs_nothing_and_is_answered_the_first_answer(client, counted):
    first = client.post('/api/v1/scope/illumination/leds_off', headers=_keyed('k1'))
    again = client.post('/api/v1/scope/illumination/leds_off', headers=_keyed('k1'))

    assert len(counted.calls) == 1
    assert (again.status_code, again.content) == (first.status_code, first.content)
    assert again.headers['Idempotent-Replayed'] == 'true'
    assert 'Idempotent-Replayed' not in first.headers
    assert again.headers['X-Request-ID'] != first.headers['X-Request-ID']


def test_a_retry_while_the_first_still_runs_waits_for_its_answer(client, counted):
    counted.clear()
    answers = []

    def post():
        answers.append(
            client.post(
                '/api/v1/scope/illumination/leds_off',
                headers={**_keyed('k2'), 'Prefer': 'wait=10'},
            )
        )

    posting = [threading.Thread(target=post) for _ in range(2)]
    for t in posting:
        t.start()
    counted.set()
    for t in posting:
        t.join(20)

    assert len(counted.calls) == 1
    assert sorted(a.headers.get('Idempotent-Replayed', 'false') for a in answers) == [
        'false',
        'true',
    ]
    assert answers[0].content == answers[1].content


def test_a_body_is_the_same_body_whatever_its_keys_order(client, session):
    home_sim_scope(session.scope)
    path = '/api/v1/scope/illumination/led_on'
    sent = {'Content-Type': 'application/json', **_keyed('k3')}
    first = client.post(
        path, content=json.dumps({'channel': LED0, 'illumination_ma': 10.0}), headers=sent
    )
    again = client.post(
        path, content=json.dumps({'illumination_ma': 10.0, 'channel': LED0}), headers=sent
    )

    assert again.headers['Idempotent-Replayed'] == 'true'
    assert again.content == first.content


def test_the_same_key_with_another_body_is_refused(client, counted):
    client.post('/api/v1/scope/illumination/leds_off', headers=_keyed('k4'))
    other = client.post(
        '/api/v1/scope/illumination/led_off', json={'channel': LED0}, headers=_keyed('k4')
    )

    assert (other.status_code, other.json()['reason']) == (422, 'idempotency_key_reused')


def test_a_started_operation_retried_is_its_one_job(client, session):
    home_sim_scope(session.scope)
    path = '/api/v1/scope/motion/start_move_z_by'
    move = {'dz_um': 10.0}
    first = client.post(path, json=move, headers=_keyed('k5'))
    again = client.post(path, json=move, headers=_keyed('k5'))

    assert (first.status_code, again.status_code) == (202, 202)
    assert again.json()['id'] == first.json()['id']
    assert again.headers['Location'] == first.headers['Location']
    assert [j['id'] for j in client.get('/api/v1/jobs').json()] == [first.json()['id']]


def test_keys_are_held_apart_from_jobs_and_the_oldest_is_let_go(client, counted, monkeypatch):
    monkeypatch.setattr(rest.jobs, 'FINISHED_LIMIT', 1)
    monkeypatch.setattr(rest.idempotency, 'KEY_LIMIT', 1)
    counted.clear()
    job = client.post('/api/v1/scope/illumination/leds_off', headers={'Prefer': 'wait=0'})
    counted.set()
    client.get(job.headers['Location'], headers={'Prefer': 'wait=10'})

    client.post('/api/v1/scope/illumination/leds_off', headers=_keyed('old'))
    client.post('/api/v1/scope/illumination/leds_off', headers=_keyed('new'))
    retried_old = client.post('/api/v1/scope/illumination/leds_off', headers=_keyed('old'))

    # A keyed call is no job: the finished job outlives a bound of one.
    assert client.get(job.headers['Location']).status_code == 200
    # The oldest key was let go past a bound of one: its retry ran again.
    assert 'Idempotent-Replayed' not in retried_old.headers
    assert len(counted.calls) == 4


def test_a_request_refused_before_it_ran_holds_no_key(client, counted, monkeypatch):
    monkeypatch.setattr(rest.jobs, 'LIVE_LIMIT', 0)
    refused = client.post('/api/v1/scope/illumination/leds_off', headers=_keyed('k6'))
    monkeypatch.setattr(rest.jobs, 'LIVE_LIMIT', 256)
    ran = client.post('/api/v1/scope/illumination/leds_off', headers=_keyed('k6'))

    assert refused.status_code == 503
    assert (ran.status_code, 'Idempotent-Replayed' in ran.headers) == (200, False)
    assert len(counted.calls) == 1
