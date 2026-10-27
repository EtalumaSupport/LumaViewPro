# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Anything slow is a job, for every call alike.

A call runs on a thread of its own and its route waits what the client
says it will wait (``Prefer: wait``, 2 s by default, at most 60 s, the
wait applied named in ``Preference-Applied``). A call still running then
is ``202 Accepted`` with the job's address, and the job, read with its
own ``Prefer: wait``, ends ``completed`` with the result a 200 would have
carried or ``failed`` with the problem; reading it is 200 either way. A
member's own progress report is the job's ``progress``. A running call a
member hands back is a job too: the call is answered ``202`` with that
job once it returns, whatever the client's wait, and makes no job of its
own. ``DELETE`` on it while it runs is
its operation's stop: ``202`` with the job, which ends as the stop leaves
it -- a stopped run ``aborted``, a still, which has no stop, as it would
have -- and a stop the scope refuses is the answer. A call adopted as a
job holds no operation, and ``DELETE`` on it while it runs is ``409``
``not_cancellable``. A finished job is forgotten by ``DELETE``. Past the
live limit a call is refused before it runs, and many clients waiting on
jobs hold no thread a new call needs.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time

import pytest
from fastapi.testclient import TestClient

import rest.jobs
from modules.notification_center import _UNTYPED_FAULT_BODY, UNTYPED_FAULT_TITLE
from modules.scope_session import ScopeSession
from rest import problems
from rest.app import build_app
from tests.scope_fakes import home_sim_scope
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


def test_a_job_read_as_finished_has_its_end_time_however_soon_it_is_read():
    # Read in the same loop pass its answer settles, before the loop runs
    # anything else: a reader on the loop can be scheduled there.
    async def read_as_it_settles():
        jobs = rest.jobs.JobRegistry(ended=lambda job: None)
        answered = asyncio.get_running_loop().create_future()
        job = jobs.adopt(
            answered, member='scope/illumination/leds_off', requested=problems.now(), progress=None
        )
        await asyncio.sleep(0)
        answer = problems.result(None)
        answered.set_result(answer)
        return jobs.get(job.id).view(), answer

    view, answer = asyncio.run(read_as_it_settles())

    assert view['status'] == 'completed'
    assert view['ended'] == answer.ended.isoformat()


@pytest.mark.parametrize('prefer', ['wait=3; note=x', 'wait="3"', 'note=x, wait=3'])
def test_a_wait_is_read_in_every_form_rfc_7240_writes(client, prefer):
    answered = client.get('/api/v1/app_version', headers={'Prefer': prefer})

    assert answered.headers['Preference-Applied'] == 'wait=3'


@pytest.mark.parametrize('prefer', ['respond-async', 'respond-async, wait=30'])
def test_a_call_asked_to_respond_async_is_answered_at_once_as_its_job(client, held, prefer):
    """``respond-async`` is a wait of none, and wins over a ``wait`` beside it."""
    accepted = client.post('/api/v1/scope/illumination/leds_off', headers={'Prefer': prefer})

    assert accepted.status_code == 202
    assert accepted.headers['Preference-Applied'] == 'respond-async'
    held.set()
    client.get(accepted.headers['Location'], headers={'Prefer': 'wait=10'})


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
        UNTYPED_FAULT_TITLE,
        _UNTYPED_FAULT_BODY,
    )


def test_an_answer_the_wire_cannot_carry_is_a_failed_job_and_the_listing_still_reads(
    client, session, monkeypatch
):
    # JSON has no NaN. Found only when the answer was rendered, a NaN held
    # by a job failed every read of it and of the whole job listing; refused
    # where the call ends, the job holds the problem.
    release = threading.Event()

    # a stand-in by design: no simulated member answers NaN; the camera's
    # cached gain did, from a fault the simulator cannot produce.
    def answers_nan():
        assert release.wait(10)
        return float('nan')

    monkeypatch.setattr(session.scope.camera, 'read_gain_db', answers_nan)
    accepted = client.get('/api/v1/scope/camera/read_gain_db', headers={'Prefer': 'wait=0'})
    release.set()

    ended = client.get(accepted.headers['Location'], headers={'Prefer': 'wait=10'})

    assert ended.status_code == 200
    job = ended.json()
    assert job['status'] == 'failed'
    assert (job['error']['status'], job['error']['kind']) == (500, 'fault')
    assert job['error']['reason'] == 'no_wire_form'
    assert client.get('/api/v1/jobs').status_code == 200


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
    assert job['progress']['progress_pct'] == 100.0
    assert job['result']['path']['name'].startswith('reports/')
    assert (live / job['result']['path']['name']).is_file()


def test_a_running_call_a_member_hands_back_is_a_job(client):
    started = client.post('/api/v1/scope/motion/start_home')

    assert started.status_code == 202
    job = started.json()
    assert started.headers['Location'] == f'/api/v1/jobs/{job["id"]}'
    assert started.headers['Retry-After'] == str(rest.jobs.RETRY_AFTER_S)
    assert job['member'] == 'scope/motion/start_home'
    ended = client.get(f'/api/v1/jobs/{job["id"]}', headers={'Prefer': 'wait=30'}).json()
    assert ended['status'] == 'completed'
    assert client.get('/api/v1/jobs').json()[0]['id'] == job['id']


def _posted(client, path, **kwargs):
    """POST *path* on a thread of its own; the thread, and a list its answer lands in."""
    answers = []
    poster = threading.Thread(target=lambda: answers.append(client.post(path, **kwargs)))
    poster.start()
    return poster, answers


def test_a_start_that_outlives_the_clients_wait_is_one_job_its_operation(
    client, session, monkeypatch
):
    """The client is handed the operation's own job, never a job of the call around it."""
    home_sim_scope(session.scope)
    reached, release = threading.Event(), threading.Event()
    real = session.scope.motion.start_move_z_by

    def dispatched_once_released(*args, **kwargs):
        reached.set()
        assert release.wait(10)
        return real(*args, **kwargs)

    # a stand-in by design: holds the start past the client's wait; the
    # simulated board dispatches a move at once.
    monkeypatch.setattr(session.scope.motion, 'start_move_z_by', dispatched_once_released)
    poster, answers = _posted(
        client,
        '/api/v1/scope/motion/start_move_z_by',
        json={'dz_um': 10.0},
        headers={'Prefer': 'wait=0'},
    )
    assert reached.wait(10)
    release.set()
    poster.join(10)

    (started,) = answers
    assert started.status_code == 202
    job = started.json()
    assert started.headers['Location'] == f'/api/v1/jobs/{job["id"]}'
    # Listed once the answered job has ended: by then the call has returned
    # whatever it hands back, so a job of its operation would be held too.
    client.get(started.headers['Location'], headers={'Prefer': 'wait=30'})
    assert [j['id'] for j in client.get('/api/v1/jobs').json()] == [job['id']]


def test_a_run_started_past_the_clients_wait_is_stopped_by_the_job_it_answered(
    client, session, monkeypatch
):
    home_sim_scope(session.scope)
    session.settings['BF']['acquire'] = 'image'
    reached, release = _scans_held_at_their_first_step(session, monkeypatch)
    runner = session.create_protocol_runner()
    held_start = runner.start_single_scan
    dispatching, dispatch = threading.Event(), threading.Event()

    def dispatched_once_released(*args, **kwargs):
        dispatching.set()
        assert dispatch.wait(10)
        return held_start(*args, **kwargs)

    # a stand-in by design: holds the run's start past the client's wait; a
    # simulated run prepares inside it.
    monkeypatch.setattr(runner, 'start_single_scan', dispatched_once_released)
    protocol = client.post('/api/v1/create_empty_protocol').json()['handle']
    assert client.post('/api/v1/add_step', json={'protocol': protocol}).status_code == 200
    handle = client.post('/api/v1/create_protocol_runner').json()['handle']
    try:
        poster, answers = _posted(
            client,
            f'/api/v1/handles/ProtocolRunner/{handle}/start_single_scan',
            json={'protocol': protocol, 'enable_image_saving': False},
            headers={'Prefer': 'wait=0'},
        )
        assert dispatching.wait(10)
        dispatch.set()
        poster.join(10)
        assert reached.wait(30)
        (started,) = answers
        location = started.headers['Location']

        stopping = client.delete(location)

        assert stopping.status_code == 202
        assert stopping.json()['progress']['is_stopping'] is True
    finally:
        dispatch.set()
        release.set()
    ended = client.get(location, headers={'Prefer': 'wait=30'}).json()
    assert (ended['result']['status'], ended['result']['reason']) == ('aborted', 'stopped')


def test_a_start_is_answered_on_return_whatever_the_clients_wait(client, session, monkeypatch):
    """``Prefer: wait`` is a blocking call's: a start answers its running job once dispatched."""
    home_sim_scope(session.scope)
    session.settings['BF']['acquire'] = 'image'
    reached, release = _scans_held_at_their_first_step(session, monkeypatch)
    protocol = client.post('/api/v1/create_empty_protocol').json()['handle']
    assert client.post('/api/v1/add_step', json={'protocol': protocol}).status_code == 200
    handle = client.post('/api/v1/create_protocol_runner').json()['handle']
    try:
        poster, answers = _posted(
            client,
            f'/api/v1/handles/ProtocolRunner/{handle}/start_single_scan',
            json={'protocol': protocol, 'enable_image_saving': False},
            headers={'Prefer': 'wait=60'},
        )
        assert reached.wait(30)
        # Bounded far below the client's wait: a route that spent the wait on
        # the run would still be waiting, since the run is held here.
        poster.join(10)

        (started,) = answers
        assert started.status_code == 202
        assert started.json()['status'] == 'running'
    finally:
        release.set()
        poster.join(30)
    client.get(started.headers['Location'], headers={'Prefer': 'wait=30'})


def test_a_finished_job_is_forgotten_and_a_running_adopted_call_is_not_stopped(client, held):
    accepted = client.post('/api/v1/scope/illumination/leds_off', headers={'Prefer': 'wait=0'})
    location = accepted.headers['Location']

    running = client.delete(location)
    assert (running.status_code, running.json()['reason']) == (409, 'not_cancellable')
    assert client.get(location).json()['status'] == 'running'

    held.set()
    client.get(location, headers={'Prefer': 'wait=10'})
    assert client.delete(location).status_code == 204
    assert client.get(location).status_code == 404


def _scans_held_at_their_first_step(session, monkeypatch):
    """The shared runner's scans, each held at its first step until *release* is set.

    ``step_started`` is delivered on the run's own thread on a host with no
    UI dispatcher, as the server is, so the run waits in it.
    """
    reached, release = threading.Event(), threading.Event()
    runner = session.create_protocol_runner()
    real = runner.start_single_scan

    def held(*args, events, **kwargs):
        def step_started(index):
            events.step_started(index)
            reached.set()
            release.wait(30)

        return real(*args, events=dataclasses.replace(events, step_started=step_started), **kwargs)

    monkeypatch.setattr(runner, 'start_single_scan', held)
    return reached, release


def test_a_run_job_deleted_while_it_runs_is_stopped_and_ends_aborted(client, session, monkeypatch):
    home_sim_scope(session.scope)
    session.settings['BF']['acquire'] = 'image'
    reached, release = _scans_held_at_their_first_step(session, monkeypatch)
    protocol = client.post('/api/v1/create_empty_protocol').json()['handle']
    assert client.post('/api/v1/add_step', json={'protocol': protocol}).status_code == 200
    runner = client.post('/api/v1/create_protocol_runner').json()['handle']
    try:
        job = client.post(
            f'/api/v1/handles/ProtocolRunner/{runner}/start_single_scan',
            json={'protocol': protocol, 'enable_image_saving': False},
        ).json()
        assert reached.wait(30)
        location = f'/api/v1/jobs/{job["id"]}'
        running = client.get(location).json()
        assert running['status'] == 'running'
        assert (running['progress']['is_live'], running['progress']['is_stopping']) == (True, False)

        stopping = client.delete(location)

        assert stopping.status_code == 202
        assert stopping.json()['id'] == job['id']
        assert stopping.json()['progress']['is_stopping'] is True
    finally:
        release.set()
    ended = client.get(location, headers={'Prefer': 'wait=30'}).json()
    assert ended['status'] == 'completed'
    assert (ended['result']['status'], ended['result']['reason']) == ('aborted', 'stopped')


def test_a_job_is_passed_as_an_argument_by_its_id(client, session, monkeypatch):
    home_sim_scope(session.scope)
    zipping = threading.Event()
    real_zip = session.make_logs_zip

    def zips_once_released(*args, **kwargs):
        assert zipping.wait(10)
        return real_zip(*args, **kwargs)

    # Held so it is a job, adopted, when the client's wait ends.
    monkeypatch.setattr(session, 'make_logs_zip', zips_once_released)
    session.settings['BF']['acquire'] = 'image'
    reached, release = _scans_held_at_their_first_step(session, monkeypatch)
    protocol = client.post('/api/v1/create_empty_protocol').json()['handle']
    assert client.post('/api/v1/add_step', json={'protocol': protocol}).status_code == 200
    runner = client.post('/api/v1/create_protocol_runner').json()['handle']
    try:
        run = client.post(
            f'/api/v1/handles/ProtocolRunner/{runner}/start_single_scan',
            json={'protocol': protocol, 'enable_image_saving': False},
        ).json()
        assert reached.wait(30)

        # The scope is held by the run itself, and by something else for anything else.
        assert client.get('/api/v1/held_by_other', params={'run': run['id']}).json() is False
        assert client.get('/api/v1/held_by_other').json() is True
        unknown = client.get('/api/v1/held_by_other', params={'run': '999'})
        assert (unknown.status_code, unknown.json()['reason']) == (404, 'not_found')
        adopted = client.post(
            '/api/v1/make_logs_zip', json={'output_dir': 'reports'}, headers={'Prefer': 'wait=0'}
        ).json()
        no_operation = client.get('/api/v1/held_by_other', params={'run': adopted['id']})
        assert (no_operation.status_code, no_operation.json()['reason']) == (
            422,
            'invalid_request',
        )
    finally:
        release.set()
        zipping.set()


def test_a_still_job_deleted_while_it_runs_is_accepted_and_ends_as_it_would_have(
    client, session, monkeypatch
):
    """A still has no stop, and asking it to stop is not refused (Eric, 2026-10-13)."""
    release = threading.Event()
    real = session.manual_capture._capture_and_save

    def held(request):
        assert release.wait(10)
        return real(request)

    # a stand-in by design: holds the still on the camera lane so the DELETE
    # meets it running; the simulated camera captures at once.
    monkeypatch.setattr(session.manual_capture, '_capture_and_save', held)
    try:
        job = client.post('/api/v1/manual_capture/capture', json={'false_color_on': False}).json()
        location = f'/api/v1/jobs/{job["id"]}'

        asked = client.delete(location)

        assert asked.status_code == 202
        assert asked.json()['status'] == 'running'
    finally:
        release.set()
    ended = client.get(location, headers={'Prefer': 'wait=30'}).json()
    assert ended['status'] == 'completed'
    assert len(ended['result']) == 1


def test_a_stop_the_scope_refuses_is_the_deletes_answer(tmp_path, monkeypatch):
    """A home on the EL-0940, whose board takes no stop while it homes."""
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850', simulator_tier='firmware'),
        simulate=True,
    )
    scope = session.scope
    homing = threading.Event()
    real = scope._motion_driver.zhome

    def announced(*args, **kwargs):
        homing.set()
        return real(*args, **kwargs)

    # a stand-in by design: tells the test the board has begun the home, a
    # moment the simulated board does not announce.
    monkeypatch.setattr(scope._motion_driver, 'zhome', announced)
    try:
        with TestClient(build_app(session)) as client:
            job = client.post('/api/v1/scope/motion/start_home', json={'axis': 'Z'}).json()
            assert homing.wait(30)

            refused = client.delete(f'/api/v1/jobs/{job["id"]}')

            assert (refused.status_code, refused.json()['reason']) == (409, 'home_not_stoppable')
            ended = client.get(f'/api/v1/jobs/{job["id"]}', headers={'Prefer': 'wait=60'})
            assert ended.json()['status'] == 'completed'
    finally:
        session.shutdown()


def test_a_call_past_the_live_limit_is_refused_before_it_runs(client, held, monkeypatch):
    monkeypatch.setattr(rest.jobs, 'LIVE_LIMIT', 0)

    refused = client.post('/api/v1/scope/illumination/leds_off')

    assert (refused.status_code, refused.json()['reason']) == (503, 'overloaded')
    assert refused.headers['Retry-After'] == str(rest.jobs.RETRY_AFTER_S)
    assert held.calls == []


def test_a_call_that_hands_out_a_job_is_refused_while_the_limit_of_jobs_runs(
    client, session, monkeypatch
):
    # The first job is a move held part of the way, so it runs until the
    # test lets it go: a home on the fast simulator could end before the
    # second call, and that call would be the one job running, not refused.
    home_sim_scope(session.scope)
    held = session.scope.simulation.hold_axis('Z')
    first = client.post('/api/v1/scope/motion/start_move_z_by', json={'dz_um': 100.0}).json()
    assert held.wait(10), 'the first job never got part of the way'
    monkeypatch.setattr(rest.jobs, 'LIVE_LIMIT', 1)

    refused = client.post('/api/v1/scope/motion/start_home')
    # A call that hands out no job is not refused for running jobs.
    answered = client.get('/api/v1/app_version')

    assert (refused.status_code, refused.json()['reason']) == (503, 'overloaded')
    assert answered.status_code == 200
    session.scope.simulation.clear()
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
    """A new call is answered while 45 clients still wait on a job.

    A wait that held what a new call needs would answer it only once a
    waiter's own wait ran out, so the call is bounded by the waiters' wait,
    never by how fast the host is: a loaded host answers it in about half
    a second, a held one in no less than the wait.
    """
    waiters_wait_s = 10
    accepted = client.post('/api/v1/scope/illumination/leds_off', headers={'Prefer': 'wait=0'})
    location = accepted.headers['Location']
    waiting = [
        threading.Thread(
            target=client.get,
            args=(location,),
            kwargs={'headers': {'Prefer': f'wait={waiters_wait_s}'}},
        )
        for _ in range(45)
    ]
    for t in waiting:
        t.start()
    time.sleep(0.2)

    start = time.monotonic()
    answered = client.get('/api/v1/app_version')
    took = time.monotonic() - start
    still_waiting = sum(t.is_alive() for t in waiting)

    held.set()
    for t in waiting:
        t.join(waiters_wait_s)
    assert answered.status_code == 200
    assert still_waiting == len(waiting), (
        f'answered only after {len(waiting) - still_waiting} waiters gave up'
    )
    assert took < waiters_wait_s / 2, f'{took:.3f} s behind {len(waiting)} waiting clients'
