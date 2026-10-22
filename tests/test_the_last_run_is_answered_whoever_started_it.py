# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``ScopeSession.last_run`` answers the last run's operation, whoever started it, live or ended.

A client watching the scope learns of a run the GUI, a script or the SDK
started from the ``status`` edges alone; this read hands it that run, and on
the wire the run is a job -- one per run, whoever reads it -- so the client
follows its progress and hears its ``job_ended``. Every ``ProtocolRunner``
start is a run: a scan, a protocol, a composite, an autofocus, a z-stack.
The read answers the last run, live or ended, and None only before any:
a run goes idle before its operation settles, inside its ``run_ended``
delivery, and a run short against the client's round trip from the edge
would otherwise end before the read, its outcome never heard.

A POST, not a GET: a read that hands out a running call is refused at the
route build.
"""

from __future__ import annotations

import contextlib
import json
import threading

import httpx
import pytest
import uvicorn

from modules.run_events import RunEvents
from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    live = tmp_path / 'live'
    live.mkdir()
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    yield s
    s.shutdown()


@contextlib.contextmanager
def _serving(session):
    server = uvicorn.Server(
        uvicorn.Config(build_app(session), host='127.0.0.1', port=0, log_level='warning')
    )
    thread = threading.Thread(target=server.run)
    thread.start()
    try:
        while not server.started:
            assert thread.is_alive(), 'the server ended before it started'
            thread.join(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=10) as client:
            yield client
    finally:
        server.should_exit = True
        thread.join(10)
        assert not thread.is_alive()


def _events_until(lines, name: str) -> list[dict]:
    """Each event read from a stream's *lines* up to and including the first named *name*."""
    seen: list[dict] = []
    event: dict = {}
    for line in lines:
        if not line:
            if event:
                seen.append(event)
                if event['event'] == name:
                    return seen
                event = {}
            continue
        if line.startswith(':'):
            continue
        field, _, value = line.partition(': ')
        event[field] = json.loads(value) if field == 'data' else value
    raise AssertionError(f'the stream ended before {name}')


def _start_a_scan(session, **events):
    """A run started in-process, as the GUI or a script starts one."""
    home_sim_scope(session.scope)
    session.set_layer_acquire('BF', 'image')
    protocol = session.create_empty_protocol()
    session.add_step(protocol)
    return session.create_protocol_runner().start_single_scan(
        protocol=protocol, enable_image_saving=False, events=RunEvents(**events)
    )


def test_none_is_answered_before_any_run(session):
    assert session.last_run() is None
    with _serving(session) as client:
        answered = client.post('/api/v1/last_run')

    assert (answered.status_code, answered.json()) == (200, None)


def test_a_run_another_caller_started_is_one_job_whose_end_reaches_the_stream(session):
    """Read between the run going idle and its operation settling: still answered, as the running job."""
    ending = threading.Event()
    release = threading.Event()

    def run_ended(*_args, **_kwargs):
        # Inside the run's run_ended delivery: idle, its operation not yet settled.
        ending.set()
        assert release.wait(10)

    with _serving(session) as client, client.stream('GET', '/api/v1/events') as response:
        stream = response.iter_lines()
        _events_until(stream, 'status')
        try:
            run = _start_a_scan(session, run_ended=run_ended)
            assert ending.wait(10)
            assert not session.sequenced_capture_runner.run_live
            read = client.post('/api/v1/last_run')
            again = client.post('/api/v1/last_run')
        finally:
            release.set()
        ended = _events_until(stream, 'job_ended')[-1]

    assert session.last_run() is run
    assert read.status_code == 202, read.text
    job = read.json()
    assert job['status'] == 'running'
    assert read.headers['location'].endswith(f'/jobs/{job["id"]}')
    assert again.json()['id'] == job['id']
    assert (ended['data']['id'], ended['data']['status']) == (job['id'], 'completed')
    assert ended['data']['result']['status'] == run.result().status


def test_a_run_read_after_it_ended_is_its_finished_job(session):
    """A reader too late to see the run live still gets its outcome, and its job_ended."""
    run = _start_a_scan(session)
    outcome = run.result(timeout=30)

    with _serving(session) as client, client.stream('GET', '/api/v1/events') as response:
        stream = response.iter_lines()
        _events_until(stream, 'status')
        read = client.post('/api/v1/last_run')
        ended = _events_until(stream, 'job_ended')[-1]

    # The job's end is told at the read: its operation was already done.
    assert read.status_code == 202, read.text
    assert (ended['data']['id'], ended['data']['status']) == (read.json()['id'], 'completed')
    assert ended['data']['result']['status'] == outcome.status
