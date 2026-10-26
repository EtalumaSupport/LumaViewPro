# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The server closes the session with its clients connected (``rest.server``).

On SIGINT or SIGTERM, or its host's ``close()``, a new connection is
refused and a member asked on an open one is refused ``server_closing``,
while jobs and the stream are still served; the host's shutdown closes the
session while the stream carries what the close reports; then ``closing``
ends every stream, a client still waiting on a job is answered what the job
is then, the calls' threads are joined with a bound, and the server stops.
A later signal changes nothing. The close runs once, whichever thread asks
first; a server not serving closes nothing, and its host shuts its session
down itself.

Driven in-process: the server runs on a thread and its signal handler is
called as the signal would call it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import threading

import httpx
import pytest
import uvicorn

import rest.app
import rest.jobs
import rest.server
from modules.scope_session import ScopeSession
from rest.app import build_app
from rest.server import Server
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0


@pytest.fixture
def session(tmp_path):
    live = tmp_path / 'live'
    live.mkdir()
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    yield s
    s.shutdown()


def _server(session, shut_down=None) -> Server:
    app = build_app(session)
    return Server(
        uvicorn.Config(app, host='127.0.0.1', port=0, log_level='warning'),
        session,
        app.state.closing,
        shut_down=shut_down or session.shutdown,
    )


@contextlib.contextmanager
def _serving(session, shut_down=None):
    server = _server(session, shut_down)
    app = server.config.app
    thread = threading.Thread(target=server.run)
    thread.start()
    try:
        while not server.serving:
            assert thread.is_alive(), 'the server ended before it served'
            thread.join(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=10) as client:
            yield server, client, app.state.closing
    finally:
        server.should_exit = True
        thread.join(10)
        assert not thread.is_alive()


def _events(response: httpx.Response):
    """Each event a stream sends, ``{"id", "event", "data"}``, until it ends."""
    event: dict = {}
    for line in response.iter_lines():
        if not line:
            if event:
                yield event
                event = {}
            continue
        if line.startswith(':'):
            continue
        field, _, value = line.partition(': ')
        event[field] = json.loads(value) if field == 'data' else value


def _until(events, name: str) -> list[dict]:
    seen = []
    for event in events:
        seen.append(event)
        if event['event'] == name:
            return seen
    raise AssertionError(f'the stream ended before {name}')


def test_a_signal_closes_the_session_with_its_clients_connected(session):
    # A stand-in by design: the subject is the server's close, and the
    # session's close is held open so the test can ask during it.
    asked = threading.Event()
    release = threading.Event()
    real_shutdown = session.shutdown

    def held_shutdown():
        asked.set()
        assert release.wait(10)
        real_shutdown()

    with (
        _serving(session, shut_down=held_shutdown) as (server, client, _closing),
        httpx.Client(base_url=client.base_url, timeout=10) as connected,
    ):
        lit = connected.post(
            '/api/v1/scope/illumination/led_on', json={'channel': LED0, 'illumination_ma': 10.0}
        )
        assert lit.status_code == 200
        try:
            with client.stream('GET', '/api/v1/events') as response:
                events = _events(response)
                _until(events, 'status')

                server.handle_exit(signal.SIGTERM, None)
                assert asked.wait(10)
                server.handle_exit(signal.SIGINT, None)

                # Asked on the connection opened before the signal.
                refused = connected.get('/api/v1/status')
                jobs = connected.get('/api/v1/jobs')
                with pytest.raises(httpx.ConnectError):
                    httpx.get(client.base_url.join('/api'))

                release.set()
                told = _until(events, 'closing')
                after = list(events)
        finally:
            release.set()

    assert (refused.status_code, refused.json()['reason']) == (503, 'server_closing')
    assert jobs.status_code == 200
    # The close's own report reached the stream before closing: the LED it turned off.
    assert {'address': 0, 'name': LED0, 'state': 'dark', 'illumination_ma': None} in [
        e['data'] for e in told if e['event'] == 'led'
    ]
    assert after == []
    assert not server.force_exit
    assert server.close_failed is None
    assert session.live_work.closed


def test_the_close_names_a_call_still_running_and_joins_it_once_it_ends(session, monkeypatch):
    release = threading.Event()
    real = session.scope.illumination.leds_off

    def leds_off():
        assert release.wait(10)
        return real()

    monkeypatch.setattr(session.scope.illumination, 'leds_off', leds_off)

    with _serving(session) as (_server, client, closing):
        accepted = client.post('/api/v1/scope/illumination/leds_off', headers={'Prefer': 'wait=0'})
        assert accepted.status_code == 202

        running = closing.join_jobs(0.1)
        release.set()
        ended = closing.join_jobs(10)

    assert running == ['rest scope/illumination/leds_off']
    assert ended == []


def test_a_client_waiting_on_a_job_is_answered_when_the_close_finishes(session, monkeypatch):
    """A long-poll open at the close is answered what its job is then; it does not hold the exit.

    The job is a call the session's close does not wait for, as a call hung
    in a driver would be: the close finishes the session's own work first,
    and waits for the call's thread no longer than its bound.
    """
    monkeypatch.setattr(rest.server, 'JOIN_S', 0.1)
    release = threading.Event()
    real = session.scope.camera.read_gain_db

    def hung():
        assert release.wait(30)
        return real()

    # a stand-in by design: a call that outlives the close, of a member the
    # close itself never calls; no simulated member hangs.
    monkeypatch.setattr(session.scope.camera, 'read_gain_db', hung)
    waiting = threading.Event()
    real_get = rest.jobs.JobRegistry.get

    def got(self, job_id):
        found = real_get(self, job_id)
        waiting.set()
        return found

    # a stand-in by design: tells the test the long-poll has reached its
    # wait, which the read begins right after this, on the server's loop.
    monkeypatch.setattr(rest.jobs.JobRegistry, 'get', got)
    answers = []
    try:
        with _serving(session) as (server, client, _closing):
            job = client.get(
                '/api/v1/scope/camera/read_gain_db', headers={'Prefer': 'respond-async'}
            ).json()
            polling = threading.Thread(
                target=lambda: answers.append(
                    client.get(f'/api/v1/jobs/{job["id"]}', headers={'Prefer': 'wait=60'})
                )
            )
            polling.start()
            assert waiting.wait(10)

            server.handle_exit(signal.SIGTERM, None)
            # Far below the long-poll's 60 s: answered by the close, not by its wait.
            polling.join(10)
            assert not polling.is_alive()
    finally:
        release.set()

    (answered,) = answers
    assert (answered.status_code, answered.json()['status']) == (200, 'running')


def _closed_on_a_thread(server: Server) -> tuple[threading.Thread, list]:
    answered: list = []

    def close():
        try:
            answered.append(server.close())
        except Exception as e:
            answered.append(e)

    thread = threading.Thread(target=close)
    thread.start()
    return thread, answered


def test_the_hosts_close_closes_the_session_through_the_server_with_its_clients_connected(
    session,
):
    """A host with no signal -- the GUI -- closes the server from a thread of its own.

    The call returns once the server has stopped: the session closed by the
    host's shutdown, once, while the stream carried the close's report, then
    the stream ended. It answers True: the host has nothing left to shut down.
    """
    shut = []

    def shut_down():
        shut.append(threading.current_thread())
        session.shutdown()

    with _serving(session, shut_down=shut_down) as (server, client, _closing):
        assert (
            client.post(
                '/api/v1/scope/illumination/led_on',
                json={'channel': LED0, 'illumination_ma': 10.0},
            ).status_code
            == 200
        )
        with client.stream('GET', '/api/v1/events') as response:
            events = _events(response)
            _until(events, 'status')
            closer, answered = _closed_on_a_thread(server)
            told = _until(events, 'closing')
            after = list(events)
        closer.join(10)
        assert not closer.is_alive()

    assert answered == [True]
    assert len(shut) == 1 and shut[0] is not closer
    assert {'address': 0, 'name': LED0, 'state': 'dark', 'illumination_ma': None} in [
        e['data'] for e in told if e['event'] == 'led'
    ]
    assert after == []
    assert not server.serving
    assert session.live_work.closed


def test_the_close_asked_from_two_threads_at_once_runs_once(session):
    """The GUI's close thread and its on_stop can both ask: one closes, the other waits for it.

    The one that waited answers False, so its host's own shutdown follows --
    a no-op once the session has closed, as it is today.
    """
    entered = threading.Event()
    release = threading.Event()
    shut = []

    def shut_down():
        shut.append(1)
        entered.set()
        assert release.wait(10)
        session.shutdown()

    try:
        with _serving(session, shut_down=shut_down) as (server, _client, _closing):
            first, first_answer = _closed_on_a_thread(server)
            assert entered.wait(10)
            second, second_answer = _closed_on_a_thread(server)
            # The second is waiting on the first's close, not returned.
            second.join(0.2)
            assert second.is_alive()
            release.set()
            first.join(10)
            second.join(10)
    finally:
        release.set()

    assert (first_answer, second_answer, shut) == ([True], [False], [1])


def test_a_close_after_the_server_stopped_closes_nothing(session):
    """A close once ``run()`` has returned hands off to nothing: the host shuts down itself."""
    shut = []
    with _serving(session, shut_down=lambda: shut.append(1)) as (server, _client, _closing):
        pass

    assert (server.close(), shut, server.serving) == (False, [], False)


def test_a_close_before_the_server_runs_closes_nothing_and_it_never_serves(session):
    shut = []
    server = _server(session, shut_down=lambda: shut.append(1))

    assert server.close() is False
    # Run afterwards, it returns without serving: a server serving a closed
    # session would answer every client with a closed scope.
    ran = threading.Thread(target=server.run)
    ran.start()
    ran.join(10)
    assert not ran.is_alive()
    assert (shut, server.serving, server.started) == ([], False, False)


def test_a_close_while_the_server_starts_stops_it_and_leaves_the_session_to_its_host(
    session, monkeypatch
):
    """Between its thread's start and its loop's startup the server closes nothing.

    It is told to stop, and ``close()`` returns once ``run()`` has, without
    the server serving; the host closes the session itself, so no server is
    left serving a closed one.
    """
    starting = threading.Event()
    release = threading.Event()
    real_startup = uvicorn.Server.startup

    async def held_startup(self, sockets=None):
        starting.set()
        assert await asyncio.to_thread(release.wait, 10)
        await real_startup(self, sockets)

    # a stand-in by design: holds the start inside its own frame, the window
    # a close lands in; the simulator has no hook there.
    monkeypatch.setattr(uvicorn.Server, 'startup', held_startup)
    shut = []
    server = _server(session, shut_down=lambda: shut.append(1))
    ran = threading.Thread(target=server.run)
    ran.start()
    try:
        assert starting.wait(10)
        closer, answered = _closed_on_a_thread(server)
        # Waiting for the start to end, not returned.
        closer.join(0.2)
        assert closer.is_alive()
    finally:
        release.set()
    closer.join(10)
    ran.join(10)

    assert answered == [False]
    assert not ran.is_alive()
    assert (shut, server.serving) == ([], False)


def test_a_failed_shutdown_is_raised_to_the_close_that_ran_it(session):
    """The host's shutdown that raises is raised from ``close()``, as it was from the host's own call.

    The server still finishes its clients and stops.
    """

    def shut_down():
        raise RuntimeError('the scope did not disconnect')

    with _serving(session, shut_down=shut_down) as (server, _client, _closing):
        closer, answered = _closed_on_a_thread(server)
        closer.join(10)

    (raised,) = answered
    assert isinstance(raised, RuntimeError) and str(raised) == 'the scope did not disconnect'
    assert server.close_failed is raised
    assert not server.serving


def test_the_close_joins_a_call_still_running_before_the_server_stops(session, monkeypatch):
    """The server's close waits for its calls' threads, so a GUI host, which has no launcher, needs nothing else."""
    release = threading.Event()
    real = session.scope.illumination.leds_off
    ended = []

    def leds_off():
        assert release.wait(10)
        ended.append(1)
        return real()

    # a stand-in by design: a call still running when the close finishes its clients.
    monkeypatch.setattr(session.scope.illumination, 'leds_off', leds_off)
    finished = threading.Event()
    real_finish = rest.app.Closing.finish

    def finish(self):
        real_finish(self)
        finished.set()

    # a stand-in by design: tells the test the close has answered its
    # clients and reached its join.
    monkeypatch.setattr(rest.app.Closing, 'finish', finish)
    try:
        with _serving(session, shut_down=lambda: None) as (server, client, _closing):
            assert (
                client.post(
                    '/api/v1/scope/illumination/leds_off', headers={'Prefer': 'wait=0'}
                ).status_code
                == 202
            )
            closer, answered = _closed_on_a_thread(server)
            assert finished.wait(10)
            closer.join(0.2)
            assert closer.is_alive() and ended == []
            release.set()
            closer.join(10)
    finally:
        release.set()

    assert (answered, ended) == ([True], [1])
