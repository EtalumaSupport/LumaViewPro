# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The server closes the session with its clients connected (``rest.server``).

On SIGINT or SIGTERM a new connection is refused and a member asked on an
open one is refused ``server_closing``, while jobs and the stream are still
served; the session closes while the stream carries what the close reports;
then ``closing`` ends every stream and the server stops. A later signal
changes nothing, and the calls' threads are joined with a bound, the ones
still running named.

Driven in-process: the server runs on a thread and its signal handler is
called as the signal would call it.
"""

from __future__ import annotations

import contextlib
import json
import signal
import threading

import httpx
import pytest
import uvicorn

from modules.scope_session import ScopeSession
from rest.app import build_app
from rest.server import Server
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
    app = build_app(session)
    server = Server(
        uvicorn.Config(app, host='127.0.0.1', port=0, log_level='warning'),
        session,
        app.state.closing,
    )
    thread = threading.Thread(target=server.run)
    thread.start()
    try:
        while not server.started:
            assert thread.is_alive(), 'the server ended before it started'
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


def test_a_signal_closes_the_session_with_its_clients_connected(session, monkeypatch):
    # A stand-in by design: the subject is the server's close, and the
    # session's close is held open so the test can ask during it.
    asked = threading.Event()
    release = threading.Event()
    real_shutdown = session.shutdown

    def held_shutdown():
        asked.set()
        assert release.wait(10)
        real_shutdown()

    monkeypatch.setattr(session, 'shutdown', held_shutdown)

    with (
        _serving(session) as (server, client, _closing),
        httpx.Client(base_url=client.base_url, timeout=10) as connected,
    ):
        lit = connected.post(
            '/api/v1/scope/illumination/led_on', json={'channel': 'Blue', 'illumination_ma': 10.0}
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
    assert {'channel': 'Blue', 'on': False, 'illumination_ma': 0.0} in [
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
