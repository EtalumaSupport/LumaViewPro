# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The event stream carries what the scope tells its listeners, to a wire client.

``GET /api/v1/events`` (``rest.events``): the first event is ``status``;
each listener member's event follows as its declared record; a run's events
each name the run by its handle, its first ones included, which can come
before the call that started it returns; an outcome is on the stream only
when nobody asked for it, once per ``outcome_id``; a reconnect with
``Last-Event-ID`` is sent what it missed, or ``reset`` and ``status`` once
that is no longer held; a quiet stream carries a comment.

Read from a real uvicorn server: a stream never ends, and only a socket
lets a client stop reading one.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import threading

import httpx
import pytest
import uvicorn

import rest.events
from modules.notification_center import notifications
from modules.scope_session import ScopeSession
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


@pytest.fixture
def client(session):
    with _serving(session) as client:
        yield client


class _Stream:
    """One client's read of the stream, event by event."""

    def __init__(self, response: httpx.Response) -> None:
        self._lines = response.iter_lines()

    def next(self) -> dict:
        """The next event or comment: ``{"id", "event", "data"}``, or ``{"comment"}``."""
        event: dict = {}
        for line in self._lines:
            if not line:
                if event:
                    return event
                continue
            if line.startswith(':'):
                return {'comment': line[1:].strip()}
            field, _, value = line.partition(': ')
            event[field] = json.loads(value) if field == 'data' else value
        raise AssertionError('the stream ended')

    def until(self, name: str) -> list[dict]:
        """Every event up to and including the first named *name*."""
        seen = []
        while True:
            event = self.next()
            if 'comment' in event:
                continue
            seen.append(event)
            if event['event'] == name:
                return seen


@contextlib.contextmanager
def _reading(client, **headers):
    with client.stream('GET', '/api/v1/events', headers=headers) as response:
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('text/event-stream')
        yield _Stream(response)


def test_the_first_event_is_the_status_with_the_current_id(client):
    with _reading(client) as stream:
        first = stream.next()

    assert first['event'] == 'status'
    assert first['id'].isdigit()
    assert set(first['data']) == {'live_work', 'axes', 'parts', 'camera_streaming'}


def test_a_listener_members_event_is_its_record(client):
    with _reading(client) as stream:
        stream.until('status')
        assert (
            client.post(
                '/api/v1/scope/illumination/led_on',
                json={'channel': 'Blue', 'illumination_ma': 10.0},
            ).status_code
            == 200
        )
        led = stream.until('led')[-1]

    assert led['data'] == {'channel': 'Blue', 'on': True, 'illumination_ma': 10.0}
    assert int(led['id']) > 0


def test_every_event_of_a_run_names_the_run_its_call_handed_out(client, session, monkeypatch):
    home_sim_scope(session.scope)
    session.settings['BF']['acquire'] = 'image'
    protocol = client.post('/api/v1/create_empty_protocol').json()['handle']
    assert client.post('/api/v1/add_step', json={'protocol': protocol}).status_code == 200
    runner = client.post('/api/v1/create_protocol_runner').json()['handle']
    # The run is dispatched before its call returns, so its first event can
    # come first; this call returns only once it has, every time.
    the_runner = session.create_protocol_runner()
    real = the_runner.run_single_scan

    def returns_after_the_first_event(*args, events, **kwargs):
        told = threading.Event()

        def scan_started(*a):
            events.scan_started(*a)
            told.set()

        run = real(*args, events=dataclasses.replace(events, scan_started=scan_started), **kwargs)
        assert told.wait(10)
        return run

    monkeypatch.setattr(the_runner, 'run_single_scan', returns_after_the_first_event)

    with _reading(client) as stream:
        stream.until('status')
        run = client.post(
            f'/api/v1/handles/ProtocolRunner/{runner}/run_single_scan',
            json={'protocol': protocol, 'enable_image_saving': False},
            # The call's answer, not its job: a loaded machine can take
            # longer than the default wait to reach the first event.
            headers={'Prefer': 'wait=60'},
        ).json()
        events = stream.until('run_ended')

    run_events = [e for e in events if e['event'] in {'scan_started', 'step_started', 'run_ended'}]
    assert run_events[0]['event'] == 'scan_started'
    assert {json.dumps(e['data']['run']) for e in run_events} == {json.dumps(run)}
    assert run['type'] == 'RunHandle'
    # The run's own copy of its protocol is no client's: it is left out.
    assert 'protocol' not in run_events[-1]['data']
    # A path crosses as its name in the live folder.
    assert run_events[-1]['data']['run_dir']['name'].startswith('ProtocolData/')


def test_an_outcome_nobody_asked_for_is_sent_once_and_an_answered_one_never(client):
    unasked = RuntimeError('unasked outcome')
    with _reading(client) as stream:
        stream.until('status')
        notifications.report_outcome(
            RuntimeError('answered outcome'), solicited=True, category='Answered'
        )
        notifications.report_outcome(unasked, solicited=False, category='Unasked')
        notifications.report_outcome(unasked, solicited=False, category='Unasked')
        client.post(
            '/api/v1/scope/illumination/led_on', json={'channel': 'Blue', 'illumination_ma': 10.0}
        )
        events = stream.until('led')

    outcomes = [e['data'] for e in events if e['event'] == 'outcome']
    assert [o['category'] for o in outcomes] == ['Unasked']
    assert outcomes[0]['solicited'] is False


def test_a_reconnect_is_sent_what_it_missed_and_no_status(client):
    led = {'channel': 'Blue', 'illumination_ma': 10.0}
    with _reading(client) as stream:
        stream.until('status')
        client.post('/api/v1/scope/illumination/led_on', json=led)
        last = stream.until('led')[-1]['id']
    client.post('/api/v1/scope/illumination/led_off', json={'channel': 'Blue'})

    with _reading(client, **{'Last-Event-ID': last}) as stream:
        resent = stream.until('led')

    assert resent[0]['event'] != 'status'
    assert int(resent[0]['id']) == int(last) + 1
    assert resent[-1]['data']['on'] is False


def test_a_reconnect_past_what_is_held_is_sent_reset_then_status(session, monkeypatch):
    monkeypatch.setattr(rest.events, 'BUFFER', 2)
    with _serving(session) as client:
        with _reading(client) as stream:
            stream.until('status')
            for _ in range(3):
                client.post(
                    '/api/v1/scope/illumination/led_on',
                    json={'channel': 'Blue', 'illumination_ma': 10.0},
                )
                client.post('/api/v1/scope/illumination/led_off', json={'channel': 'Blue'})
            stream.until('led')

        with _reading(client, **{'Last-Event-ID': '0'}) as stream:
            first, second = stream.next(), stream.next()

    assert (first['event'], second['event']) == ('reset', 'status')
    assert second['id'] == first['id']


def test_a_quiet_stream_carries_a_comment(client, monkeypatch):
    monkeypatch.setattr(rest.events, 'KEEPALIVE_S', 0.05)
    with _reading(client) as stream:
        stream.until('status')
        quiet = stream.next()

    assert quiet == {'comment': 'keepalive'}
