# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The event stream carries what the scope tells its listeners, to a wire client.

``GET /api/v1/events`` (``rest.events``): the first event is ``status``;
each listener member's event follows as its declared record; a run's events
each name the run by its job, its first ones included, which can come
before the call that started it returns; every job's end is ``job_ended``,
with the job, a run's after its ``run_ended``; a field that can hold a live
object is left out whatever it holds, as the stream's description says;
an outcome is on the stream only
when nobody asked for it, once per ``outcome_id``; a reconnect with
``Last-Event-ID`` is sent what it missed, or ``reset`` and ``status`` once
that is no longer held; a quiet stream carries a comment; a stream that
has closed hears nothing more; ``closing`` is the last event of every
stream, and the only one of a stream begun or resumed after it; the first
event says how soon to reconnect (``retry:``). A poll
(``GET /api/v1/events/held?after=<id>``) answers what a stream resumed from
that id is sent, as JSON, and the id to ask after next.

Read from a real uvicorn server: a stream never ends, and only a socket
lets a client stop reading one.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import threading

import httpx
import pytest
import uvicorn

import rest.events
from modules import wire_encoding
from modules.api_surface import api_fields
from modules.notification_center import notifications
from modules.protocol import Protocol
from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0


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
def _serving(session, app=None):
    server = uvicorn.Server(
        uvicorn.Config(app or build_app(session), host='127.0.0.1', port=0, log_level='warning')
    )
    thread = threading.Thread(target=server.run)
    thread.start()
    try:
        while not server.started:
            assert thread.is_alive(), 'the server ended before it started'
            thread.join(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=10) as client:
            client.server = server
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
    assert set(first['data']) == {
        'live_work',
        'lock',
        'control',
        'run_refused',
        'axes',
        'turret',
        'parts',
        'camera_streaming',
    }


def test_a_listener_members_event_is_its_record(client):
    with _reading(client) as stream:
        stream.until('status')
        assert (
            client.post(
                '/api/v1/scope/illumination/led_on',
                json={'channel': LED0, 'illumination_ma': 10.0},
            ).status_code
            == 200
        )
        led = stream.until('led')[-1]

    assert led['data'] == {'address': 0, 'name': LED0, 'state': 'lit', 'illumination_ma': 10.0}
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
    real = the_runner.start_single_scan

    def returns_after_the_first_event(*args, events, **kwargs):
        told = threading.Event()

        def scan_started(*a):
            events.scan_started(*a)
            told.set()

        run = real(*args, events=dataclasses.replace(events, scan_started=scan_started), **kwargs)
        assert told.wait(10)
        return run

    monkeypatch.setattr(the_runner, 'start_single_scan', returns_after_the_first_event)

    with _reading(client) as stream:
        stream.until('status')
        run = client.post(
            f'/api/v1/handles/ProtocolRunner/{runner}/start_single_scan',
            json={'protocol': protocol, 'enable_image_saving': False},
        ).json()
        events = stream.until('run_ended')
        ended = stream.until('job_ended')[-1]

    run_events = [e for e in events if e['event'] in {'scan_started', 'step_started', 'run_ended'}]
    assert run_events[0]['event'] == 'scan_started'
    # The call answers the run's job; every one of its events names that job.
    assert run['member'].endswith('/start_single_scan'), run
    assert {json.dumps(e['data']['run']) for e in run_events} == {json.dumps({'job': run['id']})}
    # The run's own copy of its protocol is no client's: it is left out.
    assert 'protocol' not in run_events[-1]['data']
    # A path crosses as its name in the live folder.
    assert run_events[-1]['data']['run_dir']['name'].startswith('ProtocolData/')
    # The run's job ends after its run_ended, with the run's outcome.
    assert (ended['data']['id'], ended['data']['status']) == (run['id'], 'completed')
    assert ended['data']['result']['status'] == 'completed'


def _run_one_step(client, session):
    home_sim_scope(session.scope)
    session.settings['BF']['acquire'] = 'image'
    protocol = client.post('/api/v1/create_empty_protocol').json()['handle']
    assert client.post('/api/v1/add_step', json={'protocol': protocol}).status_code == 200
    runner = client.post('/api/v1/create_protocol_runner').json()['handle']
    return client.post(
        f'/api/v1/handles/ProtocolRunner/{runner}/start_single_scan',
        json={'protocol': protocol, 'enable_image_saving': False},
    )


def test_each_run_state_edge_sends_the_status_in_the_order_heard(client, session):
    with _reading(client) as stream:
        stream.until('status')
        assert _run_one_step(client, session).status_code in (200, 202)
        events = stream.until('job_ended')

        def last_state():
            return [e for e in events if e.get('event') == 'run_state'][-1]['data']

        # The run lets go before its job ends, but a file write can hold the
        # controls a moment longer; its release is an edge of its own.
        while last_state()['lock']['run_lockout']:
            events.append(stream.next())

    states = [e['data'] for e in events if e['event'] == 'run_state']
    assert {
        'live_work',
        'lock',
        'control',
        'run_refused',
        'axes',
        'turret',
        'parts',
        'camera_streaming',
    } == set(states[0])
    lockouts = [s['lock']['run_lockout'] for s in states]
    # Held from the run's grant until it lets go, and never held again after.
    assert True in lockouts, lockouts
    first_held = lockouts.index(True)
    last_held = len(lockouts) - 1 - lockouts[::-1].index(True)
    assert all(lockouts[first_held : last_held + 1]), lockouts
    assert not any(lockouts[last_held + 1 :]) and not lockouts[-1], lockouts
    assert states[first_held]['run_refused']['reason'] == 'already_running'
    assert states[-1]['live_work']['work'] == [] and states[-1]['run_refused'] is None


def test_a_status_read_waiting_on_the_camera_does_not_hold_the_edge(client, session, monkeypatch):
    # A stand-in by design: the simulator cannot yet hold its camera in a
    # grab stop, which on a Basler makes is_streaming wait (P-16, measured).
    released = threading.Event()
    real = session.scope.imaging.is_streaming
    transitioning = threading.current_thread()
    asked_on = []

    def waits_on_a_grab_stop():
        asked_on.append(threading.current_thread())
        if threading.current_thread() is not transitioning:
            released.wait(10)
        return real()

    with _reading(client) as stream:
        stream.until('status')
        monkeypatch.setattr(session.scope.imaging, 'is_streaming', waits_on_a_grab_stop)
        try:
            with session.diagnostic_claim():
                pass
        finally:
            released.set()
        events = stream.until('run_state')

    # The grant and the release are made on this thread; the camera is never
    # asked on it, and the grant's event still says what held the scope.
    assert asked_on and transitioning not in asked_on, asked_on
    assert events[-1]['data']['lock']['run_named'] == 'A diagnostic is in progress.'


@api_fields('outcome', 'protocol')
@dataclasses.dataclass(frozen=True)
class _EndedWithoutProtocol:
    outcome: str
    protocol: Protocol | None


def test_a_field_that_can_hold_a_live_object_is_left_out_while_it_holds_none(session, live):
    # No event's record declares an optional live object yet, so the record
    # is the test's own, sent on a stream the way the server builds one; the
    # server does not hand its stream out, so this reads the stream itself.
    classes, aliases = wire_encoding.project_classes(), wire_encoding.project_aliases()
    stream = rest.events.EventStream(
        session,
        lambda value: wire_encoding.encode(
            value,
            live_folder=live,
            handle=lambda obj: pytest.fail('nothing to hand out'),
            job=lambda future: pytest.fail('no job'),
        ),
        wire_encoding.handed_out(ScopeSession, classes, aliases),
    )

    async def the_run_ended_event() -> str:
        stream.open(asyncio.get_running_loop())
        try:
            reading = stream.read(None)
            await anext(reading)
            stream.send('run_ended', _EndedWithoutProtocol('completed', None))
            async for frame in reading:
                if frame.startswith('id: ') and '\nevent: run_ended\n' in frame:
                    return frame
            raise AssertionError('the stream ended')
        finally:
            stream.close()

    frame = asyncio.run(the_run_ended_event())

    assert json.loads(frame.split('data: ', 1)[1]) == {'outcome': 'completed'}


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
            '/api/v1/scope/illumination/led_on', json={'channel': LED0, 'illumination_ma': 10.0}
        )
        events = stream.until('led')

    outcomes = [e['data'] for e in events if e['event'] == 'outcome']
    assert [o['category'] for o in outcomes] == ['Unasked']
    assert outcomes[0]['solicited'] is False


def test_a_reconnect_is_sent_what_it_missed_and_no_status(client):
    led = {'channel': LED0, 'illumination_ma': 10.0}
    with _reading(client) as stream:
        stream.until('status')
        client.post('/api/v1/scope/illumination/led_on', json=led)
        last = stream.until('led')[-1]['id']
    client.post('/api/v1/scope/illumination/led_off', json={'channel': LED0})

    with _reading(client, **{'Last-Event-ID': last}) as stream:
        resent = stream.until('led')

    assert resent[0]['event'] != 'status'
    assert int(resent[0]['id']) == int(last) + 1
    assert resent[-1]['data']['state'] == 'dark'


def test_a_reconnect_past_what_is_held_is_sent_reset_then_status(session, monkeypatch):
    monkeypatch.setattr(rest.events, 'BUFFER', 2)
    with _serving(session) as client:
        with _reading(client) as stream:
            stream.until('status')
            for _ in range(3):
                client.post(
                    '/api/v1/scope/illumination/led_on',
                    json={'channel': LED0, 'illumination_ma': 10.0},
                )
                client.post('/api/v1/scope/illumination/led_off', json={'channel': LED0})
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


def test_a_stream_that_has_closed_hears_nothing_more(session):
    # Served twice: a stream still hearing after its first close would send
    # each event of the second serving twice.
    app = build_app(session)
    with _serving(session, app):
        pass
    with _serving(session, app) as client, _reading(client) as stream:
        stream.until('status')
        client.post(
            '/api/v1/scope/illumination/led_on',
            json={'channel': LED0, 'illumination_ma': 10.0},
        )
        first = stream.until('led')[-1]
        client.post('/api/v1/scope/illumination/led_off', json={'channel': LED0})
        second = stream.until('led')[-1]

    assert first['data']['state'] == 'lit'
    assert second['data']['state'] == 'dark'


def test_closing_is_the_last_event_of_every_stream_and_of_one_begun_after(session):
    app = build_app(session)
    with _serving(session, app) as client:
        with client.stream('GET', '/api/v1/events') as response:
            stream = _Stream(response)
            stream.until('status')
            loop = client.server.servers[0].get_loop()
            loop.call_soon_threadsafe(app.state.closing.finish)
            last = stream.until('closing')[-1]
            with pytest.raises(AssertionError, match='the stream ended'):
                stream.next()
        with _reading(client) as late:
            only = late.next()
            with pytest.raises(AssertionError, match='the stream ended'):
                late.next()

    assert last['data'] == {}
    assert (only['event'], only['id']) == ('closing', last['id'])


def test_every_jobs_end_is_an_event_with_the_job(client, session, monkeypatch):
    """A move's, a still's and a blocking call's held as a job: each one ``job_ended``."""
    home_sim_scope(session.scope)
    release = threading.Event()
    real = session.scope.illumination.leds_off

    def leds_off():
        assert release.wait(10)
        return real()

    # a stand-in by design: holds the call past the client's wait so it is
    # held as a job; the simulated board answers at once.
    monkeypatch.setattr(session.scope.illumination, 'leds_off', leds_off)
    with _reading(client) as stream:
        stream.until('status')
        move = client.post('/api/v1/scope/motion/start_move_z_by', json={'dz_um': 10.0}).json()
        still = client.post('/api/v1/manual_capture/capture', json={'false_color_on': False}).json()
        held = client.post(
            '/api/v1/scope/illumination/leds_off', headers={'Prefer': 'respond-async'}
        ).json()
        release.set()
        ended = {}
        while len(ended) < 3:
            event = stream.until('job_ended')[-1]
            ended[event['data']['id']] = event['data']

    assert set(ended) == {move['id'], still['id'], held['id']}
    assert {e['status'] for e in ended.values()} == {'completed'}
    assert ended[held['id']]['member'] == 'scope/illumination/leds_off'


def test_a_stream_resumed_from_closing_is_sent_closing_and_nothing_after(session, monkeypatch):
    """A job that ends after ``closing`` sends nothing a reader could be sent."""
    release = threading.Event()
    real = session.scope.illumination.leds_off

    def leds_off():
        assert release.wait(10)
        return real()

    # a stand-in by design: holds the call so it ends after the close.
    monkeypatch.setattr(session.scope.illumination, 'leds_off', leds_off)
    app = build_app(session)
    with _serving(session, app) as client:
        held = client.post(
            '/api/v1/scope/illumination/leds_off', headers={'Prefer': 'respond-async'}
        ).json()
        with _reading(client) as stream:
            stream.until('status')
            loop = client.server.servers[0].get_loop()
            loop.call_soon_threadsafe(app.state.closing.finish)
            closing = stream.until('closing')[-1]
        release.set()
        # Joined, not long-polled: once the close has finished, a wait on a
        # job is answered at once. A call's thread ends after its job settles.
        assert app.state.closing.join_jobs(10) == []
        assert client.get(f'/api/v1/jobs/{held["id"]}').json()['status'] == 'completed'

        with _reading(client, **{'Last-Event-ID': closing['id']}) as resumed:
            only = resumed.next()
            with pytest.raises(AssertionError, match='the stream ended'):
                resumed.next()

    assert (only['event'], only['id']) == ('closing', closing['id'])


def test_the_first_event_says_how_soon_to_reconnect(client):
    with _reading(client) as stream:
        first = stream.next()

    assert (first['event'], first['retry']) == ('status', str(rest.events.RECONNECT_MS))


def test_a_poll_answers_the_events_held_after_an_id_and_the_id_to_ask_after_next(client):
    begun = client.get('/api/v1/events/held')
    assert begun.status_code == 200
    (status,) = begun.json()['events']
    assert status['event'] == 'status'
    assert begun.json()['next'] == status['id']

    client.post(
        '/api/v1/scope/illumination/led_on', json={'channel': LED0, 'illumination_ma': 10.0}
    )
    polled = client.get('/api/v1/events/held', params={'after': begun.json()['next']}).json()

    leds = [e for e in polled['events'] if e['event'] == 'led']
    assert leds[-1]['data']['state'] == 'lit'
    assert polled['next'] == polled['events'][-1]['id'] > status['id']
    assert client.get('/api/v1/events/held', params={'after': polled['next']}).json() == {
        'events': [],
        'next': polled['next'],
    }


def test_a_poll_after_an_id_no_longer_held_answers_reset_then_status(session, monkeypatch):
    monkeypatch.setattr(rest.events, 'BUFFER', 2)
    with _serving(session) as client:
        for _ in range(3):
            client.post(
                '/api/v1/scope/illumination/led_on',
                json={'channel': LED0, 'illumination_ma': 10.0},
            )
            client.post('/api/v1/scope/illumination/led_off', json={'channel': LED0})

        polled = client.get('/api/v1/events/held', params={'after': 0}).json()

    assert [e['event'] for e in polled['events']] == ['reset', 'status']
    assert polled['next'] == polled['events'][-1]['id']


def test_a_poll_once_the_server_closes_ends_with_closing(session):
    app = build_app(session)
    with _serving(session, app) as client:
        before = client.get('/api/v1/events/held').json()['next']
        loop = client.server.servers[0].get_loop()
        # Run on the loop ahead of any request asked after it.
        loop.call_soon_threadsafe(app.state.closing.finish)
        polled = client.get('/api/v1/events/held', params={'after': before}).json()
        after = client.get('/api/v1/events/held', params={'after': polled['next']}).json()

    assert polled['events'][-1]['event'] == 'closing'
    assert [e['event'] for e in after['events']] == ['closing']
