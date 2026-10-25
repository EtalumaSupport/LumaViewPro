# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A wire client is refused in the words, and with the reason, a Python caller is.

Every answer that is not a result is an RFC 9457 problem
(``application/problem+json``): ``type`` names the reason
(``urn:lumascope:problem:<reason>``, or ``about:blank`` with none), ``title``
and ``detail`` are the outcome's own, ``instance`` names the request, and
``kind``, ``reason`` and ``remedy`` are what a client branches on, beside
the fields the outcome's type publishes (a refused argument's name). A
member's outcome is read as every host reads it (``outcome_of``) and
reported once, to the log only: the problem is the answer. A refusal is 422
when the request as sent cannot succeed and 409 when the scope's state
refused it, as its type declares, so a client retries only a 409; a fault is
500. The server's own answers -- no such route, method, handle or body --
carry their own reasons. The OpenAPI description declares every answer that
is not a result as such a problem, on every route.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from modules.exceptions import (
    ArgumentRefusedError,
    RunWaitExpiredError,
    LiveFolderPathRefusedError,
    Remedy,
)
from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

UUID_URN = re.compile(r'urn:uuid:[0-9a-f-]{36}')


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
def reported(monkeypatch):
    calls = []
    monkeypatch.setattr(
        'modules.notification_center.notifications.report_outcome',
        lambda exc, **kw: calls.append((exc, kw)),
    )
    return calls


def _problem(answer, status):
    assert answer.status_code == status
    assert answer.headers['content-type'] == 'application/problem+json'
    body = answer.json()
    assert body['status'] == status
    assert UUID_URN.fullmatch(body['instance'])
    return body


def test_a_refusal_is_the_python_callers_refusal_and_is_logged_once(client, session, reported):
    with pytest.raises(LiveFolderPathRefusedError) as python:
        session.live_folder_path('../outside')

    # The request as sent can never succeed: 422.
    body = _problem(client.post('/api/v1/make_logs_zip', json={'output_dir': '../outside'}), 422)

    assert body['type'] == 'urn:lumascope:problem:outside_live_folder'
    assert (body['title'], body['detail']) == (python.value.title, str(python.value))
    assert (body['kind'], body['reason'], body['remedy']) == (
        'refusal',
        'outside_live_folder',
        None,
    )
    ((exception, how),) = reported
    assert isinstance(exception, LiveFolderPathRefusedError)
    assert how == {'solicited': True, 'category': 'REST', 'log_only': True}


def test_a_refusal_the_scopes_state_gave_is_409(client, live):
    # An unplugged drive: the same request succeeds once it is back.
    live.rmdir()

    body = _problem(client.post('/api/v1/make_logs_zip', json={'output_dir': 'reports'}), 409)

    assert body['reason'] == 'capture_location_unusable'


def test_a_fault_is_500_titled_by_its_class_in_its_own_words(client, session, monkeypatch):
    def fails():
        raise ValueError('bad shape (3,)')

    monkeypatch.setattr(session.scope.illumination, 'leds_off', fails)

    body = _problem(client.post('/api/v1/scope/illumination/leds_off'), 500)

    assert (body['type'], body['title'], body['detail']) == (
        'about:blank',
        'ValueError',
        'bad shape (3,)',
    )
    assert (body['kind'], body['reason']) == ('fault', None)


def test_a_hardware_fault_answers_500_naming_which_fault_it_was(client, session, monkeypatch):
    """Each fault a driver raises answers its own reason and title, never one shared by all."""
    import drivers.exceptions as driver_exceptions
    from drivers.exceptions import HardwareError, MotionInterlockError

    faults = [
        cls
        for cls in vars(driver_exceptions).values()
        if isinstance(cls, type)
        and issubclass(cls, HardwareError)
        # Built with its interlock's reason; the API answers it as a refusal.
        and cls not in (HardwareError, MotionInterlockError)
    ]
    assert len(faults) >= 10, 'the discovery drifted'
    wrong = {}
    for cls in faults:

        def fails(cls=cls):
            raise cls('the driver said so')

        monkeypatch.setattr(session.scope.illumination, 'leds_off', fails)
        body = client.post('/api/v1/scope/illumination/leds_off').json()
        got = (body['status'], body['kind'], body['reason'], body['type'], body['title'])
        want = (500, 'fault', cls.reason, f'urn:lumascope:problem:{cls.reason}', cls.title)
        if got != want:
            wrong[cls.__name__] = got
    assert wrong == {}
    assert len({cls.reason for cls in faults}) == len(faults), 'two faults share a reason'


def test_a_move_another_request_stopped_answers_409_as_a_notice(client, session):
    home_sim_scope(session.scope)
    held = session.scope.simulation.hold_axis('X')
    started = client.post(
        '/api/v1/scope/motion/move_xy',
        json={'x_um': 60000},
        headers={'Prefer': 'wait=0'},
    )
    assert started.status_code == 202
    assert held.wait(10), 'the move never got part of the way'

    assert client.post('/api/v1/scope/motion/stop_motion').status_code == 200
    job = client.get(started.headers['Location'], headers={'Prefer': 'wait=10'}).json()

    assert job['status'] == 'failed'
    error = job['error']
    assert (error['status'], error['kind'], error['reason'], error['title']) == (
        409,
        'notice',
        'stopped',
        'Move Stopped',
    )
    assert (
        error['detail'] == 'The X move did not complete: the motors were stopped before it arrived.'
    )


def test_a_quiet_outcome_answers_by_its_cause_and_says_it_is_quiet(client, session, monkeypatch):
    def expired():
        raise RunWaitExpiredError('run_composite', 5.0, None)

    monkeypatch.setattr(session.scope.illumination, 'leds_off', expired)

    # The run may still end with the caller changing nothing: the scope's
    # state's.
    body = _problem(client.post('/api/v1/scope/illumination/leds_off'), 409)

    assert (body['kind'], body['reason'], body['detail']) == (
        'quiet',
        'still_running',
        'run_composite: the run was still going when the wait ran out after 5 s; '
        "it goes on, held by this error's operation",
    )


def test_a_refusal_carries_the_fields_its_type_publishes(client, session, monkeypatch):
    def refused():
        raise ArgumentRefusedError('not_a_number', argument='illumination_ma', value='bright')

    monkeypatch.setattr(session.scope.illumination, 'leds_off', refused)

    body = _problem(client.post('/api/v1/scope/illumination/leds_off'), 422)

    # A client corrects the request from the fields, not the words.
    assert (body['reason'], body['argument']) == ('not_a_number', 'illumination_ma')


def test_a_name_that_is_no_axis_answers_with_the_names_the_member_takes(client):
    body = _problem(
        client.post('/api/v1/scope/motion/set_precision_mode', json={'axis': 'x', 'enabled': True}),
        422,
    )

    assert (body['reason'], body['argument'], body['offered']) == (
        'not_offered',
        'axis',
        ['X', 'Y', 'Z', 'T'],
    )


@pytest.mark.parametrize(
    ('method', 'route', 'body', 'reason', 'published'),
    [
        (
            'PUT',
            'channel_acquire',
            {'channel': 'BF', 'mode': 'bogus'},
            'not_offered',
            {'argument': 'mode', 'offered': ['image', 'video']},
        ),
        (
            'PUT',
            'scope/camera/image_mode',
            {'mode': 'bogus'},
            'not_offered',
            {'argument': 'mode'},
        ),
        (
            'POST',
            'update_settings',
            {'path': 'image_mode', 'value': '8bit'},
            'has_member',
            {'path': 'image_mode', 'member': 'scope.camera.set_image_mode'},
        ),
    ],
)
def test_a_session_argument_its_member_refuses_is_422_names_it_and_changes_nothing(
    client, session, method, route, body, reason, published
):
    before = session.get_settings_snapshot()

    answer = _problem(client.request(method, f'/api/v1/{route}', json=body), 422)

    assert answer['reason'] == reason
    assert {name: answer[name] for name in published} == published
    assert session.get_settings_snapshot() == before


def test_a_reads_argument_its_member_refuses_is_422_and_names_it(client):
    """A read's query argument the member refuses is the member's refusal, as a body's is."""
    answer = _problem(client.get('/api/v1/get_setting', params={'path': 'bogus'}), 422)

    assert (answer['reason'], answer['argument']) == ('not_a_setting', 'path')


def test_a_setting_only_the_host_moves_is_refused_as_host_only_not_sent_to_a_missing_route(
    client, session
):
    """The live folder confines every path a wire caller sends, so its two
    setters are in-process. A wire write to either setting is refused as the
    host's, never with advice to call a member the wire has no route for."""
    before = session.get_settings_snapshot()
    for path, member in (
        ('live_folder', 'set_live_folder'),
        ('protocol.filepath', 'set_protocol_filepath'),
    ):
        body = {'path': path, 'value': '/elsewhere'}
        answer = _problem(client.post('/api/v1/update_settings', json=body), 422)
        assert (answer['reason'], answer['member']) == ('host_only', member)
        assert client.post(f'/api/v1/{member}', json={}).status_code == 404
    assert session.get_settings_snapshot() == before


def test_a_refusals_remedy_is_sent_as_the_record_apply_remedy_takes(client, session, monkeypatch):
    remedy = Remedy('recover_file_writer', 'Recover', 'Wait')
    refusal = LiveFolderPathRefusedError('outside_live_folder', 'x', 'Not there.')
    refusal.remedy = remedy

    def refuses():
        raise refusal

    monkeypatch.setattr(session.scope.illumination, 'leds_off', refuses)

    body = _problem(client.post('/api/v1/scope/illumination/leds_off'), 422)

    assert body['remedy'] == {
        'member': 'recover_file_writer',
        'confirm_text': 'Recover',
        'cancel_text': 'Wait',
    }


@pytest.mark.parametrize(
    ('ask', 'status', 'reason'),
    [
        (lambda c: c.get('/api/v1/no_such_member'), 404, 'not_found'),
        (lambda c: c.get('/api/v1/scope/illumination/leds_off'), 405, 'method_not_allowed'),
        (lambda c: c.get('/api/v1/handles/Protocol/9/num_steps'), 404, 'not_found'),
        (lambda c: c.get('/api/v1/status?verbose=1'), 422, 'invalid_request'),
        (
            lambda c: c.post(
                '/api/v1/scope/illumination/led_on',
                content=b'{"channel": "\xff\xfe"}',
                headers={'content-type': 'application/json'},
            ),
            422,
            'invalid_request',
        ),
        (
            lambda c: c.post('/api/v1/scope/illumination/led_on', json={'channel': 'Red'}),
            422,
            'invalid_request',
        ),
        (
            lambda c: c.post(
                '/api/v1/scope/illumination/led_off',
                content='channel=Red',
                headers={'content-type': 'text/plain'},
            ),
            415,
            'unsupported_media_type',
        ),
    ],
    ids=[
        'no route',
        'wrong method',
        'no handle',
        'a query string',
        'a body that is not text',
        'a missing argument',
        'not JSON',
    ],
)
def test_the_servers_own_answers_carry_their_own_reasons(client, reported, ask, status, reason):
    body = _problem(ask(client), status)

    assert body['type'] == f'urn:lumascope:problem:{reason}'
    assert (body['kind'], body['reason']) == ('refusal', reason)
    # Nothing reached a member, so there is no outcome to report.
    assert reported == []


def test_a_cancel_that_declares_no_cause_is_409(client, session, monkeypatch):
    from concurrent.futures import CancelledError

    def cancelled():
        raise CancelledError()

    monkeypatch.setattr(session.scope.illumination, 'leds_off', cancelled)

    body = _problem(client.post('/api/v1/scope/illumination/leds_off'), 409)

    assert body['kind'] == 'quiet'


def test_a_call_the_server_cannot_start_is_a_fault_and_holds_no_count(client, monkeypatch):
    import threading

    import rest.jobs

    def cannot(_self):
        raise RuntimeError("can't start new thread")

    # The answer is what is under test, so the client does not re-raise it.
    with (
        TestClient(client.app, raise_server_exceptions=False) as fresh,
        monkeypatch.context() as patch,
    ):
        patch.setattr(threading.Thread, 'start', cannot)
        failed = fresh.get('/api/v1/app_version')

    body = _problem(failed, 500)
    assert (body['kind'], body['detail']) == ('fault', "can't start new thread")
    # The failed start left no call counted: one call still fits under a limit of one.
    monkeypatch.setattr(rest.jobs, 'LIVE_LIMIT', 1)
    assert client.get('/api/v1/app_version').status_code == 200


def test_an_invalid_body_names_each_argument_that_does_not_fit(client):
    body = _problem(client.post('/api/v1/scope/illumination/led_on', json={'channel': 1.5}), 422)

    assert {tuple(e['loc']) for e in body['errors']} >= {('body', 'illumination_ma')}
    assert all(set(e) == {'loc', 'msg', 'type'} for e in body['errors'])


def test_the_openapi_declares_every_answer_that_is_not_a_result_as_a_problem(client):
    openapi = client.get('/api/v1/openapi.json').json()
    operations = [op for path in openapi['paths'].values() for op in path.values()]
    refused = client.post(
        '/api/v1/scope/motion/set_precision_mode', json={'axis': 'x', 'enabled': True}
    )

    # Every route keeps the ranges; a status its member declares (its
    # Raises: section) is a problem too.
    missing_ranges = [
        op['operationId'] for op in operations if not {'4XX', '5XX'} <= set(op['responses'])
    ]
    media = {
        media
        for op in operations
        for code, response in op['responses'].items()
        if not code.startswith('2')
        for media in response['content']
    }
    schema = operations[0]['responses']['4XX']['content']['application/problem+json']['schema']
    assert missing_ranges == []
    assert media == {'application/problem+json'}
    assert 'HTTPValidationError' not in str(openapi)
    assert set(schema['required']) <= set(refused.json())
