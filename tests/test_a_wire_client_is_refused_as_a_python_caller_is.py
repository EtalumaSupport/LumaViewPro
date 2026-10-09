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
carry their own reasons.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from modules.exceptions import (
    ArgumentRefusedError,
    LiveFolderPathRefusedError,
    Remedy,
    RunAlreadyEndedError,
)
from modules.scope_session import ScopeSession
from rest.app import build_app
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


def test_a_quiet_outcome_answers_by_its_cause_and_says_it_is_quiet(client, session, monkeypatch):
    def ended():
        raise RunAlreadyEndedError('The run has already ended.')

    monkeypatch.setattr(session.scope.illumination, 'leds_off', ended)

    # An ended run's handle never names a live run again: the request's.
    body = _problem(client.post('/api/v1/scope/illumination/leds_off'), 422)

    assert (body['kind'], body['reason'], body['detail']) == (
        'quiet',
        'run_already_ended',
        'The run has already ended.',
    )


def test_a_refusal_carries_the_fields_its_type_publishes(client, session, monkeypatch):
    def refused():
        raise ArgumentRefusedError('not_a_number', argument='illumination_ma', value='bright')

    monkeypatch.setattr(session.scope.illumination, 'leds_off', refused)

    body = _problem(client.post('/api/v1/scope/illumination/leds_off'), 422)

    # A client corrects the request from the fields, not the words.
    assert (body['reason'], body['argument']) == ('not_a_number', 'illumination_ma')


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
        (lambda c: c.post('/api/v1/handles/Protocol/9/num_steps'), 404, 'not_found'),
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
                '/api/v1/scope/illumination/get_led_state',
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
