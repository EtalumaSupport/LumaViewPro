# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A wire client is refused in the words, and with the reason, a Python caller is.

Every answer that is not a result is an RFC 9457 problem
(``application/problem+json``): ``type`` names the reason
(``urn:lumascope:problem:<reason>``, or ``about:blank`` with none), ``title``
and ``detail`` are the outcome's own, ``instance`` names the request, and
``kind``, ``reason`` and ``remedy`` are what a client branches on. A
member's outcome is read as every host reads it (``outcome_of``) and
reported once, to the log only: the problem is the answer. Every refusal is
409 and a fault 500 until each refusal declares its cause. The server's own
answers -- no such route, method, handle or body -- carry their own reasons.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from modules.exceptions import LiveFolderPathRefusedError, Remedy, RunAlreadyEndedError
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
    return TestClient(build_app(session))


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

    body = _problem(client.post('/api/v1/make_logs_zip', json={'output_dir': '../outside'}), 409)

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


def test_a_quiet_outcome_is_409_and_says_it_is_quiet(client, session, monkeypatch):
    def ended():
        raise RunAlreadyEndedError('The run has already ended.')

    monkeypatch.setattr(session.scope.illumination, 'leds_off', ended)

    body = _problem(client.post('/api/v1/scope/illumination/leds_off'), 409)

    assert (body['kind'], body['detail']) == ('quiet', 'The run has already ended.')


def test_a_refusals_remedy_is_sent_as_the_record_apply_remedy_takes(client, session, monkeypatch):
    remedy = Remedy('recover_file_writer', 'Recover', 'Wait')
    refusal = LiveFolderPathRefusedError('outside_live_folder', 'x', 'Not there.')
    refusal.remedy = remedy

    def refuses():
        raise refusal

    monkeypatch.setattr(session.scope.illumination, 'leds_off', refuses)

    body = _problem(client.post('/api/v1/scope/illumination/leds_off'), 409)

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


def test_an_invalid_body_names_each_argument_that_does_not_fit(client):
    body = _problem(client.post('/api/v1/scope/illumination/led_on', json={'channel': 1.5}), 422)

    assert {tuple(e['loc']) for e in body['errors']} >= {('body', 'illumination_ma')}
    assert all(set(e) == {'loc', 'msg', 'type'} for e in body['errors'])
