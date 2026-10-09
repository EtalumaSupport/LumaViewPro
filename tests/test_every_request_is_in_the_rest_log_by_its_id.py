# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every request the server answers is in the REST log, by the id its answer carries.

Each is one record on ``lvp_logger``'s ``rest`` child, marked
``api_request`` so the REST log's handler keeps it, naming the request's
id, method, path and status; the answer carries the same id as
``X-Request-ID``, and a problem names it as its ``instance``, so a client's
report of one answer finds its line.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield s
    s.shutdown()


@pytest.fixture
def client(session):
    with TestClient(build_app(session)) as client:
        yield client


def _logged(caplog, request_id):
    return [
        r
        for r in caplog.records
        if r.name == 'lvp_logger.rest' and getattr(r, 'request_id', None) == request_id
    ]


def test_a_result_is_logged_once_by_the_id_its_answer_carries(client, caplog):
    caplog.set_level(logging.INFO, logger='lvp_logger.rest')

    answer = client.get('/api/v1/app_version')

    (record,) = _logged(caplog, answer.headers['X-Request-ID'])
    assert record.api_request is True
    assert 'GET /api/v1/app_version -> 200' in record.getMessage()


def test_a_problem_names_the_same_id_its_log_line_does(client, caplog):
    caplog.set_level(logging.INFO, logger='lvp_logger.rest')

    answer = client.get('/api/v1/no_such_member')

    request_id = answer.headers['X-Request-ID']
    assert answer.json()['instance'] == f'urn:uuid:{request_id}'
    (record,) = _logged(caplog, request_id)
    assert 'GET /api/v1/no_such_member -> 404' in record.getMessage()


def test_a_body_refused_before_any_route_is_logged_too(client, caplog):
    caplog.set_level(logging.INFO, logger='lvp_logger.rest')

    answer = client.post(
        '/api/v1/scope/illumination/leds_off',
        content=b'x',
        headers={'Content-Type': 'text/plain'},
    )

    assert answer.status_code == 415
    (record,) = _logged(caplog, answer.headers['X-Request-ID'])
    assert '-> 415' in record.getMessage()
