# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A renamed member or argument answers under its new name only; the old one is gone from the wire.

A rename is a hard rename: the old route answers 404 and an old body or
query key 422, so a client still speaking the old word learns it at once
instead of having its argument dropped. One row per renamed route or key.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

# (method, old route, old arguments, new route, new arguments): a GET's
# arguments are its query, any other method's its JSON body.
RENAMED = [
    (
        'PUT',
        'layer_acquire',
        {'layer': 'BF', 'mode': 'image'},
        'layer_acquire',
        {'channel': 'BF', 'mode': 'image'},
    ),
    ('GET', 'layer_acquire', {'layer': 'BF'}, 'layer_acquire', {'channel': 'BF'}),
    (
        'PUT',
        'layer_auto_gain',
        {'layer': 'Blue', 'enabled': True},
        'layer_auto_gain',
        {'channel': 'Blue', 'enabled': True},
    ),
    ('GET', 'layer_auto_gain', {'layer': 'Blue'}, 'layer_auto_gain', {'channel': 'Blue'}),
    ('GET', 'layer_auto_gain_lock', {'layer': 'Blue'}, 'layer_auto_gain_lock', {'channel': 'Blue'}),
    (
        'PUT',
        'saved_focus',
        {'layer': 'Blue', 'z_um': 5000.0},
        'saved_focus',
        {'channel': 'Blue', 'z_um': 5000.0},
    ),
    ('GET', 'saved_focus', {'layer': 'Blue'}, 'saved_focus', {'channel': 'Blue'}),
    ('POST', 'show_channel', {'layer': 'Blue'}, 'show_channel', {'channel': 'Blue'}),
    (
        'POST',
        'scope/camera/apply',
        {'gain_db': 1.0, 'exposure_ms': 10.0, 'layer': 'BF'},
        'scope/camera/apply',
        {'gain_db': 1.0, 'exposure_ms': 10.0, 'channel': 'BF'},
    ),
]


@pytest.fixture
def client(tmp_path):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850'), simulate=True
    )
    try:
        home_sim_scope(session.scope)
        with TestClient(build_app(session)) as client:
            yield client
    finally:
        session.shutdown()


def _send(client, method: str, route: str, arguments: dict):
    if method == 'GET':
        return client.get(f'/api/v1/{route}', params=arguments)
    return client.request(method, f'/api/v1/{route}', json=arguments)


def test_each_renamed_name_answers_and_its_old_name_is_refused(client):
    wrong = {}
    for method, old_route, old, new_route, new in RENAMED:
        answer = _send(client, method, new_route, new)
        if answer.status_code != 200:
            wrong[f'{method} {new_route} {sorted(new)}'] = (answer.status_code, answer.json())
        answer = _send(client, method, old_route, old)
        if answer.status_code not in (404, 422):
            wrong[f'{method} {old_route} {sorted(old)} (old)'] = (answer.status_code, answer.json())
    assert not wrong, wrong
