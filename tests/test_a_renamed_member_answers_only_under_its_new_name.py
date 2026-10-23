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

# Stands for a new empty protocol's handle, made for each request that names it.
PROTOCOL = '<protocol>'

# (method, old route, old arguments, new route, new arguments): a GET's
# arguments are its query, any other method's its JSON body.
RENAMED = [
    (
        'PUT',
        'channel_acquire',
        {'layer': 'BF', 'mode': 'image'},
        'channel_acquire',
        {'channel': 'BF', 'mode': 'image'},
    ),
    ('GET', 'channel_acquire', {'layer': 'BF'}, 'channel_acquire', {'channel': 'BF'}),
    (
        'PUT',
        'layer_acquire',
        {'channel': 'BF', 'mode': 'image'},
        'channel_acquire',
        {'channel': 'BF', 'mode': 'image'},
    ),
    ('GET', 'layer_acquire', {'channel': 'BF'}, 'channel_acquire', {'channel': 'BF'}),
    (
        'PUT',
        'channel_auto_gain',
        {'layer': 'Blue', 'enabled': True},
        'channel_auto_gain',
        {'channel': 'Blue', 'enabled': True},
    ),
    ('GET', 'channel_auto_gain', {'layer': 'Blue'}, 'channel_auto_gain', {'channel': 'Blue'}),
    (
        'GET',
        'channel_auto_gain_lock',
        {'layer': 'Blue'},
        'channel_auto_gain_lock',
        {'channel': 'Blue'},
    ),
    (
        'PUT',
        'layer_auto_gain',
        {'channel': 'Blue', 'enabled': True},
        'channel_auto_gain',
        {'channel': 'Blue', 'enabled': True},
    ),
    ('GET', 'layer_auto_gain', {'channel': 'Blue'}, 'channel_auto_gain', {'channel': 'Blue'}),
    (
        'GET',
        'layer_auto_gain_lock',
        {'channel': 'Blue'},
        'channel_auto_gain_lock',
        {'channel': 'Blue'},
    ),
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
    (
        'POST',
        'add_step',
        {'protocol': PROTOCOL, 'layers': ['BF']},
        'add_step',
        {'protocol': PROTOCOL, 'channels': ['BF']},
    ),
    (
        'POST',
        'save_focus',
        {'protocol': PROTOCOL, 'layer': 'BF'},
        'save_focus',
        {'protocol': PROTOCOL, 'channel': 'BF'},
    ),
    (
        'POST',
        'apply_focus_to_layer_steps',
        {'protocol': PROTOCOL, 'channel': 'BF'},
        'apply_focus_to_channel_steps',
        {'protocol': PROTOCOL, 'channel': 'BF'},
    ),
    (
        'POST',
        'apply_layer_settings',
        {'protocol': PROTOCOL},
        'apply_channel_settings',
        {'protocol': PROTOCOL},
    ),
    ('GET', 'scope/layer_identity', {}, 'scope/channel_identity', {}),
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
    if PROTOCOL in arguments.values():
        handle = client.post('/api/v1/create_empty_protocol').json()['handle']
        arguments = {k: handle if v == PROTOCOL else v for k, v in arguments.items()}
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
