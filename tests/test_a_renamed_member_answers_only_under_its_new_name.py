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


def _cell_count_method(threshold: str, area: str, perimeter: str, intensity: str) -> dict:
    """The default cell-count method as a wire client sends it, its fields under the given names."""
    filters = {
        area: {'min': None, 'max': None},
        perimeter: {'min': None, 'max': None},
        'sphericity': {'min': 0.0, 'max': 1.0},
    }
    for which in ('min', 'mean', 'max'):
        filters[f'intensity_{which}{intensity}'] = {'min': 0, 'max': 100}
    return {'pixels_per_um': None, 'fluorescent_mode': True, threshold: 20, 'filters': filters}


# (method, old route, old arguments, new route, new arguments): a GET's
# arguments are its query, any other method's its JSON body. PROTOCOL in a
# route or an argument is the new protocol's id.
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
        {'channel': 'Blue', 'z_um': 5000.0},
        'saved_focus_z_um',
        {'channel': 'Blue', 'z_um': 5000.0},
    ),
    ('GET', 'saved_focus', {'channel': 'Blue'}, 'saved_focus_z_um', {'channel': 'Blue'}),
    (
        'GET',
        'scope/motion/jog_step',
        {'axis': 'Z', 'coarse': True},
        'scope/motion/jog_step_um',
        {'axis': 'Z', 'coarse': True},
    ),
    ('GET', 'scope/camera/read_frame_rate', {}, 'scope/camera/read_frame_rate_fps', {}),
    ('GET', 'scope/camera/read_exposure_ms_min', {}, 'scope/camera/read_exposure_min_ms', {}),
    (
        'PUT',
        'scope/camera/black_level',
        {'value': 4.0},
        'scope/camera/black_level',
        {'level': 4.0},
    ),
    (
        'GET',
        'scope/protocols/focus_z',
        {'then': 'save the focus'},
        'scope/protocols/focus_z_um',
        {'then': 'save the focus'},
    ),
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
    (
        'PUT',
        'labware',
        {'labware_id': '384 well microplate'},
        'labware_id',
        {'labware_id': '384 well microplate'},
    ),
    (
        'PUT',
        'labware_id',
        {'labware_name': '384 well microplate'},
        'labware_id',
        {'labware_id': '384 well microplate'},
    ),
    ('GET', 'labware', {}, 'labware_id', {}),
    (
        'POST',
        'set_protocol_labware',
        {'protocol': PROTOCOL, 'plate_key': '6 well microplate'},
        'set_protocol_labware',
        {'protocol': PROTOCOL, 'labware_id': '6 well microplate'},
    ),
    (
        'GET',
        f'handles/Protocol/{PROTOCOL}/labware',
        {},
        f'handles/Protocol/{PROTOCOL}/labware_id',
        {},
    ),
    (
        'PUT',
        'turret_objective',
        {'position': 2, 'objective_id': '10x Oly'},
        'turret_objective',
        {'slot': 2, 'objective_id': '10x Oly'},
    ),
    ('GET', 'turret_objective', {'position': 2}, 'turret_objective', {'slot': 2}),
    ('POST', 'clear_turret_objective', {'position': 4}, 'clear_turret_objective', {'slot': 4}),
    ('POST', 'scope/motion/move_turret', {'position': 1}, 'scope/motion/move_turret', {'slot': 1}),
    (
        'POST',
        'confirm_objective',
        {'objective_id': '10x Oly', 'turret_position': 1},
        'confirm_objective',
        {'objective_id': '10x Oly', 'slot': 1},
    ),
    (
        'GET',
        'scope/motion/is_current_turret_position_objective_set',
        {},
        'scope/motion/is_current_slot_objective_set',
        {},
    ),
    ('PUT', 'acceleration_limit_pct', {'val_pct': 40}, 'acceleration_limit_pct', {'limit_pct': 40}),
    (
        'POST',
        'post_processing/save_cell_count_method',
        {'method': _cell_count_method('threshold', 'area', 'perimeter', ''), 'path': 'cells.json'},
        'post_processing/save_cell_count_method',
        {
            'method': _cell_count_method('threshold_pct', 'area_um2', 'perimeter_um', '_pct'),
            'path': 'cells.json',
        },
    ),
]


@pytest.fixture
def client(tmp_path):
    # A turret model, with an objective in the slot the startup home leaves
    # in the light path, so the turret's members answer as well.
    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            microscope='LS850T',
            objective_confirmed=True,
            turret_objectives={'1': '10x Oly', '2': '10x Oly', '3': None, '4': None},
        ),
        simulate=True,
    )
    try:
        home_sim_scope(session.scope)
        with TestClient(build_app(session)) as client:
            yield client
    finally:
        session.shutdown()


def _send(client, method: str, route: str, arguments: dict):
    if PROTOCOL in route or PROTOCOL in arguments.values():
        handle = client.post('/api/v1/create_empty_protocol').json()['handle']
        route = route.replace(PROTOCOL, handle)
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
