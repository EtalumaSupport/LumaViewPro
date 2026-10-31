# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each setting the Session owns has a read of its own, answering what its setter wrote.

A setting changed by its own member (``SETTINGS_WITH_A_MEMBER``) was read
only by its dotted path through ``get_setting``, or inside a dict on
another object, so a caller wrote it by one name and read it by another.
Each now has a named read on the Session, beside its setter.
"""

import logging

import pytest
from fastapi.testclient import TestClient

from modules import settings_paths
from modules.api_surface import sets_of
from modules.exceptions import (
    ArgumentRefusedError,
    HardwareCommandRefusedError,
    PositionOutOfRangeError,
)
from modules.lumascope_api.camera import CameraAPI
from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0, LED1


def _shutdown(session):
    try:
        session.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


@pytest.fixture
def session(tmp_path):
    built = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850'), simulate=True
    )
    home_sim_scope(built.scope)
    yield built
    _shutdown(built)


def test_each_read_answers_what_its_setter_wrote(session):
    # Every value differs from the shipped one, so a read of the template,
    # or of nothing, answers wrong.
    session.scope.motion.move_xy(x_um=30000)
    session.scope.motion.move_z(6000)
    writes = {
        'model': (lambda: session.select_model('LS560'), lambda: session.model, 'LS560'),
        'objective_id': (
            lambda: session.select_objective('10x Oly'),
            lambda: session.objective_id,
            '10x Oly',
        ),
        'labware_id': (
            lambda: session.select_labware('384 well microplate'),
            lambda: session.labware_id,
            '384 well microplate',
        ),
        'scale_bar_enabled': (
            lambda: session.set_scale_bar(True),
            lambda: session.scale_bar_enabled,
            True,
        ),
        'acceleration_limit_pct': (
            lambda: session.set_acceleration_limit(40),
            lambda: session.acceleration_limit_pct,
            40,
        ),
        'channel_acquire': (
            lambda: session.set_channel_acquire(LED0, 'video'),
            lambda: session.channel_acquire(LED0),
            'video',
        ),
        'channel_auto_gain': (
            lambda: session.set_channel_auto_gain(LED1, True),
            lambda: session.channel_auto_gain(LED1),
            True,
        ),
    }
    wrong = {}
    for name, (write, read, expected) in writes.items():
        write()
        if read() != expected:
            wrong[name] = (read(), expected)
    saved = session.save_bookmark(('X', 'Y', 'Z'))
    if session.bookmark != saved:
        wrong['bookmark'] = (session.bookmark, saved)
    assert wrong == {}, f'read, expected: {wrong}'


@pytest.mark.parametrize('read', ['channel_acquire', 'channel_auto_gain'])
def test_a_layer_read_refuses_a_name_that_is_no_layer(session, read):
    with pytest.raises(ArgumentRefusedError) as raised:
        getattr(session, read)('Purple')
    assert raised.value.reason == 'channel_unknown'


def test_the_objective_is_none_while_the_turret_slot_is_unknown(tmp_path):
    # Every slot holds the objective, so once the turret's slot is known
    # the answer is it, whichever slot the home leaves in the light path;
    # before the home no slot is, so none is the active objective.
    settings = complete_settings(live_folder=str(tmp_path), microscope='LS850T')
    settings['turret_objectives'] = dict.fromkeys(range(1, 5), '10x Oly')
    turreted = ScopeSession.create(settings, simulate=True)
    try:
        assert turreted.objective_id is None
        home_sim_scope(turreted.scope)
        assert turreted.objective_id == '10x Oly'
    finally:
        _shutdown(turreted)


def test_a_turret_slot_reads_what_was_bound_to_it(tmp_path):
    turreted = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'), simulate=True
    )
    try:
        turreted.assign_turret_objective(2, '10x Oly')
        assert turreted.turret_objective(2) == '10x Oly'
        assert turreted.turret_objective(3) is None
        with pytest.raises(PositionOutOfRangeError) as refused:
            turreted.turret_objective(5)
        assert (refused.value.axis, refused.value.position) == ('T', 5)
    finally:
        _shutdown(turreted)


def test_a_turret_slot_is_refused_on_a_scope_with_no_turret(session):
    with pytest.raises(HardwareCommandRefusedError) as refused:
        session.turret_objective(1)
    assert refused.value.reason == 'axis_absent'


def test_the_turret_map_answers_what_each_slot_reads_in_process_and_on_the_wire(tmp_path):
    turreted = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'), simulate=True
    )
    try:
        turreted.assign_turret_objective(2, '10x Oly')
        expected = {slot: turreted.turret_objective(slot) for slot in range(1, 5)}
        assert turreted.scope.runtime_state.get_turret_config() == expected
        with TestClient(build_app(turreted)) as client:
            answer = client.get('/api/v1/scope/runtime_state/get_turret_config')
        # The slot keys cross as JSON object keys, strings.
        assert (answer.status_code, answer.json()) == (
            200,
            {str(slot): objective for slot, objective in expected.items()},
        )
    finally:
        _shutdown(turreted)


def test_the_turret_map_is_empty_on_a_scope_with_no_turret_and_the_stored_slots_are_kept(
    tmp_path,
):
    # Assignments stored under a turreted model are no turret's on this one,
    # as turret_objective refuses every slot here; the setting is untouched.
    settings = complete_settings(live_folder=str(tmp_path), microscope='LS850')
    settings['turret_objectives'] = dict.fromkeys(range(1, 5), '10x Oly')
    turretless = ScopeSession.create(settings, simulate=True)
    try:
        assert turretless.scope.runtime_state.get_turret_config() == {}
        with TestClient(build_app(turretless)) as client:
            answer = client.get('/api/v1/scope/runtime_state/get_turret_config')
        assert (answer.status_code, answer.json()) == (200, {})
        assert turretless.settings['turret_objectives'] == dict.fromkeys(range(1, 5), '10x Oly')
    finally:
        _shutdown(turretless)


def test_get_setting_refuses_a_setting_with_its_own_read_and_names_it(session):
    # The table's paths, a layer's for '*': each owned setting with a read
    # is refused, naming it; an action's setting and a host's are answered.
    wrong = {}
    for pattern, member in settings_paths.SETTINGS_WITH_A_MEMBER.items():
        path = pattern.replace('*', LED0, 1)
        owner = ScopeSession if member.owner == 'ScopeSession' else CameraAPI
        read = sets_of(getattr(owner, member.name))
        try:
            session.get_setting(path)
        except ArgumentRefusedError as refused:
            answered = (refused.reason, refused.member)
        else:
            answered = 'answered'
        expected = 'answered' if read is None else ('has_member', member.path_of(read))
        if answered != expected:
            wrong[path] = (answered, expected)
    assert wrong == {}
    # Below an owned key as at it: the refusal is the setting's.
    with pytest.raises(ArgumentRefusedError) as refused:
        session.get_setting('binning.size')
    assert (refused.value.reason, refused.value.member) == ('has_member', 'scope.camera.binning')


def test_a_wire_get_setting_of_an_owned_setting_names_its_read(session):
    with TestClient(build_app(session)) as client:
        refused = client.get('/api/v1/get_setting', params={'path': 'scale_bar.enabled'})
        answered = client.get('/api/v1/get_setting', params={'path': 'live_folder'})
    assert refused.status_code == 422
    assert (refused.json()['reason'], refused.json()['member']) == (
        'has_member',
        'scale_bar_enabled',
    )
    assert answered.status_code == 200
