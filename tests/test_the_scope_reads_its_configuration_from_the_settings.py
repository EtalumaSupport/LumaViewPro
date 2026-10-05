# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The settings are the one store for the configuration the scope acts on.

The labware, the stage offset, the turret map, the objective selected on a
scope with no turret and whether the scale bar is drawn are read by the
scope from its session's settings whenever it acts on them. The scope holds
no copy and has no setter, so nothing below the Session can make the scope
act on one value while the settings say another; a reader gets a copy, and
a plate transform keeps the frame it was bound in.
"""

import pytest

from modules.exceptions import ConfigError
from modules.lumascope_api import Lumascope
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield session
    finally:
        session.shutdown()


class TestAWriteIsWhatTheScopeActsOn:
    def test_the_labware(self, session):
        other = next(
            name
            for name in session.wellplate_loader.get_plate_list()
            if name != session.settings['protocol']['labware']
        )
        session.select_labware(other)
        assert session.scope.runtime_state.get_labware().config == (
            session.wellplate_loader.get_plate(plate_key=other).config
        )

    def test_the_stage_offset(self, session):
        session.update_settings('stage_offset.x', 11000.0)
        assert session.scope.runtime_state.get_stage_offset()['x'] == 11000.0

    def test_the_turret_map(self, session):
        objective = session.objective_helper.get_objectives_list()[0]
        session.assign_turret_objective(2, objective)
        assert session.scope.runtime_state.get_turret_config()[2] == objective
        session.clear_turret_objective(2)
        assert session.scope.runtime_state.get_turret_config()[2] is None

    def test_the_objective_with_no_turret(self, session):
        assert not session.scope.runtime_state.is_turreted()
        objective = next(
            o
            for o in session.objective_helper.get_objectives_list()
            if o != session.settings['objective_id']
        )
        session.select_objective(objective)
        assert session.scope.runtime_state.get_current_objective_id() == objective

    def test_the_scale_bar(self, session):
        for enabled in (True, False):
            session.set_scale_bar(enabled)
            assert session.scope.imaging.scale_bar_config['enabled'] is enabled


def test_the_live_views_colour_write_never_changes_whether_the_bar_is_drawn(session):
    # The live view sets the colour on every frame; a toggle between two of
    # those writes used to be undone in the overlay and not in the settings.
    session.set_scale_bar(True)
    session.scope.imaging.set_scale_bar_color('Red')
    session.set_scale_bar(False)
    session.scope.imaging.set_scale_bar_color('BF')
    assert session.scope.imaging.scale_bar_config == {'enabled': False, 'color': 'BF'}
    assert session.settings['scale_bar']['enabled'] is False


def test_changing_what_a_getter_returned_changes_no_setting(session):
    state = session.scope.runtime_state
    state.get_turret_config()[1] = 'NOT_IN_CATALOGUE'
    state.get_stage_offset()['x'] = 0.0
    assert session.settings['turret_objectives'][1] != 'NOT_IN_CATALOGUE'
    assert session.settings['stage_offset']['x'] != 0.0


def test_a_bound_plate_transform_keeps_its_frame(session):
    to_plate = session.scope.runtime_state.plate_transform()
    before = to_plate(60000, 40000)
    session.update_settings('stage_offset.x', session.settings['stage_offset']['x'] + 5500)
    assert to_plate(60000, 40000) == before
    assert session.scope.runtime_state.stage_to_plate(60000, 40000) != before


def test_a_scope_no_session_bound_refuses_each_read_by_name():
    scope = Lumascope(simulate=True, warn_pre_release=False, register_atexit=False)
    try:
        state = scope.runtime_state
        for read in (state.get_labware, state.get_stage_offset, state.get_turret_config):
            with pytest.raises(ConfigError, match='ScopeSession'):
                read()
        with pytest.raises(ConfigError, match='ScopeSession'):
            scope.imaging.scale_bar_config  # noqa: B018 -- the read is the subject
    finally:
        scope.disconnect()


def test_a_second_session_leaves_the_scope_reading_the_first(session):
    # The lanes refuse a second session over one scope before it binds, so
    # the scope never acts on settings nobody can see.
    other = complete_settings(live_folder=session.settings['live_folder'])
    other['stage_offset'] = {'x': 1.0, 'y': 2.0}
    with pytest.raises(RuntimeError, match='already asks an activity claim'):
        ScopeSession.create(other, scope=session.scope)
    assert session.scope.runtime_state.get_stage_offset() == session.settings['stage_offset']


class TestBringUpRefusesWhatItCannotActOn:
    def test_an_unknown_stored_objective(self, tmp_path):
        settings = complete_settings(live_folder=str(tmp_path))
        settings['objective_id'] = 'NOT_AN_OBJECTIVE'
        with pytest.raises(ConfigError, match='NOT_AN_OBJECTIVE'):
            ScopeSession.create(settings, simulate=True)

    @pytest.mark.parametrize('key', ['scale_bar', 'turret_objectives'])
    def test_a_missing_key(self, tmp_path, key):
        settings = complete_settings(live_folder=str(tmp_path))
        del settings[key]
        with pytest.raises(ConfigError, match=key):
            ScopeSession.create(settings, simulate=True)
