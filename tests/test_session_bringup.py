# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A factory-built session is a configured session.

`ScopeSession.create`, for a scope it builds, runs
the settings-to-scope bring-up before it returns: the settings prepared
as the file's are, the turret slot keys normalized, the slot-1 objective
adopted, the labware selected, the scope initialized, the camera start
gate released. A session that came back from a factory can therefore
save an image, refuses a settings block of the wrong kind by name, and
never leaves threads behind when it refuses. What the preparation makes
of a hand-built dict is pinned in
`tests/test_a_settings_dict_handed_to_create_is_prepared_as_the_file_is.py`.
"""

import threading
import time

import numpy as np
import pytest

from modules.exceptions import ArgumentRefusedError
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.frame_records import frame_record, plate


def _wait_for_thread_count(target, deadline_s=2.0):
    end = time.monotonic() + deadline_s
    while threading.active_count() > target and time.monotonic() < end:
        time.sleep(0.02)
    return threading.active_count()


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


class TestAFactorySessionIsConfigured:
    def test_the_scope_carries_the_objective_and_labware(self, session):
        assert (
            session.scope.runtime_state.get_current_objective_id()
            == session.settings['objective_id']
        )
        assert session.scope.runtime_state.get_labware() is not None
        assert session.scope.imaging.is_streaming()

    def test_a_bare_session_captures_and_saves(self, session, tmp_path):
        from modules.image_save import save_image

        frame = session.scope.imaging.capture_and_wait()
        assert isinstance(frame, np.ndarray)
        path = save_image(
            session.scope,
            array=frame,
            save_folder=str(tmp_path),
            file_root='bringup',
            append='',
            tail_id_mode=None,
            channel='BF',
            false_color_on=False,
            save_encoding='8bit',
            significant_bits=8,
            objective_id=session.scope.runtime_state.get_current_objective_id(),
            frame_record=frame_record(),
            labware=plate(),
            well_label=None,
        )
        assert path is not None


class TestSettingsHandedToAFactory:
    def test_string_turret_keys_are_normalized_and_slot_one_answers(self, tmp_path):
        """A caller dict carries JSON string keys; the file pipeline never saw
        it. The slot map must still be keyed by int, so the turret in slot 1
        answers slot 1's assignment."""
        raw = complete_settings(
            live_folder=str(tmp_path),
            microscope='LS850T',
            objective_id='20x Oly',
        )
        raw['turret_objectives'] = {'1': '10x Oly', '2': None, '3': None, '4': None}
        s = ScopeSession.create(raw, simulate=True)
        try:
            assert s.settings['turret_objectives'][1] == '10x Oly'
            home_sim_scope(s.scope)
            s.scope.motion.move_turret(1)
            assert s.scope.runtime_state.get_current_objective_id() == '10x Oly'
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_root_without_a_template_refuses_by_root(self, tmp_path):
        with pytest.raises(ArgumentRefusedError) as refused:
            ScopeSession.create(
                ScopeSession.load_user_settings(str(tmp_path)),
                source_path=str(tmp_path),
                simulate=True,
            )
        assert (refused.value.reason, refused.value.value) == ('not_an_installation', str(tmp_path))


class TestARefusingFactoryLeavesNothingBehind:
    def test_thread_count_returns_to_baseline(self):
        baseline = threading.active_count()
        with pytest.raises(ArgumentRefusedError) as refused:
            ScopeSession.create(complete_settings(frame='1900 by 1900'), simulate=True)
        assert (refused.value.reason, refused.value.argument) == ('wrong_kind', 'settings.frame')
        # Other tests' threads may finish during the wait; what matters is
        # that the refusing factory left none of its own behind.
        assert _wait_for_thread_count(baseline) <= baseline


class TestDisableHomingIsNoStartupMotion:
    """`disable_homing=True` means no startup motion on any axis: the
    turret is left where it is, like the stage, and no turret position
    is recorded. Positioning the turret without a home would drive an
    absolute move against a reference the caller asked us not to
    establish."""

    def test_neither_the_home_nor_the_turret_move_is_issued(self, session):
        attempts = []
        session.settings['turret_position'] = 3

        session.start_application_session(
            disable_homing=True,
            home_fn=lambda axis: attempts.append(('home', axis)),
            turret_fn=lambda position: attempts.append(('turret', position)),
        )

        assert attempts == [], f'homing disabled must issue no motion, got {attempts}'
        assert session.settings['turret_position'] == 3, (
            'the recorded turret position must be left as it was'
        )

    def test_the_stage_is_not_placed_at_the_sample_either(self, session):
        """No startup motion means none, including the simulator's own
        placement.

        The placement cannot run here even in principle: it is an absolute
        move, and an absolute move on an axis that was never homed raises
        rather than guessing where it is. Skipping the home therefore has
        to skip the placement too, and this pins that they stay together.
        """
        session.start_application_session(disable_homing=True)

        assert session.scope.motion.get_current_position('Z') == pytest.approx(0.0, abs=1.0), (
            'homing disabled must leave the stage untouched at its unhomed origin'
        )


class TestTheSimulatorStartsWhereItsSampleIs:
    """Bring-up leaves a simulated stage at the height its own camera says
    the sample sits at, not at the floor where homing leaves it.

    Homing to the bottom of travel is correct and shared with the real
    instrument. The difference is what happens next: a real operator
    focuses, and in the simulator nobody did, so every session started
    5 mm from the specimen -- far enough that a z-stack asked for slices
    below zero and an autofocus sweep began on nothing.
    """

    def test_bringup_leaves_z_off_the_floor_and_near_the_sample(self, session):
        session.start_application_session()

        z = session.scope.motion.get_current_position('Z')
        assert z > 0.0, 'bring-up left the stage at the bottom of travel, where no sample is'

        # Read from the camera rather than restated here: the point is that
        # one simulated scene holds one idea of where the sample is, so a
        # literal in this test would defeat what it checks. The stage lands
        # at the sample's nominal height, as an operator focusing by eye
        # would, and the true focus is near it, for an autofocus to find.
        camera = session.scope._camera_driver
        nominal = camera.get_nominal_focal_z()
        assert z == pytest.approx(nominal, abs=1.0), (
            f'the stage settled at {z} um but the sample nominally sits at {nominal} um'
        )
        assert camera.get_focal_z() == nominal, 'with no spread the sample is in focus there'

    def test_a_focal_spread_puts_the_focus_within_it_of_the_stage(self, tmp_path):
        s = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path)), simulate=True, sim_focal_spread_um=25.0
        )
        try:
            camera = s.scope._camera_driver
            assert abs(camera.get_focal_z() - camera.get_nominal_focal_z()) <= 25.0
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_focal_spread_is_refused_on_a_real_scope(self, tmp_path):
        with pytest.raises(ArgumentRefusedError) as refused:
            ScopeSession.create(
                complete_settings(live_folder=str(tmp_path)), sim_focal_spread_um=25.0
            )
        assert (refused.value.reason, refused.value.argument) == (
            'needs_simulated_scope',
            'sim_focal_spread_um',
        )
