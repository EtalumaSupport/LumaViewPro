# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A factory-built session is a configured session.

`ScopeSession.create`, for a scope it builds, runs
the settings-to-scope bring-up before it returns: the turret slot keys
normalized, the slot-1 objective adopted, the labware selected, the
scope initialized, the camera start gate released. A session that came
back from a factory can therefore save an image, refuses settings that
cannot configure a scope by name, and never leaves threads behind when
it refuses.
"""

import threading
import time

import numpy as np
import pytest

from modules.exceptions import ConfigError
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings, complete_settings_without


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
    def test_the_helpers_are_real_and_the_engine_holds_them(self, session):
        assert session.objective_helper is not None
        assert session.wellplate_loader is not None
        assert session.coordinate_transformer is not None
        assert session.sequenced_capture_runner._wellplate_loader is session.wellplate_loader

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
        )
        assert path is not None


class TestSettingsThatCannotConfigureAScope:
    def test_missing_frame_refuses_by_key(self):
        with pytest.raises(ConfigError, match='frame'):
            ScopeSession.create(complete_settings_without('frame'), simulate=True)

    def test_missing_objective_refuses_by_key(self):
        with pytest.raises(ConfigError, match='objective_id'):
            ScopeSession.create(complete_settings_without('objective_id'), simulate=True)

    def test_a_turret_scope_needs_no_stored_objective(self, tmp_path):
        """On a turret scope the objective is the slot's assignment; a stored
        id names nothing, so its absence is not a reason to refuse."""
        session = ScopeSession.create(
            complete_settings_without(
                'objective_id', live_folder=str(tmp_path), microscope='LS850T'
            ),
            simulate=True,
        )
        session.shutdown()

    def test_an_unshipped_objective_refuses_by_value(self):
        with pytest.raises(ConfigError, match='banana'):
            ScopeSession.create(complete_settings(objective_id='banana'), simulate=True)

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
        with pytest.raises(ConfigError, match=r'settings\.json'):
            ScopeSession.create(
                ScopeSession.load_user_settings(str(tmp_path)),
                source_path=str(tmp_path),
                simulate=True,
            )

    def test_a_root_without_labware_refuses_by_file(self, tmp_path):
        import pathlib
        import shutil

        repo = pathlib.Path(__file__).resolve().parent.parent
        data = tmp_path / 'data'
        data.mkdir()
        shutil.copy(repo / 'data' / 'settings.json', data / 'settings.json')
        shutil.copy(repo / 'data' / 'objectives.json', data / 'objectives.json')
        with pytest.raises(ConfigError, match=r'labware\.json'):
            ScopeSession.create(
                ScopeSession.load_user_settings(str(tmp_path)),
                source_path=str(tmp_path),
                simulate=True,
            )


class TestARefusingFactoryLeavesNothingBehind:
    def test_thread_count_returns_to_baseline(self):
        baseline = threading.active_count()
        with pytest.raises(ConfigError):
            ScopeSession.create(complete_settings_without('frame'), simulate=True)
        # Other tests' threads may finish during the wait; what matters is
        # that the refusing factory left none of its own behind.
        assert _wait_for_thread_count(baseline) <= baseline

    def test_a_callers_lanes_survive_the_refusal(self):
        from modules.sequential_io_executor import SequentialIOExecutor

        io = SequentialIOExecutor(name='IO_CALLER')
        cam = SequentialIOExecutor(name='CAMERA_CALLER')
        io.start()
        cam.start()
        try:
            with pytest.raises(ConfigError):
                ScopeSession.create(
                    settings=complete_settings_without('frame'),
                    io_executor=io,
                    camera_executor=cam,
                )
            assert io.accepts_work() and cam.accepts_work()
        finally:
            io.shutdown()
            cam.shutdown()


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
            home_fn=lambda axis: attempts.append(('home', axis)) or True,
            turret_fn=lambda position: attempts.append(('turret', position)),
        )

        assert attempts == [], f'homing disabled must issue no motion, got {attempts}'
        assert session.settings['turret_position'] == 3, (
            'the recorded turret position must be left as it was'
        )

    def test_a_fresh_session_with_the_default_motion_returns_quietly(self, session):
        session.settings['turret_position'] = 3

        session.start_application_session(disable_homing=True)

        assert session.settings['turret_position'] == 3

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
    """Bring-up leaves a simulated stage at the plane its own camera calls
    sharp, not at the floor where homing leaves it.

    Homing to the bottom of travel is correct and shared with the real
    instrument. The difference is what happens next: a real operator
    focuses, and in the simulator nobody did, so every session started
    5 mm from the specimen -- far enough that a z-stack asked for slices
    below zero and an autofocus sweep began on nothing.
    """

    def test_bringup_leaves_z_off_the_floor_and_at_the_declared_plane(self, session):
        session.start_application_session()

        z = session.scope.motion.get_current_position('Z')
        assert z > 0.0, 'bring-up left the stage at the bottom of travel, where no sample is'

        # Read from the camera rather than restated here: the point of the
        # change is that one simulated scene holds one idea of where the
        # sample is, so a literal in this test would defeat what it checks.
        declared_plane = session.scope._camera_driver.get_focal_z()
        assert z == pytest.approx(declared_plane, abs=1.0), (
            f'the stage settled at {z} um but the camera calls {declared_plane} um sharp'
        )
