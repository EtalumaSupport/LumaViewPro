# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for simulated LED and motor boards.

Verifies that simulators are drop-in replacements for real hardware:
- Same public API surface
- Correct state tracking
- Thread safety
- Position math matches real boards
"""

import itertools
import pytest
import threading
import time

import numpy as np

# Heavy deps are mocked by tests/conftest.py at module-import time.

from drivers.simulated_ledboard import SimulatedLEDBoard
from drivers.simulated_motorboard import SimulatedMotorBoard
from drivers.simulated_camera import SimulatedCamera
from drivers.simulated_specimen import DRIFT_STEPS, Specimen
from tests.camera_fakes import grab_a_frame_made_after_now

# Imported for its side effect too: drivers/motorboard.py installs the
# AMAX/DMAX probe-warning filter at import time, and
# test_amax_dmax_probe_warning_suppressed measures that filter.
from drivers.motorboard import MotorBoard
from tests.scope_fakes import build_scope
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS


# ---------------------------------------------------------------------------
# LED Simulator Tests
# ---------------------------------------------------------------------------


class TestSimulatedLEDBoard:
    # Driver-side state-query tests (test_initial_state, test_led_on_off,
    # test_led_on_fast, test_leds_off_all, test_leds_off_fast,
    # test_get_led_state, test_get_led_states) retired in Wave 7 Phase
    # 3d.5 -- get_led_ma / is_led_on / get_led_state / get_led_states
    # are no longer on the driver protocol. Equivalent API-side coverage
    # lives in tests/test_lumascope_api.py and
    # tests/test_state_observer.py.

    def test_exchange_command(self):
        board = SimulatedLEDBoard()
        resp = board.exchange_command('STATUS')
        assert resp is not None
        assert 'STATUS' in resp

    def test_disconnect_reconnect(self):
        board = SimulatedLEDBoard()
        board.disconnect()
        assert not board.is_connected()
        # exchange_command should auto-reconnect
        resp = board.exchange_command('STATUS')
        assert resp is not None
        assert board.is_connected()

    def test_thread_safety(self):
        """Concurrent LED on/off should not corrupt state."""
        board = SimulatedLEDBoard()
        errors = []

        def toggle(channel, iterations):
            try:
                for _ in range(iterations):
                    board.led_on(channel, 100)
                    board.led_off(channel)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=toggle, args=(ch, 50)) for ch in board.available_channels()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f'Thread safety errors: {errors}'
        # All LEDs should be off after all toggles complete, as the
        # simulated board's own current read reports them.
        lit = {ch: board.read_led_current(ch) for ch in board.available_channels()}
        assert set(lit.values()) == {0.0}, f'not reset after concurrent toggle: {lit}'


# ---------------------------------------------------------------------------
# Motor Simulator Tests
# ---------------------------------------------------------------------------


class TestSimulatedMotorBoard:
    def test_initial_state(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        assert board.found is True
        assert board.is_connected()
        assert board.has_homed() is False
        assert board.has_turret() is False  # default model LS850 (no turret)

    def test_no_turret_model(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model='LS850')
        assert board.has_turret() is False

    def test_homing_xyz(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='instant')
        board.home()
        assert board.has_homed() is True
        assert board.current_pos('X') == 0
        assert board.current_pos('Y') == 0
        assert board.current_pos('Z') == 0

    def test_zhome(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        board.move_abs_pos('Z', 5000)
        assert board.current_pos('Z') > 0
        board.zhome()
        assert board.current_pos('Z') == 0

    def test_thome(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        board.thome()
        assert board.has_thomed() is True

    def test_move_absolute_z(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        board.move_abs_pos('Z', 7000)
        pos = board.current_pos('Z')
        assert abs(pos - 7000) < 1  # within rounding

    def test_move_absolute_xy(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        board.move_abs_pos('X', 60000)
        board.move_abs_pos('Y', 40000)
        assert abs(board.current_pos('X') - 60000) < 1
        assert abs(board.current_pos('Y') - 40000) < 1

    def test_a_target_past_travel_is_driven_not_clamped(self):
        """Travel is the motion API's refusal, as on the real board; a
        driver that clamped made a refused move look like one that
        succeeded and stopped short."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        board.move_abs_pos('Z', 99999)
        assert abs(board.current_pos('Z') - 99999) < 1

    def test_target_status(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        board.move_abs_pos('X', 50000)
        # Fast mode: position updates instantly but target_status is False
        # for ~3ms (simulates motion monitor detection window)
        import time

        time.sleep(0.005)
        assert board.target_status('X') is True

    def test_conversion_z(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        um = 5000
        ustep = board.z_um2ustep(um)
        um_back = board.z_ustep2um(ustep)
        assert abs(um - um_back) < 0.01

    def test_conversion_xy(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        um = 60000
        ustep = board.xy_um2ustep(um)
        um_back = board.xy_ustep2um(ustep)
        assert abs(um - um_back) < 0.1

    def test_conversion_turret(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        pos = 3
        ustep = board.t_pos2ustep(pos)
        pos_back = board.t_ustep2pos(ustep)
        assert pos == pos_back

    def test_fullinfo(self):
        board = SimulatedMotorBoard(
            motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model='LS850T', serial_number='TEST-123'
        )
        info = board.fullinfo()
        assert info['model'] == 'LS850T'
        assert info['serial_number'] == 'TEST-123'

    def test_exchange_command_info(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        resp = board.exchange_command('INFO')
        assert 'SIMULATED' in resp

    def test_disconnect_reconnect(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        board.disconnect()
        assert not board.is_connected()
        resp = board.exchange_command('INFO')
        assert resp is not None
        assert board.is_connected()

    def test_acceleration_stubs(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        assert board.acceleration_limit('X', 'acceleration') == 30000
        limits = board.acceleration_limits()
        assert 'X' in limits
        assert 'Y' in limits

    def test_amax_dmax_probe_warning_suppressed(self, caplog):
        """Legacy firmware probe failures for AMAXX/DMAXX/AMAXY/DMAXY
        are filtered out of LVP.serial WARNING records.
        drivers/motorboard.py installs the filter at import time."""
        import logging

        serial_log = logging.getLogger('LVP.serial')
        with caplog.at_level(logging.WARNING, logger='LVP.serial'):
            for cmd in ('AMAXX', 'DMAXX', 'AMAXY', 'DMAXY'):
                serial_log.warning(
                    f"[XYZ Class ] FIRMWARE ERROR: {cmd} -> ERROR: command '{cmd}' not found:"
                )
        suppressed = [r for r in caplog.records if 'FIRMWARE ERROR' in r.getMessage()]
        assert suppressed == [], (
            f'AMAX/DMAX probe warnings must be filtered; leaked {len(suppressed)} records'
        )

    def test_other_firmware_errors_still_propagate(self, caplog):
        """Filter must drop ONLY the AMAX/DMAX probe records. Any other
        FIRMWARE ERROR (real protocol-level failure) propagates."""
        import logging

        serial_log = logging.getLogger('LVP.serial')
        with caplog.at_level(logging.WARNING, logger='LVP.serial'):
            serial_log.warning(
                '[XYZ Class ] FIRMWARE ERROR: MOVE -> ERROR: motor stalled at limit switch'
            )
            serial_log.warning(
                '[LED Class ] FIRMWARE ERROR: ILLUMS -> ERROR: channel not available'
            )
        passed = [r for r in caplog.records if 'FIRMWARE ERROR' in r.getMessage()]
        assert len(passed) == 2, (
            f'Real FIRMWARE ERROR records must propagate; got {len(passed)} (expected 2)'
        )

    def test_axes_config(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        config = board.get_axes_config()
        assert 'X' in config
        assert 'Y' in config
        assert 'Z' in config
        assert 'T' in config

    def test_axis_limits(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        z_limits = board.get_axis_limits('Z')
        assert z_limits['min'] == 0
        assert z_limits['max'] == 14000

    def test_thread_safety(self):
        """Concurrent moves should not corrupt state."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        errors = []

        def move_axis(axis, positions):
            try:
                for pos in positions:
                    board.move_abs_pos(axis, pos)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=move_axis, args=('X', [10000, 20000, 30000])),
            threading.Thread(target=move_axis, args=('Y', [10000, 20000, 30000])),
            threading.Thread(target=move_axis, args=('Z', [1000, 2000, 3000])),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    # --- detect_present_axes tests ---

    def test_detect_present_axes_ls850(self):
        """LS850 should have X, Y, Z (no turret)."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model='LS850')
        axes = board.detect_present_axes()
        assert 'X' in axes
        assert 'Y' in axes
        assert 'Z' in axes
        assert 'T' not in axes

    def test_detect_present_axes_ls820t(self):
        """LS820T should have Z and T (no XY stage)."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model='LS820T')
        axes = board.detect_present_axes()
        assert 'Z' in axes
        assert 'T' in axes
        assert 'X' not in axes
        assert 'Y' not in axes

    def test_detect_present_axes_ls850t(self):
        """LS850T should have X, Y, Z, and T."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model='LS850T')
        axes = board.detect_present_axes()
        assert 'X' in axes
        assert 'Y' in axes
        assert 'Z' in axes
        assert 'T' in axes

    # --- current_pos_steps / target_pos_steps tests ---

    def test_current_pos_steps(self):
        """After a move, current_pos_steps returns raw microstep position."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        target_um = 5000
        board.move_abs_pos('Z', target_um)
        steps = board.current_pos_steps('Z')
        assert isinstance(steps, int)
        expected_steps = board.z_um2ustep(target_um)
        assert steps == expected_steps

    def test_target_pos_steps(self):
        """target_pos_steps returns raw target microstep position."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        target_um = 7000
        board.move_abs_pos('Z', target_um)
        steps = board.target_pos_steps('Z')
        assert isinstance(steps, int)
        expected_steps = board.z_um2ustep(target_um)
        assert steps == expected_steps

    # --- homing returns tests ---

    def test_zhome_returns_once_homed(self):
        """zhome() returns None once homed, and raises when it cannot."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        result = board.zhome()
        assert result is None

    def test_home_returns_once_homed(self):
        """home() returns None once homed, and raises when it cannot."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='instant')
        result = board.home()
        assert result is None

    def test_thome_returns_once_homed(self):
        """thome() returns None once homed, and raises when it cannot."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model='LS850T')
        result = board.thome()
        assert result is None

    def test_thome_no_turret(self):
        """thome() on a non-turret model returns as a homed turret does."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model='LS850')
        result = board.thome()
        assert result is None


class TestATravelHold:
    """A held move halts part of the way, reports itself not arrived, and
    finishes only once released; a STOP or a new target ends the hold."""

    # About half a second of realistic X travel.
    TARGET_UM = 20000

    @staticmethod
    def _held(board, axis='X', at_fraction=0.5):
        hold = board.hold_travel(axis, at_fraction=at_fraction)
        board.move_abs_pos(axis, TestATravelHold.TARGET_UM)
        # A read is what finds the stage at the hold, as the monitor's poll does.
        deadline = time.monotonic() + 10.0
        while not hold.reached.is_set() and time.monotonic() < deadline:
            board.target_status(axis)
            time.sleep(0.005)
        assert hold.reached.is_set(), 'the move never reached its hold'
        return hold

    @staticmethod
    def _arrives(board, axis='X'):
        deadline = time.monotonic() + 10.0
        while not board.target_status(axis):
            assert time.monotonic() < deadline, 'the released move never arrived'
            time.sleep(0.005)

    @staticmethod
    def _board():
        return SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='realistic')

    def test_a_held_move_stands_part_of_the_way_and_has_not_arrived(self):
        board = self._board()
        self._held(board)
        at = board.current_pos('X')
        assert at == pytest.approx(self.TARGET_UM / 2, rel=0.01)
        # Past the time the whole move takes, it still stands there.
        time.sleep(0.6)
        assert board.current_pos('X') == at
        assert board.target_status('X') is False

    def test_a_released_move_travels_on_and_arrives(self):
        board = self._board()
        hold = self._held(board)
        hold.release()
        self._arrives(board)
        assert board.current_pos('X') == pytest.approx(self.TARGET_UM, abs=1)

    def test_a_stop_ends_the_hold_where_the_stage_stood(self):
        board = self._board()
        self._held(board)
        board.exchange_command('STOP')
        assert board.target_status('X') is True
        assert board.current_pos('X') == pytest.approx(self.TARGET_UM / 2, rel=0.01)
        # The hold was that move's: the next one is not held.
        board.move_abs_pos('X', 0)
        self._arrives(board)

    def test_a_new_target_ends_the_hold(self):
        board = self._board()
        self._held(board)
        board.move_abs_pos('X', 5000)
        self._arrives(board)
        assert board.current_pos('X') == pytest.approx(5000, abs=1)

    def test_a_home_ends_the_hold(self):
        board = self._board()
        self._held(board, axis='Z')
        board.zhome()
        assert board.current_pos('Z') == 0
        # The hold went with the move it held: the axis takes a new one.
        board.hold_travel('Z').release()

    @pytest.mark.parametrize('timing', ['instant', 'fast'])
    def test_a_move_with_no_travel_time_is_held_and_arrives_once_released(self, timing):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing=timing)
        hold = self._held(board)
        assert board.current_pos('X') == pytest.approx(self.TARGET_UM / 2, rel=0.01)
        assert board.target_status('X') is False
        hold.release()
        self._arrives(board)
        assert board.current_pos('X') == pytest.approx(self.TARGET_UM, abs=1)

    @pytest.mark.parametrize('at_fraction', [0.0, 1.0, 1.5])
    def test_a_hold_is_part_of_the_way_along(self, at_fraction):
        with pytest.raises(ValueError, match='part of the way'):
            self._board().hold_travel('X', at_fraction=at_fraction)

    def test_an_axis_takes_one_hold(self):
        board = self._board()
        board.hold_travel('X')
        with pytest.raises(ValueError, match='already has a hold'):
            board.hold_travel('X')


# ---------------------------------------------------------------------------
# Multi-Model Tests -- verify all microscope models work correctly
# ---------------------------------------------------------------------------

# All shipping models: Lumi (Z only), LS820 (Z only), LS850 (XYZ), LS850T (XYZ+turret)
ALL_MODELS = ['Lumi', 'LS820', 'LS850', 'LS850T']
TURRET_MODELS = ['LS850T']
NON_TURRET_MODELS = ['Lumi', 'LS820', 'LS850']


class TestAllModels:
    """Verify each microscope model initializes correctly."""

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_model_creates_without_error(self, model):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model=model)
        assert board.found is True
        assert board.is_connected()

    @pytest.mark.parametrize('model', TURRET_MODELS)
    def test_turret_model_detected(self, model):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model=model)
        assert board.has_turret() is True

    @pytest.mark.parametrize('model', NON_TURRET_MODELS)
    def test_non_turret_model_detected(self, model):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model=model)
        assert board.has_turret() is False

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_fullinfo_reports_model(self, model):
        board = SimulatedMotorBoard(
            motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model=model, serial_number='SN-TEST'
        )
        info = board.fullinfo()
        assert info['model'] == model
        assert info['serial_number'] == 'SN-TEST'

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_z_axis_works(self, model):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model=model)
        board.move_abs_pos('Z', 5000)
        assert abs(board.current_pos('Z') - 5000) < 1

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_homing_works(self, model):
        board = SimulatedMotorBoard(
            motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model=model, timing='instant'
        )
        board.home()
        assert board.has_homed() is True

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_motorconfig_travel_limits(self, model):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model=model)
        mc = board.motorconfig
        # All models should have valid travel limits
        assert mc.travel_limit_mm('X') > 0
        assert mc.travel_limit_mm('Y') > 0
        assert mc.travel_limit_mm('Z') > 0

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_axes_config_populated(self, model):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model=model)
        config = board.get_axes_config()
        assert 'X' in config
        assert 'Y' in config
        assert 'Z' in config
        for axis in ('X', 'Y', 'Z'):
            assert config[axis]['limits']['max'] > 0

    @pytest.mark.parametrize('model', TURRET_MODELS)
    def test_turret_positions(self, model):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model=model)
        mc = board.motorconfig
        for pos in range(1, 5):
            usteps = mc.turret_position_usteps(pos)
            assert isinstance(usteps, int)

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_center_command(self, model):
        """CENTER should move to stage center for all models."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, model=model)
        resp = board.exchange_command('CENTER')
        assert resp is not None
        x = board.current_pos('X')
        y = board.current_pos('Y')
        mc = board.motorconfig
        expected_x = mc.travel_limit_um('X') / 2
        expected_y = mc.travel_limit_um('Y') / 2
        assert abs(x - expected_x) < 1
        assert abs(y - expected_y) < 1


class TestMotorConfigDefaults:
    """Verify motorconfig defaults are sensible."""

    def test_defaults_file_exists(self):
        import pathlib

        f = pathlib.Path('data/motorconfig_defaults.json')
        assert f.is_file()

    def test_defaults_load(self):
        from drivers.motorconfig import MotorConfig

        mc = MotorConfig(SHIPPED_MOTOR_DEFAULTS)
        assert mc.model() in ('LS850', 'LS850T')
        assert mc.travel_limit_mm('X') == 120
        assert mc.travel_limit_mm('Y') == 80
        assert mc.travel_limit_mm('Z') == 14
        assert mc.usteps_per_mm('Z') == 170666
        assert mc.lens_focal_length() == 47.8
        assert mc.pixel_size() == 2.0

    def test_update_from_board_overrides(self):
        from drivers.motorconfig import MotorConfig

        mc = MotorConfig(SHIPPED_MOTOR_DEFAULTS)
        mc.update_from_board({'Axis Travel Limit': {'Z': 20}})
        assert mc.travel_limit_mm('Z') == 20
        # X/Y unchanged
        assert mc.travel_limit_mm('X') == 120

    def test_missing_section_returns_default(self):
        from drivers.motorconfig import MotorConfig

        mc = MotorConfig(SHIPPED_MOTOR_DEFAULTS)
        # Non-existent section should return default without crashing
        val = mc._axis_lookup('Nonexistent Section', 'X', default=42)
        assert val == 42

    def test_optics_raises_when_no_optics_declared(self):
        import pytest

        from drivers.exceptions import HardwareError
        from drivers.motorconfig import MotorConfig

        # A config with no Optics section cannot report a scale. The accessors
        # raise rather than substitute a fabricated default; the capability
        # builder catches this and degrades the scale honestly (no scale bar /
        # field of view) instead of writing an invented pixel size.
        mc = MotorConfig.__new__(MotorConfig)
        mc._config = {}
        mc._defaults = {}
        with pytest.raises(HardwareError):
            mc.lens_focal_length()
        with pytest.raises(HardwareError):
            mc.pixel_size()


class TestScaleBarObjectiveInit:
    """The scale bar and the objective a capture uses are the settings' answers."""

    def test_objective_none_at_init(self):
        """A scope not yet brought up has no objective."""

        scope = build_scope(simulate=True)
        assert scope.runtime_state.get_current_objective() is None

    def test_the_selected_objective_is_the_settings(self):
        """With no turret, the objective is the one the settings select."""

        from tests.scope_fakes import bind_settings_like_a_session, record_turret_answer

        scope = record_turret_answer(build_scope(simulate=True, sim_model='LS850'))
        bind_settings_like_a_session(scope, objective_id='20x Oly')
        assert scope.runtime_state.get_current_objective().magnification == 20

    def test_scale_bar_disabled_without_objective(self):
        """Scale bar enabled but no objective -> use_scale_bar forced False."""

        from tests.scope_fakes import bind_settings_like_a_session

        scope = build_scope(simulate=True)
        bind_settings_like_a_session(scope, scale_bar={'enabled': True})
        assert scope.imaging.scale_bar_config['enabled'] is True
        assert scope.runtime_state.get_current_objective() is None

    def test_scale_bar_works_with_objective(self):
        """Scale bar with objective set should proceed."""

        from tests.scope_fakes import bind_settings_like_a_session, record_turret_answer

        scope = record_turret_answer(build_scope(simulate=True, sim_model='LS850'))
        bind_settings_like_a_session(scope, objective_id='20x Oly', scale_bar={'enabled': True})
        assert scope.imaging.scale_bar_config['enabled'] is True
        assert scope.runtime_state.get_current_objective() is not None


# ---------------------------------------------------------------------------
# Camera Simulator Tests
# ---------------------------------------------------------------------------


class TestSimulatedCamera:
    def test_connects_on_init(self):
        cam = SimulatedCamera()
        assert cam.active is True
        assert cam.is_connected()
        assert cam.model_name == 'SimulatedCamera-3840x2160'

    def test_disconnect_reconnect(self):
        cam = SimulatedCamera()
        cam.disconnect()
        assert not cam.is_connected()
        cam.connect()
        assert cam.is_connected()

    def test_default_frame_size(self):
        cam = SimulatedCamera()
        size = cam.get_frame_size()
        assert size['width'] == 3840
        assert size['height'] == 2160

    def test_set_frame_size(self):
        cam = SimulatedCamera()
        cam.set_frame_size(960, 600)
        size = cam.get_frame_size()
        assert size['width'] == 960
        assert size['height'] == 600

    def test_a_frame_size_off_the_grid_is_delivered(self):
        cam = SimulatedCamera()
        cam.set_frame_size(100, 7)  # Not multiples of 48/4
        assert cam.get_frame_size() == {'width': 100, 'height': 7}

    def test_set_frame_size_returns_delivered_geometry(self):
        cam = SimulatedCamera()
        delivered = cam.set_frame_size(640, 482)
        # Off the 48/4 grid, delivered exactly; the return must equal what
        # get_frame_size then reports, so callers can cache it without a read-back.
        assert delivered == {'width': 640, 'height': 482}
        assert delivered == cam.get_frame_size()

    def test_min_max_frame_size(self):
        cam = SimulatedCamera()
        mins = cam.get_min_frame_size()
        maxs = cam.get_max_frame_size()
        assert mins['width'] < maxs['width']
        assert mins['height'] < maxs['height']

    # -- Exposure --

    def test_set_get_exposure(self):
        cam = SimulatedCamera()
        cam.exposure_t(50.0)  # 50 ms
        assert cam.get_exposure_t() == 50.0

    def test_exposure_rejects_over_max(self):
        cam = SimulatedCamera()
        original = cam.get_exposure_t()
        cam.exposure_t(999_999)  # way over max
        assert cam.get_exposure_t() == original  # unchanged

    # -- Gain --

    def test_set_get_gain(self):
        cam = SimulatedCamera()
        cam.gain(5.0)
        assert cam.get_gain() == 5.0

    def test_auto_gain(self):
        cam = SimulatedCamera()
        result = cam.auto_gain(state=True, target_brightness=0.3, min_gain_db=1.0, max_gain_db=10.0)
        assert result is True
        # Gain should converge to mid-range
        assert 1.0 <= cam.get_gain() <= 10.0

    def test_auto_gain_once(self):
        cam = SimulatedCamera()
        result = cam.auto_gain_once(
            state=True, target_brightness=0.5, min_gain_db=2.0, max_gain_db=8.0
        )
        assert result is True
        assert 2.0 <= cam.get_gain() <= 8.0

    def test_update_auto_gain_target(self):
        cam = SimulatedCamera()
        result = cam.update_auto_gain_target_brightness(0.7)
        assert result is True

    def test_update_auto_gain_min_max(self):
        cam = SimulatedCamera()
        result = cam.update_auto_gain_min_max(min_gain_db=0.5, max_gain_db=15.0)
        assert result is True

    # -- Pixel format --

    def test_default_pixel_format(self):
        cam = SimulatedCamera()
        assert cam.get_pixel_format() == 'Mono8'

    def test_set_pixel_format(self):
        cam = SimulatedCamera()
        assert cam.set_pixel_format('Mono12') is True
        assert cam.get_pixel_format() == 'Mono12'

    def test_reject_unsupported_pixel_format(self):
        cam = SimulatedCamera()
        assert cam.set_pixel_format('RGB24') is False
        assert cam.get_pixel_format() == 'Mono8'  # unchanged

    def test_supported_pixel_formats(self):
        cam = SimulatedCamera()
        formats = cam.get_supported_pixel_formats()
        assert 'Mono8' in formats
        assert 'Mono12' in formats

    # -- Binning --

    def test_set_binning(self):
        cam = SimulatedCamera()
        assert cam.set_binning_size(2) is True
        assert cam.get_binning_size() == 2

    def test_reject_invalid_binning(self):
        cam = SimulatedCamera()
        assert cam.set_binning_size(8) is False
        assert cam.get_binning_size() == 1  # unchanged

    # -- Grab / image generation --

    def test_grab_returns_image(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        frame = grab_a_frame_made_after_now(cam)
        assert frame.timestamp is not None
        assert isinstance(frame.image, np.ndarray)
        assert frame.image.shape == (2160, 3840)
        assert frame.image.dtype == np.uint8

    def test_grab_new_capture(self):
        cam = SimulatedCamera()
        # Driver contract is float seconds (verified across all five camera
        # drivers). `timeout=1000` here would have been 1000 seconds and
        # passed only because the simulator doesn't honor the timeout.
        cam.open_and_start()
        frame = cam.grab_new_capture(timeout_s=5.0)
        assert frame is not None
        assert frame.timestamp is not None
        assert isinstance(frame.image, np.ndarray)

    def test_grab_respects_binning(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_binning_size(2)
        frame = grab_a_frame_made_after_now(cam)
        assert frame.image.shape == (1080, 1920)

    def test_a_format_change_stores_no_frame_made_under_the_old_format(self):
        """A pixel-format change stops the stream, applies and restarts it, as
        on a real body, so the next frame stored is in the new format.

        The slow transfer is what makes the race certain: a frame made under
        Mono8 just before the change would otherwise land after the setter
        returned, and the capture that waited for the next frame would get it.
        """
        cam = SimulatedCamera(width=48, height=24)
        cam._grab_delay = 0.2
        cam.open_and_start()
        grab_a_frame_made_after_now(cam)
        assert cam.set_pixel_format('Mono12') is True
        frame = cam.grab_new_capture(timeout_s=5.0)
        assert frame is not None
        assert frame.image.dtype == np.uint16, (
            'a frame made under the old format was stored after the change'
        )

    def test_grab_mono12_dtype(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_pixel_format('Mono12')
        frame = grab_a_frame_made_after_now(cam)
        assert frame.image.dtype == np.uint16

    def test_grab_not_grabbing_returns_false(self):
        cam = SimulatedCamera()
        cam.stop_grabbing()
        assert cam.grab() is None

    def test_image_brightness_varies_with_exposure(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.exposure_t(1.0)  # 1ms -- dim
        frame = grab_a_frame_made_after_now(cam)
        dim = frame.image.mean()

        cam.exposure_t(100.0)  # 100ms -- bright
        frame = grab_a_frame_made_after_now(cam)
        bright = frame.image.mean()

        assert bright > dim

    def test_image_brightness_varies_with_gain(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.gain(0.1)
        frame = grab_a_frame_made_after_now(cam)
        low = frame.image.mean()

        cam.gain(20.0)
        frame = grab_a_frame_made_after_now(cam)
        high = frame.image.mean()

        assert high > low

    def test_twenty_db_of_gain_is_ten_times_the_signal(self):
        # Gain is in dB, so 20 dB is a factor of 10 in amplitude, not 20.
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_pixel_format('Mono12')
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        cam.exposure_t(1.0)
        cam.gain(0.0)
        frame = grab_a_frame_made_after_now(cam)
        unity = frame.image.mean()

        cam.gain(20.0)
        frame = grab_a_frame_made_after_now(cam)
        amplified = frame.image.mean()

        assert amplified / unity == pytest.approx(10.0, rel=0.05)

    # -- Test patterns --

    def test_black_pattern(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='Black')
        frame = grab_a_frame_made_after_now(cam)
        assert frame.image.max() == 0

    def test_white_pattern(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='White')
        frame = grab_a_frame_made_after_now(cam)
        assert frame.image.max() == 255

    def test_noise_pattern(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='Noise')
        frame = grab_a_frame_made_after_now(cam)
        # Noise should have some variance
        assert frame.image.std() > 0

    def test_disable_pattern_returns_the_specimen(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='Black')
        cam.set_test_pattern(enabled=False)
        frame = grab_a_frame_made_after_now(cam)
        # The specimen field has variation; the black pattern it replaced does not
        assert frame.image.std() > 0


class TestNoPatternRequestedRendersTheSpecimen:
    """Turning a test pattern off, or never asking for one, must show the
    specimen field -- not a static ramp.

    The ramp was reachable three ways: the constructor default, the
    disable path, and any unrecognized pattern name.
    """

    def test_disabling_a_pattern_restores_the_moving_specimen(self):
        cam = SimulatedCamera(width=480, height=300)
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='Black')
        cam.set_test_pattern(enabled=False)
        frames = [
            np.asarray(cam._generate_image(cam._start_exposure()), dtype=float) for _ in range(4)
        ]
        diffs = [float(np.mean(np.abs(a - b))) for a, b in itertools.pairwise(frames)]
        assert min(f.std() for f in frames) > 0
        assert min(diffs) > 0, 'consecutive frames are identical -- this is a static image'

    def test_a_fresh_camera_renders_the_moving_specimen(self):
        cam = SimulatedCamera(width=480, height=300)
        frames = [
            np.asarray(cam._generate_image(cam._start_exposure()), dtype=float) for _ in range(4)
        ]
        diffs = [float(np.mean(np.abs(a - b))) for a, b in itertools.pairwise(frames)]
        assert min(diffs) > 0, 'the constructor default is a static image'

    def test_an_unknown_pattern_name_warns_and_still_renders(self):
        # The suite mocks lvp_logger, so the driver's `logger` is a MagicMock
        # and no handler or caplog capture can see the call -- assert on the
        # mock, which is how the rest of the suite checks driver logging.
        import drivers.simulated_camera as sim_cam

        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='NotAPattern')
        sim_cam.logger.warning.reset_mock()
        img = np.asarray(cam._generate_image(cam._start_exposure()), dtype=float)

        assert img.std() > 0, 'a bad name must still produce a usable frame'
        sim_cam.logger.warning.assert_called()
        # set_test_pattern lowercases the name, so the message carries it lowercased
        emitted = ' '.join(str(c) for c in sim_cam.logger.warning.call_args_list).lower()
        assert 'notapattern' in emitted, (
            'an unrecognized pattern name must say so, not fall back silently'
        )

    def test_the_fallback_scales_for_16_bit_pixel_formats(self):
        """Mono12 renders against a 4095 range while the source frames top out
        at 255. Without the uint16 scaling the fallback comes out nearly black,
        which is exactly what a second copy of the render path drops."""
        cam = SimulatedCamera()
        cam.set_test_pattern(enabled=False)
        cam._pixel_format = 'Mono12'
        img = np.asarray(cam._generate_image(cam._start_exposure()), dtype=float)
        assert img.max() > 255, (
            f'12-bit fallback peaks at {img.max()} -- not scaled to the 4095 range'
        )

    # -- Grabbing state --

    def test_start_stop_grabbing(self):
        cam = SimulatedCamera()
        assert cam.is_grabbing() is False  # connect() no longer eager-starts
        cam.open_and_start()
        assert cam.is_grabbing() is True  # gate released
        cam.stop_grabbing()
        assert cam.is_grabbing() is False
        cam.start_grabbing()  # restart primitive (gate already open)
        assert cam.is_grabbing() is True

    # -- update_camera_config context manager --

    def test_update_camera_config_stops_restarts_grabbing(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        assert cam.is_grabbing() is True
        with cam.update_camera_config():
            assert cam.is_grabbing() is False
        assert cam.is_grabbing() is True

    # -- Temperature --

    def test_temperatures(self):
        cam = SimulatedCamera()
        temps = cam.get_all_temperatures()
        assert 'sensor' in temps
        assert temps['sensor'] > 0

    # -- Max exposure --

    def test_max_exposure_set(self):
        cam = SimulatedCamera()
        assert cam.max_exposure == 1_000

    # -- Thread safety --

    def test_thread_safety(self):
        cam = SimulatedCamera()
        errors = []

        def grab_loop(n):
            try:
                for _ in range(n):
                    cam.grab()
            except Exception as e:
                errors.append(e)

        def settings_loop(n):
            try:
                for i in range(n):
                    cam.exposure_t(10.0 + i)
                    cam.gain(1.0 + i * 0.1)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=grab_loop, args=(50,)),
            threading.Thread(target=grab_loop, args=(50,)),
            threading.Thread(target=settings_loop, args=(50,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    # -- Z-dependent focus simulation --

    def test_focus_target_pattern(self):
        cam = SimulatedCamera(width=480, height=300)
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        frame = grab_a_frame_made_after_now(cam)
        # Focus target should have features -- not uniform
        assert frame.image.std() > 5

    def test_set_get_z_position(self):
        cam = SimulatedCamera()
        cam.set_z_position(3000.0)
        assert cam.get_z_position() == 3000.0

    def test_set_get_focal_z(self):
        cam = SimulatedCamera()
        cam.set_focal_z(7000.0)
        assert cam.get_focal_z() == 7000.0

    def test_focus_score_peaks_at_focal_z(self):
        """Vollath F4 focus score should be highest at the focal point."""
        from modules.autofocus_functions import focus_vollath4_original

        cam = SimulatedCamera(width=480, height=300)
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        cam.set_focal_z(5000.0)

        scores = {}
        for z in [3000, 4000, 4500, 4800, 5000, 5200, 5500, 6000, 7000]:
            cam.set_z_position(float(z))
            frame = grab_a_frame_made_after_now(cam)
            scores[z] = focus_vollath4_original(image=frame.image)

        # Best score should be at z=5000 (focal point)
        best_z = max(scores, key=scores.get)
        assert best_z == 5000, f'Expected best focus at 5000, got {best_z}. Scores: {scores}'

    def test_focus_score_decreases_with_defocus(self):
        """Focus score should decrease as we move away from focal point."""
        from modules.autofocus_functions import focus_vollath4_original

        cam = SimulatedCamera(width=480, height=300)
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        cam.set_focal_z(5000.0)

        # Get scores at increasing distances from focus, in depths of field
        # (a camera built on its own renders with a 10 um depth of field)
        cam.set_z_position(5000.0)
        frame = grab_a_frame_made_after_now(cam)
        score_at_focus = focus_vollath4_original(image=frame.image)

        cam.set_z_position(5010.0)
        frame = grab_a_frame_made_after_now(cam)
        score_near = focus_vollath4_original(image=frame.image)

        cam.set_z_position(5040.0)
        frame = grab_a_frame_made_after_now(cam)
        score_far = focus_vollath4_original(image=frame.image)

        assert score_at_focus > score_near > score_far, (
            f'Scores should decrease: focus={score_at_focus}, near={score_near}, far={score_far}'
        )

    def test_focus_curve_is_symmetric(self):
        """Focus scores should be roughly symmetric around focal point."""
        from modules.autofocus_functions import focus_vollath4_original

        cam = SimulatedCamera(width=480, height=300)
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        cam.set_focal_z(5000.0)

        cam.set_z_position(4980.0)
        frame = grab_a_frame_made_after_now(cam)
        score_below = focus_vollath4_original(image=frame.image)

        cam.set_z_position(5020.0)
        frame = grab_a_frame_made_after_now(cam)
        score_above = focus_vollath4_original(image=frame.image)

        # Within 20% of each other (both 20 um, two depths of field, from focus)
        ratio = score_below / score_above if score_above != 0 else float('inf')
        assert 0.8 < ratio < 1.2, f'Asymmetric: below={score_below}, above={score_above}'

    def test_a_z_told_to_the_sensor_is_where_the_next_frames_are_exposed(self):
        """The scope tells the sensor where Z stands; frames exposed after it are made there."""
        cam = SimulatedCamera(width=480, height=300)
        cam.set_z_position(5000.0)
        cam.open_and_start()
        try:
            at_focus = grab_a_frame_made_after_now(cam)
            cam.set_z_position(3000.0)
            moved = grab_a_frame_made_after_now(cam)
        finally:
            cam.stop_grabbing()

        assert cam.exposed_under(at_focus.chunks['FrameID']).z_um == 5000.0
        assert cam.exposed_under(moved.chunks['FrameID']).z_um == 3000.0

    def test_the_specimen_is_sharpest_at_the_focal_plane(self):
        """The specimen a live view shows blurs with defocus, so a sweep's
        best score is at the sample's focal plane, from either side."""
        from modules.autofocus_functions import focus_vollath4_original

        dof = 4.0
        cam = SimulatedCamera(width=480, height=300, depth_of_field_func=lambda: dof)
        cam.open_and_start()
        cam.set_focal_z(5007.0)

        scores = {}
        for z in [4995.0, 4999.0, 5003.0, 5005.0, 5007.0, 5009.0, 5011.0, 5015.0, 5019.0]:
            cam.set_z_position(z)
            frame = grab_a_frame_made_after_now(cam)
            scores[z] = focus_vollath4_original(image=frame.image)

        best_z = max(scores, key=scores.get)
        assert best_z == 5007.0, f'best focus at {best_z}, not the focal plane. Scores: {scores}'

    def test_a_shallower_depth_of_field_falls_off_faster(self):
        """The same distance from focus blurs a high-NA objective more."""
        from modules.autofocus_functions import focus_vollath4_original

        def score_10_um_off(dof):
            cam = SimulatedCamera(width=480, height=300, depth_of_field_func=lambda: dof)
            cam.open_and_start()
            cam.set_focal_z(5000.0)
            cam.set_z_position(5010.0)
            frame = grab_a_frame_made_after_now(cam)
            return focus_vollath4_original(image=frame.image)

        assert score_10_um_off(dof=55.5) > score_10_um_off(dof=1.7)

    def test_no_blur_at_focal_point(self):
        """Image at focal point should be identical to unblurred target."""
        cam = SimulatedCamera(width=480, height=300)
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        cam.set_focal_z(5000.0)

        cam.set_z_position(5000.0)
        frame = grab_a_frame_made_after_now(cam)
        sharp = frame.image.copy()

        # Defocused image should differ
        cam.set_z_position(7000.0)
        frame = grab_a_frame_made_after_now(cam)
        blurred = frame.image

        assert not np.array_equal(sharp, blurred)
        # Blurred image should have lower variance (smoother)
        assert blurred.astype(float).std() < sharp.astype(float).std()

    # -- Timing modes --

    def test_timing_mode_fast(self):
        cam = SimulatedCamera(timing='fast')
        assert cam._grab_delay == 0.0

    def test_timing_mode_realistic(self):
        cam = SimulatedCamera(timing='realistic')
        assert cam._grab_delay > 0

    def test_timing_mode_switch(self):
        cam = SimulatedCamera(timing='fast')
        cam.set_timing_mode('realistic')
        assert cam._grab_delay > 0
        cam.set_timing_mode('fast')
        assert cam._grab_delay == 0.0

    # -- Camera profile --

    def test_profile_loaded_on_connect(self):
        cam = SimulatedCamera()
        assert cam.profile is not None
        assert cam.profile.model_name == 'SimulatedCamera-3840x2160'

    def test_profile_sensor_info(self):
        cam = SimulatedCamera()
        assert cam.profile.sensor == 'Simulated'
        assert cam.profile.pixel_size_um == 2.0
        assert cam.profile.shutter == 'rolling'

    def test_profile_sets_max_exposure(self):
        cam = SimulatedCamera()
        # Camera.max_exposure is a property derived from profile.exposure_max_us.
        # Single source of truth (Rule 2).
        assert cam.max_exposure == cam.profile.exposure_max_us / 1000.0

    def test_profile_binning_sizes(self):
        cam = SimulatedCamera()
        assert cam.profile.binning_sizes == [1, 2, 4]

    def test_profile_pixel_formats(self):
        cam = SimulatedCamera()
        assert 'Mono8' in cam.profile.pixel_formats
        assert 'Mono12' in cam.profile.pixel_formats

    def test_profile_gain_info(self):
        cam = SimulatedCamera()
        assert cam.profile.gain.total_min_db == 0.0
        assert cam.profile.gain.total_max_db == 48.0

    def test_profile_native_resolution(self):
        cam = SimulatedCamera()
        assert cam.profile.native_resolution == {'width': 3840, 'height': 2160}

    def test_profile_capabilities(self):
        cam = SimulatedCamera()
        assert cam.profile.has_auto_gain is True
        assert cam.profile.has_auto_exposure is True
        assert cam.profile.driver == 'simulated'

    # -- update_camera_config exception safety --

    def test_update_camera_config_restarts_after_exception(self):
        """Grabbing must restart even if config change throws."""
        cam = SimulatedCamera()
        cam.open_and_start()
        assert cam.is_grabbing() is True
        with pytest.raises(ValueError), cam.update_camera_config():
            assert cam.is_grabbing() is False
            raise ValueError('simulated config failure')
        # Grabbing must be restored despite the exception
        assert cam.is_grabbing() is True

    # -- update_camera_config re-entrancy (CAM-4) --

    def test_update_camera_config_reentrant_single_stop_start(self):
        """Nested update_camera_config must stop+start exactly once.

        Regression test for CAM-4: previously the abstract context
        manager always read is_grabbing() and trusted the outer call to
        have already stopped before the inner call queried. Depth
        tracking now makes the invariant explicit -- only the outermost
        invocation toggles the grab loop.
        """
        cam = SimulatedCamera()
        cam.open_and_start()
        stop_calls = []
        start_calls = []
        orig_stop = cam.stop_grabbing
        orig_start = cam.start_grabbing

        def counting_stop():
            stop_calls.append(True)
            orig_stop()

        def counting_start():
            start_calls.append(True)
            orig_start()

        cam.stop_grabbing = counting_stop
        cam.start_grabbing = counting_start

        assert cam.is_grabbing() is True
        with cam.update_camera_config():
            assert cam.is_grabbing() is False
            with cam.update_camera_config():
                # Inner level must NOT call stop_grabbing again -- would
                # be a no-op anyway (already stopped) but the invariant
                # is "only outer level toggles".
                assert cam.is_grabbing() is False
                with cam.update_camera_config():
                    assert cam.is_grabbing() is False
        assert cam.is_grabbing() is True

        assert len(stop_calls) == 1, f'expected exactly 1 stop, got {len(stop_calls)}'
        assert len(start_calls) == 1, f'expected exactly 1 start, got {len(start_calls)}'

    def test_update_camera_config_reentrant_inner_exception(self):
        """If an inner level raises, the outer level still restarts."""
        cam = SimulatedCamera()
        cam.open_and_start()
        assert cam.is_grabbing() is True
        with pytest.raises(ValueError), cam.update_camera_config():
            assert cam.is_grabbing() is False
            with cam.update_camera_config():
                raise ValueError('inner failure')
        assert cam.is_grabbing() is True

    def test_update_camera_config_reentrant_when_not_grabbing(self):
        """Nested call when grabbing was already stopped is a no-op."""
        cam = SimulatedCamera()
        cam.stop_grabbing()
        assert cam.is_grabbing() is False

        stop_calls = []
        start_calls = []
        orig_stop = cam.stop_grabbing
        orig_start = cam.start_grabbing

        def counting_stop():
            stop_calls.append(True)
            orig_stop()

        def counting_start():
            start_calls.append(True)
            orig_start()

        cam.stop_grabbing = counting_stop
        cam.start_grabbing = counting_start

        with cam.update_camera_config(), cam.update_camera_config():
            pass

        # Was not grabbing on entry; must not start anything.
        assert len(stop_calls) == 0
        assert len(start_calls) == 0
        assert cam.is_grabbing() is False


class TestTheSpecimenField:
    """The specimen is one sample on the stage, drifting one pixel around a circle.

    A field that changed from frame to frame changed what a focus metric
    scored at one Z, so a sweep found its peak wherever the change put it.
    """

    def test_never_fully_black_or_blown_out(self):
        field = Specimen().view(0.0, 0.0, 600, 800, None)
        assert field.min() > 0, 'a crushed field reads as a dead camera'
        assert field.max() < 255, 'a blown field hides the exposure control'

    def test_the_same_place_is_the_same_sample_every_time(self):
        assert np.array_equal(
            Specimen().view(1500.0, -700.0, 600, 800, 0.75),
            Specimen().view(1500.0, -700.0, 600, 800, 0.75),
        )

    def test_matches_requested_size(self):
        field = Specimen().view(0.0, 0.0, 1200, 1920, None)
        assert field.shape == (1200, 1920)
        assert field.dtype == np.uint8

    def test_the_drift_moves_the_frame_one_pixel_a_step_including_the_wrap(self):
        specimen = Specimen()
        frames = [specimen.view(0.0, 0.0, 600, 800, None, shift=step) for step in DRIFT_STEPS]
        assert len({f.tobytes() for f in frames}) == len(frames), (
            'identical frames make a running stream indistinguishable from a frozen one'
        )
        # One pixel of shift on this field changes about 4 grey levels on
        # average; a larger drift would show as more.
        for a, b in zip(frames, frames[1:] + frames[:1], strict=True):
            change = float(np.abs(a.astype(int) - b.astype(int)).mean())
            assert 0 < change < 6.0, f'a drift step changed {change} grey levels'

    def test_the_drift_does_not_move_a_focus_score(self):
        """A sweep's scores differ by focus, not by which drift step a frame was."""
        from modules.autofocus_functions import focus_vollath4_original

        cam = SimulatedCamera(width=480, height=300, depth_of_field_func=lambda: 4.0)
        cam.set_focal_z(5000.0)
        cam.set_z_position(5000.0)
        cam.open_and_start()
        try:
            # Four frames, each its own drift step.
            at_focus = [
                focus_vollath4_original(image=grab_a_frame_made_after_now(cam).image)
                for _ in range(4)
            ]
            cam.set_z_position(5001.0)
            quarter_dof_off = focus_vollath4_original(image=grab_a_frame_made_after_now(cam).image)
        finally:
            cam.stop_grabbing()
        drift_spread = max(at_focus) - min(at_focus)
        assert drift_spread < (min(at_focus) - quarter_dof_off) / 10


class TestCameraProfiles:
    """Tests for drivers/camera_profiles.py lookup and defaults."""

    def test_lookup_known_pylon_model(self):
        from drivers.camera_profiles import lookup_profile

        p = lookup_profile('daA3840-45um')
        assert p.model_name == 'daA3840-45um'
        assert p.sensor == 'Sony IMX334LLR-C'
        assert p.pixel_size_um == 2.0
        assert p.shutter == 'rolling'
        assert p.driver == 'pylon'
        assert p.exposure_max_us == 1_000_000

    def test_lookup_known_ace2_model(self):
        from drivers.camera_profiles import lookup_profile

        p = lookup_profile('a2A3536-31umBAS')
        assert p.sensor == 'Sony IMX676-AAMR1-C'
        assert p.exposure_max_us == 10_000_000
        assert p.gain.analog_max_db == 30.0

    def test_lookup_known_ids_model(self):
        from drivers.camera_profiles import lookup_profile

        p = lookup_profile('U3-34L0XCP-M')
        assert p.driver == 'ids'
        assert p.sensor == 'Sony IMX676-AAMR1-C'
        assert p.pixel_size_um == 2.0
        assert p.native_resolution == {'width': 3552, 'height': 3552}
        assert p.binning_sizes == [1, 2]
        assert p.gain.gain_selector == 'AnalogAll'
        assert p.has_auto_gain is False
        assert p.has_auto_exposure is False
        assert 'Mono8' not in p.pixel_formats  # No native Mono8
        assert 'Mono10g40IDS' in p.pixel_formats
        assert p.exposure_max_us == 2_000_000

    def test_lookup_simulated(self):
        from drivers.camera_profiles import lookup_profile

        p = lookup_profile('SimulatedCamera-3840x2160')
        assert p.driver == 'simulated'
        assert p.gain.total_min_db == 0.0

    def test_lookup_unknown_returns_default(self):
        from drivers.camera_profiles import lookup_profile

        p = lookup_profile('TotallyUnknownCamera-XYZ')
        assert p.model_name == 'TotallyUnknownCamera-XYZ'
        assert p.driver == 'unknown'
        assert p.exposure_max_us == 1_000_000
        assert p.binning_sizes == [1]

    def test_lookup_none_returns_default(self):
        from drivers.camera_profiles import lookup_profile

        p = lookup_profile(None)
        assert p.driver == 'unknown'

    def test_lookup_substring_match(self):
        from drivers.camera_profiles import lookup_profile

        # Model name might have extra text from SDK
        p = lookup_profile('Basler daA3840-45um (12345678)')
        assert p.sensor == 'Sony IMX334LLR-C'

    def test_lookup_returns_copy(self):
        """Modifying a returned profile should not affect the registry."""
        from drivers.camera_profiles import lookup_profile

        p1 = lookup_profile('daA3840-45um')
        p1.exposure_max_us = 99_000
        p1.gain.total_min_db = -99.0

        p2 = lookup_profile('daA3840-45um')
        assert p2.exposure_max_us == 1_000_000
        assert p2.gain.total_min_db is None  # Not modified

    def test_dynamic_fields_initially_none(self):
        from drivers.camera_profiles import lookup_profile

        p = lookup_profile('daA3840-45um')
        # exposure_max_us has a static default per profile entry; it's
        # the single source of truth, overwritten dynamically only when
        # the SDK / driver narrows it.
        assert p.exposure_min_us is None
        assert p.gain.total_min_db is None  # Pylon profiles don't preset these
        assert p.gain.total_max_db is None

    def test_profile_dataclass_defaults(self):
        from drivers.camera_profiles import CameraProfile

        p = CameraProfile()
        assert p.model_name == ''
        assert p.pixel_formats == []
        assert p.binning_sizes == [1]
        assert p.alignment == {'width': 2, 'height': 2}


class TestTimingModes:
    """Verify timing mode switching across all simulators."""

    def test_motor_instant_mode(self):
        m = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='instant')
        assert m._cmd_delay == 0.0
        assert m._simulate_move_duration is False
        assert m._fast_move_duration == 0.0

    def test_motor_fast_mode(self):
        m = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='fast')
        assert m._cmd_delay > 0  # 1ms minimum -- nothing returns instantly
        assert m._simulate_move_duration is True
        assert m._fast_move_duration > 0  # Brief ~3ms per move

    def test_motor_realistic_mode(self):
        m = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='realistic')
        assert m._cmd_delay > 0
        assert m._simulate_move_duration is True

    def test_motor_switch_mode(self):
        m = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='fast')
        m.set_timing_mode('realistic')
        assert m._simulate_move_duration is True
        m.set_timing_mode('fast')
        assert m._simulate_move_duration is True
        assert m._fast_move_duration > 0
        m.set_timing_mode('instant')
        assert m._simulate_move_duration is False
        assert m._cmd_delay == 0.0

    def test_motor_realistic_move_not_instant(self):
        """In realistic mode, target_status returns False during move."""
        m = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='realistic')
        m._homed['Z'] = True
        # 1000 usteps gives ~0.5 s expected duration with TMC ramp params --
        # still proves "not instant" via the immediate-False check, while
        # leaving ~10x headroom against the 5 s deadline. Pre-shrink the
        # test used 10 000 usteps which produced ~4.7 s expected duration
        # and only ~0.3 s of margin; under heavy concurrent test load
        # (memory pressure, GC pauses) the deadline blew intermittently.
        m.move_abs_pos('Z', 1000.0)
        # Should not have arrived yet
        assert m.target_status('Z') is False
        # Wait for move to complete
        deadline = time.monotonic() + 5.0
        while not m.target_status('Z'):
            time.sleep(0.01)
            if time.monotonic() > deadline:
                raise TimeoutError('Motor never reached target')
        assert m.current_pos('Z') == pytest.approx(1000.0, abs=1.0)

    def test_motor_fast_move_brief_delay(self):
        """In fast mode, position updates instantly but target_status has ~3ms delay."""
        import time

        m = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='fast')
        m.move_abs_pos('Z', 10000.0)
        # Position is instant
        assert m.current_pos('Z') == pytest.approx(10000.0, abs=1.0)
        # target_status needs brief delay
        time.sleep(0.005)
        assert m.target_status('Z') is True

    @staticmethod
    def _waits_of_a_full_home(monkeypatch, timing):
        """Every wait the board takes through a full home on ``timing``.

        A simulated home's duration is the wait the board takes for it, so
        the waits are read from the driver's own ``time`` name rather than
        timed on the wall: a real-time bound would be the clock of a loaded
        machine, and the realistic home is seconds of it.
        """
        import types

        import drivers.simulated_motorboard as simulated_motorboard

        waits = []
        clock = types.SimpleNamespace(sleep=waits.append, monotonic=time.monotonic)
        monkeypatch.setattr(simulated_motorboard, 'time', clock)
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing=timing)
        board.home()
        return board, waits

    def test_a_home_on_the_fast_timing_waits_no_longer_than_a_fast_move(self, monkeypatch):
        """The fast tier homes as it moves, in its own brief time, never the
        bench's homing durations (it slept those from 2026-03-31 to 10-12)."""
        board, waits = self._waits_of_a_full_home(monkeypatch, 'fast')

        assert board.has_homed()
        assert waits and max(waits) <= board._fast_move_duration

    def test_a_home_on_the_realistic_timing_waits_the_benchs_homing_time(self, monkeypatch):
        """The realistic timing keeps the bench's home: an XYZ home waits its
        homing duration."""
        board, waits = self._waits_of_a_full_home(monkeypatch, 'realistic')

        assert board.has_homed()
        assert SimulatedMotorBoard.HOMING_DURATIONS['XYZ'] in waits

    def test_the_fast_timing_fan_spins_no_longer_than_a_fast_move(self):
        """A support report's fan test waits the board's spin times; on the
        fast tier they are its own brief time, not the real fan's 5 s."""
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='fast')

        assert board.fan_spin_up_s <= board._fast_move_duration
        assert board.fan_spin_down_s <= board._fast_move_duration

    def test_the_realistic_timing_fan_spins_as_the_real_boards(self):
        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='realistic')

        assert (board.fan_spin_up_s, board.fan_spin_down_s) == (
            MotorBoard.fan_spin_up_s,
            MotorBoard.fan_spin_down_s,
        )

    def test_led_instant_mode(self):
        led = SimulatedLEDBoard(timing='instant')
        assert led._delay == 0.0

    def test_led_fast_mode(self):
        led = SimulatedLEDBoard(timing='fast')
        assert led._delay > 0  # 1ms minimum -- nothing returns instantly

    def test_led_realistic_mode(self):
        led = SimulatedLEDBoard(timing='realistic')
        assert led._delay > 0

    def test_led_switch_mode(self):
        led = SimulatedLEDBoard(timing='fast')
        led.set_timing_mode('realistic')
        assert led._delay > 0
        led.set_timing_mode('fast')
        assert led._delay > 0  # fast mode now has 1ms delay
        led.set_timing_mode('instant')
        assert led._delay == 0.0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='turbo')
        with pytest.raises(ValueError):
            SimulatedLEDBoard(timing='turbo')
        with pytest.raises(ValueError):
            SimulatedCamera(timing='turbo')


class TestFailureInjection:
    """Verify failure injection for testing error recovery paths."""

    # --- Motor board ---

    def test_motor_fail_on_specific_command(self):
        """Motor board should return None for targeted commands only."""
        m = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, fail_on={'ZHOME'})
        assert m.exchange_command('INFO') is not None  # OK
        assert m.exchange_command('ZHOME') is None  # targeted failure
        assert m.exchange_command('INFO') is not None  # still connected
        assert m.driver is not None  # not disconnected

    def test_motor_fail_on_multiple_commands(self):
        """Multiple commands can be targeted for failure."""
        m = SimulatedMotorBoard(
            motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS,
            fail_on={'ZHOME', 'THOME'},
            timing='instant',
        )
        assert m.exchange_command('ZHOME') is None
        assert m.exchange_command('THOME') is None
        assert m.exchange_command('HOME') is not None  # not in fail set

    def test_motor_no_failure_by_default(self):
        """Without fail params, simulator works normally."""
        m = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        for _ in range(100):
            assert m.exchange_command('INFO') is not None

    # --- LED board ---

    def test_led_fail_on_specific_command(self):
        """LED board should return None for targeted commands only."""
        led = SimulatedLEDBoard(fail_on={'LEDS_ENT'})
        assert led.exchange_command('LEDS_ENT') is None  # targeted
        assert led.exchange_command('LED0_100') is not None  # OK
        assert led.driver is not None  # still connected

    def test_led_no_failure_by_default(self):
        """Without fail params, LED simulator works normally."""
        led = SimulatedLEDBoard()
        for _ in range(100):
            assert led.exchange_command('LED0_100') is not None

    def test_an_unplugged_motor_board_is_found_dead_by_its_next_command(self):
        m = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        m.unplug()
        assert m.is_connected(), 'a pulled cable is found by a write, not before'
        assert m.exchange_command('INFO') is None
        assert (m.is_connected(), m.found) == (False, False)
        m.connect()
        assert not m.is_connected(), 'no port opens while the cable is out'
        m.replug()
        assert m.exchange_command('INFO') is not None
        assert m.is_connected()

    def test_a_counted_drop_takes_that_many_of_its_command(self):
        m = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        m.drop_replies('ACTUAL_RZ', count=2)
        assert m.exchange_command('INFO') is not None
        assert m.exchange_command('ACTUAL_RZ') is None
        assert m.exchange_command('ACTUAL_RZ') is None
        assert m.exchange_command('ACTUAL_RZ') is not None

    def test_a_drop_of_any_command_lasts_until_lifted(self):
        m = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        m.drop_replies()
        assert m.exchange_command('INFO') is None
        assert m.exchange_command('ACTUAL_RX') is None
        m.stop_dropping()
        assert m.exchange_command('INFO') is not None

    def test_a_drop_is_of_one_command_or_more(self):
        m = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        with pytest.raises(ValueError, match='one command or more'):
            m.drop_replies('INFO', count=0)

    def test_led_fail_on_matches_by_first_word_as_the_motor_board_does(self):
        led = SimulatedLEDBoard(fail_on={'LEDS_ENT'})
        assert led.exchange_command('LEDS_ENT 1') is None


class TestTheCamerasFaultDoors:
    """A simulated camera refuses a node write once, leaves the bus, and
    takes back a stall, through the doors ``scope.simulation`` drives."""

    @pytest.mark.parametrize(
        ('answer', 'returned'), [('false', False), ('none', None), ('true', True)]
    )
    def test_a_refused_write_answers_once_and_moves_nothing(self, answer, returned):
        cam = SimulatedCamera()
        cam.gain(2.0)
        cam.refuse_write('gain', answer)
        assert cam.gain(8.0) is returned
        assert cam.get_gain() == pytest.approx(2.0)
        cam.gain(8.0)
        assert cam.get_gain() == pytest.approx(8.0)

    def test_a_raising_write_raises(self):
        cam = SimulatedCamera()
        cam.refuse_write('exposure', 'raise')
        with pytest.raises(RuntimeError, match='exposure node write failed'):
            cam.exposure_t(5.0)

    def test_a_write_the_camera_does_not_answer_raises_and_keeps_the_camera(self):
        # A timed-out write: the camera may answer the next one, so it is
        # not taken for gone, and nothing moved.
        from drivers.exceptions import CameraNoAnswerError

        cam = SimulatedCamera()
        cam.gain(2.0)
        cam.refuse_write('gain', 'no_answer')
        with pytest.raises(CameraNoAnswerError) as raised:
            cam.gain(8.0)
        assert raised.value.reason == 'camera_no_answer'
        assert not cam.is_device_removed()
        assert cam.get_gain() == pytest.approx(2.0)
        cam.gain(8.0)
        assert cam.get_gain() == pytest.approx(8.0)

    def test_a_refusal_no_write_met_is_taken_back(self):
        cam = SimulatedCamera()
        cam.refuse_write('gain', 'false')
        cam.allow_write('gain')
        assert cam.gain(8.0) is not False

    def test_a_node_or_answer_it_lacks_is_refused(self):
        cam = SimulatedCamera()
        with pytest.raises(ValueError, match='focus'):
            cam.refuse_write('focus', 'false')
        with pytest.raises(ValueError, match='maybe'):
            cam.refuse_write('gain', 'maybe')

    def test_a_camera_that_leaves_the_bus_is_removed(self):
        cam = SimulatedCamera()
        cam.leave_bus()
        assert cam.is_device_removed()

    def test_a_camera_that_leaves_the_bus_delivers_no_more_frames(self):
        """A removed camera's stream ends, so a recording on it is lost, not completed."""
        cam = SimulatedCamera()
        cam.open_and_start()
        try:
            handler = cam.cam_image_handler
            assert handler.wait_for_frame_after(handler.frames_delivered, 2.0)
            cam.leave_bus()
            # A frame already on its way when the camera left may still land.
            handler.wait_for_frame_after(handler.frames_delivered, 0.1)
            assert not handler.wait_for_frame_after(handler.frames_delivered, 0.5)
        finally:
            cam.stop_grabbing()


class TestAnUnlitFieldIsDark:
    """The simulated camera answers to light.

    A real camera in a dark box returns a dark frame however long the
    exposure. The simulator could not: nothing in it read illumination,
    and a brightness floor kept the field visible at any gain and any
    exposure. So the whole illumination-failure class -- an LED that
    never came on, a channel dark through a run -- was reproducible only
    on a bench, and a sim capture with no LED lit looked like a good one.
    """

    def test_the_same_camera_is_dark_unlit_and_bright_lit(self):
        cam = SimulatedCamera()
        cam.set_illumination(0.0)
        cam.open_and_start()

        frame = grab_a_frame_made_after_now(cam)
        dark = frame.image.max()

        cam.set_illumination(50.0)
        frame = grab_a_frame_made_after_now(cam)
        bright = frame.image.max()

        assert dark == 0, f'an unlit field must be black, peaked at {dark}'
        assert bright > 0, 'a lit field must show the specimen'

    def test_a_camera_with_no_illumination_source_still_renders(self):
        """A camera built on its own has no scope to ask about light, so it
        renders as it always did rather than describing a fault."""
        cam = SimulatedCamera()
        cam.open_and_start()

        frame = grab_a_frame_made_after_now(cam)

        assert frame.image.max() > 0
