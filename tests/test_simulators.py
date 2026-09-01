# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for simulated LED and motor boards.

Verifies that simulators are drop-in replacements for real hardware:
- Same public API surface
- Correct state tracking
- Thread safety
- Position math matches real boards
"""

import ast
import inspect
import itertools
import pytest
import threading
import time

import numpy as np

# Heavy deps are mocked by tests/conftest.py at module-import time.

from tests.ast_seams import iter_package_modules

from drivers.simulated_ledboard import SimulatedLEDBoard
from drivers.simulated_motorboard import SimulatedMotorBoard
from drivers.simulated_camera import SimulatedCamera
from drivers.camera import Camera
from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard


# ---------------------------------------------------------------------------
# LED Simulator Tests
# ---------------------------------------------------------------------------


class TestSimulatedLEDBoard:
    def test_api_surface_matches_real(self):
        """Simulated board has all public methods of the real board."""
        real_methods = {m for m in dir(LEDBoard) if not m.startswith('_')}
        sim_methods = {m for m in dir(SimulatedLEDBoard) if not m.startswith('_')}
        missing = real_methods - sim_methods
        assert not missing, f'SimulatedLEDBoard missing methods: {missing}'

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

    def test_color_channel_conversion(self):
        board = SimulatedLEDBoard()
        assert board.color2ch('Blue') == 0
        assert board.color2ch('Green') == 1
        assert board.color2ch('Red') == 2
        assert board.color2ch('BF') == 3
        assert board.ch2color(0) == 'Blue'
        assert board.ch2color(3) == 'BF'

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

        threads = [threading.Thread(target=toggle, args=(ch, 50)) for ch in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f'Thread safety errors: {errors}'
        # All LEDs should be off after all toggles complete; check the
        # driver-internal cache (now Phase-3d.5 dead state) directly
        # since the protocol-level reader was retired.
        for color, illumination_ma in board.led_ma.items():
            assert illumination_ma == -1, f'{color} not reset after concurrent toggle'


# ---------------------------------------------------------------------------
# Motor Simulator Tests
# ---------------------------------------------------------------------------


class TestSimulatedMotorBoard:
    def test_api_surface_matches_real(self):
        """Simulated board has all public methods of the real board."""
        real_methods = {m for m in dir(MotorBoard) if not m.startswith('_')}
        sim_methods = {m for m in dir(SimulatedMotorBoard) if not m.startswith('_')}
        missing = real_methods - sim_methods
        assert not missing, f'SimulatedMotorBoard missing methods: {missing}'

    def test_initial_state(self):
        board = SimulatedMotorBoard()
        assert board.found is True
        assert board.is_connected()
        assert board.has_homed() is False
        assert board.has_turret() is False  # default model LS850 (no turret)

    def test_no_turret_model(self):
        board = SimulatedMotorBoard(model='LS850')
        assert board.has_turret() is False

    def test_homing_xyz(self):
        board = SimulatedMotorBoard(timing='instant')
        board.home()
        assert board.has_homed() is True
        assert board.current_pos('X') == 0
        assert board.current_pos('Y') == 0
        assert board.current_pos('Z') == 0

    def test_zhome(self):
        board = SimulatedMotorBoard()
        board.move_abs_pos('Z', 5000)
        assert board.current_pos('Z') > 0
        board.zhome()
        assert board.current_pos('Z') == 0

    def test_thome(self):
        board = SimulatedMotorBoard()
        board.thome()
        assert board.has_thomed() is True

    def test_move_absolute_z(self):
        board = SimulatedMotorBoard()
        board.move_abs_pos('Z', 7000, overshoot_enabled=False)
        pos = board.current_pos('Z')
        assert abs(pos - 7000) < 1  # within rounding

    def test_move_absolute_xy(self):
        board = SimulatedMotorBoard()
        board.move_abs_pos('X', 60000)
        board.move_abs_pos('Y', 40000)
        assert abs(board.current_pos('X') - 60000) < 1
        assert abs(board.current_pos('Y') - 40000) < 1

    def test_move_relative(self):
        board = SimulatedMotorBoard()
        board.move_abs_pos('X', 50000)
        board.move_rel_pos('X', 10000)
        assert abs(board.current_pos('X') - 60000) < 1

    def test_limits_enforced(self):
        board = SimulatedMotorBoard()
        board.move_abs_pos('Z', 99999, overshoot_enabled=False)
        pos = board.current_pos('Z')
        assert pos <= 14000 + 1  # Z max is 14000

    def test_limits_ignored(self):
        board = SimulatedMotorBoard()
        board.move_abs_pos('Z', 99999, overshoot_enabled=False, ignore_limits=True)
        pos = board.current_pos('Z')
        assert pos > 14000

    def test_target_status(self):
        board = SimulatedMotorBoard()
        board.move_abs_pos('X', 50000)
        # Fast mode: position updates instantly but target_status is False
        # for ~3ms (simulates motion monitor detection window)
        import time

        time.sleep(0.005)
        assert board.target_status('X') is True

    def test_conversion_z(self):
        board = SimulatedMotorBoard()
        um = 5000
        ustep = board.z_um2ustep(um)
        um_back = board.z_ustep2um(ustep)
        assert abs(um - um_back) < 0.01

    def test_conversion_xy(self):
        board = SimulatedMotorBoard()
        um = 60000
        ustep = board.xy_um2ustep(um)
        um_back = board.xy_ustep2um(ustep)
        assert abs(um - um_back) < 0.1

    def test_conversion_turret(self):
        board = SimulatedMotorBoard()
        pos = 3
        ustep = board.t_pos2ustep(pos)
        pos_back = board.t_ustep2pos(ustep)
        assert pos == pos_back

    def test_fullinfo(self):
        board = SimulatedMotorBoard(model='LS850T', serial_number='TEST-123')
        info = board.fullinfo()
        assert info['model'] == 'LS850T'
        assert info['serial_number'] == 'TEST-123'

    def test_exchange_command_info(self):
        board = SimulatedMotorBoard()
        resp = board.exchange_command('INFO')
        assert 'SIMULATED' in resp

    def test_disconnect_reconnect(self):
        board = SimulatedMotorBoard()
        board.disconnect()
        assert not board.is_connected()
        resp = board.exchange_command('INFO')
        assert resp is not None
        assert board.is_connected()

    def test_acceleration_stubs(self):
        board = SimulatedMotorBoard()
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
        board = SimulatedMotorBoard()
        config = board.get_axes_config()
        assert 'X' in config
        assert 'Y' in config
        assert 'Z' in config
        assert 'T' in config

    def test_axis_limits(self):
        board = SimulatedMotorBoard()
        z_limits = board.get_axis_limits('Z')
        assert z_limits['min'] == 0
        assert z_limits['max'] == 14000

    def test_thread_safety(self):
        """Concurrent moves should not corrupt state."""
        board = SimulatedMotorBoard()
        errors = []

        def move_axis(axis, positions):
            try:
                for pos in positions:
                    board.move_abs_pos(axis, pos, overshoot_enabled=False)
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

    def test_overshoot_z(self):
        """Z overshoot should work without errors."""
        board = SimulatedMotorBoard()
        board.move_abs_pos('Z', 5000, overshoot_enabled=False)
        board.move_abs_pos('Z', 3000, overshoot_enabled=True)
        pos = board.current_pos('Z')
        assert abs(pos - 3000) < 1

    # --- detect_present_axes tests ---

    def test_detect_present_axes_ls850(self):
        """LS850 should have X, Y, Z (no turret)."""
        board = SimulatedMotorBoard(model='LS850')
        axes = board.detect_present_axes()
        assert 'X' in axes
        assert 'Y' in axes
        assert 'Z' in axes
        assert 'T' not in axes

    def test_detect_present_axes_ls820t(self):
        """LS820T should have Z and T (no XY stage)."""
        board = SimulatedMotorBoard(model='LS820T')
        axes = board.detect_present_axes()
        assert 'Z' in axes
        assert 'T' in axes
        assert 'X' not in axes
        assert 'Y' not in axes

    def test_detect_present_axes_ls850t(self):
        """LS850T should have X, Y, Z, and T."""
        board = SimulatedMotorBoard(model='LS850T')
        axes = board.detect_present_axes()
        assert 'X' in axes
        assert 'Y' in axes
        assert 'Z' in axes
        assert 'T' in axes

    # --- current_pos_steps / target_pos_steps tests ---

    def test_current_pos_steps(self):
        """After a move, current_pos_steps returns raw microstep position."""
        board = SimulatedMotorBoard()
        target_um = 5000
        board.move_abs_pos('Z', target_um, overshoot_enabled=False)
        steps = board.current_pos_steps('Z')
        assert isinstance(steps, int)
        expected_steps = board.z_um2ustep(target_um)
        assert steps == expected_steps

    def test_target_pos_steps(self):
        """target_pos_steps returns raw target microstep position."""
        board = SimulatedMotorBoard()
        target_um = 7000
        board.move_abs_pos('Z', target_um, overshoot_enabled=False)
        steps = board.target_pos_steps('Z')
        assert isinstance(steps, int)
        expected_steps = board.z_um2ustep(target_um)
        assert steps == expected_steps

    # --- homing return value tests ---

    def test_zhome_returns_bool(self):
        """zhome() should return True on success."""
        board = SimulatedMotorBoard()
        result = board.zhome()
        assert result is True

    def test_home_returns_bool(self):
        """home() should return True on success."""
        board = SimulatedMotorBoard(timing='instant')
        result = board.home()
        assert result is True

    def test_thome_returns_bool(self):
        """thome() should return True on success."""
        board = SimulatedMotorBoard(model='LS850T')
        result = board.thome()
        assert result is True

    def test_thome_no_turret(self):
        """thome() on a non-turret model should still return True."""
        board = SimulatedMotorBoard(model='LS850')
        result = board.thome()
        assert result is True


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
        board = SimulatedMotorBoard(model=model)
        assert board.found is True
        assert board.is_connected()

    @pytest.mark.parametrize('model', TURRET_MODELS)
    def test_turret_model_detected(self, model):
        board = SimulatedMotorBoard(model=model)
        assert board.has_turret() is True

    @pytest.mark.parametrize('model', NON_TURRET_MODELS)
    def test_non_turret_model_detected(self, model):
        board = SimulatedMotorBoard(model=model)
        assert board.has_turret() is False

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_fullinfo_reports_model(self, model):
        board = SimulatedMotorBoard(model=model, serial_number='SN-TEST')
        info = board.fullinfo()
        assert info['model'] == model
        assert info['serial_number'] == 'SN-TEST'

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_z_axis_works(self, model):
        board = SimulatedMotorBoard(model=model)
        board.move_abs_pos('Z', 5000, overshoot_enabled=False)
        assert abs(board.current_pos('Z') - 5000) < 1

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_homing_works(self, model):
        board = SimulatedMotorBoard(model=model, timing='instant')
        board.home()
        assert board.has_homed() is True

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_motorconfig_travel_limits(self, model):
        board = SimulatedMotorBoard(model=model)
        mc = board.motorconfig
        # All models should have valid travel limits
        assert mc.travel_limit_mm('X') > 0
        assert mc.travel_limit_mm('Y') > 0
        assert mc.travel_limit_mm('Z') > 0

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_axes_config_populated(self, model):
        board = SimulatedMotorBoard(model=model)
        config = board.get_axes_config()
        assert 'X' in config
        assert 'Y' in config
        assert 'Z' in config
        for axis in ('X', 'Y', 'Z'):
            assert config[axis]['limits']['max'] > 0

    @pytest.mark.parametrize('model', TURRET_MODELS)
    def test_turret_positions(self, model):
        board = SimulatedMotorBoard(model=model)
        mc = board.motorconfig
        for pos in range(1, 5):
            usteps = mc.turret_position_usteps(pos)
            assert isinstance(usteps, int)

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_center_command(self, model):
        """CENTER should move to stage center for all models."""
        board = SimulatedMotorBoard(model=model)
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
        import pathlib

        mc = MotorConfig(defaults_file=pathlib.Path('data/motorconfig_defaults.json'))
        assert mc.model() in ('LS850', 'LS850T')
        assert mc.travel_limit_mm('X') == 120
        assert mc.travel_limit_mm('Y') == 80
        assert mc.travel_limit_mm('Z') == 14
        assert mc.usteps_per_mm('Z') == 170666
        assert mc.lens_focal_length() == 47.8
        assert mc.pixel_size() == 2.0

    def test_update_from_board_overrides(self):
        from drivers.motorconfig import MotorConfig
        import pathlib

        mc = MotorConfig(defaults_file=pathlib.Path('data/motorconfig_defaults.json'))
        mc.update_from_board({'Axis Travel Limit': {'Z': 20}})
        assert mc.travel_limit_mm('Z') == 20
        # X/Y unchanged
        assert mc.travel_limit_mm('X') == 120

    def test_missing_section_returns_default(self):
        from drivers.motorconfig import MotorConfig
        import pathlib

        mc = MotorConfig(defaults_file=pathlib.Path('data/motorconfig_defaults.json'))
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
    """Verify that set_objective enables scale bar rendering."""

    def test_objective_none_at_init(self):
        """Lumascope starts with no objective set."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        assert scope.runtime_state._objective is None

    def test_set_objective_populates(self):
        """set_objective() should populate _objective dict."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        scope.runtime_state.set_objective('20x Oly')
        assert scope.runtime_state._objective is not None
        assert scope.runtime_state._objective['magnification'] == 20

    def test_scale_bar_disabled_without_objective(self):
        """Scale bar enabled but no objective -> use_scale_bar forced False."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        scope.imaging.set_scale_bar(enabled=True)
        assert scope.imaging._scale_bar['enabled'] is True
        assert scope.runtime_state._objective is None
        # Internal logic forces use_scale_bar = False when _objective is None

    def test_scale_bar_works_with_objective(self):
        """Scale bar with objective set should proceed."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        scope.runtime_state.set_objective('20x Oly')
        scope.imaging.set_scale_bar(enabled=True)
        assert scope.imaging._scale_bar['enabled'] is True
        assert scope.runtime_state._objective is not None


# ---------------------------------------------------------------------------
# Camera Simulator Tests
# ---------------------------------------------------------------------------


class TestSimulatedCamera:
    def test_api_surface_matches_base(self):
        """Simulated camera implements all abstract methods from Camera ABC."""
        abstract_methods = {
            m for m in dir(Camera) if not m.startswith('_') and callable(getattr(Camera, m, None))
        }
        sim_methods = {m for m in dir(SimulatedCamera) if not m.startswith('_')}
        missing = abstract_methods - sim_methods
        assert not missing, f'SimulatedCamera missing methods: {missing}'

    def test_connects_on_init(self):
        cam = SimulatedCamera()
        assert cam.active is True
        assert cam.is_connected()
        assert cam.model_name == 'SimulatedCamera-1920x1200'

    def test_disconnect_reconnect(self):
        cam = SimulatedCamera()
        cam.disconnect()
        assert not cam.is_connected()
        cam.connect()
        assert cam.is_connected()

    def test_default_frame_size(self):
        cam = SimulatedCamera()
        size = cam.get_frame_size()
        assert size['width'] == 1920
        assert size['height'] == 1200

    def test_set_frame_size(self):
        cam = SimulatedCamera()
        cam.set_frame_size(960, 600)
        size = cam.get_frame_size()
        assert size['width'] == 960
        assert size['height'] == 600

    def test_frame_size_snaps_to_valid(self):
        cam = SimulatedCamera()
        cam.set_frame_size(100, 7)  # Not multiples of 48/4
        size = cam.get_frame_size()
        assert size['width'] % 48 == 0
        assert size['height'] % 4 == 0

    def test_set_frame_size_returns_delivered_geometry(self):
        cam = SimulatedCamera()
        delivered = cam.set_frame_size(640, 482)
        # Snapped to the 48/4 grid; the return must equal what get_frame_size
        # then reports, so callers can cache it without a read-back.
        assert delivered == {'width': 624, 'height': 480}
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
        result, ts = cam.grab()
        assert result is True
        assert ts is not None
        assert isinstance(cam.array, np.ndarray)
        assert cam.array.shape == (1200, 1920)
        assert cam.array.dtype == np.uint8

    def test_grab_new_capture(self):
        cam = SimulatedCamera()
        # Driver contract is float seconds (verified across all five camera
        # drivers). `timeout=1000` here would have been 1000 seconds and
        # passed only because the simulator doesn't honor the timeout.
        cam.open_and_start()
        result, ts = cam.grab_new_capture(timeout_s=5.0)
        assert result is True
        assert ts is not None
        assert isinstance(cam.array, np.ndarray)

    def test_grab_respects_binning(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_binning_size(2)
        cam.grab()
        assert cam.array.shape == (600, 960)

    def test_grab_mono12_dtype(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_pixel_format('Mono12')
        cam.grab()
        assert cam.array.dtype == np.uint16

    def test_grab_not_grabbing_returns_false(self):
        cam = SimulatedCamera()
        cam.stop_grabbing()
        result, _ts = cam.grab()
        assert result is False

    def test_image_brightness_varies_with_exposure(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.exposure_t(1.0)  # 1ms -- dim
        cam.grab()
        dim = cam.array.mean()

        cam.exposure_t(100.0)  # 100ms -- bright
        cam.grab()
        bright = cam.array.mean()

        assert bright > dim

    def test_image_brightness_varies_with_gain(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.gain(0.1)
        cam.grab()
        low = cam.array.mean()

        cam.gain(10.0)
        cam.grab()
        high = cam.array.mean()

        assert high > low

    # -- Test patterns --

    def test_black_pattern(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='Black')
        cam.grab()
        assert cam.array.max() == 0

    def test_white_pattern(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='White')
        cam.grab()
        assert cam.array.max() == 255

    def test_noise_pattern(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='Noise')
        cam.grab()
        # Noise should have some variance
        assert cam.array.std() > 0

    def test_disable_pattern_returns_the_specimen(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='Black')
        cam.set_test_pattern(enabled=False)
        cam.grab()
        # The specimen field has variation; the black pattern it replaced does not
        assert cam.array.std() > 0


class TestNoPatternRequestedRendersTheSpecimen:
    """Turning a test pattern off, or never asking for one, must show the
    specimen field -- not the static ramp the cycle was built to replace.

    The ramp was reachable three ways: the constructor default, the
    disable path, and any unrecognized pattern name. Landing on it undid
    the cycle for the rest of the session.
    """

    def test_disabling_a_pattern_restores_the_moving_specimen(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='Black')
        cam.set_test_pattern(enabled=False)
        frames = [np.asarray(cam._generate_image(), dtype=float) for _ in range(4)]
        spread = max(f.max() for f in frames) - min(f.min() for f in frames)
        diffs = [float(np.mean(np.abs(a - b))) for a, b in itertools.pairwise(frames)]
        assert spread > 0
        assert max(diffs) > 0, 'consecutive frames are identical -- this is a static image'

    def test_a_fresh_camera_renders_the_specimen_without_load_cycle_images(self):
        cam = SimulatedCamera()
        frames = [np.asarray(cam._generate_image(), dtype=float) for _ in range(4)]
        diffs = [float(np.mean(np.abs(a - b))) for a, b in itertools.pairwise(frames)]
        assert max(diffs) > 0, 'the constructor default is a static image'

    def test_an_unknown_pattern_name_warns_and_still_renders(self):
        # The suite mocks lvp_logger, so the driver's `logger` is a MagicMock
        # and no handler or caplog capture can see the call -- assert on the
        # mock, which is how the rest of the suite checks driver logging.
        import drivers.simulated_camera as sim_cam

        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='NotAPattern')
        sim_cam.logger.warning.reset_mock()
        img = np.asarray(cam._generate_image(), dtype=float)

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
        img = np.asarray(cam._generate_image(), dtype=float)
        assert img.max() > 255, (
            f'12-bit fallback peaks at {img.max()} -- not scaled to the 4095 range'
        )

    def test_disabling_a_pattern_does_not_enable_exposure_pacing(self):
        """grab() paces frame delivery on exposure ONLY for 'image_cycle'.
        The disable path must not borrow that behaviour by reusing the name."""
        cam = SimulatedCamera()
        cam.open_and_start()
        cam.set_test_pattern(enabled=False)
        assert cam._test_pattern != 'image_cycle'

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
        assert cam.max_exposure == 10_000

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
        cam = SimulatedCamera(width=480, height=300, grab_delay=0)
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        cam.grab()
        # Focus target should have features -- not uniform
        assert cam.array.std() > 5

    def test_set_get_z_position(self):
        cam = SimulatedCamera(grab_delay=0)
        cam.set_z_position(3000.0)
        assert cam.get_z_position() == 3000.0

    def test_set_get_focal_z(self):
        cam = SimulatedCamera(grab_delay=0)
        cam.set_focal_z(7000.0)
        assert cam.get_focal_z() == 7000.0

    def test_focus_score_peaks_at_focal_z(self):
        """Vollath F4 focus score should be highest at the focal point."""
        from modules.autofocus_functions import focus_vollath4_original

        cam = SimulatedCamera(width=480, height=300, grab_delay=0)
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        cam.set_focal_z(5000.0)
        cam.set_blur_per_um(0.01)

        scores = {}
        for z in [3000, 4000, 4500, 4800, 5000, 5200, 5500, 6000, 7000]:
            cam.set_z_position(float(z))
            cam.grab()
            scores[z] = focus_vollath4_original(image=cam.array)

        # Best score should be at z=5000 (focal point)
        best_z = max(scores, key=scores.get)
        assert best_z == 5000, f'Expected best focus at 5000, got {best_z}. Scores: {scores}'

    def test_focus_score_decreases_with_defocus(self):
        """Focus score should decrease as we move away from focal point."""
        from modules.autofocus_functions import focus_vollath4_original

        cam = SimulatedCamera(width=480, height=300, grab_delay=0)
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        cam.set_focal_z(5000.0)
        cam.set_blur_per_um(0.01)

        # Get scores at increasing distances from focus
        cam.set_z_position(5000.0)
        cam.grab()
        score_at_focus = focus_vollath4_original(image=cam.array)

        cam.set_z_position(5500.0)
        cam.grab()
        score_near = focus_vollath4_original(image=cam.array)

        cam.set_z_position(6500.0)
        cam.grab()
        score_far = focus_vollath4_original(image=cam.array)

        assert score_at_focus > score_near > score_far, (
            f'Scores should decrease: focus={score_at_focus}, near={score_near}, far={score_far}'
        )

    def test_focus_curve_is_symmetric(self):
        """Focus scores should be roughly symmetric around focal point."""
        from modules.autofocus_functions import focus_vollath4_original

        cam = SimulatedCamera(width=480, height=300, grab_delay=0)
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        cam.set_focal_z(5000.0)
        cam.set_blur_per_um(0.01)

        cam.set_z_position(4000.0)
        cam.grab()
        score_below = focus_vollath4_original(image=cam.array)

        cam.set_z_position(6000.0)
        cam.grab()
        score_above = focus_vollath4_original(image=cam.array)

        # Within 20% of each other (both 1000um from focus)
        ratio = score_below / score_above if score_above != 0 else float('inf')
        assert 0.8 < ratio < 1.2, f'Asymmetric: below={score_below}, above={score_above}'

    def test_z_position_func_callback(self):
        """Camera auto-queries Z from callback when generating focus_target."""
        z_val = [5000.0]
        cam = SimulatedCamera(width=480, height=300, grab_delay=0, z_position_func=lambda: z_val[0])
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        cam.set_focal_z(5000.0)

        # At focus
        cam.grab()
        assert cam.get_z_position() == 5000.0

        # Move via callback
        z_val[0] = 3000.0
        cam.grab()
        assert cam.get_z_position() == 3000.0

    def test_no_blur_at_focal_point(self):
        """Image at focal point should be identical to unblurred target."""
        cam = SimulatedCamera(width=480, height=300, grab_delay=0)
        cam.open_and_start()
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        cam.set_focal_z(5000.0)

        cam.set_z_position(5000.0)
        cam.grab()
        sharp = cam.array.copy()

        # Defocused image should differ
        cam.set_z_position(7000.0)
        cam.grab()
        blurred = cam.array

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
        assert cam.profile.model_name == 'SimulatedCamera-1920x1200'

    def test_profile_sensor_info(self):
        cam = SimulatedCamera()
        assert cam.profile.sensor == 'Simulated'
        assert cam.profile.pixel_size_um == 2.0
        assert cam.profile.shutter == 'global'

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
        assert cam.profile.gain.total_max_db == 20.0
        assert cam.profile.gain.analog_max_db == 20.0

    def test_profile_native_resolution(self):
        cam = SimulatedCamera()
        assert cam.profile.native_resolution == {'width': 1920, 'height': 1200}

    def test_profile_capabilities(self):
        cam = SimulatedCamera()
        assert cam.profile.has_auto_gain is True
        assert cam.profile.has_auto_exposure is True
        assert cam.profile.has_temperature is True
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


class TestSpecimenCycleFrames:
    """The fallback cycle frames must drift, not strobe.

    The cycle advances once per generated frame, so four unrelated patterns
    flash at frame rate -- the earlier horizontal-ramp / vertical-ramp /
    bullseye / checkerboard set did exactly that, and it was unwatchable in
    simulate mode. These pin the properties that make the replacement calm:
    neighbouring frames stay close, none is fully black or blown out, and the
    wrap back to the first frame is the same size step as the others.
    """

    def test_frames_are_distinct_so_a_live_stream_is_visible(self):
        frames = SimulatedCamera._make_specimen_frames(600, 800)
        assert len(frames) == 4
        assert len({f.tobytes() for f in frames}) == 4, (
            'identical frames make a running stream indistinguishable from a frozen one'
        )

    def test_consecutive_frames_stay_close_including_the_wrap(self):
        frames = SimulatedCamera._make_specimen_frames(600, 800)
        deltas = [
            float(np.abs(frames[i].astype(int) - frames[(i + 1) % len(frames)].astype(int)).mean())
            for i in range(len(frames))
        ]
        # Calibrated against the live display, not against the old patterns:
        # an 8 px sampling offset (~12 grey levels) was rejected as looking
        # like the sample being jostled, and 1 px (~1.6) was accepted. The
        # ceiling sits between 1 px and 2 px so a creeping increase trips it
        # while the accepted setting has ~2x headroom for frame-size variation.
        # The wrap is included deliberately: sampling around a circle is what
        # keeps the last->first step the same size as the others.
        assert max(deltas) < 3.0, f'frames too far apart, this reads as jostling: {deltas}'
        assert min(deltas) > 1.0, f'frames too similar to read as motion: {deltas}'

    def test_never_fully_black_or_blown_out(self):
        for frame in SimulatedCamera._make_specimen_frames(600, 800):
            assert frame.min() > 0, 'a crushed frame reads as a dead camera'
            assert frame.max() < 255, 'a blown frame hides the exposure control'

    def test_deterministic_for_a_fixed_seed(self):
        first = SimulatedCamera._make_specimen_frames(600, 800)
        second = SimulatedCamera._make_specimen_frames(600, 800)
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))

    def test_matches_requested_frame_size(self):
        for frame in SimulatedCamera._make_specimen_frames(1200, 1920):
            assert frame.shape == (1200, 1920)
            assert frame.dtype == np.uint8


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
        assert p.has_temperature is True

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

        p = lookup_profile('SimulatedCamera-1920x1200')
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
        assert p.alignment == {'width': 4, 'height': 4}


class TestTimingModes:
    """Verify timing mode switching across all simulators."""

    def test_motor_instant_mode(self):
        m = SimulatedMotorBoard(timing='instant')
        assert m._cmd_delay == 0.0
        assert m._simulate_move_duration is False
        assert m._fast_move_duration == 0.0

    def test_motor_fast_mode(self):
        m = SimulatedMotorBoard(timing='fast')
        assert m._cmd_delay > 0  # 1ms minimum -- nothing returns instantly
        assert m._simulate_move_duration is True
        assert m._fast_move_duration > 0  # Brief ~3ms per move

    def test_motor_realistic_mode(self):
        m = SimulatedMotorBoard(timing='realistic')
        assert m._cmd_delay > 0
        assert m._simulate_move_duration is True

    def test_motor_switch_mode(self):
        m = SimulatedMotorBoard(timing='fast')
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
        m = SimulatedMotorBoard(timing='realistic')
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

        m = SimulatedMotorBoard(timing='fast')
        m.move_abs_pos('Z', 10000.0)
        # Position is instant
        assert m.current_pos('Z') == pytest.approx(10000.0, abs=1.0)
        # target_status needs brief delay
        time.sleep(0.005)
        assert m.target_status('Z') is True

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
            SimulatedMotorBoard(timing='turbo')
        with pytest.raises(ValueError):
            SimulatedLEDBoard(timing='turbo')
        with pytest.raises(ValueError):
            SimulatedCamera(timing='turbo')


class TestFailureInjection:
    """Verify failure injection for testing error recovery paths."""

    # --- Motor board ---

    def test_motor_fail_after_disconnects(self):
        """Motor board should return None after N commands."""
        m = SimulatedMotorBoard(fail_after=3)
        assert m.exchange_command('INFO') is not None  # cmd 1
        assert m.exchange_command('INFO') is not None  # cmd 2
        assert m.exchange_command('INFO') is not None  # cmd 3
        assert m.exchange_command('INFO') is None  # cmd 4 -- disconnected
        assert m.driver is None

    def test_motor_fail_after_sets_found_false(self):
        """After injected disconnect, found should be False."""
        m = SimulatedMotorBoard(fail_after=1)
        assert m.found is True
        m.exchange_command('INFO')  # cmd 1 -- succeeds
        m.exchange_command('INFO')  # cmd 2 -- fails
        assert m.found is False

    def test_motor_fail_on_specific_command(self):
        """Motor board should return None for targeted commands only."""
        m = SimulatedMotorBoard(fail_on={'ZHOME'})
        assert m.exchange_command('INFO') is not None  # OK
        assert m.exchange_command('ZHOME') is None  # targeted failure
        assert m.exchange_command('INFO') is not None  # still connected
        assert m.driver is not None  # not disconnected

    def test_motor_fail_on_multiple_commands(self):
        """Multiple commands can be targeted for failure."""
        m = SimulatedMotorBoard(fail_on={'ZHOME', 'THOME'}, timing='instant')
        assert m.exchange_command('ZHOME') is None
        assert m.exchange_command('THOME') is None
        assert m.exchange_command('HOME') is not None  # not in fail set

    def test_motor_no_failure_by_default(self):
        """Without fail params, simulator works normally."""
        m = SimulatedMotorBoard()
        for _ in range(100):
            assert m.exchange_command('INFO') is not None

    def test_motor_fail_after_affects_move(self):
        """Mid-protocol disconnect: move starts OK, then fails."""
        m = SimulatedMotorBoard(fail_after=5, timing='instant')
        m.exchange_command('HOME')  # cmd 1
        m.move_abs_pos('Z', 5000)  # uses multiple commands
        # Eventually commands fail
        result = m.exchange_command('ACTUAL_RZ')  # noqa: F841 -- deferred
        # After enough commands, should get None
        # (exact count depends on internal commands used by move_abs_pos)

    # --- LED board ---

    def test_led_fail_after_disconnects(self):
        """LED board should return None after N commands."""
        led = SimulatedLEDBoard(fail_after=2)
        assert led.exchange_command('LEDS_ENT') is not None  # cmd 1
        assert led.exchange_command('LED0_100') is not None  # cmd 2
        assert led.exchange_command('LEDS_OFF') is None  # cmd 3 -- disconnected
        assert led.driver is None

    def test_led_fail_after_sets_found_false(self):
        """After injected disconnect, found should be False."""
        led = SimulatedLEDBoard(fail_after=1)
        assert led.found is True
        led.exchange_command('LEDS_ENT')  # cmd 1 -- succeeds
        led.exchange_command('LED0_100')  # cmd 2 -- fails
        assert led.found is False

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

    def test_led_fast_path_also_fails(self):
        """_write_command_fast should also respect fail_after."""
        led = SimulatedLEDBoard(fail_after=2)
        led._write_command_fast('LED0_100')  # cmd 1
        led._write_command_fast('LED1_100')  # cmd 2
        led._write_command_fast('LED2_100')  # cmd 3 -- should disconnect
        assert led.driver is None
        assert led.found is False


# ---------------------------------------------------------------------------
# Drop-in-replacement guards (all three board pairs)
# ---------------------------------------------------------------------------
#
# The three `test_api_surface_matches_real` / `_matches_base` tests above
# ask one question -- does the simulator have every public NAME the real
# board has -- and all three passed while a documented SDK method was
# crashing against the simulator. `IlluminationAPI.wait_until_led_on()`
# calls `self._driver.wait_until_on(timeout_s)`; the real board accepts
# `timeout_s`, the simulator's override did not, so the call raised
# TypeError in every simulated run. A name-only comparison cannot see
# that, and the one test that touched the method on a sim scope set
# `_led_driver = None` first, so it never reached the driver.
#
# These guards close the two gaps a name check leaves.
#
# Why the drift exists at all, which is the part worth remembering:
#
#     SimulatedCamera(Camera)   inherits the ABC   -> 0 divergences
#     SimulatedLEDBoard         inherits nothing   -> 4 divergences
#     SimulatedMotorBoard       inherits nothing   -> 2 divergences
#
# The pair that shares a contract has never drifted; the two that share
# none hold every divergence. So the root is not six stale signatures, it
# is that nothing makes LED/motor drift UNREPRESENTABLE. Hand-mirroring
# the parameters would add defaults that do nothing on a simulator --
# decorative parity that reads as fixed while the next divergence is
# still free to appear. The five inert divergences are therefore
# allowlisted below against that structural fix, not patched here. The
# one that was actually crashing (`wait_until_on` missing `timeout_s`)
# is fixed in the simulator, where the parameter now bounds a loop that
# could previously spin forever.


_BOARD_PAIRS = (
    ('LEDBoard', LEDBoard, SimulatedLEDBoard),
    ('MotorBoard', MotorBoard, SimulatedMotorBoard),
    ('Camera', Camera, SimulatedCamera),
)

# Sim-only public names per board, recorded at introduction. This is a
# RATCHET, not an allowlist: it carries no per-name justification and
# exists only so new sim-only surface cannot accrete unnoticed. A
# deliberate addition raises the number here in the same commit.
#
# What the current entries cover: timing-mode controls and TIMING_*
# constants (test-harness affordances, on every simulator),
# `load_cycle_images` plus the camera's virtual-specimen focus modeling
# (`set_focal_z`, `set_blur_per_um`, ...), and six firmware-update
# methods on the motor simulator that anticipate the firmware-updating
# work landing on the FW branch. Those six have no caller yet; when that
# code calls them, `test_no_production_code_calls_simulator_only_names`
# below will require the REAL board to gain the same names, which is the
# point.
_SIM_ONLY_NAME_BUDGET = {
    'LEDBoard': 4,
    'MotorBoard': 12,
    'Camera': 13,
}

# Parameter divergences that stay until the simulators gain a shared
# contract with the real boards. Every one of these parameters is inert
# on a simulator -- there is no firmware to soft-reset, no empty response
# to stop on, no unsupported-command warning to suppress -- so adding
# them by hand would mean defaults that do nothing, and the next
# divergence would still be free to appear. Keyed by (board, method) so
# the entry survives reformatting.
_PARAM_DIVERGENCE_ALLOWLIST = {
    ('LEDBoard', 'enter_raw_repl'),
    ('LEDBoard', 'exchange_command'),
    ('LEDBoard', 'led_on'),
    ('MotorBoard', 'enter_raw_repl'),
    ('MotorBoard', 'exchange_command'),
}

# Production sites allowed to touch simulator-only surface. The single
# entry is the construction branch that BUILDS the simulator: inside
# `if simulate:` the driver provably is a SimulatedCamera, which is the
# one place production code can know that. Keyed by (file, name).
_SIM_ONLY_CALL_ALLOWLIST = {
    ('modules/lumascope_api/_lumascope.py', 'load_cycle_images'),
}


def _public_names(cls):
    return {name for name in dir(cls) if not name.startswith('_')}


def _sim_only_names(real, sim):
    return _public_names(sim) - _public_names(real)


@pytest.mark.parametrize('label,real,sim', _BOARD_PAIRS, ids=[p[0] for p in _BOARD_PAIRS])
def test_simulator_accepts_the_same_parameters_as_the_real_board(label, real, sim):
    """A call that works on the real board must work on the simulator.

    Compares PARAMETER NAMES only, deliberately. Annotations diverge
    harmlessly all over these classes -- the simulators are annotated
    more completely than the real boards, 15 such differences on the LED
    pair alone -- and an annotation never changes whether a call is
    accepted. A parameter name does.

    Bidirectional, because both directions are real hazards: a parameter
    the simulator lacks crashes every simulated run of a production code
    path, and a parameter only the simulator has invites test code to
    depend on something hardware will reject.
    """
    divergences = []
    for name in sorted(_public_names(real) & _public_names(sim)):
        try:
            real_params = list(inspect.signature(getattr(real, name)).parameters)
            sim_params = list(inspect.signature(getattr(sim, name)).parameters)
        except (TypeError, ValueError):
            continue  # C-implemented or otherwise non-introspectable
        if real_params != sim_params and (label, name) not in _PARAM_DIVERGENCE_ALLOWLIST:
            divergences.append(f'  {name}:\n      real={real_params}\n      sim ={sim_params}')

    assert not divergences, (
        f'{sim.__name__} is not a drop-in replacement for {label} -- these '
        f'methods take different parameters, so a caller written against one '
        f'breaks against the other:\n' + '\n'.join(divergences)
    )


@pytest.mark.parametrize('label,real,sim', _BOARD_PAIRS, ids=[p[0] for p in _BOARD_PAIRS])
def test_no_production_code_calls_simulator_only_names(label, real, sim):
    """Production code may not depend on surface only the simulator has.

    Simulators legitimately carry extra surface -- timing controls,
    virtual-specimen modeling, image-cycle loading -- so demanding an
    empty sim-only set would mean a large allowlist whose upkeep exceeds
    its signal. The hazard worth guarding is narrower and exact: code
    under `modules/` or `ui/` that calls a name only the simulator has
    works in every test and fails on hardware.

    Tests are excluded on purpose. Driving simulator-only affordances is
    what test code is FOR -- `conftest.sim_scope` calls `set_timing_mode`
    and `load_cycle_images`.
    """
    sim_only = _sim_only_names(real, sim)
    if not sim_only:
        pytest.skip(f'{sim.__name__} has no simulator-only public names')

    hits = []
    for rel_path, tree in iter_package_modules(('modules', 'ui')):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in sim_only:
                continue
            if (rel_path, node.attr) in _SIM_ONLY_CALL_ALLOWLIST:
                continue
            hits.append(f'  {rel_path}:{node.lineno}: .{node.attr}')

    assert not hits, (
        f'Production code references {sim.__name__}-only surface, which does '
        f'not exist on {label} and will fail on hardware:\n'
        + '\n'.join(sorted(hits))
        + f'\n\nFix by adding the name to {label} (making it real) or by '
        f'removing the production dependency.'
    )


def test_no_allowlist_entry_outlives_its_divergence():
    """Delete an exemption the moment the thing it excuses is gone.

    A stale entry is worse than no entry: it silently excuses the NEXT
    divergence on the same method, which is exactly the regression this
    guard exists to catch. Both allowlists are checked, so landing the
    shared-contract fix forces the entries out in the same commit.
    """
    stale_params = []
    for label, real, sim in _BOARD_PAIRS:
        for name in sorted(_public_names(real) & _public_names(sim)):
            if (label, name) not in _PARAM_DIVERGENCE_ALLOWLIST:
                continue
            try:
                real_params = list(inspect.signature(getattr(real, name)).parameters)
                sim_params = list(inspect.signature(getattr(sim, name)).parameters)
            except (TypeError, ValueError):
                continue
            if real_params == sim_params:
                stale_params.append((label, name))

    shared = {
        (label, name)
        for label, real, sim in _BOARD_PAIRS
        for name in _public_names(real) & _public_names(sim)
    }
    unknown = sorted(entry for entry in _PARAM_DIVERGENCE_ALLOWLIST if entry not in shared)

    assert not stale_params, (
        f'These methods now agree and their _PARAM_DIVERGENCE_ALLOWLIST '
        f'entries must be deleted: {sorted(stale_params)}'
    )
    assert not unknown, (
        f'These _PARAM_DIVERGENCE_ALLOWLIST entries name a method that no '
        f'longer exists on both classes: {unknown}'
    )

    sim_only_names = set()
    for _, real, sim in _BOARD_PAIRS:
        sim_only_names |= _sim_only_names(real, sim)
    stale_calls = sorted(
        entry for entry in _SIM_ONLY_CALL_ALLOWLIST if entry[1] not in sim_only_names
    )
    assert not stale_calls, (
        f'These _SIM_ONLY_CALL_ALLOWLIST entries name surface that is no '
        f'longer simulator-only, so the exemption must be deleted: {stale_calls}'
    )


@pytest.mark.parametrize('label,real,sim', _BOARD_PAIRS, ids=[p[0] for p in _BOARD_PAIRS])
def test_simulator_only_surface_does_not_grow(label, real, sim):
    """Ratchet: sim-only public surface may not accrete unnoticed.

    Every name here is one the real board does not have, so each is a
    place production code could come to depend on something hardware
    cannot do. Growth should be a decision, not a side effect.
    """
    sim_only = _sim_only_names(real, sim)
    budget = _SIM_ONLY_NAME_BUDGET[label]
    assert len(sim_only) <= budget, (
        f'{sim.__name__} now has {len(sim_only)} public names absent from '
        f'{label}, over the recorded {budget}: {sorted(sim_only)}\n\n'
        f'If the addition is deliberate, raise _SIM_ONLY_NAME_BUDGET '
        f"['{label}'] in this commit and say why in the message. If it is "
        f'not, the name probably belongs on {label} too.'
    )
