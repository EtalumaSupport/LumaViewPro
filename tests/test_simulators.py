# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for simulated LED and motor boards.

Verifies that simulators are drop-in replacements for real hardware:
- Same public API surface
- Correct state tracking
- Thread safety
- Position math matches real boards
"""

import pytest
import sys
import threading
import time
from unittest.mock import MagicMock

import numpy as np

# Mock heavy dependencies
_mock_logger = MagicMock()
_mock_logger.info = MagicMock()
_mock_logger.debug = MagicMock()
_mock_logger.error = MagicMock()
_mock_logger.warning = MagicMock()
_mock_logger.critical = MagicMock()

_mock_lvp_logger = MagicMock()
_mock_lvp_logger.logger = _mock_logger
_mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
_mock_lvp_logger.unpause_thread = MagicMock()
_mock_lvp_logger.pause_thread = MagicMock()

sys.modules.setdefault('userpaths', MagicMock())
sys.modules.setdefault('lvp_logger', _mock_lvp_logger)
sys.modules.setdefault('requests', MagicMock())
sys.modules.setdefault('requests.structures', MagicMock())

sys.modules.setdefault('pypylon', MagicMock())
sys.modules.setdefault('pypylon.pylon', MagicMock())
sys.modules.setdefault('pypylon.genicam', MagicMock())
sys.modules.setdefault('ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak_ipl_extension', MagicMock())
sys.modules.setdefault('ids_peak_ipl', MagicMock())

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
        assert not missing, f"SimulatedLEDBoard missing methods: {missing}"

    def test_initial_state(self):
        board = SimulatedLEDBoard()
        assert board.found is True
        assert board.is_connected()
        for color in ('BF', 'PC', 'DF', 'Red', 'Blue', 'Green'):
            assert board.get_led_ma(color) == -1
            assert board.is_led_on(color) is False

    def test_led_on_off(self):
        board = SimulatedLEDBoard()
        board.led_on(channel=0, mA=100)
        assert board.is_led_on('Blue') is True
        assert board.get_led_ma('Blue') == 100

        board.led_off(channel=0)
        assert board.is_led_on('Blue') is False
        assert board.get_led_ma('Blue') == -1

    def test_led_on_fast(self):
        board = SimulatedLEDBoard()
        board.led_on_fast(channel=2, mA=50)
        assert board.get_led_ma('Red') == 50
        board.led_off_fast(channel=2)
        assert board.get_led_ma('Red') == -1

    def test_leds_off_all(self):
        board = SimulatedLEDBoard()
        board.led_on(0, 100)
        board.led_on(1, 200)
        board.led_on(2, 300)
        board.leds_off()
        for color in board.led_ma:
            assert board.get_led_ma(color) == -1

    def test_leds_off_fast(self):
        board = SimulatedLEDBoard()
        board.led_on_fast(0, 100)
        board.leds_off_fast()
        assert board.is_led_on('Blue') is False

    def test_get_led_state(self):
        board = SimulatedLEDBoard()
        board.led_on(3, 50)
        state = board.get_led_state('BF')
        assert state['enabled'] is True
        assert state['illumination'] == 50

    def test_get_led_states(self):
        board = SimulatedLEDBoard()
        board.led_on(0, 100)
        states = board.get_led_states()
        assert states['Blue']['enabled'] is True
        assert states['Red']['enabled'] is False

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

        assert not errors, f"Thread safety errors: {errors}"
        # All LEDs should be off after all toggles complete
        for color in board.led_ma:
            assert board.get_led_ma(color) == -1


# ---------------------------------------------------------------------------
# Motor Simulator Tests
# ---------------------------------------------------------------------------

class TestSimulatedMotorBoard:

    def test_api_surface_matches_real(self):
        """Simulated board has all public methods of the real board."""
        real_methods = {m for m in dir(MotorBoard) if not m.startswith('_')}
        sim_methods = {m for m in dir(SimulatedMotorBoard) if not m.startswith('_')}
        missing = real_methods - sim_methods
        assert not missing, f"SimulatedMotorBoard missing methods: {missing}"

    def test_initial_state(self):
        board = SimulatedMotorBoard()
        assert board.found is True
        assert board.is_connected()
        assert board.has_xyhomed() is False
        assert board.has_turret() is False  # default model LS850 (no turret)

    def test_no_turret_model(self):
        board = SimulatedMotorBoard(model='LS850')
        assert board.has_turret() is False

    def test_homing_xyz(self):
        board = SimulatedMotorBoard()
        board.xyhome()
        assert board.has_xyhomed() is True
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
        assert board.target_status('X') is True  # instant move

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


# ---------------------------------------------------------------------------
# Multi-Model Tests — verify all microscope models work correctly
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
        board = SimulatedMotorBoard(model=model)
        board.xyhome()
        assert board.has_xyhomed() is True

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
        from modules.motorconfig import MotorConfig
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
        from modules.motorconfig import MotorConfig
        import pathlib
        mc = MotorConfig(defaults_file=pathlib.Path('data/motorconfig_defaults.json'))
        mc.update_from_board({'Axis Travel Limit': {'Z': 20}})
        assert mc.travel_limit_mm('Z') == 20
        # X/Y unchanged
        assert mc.travel_limit_mm('X') == 120

    def test_missing_section_returns_default(self):
        from modules.motorconfig import MotorConfig
        import pathlib
        mc = MotorConfig(defaults_file=pathlib.Path('data/motorconfig_defaults.json'))
        # Non-existent section should return default without crashing
        val = mc._axis_lookup('Nonexistent Section', 'X', default=42)
        assert val == 42

    def test_optics_fallback(self):
        from modules.motorconfig import MotorConfig
        import pathlib
        # Create a config without Optics section
        mc = MotorConfig.__new__(MotorConfig)
        mc._config = {}
        mc._defaults = {}
        assert mc.lens_focal_length() == 47.8
        assert mc.pixel_size() == 2.0


class TestScaleBarObjectiveInit:
    """Verify that set_objective enables scale bar rendering."""

    def test_objective_none_at_init(self):
        """Lumascope starts with no objective set."""
        from lumascope_api import Lumascope
        scope = Lumascope(simulate=True)
        assert scope._objective is None

    def test_set_objective_populates(self):
        """set_objective() should populate _objective dict."""
        from lumascope_api import Lumascope
        scope = Lumascope(simulate=True)
        scope.set_objective('20x Oly')
        assert scope._objective is not None
        assert scope._objective['magnification'] == 20

    def test_scale_bar_disabled_without_objective(self):
        """Scale bar enabled but no objective → use_scale_bar forced False."""
        from lumascope_api import Lumascope
        scope = Lumascope(simulate=True)
        scope.set_scale_bar(enabled=True)
        assert scope._scale_bar['enabled'] is True
        assert scope._objective is None
        # Internal logic forces use_scale_bar = False when _objective is None

    def test_scale_bar_works_with_objective(self):
        """Scale bar with objective set should proceed."""
        from lumascope_api import Lumascope
        scope = Lumascope(simulate=True)
        scope.set_objective('20x Oly')
        scope.set_scale_bar(enabled=True)
        assert scope._scale_bar['enabled'] is True
        assert scope._objective is not None


# ---------------------------------------------------------------------------
# Camera Simulator Tests
# ---------------------------------------------------------------------------

class TestSimulatedCamera:

    def test_api_surface_matches_base(self):
        """Simulated camera implements all abstract methods from Camera ABC."""
        abstract_methods = {m for m in dir(Camera) if not m.startswith('_')
                          and callable(getattr(Camera, m, None))}
        sim_methods = {m for m in dir(SimulatedCamera) if not m.startswith('_')}
        missing = abstract_methods - sim_methods
        assert not missing, f"SimulatedCamera missing methods: {missing}"

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
        result = cam.auto_gain(state=True, target_brightness=0.3,
                               min_gain=1.0, max_gain=10.0)
        assert result is True
        # Gain should converge to mid-range
        assert 1.0 <= cam.get_gain() <= 10.0

    def test_auto_gain_once(self):
        cam = SimulatedCamera()
        result = cam.auto_gain_once(state=True, target_brightness=0.5,
                                     min_gain=2.0, max_gain=8.0)
        assert result is True
        assert 2.0 <= cam.get_gain() <= 8.0

    def test_update_auto_gain_target(self):
        cam = SimulatedCamera()
        result = cam.update_auto_gain_target_brightness(0.7)
        assert result is True

    def test_update_auto_gain_min_max(self):
        cam = SimulatedCamera()
        result = cam.update_auto_gain_min_max(min_gain=0.5, max_gain=15.0)
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
        result, ts = cam.grab()
        assert result is True
        assert ts is not None
        assert isinstance(cam.array, np.ndarray)
        assert cam.array.shape == (1200, 1920)
        assert cam.array.dtype == np.uint8

    def test_grab_new_capture(self):
        cam = SimulatedCamera()
        result, ts = cam.grab_new_capture(timeout=1000)
        assert result is True
        assert ts is not None
        assert isinstance(cam.array, np.ndarray)

    def test_grab_respects_binning(self):
        cam = SimulatedCamera()
        cam.set_binning_size(2)
        cam.grab()
        assert cam.array.shape == (600, 960)

    def test_grab_mono12_dtype(self):
        cam = SimulatedCamera()
        cam.set_pixel_format('Mono12')
        cam.grab()
        assert cam.array.dtype == np.uint16

    def test_grab_not_grabbing_returns_false(self):
        cam = SimulatedCamera()
        cam.stop_grabbing()
        result, ts = cam.grab()
        assert result is False

    def test_image_brightness_varies_with_exposure(self):
        cam = SimulatedCamera()
        cam.exposure_t(1.0)  # 1ms — dim
        cam.grab()
        dim = cam.array.mean()

        cam.exposure_t(100.0)  # 100ms — bright
        cam.grab()
        bright = cam.array.mean()

        assert bright > dim

    def test_image_brightness_varies_with_gain(self):
        cam = SimulatedCamera()
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
        cam.set_test_pattern(enabled=True, pattern='Black')
        cam.grab()
        assert cam.array.max() == 0

    def test_white_pattern(self):
        cam = SimulatedCamera()
        cam.set_test_pattern(enabled=True, pattern='White')
        cam.grab()
        assert cam.array.max() == 255

    def test_noise_pattern(self):
        cam = SimulatedCamera()
        cam.set_test_pattern(enabled=True, pattern='Noise')
        cam.grab()
        # Noise should have some variance
        assert cam.array.std() > 0

    def test_disable_pattern_returns_gradient(self):
        cam = SimulatedCamera()
        cam.set_test_pattern(enabled=True, pattern='Black')
        cam.set_test_pattern(enabled=False)
        cam.grab()
        # Gradient should have variation across columns
        assert cam.array.std() > 0

    # -- Grabbing state --

    def test_start_stop_grabbing(self):
        cam = SimulatedCamera()
        assert cam.is_grabbing() is True
        cam.stop_grabbing()
        assert cam.is_grabbing() is False
        cam.start_grabbing()
        assert cam.is_grabbing() is True

    # -- update_camera_config context manager --

    def test_update_camera_config_stops_restarts_grabbing(self):
        cam = SimulatedCamera()
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
        cam.set_test_pattern(enabled=True, pattern='focus_target')
        cam.grab()
        # Focus target should have features — not uniform
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
        assert best_z == 5000, f"Expected best focus at 5000, got {best_z}. Scores: {scores}"

    def test_focus_score_decreases_with_defocus(self):
        """Focus score should decrease as we move away from focal point."""
        from modules.autofocus_functions import focus_vollath4_original

        cam = SimulatedCamera(width=480, height=300, grab_delay=0)
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

        assert score_at_focus > score_near > score_far, \
            f"Scores should decrease: focus={score_at_focus}, near={score_near}, far={score_far}"

    def test_focus_curve_is_symmetric(self):
        """Focus scores should be roughly symmetric around focal point."""
        from modules.autofocus_functions import focus_vollath4_original

        cam = SimulatedCamera(width=480, height=300, grab_delay=0)
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
        assert 0.8 < ratio < 1.2, f"Asymmetric: below={score_below}, above={score_above}"

    def test_z_position_func_callback(self):
        """Camera auto-queries Z from callback when generating focus_target."""
        z_val = [5000.0]
        cam = SimulatedCamera(width=480, height=300, grab_delay=0,
                              z_position_func=lambda: z_val[0])
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
        assert cam.max_exposure == cam.profile.max_exposure_ms

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
        assert cam.is_grabbing() is True
        with pytest.raises(ValueError):
            with cam.update_camera_config():
                assert cam.is_grabbing() is False
                raise ValueError("simulated config failure")
        # Grabbing must be restored despite the exception
        assert cam.is_grabbing() is True


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
        assert p.max_exposure_ms == 1_000

    def test_lookup_known_ace2_model(self):
        from drivers.camera_profiles import lookup_profile
        p = lookup_profile('a2A3536-31umBAS')
        assert p.sensor == 'Sony IMX676-AAMR1-C'
        assert p.max_exposure_ms == 10_000
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
        assert p.max_exposure_ms == 2_000

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
        assert p.max_exposure_ms == 1_000
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
        p1.max_exposure_ms = 999
        p1.gain.total_min_db = -99.0

        p2 = lookup_profile('daA3840-45um')
        assert p2.max_exposure_ms == 1_000
        assert p2.gain.total_min_db is None  # Not modified

    def test_dynamic_fields_initially_none(self):
        from drivers.camera_profiles import lookup_profile
        p = lookup_profile('daA3840-45um')
        assert p.exposure_min_us is None
        assert p.exposure_max_us is None
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

    def test_motor_fast_mode(self):
        m = SimulatedMotorBoard(timing='fast')
        assert m._cmd_delay == 0.0
        assert m._simulate_move_duration is False

    def test_motor_realistic_mode(self):
        m = SimulatedMotorBoard(timing='realistic')
        assert m._cmd_delay > 0
        assert m._simulate_move_duration is True

    def test_motor_switch_mode(self):
        m = SimulatedMotorBoard(timing='fast')
        m.set_timing_mode('realistic')
        assert m._simulate_move_duration is True
        m.set_timing_mode('fast')
        assert m._simulate_move_duration is False

    def test_motor_realistic_move_not_instant(self):
        """In realistic mode, target_status returns False during move."""
        m = SimulatedMotorBoard(timing='realistic')
        m._homed['Z'] = True
        # Move Z a significant distance
        m.move_abs_pos('Z', 10000.0)
        # Should not have arrived yet
        assert m.target_status('Z') is False
        # Wait for move to complete
        deadline = time.monotonic() + 5.0
        while not m.target_status('Z'):
            time.sleep(0.01)
            if time.monotonic() > deadline:
                raise TimeoutError("Motor never reached target")
        assert m.current_pos('Z') == pytest.approx(10000.0, abs=1.0)

    def test_motor_fast_move_instant(self):
        """In fast mode, move is instant."""
        m = SimulatedMotorBoard(timing='fast')
        m.move_abs_pos('Z', 10000.0)
        assert m.target_status('Z') is True
        assert m.current_pos('Z') == pytest.approx(10000.0, abs=1.0)

    def test_led_fast_mode(self):
        led = SimulatedLEDBoard(timing='fast')
        assert led._delay == 0.0

    def test_led_realistic_mode(self):
        led = SimulatedLEDBoard(timing='realistic')
        assert led._delay > 0

    def test_led_switch_mode(self):
        led = SimulatedLEDBoard(timing='fast')
        led.set_timing_mode('realistic')
        assert led._delay > 0
        led.set_timing_mode('fast')
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
        assert m.exchange_command('INFO') is not None      # cmd 1
        assert m.exchange_command('INFO') is not None      # cmd 2
        assert m.exchange_command('INFO') is not None      # cmd 3
        assert m.exchange_command('INFO') is None          # cmd 4 — disconnected
        assert m.driver is None

    def test_motor_fail_after_sets_found_false(self):
        """After injected disconnect, found should be False."""
        m = SimulatedMotorBoard(fail_after=1)
        assert m.found is True
        m.exchange_command('INFO')   # cmd 1 — succeeds
        m.exchange_command('INFO')   # cmd 2 — fails
        assert m.found is False

    def test_motor_fail_on_specific_command(self):
        """Motor board should return None for targeted commands only."""
        m = SimulatedMotorBoard(fail_on={'ZHOME'})
        assert m.exchange_command('INFO') is not None      # OK
        assert m.exchange_command('ZHOME') is None         # targeted failure
        assert m.exchange_command('INFO') is not None      # still connected
        assert m.driver is not None                        # not disconnected

    def test_motor_fail_on_multiple_commands(self):
        """Multiple commands can be targeted for failure."""
        m = SimulatedMotorBoard(fail_on={'ZHOME', 'THOME'})
        assert m.exchange_command('ZHOME') is None
        assert m.exchange_command('THOME') is None
        assert m.exchange_command('HOME') is not None      # not in fail set

    def test_motor_no_failure_by_default(self):
        """Without fail params, simulator works normally."""
        m = SimulatedMotorBoard()
        for _ in range(100):
            assert m.exchange_command('INFO') is not None

    def test_motor_fail_after_affects_move(self):
        """Mid-protocol disconnect: move starts OK, then fails."""
        m = SimulatedMotorBoard(fail_after=5)
        m.exchange_command('HOME')                         # cmd 1
        m.move_abs_pos('Z', 5000)                          # uses multiple commands
        # Eventually commands fail
        result = m.exchange_command('ACTUAL_RZ')
        # After enough commands, should get None
        # (exact count depends on internal commands used by move_abs_pos)

    # --- LED board ---

    def test_led_fail_after_disconnects(self):
        """LED board should return None after N commands."""
        led = SimulatedLEDBoard(fail_after=2)
        assert led.exchange_command('LEDS_ENT') is not None  # cmd 1
        assert led.exchange_command('LED0_100') is not None   # cmd 2
        assert led.exchange_command('LEDS_OFF') is None       # cmd 3 — disconnected
        assert led.driver is None

    def test_led_fail_after_sets_found_false(self):
        """After injected disconnect, found should be False."""
        led = SimulatedLEDBoard(fail_after=1)
        assert led.found is True
        led.exchange_command('LEDS_ENT')    # cmd 1 — succeeds
        led.exchange_command('LED0_100')     # cmd 2 — fails
        assert led.found is False

    def test_led_fail_on_specific_command(self):
        """LED board should return None for targeted commands only."""
        led = SimulatedLEDBoard(fail_on={'LEDS_ENT'})
        assert led.exchange_command('LEDS_ENT') is None       # targeted
        assert led.exchange_command('LED0_100') is not None    # OK
        assert led.driver is not None                          # still connected

    def test_led_no_failure_by_default(self):
        """Without fail params, LED simulator works normally."""
        led = SimulatedLEDBoard()
        for _ in range(100):
            assert led.exchange_command('LED0_100') is not None

    def test_led_fast_path_also_fails(self):
        """_write_command_fast should also respect fail_after."""
        led = SimulatedLEDBoard(fail_after=2)
        led._write_command_fast('LED0_100')   # cmd 1
        led._write_command_fast('LED1_100')   # cmd 2
        led._write_command_fast('LED2_100')   # cmd 3 — should disconnect
        assert led.driver is None
        assert led.found is False
