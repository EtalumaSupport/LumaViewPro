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

from simulated_ledboard import SimulatedLEDBoard
from simulated_motorboard import SimulatedMotorBoard
from camera.simulated_camera import SimulatedCamera
from camera.camera import Camera
from ledboard import LEDBoard
from motorboard import MotorBoard


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
        assert board.has_turret() is True  # default model LS720T

    def test_no_turret_model(self):
        board = SimulatedMotorBoard(model='LS720')
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
        board = SimulatedMotorBoard(model='LS720T', serial_number='TEST-123')
        info = board.fullinfo()
        assert info['model'] == 'LS720T'
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
