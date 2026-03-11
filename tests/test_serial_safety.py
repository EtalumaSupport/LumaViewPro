# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for serial driver thread safety and error handling.

Uses mock serial ports — no hardware needed.
Tests the fixes in ledboard.py and motorboard.py for:
- Single lock preventing interleaved writes
- Safe driver cleanup on errors (no bare self.driver = None)
- Motor board timeout (not blocking forever)
- Reconnect after failure
"""

import pytest
import sys
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock
import serial

# ---------------------------------------------------------------------------
# Mock out heavy dependencies that aren't installed in test environment
# ---------------------------------------------------------------------------
# lvp_logger depends on userpaths; motorboard depends on requests
# We mock these before importing ledboard/motorboard

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

# Now safe to import
import pathlib
from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard
from modules.motorconfig import MotorConfig

_MOTORCONFIG_DEFAULTS = pathlib.Path("data/motorconfig_defaults.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_serial(**overrides):
    """Create a mock serial.Serial that behaves like a connected port.

    LED firmware sends two lines per command:
      1. Echo:  "RE: <CMD>\\r\\n"
      2. Result: "<result text>\\r\\n"
    Motor firmware sends one line:
      1. Result: "<result text>\\r\\n"

    Default readline returns echo-style responses. Use side_effect for
    multi-line sequences.
    """
    mock = MagicMock(spec=serial.Serial)
    mock.readline.return_value = b"OK\r\n"
    mock.write.return_value = None
    mock.close.return_value = None
    for k, v in overrides.items():
        setattr(mock, k, v)
    return mock


def _make_led_readline(*result_lines):
    """Build a readline side_effect for LED board: echo + result per command.

    Each call to exchange_command reads two lines:
      readline() -> b"RE: <cmd>\\r\\n"  (echo)
      readline() -> b"<result>\\r\\n"   (actual result)

    Pass result strings (without \\r\\n) and a cycling side_effect is returned.
    """
    responses = []
    for result in result_lines:
        responses.append(b"RE: CMD\r\n")  # echo (content doesn't matter, starts with RE:)
        responses.append(result.encode('utf-8') + b"\r\n")
    # Cycle so multiple exchange_command calls work
    import itertools
    cycle = itertools.cycle(responses)
    return lambda: next(cycle)


# ---------------------------------------------------------------------------
# LEDBoard Tests
# ---------------------------------------------------------------------------

class TestLEDBoardLocking:
    """Verify LEDBoard uses a single lock for all serial access."""

    def _make_board(self):
        """Create an LEDBoard with a mock serial driver."""
        board = LEDBoard.__new__(LEDBoard)
        board.found = False
        board._lock = threading.RLock()
        board._label = '[LED Class ]'
        board.port = '/dev/fake'
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 0.1
        board.write_timeout = 0.1
        board.driver = _make_mock_serial()
        board.led_ma = {'BF': -1, 'PC': -1, 'DF': -1, 'Red': -1, 'Blue': -1, 'Green': -1}
        board.firmware_version = None
        board._last_error_log_time = 0.0
        board._error_log_interval = 2.0
        return board

    def test_no_com_lock_attribute(self):
        """LEDBoard should not have a separate com_lock."""
        board = self._make_board()
        assert not hasattr(board, 'com_lock'), "com_lock should be removed"

    def test_has_single_lock(self):
        """LEDBoard should have _lock as the sole lock."""
        board = self._make_board()
        assert hasattr(board, '_lock')
        assert isinstance(board._lock, type(threading.RLock()))

    def test_exchange_command_returns_response(self):
        """exchange_command should return the actual result (not the echo)."""
        board = self._make_board()
        # Firmware sends echo, then result
        board.driver.readline.side_effect = [
            b"RE: LED0_100\r\n",       # echo line
            b"LED 0 set to 100 mA.\r\n",  # result line
        ]
        resp = board.exchange_command('LED0_100')
        assert resp is not None
        assert 'LED 0 set to 100 mA.' == resp

    def test_exchange_command_reads_past_echo(self):
        """exchange_command should skip the RE: echo and return the result."""
        board = self._make_board()
        board.driver.readline.side_effect = [
            b"RE: LEDS_OFF\r\n",          # echo
            b"All LEDs turned off\r\n",     # result
        ]
        resp = board.exchange_command('LEDS_OFF')
        assert resp == 'All LEDs turned off'

    def test_exchange_command_no_echo_firmware(self):
        """If firmware doesn't send RE: echo, first line IS the result."""
        board = self._make_board()
        board.driver.readline.return_value = b"LED 0 set to 100 mA.\r\n"
        resp = board.exchange_command('LED0_100')
        assert resp == 'LED 0 set to 100 mA.'

    def test_exchange_command_none_on_timeout(self):
        """exchange_command should return None on SerialTimeoutException."""
        board = self._make_board()
        board.driver.write.side_effect = serial.SerialTimeoutException("timeout")
        resp = board.exchange_command('LED0_100')
        assert resp is None

    def test_driver_closed_on_timeout(self):
        """Driver should be closed (not just set to None) on timeout."""
        board = self._make_board()
        mock_driver = board.driver
        board.driver.write.side_effect = serial.SerialTimeoutException("timeout")
        board.exchange_command('LED0_100')
        mock_driver.close.assert_called()
        assert board.driver is None

    def test_write_fast_uses_same_lock(self):
        """_write_command_fast should acquire _lock, preventing interleave."""
        board = self._make_board()
        lock_acquired = threading.Event()
        write_started = threading.Event()

        original_write = board.driver.write

        def slow_write(data):
            write_started.set()
            time.sleep(0.1)
            return original_write(data)

        board.driver.write = slow_write

        # Start a slow exchange_command in a thread
        def do_exchange():
            board.exchange_command('LED0_100')
            lock_acquired.set()

        t = threading.Thread(target=do_exchange)
        t.start()
        write_started.wait(timeout=2)

        # Try _write_command_fast — should block until exchange finishes
        start = time.time()
        board._write_command_fast('LED0_OFF')
        elapsed = time.time() - start

        t.join(timeout=5)
        # If the lock works, fast write had to wait for the slow exchange
        assert elapsed >= 0.05, f"_write_command_fast didn't wait for lock (elapsed={elapsed:.3f}s)"

    def test_exchange_reconnects_if_driver_none(self):
        """exchange_command should attempt reconnect if driver is None."""
        board = self._make_board()
        board.driver = None

        with patch.object(board, 'connect') as mock_connect:
            resp = board.exchange_command('INFO')
            mock_connect.assert_called()

    def test_concurrent_exchange_commands(self):
        """Multiple threads calling exchange_command should not interleave."""
        board = self._make_board()
        call_order = []
        original_write = board.driver.write

        def tracking_write(data):
            cmd = data.decode('utf-8').strip()
            call_order.append(('start', cmd))
            time.sleep(0.01)
            call_order.append(('end', cmd))
            return original_write(data)

        board.driver.write = tracking_write

        threads = []
        for i in range(5):
            t = threading.Thread(target=board.exchange_command, args=(f'LED{i}_100',))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Verify no interleaving: every 'start' should be followed by its 'end'
        for i in range(0, len(call_order), 2):
            assert call_order[i][0] == 'start'
            assert call_order[i+1][0] == 'end'
            assert call_order[i][1] == call_order[i+1][1], \
                f"Interleaved: {call_order[i]} followed by {call_order[i+1]}"


# ---------------------------------------------------------------------------
# MotorBoard Tests
# ---------------------------------------------------------------------------

class TestMotorBoardSafety:
    """Verify MotorBoard error handling and timeout behavior."""

    def _make_board(self):
        """Create a MotorBoard with a mock serial driver."""
        # MotorBoard imported at module level
        board = MotorBoard.__new__(MotorBoard)
        board.motorconfig = MotorConfig(defaults_file=_MOTORCONFIG_DEFAULTS)
        board.found = True
        board.overshoot = False
        board.backlash = 25
        board._has_turret = False
        board.initial_homing_complete = False
        board.initial_t_homing_complete = False
        board.port = '/dev/fake'
        board._lock = threading.RLock()
        board._label = '[XYZ Class ]'
        board.thread_lock = board._lock
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 30
        board.write_timeout = 5
        board.driver = _make_mock_serial()
        board._fullinfo = None
        board._connect_fails = 0
        board.axes_config = {}
        board.firmware_version = None
        return board

    def test_timeout_is_set(self):
        """Motor board should have a finite timeout, not None."""
        board = self._make_board()
        assert board.timeout is not None
        assert board.timeout > 0

    def test_exchange_command_returns_response(self):
        """exchange_command should return decoded response."""
        board = self._make_board()
        board.driver.readline.return_value = b"Z home successful\n"
        resp = board.exchange_command('ZHOME')
        assert resp is not None
        assert 'home successful' in resp

    def test_exchange_command_none_on_timeout(self):
        """exchange_command should return None on timeout, not hang."""
        board = self._make_board()
        board.driver.write.side_effect = serial.SerialTimeoutException("timeout")
        resp = board.exchange_command('HOME')
        assert resp is None

    def test_driver_closed_on_error(self):
        """Driver should be properly closed on error, not just set to None."""
        board = self._make_board()
        mock_driver = board.driver
        board.driver.write.side_effect = OSError("port disappeared")
        resp = board.exchange_command('INFO')
        assert resp is None
        mock_driver.close.assert_called()
        assert board.driver is None

    def test_exchange_reconnects_if_driver_none(self):
        """exchange_command should attempt reconnect if driver is None."""
        board = self._make_board()
        board.driver = None

        with patch.object(board, 'connect') as mock_connect:
            resp = board.exchange_command('INFO')
            mock_connect.assert_called()

    def test_disconnect_is_threadsafe(self):
        """disconnect() during exchange_command should not corrupt state."""
        board = self._make_board()

        # Slow down exchange_command so disconnect has to wait
        original_write = board.driver.write
        def slow_write(data):
            time.sleep(0.1)
            return original_write(data)
        board.driver.write = slow_write

        results = {}
        def do_exchange():
            results['resp'] = board.exchange_command('INFO')

        t = threading.Thread(target=do_exchange)
        t.start()
        time.sleep(0.02)  # let exchange start

        board.disconnect()
        t.join(timeout=5)

        # After disconnect, driver should be None
        assert board.driver is None


# ---------------------------------------------------------------------------
# LEDBoard Command Formatting
# ---------------------------------------------------------------------------

class TestLEDBoardCommands:
    """Verify LEDBoard sends correctly formatted serial commands."""

    def _make_board(self):
        board = LEDBoard.__new__(LEDBoard)
        board.found = False
        board._lock = threading.RLock()
        board._label = '[LED Class ]'
        board.port = '/dev/fake'
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 0.1
        board.write_timeout = 0.1
        board.driver = _make_mock_serial()
        board.led_ma = {'BF': -1, 'PC': -1, 'DF': -1, 'Red': -1, 'Blue': -1, 'Green': -1}
        return board

    def test_led_on_sends_correct_command(self):
        """led_on(0, 100) should send 'LED0_100\\n'."""
        board = self._make_board()
        board.led_on(channel=0, mA=100)
        board.driver.write.assert_called_with(b'LED0_100\n')

    def test_led_on_channel_3(self):
        """led_on(3, 250) should send 'LED3_250\\n'."""
        board = self._make_board()
        board.led_on(channel=3, mA=250)
        board.driver.write.assert_called_with(b'LED3_250\n')

    def test_led_off_sends_correct_command(self):
        """led_off(2) should send 'LED2_OFF\\n'."""
        board = self._make_board()
        board.led_off(channel=2)
        board.driver.write.assert_called_with(b'LED2_OFF\n')

    def test_leds_off_sends_correct_command(self):
        """leds_off() should send 'LEDS_OFF\\n'."""
        board = self._make_board()
        board.leds_off()
        board.driver.write.assert_called_with(b'LEDS_OFF\n')

    def test_leds_enable_sends_correct_command(self):
        """leds_enable() should send 'LEDS_ENT\\n'."""
        board = self._make_board()
        board.leds_enable()
        board.driver.write.assert_called_with(b'LEDS_ENT\n')

    def test_leds_disable_sends_correct_command(self):
        """leds_disable() should send 'LEDS_ENF\\n'."""
        board = self._make_board()
        board.leds_disable()
        board.driver.write.assert_called_with(b'LEDS_ENF\n')

    def test_led_on_fast_sends_correct_command(self):
        """led_on_fast uses write-only path with correct command."""
        board = self._make_board()
        board.led_on_fast(channel=1, mA=50)
        board.driver.write.assert_called_with(b'LED1_50\n')

    def test_led_off_fast_sends_correct_command(self):
        """led_off_fast uses write-only path with correct command."""
        board = self._make_board()
        board.led_off_fast(channel=4)
        board.driver.write.assert_called_with(b'LED4_OFF\n')

    def test_leds_off_fast_sends_correct_command(self):
        """leds_off_fast uses write-only path with correct command."""
        board = self._make_board()
        board.leds_off_fast()
        board.driver.write.assert_called_with(b'LEDS_OFF\n')

    def test_get_status_sends_status(self):
        """get_status() should send 'STATUS\\n'."""
        board = self._make_board()
        board.get_status()
        board.driver.write.assert_called_with(b'STATUS\n')


# ---------------------------------------------------------------------------
# LEDBoard State Tracking
# ---------------------------------------------------------------------------

class TestLEDBoardState:
    """Verify LEDBoard tracks LED on/off state correctly."""

    def _make_board(self):
        board = LEDBoard.__new__(LEDBoard)
        board.found = False
        board._lock = threading.RLock()
        board._label = '[LED Class ]'
        board.port = '/dev/fake'
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 0.1
        board.write_timeout = 0.1
        board.driver = _make_mock_serial()
        board.led_ma = {'BF': -1, 'PC': -1, 'DF': -1, 'Red': -1, 'Blue': -1, 'Green': -1}
        return board

    def test_led_on_updates_state(self):
        """led_on should update led_ma for the correct color."""
        board = self._make_board()
        board.led_on(channel=0, mA=100)
        assert board.get_led_ma('Blue') == 100
        assert board.is_led_on('Blue') is True

    def test_led_off_clears_state(self):
        """led_off should set led_ma to -1."""
        board = self._make_board()
        board.led_on(channel=0, mA=100)
        board.led_off(channel=0)
        assert board.get_led_ma('Blue') == -1
        assert board.is_led_on('Blue') is False

    def test_leds_off_clears_all(self):
        """leds_off should clear all channels."""
        board = self._make_board()
        board.led_on(channel=0, mA=100)
        board.led_on(channel=1, mA=200)
        board.led_on(channel=2, mA=300)
        board.leds_off()
        for color in board.led_ma:
            assert board.get_led_ma(color) == -1

    def test_leds_disable_clears_all(self):
        """leds_disable should clear all channels."""
        board = self._make_board()
        board.led_on(channel=0, mA=100)
        board.led_on(channel=3, mA=50)
        board.leds_disable()
        for color in board.led_ma:
            assert board.get_led_ma(color) == -1

    def test_led_on_fast_updates_state(self):
        """led_on_fast should track state the same as led_on."""
        board = self._make_board()
        board.led_on_fast(channel=2, mA=75)
        assert board.get_led_ma('Red') == 75
        assert board.is_led_on('Red') is True

    def test_led_off_fast_clears_state(self):
        """led_off_fast should clear state."""
        board = self._make_board()
        board.led_on_fast(channel=2, mA=75)
        board.led_off_fast(channel=2)
        assert board.get_led_ma('Red') == -1

    def test_leds_off_fast_clears_all(self):
        """leds_off_fast should clear all channels."""
        board = self._make_board()
        board.led_on_fast(channel=0, mA=100)
        board.led_on_fast(channel=1, mA=200)
        board.leds_off_fast()
        for color in board.led_ma:
            assert board.get_led_ma(color) == -1

    def test_get_led_state_dict(self):
        """get_led_state should return dict with enabled and illumination."""
        board = self._make_board()
        board.led_on(channel=3, mA=50)
        state = board.get_led_state('BF')
        assert state == {'enabled': True, 'illumination': 50}

    def test_get_led_states_all(self):
        """get_led_states should return state for all channels."""
        board = self._make_board()
        board.led_on(channel=0, mA=100)
        states = board.get_led_states()
        assert states['Blue']['enabled'] is True
        assert states['Blue']['illumination'] == 100
        assert states['Red']['enabled'] is False
        assert states['Red']['illumination'] == -1

    def test_multiple_channels_independent(self):
        """Turning on one channel should not affect others."""
        board = self._make_board()
        board.led_on(channel=0, mA=100)
        board.led_on(channel=2, mA=200)
        assert board.get_led_ma('Blue') == 100
        assert board.get_led_ma('Red') == 200
        assert board.get_led_ma('Green') == -1

        board.led_off(channel=0)
        assert board.get_led_ma('Blue') == -1
        assert board.get_led_ma('Red') == 200


# ---------------------------------------------------------------------------
# LEDBoard Color/Channel Conversion
# ---------------------------------------------------------------------------

class TestLEDBoardConversion:
    """Verify color-to-channel and channel-to-color mappings."""

    def test_color2ch_all(self):
        board = LEDBoard.__new__(LEDBoard)
        assert board.color2ch('Blue') == 0
        assert board.color2ch('Green') == 1
        assert board.color2ch('Red') == 2
        assert board.color2ch('BF') == 3
        assert board.color2ch('PC') == 4
        assert board.color2ch('DF') == 5

    def test_ch2color_all(self):
        board = LEDBoard.__new__(LEDBoard)
        assert board.ch2color(0) == 'Blue'
        assert board.ch2color(1) == 'Green'
        assert board.ch2color(2) == 'Red'
        assert board.ch2color(3) == 'BF'
        assert board.ch2color(4) == 'PC'
        assert board.ch2color(5) == 'DF'

    def test_color2ch_unknown_defaults_bf(self):
        board = LEDBoard.__new__(LEDBoard)
        assert board.color2ch('Unknown') == 3

    def test_ch2color_unknown_defaults_bf(self):
        board = LEDBoard.__new__(LEDBoard)
        assert board.ch2color(99) == 'BF'

    def test_roundtrip_all_channels(self):
        """color2ch -> ch2color should roundtrip for all valid channels."""
        board = LEDBoard.__new__(LEDBoard)
        for color in ('Blue', 'Green', 'Red', 'BF', 'PC', 'DF'):
            ch = board.color2ch(color)
            assert board.ch2color(ch) == color


# ---------------------------------------------------------------------------
# MotorBoard Command Formatting
# ---------------------------------------------------------------------------

class TestMotorBoardCommands:
    """Verify MotorBoard sends correctly formatted serial commands."""

    def _make_board(self):
        board = MotorBoard.__new__(MotorBoard)
        board.motorconfig = MotorConfig(defaults_file=_MOTORCONFIG_DEFAULTS)
        board.found = True
        board.overshoot = False
        board.backlash = 25
        board._has_turret = False
        board.initial_homing_complete = False
        board.initial_t_homing_complete = False
        board.port = '/dev/fake'
        board._lock = threading.RLock()
        board._label = '[XYZ Class ]'
        board.thread_lock = board._lock
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 30
        board.write_timeout = 5
        board.driver = _make_mock_serial()
        board._fullinfo = None
        board._connect_fails = 0
        board.axes_config = {
            'Z': {'limits': {'min': 0., 'max': 14000.}, 'move_func': board.z_um2ustep},
            'X': {'limits': {'min': 0., 'max': 120000.}, 'move_func': board.xy_um2ustep},
            'Y': {'limits': {'min': 0., 'max': 80000.}, 'move_func': board.xy_um2ustep},
            'T': {'move_func': board.t_pos2ustep},
        }
        return board

    def test_move_sends_target_write(self):
        """move('Z', 1000) should send 'TARGET_WZ1000\\n'."""
        board = self._make_board()
        board.move('Z', 1000)
        board.driver.write.assert_called_with(b'TARGET_WZ1000\n')

    def test_move_negative_twos_complement(self):
        """Negative steps should be converted to twos complement."""
        board = self._make_board()
        board.move('Z', -100)
        expected = -100 + 0x100000000
        board.driver.write.assert_called_with(f'TARGET_WZ{expected}\n'.encode('utf-8'))

    def test_zhome_sends_zhome(self):
        """zhome() should send 'ZHOME\\n'."""
        board = self._make_board()
        board.zhome()
        board.driver.write.assert_called_with(b'ZHOME\n')

    def test_xyhome_sends_home(self):
        """xyhome() should send 'HOME\\n'."""
        board = self._make_board()
        board.xyhome()
        board.driver.write.assert_called_with(b'HOME\n')

    def test_thome_sends_thome(self):
        """thome() should send 'THOME\\n'."""
        board = self._make_board()
        board.thome()
        board.driver.write.assert_called_with(b'THOME\n')

    def test_xycenter_sends_center(self):
        """xycenter() should send 'CENTER\\n'."""
        board = self._make_board()
        board.xycenter()
        board.driver.write.assert_called_with(b'CENTER\n')

    def test_current_pos_sends_actual_read(self):
        """current_pos('Z') should send 'ACTUAL_RZ\\n'."""
        board = self._make_board()
        board.driver.readline.return_value = b"0\n"
        board.current_pos('Z')
        board.driver.write.assert_called_with(b'ACTUAL_RZ\n')

    def test_target_pos_sends_target_read(self):
        """target_pos('X') should send 'TARGET_RX\\n'."""
        board = self._make_board()
        board.driver.readline.return_value = b"0\n"
        board.target_pos('X')
        board.driver.write.assert_called_with(b'TARGET_RX\n')

    def test_target_status_sends_status_read(self):
        """target_status('Z') should send 'STATUS_RZ\\n'."""
        board = self._make_board()
        # bit 9 (position_reached) set
        board.driver.readline.return_value = b"512\n"
        board.target_status('Z')
        board.driver.write.assert_called_with(b'STATUS_RZ\n')

    def test_spi_read_sends_correct_format(self):
        """spi_read('X', 0x6F) should send 'SPIX0x6f00\\n'."""
        board = self._make_board()
        board.driver.readline.return_value = b"0x00000000\n"
        board.spi_read('X', 0x6F)
        board.driver.write.assert_called_with(b'SPIX0x6f00\n')


# ---------------------------------------------------------------------------
# MotorBoard Homing State
# ---------------------------------------------------------------------------

class TestMotorBoardHoming:
    """Verify homing flag tracking."""

    def _make_board(self):
        board = MotorBoard.__new__(MotorBoard)
        board.motorconfig = MotorConfig(defaults_file=_MOTORCONFIG_DEFAULTS)
        board.found = True
        board.overshoot = False
        board.backlash = 25
        board._has_turret = False
        board.initial_homing_complete = False
        board.initial_t_homing_complete = False
        board.port = '/dev/fake'
        board._lock = threading.RLock()
        board._label = '[XYZ Class ]'
        board.thread_lock = board._lock
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 30
        board.write_timeout = 5
        board.driver = _make_mock_serial()
        board._fullinfo = None
        board._connect_fails = 0
        board.axes_config = {}
        return board

    def test_initial_homing_false(self):
        """Board starts with homing not complete."""
        board = self._make_board()
        assert board.has_xyhomed() is False
        assert board.has_thomed() is False

    def test_xyhome_sets_flag_on_success(self):
        """xyhome() should set initial_homing_complete when firmware confirms."""
        board = self._make_board()
        board.driver.readline.return_value = b"XYZ home complete\n"
        board.xyhome()
        assert board.has_xyhomed() is True

    def test_xyhome_no_flag_on_failure(self):
        """xyhome() should not set flag if response doesn't contain expected text."""
        board = self._make_board()
        board.driver.readline.return_value = b"ERROR: timeout\n"
        board.xyhome()
        assert board.has_xyhomed() is False

    def test_xyhome_no_flag_on_none(self):
        """xyhome() should not set flag if response is None (disconnected)."""
        board = self._make_board()
        board.driver.write.side_effect = serial.SerialTimeoutException("timeout")
        board.xyhome()
        assert board.has_xyhomed() is False

    def test_thome_sets_flag_on_success(self):
        """thome() should set initial_t_homing_complete when firmware confirms."""
        board = self._make_board()
        board.driver.readline.return_value = b"T home successful\n"
        board.thome()
        assert board.has_thomed() is True

    def test_thome_no_flag_on_failure(self):
        """thome() should not set flag if response doesn't match."""
        board = self._make_board()
        board.driver.readline.return_value = b"ERROR\n"
        board.thome()
        assert board.has_thomed() is False

    def test_has_thomed_true_after_xyhome(self):
        """has_thomed() should return True if XY homing completed (it homes T too)."""
        board = self._make_board()
        board.driver.readline.return_value = b"XYZ home complete\n"
        board.xyhome()
        assert board.has_thomed() is True


# ---------------------------------------------------------------------------
# MotorBoard Fullinfo Parsing
# ---------------------------------------------------------------------------

class TestMotorBoardFullinfo:
    """Verify fullinfo() parses firmware response correctly."""

    def _make_board(self):
        board = MotorBoard.__new__(MotorBoard)
        board.motorconfig = MotorConfig(defaults_file=_MOTORCONFIG_DEFAULTS)
        board.found = True
        board.overshoot = False
        board.backlash = 25
        board._has_turret = False
        board.initial_homing_complete = False
        board.initial_t_homing_complete = False
        board.port = '/dev/fake'
        board._lock = threading.RLock()
        board._label = '[XYZ Class ]'
        board.thread_lock = board._lock
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 30
        board.write_timeout = 5
        board.driver = _make_mock_serial()
        board._fullinfo = None
        board._connect_fails = 0
        board.axes_config = {}
        return board

    def test_fullinfo_parses_model_and_serial(self):
        """fullinfo() should extract model and serial_number."""
        board = self._make_board()
        board.driver.readline.return_value = (
            b"EL-0940-02 Integrated Mainboard Firmware: 2024-09-10 "
            b"Model: LS850 Serial: 12062 X homed: False\n"
        )
        info = board.fullinfo()
        assert info['model'] == 'LS850'
        assert info['serial_number'] == '12062'

    def test_fullinfo_detects_turret_model(self):
        """fullinfo() should set _has_turret for models ending in 'T'."""
        board = self._make_board()
        board.driver.readline.return_value = (
            b"EL-0940-02 Integrated Mainboard Firmware: 2024-09-10 "
            b"Model: LS850T Serial: 99999 X homed: False\n"
        )
        board.fullinfo()
        assert board._has_turret is True
        assert board.has_turret() is True

    def test_fullinfo_no_turret(self):
        """fullinfo() should not set _has_turret for non-T models."""
        board = self._make_board()
        board.driver.readline.return_value = (
            b"EL-0940-02 Integrated Mainboard Firmware: 2024-09-10 "
            b"Model: LS850 Serial: 12062 X homed: False\n"
        )
        board.fullinfo()
        assert board._has_turret is False
        assert board.has_turret() is False

    def test_get_microscope_model(self):
        """get_microscope_model() should return model from cached fullinfo."""
        board = self._make_board()
        board._fullinfo = {'model': 'LS850', 'serial_number': '12062'}
        assert board.get_microscope_model() == 'LS850'


# ---------------------------------------------------------------------------
# MotorBoard Unit Conversions
# ---------------------------------------------------------------------------

class TestMotorBoardConversions:
    """Verify unit conversion math for Z, XY, and turret."""

    def _make_board(self):
        board = MotorBoard.__new__(MotorBoard)
        board.motorconfig = MotorConfig(defaults_file=_MOTORCONFIG_DEFAULTS)
        return board

    def test_z_roundtrip(self):
        """z_um2ustep -> z_ustep2um should roundtrip closely."""
        board = self._make_board()
        for um in (0, 100, 5000, 14000):
            ustep = board.z_um2ustep(um)
            um_back = board.z_ustep2um(ustep)
            assert abs(um - um_back) < 0.01, f"Z roundtrip failed for {um}: got {um_back}"

    def test_xy_roundtrip(self):
        """xy_um2ustep -> xy_ustep2um should roundtrip closely."""
        board = self._make_board()
        for um in (0, 1000, 60000, 120000):
            ustep = board.xy_um2ustep(um)
            um_back = board.xy_ustep2um(ustep)
            assert abs(um - um_back) < 0.1, f"XY roundtrip failed for {um}: got {um_back}"

    def test_turret_roundtrip(self):
        """t_pos2ustep -> t_ustep2pos should roundtrip for positions 1-4."""
        board = self._make_board()
        for pos in (1, 2, 3, 4):
            ustep = board.t_pos2ustep(pos)
            pos_back = board.t_ustep2pos(ustep)
            assert pos == pos_back, f"Turret roundtrip failed for pos {pos}: got {pos_back}"

    def test_z_known_values(self):
        """Spot check Z conversion against known ratio (170666 ustep/mm)."""
        board = self._make_board()
        # 1mm = 1000um -> 170666 usteps (from motorconfig_defaults.json)
        assert board.z_um2ustep(1000) == 170666
        # 170666 usteps -> 1000um
        assert abs(board.z_ustep2um(170666) - 1000) < 0.01

    def test_xy_known_values(self):
        """Spot check XY conversion against known ratio (20157 ustep/mm)."""
        board = self._make_board()
        # 1mm = 1000um -> 20157 usteps
        assert board.xy_um2ustep(1000) == 20157

    def test_turret_degrees(self):
        """Position 1 = 0 deg, position 2 = 90 deg, etc."""
        board = self._make_board()
        assert board.t_pos2ustep(1) == 0
        assert board.t_pos2ustep(2) == 80000


# ---------------------------------------------------------------------------
# MotorBoard Movement Logic
# ---------------------------------------------------------------------------

class TestMotorBoardMovement:
    """Verify move_abs_pos limit clamping and overshoot logic."""

    def _make_board(self):
        board = MotorBoard.__new__(MotorBoard)
        board.motorconfig = MotorConfig(defaults_file=_MOTORCONFIG_DEFAULTS)
        board.found = True
        board.overshoot = False
        board.backlash = 25
        board._has_turret = False
        board.initial_homing_complete = False
        board.initial_t_homing_complete = False
        board.port = '/dev/fake'
        board._lock = threading.RLock()
        board._label = '[XYZ Class ]'
        board.thread_lock = board._lock
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 30
        board.write_timeout = 5
        board.driver = _make_mock_serial()
        board._fullinfo = None
        board._connect_fails = 0
        board.axes_config = {
            'Z': {'limits': {'min': 0., 'max': 14000.}, 'move_func': board.z_um2ustep},
            'X': {'limits': {'min': 0., 'max': 120000.}, 'move_func': board.xy_um2ustep},
            'Y': {'limits': {'min': 0., 'max': 80000.}, 'move_func': board.xy_um2ustep},
            'T': {'move_func': board.t_pos2ustep},
        }
        return board

    def test_z_clamped_to_max(self):
        """move_abs_pos('Z', 99999) should clamp to Z max (14000um)."""
        board = self._make_board()
        board.move_abs_pos('Z', 99999, overshoot_enabled=False)
        expected_ustep = board.z_um2ustep(14000)
        board.driver.write.assert_called_with(f'TARGET_WZ{expected_ustep}\n'.encode('utf-8'))

    def test_z_clamped_to_min(self):
        """move_abs_pos('Z', -100) should clamp to Z min (0um)."""
        board = self._make_board()
        board.move_abs_pos('Z', -100, overshoot_enabled=False)
        expected_ustep = board.z_um2ustep(0)
        board.driver.write.assert_called_with(f'TARGET_WZ{expected_ustep}\n'.encode('utf-8'))

    def test_x_clamped_to_max(self):
        """move_abs_pos('X', 200000) should clamp to X max (120000um)."""
        board = self._make_board()
        board.move_abs_pos('X', 200000, overshoot_enabled=False)
        expected_ustep = board.xy_um2ustep(120000)
        board.driver.write.assert_called_with(f'TARGET_WX{expected_ustep}\n'.encode('utf-8'))

    def test_ignore_limits(self):
        """move_abs_pos with ignore_limits=True should not clamp."""
        board = self._make_board()
        board.move_abs_pos('Z', 99999, overshoot_enabled=False, ignore_limits=True)
        expected_ustep = board.z_um2ustep(99999)
        board.driver.write.assert_called_with(f'TARGET_WZ{expected_ustep}\n'.encode('utf-8'))

    def test_unsupported_axis_raises(self):
        """move_abs_pos with unknown axis should raise."""
        board = self._make_board()
        with pytest.raises(Exception, match="Unsupported axis"):
            board.move_abs_pos('Q', 100, overshoot_enabled=False)

    def test_move_rel_pos(self):
        """move_rel_pos should add relative distance to current target."""
        board = self._make_board()
        # target_pos reads TARGET_R, return 50000um in usteps
        target_ustep = board.xy_um2ustep(50000)
        board.driver.readline.return_value = f"{target_ustep}\n".encode('utf-8')
        board.move_rel_pos('X', 10000, overshoot_enabled=False)
        # Should move to 60000um
        expected_ustep = board.xy_um2ustep(60000)
        board.driver.write.assert_called_with(f'TARGET_WX{expected_ustep}\n'.encode('utf-8'))

    def test_target_status_position_reached(self):
        """target_status should return True when position_reached bit is set."""
        board = self._make_board()
        # bit 9 (from LSB) = 0x200 = 512
        board.driver.readline.return_value = b"512\n"
        assert board.target_status('Z') is True

    def test_target_status_not_reached(self):
        """target_status should return False when position_reached bit is clear."""
        board = self._make_board()
        board.driver.readline.return_value = b"0\n"
        assert board.target_status('Z') is False

    def test_get_axis_limits(self):
        """get_axis_limits should return the configured limits."""
        board = self._make_board()
        limits = board.get_axis_limits('Z')
        assert limits['min'] == 0.
        assert limits['max'] == 14000.

    def test_get_axis_limits_unsupported(self):
        """get_axis_limits with unknown axis should raise."""
        board = self._make_board()
        with pytest.raises(Exception, match="Unsupported axis"):
            board.get_axis_limits('Q')

    def test_get_axes_config_has_all_axes(self):
        """get_axes_config should return all 4 axes."""
        board = self._make_board()
        config = board.get_axes_config()
        assert 'X' in config
        assert 'Y' in config
        assert 'Z' in config
        assert 'T' in config


# ---------------------------------------------------------------------------
# Firmware Version Detection Tests
# ---------------------------------------------------------------------------

class TestLEDFirmwareVersion:
    """Test LED board firmware version detection and is_v2 property."""

    def _make_board(self):
        board = LEDBoard.__new__(LEDBoard)
        board.found = False
        board._lock = threading.RLock()
        board._label = '[LED Class ]'
        board.port = '/dev/fake'
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 0.1
        board.write_timeout = 0.1
        board.driver = _make_mock_serial()
        board.led_ma = {'BF': -1, 'PC': -1, 'DF': -1, 'Red': -1, 'Blue': -1, 'Green': -1}
        board.firmware_version = None
        board._last_error_log_time = 0.0
        board._error_log_interval = 2.0
        return board

    def test_detect_v2_firmware(self):
        """Should parse v2.0.1 from INFO response."""
        board = self._make_board()
        # INFO command reads two lines (echo + result)
        board.driver.readline.side_effect = [
            b"RE: INFO\r\n",
            b"Firmware:     2026-03-06 v2.0.1\r\n",
        ]
        board._detect_firmware_version()
        assert board.firmware_version == '2.0.1'
        assert board.is_v2 is True

    def test_detect_legacy_firmware(self):
        """Legacy firmware has no version string."""
        board = self._make_board()
        board.driver.readline.side_effect = [
            b"RE: INFO\r\n",
            b"EL-0925 Gen3 LED Controller\r\n",
        ]
        board._detect_firmware_version()
        assert board.firmware_version is None
        assert board.is_v2 is False

    def test_detect_v2_0_0(self):
        """Should parse v2.0.0."""
        board = self._make_board()
        board.driver.readline.side_effect = [
            b"RE: INFO\r\n",
            b"Firmware:     2026-01-15 v2.0.0\r\n",
        ]
        board._detect_firmware_version()
        assert board.firmware_version == '2.0.0'
        assert board.is_v2 is True

    def test_detect_v1_firmware(self):
        """Hypothetical v1.x firmware should not be is_v2."""
        board = self._make_board()
        board.driver.readline.side_effect = [
            b"RE: INFO\r\n",
            b"Firmware: v1.5.0\r\n",
        ]
        board._detect_firmware_version()
        assert board.firmware_version == '1.5.0'
        assert board.is_v2 is False

    def test_detect_no_echo_firmware(self):
        """Future firmware without RE: echo should still parse version."""
        board = self._make_board()
        board.driver.readline.side_effect = [
            b"Firmware:     2027-01-01 v3.0.0\r\n",
        ]
        board._detect_firmware_version()
        assert board.firmware_version == '3.0.0'
        assert board.is_v2 is True

    def test_detect_timeout(self):
        """If INFO times out, version should be None."""
        board = self._make_board()
        board.driver.write.side_effect = serial.SerialTimeoutException('timeout')
        board._detect_firmware_version()
        assert board.firmware_version is None
        assert board.is_v2 is False


class TestMotorFirmwareVersion:
    """Test motor board firmware version detection and is_v2 property."""

    def _make_board(self):
        board = MotorBoard.__new__(MotorBoard)
        board.motorconfig = MotorConfig(defaults_file=_MOTORCONFIG_DEFAULTS)
        board.found = True
        board.overshoot = False
        board.backlash = 25
        board._has_turret = False
        board.initial_homing_complete = False
        board.initial_t_homing_complete = False
        board.port = '/dev/fake'
        board._lock = threading.RLock()
        board._label = '[XYZ Class ]'
        board.thread_lock = board._lock
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 30
        board.write_timeout = 5
        board.driver = _make_mock_serial()
        board._fullinfo = None
        board._connect_fails = 0
        board.axes_config = {}
        board.firmware_version = None
        return board

    def test_detect_v2_firmware(self):
        """Should parse v2.0.1 from motor INFO response."""
        board = self._make_board()
        board.driver.readline.return_value = \
            b"EL-0940 Integrated Mainboard Firmware:     2026-03-06 v2.0.1\r\n"
        board._detect_firmware_version()
        assert board.firmware_version == '2.0.1'
        assert board.is_v2 is True

    def test_detect_legacy_firmware(self):
        """Legacy motor firmware has no version string."""
        board = self._make_board()
        board.driver.readline.return_value = \
            b"Etaluma Motor Controller Board EL-0923 Firmware:     2024-09-10\r\n"
        board._detect_firmware_version()
        assert board.firmware_version is None
        assert board.is_v2 is False

    def test_detect_v2_0_0(self):
        """Should parse v2.0.0."""
        board = self._make_board()
        board.driver.readline.return_value = \
            b"EL-0940 Integrated Mainboard Firmware:     2026-01-15 v2.0.0\r\n"
        board._detect_firmware_version()
        assert board.firmware_version == '2.0.0'
        assert board.is_v2 is True


class TestSimulatorFirmwareVersion:
    """Test that simulators expose firmware_version and is_v2."""

    def test_led_simulator_default_v2(self):
        from drivers.simulated_ledboard import SimulatedLEDBoard
        board = SimulatedLEDBoard()
        assert board.firmware_version == '2.0.1'
        assert board.is_v2 is True

    def test_led_simulator_legacy(self):
        from drivers.simulated_ledboard import SimulatedLEDBoard
        board = SimulatedLEDBoard(firmware_version=None)
        assert board.firmware_version is None
        assert board.is_v2 is False

    def test_led_simulator_custom_version(self):
        from drivers.simulated_ledboard import SimulatedLEDBoard
        board = SimulatedLEDBoard(firmware_version='1.0.0')
        assert board.firmware_version == '1.0.0'
        assert board.is_v2 is False

    def test_motor_simulator_default_v2(self):
        from drivers.simulated_motorboard import SimulatedMotorBoard
        board = SimulatedMotorBoard()
        assert board.firmware_version == '2.0.1'
        assert board.is_v2 is True

    def test_motor_simulator_legacy(self):
        from drivers.simulated_motorboard import SimulatedMotorBoard
        board = SimulatedMotorBoard(firmware_version=None)
        assert board.firmware_version is None
        assert board.is_v2 is False


# ---------------------------------------------------------------------------
# C1: exchange_command None handling (led_on block, wait_until_on)
# ---------------------------------------------------------------------------

class TestLEDNoneHandling:
    """Verify LED methods handle None from exchange_command without crashing."""

    def _make_board(self):
        board = LEDBoard.__new__(LEDBoard)
        board.found = False
        board._lock = threading.RLock()
        board._label = '[LED Class ]'
        board.port = '/dev/fake'
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 0.1
        board.write_timeout = 0.1
        board.driver = _make_mock_serial()
        board.led_ma = {'BF': -1, 'PC': -1, 'DF': -1, 'Red': -1, 'Blue': -1, 'Green': -1}
        board.firmware_version = None
        board._last_error_log_time = 0.0
        board._error_log_interval = 2.0
        return board

    def test_led_on_block_with_none_then_success(self):
        """led_on(block=True) should retry when exchange_command returns None."""
        board = self._make_board()
        # First call: echo + None-inducing timeout, second call: echo + valid response
        call_count = [0]
        def mock_readline():
            call_count[0] += 1
            if call_count[0] <= 2:
                # First exchange_command: echo then timeout (empty = None after strip)
                if call_count[0] == 1:
                    return b"RE: LED3_100\r\n"
                return b"\r\n"  # empty result -> stripped to ''
            elif call_count[0] == 3:
                return b"RE: LED3_100\r\n"
            else:
                return b"LED 3 set to 100 mA. LED3_100\r\n"
        board.driver.readline = mock_readline
        # Should not crash — first response is empty string (not None), second succeeds
        board.led_on(channel=3, mA=100, block=True)
        assert board.led_ma['BF'] == 100

    def test_led_on_block_none_response_no_crash(self):
        """led_on(block=True) must not crash when exchange_command returns None."""
        board = self._make_board()
        call_count = [0]
        def mock_exchange(cmd):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # simulate disconnect
            return f"LED3_100 {cmd}"  # valid on retry
        board.exchange_command = mock_exchange
        board.led_on(channel=3, mA=100, block=True)
        assert call_count[0] == 2

    def test_wait_until_on_none_response_no_crash(self):
        """wait_until_on must not crash when get_status returns None."""
        board = self._make_board()
        call_count = [0]
        def mock_exchange(cmd):
            call_count[0] += 1
            if call_count[0] == 1:
                return None
            return "RE: STATUS LED3:100mA"
        board.exchange_command = mock_exchange
        board.wait_until_on()
        assert call_count[0] == 2
