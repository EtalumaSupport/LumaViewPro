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
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock
import serial

# Heavy deps are mocked by tests/conftest.py at module-import time.

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
    mock.in_waiting = 0
    mock.read.return_value = b""
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
        board._state_lock = threading.Lock()
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

    def test_driver_stays_open_on_timeout(self):
        """H-17: Timeout is transient — driver stays open for retry.

        Only fatal exceptions close the driver. SerialTimeoutException
        keeps it open so the next command can succeed without reconnecting.
        """
        board = self._make_board()
        mock_driver = board.driver
        board.driver.write.side_effect = serial.SerialTimeoutException("timeout")
        board.exchange_command('LED0_100')
        mock_driver.close.assert_not_called()
        assert board.driver is not None

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
        board._state_lock = threading.Lock()
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
        board._state_lock = threading.Lock()
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

    def test_get_status_returns_none(self):
        """get_status() returns None — LED firmware has no STATUS command."""
        board = self._make_board()
        result = board.get_status()
        assert result is None
        # Should NOT send any serial command (STATUS not implemented in firmware)
        board.driver.write.assert_not_called()


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
        board._state_lock = threading.Lock()
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
        board._state_lock = threading.Lock()
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

    def test_home_sends_home(self):
        """home() should send 'HOME\\n'."""
        board = self._make_board()
        board.home()
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
        board._state_lock = threading.Lock()
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
        assert board.has_homed() is False
        assert board.has_thomed() is False

    def test_home_sets_flag_on_success(self):
        """home() should set initial_homing_complete when firmware confirms."""
        board = self._make_board()
        board.driver.readline.return_value = b"XYZ home complete\n"
        board.home()
        assert board.has_homed() is True

    def test_home_sets_flag_on_partial_success(self):
        """home() should set initial_homing_complete on partial home — the
        firmware homed Z (and T if present) before reporting that X or Y
        is not physically wired on this board (LS820 case, #618 follow-up)."""
        board = self._make_board()
        board.driver.readline.return_value = b"ERROR: X not present\n"
        board.home()
        assert board.has_homed() is True, (
            "Partial home (Z homed before firmware reported missing X/Y) "
            "must set the homed flag — Z is at its reference position"
        )

    def test_home_no_flag_on_real_failure(self):
        """home() should not set flag for a real failure (timeout, hardware
        error) — distinct from the LS820 partial-home case above."""
        board = self._make_board()
        board.driver.readline.return_value = b"ERROR: timeout\n"
        board.home()
        assert board.has_homed() is False

    def test_home_no_flag_on_none(self):
        """home() should not set flag if response is None (disconnected)."""
        board = self._make_board()
        board.driver.write.side_effect = serial.SerialTimeoutException("timeout")
        board.home()
        assert board.has_homed() is False

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

    def test_has_thomed_true_after_home(self):
        """has_thomed() should return True if home() completed (it homes T too)."""
        board = self._make_board()
        board.driver.readline.return_value = b"XYZ home complete\n"
        board.home()
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
        board._state_lock = threading.Lock()
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
        board._state_lock = threading.Lock()
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
        board._state_lock = threading.Lock()
        board.firmware_version = None
        board.firmware_date = None
        board.firmware_responding = False
        board.protocol_version = None
        board._last_error_log_time = 0.0
        board._error_log_interval = 2.0
        return board

    def test_detect_v2_firmware(self):
        """Should parse v2.0.1 from INFO response."""
        board = self._make_board()
        # exchange_command('INFO', response_numlines=6) reads up to 6 lines;
        # first line is echo (RE: prefix), rest are content or empty (timeout).
        board.driver.readline.side_effect = [
            b"RE: INFO\r\n",
            b"Firmware:     2026-03-06 v2.0.1\r\n",
            b"\r\n", b"\r\n", b"\r\n", b"\r\n", b"\r\n", b"\r\n",
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
            b"\r\n", b"\r\n", b"\r\n", b"\r\n", b"\r\n", b"\r\n",
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
            b"\r\n", b"\r\n", b"\r\n", b"\r\n", b"\r\n", b"\r\n",
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
            b"\r\n", b"\r\n", b"\r\n", b"\r\n", b"\r\n", b"\r\n",
        ]
        board._detect_firmware_version()
        assert board.firmware_version == '1.5.0'
        assert board.is_v2 is False

    def test_detect_no_echo_firmware(self):
        """Future firmware without RE: echo should still parse version."""
        board = self._make_board()
        board.driver.readline.side_effect = [
            b"Firmware:     2027-01-01 v3.0.0\r\n",
            b"\r\n", b"\r\n", b"\r\n", b"\r\n", b"\r\n", b"\r\n",
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
        board._state_lock = threading.Lock()
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
        board._state_lock = threading.Lock()
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

    def test_wait_until_on_returns_immediately(self):
        """wait_until_on returns immediately — STATUS not implemented in firmware."""
        board = self._make_board()
        # Should return without sending any commands
        board.wait_until_on()
        board.driver.write.assert_not_called()


# ==========================================================================
# Concurrency / State Lock Tests (C-1 through C-4)
# ==========================================================================

class TestLEDBoardStateLock:
    """Verify LEDBoard _state_lock protects led_ma from concurrent access."""

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
        board._state_lock = threading.Lock()
        return board

    def test_led_on_uses_state_lock(self):
        """led_on() should acquire _state_lock when updating led_ma (after serial succeeds)."""
        board = self._make_board()
        acquired = []
        original_lock = board._state_lock

        class TrackingLock:
            def __enter__(self_lock):
                acquired.append('enter')
                return original_lock.__enter__()
            def __exit__(self_lock, *args):
                acquired.append('exit')
                return original_lock.__exit__(*args)

        board._state_lock = TrackingLock()
        board.led_on(channel=0, mA=100)
        assert 'enter' in acquired, "_state_lock was not acquired during led_on()"
        assert board.led_ma['Blue'] == 100

    def test_led_on_no_state_update_on_failure(self):
        """led_on() should NOT update state cache if serial returns None."""
        board = self._make_board()
        board.driver.write.side_effect = serial.SerialTimeoutException('write timeout')
        board.led_on(channel=0, mA=100)
        assert board.led_ma['Blue'] == -1, "State cache should not update on serial failure"

    def test_led_off_uses_state_lock(self):
        """led_off() should acquire _state_lock when updating led_ma (after serial succeeds)."""
        board = self._make_board()
        board.led_ma['Blue'] = 200
        acquired = []
        original_lock = board._state_lock

        class TrackingLock:
            def __enter__(self_lock):
                acquired.append('enter')
                return original_lock.__enter__()
            def __exit__(self_lock, *args):
                return original_lock.__exit__(*args)

        board._state_lock = TrackingLock()
        board.led_off(channel=0)
        assert 'enter' in acquired
        assert board.led_ma['Blue'] == -1

    def test_led_off_no_state_update_on_failure(self):
        """led_off() should NOT update state cache if serial returns None."""
        board = self._make_board()
        board.led_ma['Blue'] = 200
        board.driver.write.side_effect = serial.SerialTimeoutException('write timeout')
        board.led_off(channel=0)
        assert board.led_ma['Blue'] == 200, "State cache should not update on serial failure"

    def test_get_led_states_returns_consistent_snapshot(self):
        """get_led_states() should return a consistent snapshot under _state_lock."""
        board = self._make_board()
        board.led_ma['Blue'] = 100
        board.led_ma['Red'] = 200
        states = board.get_led_states()
        assert states['Blue'] == {'enabled': True, 'illumination': 100}
        assert states['Red'] == {'enabled': True, 'illumination': 200}
        assert states['BF'] == {'enabled': False, 'illumination': -1}

    def test_concurrent_led_on_off(self):
        """Concurrent led_on and led_off should not corrupt led_ma."""
        board = self._make_board()
        errors = []

        def toggle_on():
            try:
                for _ in range(50):
                    board.led_on(channel=0, mA=100)
            except Exception as e:
                errors.append(e)

        def toggle_off():
            try:
                for _ in range(50):
                    board.led_off(channel=0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle_on), threading.Thread(target=toggle_off)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors, f"Concurrent access raised errors: {errors}"
        # Final state should be either 100 or -1, never anything else
        assert board.led_ma['Blue'] in (100, -1)

    def test_leds_off_clears_all(self):
        """leds_off() should atomically clear all channels under _state_lock."""
        board = self._make_board()
        board.led_ma['Blue'] = 100
        board.led_ma['Red'] = 200
        board.led_ma['Green'] = 300
        board.leds_off()
        for color, val in board.led_ma.items():
            assert val == -1, f"{color} was not cleared by leds_off()"

    def test_leds_off_fast_clears_all(self):
        """leds_off_fast() should atomically clear all channels."""
        board = self._make_board()
        board.led_ma['Blue'] = 100
        board.led_ma['Red'] = 200
        # Mock _write_command_fast
        board._write_command_fast = MagicMock()
        board.leds_off_fast()
        for color, val in board.led_ma.items():
            assert val == -1

    def test_led_on_fast_uses_state_lock(self):
        """led_on_fast() should protect led_ma with _state_lock."""
        board = self._make_board()
        board._write_command_fast = MagicMock()
        board.led_on_fast(channel=2, mA=500)
        assert board.led_ma['Red'] == 500

    def test_led_off_fast_uses_state_lock(self):
        """led_off_fast() should protect led_ma with _state_lock."""
        board = self._make_board()
        board.led_ma['Red'] = 500
        board._write_command_fast = MagicMock()
        board.led_off_fast(channel=2)
        assert board.led_ma['Red'] == -1

    def test_get_led_ma_thread_safe(self):
        """get_led_ma() should read under _state_lock."""
        board = self._make_board()
        board.led_ma['Green'] = 150
        assert board.get_led_ma('Green') == 150
        assert board.get_led_ma('BF') == -1
        assert board.get_led_ma('nonexistent') == -1

    def test_is_led_on_thread_safe(self):
        """is_led_on() should read under _state_lock."""
        board = self._make_board()
        board.led_ma['Green'] = 150
        assert board.is_led_on('Green') is True
        assert board.is_led_on('BF') is False


class TestSerialDesyncRecovery:
    """Verify that serial response desync is recovered via input buffer flush.

    Scenario: if readline() times out (returns b""), the firmware's response
    sits in the input buffer. Without the flush fix, the next exchange_command
    reads the stale response, creating a permanent desync cascade.
    """

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
        board._state_lock = threading.Lock()
        board._last_error_log_time = 0.0
        board._error_log_interval = 2.0
        return board

    def test_stale_buffer_flushed_before_write(self):
        """exchange_command should flush stale input bytes before writing."""
        board = self._make_board()
        # Simulate 20 stale bytes sitting in the input buffer initially,
        # then 0 after the flush (PropertyMock cycles through values).
        # The post-response drain also checks in_waiting, so we need
        # 20 for the pre-flush, then 0 for the post-response drain.
        type(board.driver).in_waiting = PropertyMock(side_effect=[20, 0])
        board.driver.readline.return_value = b"OK\r\n"

        response = board.exchange_command('STATUS')

        # Verify stale data was read (flushed) before the command was written
        board.driver.read.assert_called_once_with(20)
        assert response == 'OK'

    def test_no_flush_when_buffer_empty(self):
        """exchange_command should not flush when input buffer is empty."""
        board = self._make_board()
        board.driver.in_waiting = 0
        board.driver.readline.return_value = b"OK\r\n"

        response = board.exchange_command('STATUS')

        board.driver.read.assert_not_called()
        assert response == 'OK'

    def test_rapid_led_on_off_state_consistent(self):
        """Rapid led_on/leds_off cycling (25x) should keep state consistent.

        This reproduces the characterization test pattern that caused LED
        dropout: 25 iterations of leds_off -> led_on -> leds_off.
        """
        board = self._make_board()
        board.driver.in_waiting = 0

        for i in range(25):
            board.leds_off()
            assert board.led_ma['BF'] == -1, f"Iteration {i}: BF should be off after leds_off"
            board.led_on(channel=3, mA=20)
            assert board.led_ma['BF'] == 20, f"Iteration {i}: BF should be 20 after led_on"
            board.leds_off()
            assert board.led_ma['BF'] == -1, f"Iteration {i}: BF should be off after final leds_off"

    def test_rapid_led_cycling_with_timeouts(self):
        """LED state stays correct even when some commands timeout.

        Simulates intermittent serial timeouts during rapid cycling.
        With the flush fix, stale responses are cleared before each command.
        """
        board = self._make_board()
        call_count = [0]
        original_readline = board.driver.readline

        def flaky_readline():
            call_count[0] += 1
            # Every 7th readline returns empty (simulates timeout)
            if call_count[0] % 7 == 0:
                return b""
            return b"OK\r\n"

        board.driver.readline = flaky_readline
        # After a timeout, stale bytes appear in buffer
        stale_count = [0]
        def dynamic_in_waiting():
            # After a timeout, simulate stale data
            if call_count[0] > 0 and call_count[0] % 7 == 1:
                return 15  # stale bytes from missed response
            return 0
        type(board.driver).in_waiting = property(lambda self: dynamic_in_waiting())

        # Run 25 cycles — should not crash or desync
        successes = 0
        for i in range(25):
            board.leds_off()
            board.led_on(channel=3, mA=20)
            if board.led_ma['BF'] == 20:
                successes += 1
            board.leds_off()

        # Most cycles should succeed (some may fail due to simulated timeouts)
        # The key is: no crashes, no permanent desync cascade
        assert successes > 15, f"Only {successes}/25 cycles succeeded — desync cascade?"


class TestSilentBoardHandling:
    """Regression tests for #619 Phase B — silent board detection.

    A "silent" board sends zero bytes during the entire connect
    sequence. Distinguished from a legacy (pre-v3.0) board that
    responds to INFO with unparseable text but still echoes bytes.
    Silent boards are hung firmware or a stuck USB hub and cannot
    recover via Ctrl-D — they need a hardware power cycle.

    The structural fix:
    - _reset_firmware tracks bytes_ever_seen and skips the Ctrl-D
      soft-reset recovery path when no bytes were captured
    - _reset_firmware marks firmware_silent = True when all retries
      produce zero bytes
    - connect() surfaces firmware_silent via notifications.error
      instead of silently degrading to "legacy, no version info"
    - exchange_command() fails fast (returns None immediately)
      when firmware_silent is True
    """

    def _make_silent_board(self):
        """Build an LEDBoard whose serial driver returns zero bytes
        for every read and reports in_waiting=0 forever. Simulates
        the #619 hung-firmware scenario at the driver level."""
        board = LEDBoard.__new__(LEDBoard)
        board.found = True
        board._lock = threading.RLock()
        board._label = '[LED Class ]'
        board.port = '/dev/fake_silent'
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 0.01  # Keep test fast; we don't care about real timing
        board.write_timeout = 0.01
        board.led_ma = {'BF': -1, 'PC': -1, 'DF': -1,
                        'Red': -1, 'Blue': -1, 'Green': -1}
        board._state_lock = threading.Lock()
        board.firmware_version = None
        board.firmware_date = None
        board.firmware_responding = False
        board.firmware_silent = False
        board._detect_response_bytes = 0
        board._last_error_log_time = 0.0
        board._error_log_interval = 2.0
        board._min_command_interval = 0.0
        board._last_command_time = 0.0
        board._in_raw_repl = False
        from drivers.serialboard import ProtocolVersion
        board.protocol_version = ProtocolVersion.LEGACY
        # Silent driver: every read is empty, in_waiting is always 0,
        # writes succeed (bytes go into the void)
        board.driver = _make_mock_serial()
        board.driver.readline.return_value = b""
        board.driver.read.return_value = b""
        board.driver.in_waiting = 0
        return board

    def _make_responsive_legacy_board(self):
        """Build an LEDBoard that responds to INFO with unparseable
        bytes (old LED firmware that has no version string). This is
        the case the silent-board path must NOT trigger on — it's a
        genuinely legacy board, not a hung one."""
        board = self._make_silent_board()
        # readline returns SOMETHING (the legacy firmware's INFO reply)
        # on the first few calls. Content doesn't need to be parseable,
        # it just needs to be non-empty so _detect_response_bytes > 0.
        import itertools
        responses = itertools.cycle([
            b"EL-0940 Mainboard\r\n",
            b"No version\r\n",
            b"\r\n",
            b"\r\n",
            b"\r\n",
            b"\r\n",
        ])
        board.driver.readline.side_effect = lambda: next(responses)
        return board

    def test_silent_board_marks_firmware_silent_true(self):
        """After _reset_firmware on a silent board, firmware_silent
        must be True and firmware_responding must be False."""
        board = self._make_silent_board()
        board._reset_firmware()
        assert board.firmware_silent is True
        assert board.firmware_responding is False

    def test_silent_board_still_attempts_ctrl_d_recovery(self):
        """The soft-reset recovery path (step 4) must run even when
        `bytes_ever_seen == 0`. Ctrl-D is the only way to wake up a
        board that was just used by Thonny and left in raw REPL
        state — in that state, MicroPython doesn't echo or execute
        anything until Ctrl-D arrives, so the board looks silent to
        drain + INFO detect but is actually alive.

        History: an earlier version of `_reset_firmware()` skipped
        step 4 entirely when `bytes_ever_seen == 0`, assuming a
        silent board wouldn't respond to Ctrl-D either. That skip
        was correct for the in-house bench brick case (board
        genuinely hung, Ctrl-D did nothing) but wrong for the
        Thonny-left-in-raw-REPL case (board alive, Ctrl-D is the
        wake-up signal). Verified in-session 2026-04-14 on a real
        LS850T test board after a Thonny connect/disconnect cycle:
        LVP with the skip in place falsely flagged both boards as
        silent and LVP couldn't recover them.

        **The guiding principle: robust startup trumps 5 seconds
        of optimization.** Always run the recovery sequence
        before declaring a board silent.

        This test asserts Ctrl-B (b'\\x02', exit raw REPL) and
        Ctrl-D (b'\\x04', soft reset) are both written during the
        recovery path, even on a silent board. It replaces the
        earlier `test_silent_board_skips_ctrl_d` which pinned the
        now-reverted optimization.
        """
        board = self._make_silent_board()
        board._reset_firmware()
        all_writes = [call.args[0] for call in board.driver.write.call_args_list]
        assert b'\x04' in all_writes, (
            f"Ctrl-D (b'\\x04') MUST be sent during recovery even "
            f"on a silent board, to handle Thonny-left-in-raw-REPL "
            f"state. Writes were: {all_writes}"
        )
        assert b'\x02' in all_writes, (
            f"Ctrl-B (b'\\x02', raw REPL exit) must also be sent as "
            f"part of the recovery sequence. Writes were: {all_writes}"
        )

    def test_silent_board_exchange_command_fails_fast(self):
        """After a silent _reset_firmware, exchange_command on any
        non-INFO command must return None immediately without
        touching the driver."""
        board = self._make_silent_board()
        board._reset_firmware()
        assert board.firmware_silent is True

        # Clear the write history so we only see calls after the reset.
        board.driver.write.reset_mock()
        board.driver.readline.reset_mock()

        t0 = time.monotonic()
        result = board.exchange_command('LEDS_OFF')
        elapsed_ms = (time.monotonic() - t0) * 1000

        assert result is None, "Silent board must return None from exchange_command"
        assert elapsed_ms < 100, (
            f"Silent board exchange must fail fast (<100ms), took {elapsed_ms:.0f}ms — "
            f"suggests the timeout path was hit"
        )
        board.driver.write.assert_not_called()

    def test_silent_board_exchange_allows_info(self):
        """INFO is exempted from fail-fast so a reconnect attempt can
        re-probe the board (e.g. after the user power-cycles it).
        Otherwise firmware_silent becomes sticky."""
        board = self._make_silent_board()
        board.firmware_silent = True  # simulate prior silent connect

        # INFO should still reach the driver. We don't care about the
        # response — just that write was called.
        board.driver.write.reset_mock()
        board.exchange_command('INFO')
        # At least one write happened (driver.write called with b'INFO\n')
        assert board.driver.write.called, (
            "INFO must be allowed through exchange_command even when "
            "firmware_silent is True, so reconnect probes can clear the flag"
        )

    def test_responsive_legacy_board_does_not_trigger_silent(self):
        """Pre-v3.0 LED boards respond to INFO with unparseable text
        (no version string) but still echo bytes. They must take the
        normal legacy path — firmware_silent must stay False."""
        board = self._make_responsive_legacy_board()
        board._reset_firmware()
        assert board.firmware_silent is False, (
            "A board that sent ANY bytes is not silent — it's legacy"
        )

    def test_silent_board_reset_clears_stale_silent_flag(self):
        """After a power cycle, the user reconnects. _reset_firmware
        must clear any stale silent flag at the start so a board
        that now responds gets a fresh verdict."""
        board = self._make_silent_board()
        board.firmware_silent = True  # simulate carried-over stale flag
        board._detect_response_bytes = 0

        # Swap to a responsive driver mid-test (simulate power cycle)
        import itertools
        responses = itertools.cycle([
            b"Firmware: 2026-03-18 v3.0.4\r\n",
            b"\r\n", b"\r\n", b"\r\n", b"\r\n", b"\r\n",
        ])
        board.driver.readline.side_effect = lambda: next(responses)

        board._reset_firmware()
        assert board.firmware_silent is False, (
            "_reset_firmware must clear stale firmware_silent at entry"
        )

    def test_silent_board_cannot_auto_reconnect_loop(self):
        """Repeated exchange_command on a silent board must not loop
        forever — each call returns None fast without calling
        self.connect()."""
        board = self._make_silent_board()
        board.firmware_silent = True
        board.driver.write.reset_mock()

        for _ in range(10):
            result = board.exchange_command('LEDS_OFF')
            assert result is None

        board.driver.write.assert_not_called()


class TestExchangeCommandStopOnEmpty:
    """Regression for the motor INFO 2.5s wasted timeout.

    _detect_firmware_version calls exchange_command('INFO',
    response_numlines=6, timeout=0.5, stop_on_empty=True). The motor
    firmware sends INFO as a single line but the reader was waiting
    the full per-line timeout × 5 on all the empty subsequent lines,
    wasting 2.5s on every healthy motor connect.

    stop_on_empty=True breaks out of the readline loop once an empty
    line arrives after non-empty content. Safe because neither motor
    nor LED INFO contains intentional empty lines inside the content.
    """

    def _make_led_board(self):
        board = LEDBoard.__new__(LEDBoard)
        board.found = True
        board._lock = threading.RLock()
        board._label = '[LED Class ]'
        board.port = '/dev/fake'
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 0.1
        board.write_timeout = 0.1
        board.led_ma = {}
        board._state_lock = threading.Lock()
        board.firmware_version = None
        board._last_error_log_time = 0.0
        board._error_log_interval = 2.0
        board._min_command_interval = 0.0
        board._last_command_time = 0.0
        board.driver = _make_mock_serial()
        return board

    def test_stop_on_empty_breaks_after_first_empty(self):
        """Motor INFO case: one content line then empty lines — must
        break on first empty line after content, not read all 6."""
        board = self._make_led_board()
        # Motor INFO reply style: one content line, then empty lines
        # (which represent readline timeouts in the real driver).
        board.driver.readline.side_effect = [
            b"EL-0940-04 Integrated Mainboard Firmware: 2026-04-01 v3.0.9\r\n",
            b"",  # empty — should break the loop
            b"SHOULD_NOT_READ\r\n",  # must not reach this line
            b"ALSO_SHOULD_NOT_READ\r\n",
            b"ALSO_SHOULD_NOT_READ\r\n",
            b"ALSO_SHOULD_NOT_READ\r\n",
        ]
        resp = board.exchange_command('INFO', response_numlines=6,
                                      stop_on_empty=True)
        assert isinstance(resp, list)
        assert len(resp) == 6  # padded to requested length
        assert 'v3.0.9' in resp[0]
        # The break must prevent subsequent reads from the side_effect
        # list. We assert this by checking the padding (empty strings)
        # rather than counting readline calls (which include the echo
        # handling that may consume additional entries).
        assert all(ln == '' for ln in resp[1:]), (
            f"stop_on_empty should break on first empty; got {resp}"
        )

    def test_stop_on_empty_reads_full_multiline_led_info(self):
        """LED INFO case: all 6 lines have content — stop_on_empty
        must NOT trigger because no empty line appears."""
        board = self._make_led_board()
        board.driver.readline.side_effect = [
            b"Version:      EL-0940 Integrated Mainboard\r\n",
            b"Firmware:     2026-04-01 v3.0.7\r\n",
            b"Copyright:    Etaluma, Inc.\r\n",
            b"Calibration:  Default\r\n",
            b"Reset cause:  Power-on\r\n",
            b"Heap free:    154512 bytes\r\n",
        ]
        resp = board.exchange_command('INFO', response_numlines=6,
                                      stop_on_empty=True)
        assert isinstance(resp, list)
        assert len(resp) == 6
        assert 'v3.0.7' in resp[1]
        assert 'Heap free' in resp[5]
        # All 6 lines present means the break was NOT taken — safe.

    def test_stop_on_empty_without_any_content_reads_all(self):
        """Silent board case: every line is empty — must read all 6
        (the 'no content at all' signal is what Phase B's silent
        detection relies on)."""
        board = self._make_led_board()
        board.driver.readline.side_effect = [b"", b"", b"", b"", b"", b""]
        resp = board.exchange_command('INFO', response_numlines=6,
                                      stop_on_empty=True)
        assert isinstance(resp, list)
        assert len(resp) == 6
        assert all(ln == '' for ln in resp)
        # Phase B's silent detection sees 6 empty lines and concludes
        # "silent board" regardless of whether we break or not.

    def test_default_behavior_unchanged(self):
        """Regression: stop_on_empty defaults to False. Existing
        single-line command callers (STATUS, LED0_100, etc.) must
        get exactly their requested number of lines."""
        board = self._make_led_board()
        board.driver.readline.side_effect = [
            b"RE: LED0_100\r\n",
            b"LED 0 set to 100 mA.\r\n",
        ]
        resp = board.exchange_command('LED0_100')
        assert resp == 'LED 0 set to 100 mA.'


class TestMotorBoardStateLock:
    """Verify MotorBoard _state_lock protects state flags from concurrent access."""

    def _make_board(self):
        board = MotorBoard.__new__(MotorBoard)
        board.motorconfig = MotorConfig(defaults_file=_MOTORCONFIG_DEFAULTS)
        board.found = True
        board._state_lock = threading.Lock()
        board.overshoot = False
        board.backlash = 25
        board._has_turret = False
        board.initial_homing_complete = False
        board.initial_t_homing_complete = False
        board._fullinfo = {"model": "LS720", "serial_number": "12345"}
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
        board.axes_config = {
            'Z': {'limits': {'min': 0., 'max': 20000.}, 'move_func': board.z_um2ustep},
            'X': {'limits': {'min': 0., 'max': 100000.}, 'move_func': board.xy_um2ustep},
            'Y': {'limits': {'min': 0., 'max': 100000.}, 'move_func': board.xy_um2ustep},
            'T': {'move_func': board.t_pos2ustep},
        }
        return board

    def test_home_sets_homing_complete_under_lock(self):
        """home() should set initial_homing_complete under _state_lock."""
        board = self._make_board()
        board.exchange_command = MagicMock(return_value="XYZ home complete")
        board.home()
        assert board.has_homed() is True

    def test_thome_sets_t_homing_complete_under_lock(self):
        """thome() should set initial_t_homing_complete under _state_lock."""
        board = self._make_board()
        board.exchange_command = MagicMock(return_value="T home successful")
        board.thome()
        assert board.has_thomed() is True

    def test_has_turret_reads_under_lock(self):
        """has_turret() should read _has_turret under _state_lock."""
        board = self._make_board()
        assert board.has_turret() is False
        with board._state_lock:
            board._has_turret = True
        assert board.has_turret() is True

    def test_has_thomed_combines_both_flags(self):
        """has_thomed() returns True if either homing flag is set."""
        board = self._make_board()
        assert board.has_thomed() is False
        with board._state_lock:
            board.initial_t_homing_complete = True
        assert board.has_thomed() is True

    def test_get_microscope_model_reads_under_lock(self):
        """get_microscope_model() should read _fullinfo under _state_lock."""
        board = self._make_board()
        assert board.get_microscope_model() == "LS720"

    def test_fullinfo_sets_has_turret_for_T_model(self):
        """fullinfo() should set _has_turret when model ends in T."""
        board = self._make_board()
        board.exchange_command = MagicMock(
            return_value="Etaluma Motor Controller Board Model: LS720T Serial: 99999"
        )
        info = board.fullinfo()
        assert info['model'] == 'LS720T'
        assert board.has_turret() is True

    def test_concurrent_homing_flag_access(self):
        """Concurrent reads/writes of homing flags should not raise."""
        board = self._make_board()
        board.exchange_command = MagicMock(return_value="XYZ home complete")
        errors = []

        def do_home():
            try:
                for _ in range(20):
                    board.home()
            except Exception as e:
                errors.append(e)

        def check_homed():
            try:
                for _ in range(20):
                    board.has_homed()
                    board.has_thomed()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_home), threading.Thread(target=check_homed)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors, f"Concurrent homing access raised: {errors}"


class TestCameraStateLock:
    """Verify Camera _state_lock protects active/device_removed flags."""

    def test_mark_disconnected_sets_both_flags_atomically(self):
        """_mark_disconnected should set both flags under _state_lock."""
        from drivers.camera import Camera

        # Create a minimal concrete Camera subclass
        class StubCamera(Camera):
            def connect(self): self.active = True; return True
            def disconnect(self): self.active = None; return True
            def is_connected(self): return self.active is not None and self.active is not False
            def init_camera_config(self): pass
            def start_grabbing(self): pass
            def stop_grabbing(self): pass
            def is_grabbing(self): return False
            def set_frame_size(self, w, h): pass
            def get_min_frame_size(self): return {'w': 1, 'h': 1}
            def get_max_frame_size(self): return {'w': 4096, 'h': 4096}
            def get_frame_size(self): return {'w': 1024, 'h': 1024}
            def set_pixel_format(self, f): return True
            def get_pixel_format(self): return 'Mono8'
            def get_supported_pixel_formats(self): return ('Mono8',)
            def exposure_t(self, t): pass
            def get_exposure_t(self): return 10.0
            def auto_exposure_t(self, state=True): pass
            def find_model_name(self): self.model_name = 'TestCam'
            def get_all_temperatures(self): return {}
            def set_max_acquisition_frame_rate(self, enabled, fps=1.0): pass
            def set_binning_size(self, size): return True
            def get_binning_size(self): return 1
            def grab_new_capture(self, timeout): return True, None
            def update_auto_gain_target_brightness(self, v): pass
            def update_auto_gain_min_max(self, min_g, max_g): pass
            def get_gain(self): return 1.0
            def gain(self, g): pass
            def auto_gain(self, state=True, **kw): pass
            def auto_gain_once(self, state=True, **kw): pass
            def set_test_pattern(self, enabled=False, pattern='Black'): pass

        cam = StubCamera()
        assert cam.active is True
        assert cam._device_removed is False

        cam._mark_disconnected()
        assert cam.active is None
        assert cam._device_removed is True
        assert cam.is_device_removed() is True

    def test_grab_returns_false_after_disconnect(self):
        """grab() should check flags under _state_lock and return False."""
        from drivers.camera import Camera, ImageHandlerBase

        class StubCamera(Camera):
            def connect(self): self.active = True; return True
            def disconnect(self): self.active = None; return True
            def is_connected(self): return self.active is not None
            def init_camera_config(self): pass
            def start_grabbing(self): pass
            def stop_grabbing(self): pass
            def is_grabbing(self): return False
            def set_frame_size(self, w, h): pass
            def get_min_frame_size(self): return {'w': 1, 'h': 1}
            def get_max_frame_size(self): return {'w': 4096, 'h': 4096}
            def get_frame_size(self): return {'w': 1024, 'h': 1024}
            def set_pixel_format(self, f): return True
            def get_pixel_format(self): return 'Mono8'
            def get_supported_pixel_formats(self): return ('Mono8',)
            def exposure_t(self, t): pass
            def get_exposure_t(self): return 10.0
            def auto_exposure_t(self, state=True): pass
            def find_model_name(self): self.model_name = 'TestCam'
            def get_all_temperatures(self): return {}
            def set_max_acquisition_frame_rate(self, enabled, fps=1.0): pass
            def set_binning_size(self, size): return True
            def get_binning_size(self): return 1
            def grab_new_capture(self, timeout): return True, None
            def update_auto_gain_target_brightness(self, v): pass
            def update_auto_gain_min_max(self, min_g, max_g): pass
            def get_gain(self): return 1.0
            def gain(self, g): pass
            def auto_gain(self, state=True, **kw): pass
            def auto_gain_once(self, state=True, **kw): pass
            def set_test_pattern(self, enabled=False, pattern='Black'): pass

        cam = StubCamera()
        assert cam.active is True

        # Normal grab without image handler should return False (no handler)
        result, ts = cam.grab()
        assert result is False

        # After disconnect, grab should return False immediately
        cam._mark_disconnected()
        result, ts = cam.grab()
        assert result is False

    def test_concurrent_mark_disconnected(self):
        """Multiple threads calling _mark_disconnected should not raise."""
        from drivers.camera import Camera

        class StubCamera(Camera):
            def connect(self): self.active = True; return True
            def disconnect(self): return True
            def is_connected(self): return self.active is not None
            def init_camera_config(self): pass
            def start_grabbing(self): pass
            def stop_grabbing(self): pass
            def is_grabbing(self): return False
            def set_frame_size(self, w, h): pass
            def get_min_frame_size(self): return {'w': 1, 'h': 1}
            def get_max_frame_size(self): return {'w': 4096, 'h': 4096}
            def get_frame_size(self): return {'w': 1024, 'h': 1024}
            def set_pixel_format(self, f): return True
            def get_pixel_format(self): return 'Mono8'
            def get_supported_pixel_formats(self): return ('Mono8',)
            def exposure_t(self, t): pass
            def get_exposure_t(self): return 10.0
            def auto_exposure_t(self, state=True): pass
            def find_model_name(self): self.model_name = 'TestCam'
            def get_all_temperatures(self): return {}
            def set_max_acquisition_frame_rate(self, enabled, fps=1.0): pass
            def set_binning_size(self, size): return True
            def get_binning_size(self): return 1
            def grab_new_capture(self, timeout): return True, None
            def update_auto_gain_target_brightness(self, v): pass
            def update_auto_gain_min_max(self, min_g, max_g): pass
            def get_gain(self): return 1.0
            def gain(self, g): pass
            def auto_gain(self, state=True, **kw): pass
            def auto_gain_once(self, state=True, **kw): pass
            def set_test_pattern(self, enabled=False, pattern='Black'): pass

        cam = StubCamera()
        errors = []

        def disconnect_loop():
            try:
                for _ in range(50):
                    cam._mark_disconnected()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=disconnect_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors
        assert cam._device_removed is True
        assert cam.active is None
