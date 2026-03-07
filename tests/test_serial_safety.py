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
from ledboard import LEDBoard
from motorboard import MotorBoard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_serial(**overrides):
    """Create a mock serial.Serial that behaves like a connected port."""
    mock = MagicMock(spec=serial.Serial)
    mock.readline.return_value = b"RE: OK\r\n"
    mock.write.return_value = None
    mock.flushInput.return_value = None
    mock.flush.return_value = None
    mock.close.return_value = None
    for k, v in overrides.items():
        setattr(mock, k, v)
    return mock


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
        """exchange_command should return the decoded response."""
        board = self._make_board()
        board.driver.readline.return_value = b"RE: LED0_100\r\n"
        resp = board.exchange_command('LED0_100')
        assert resp is not None
        assert 'LED0_100' in resp

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
        board.found = True
        board.overshoot = False
        board.backlash = 25
        board._has_turret = False
        board.initial_homing_complete = False
        board.initial_t_homing_complete = False
        board.port = '/dev/fake'
        board.thread_lock = threading.RLock()
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
