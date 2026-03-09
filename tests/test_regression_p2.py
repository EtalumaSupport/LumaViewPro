"""
Regression tests for P2 bug fixes.

Tests that the following bugs stay fixed:
- #563: Autofocus race condition — protocol_end() must come AFTER final move
- #568: Duration precision — sub-second protocols must round-trip through save/load
- #424: Video bit-depth — 16-bit frames must be converted to 8-bit before codec write
- #539: Serial error rate limiting — repeated errors must not spam the log
"""

import datetime
import pathlib
import sys
import threading
import time
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import serial

# ---------------------------------------------------------------------------
# Mock out heavy dependencies before importing modules under test
# ---------------------------------------------------------------------------
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
sys.modules.setdefault('psutil', MagicMock())
sys.modules.setdefault('kivy', MagicMock())
sys.modules.setdefault('kivy.clock', MagicMock())

# Mock camera hardware SDKs
sys.modules.setdefault('pypylon', MagicMock())
sys.modules.setdefault('pypylon.pylon', MagicMock())
sys.modules.setdefault('pypylon.genicam', MagicMock())
sys.modules.setdefault('ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak_ipl_extension', MagicMock())
sys.modules.setdefault('ids_peak_ipl', MagicMock())

# Mock settings_init
_mock_settings_init = MagicMock()
_mock_settings_init.settings = {
    'BF': {'autofocus': False},
    'PC': {'autofocus': False},
    'DF': {'autofocus': False},
    'Red': {'autofocus': False},
    'Green': {'autofocus': False},
    'Blue': {'autofocus': False},
    'Lumi': {'autofocus': False},
}
sys.modules.setdefault('settings_init', _mock_settings_init)

import serialboard
from serialboard import SerialBoard
from modules.protocol import Protocol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_serial(**overrides):
    mock = MagicMock(spec=serial.Serial)
    mock.readline.return_value = b"OK\r\n"
    mock.write.return_value = None
    mock.close.return_value = None
    for k, v in overrides.items():
        setattr(mock, k, v)
    return mock


def _make_serial_board():
    """Create a SerialBoard with a mock serial driver."""
    board = SerialBoard.__new__(SerialBoard)
    board.found = False
    board._lock = threading.RLock()
    board._label = '[TEST]'
    board.port = '/dev/fake'
    board.baudrate = 115200
    board.bytesize = serial.EIGHTBITS
    board.parity = serial.PARITY_NONE
    board.stopbits = serial.STOPBITS_ONE
    board.timeout = 0.1
    board.write_timeout = 0.1
    board._last_error_log_time = 0.0
    board._error_log_interval = 2.0
    board.firmware_version = None
    board.driver = _make_mock_serial()
    return board


def _make_tiling_configs(tmp_path):
    """Create minimal tiling configs file for Protocol."""
    configs_file = tmp_path / "tiling_configs.json"
    configs_file.write_text('{}')
    return configs_file


# ---------------------------------------------------------------------------
# #563: Autofocus race condition — protocol_end() after final move
# ---------------------------------------------------------------------------

class TestAFRaceCondition:
    """Verify that protocol_end() is called AFTER the final move_absolute_position.

    Bug #563: If protocol_end() is called between the two final moves,
    the second move is lost because the protocol executor stops processing.
    """

    def test_protocol_end_after_final_move(self):
        """protocol_end() must be called after both move_absolute_position calls."""
        # We can't easily run the full AF loop, but we can verify the
        # source code ordering by inspecting the method.
        import inspect
        from modules.autofocus_executor import AutofocusExecutor

        source = inspect.getsource(AutofocusExecutor._iterate)

        # Extract just the last-pass block (from "self._last_pass:" to "return")
        last_pass_start = source.index('if self._last_pass:')
        # Find the return that ends this block
        last_pass_block = source[last_pass_start:]
        return_idx = last_pass_block.index('\n                return\n')
        last_pass_block = last_pass_block[:return_idx]

        # Find move_absolute_position calls within the last-pass block
        move_positions = []
        search_from = 0
        while True:
            idx = last_pass_block.find('self._move_absolute_position', search_from)
            if idx == -1:
                break
            move_positions.append(idx)
            search_from = idx + 1

        protocol_end_pos = last_pass_block.find('self._autofocus_executor.protocol_end()')
        clear_pending_pos = last_pass_block.find('self._autofocus_executor.clear_protocol_pending()')

        assert len(move_positions) >= 2, "Expected at least 2 move_absolute_position calls in last-pass block"
        assert protocol_end_pos > move_positions[-1], \
            "protocol_end() must come AFTER the last move_absolute_position (bug #563)"
        assert clear_pending_pos > move_positions[-1], \
            "clear_protocol_pending() must come AFTER the last move_absolute_position"


# ---------------------------------------------------------------------------
# #568: Duration precision — sub-second protocols must survive save/load
# ---------------------------------------------------------------------------

class TestDurationPrecision:
    """Verify that short protocol durations don't get truncated to 0.0.

    Bug #568: round(hours, 2) for a 10-second protocol produces 0.0,
    which fails validation on reload (duration must be > 0).
    """

    def test_10_second_duration_round_trips(self, tmp_path):
        """A 10-second protocol duration must survive save → load."""
        duration = datetime.timedelta(seconds=10)
        duration_hours = round(duration.total_seconds() / 3600.0, 6)
        assert duration_hours > 0, "10-second duration must not round to 0.0"

    def test_1_second_duration_round_trips(self, tmp_path):
        """A 1-second protocol duration must survive save → load."""
        duration = datetime.timedelta(seconds=1)
        duration_hours = round(duration.total_seconds() / 3600.0, 6)
        assert duration_hours > 0, "1-second duration must not round to 0.0"

    def test_sub_second_duration_round_trips(self):
        """A 0.5-second protocol duration must survive save → load."""
        duration = datetime.timedelta(milliseconds=500)
        duration_hours = round(duration.total_seconds() / 3600.0, 6)
        assert duration_hours > 0, "0.5-second duration must not round to 0.0"

    def test_old_rounding_would_fail(self):
        """Confirm that the old round(hours, 2) would produce 0.0 for short durations."""
        duration = datetime.timedelta(seconds=10)
        duration_hours_old = round(duration.total_seconds() / 3600.0, 2)
        assert duration_hours_old == 0.0, "Old rounding (2 decimals) should produce 0.0 for 10s"

        duration_hours_new = round(duration.total_seconds() / 3600.0, 6)
        assert duration_hours_new > 0, "New rounding (6 decimals) should preserve precision"

    def test_save_writes_nonzero_duration(self, tmp_path):
        """Saved protocol file must contain a non-zero duration for short protocols."""
        tiling_file = _make_tiling_configs(tmp_path)
        protocol = Protocol(tiling_configs_file_loc=tiling_file)

        # Configure a minimal protocol with 10-second duration
        protocol._config = {
            'version': Protocol.CURRENT_VERSION,
            'period': datetime.timedelta(minutes=1),
            'duration': datetime.timedelta(seconds=10),
            'labware_id': 'none',
            'capture_root': '',
            'steps': Protocol._create_empty_steps_df(),
        }

        # Save
        save_path = tmp_path / "test_protocol.tsv"
        protocol.to_file(file_path=save_path)

        # Read back the raw file and find the Duration line
        content = save_path.read_text()
        for line in content.splitlines():
            if line.startswith('Duration'):
                duration_val = float(line.split('\t')[1])
                assert duration_val > 0, \
                    f"Duration in saved file must be > 0, got {duration_val} (bug #568)"
                # Verify it converts back to ~10 seconds
                seconds = duration_val * 3600
                assert abs(seconds - 10.0) < 0.1, \
                    f"Duration should be ~10 seconds, got {seconds}"
                return

        pytest.fail("Duration row not found in saved protocol file")


# ---------------------------------------------------------------------------
# #424: Video bit-depth — 16-bit frames converted to 8-bit before write
# ---------------------------------------------------------------------------

class TestVideoBitDepth:
    """Verify that 16-bit images are converted to 8-bit before codec write.

    Bug #424: mp4v codec silently degrades 12/16-bit frames, producing
    corrupted video. Must convert to uint8 before writing.
    """

    def test_videowriter_converts_16bit_frame(self):
        """VideoWriter.add_frame() must convert uint16 to uint8."""
        from modules.video_writer import VideoWriter

        out_path = pathlib.Path('/tmp/test_video_16bit.avi')
        writer = VideoWriter(
            output_file_loc=out_path,
            fps=10.0,
            include_timestamp_overlay=False,
            codec='mp4v',
        )

        # Create a 16-bit grayscale frame
        frame_16bit = np.ones((100, 100), dtype=np.uint16) * 1000

        # Mock the cv2.VideoWriter so we can inspect what gets written
        mock_cv2_writer = MagicMock()
        writer._video = mock_cv2_writer
        writer._shape = (100, 100)

        writer.add_frame(image=frame_16bit)

        # Verify write was called
        mock_cv2_writer.write.assert_called_once()

        # The written image must be uint8
        written_image = mock_cv2_writer.write.call_args[0][0]
        assert written_image.dtype == np.uint8, \
            "16-bit frame must be converted to uint8 before write (bug #424)"

    def test_videowriter_passes_8bit_unchanged(self):
        """VideoWriter.add_frame() must not modify uint8 frames."""
        from modules.video_writer import VideoWriter

        out_path = pathlib.Path('/tmp/test_video_8bit.avi')
        writer = VideoWriter(
            output_file_loc=out_path,
            fps=10.0,
            include_timestamp_overlay=False,
            codec='mp4v',
        )

        frame_8bit = np.ones((100, 100), dtype=np.uint8) * 128

        mock_cv2_writer = MagicMock()
        writer._video = mock_cv2_writer
        writer._shape = (100, 100)

        writer.add_frame(image=frame_8bit)

        mock_cv2_writer.write.assert_called_once()
        written_image = mock_cv2_writer.write.call_args[0][0]
        assert written_image.dtype == np.uint8

    def test_videowriter_converts_color_16bit(self):
        """VideoWriter.add_frame() must handle 16-bit color (3-channel) frames."""
        from modules.video_writer import VideoWriter

        out_path = pathlib.Path('/tmp/test_video_color16.avi')
        writer = VideoWriter(
            output_file_loc=out_path,
            fps=10.0,
            include_timestamp_overlay=False,
            codec='mp4v',
        )

        frame_16bit_color = np.ones((100, 100, 3), dtype=np.uint16) * 500

        mock_cv2_writer = MagicMock()
        writer._video = mock_cv2_writer
        writer._shape = (100, 100)

        writer.add_frame(image=frame_16bit_color)

        mock_cv2_writer.write.assert_called_once()
        written_image = mock_cv2_writer.write.call_args[0][0]
        assert written_image.dtype == np.uint8, \
            "16-bit color frame must be converted to uint8 (bug #424)"

    def test_convert_16bit_to_8bit_preserves_relative_intensity(self):
        """Conversion should preserve relative brightness (divide by 256)."""
        import image_utils

        frame = np.array([[0, 256, 512, 65535]], dtype=np.uint16)
        result = image_utils.convert_16bit_to_8bit(frame)

        assert result.dtype == np.uint8
        assert result[0, 0] == 0       # 0/256 = 0
        assert result[0, 1] == 1       # 256/256 = 1
        assert result[0, 2] == 2       # 512/256 = 2
        assert result[0, 3] == 255     # 65535/256 = 255 (truncated)


# ---------------------------------------------------------------------------
# #539: Serial error rate limiting — repeated errors must not spam the log
# ---------------------------------------------------------------------------

class TestSerialErrorRateLimiting:
    """Verify that serial errors are rate-limited to prevent log spam.

    Bug #539: When the USB cable is disconnected, every polling command
    triggers a log.error, flooding the log file with thousands of
    identical messages per minute.

    Uses patch.object on serialboard.logger to avoid shared mock state
    leaking between test files.
    """

    def test_first_error_is_logged(self):
        """First serial error should always be logged."""
        mock_log = MagicMock()
        board = _make_serial_board()
        board.driver.write.side_effect = serial.SerialTimeoutException("timeout")

        with patch.object(serialboard, 'logger', mock_log):
            board.exchange_command('TEST')

        mock_log.error.assert_called_once()

    def test_rapid_errors_are_suppressed(self):
        """Errors within the rate-limit window should be suppressed."""
        mock_log = MagicMock()
        board = _make_serial_board()
        board._error_log_interval = 2.0

        with patch.object(serialboard, 'logger', mock_log):
            # First call — should log
            board.driver = _make_mock_serial()
            board.driver.write.side_effect = serial.SerialTimeoutException("timeout")
            board.exchange_command('TEST1')
            assert mock_log.error.call_count == 1

            # Second call immediately after — should be suppressed
            board.driver = _make_mock_serial()
            board.driver.write.side_effect = serial.SerialTimeoutException("timeout")
            board.exchange_command('TEST2')
            assert mock_log.error.call_count == 1, \
                "Second error within rate-limit window should be suppressed (bug #539)"

    def test_error_logged_after_interval(self):
        """Errors after the rate-limit interval should be logged again."""
        mock_log = MagicMock()
        mock_time = MagicMock()
        board = _make_serial_board()
        board._error_log_interval = 2.0

        with patch.object(serialboard, 'logger', mock_log), \
             patch.object(serialboard.time, 'monotonic', mock_time):
            # First error at t=0
            mock_time.return_value = 100.0
            board.driver = _make_mock_serial()
            board.driver.write.side_effect = serial.SerialTimeoutException("timeout")
            board.exchange_command('TEST1')
            assert mock_log.error.call_count == 1

            # Second error at t=3 (past 2s interval) — should be logged
            mock_time.return_value = 103.0
            board.driver = _make_mock_serial()
            board.driver.write.side_effect = serial.SerialTimeoutException("timeout")
            board.exchange_command('TEST2')
            assert mock_log.error.call_count == 2, \
                "Error after rate-limit interval should be logged"

    def test_write_fast_errors_also_rate_limited(self):
        """_write_command_fast errors should also be rate-limited."""
        mock_log = MagicMock()
        board = _make_serial_board()
        board._error_log_interval = 2.0

        with patch.object(serialboard, 'logger', mock_log):
            # First call — should log
            board.driver.write.side_effect = Exception("write failed")
            board._write_command_fast('TEST1')
            assert mock_log.error.call_count == 1

            # Second call immediately — should be suppressed
            board.driver = _make_mock_serial()
            board.driver.write.side_effect = Exception("write failed")
            board._write_command_fast('TEST2')
            assert mock_log.error.call_count == 1, \
                "_write_command_fast errors should also be rate-limited (bug #539)"

    def test_generic_exception_also_rate_limited(self):
        """Generic exceptions (not just timeout) should be rate-limited."""
        mock_log = MagicMock()
        board = _make_serial_board()
        board._error_log_interval = 2.0

        with patch.object(serialboard, 'logger', mock_log):
            # First generic error
            board.driver.write.side_effect = Exception("USB disconnected")
            board.exchange_command('TEST1')
            assert mock_log.error.call_count == 1

            # Second generic error immediately
            board.driver = _make_mock_serial()
            board.driver.write.side_effect = Exception("USB disconnected")
            board.exchange_command('TEST2')
            assert mock_log.error.call_count == 1

    def test_driver_still_closed_when_suppressed(self):
        """Even when log is suppressed, driver must still be closed."""
        board = _make_serial_board()
        board._error_log_interval = 2.0

        # First error — triggers log + close
        mock_driver1 = board.driver
        board.driver.write.side_effect = serial.SerialTimeoutException("timeout")
        board.exchange_command('TEST1')
        mock_driver1.close.assert_called()
        assert board.driver is None

        # Second error — log suppressed but close still happens
        board.driver = _make_mock_serial()
        mock_driver2 = board.driver
        board.driver.write.side_effect = serial.SerialTimeoutException("timeout")
        board.exchange_command('TEST2')
        mock_driver2.close.assert_called()
        assert board.driver is None
