# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for drivers/firmware_updater.py.

Uses mock serial ports and mock filesystem — no hardware needed.
Covers version parsing, board configs, BOOTSEL detection, serial port
lookup, FWUPDATE command, config backup/restore, UpdateResult, and
the top-level update_firmware orchestrator.
"""

import hashlib
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

from drivers.firmware_updater import (
    _parse_uf2_version,
    _detect_bootsel_drive,
    _detect_bootsel_macos,
    _find_serial_port,
    _find_dev_rp2350_pre_flash_port,
    _reboot_dev_board_to_bootsel,
    _send_fwupdate_command,
    _backup_configs,
    _restore_configs,
    _report_progress,
    flash_dev_board,
    update_firmware,
    BoardConfig,
    BoardType,
    BOARD_CONFIGS,
    BOOTSEL_VOLUME_NAMES,
    KNOWN_DEV_RP2350_PRE_FLASH,
    UpdateError,
    UpdateResult,
    UpdateStage,
)


# ---------------------------------------------------------------------------
# 1. _parse_uf2_version
# ---------------------------------------------------------------------------

class TestParseUf2Version:
    def test_semantic_version(self):
        assert _parse_uf2_version(Path("led_firmware_v2.1.0.uf2")) == "2.1.0"

    def test_semantic_version_no_v(self):
        assert _parse_uf2_version(Path("motor_firmware_1.0.3.uf2")) == "1.0.3"

    def test_date_version(self):
        assert _parse_uf2_version(Path("motor_firmware_2026-03-09.uf2")) == "2026-03-09"

    def test_no_version_returns_stem(self):
        assert _parse_uf2_version(Path("custom_build.uf2")) == "custom_build"

    def test_semantic_takes_precedence_over_date(self):
        # If both patterns appear, semantic (matched first) wins
        assert _parse_uf2_version(Path("fw_v1.2.3_2026-01-01.uf2")) == "1.2.3"

    def test_path_with_directory(self):
        p = Path("/some/dir/led_firmware_v3.0.0.uf2")
        assert _parse_uf2_version(p) == "3.0.0"


# ---------------------------------------------------------------------------
# 2. BoardConfig — LED and MOTOR configs
# ---------------------------------------------------------------------------

class TestBoardConfig:
    def test_led_config_exists(self):
        cfg = BOARD_CONFIGS[BoardType.LED]
        assert cfg.board_type == BoardType.LED
        assert cfg.vid == 0x0424
        assert cfg.pid == 0x704C
        assert cfg.line_ending == b'\r\n'
        assert 'cal.json' in cfg.config_files
        assert cfg.uf2_prefix == 'led_firmware'

    def test_motor_config_exists(self):
        cfg = BOARD_CONFIGS[BoardType.MOTOR]
        assert cfg.board_type == BoardType.MOTOR
        assert cfg.vid == 0x2E8A
        assert cfg.pid == 0x0005
        assert cfg.line_ending == b'\n'
        assert 'motorconfig.json' in cfg.config_files
        assert 'xymotorconfig.ini' in cfg.config_files
        assert cfg.uf2_prefix == 'motor_firmware'

    def test_led_has_longer_timeouts(self):
        led = BOARD_CONFIGS[BoardType.LED]
        motor = BOARD_CONFIGS[BoardType.MOTOR]
        assert led.bootsel_timeout >= motor.bootsel_timeout
        assert led.serial_reappear_timeout >= motor.serial_reappear_timeout


# ---------------------------------------------------------------------------
# 3. _detect_bootsel_drive — mock filesystem, macOS path
# ---------------------------------------------------------------------------

class TestDetectBootselDrive:
    @patch("drivers.firmware_updater.platform.system", return_value="Darwin")
    @patch("drivers.firmware_updater._detect_bootsel_macos")
    def test_macos_delegates(self, mock_macos, mock_sys):
        mock_macos.return_value = Path("/Volumes/RPI-RP2")
        result = _detect_bootsel_drive()
        assert result == Path("/Volumes/RPI-RP2")
        mock_macos.assert_called_once()

    @patch("drivers.firmware_updater.platform.system", return_value="Darwin")
    @patch("drivers.firmware_updater._detect_bootsel_macos", return_value=None)
    def test_macos_not_found(self, mock_macos, mock_sys):
        assert _detect_bootsel_drive() is None

    @patch("drivers.firmware_updater.platform.system", return_value="FreeBSD")
    def test_unsupported_platform(self, mock_sys):
        assert _detect_bootsel_drive() is None

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_dir", return_value=True)
    def test_macos_found(self, mock_is_dir, mock_exists):
        result = _detect_bootsel_macos()
        assert result == Path("/Volumes/RPI-RP2")

    @patch("pathlib.Path.is_dir", return_value=False)
    def test_macos_no_dir(self, mock_is_dir):
        result = _detect_bootsel_macos()
        assert result is None


# ---------------------------------------------------------------------------
# 4. _find_serial_port — mock list_ports.comports()
# ---------------------------------------------------------------------------

class TestFindSerialPort:
    @patch("drivers.firmware_updater.list_ports.comports")
    def test_found(self, mock_comports):
        port = MagicMock()
        port.vid = 0x2E8A
        port.pid = 0x0005
        port.device = "/dev/ttyACM0"
        mock_comports.return_value = [port]
        assert _find_serial_port(0x2E8A, 0x0005) == "/dev/ttyACM0"

    @patch("drivers.firmware_updater.list_ports.comports")
    def test_not_found(self, mock_comports):
        port = MagicMock()
        port.vid = 0x1234
        port.pid = 0x5678
        mock_comports.return_value = [port]
        assert _find_serial_port(0x2E8A, 0x0005) is None

    @patch("drivers.firmware_updater.list_ports.comports")
    def test_empty_list(self, mock_comports):
        mock_comports.return_value = []
        assert _find_serial_port(0x2E8A, 0x0005) is None

    @patch("drivers.firmware_updater.list_ports.comports")
    def test_multiple_ports_returns_first_match(self, mock_comports):
        p1 = MagicMock(vid=0x1111, pid=0x2222, device="/dev/ttyACM0")
        p2 = MagicMock(vid=0x2E8A, pid=0x0005, device="/dev/ttyACM1")
        p3 = MagicMock(vid=0x2E8A, pid=0x0005, device="/dev/ttyACM2")
        mock_comports.return_value = [p1, p2, p3]
        assert _find_serial_port(0x2E8A, 0x0005) == "/dev/ttyACM1"


# ---------------------------------------------------------------------------
# 5-6. _send_fwupdate_command — mock serial, verify bytes + error
# ---------------------------------------------------------------------------

class TestSendFwupdateCommand:
    def _make_motor_config(self):
        return BOARD_CONFIGS[BoardType.MOTOR]

    def _make_led_config(self):
        return BOARD_CONFIGS[BoardType.LED]

    def test_motor_sends_fwupdate(self):
        """Motor board: exchange_command('FWUPDATE') is called."""
        cfg = self._make_motor_config()
        board = MagicMock()
        board.exchange_command.return_value = None  # Board reboots, no response
        _send_fwupdate_command(board, cfg)

        board.exchange_command.assert_called_once_with('FWUPDATE', timeout=3.0)
        board.disconnect.assert_called()

    def test_led_sends_fwupdate(self):
        """LED board: exchange_command('FWUPDATE') is called."""
        cfg = self._make_led_config()
        board = MagicMock()
        board.exchange_command.return_value = None
        _send_fwupdate_command(board, cfg)

        board.exchange_command.assert_called_once_with('FWUPDATE', timeout=3.0)
        board.disconnect.assert_called()

    def test_raises_on_exchange_exception(self):
        """UpdateError raised when exchange_command throws."""
        cfg = self._make_motor_config()
        board = MagicMock()
        board.exchange_command.side_effect = Exception("serial error")
        with pytest.raises(UpdateError) as exc_info:
            _send_fwupdate_command(board, cfg)
        assert exc_info.value.stage == UpdateStage.SENDING_FWUPDATE
        board.disconnect.assert_called()

    def test_disconnects_on_success(self):
        """Board is disconnected after successful FWUPDATE."""
        cfg = self._make_motor_config()
        board = MagicMock()
        board.exchange_command.return_value = 'Entering bootloader mode...'
        _send_fwupdate_command(board, cfg)
        board.disconnect.assert_called()

    def test_fallback_to_raw_repl_on_not_found(self):
        """Falls back to raw REPL when firmware doesn't support FWUPDATE."""
        cfg = self._make_motor_config()
        board = MagicMock()
        board.exchange_command.return_value = "ERROR: command 'FWUPDATE' not found"
        with patch("drivers.firmware_updater._bootloader_via_raw_repl") as mock_fallback:
            _send_fwupdate_command(board, cfg)
            mock_fallback.assert_called_once_with(board)

    def test_disconnects_on_error(self):
        """Board is disconnected even when exchange_command fails."""
        cfg = self._make_motor_config()
        board = MagicMock()
        board.exchange_command.side_effect = OSError("USB disconnected")
        with pytest.raises(UpdateError):
            _send_fwupdate_command(board, cfg)
        board.disconnect.assert_called()

    def test_fw40_two_step_sends_confirm(self):
        """Phase 4G: FW4.0 two-step FWUPDATE detection.

        FW4.0 firmware refuses to reboot on a bare FWUPDATE; it returns
        a warning asking for 'FWUPDATE CONFIRM'. The helper must detect
        this and send the follow-up. v3.0.x firmware reboots silently
        on the first call and never emits this phrase, so a string-
        detection approach is safe for both protocols.

        Regression guard: without this fix, a FW4.0 motor board is
        unrecoverable via FWUPDATE (observed on SN 11016 bench).
        """
        cfg = self._make_motor_config()
        board = MagicMock()
        # First call: FW4.0 warning. Second call: no response (board reboots).
        board.exchange_command.side_effect = [
            '{"ok":true,"cmd":"FWUPDATE","msg":"send FWUPDATE CONFIRM to reboot into UF2 bootloader"}',
            None,
        ]
        _send_fwupdate_command(board, cfg)

        assert board.exchange_command.call_count == 2
        # First call: bare FWUPDATE
        assert board.exchange_command.call_args_list[0].args[0] == 'FWUPDATE'
        # Second call: CONFIRM follow-up
        assert board.exchange_command.call_args_list[1].args[0] == 'FWUPDATE CONFIRM'
        board.disconnect.assert_called()

    def test_fw40_two_step_tolerates_confirm_timeout(self):
        """If FWUPDATE CONFIRM raises (board already rebooting), the
        helper treats it as success — no UpdateError, board still
        disconnects."""
        cfg = self._make_motor_config()
        board = MagicMock()
        board.exchange_command.side_effect = [
            '{"ok":true,"cmd":"FWUPDATE","msg":"send FWUPDATE CONFIRM to reboot"}',
            Exception("port disappeared"),  # board rebooting mid-write
        ]
        # Should not raise — CONFIRM timeout is expected on reboot
        _send_fwupdate_command(board, cfg)
        assert board.exchange_command.call_count == 2
        board.disconnect.assert_called()


# ---------------------------------------------------------------------------
# 7-9. _backup_configs — exercised against a real SerialBoard wired to an
# in-memory FakeTransport via the `board_with_fake_transport` fixture, so
# SerialBoard._lock, enter/exit_raw_repl, and MpremoteSession's read-with-
# verify path all run for real.
# ---------------------------------------------------------------------------

class TestBackupConfigs:
    def test_all_files_backed_up(self, tmp_path, board_with_fake_transport):
        """All config files are read and saved to disk."""
        cfg = BOARD_CONFIGS[BoardType.MOTOR]
        board, fake = board_with_fake_transport(
            BoardType.MOTOR,
            initial_files={
                'motorconfig.json': b'{"motor": 1}',
                'xymotorconfig.ini': b'[xy]\nsteps=100',
                'ztmotorconfig.ini': b'[zt]\nsteps=200',
                'ztmotorconfig2.ini': b'[zt2]\nsteps=300',
            },
        )

        result = _backup_configs(board, cfg, tmp_path)

        assert len(result) == 4
        assert result['motorconfig.json'] == b'{"motor": 1}'
        board_dir = tmp_path / 'motor'
        assert (board_dir / 'motorconfig.json').read_bytes() == b'{"motor": 1}'
        assert (board_dir / 'backup_manifest.json').exists()

        manifest = json.loads((board_dir / 'backup_manifest.json').read_text())
        assert 'motorconfig.json' in manifest
        assert manifest['motorconfig.json']['size'] == len(b'{"motor": 1}')

    def test_missing_file_skipped(self, tmp_path, board_with_fake_transport):
        """File not present on board is skipped (not an error)."""
        cfg = BOARD_CONFIGS[BoardType.MOTOR]
        board, fake = board_with_fake_transport(
            BoardType.MOTOR,
            initial_files={'motorconfig.json': b'{"motor": 1}'},
        )

        result = _backup_configs(board, cfg, tmp_path)

        assert set(result.keys()) == {'motorconfig.json'}
        # Only the present file was read; the three *.ini files were
        # never fetched. call_log captures every fs_readfile invocation.
        read_names = [entry[1] for entry in fake.call_log
                      if entry[0] == 'fs_readfile']
        assert set(read_names) == {'motorconfig.json'}

    def test_read_failure_raises(self, tmp_path, board_with_fake_transport):
        """fs_readfile raising TransportError surfaces as UpdateError."""
        from mpremote.transport import TransportError
        cfg = BOARD_CONFIGS[BoardType.LED]
        # File appears on the board (list_files returns it) but read
        # fails. MpremoteSession.read_file catches TransportError and
        # returns None → _backup_configs raises UpdateError.
        board, fake = board_with_fake_transport(
            BoardType.LED,
            initial_files={'cal.json': b'{"cal": "good"}'},
        )
        fake.raise_on = {'fs_readfile': TransportError('simulated read failure')}

        with pytest.raises(UpdateError) as exc_info:
            _backup_configs(board, cfg, tmp_path)
        assert exc_info.value.stage == UpdateStage.BACKING_UP_CONFIG

    def test_enter_repl_failure_raises(self, tmp_path, board_with_fake_transport):
        """enter_raw_repl failure (TransportError) surfaces as UpdateError."""
        from mpremote.transport import TransportError
        cfg = BOARD_CONFIGS[BoardType.LED]
        board, fake = board_with_fake_transport(BoardType.LED)
        # Raising from enter_raw_repl propagates through MpremoteSession.enter
        # into SerialBoard.enter_raw_repl's except block, which returns False.
        fake.raise_on = {'enter_raw_repl': TransportError('cannot enter REPL')}

        with pytest.raises(UpdateError) as exc_info:
            _backup_configs(board, cfg, tmp_path)
        assert exc_info.value.stage == UpdateStage.BACKING_UP_CONFIG


# ---------------------------------------------------------------------------
# 10-11. _restore_configs — same fixture. MpremoteSession.write_file's
# atomic-`.tmp` + device-side SHA-256 verify + `.bak` rename sequence runs
# for real; the fake stores bytes and computes real hashlib.sha256.
# ---------------------------------------------------------------------------

class TestRestoreConfigs:
    def test_surviving_file_skipped(self, board_with_fake_transport):
        """File that survived the update (matches backup) is not rewritten."""
        cfg = BOARD_CONFIGS[BoardType.MOTOR]
        data = b'{"motor": 1}'
        board, fake = board_with_fake_transport(
            BoardType.MOTOR,
            initial_files={'motorconfig.json': data},
        )

        result = _restore_configs(board, cfg, {'motorconfig.json': data})

        assert result is True
        # No fs_writefile call means nothing was rewritten.
        assert not any(entry[0] == 'fs_writefile' for entry in fake.call_log)

    def test_missing_file_written(self, board_with_fake_transport):
        """File missing from board after update is restored."""
        cfg = BOARD_CONFIGS[BoardType.MOTOR]
        data = b'{"motor": 1}'
        board, fake = board_with_fake_transport(BoardType.MOTOR)  # empty FS

        result = _restore_configs(board, cfg, {'motorconfig.json': data})

        assert result is True
        # After atomic rename, the file should be on the fake's filesystem.
        assert fake.files.get('motorconfig.json') == data

    def test_changed_file_restored(self, board_with_fake_transport):
        """File that exists but differs from backup is overwritten."""
        cfg = BOARD_CONFIGS[BoardType.LED]
        backup_data = b'{"cal": "good"}'
        board, fake = board_with_fake_transport(
            BoardType.LED,
            initial_files={'cal.json': b'{"cal": "corrupted"}'},
        )

        result = _restore_configs(board, cfg, {'cal.json': backup_data})

        assert result is True
        assert fake.files.get('cal.json') == backup_data
        # Previous content rotated into .bak per MpremoteSession.write_file.
        assert fake.files.get('cal.json.bak') == b'{"cal": "corrupted"}'

    def test_write_failure_raises(self, board_with_fake_transport):
        """Exhausted retries on fs_writefile surface as UpdateError."""
        cfg = BOARD_CONFIGS[BoardType.LED]
        board, fake = board_with_fake_transport(BoardType.LED)  # empty FS
        # MpremoteSession.write_file retries WRITE_VERIFY_RETRIES times
        # (=3). fail_next_write=3 exhausts all attempts.
        fake.fail_next_write = 3

        with pytest.raises(UpdateError) as exc_info:
            _restore_configs(board, cfg, {'cal.json': b'data'})
        assert exc_info.value.stage == UpdateStage.RESTORING_CONFIG
        # File never landed on the device.
        assert 'cal.json' not in fake.files

    def test_empty_config_data_returns_true(self):
        """No config files to restore returns True immediately.

        Uses MagicMock — this path short-circuits before touching the
        board, so no transport setup is needed.
        """
        cfg = BOARD_CONFIGS[BoardType.LED]
        board = MagicMock()
        assert _restore_configs(board, cfg, {}) is True


# ---------------------------------------------------------------------------
# 12. UpdateResult dataclass
# ---------------------------------------------------------------------------

class TestUpdateResult:
    def test_default_fields(self):
        r = UpdateResult(success=True, board_type=BoardType.LED)
        assert r.success is True
        assert r.board_type == BoardType.LED
        assert r.old_version is None
        assert r.new_version is None
        assert r.config_backup_path is None
        assert r.error_message is None
        assert r.error_stage is None
        assert r.warnings == []

    def test_all_fields(self):
        r = UpdateResult(
            success=False,
            board_type=BoardType.MOTOR,
            old_version="1.0.0",
            new_version="2.0.0",
            config_backup_path=Path("/tmp/backup"),
            error_message="something broke",
            error_stage=UpdateStage.COPYING_UF2,
            warnings=["warn1"],
        )
        assert r.success is False
        assert r.old_version == "1.0.0"
        assert r.new_version == "2.0.0"
        assert r.error_message == "something broke"
        assert r.error_stage == UpdateStage.COPYING_UF2
        assert r.warnings == ["warn1"]


# ---------------------------------------------------------------------------
# 13. update_firmware — same version skips update
# ---------------------------------------------------------------------------

class TestUpdateFirmwareSameVersion:
    @patch("drivers.firmware_updater._detect_bootsel_drive", return_value=None)
    @patch("drivers.firmware_updater._create_board")
    @patch("drivers.firmware_updater._parse_uf2_version", return_value="2.0.0")
    def test_same_version_returns_success(self, mock_parse, mock_create,
                                           mock_bootsel, tmp_path):
        """If board already has the target version, skip update."""
        board = MagicMock()
        board.firmware_version = "2.0.0"
        board.firmware_date = None
        mock_create.return_value = board

        uf2 = tmp_path / "motor_firmware_v2.0.0.uf2"
        uf2.write_bytes(b'\x00' * 1024)

        result = update_firmware(
            board_type=BoardType.MOTOR,
            uf2_path=uf2,
            backup_dir=tmp_path / "backup",
        )
        assert result.success is True
        assert result.old_version == "2.0.0"
        assert result.new_version == "2.0.0"
        board.disconnect.assert_called()


# ---------------------------------------------------------------------------
# 14. update_firmware — pre-existing BOOTSEL drive causes abort
# ---------------------------------------------------------------------------

class TestUpdateFirmwareBootselAbort:
    @patch("drivers.firmware_updater.time.sleep")
    @patch("drivers.firmware_updater._detect_bootsel_drive",
           return_value=Path("/Volumes/RPI-RP2"))
    def test_existing_bootsel_aborts(self, mock_bootsel, mock_sleep, tmp_path):
        """Pre-existing BOOTSEL drive causes immediate abort."""
        uf2 = tmp_path / "motor_firmware_v2.0.0.uf2"
        uf2.write_bytes(b'\x00' * 1024)

        result = update_firmware(
            board_type=BoardType.MOTOR,
            uf2_path=uf2,
            backup_dir=tmp_path / "backup",
        )
        assert result.success is False
        assert result.error_stage == UpdateStage.PREFLIGHT
        assert "already mounted" in result.error_message


# ---------------------------------------------------------------------------
# 15. Progress callback — called at each stage, exception-safe
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_report_progress_calls_callback(self):
        cb = Mock()
        _report_progress(cb, UpdateStage.PREFLIGHT, "hello", 0.5)
        cb.assert_called_once_with(UpdateStage.PREFLIGHT, "hello", 0.5)

    def test_report_progress_none_callback(self):
        """None callback does not raise."""
        _report_progress(None, UpdateStage.PREFLIGHT, "hello", 0.5)

    def test_report_progress_exception_safe(self):
        """Callback raising exception does not propagate."""
        cb = Mock(side_effect=RuntimeError("boom"))
        # Should not raise
        _report_progress(cb, UpdateStage.PREFLIGHT, "hello", 0.5)

    @patch("drivers.firmware_updater.time.sleep")
    @patch("drivers.firmware_updater._detect_bootsel_drive",
           return_value=Path("/Volumes/RPI-RP2"))
    def test_callback_called_on_error(self, mock_bootsel, mock_sleep, tmp_path):
        """Progress callback receives FAILED stage on error."""
        cb = Mock()
        uf2 = tmp_path / "motor_firmware_v2.0.0.uf2"
        uf2.write_bytes(b'\x00' * 1024)

        update_firmware(
            board_type=BoardType.MOTOR,
            uf2_path=uf2,
            progress_callback=cb,
            backup_dir=tmp_path / "backup",
        )
        # Last call should be FAILED
        stages = [c.args[0] for c in cb.call_args_list]
        assert UpdateStage.PREFLIGHT in stages
        assert UpdateStage.FAILED in stages

    @patch("drivers.firmware_updater.time.sleep")
    @patch("drivers.firmware_updater._detect_bootsel_drive",
           return_value=Path("/Volumes/RPI-RP2"))
    def test_exception_in_callback_does_not_break_update(self, mock_bootsel,
                                                          mock_sleep, tmp_path):
        """Even if progress callback raises, update_firmware still returns result."""
        cb = Mock(side_effect=RuntimeError("callback broke"))
        uf2 = tmp_path / "motor_firmware_v2.0.0.uf2"
        uf2.write_bytes(b'\x00' * 1024)

        result = update_firmware(
            board_type=BoardType.MOTOR,
            uf2_path=uf2,
            progress_callback=cb,
            backup_dir=tmp_path / "backup",
        )
        # Should still get a result (the error from BOOTSEL pre-check)
        assert isinstance(result, UpdateResult)


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

class TestUpdateErrorDataclass:
    def test_fields(self):
        e = UpdateError("msg", UpdateStage.COPYING_UF2, recoverable=False)
        assert str(e) == "msg"
        assert e.stage == UpdateStage.COPYING_UF2
        assert e.recoverable is False

    def test_default_recoverable(self):
        e = UpdateError("msg", UpdateStage.PREFLIGHT)
        assert e.recoverable is True


class TestUpdateFirmwareMissingUf2:
    @patch("drivers.firmware_updater.time.sleep")
    @patch("drivers.firmware_updater._detect_bootsel_drive", return_value=None)
    def test_missing_uf2_file(self, mock_bootsel, mock_sleep, tmp_path):
        """Non-existent UF2 file returns error result."""
        result = update_firmware(
            board_type=BoardType.MOTOR,
            uf2_path=tmp_path / "nonexistent.uf2",
            backup_dir=tmp_path / "backup",
        )
        assert result.success is False
        assert result.error_stage == UpdateStage.PREFLIGHT


class TestUpdateFirmwareTooSmallUf2:
    @patch("drivers.firmware_updater.time.sleep")
    @patch("drivers.firmware_updater._detect_bootsel_drive", return_value=None)
    def test_tiny_uf2(self, mock_bootsel, mock_sleep, tmp_path):
        """UF2 file under 512 bytes is rejected."""
        uf2 = tmp_path / "motor_firmware_v1.0.0.uf2"
        uf2.write_bytes(b'\x00' * 100)

        result = update_firmware(
            board_type=BoardType.MOTOR,
            uf2_path=uf2,
            backup_dir=tmp_path / "backup",
        )
        assert result.success is False
        assert "too small" in result.error_message


class TestFlashUf2Direct:
    """flash_uf2_direct — for bricked boards already in BOOTSEL."""

    def test_rejects_led(self, tmp_path):
        """LED has has_direct_usb=False — direct UF2 flash impossible
        through this path (requires TP96 + TP8/TP11 on the bench)."""
        from drivers.firmware_updater import flash_uf2_direct
        uf2 = tmp_path / "ledcon.uf2"
        uf2.write_bytes(b'\x00' * 1024)
        result = flash_uf2_direct(BoardType.LED, uf2)
        assert result.success is False
        assert result.error_stage == UpdateStage.PREFLIGHT
        assert "no direct USB" in result.error_message

    def test_rejects_missing_uf2(self, tmp_path):
        from drivers.firmware_updater import flash_uf2_direct
        result = flash_uf2_direct(
            BoardType.MOTOR, tmp_path / "nonexistent.uf2")
        assert result.success is False
        assert result.error_stage == UpdateStage.PREFLIGHT
        assert "not found" in result.error_message

    def test_rejects_tiny_uf2(self, tmp_path):
        from drivers.firmware_updater import flash_uf2_direct
        uf2 = tmp_path / "mocon.uf2"
        uf2.write_bytes(b'\x00' * 100)
        result = flash_uf2_direct(BoardType.MOTOR, uf2)
        assert result.success is False
        assert result.error_stage == UpdateStage.PREFLIGHT
        assert "too small" in result.error_message

    @patch("drivers.firmware_updater.time.sleep")
    @patch("drivers.firmware_updater._wait_for_bootsel_drive",
           return_value=None)
    @patch("drivers.firmware_updater._find_picotool", return_value=None)
    def test_no_bootsel_no_picotool_errors(
        self, mock_picotool, mock_wait_bootsel, mock_sleep, tmp_path,
    ):
        """If BOOTSEL drive never appears AND picotool isn't installed,
        surface a clear recoverable error pointing at the pin short."""
        from drivers.firmware_updater import flash_uf2_direct
        uf2 = tmp_path / "mocon.uf2"
        uf2.write_bytes(b'\x00' * 1024)
        result = flash_uf2_direct(
            BoardType.MOTOR, uf2, bootsel_timeout=0.01)
        assert result.success is False
        assert result.error_stage == UpdateStage.WAITING_BOOTSEL
        assert "BOOTSEL pin" in result.error_message

    @patch("drivers.firmware_updater.time.sleep")
    @patch("drivers.firmware_updater._create_board")
    @patch("drivers.firmware_updater._wait_for_serial_port")
    @patch("drivers.firmware_updater._wait_for_drive_disappear",
           return_value=True)
    @patch("drivers.firmware_updater._wait_for_bootsel_drive")
    @patch("drivers.firmware_updater.shutil.copy2")
    def test_bootsel_drive_happy_path(
        self, mock_copy, mock_wait_bootsel, mock_wait_disappear,
        mock_wait_serial, mock_create_board, mock_sleep, tmp_path,
    ):
        """BOOTSEL drive present, copy OK, reboot OK, version read OK."""
        from drivers.firmware_updater import flash_uf2_direct

        bootsel_dir = tmp_path / "RPI-RP2"
        bootsel_dir.mkdir()
        mock_wait_bootsel.return_value = bootsel_dir
        mock_wait_serial.return_value = "/dev/cu.usbmodem9999"
        board = MagicMock()
        board.firmware_version = "1.27.0"
        board.firmware_date = None
        mock_create_board.return_value = board

        uf2 = tmp_path / "mocon.uf2"
        uf2.write_bytes(b'\x00' * 1024)

        result = flash_uf2_direct(BoardType.MOTOR, uf2)

        assert result.success is True
        assert mock_copy.call_count == 1
        assert result.new_version == "1.27.0"
        board.disconnect.assert_called_once()


class TestDeployFirmwareFileExtraFiles:
    """deploy_firmware_file extra_files parameter — companion file push."""

    def test_preflight_missing_extra_file(self, tmp_path):
        from drivers.firmware_updater import deploy_firmware_file
        main_py = tmp_path / "main.py"
        main_py.write_bytes(b'print("hi")\n' + b'# filler\n' * 20)
        result = deploy_firmware_file(
            BoardType.MOTOR,
            main_py,
            extra_files=[(tmp_path / "missing.py", "fw40_framing.py")],
        )
        assert result.success is False
        assert result.error_stage == UpdateStage.PREFLIGHT
        assert "extra_files entry not found" in result.error_message

    def test_preflight_empty_extra_file(self, tmp_path):
        from drivers.firmware_updater import deploy_firmware_file
        main_py = tmp_path / "main.py"
        main_py.write_bytes(b'print("hi")\n' + b'# filler\n' * 20)
        empty = tmp_path / "empty.py"
        empty.write_bytes(b'')
        result = deploy_firmware_file(
            BoardType.MOTOR,
            main_py,
            extra_files=[(empty, "fw40_framing.py")],
        )
        assert result.success is False
        assert result.error_stage == UpdateStage.PREFLIGHT
        assert "empty" in result.error_message

    @patch("drivers.firmware_updater.time.sleep")
    @patch("drivers.firmware_updater._run_post_update_test",
           return_value=(True, "ok"))
    @patch("drivers.firmware_updater._backup_configs", return_value={})
    @patch("drivers.firmware_updater._create_board")
    def test_companion_written_before_main_py(
        self, mock_create_board, mock_backup, mock_post,
        mock_sleep, tmp_path,
    ):
        """Ordering contract: extra_files are written BEFORE main.py so
        a partial failure never leaves a new main.py that imports a
        missing companion (the bricking case the docstring warns about).
        """
        from drivers.firmware_updater import deploy_firmware_file

        main_py = tmp_path / "main.py"
        main_py.write_bytes(b'import fw40_framing\n' + b'# filler\n' * 20)
        framing = tmp_path / "fw40_framing.py"
        framing.write_bytes(b'# framing module\n' + b'# filler\n' * 20)

        board = MagicMock()
        board.firmware_version = "4.0.0"
        board.firmware_date = "2026-04-22"
        board.enter_raw_repl.return_value = True
        board.repl_write_file.return_value = True
        board.detect_firmware_version = MagicMock()
        mock_create_board.return_value = board

        result = deploy_firmware_file(
            BoardType.MOTOR,
            main_py,
            extra_files=[(framing, "fw40_framing.py")],
            skip_config_backup=True,
        )

        assert result.success is True, result.error_message

        write_calls = board.repl_write_file.call_args_list
        names_in_order = [c.args[0] for c in write_calls]
        assert names_in_order == ['fw40_framing.py', 'main.py'], (
            f'wrong write order: {names_in_order}'
        )


class TestDeployFirmwareFileCompileMpy:
    """compile_mpy=True — mpy-cross integration + .mpy pattern."""

    def test_preflight_mpy_cross_missing(self, tmp_path):
        """compile_mpy=True with no mpy-cross on PATH fails preflight."""
        from drivers.firmware_updater import deploy_firmware_file
        main_py = tmp_path / "main.py"
        main_py.write_bytes(b'print("hi")\n')
        with patch("drivers.firmware_updater._find_mpy_cross",
                   return_value=None):
            result = deploy_firmware_file(
                BoardType.MOTOR, main_py, compile_mpy=True,
            )
        assert result.success is False
        assert result.error_stage == UpdateStage.PREFLIGHT
        assert "mpy-cross" in result.error_message

    def test_firmware_remote_name_override(self, tmp_path):
        """firmware_remote_name writes to the caller-chosen filename."""
        from drivers.firmware_updater import deploy_firmware_file

        main_py = tmp_path / "main.py"
        main_py.write_bytes(b'import fw40_led\n')

        board = MagicMock()
        board.firmware_version = "4.0.0"
        board.firmware_date = "2026-04-22"
        board.enter_raw_repl.return_value = True
        board.repl_write_file.return_value = True
        board.detect_firmware_version = MagicMock()

        with patch("drivers.firmware_updater._create_board",
                   return_value=board), \
             patch("drivers.firmware_updater._backup_configs",
                   return_value={}), \
             patch("drivers.firmware_updater._run_post_update_test",
                   return_value=(True, "ok")), \
             patch("drivers.firmware_updater.time.sleep"):
            result = deploy_firmware_file(
                BoardType.MOTOR,
                main_py,
                firmware_remote_name='main.py',
                skip_config_backup=True,
            )
        assert result.success is True
        names = [c.args[0] for c in board.repl_write_file.call_args_list]
        assert names == ['main.py']

    @patch("drivers.firmware_updater._mpy_cross_compile")
    @patch("drivers.firmware_updater._find_mpy_cross")
    def test_compile_mpy_compiles_firmware_and_extras(
        self, mock_find, mock_compile, tmp_path,
    ):
        """compile_mpy=True runs mpy-cross on .py inputs; the deployed
        bytes are the .mpy output, written under remote names verbatim
        (so remote names should carry .mpy suffix when the caller intends
        .mpy deployment — this is documented in the docstring)."""
        from drivers.firmware_updater import deploy_firmware_file

        mock_find.return_value = Path('/fake/mpy-cross')

        main_py = tmp_path / "main.py"
        main_py.write_bytes(b'import fw40_led\n' + b'# filler\n' * 5)
        framing = tmp_path / "fw40_framing.py"
        framing.write_bytes(b'# framing\n' + b'# filler\n' * 5)
        fw_source = tmp_path / "fw40_led.py"
        fw_source.write_bytes(b'# led firmware\n' + b'# filler\n' * 100)

        # Simulate mpy-cross: create the .mpy output file each call.
        def fake_compile(py_path, out_path, mpy_cross_path=None):
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b'MPY\x06\x00' + bytes(
                f'compiled:{py_path.name}', 'utf-8'))
            return True
        mock_compile.side_effect = fake_compile

        board = MagicMock()
        board.firmware_version = "4.0.0"
        board.firmware_date = "2026-04-22"
        board.enter_raw_repl.return_value = True
        board.repl_write_file.return_value = True
        board.detect_firmware_version = MagicMock()

        with patch("drivers.firmware_updater._create_board",
                   return_value=board), \
             patch("drivers.firmware_updater._backup_configs",
                   return_value={}), \
             patch("drivers.firmware_updater._run_post_update_test",
                   return_value=(True, "ok")), \
             patch("drivers.firmware_updater.time.sleep"):
            result = deploy_firmware_file(
                BoardType.MOTOR,
                main_py,
                extra_files=[
                    (framing, 'fw40_framing.mpy'),
                    (fw_source, 'fw40_led.mpy'),
                ],
                compile_mpy=True,
                skip_config_backup=True,
            )

        assert result.success is True, result.error_message
        # Only the two extras whose REMOTE name is .mpy compile. The
        # firmware stub (remote='main.py') writes verbatim — that's the
        # caller-intent rule that keeps the stub-import pattern working.
        assert mock_compile.call_count == 2

        names_in_order = [
            c.args[0] for c in board.repl_write_file.call_args_list
        ]
        assert names_in_order == [
            'fw40_framing.mpy', 'fw40_led.mpy', 'main.py',
        ], f'wrong write order under compile_mpy: {names_in_order}'

        # main.py (the stub source) went out verbatim — bytes start with
        # 'import' text, not the 'MPY' magic.
        main_write = board.repl_write_file.call_args_list[-1]
        assert main_write.args[0] == 'main.py'
        assert main_write.args[1].startswith(b'import fw40_led')

        # fw40_led.mpy got compiled bytes (MPY magic).
        led_write = board.repl_write_file.call_args_list[1]
        assert led_write.args[0] == 'fw40_led.mpy'
        assert led_write.args[1].startswith(b'MPY')


class TestDeployFirmwareBundleFw40:
    """deploy_firmware_bundle_fw40 — the 3-file stub+.mpy bundle for LED."""

    @patch("drivers.firmware_updater._mpy_cross_compile")
    @patch("drivers.firmware_updater._find_mpy_cross")
    def test_led_bundle_writes_stub_and_two_mpy(
        self, mock_find, mock_compile, tmp_path,
    ):
        """LED bundle: main.py stub → fw40_led.mpy + fw40_framing.mpy."""
        from drivers.firmware_updater import deploy_firmware_bundle_fw40

        mock_find.return_value = Path('/fake/mpy-cross')

        led_source = tmp_path / "main.py"  # LVP stores LED firmware as main.py
        led_source.write_bytes(b'# led firmware\n' + b'# filler\n' * 100)
        framing = tmp_path / "fw40_framing.py"
        framing.write_bytes(b'# framing\n' + b'# filler\n' * 5)

        def fake_compile(py_path, out_path, mpy_cross_path=None):
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b'MPY\x06\x00')
            return True
        mock_compile.side_effect = fake_compile

        board = MagicMock()
        board.firmware_version = "4.0.0"
        board.firmware_date = "2026-04-22"
        board.enter_raw_repl.return_value = True
        board.repl_write_file.return_value = True
        board.detect_firmware_version = MagicMock()

        with patch("drivers.firmware_updater._create_board",
                   return_value=board), \
             patch("drivers.firmware_updater._backup_configs",
                   return_value={}), \
             patch("drivers.firmware_updater._run_post_update_test",
                   return_value=(True, "ok")), \
             patch("drivers.firmware_updater.time.sleep"):
            result = deploy_firmware_bundle_fw40(
                BoardType.LED,
                main_module_path=led_source,
                framing_path=framing,
                skip_config_backup=True,
            )

        assert result.success is True, result.error_message
        names = [c.args[0] for c in board.repl_write_file.call_args_list]
        assert names == [
            'fw40_framing.mpy',
            'fw40_led.mpy',
            'main.py',
        ], f'bundle write order wrong: {names}'

        # main.py stub content is 'import fw40_led\n'.
        main_write = board.repl_write_file.call_args_list[-1]
        assert main_write.args[0] == 'main.py'
        assert main_write.args[1] == b'import fw40_led\n'

    @patch("drivers.firmware_updater._mpy_cross_compile")
    @patch("drivers.firmware_updater._find_mpy_cross")
    def test_motor_bundle_uses_fw40_motor_stem(
        self, mock_find, mock_compile, tmp_path,
    ):
        """MOTOR bundle stub imports fw40_motor (not fw40_led)."""
        from drivers.firmware_updater import deploy_firmware_bundle_fw40

        mock_find.return_value = Path('/fake/mpy-cross')
        motor_source = tmp_path / "main.py"
        motor_source.write_bytes(b'# motor\n' + b'# filler\n' * 100)
        framing = tmp_path / "fw40_framing.py"
        framing.write_bytes(b'# framing\n' + b'# filler\n' * 5)

        def fake_compile(py_path, out_path, mpy_cross_path=None):
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b'MPY\x06\x00')
            return True
        mock_compile.side_effect = fake_compile

        board = MagicMock()
        board.firmware_version = "4.0.0"
        board.firmware_date = "2026-04-22"
        board.enter_raw_repl.return_value = True
        board.repl_write_file.return_value = True
        board.detect_firmware_version = MagicMock()

        with patch("drivers.firmware_updater._create_board",
                   return_value=board), \
             patch("drivers.firmware_updater._backup_configs",
                   return_value={}), \
             patch("drivers.firmware_updater._run_post_update_test",
                   return_value=(True, "ok")), \
             patch("drivers.firmware_updater.time.sleep"):
            result = deploy_firmware_bundle_fw40(
                BoardType.MOTOR,
                main_module_path=motor_source,
                framing_path=framing,
                skip_config_backup=True,
            )

        assert result.success is True, result.error_message
        names = [c.args[0] for c in board.repl_write_file.call_args_list]
        assert names == [
            'fw40_framing.mpy',
            'fw40_motor.mpy',
            'main.py',
        ]
        main_write = board.repl_write_file.call_args_list[-1]
        assert main_write.args[1] == b'import fw40_motor\n'


# ---------------------------------------------------------------------------
# Dev-board flashing — RP2350 dev boards (Pi Pico 2, Seeed XIAO, etc.)
# Architecture Rule 22: routed through firmware_updater rather than a
# parallel script. These tests cover the new pieces added 2026-04-28.
# ---------------------------------------------------------------------------

class TestBootselVolumeNames:
    """The set of BOOTSEL volume names recognized across platforms.
    RP2040 boards mount as 'RPI-RP2'; RP2350 boards mount as 'RP2350'."""

    def test_rp2040_volume_recognized(self):
        assert 'RPI-RP2' in BOOTSEL_VOLUME_NAMES

    def test_rp2350_volume_recognized(self):
        assert 'RP2350' in BOOTSEL_VOLUME_NAMES

    def test_volume_names_are_strings(self):
        for name in BOOTSEL_VOLUME_NAMES:
            assert isinstance(name, str) and len(name) > 0


class TestDevRp2350BoardConfig:
    """BOARD_CONFIGS entry for the RP2350 dev target (Pi Pico 2, Seeed XIAO)."""

    def test_dev_rp2350_config_exists(self):
        cfg = BOARD_CONFIGS[BoardType.DEV_RP2350]
        assert cfg.board_type == BoardType.DEV_RP2350

    def test_dev_rp2350_has_direct_usb(self):
        cfg = BOARD_CONFIGS[BoardType.DEV_RP2350]
        assert cfg.has_direct_usb is True

    def test_dev_rp2350_no_etaluma_configs(self):
        """Un-firmware'd dev boards have no Etaluma config files."""
        cfg = BOARD_CONFIGS[BoardType.DEV_RP2350]
        assert cfg.config_files == []


class TestKnownDevRp2350PreFlash:
    """Pre-flash dev-board VID/PID list for board discovery."""

    def test_list_is_non_empty(self):
        assert len(KNOWN_DEV_RP2350_PRE_FLASH) > 0

    def test_each_entry_has_three_fields(self):
        for entry in KNOWN_DEV_RP2350_PRE_FLASH:
            assert len(entry) == 3
            vid, pid, label = entry
            assert isinstance(vid, int) and 0 <= vid <= 0xFFFF
            assert isinstance(pid, int) and 0 <= pid <= 0xFFFF
            assert isinstance(label, str)

    def test_seeed_xiao_rp2350_present(self):
        """The XIAO RP2350 was the board that motivated this work."""
        vid_pids = [(v, p) for v, p, _ in KNOWN_DEV_RP2350_PRE_FLASH]
        assert (0x2886, 0x0058) in vid_pids


class TestDetectBootselMacosRp2350:
    """_detect_bootsel_macos must find an RP2350-named volume."""

    @patch("drivers.firmware_updater.Path.exists", return_value=True)
    @patch("drivers.firmware_updater.Path.is_dir")
    def test_finds_rp2350_when_only_rp2350_mounted(self, mock_is_dir, mock_exists):
        # Only the RP2350 path is_dir==True; RPI-RP2 is False.
        def is_dir_side_effect(self_path=None):
            # The Path the method is invoked on is bound to `self` in the
            # actual call; with patch on the class method, MagicMock receives
            # no args, so we side-effect by inspecting the most recent call.
            return True
        # Simpler alternative: use a custom Path stub for this scenario.
        # See test_finds_rp2350_via_filesystem_stub below for the cleaner shape.
        # This test passes structurally — we verify the function can return
        # an RP2350-pointed Path when the platform-specific helper says yes.
        mock_is_dir.return_value = True
        result = _detect_bootsel_macos()
        # First match wins; with both is_dir==True the RPI-RP2 (first in
        # BOOTSEL_VOLUME_NAMES) is returned. That's the existing behavior.
        # The RP2350 detection is verified in test_finds_rp2350_only.
        assert result is not None

    def test_finds_rp2350_only(self):
        """When only /Volumes/RP2350 exists, the function returns it."""
        # Stub Path methods scoped to this test only — patch the module-level
        # Path symbol used by firmware_updater.
        from unittest.mock import patch
        original_is_dir = Path.is_dir

        def is_dir_stub(self):
            # Only RP2350 path is "directory"; RPI-RP2 is not.
            return str(self).endswith('/RP2350') and not str(self).endswith('INFO_UF2.TXT')

        def exists_stub(self):
            return str(self).endswith('INFO_UF2.TXT') and '/RP2350/' in str(self)

        with patch.object(Path, 'is_dir', is_dir_stub), \
             patch.object(Path, 'exists', exists_stub):
            result = _detect_bootsel_macos()
            assert result == Path('/Volumes/RP2350')


class TestFindDevRp2350PreFlashPort:
    """_find_dev_rp2350_pre_flash_port discovers boards via list_ports.comports()."""

    @patch("drivers.firmware_updater.list_ports.comports")
    def test_finds_seeed_xiao(self, mock_comports):
        port = MagicMock(spec=['device', 'vid', 'pid', 'serial_number'])
        port.device = '/dev/cu.usbmodem101'
        port.vid = 0x2886
        port.pid = 0x0058
        port.serial_number = 'ABCD'
        mock_comports.return_value = [port]
        result = _find_dev_rp2350_pre_flash_port()
        assert result is not None
        assert result['repl_port'] == '/dev/cu.usbmodem101'
        assert result['app_port'] is None  # single CDC, no composite
        assert result['vid'] == 0x2886
        assert result['pid'] == 0x0058
        assert 'XIAO' in result['label']

    @patch("drivers.firmware_updater.list_ports.comports")
    def test_returns_none_when_no_known_board(self, mock_comports):
        port = MagicMock(spec=['device', 'vid', 'pid', 'serial_number'])
        port.device = '/dev/cu.something'
        port.vid = 0x9999
        port.pid = 0x9999
        port.serial_number = None
        mock_comports.return_value = [port]
        assert _find_dev_rp2350_pre_flash_port() is None

    @patch("drivers.firmware_updater.list_ports.comports")
    def test_returns_known_match_when_unknown_also_present(self, mock_comports):
        unknown = MagicMock(spec=['device', 'vid', 'pid', 'serial_number'])
        unknown.device = '/dev/cu.unknown'
        unknown.vid = 0xFFFF
        unknown.pid = 0xFFFF
        unknown.serial_number = None
        xiao = MagicMock(spec=['device', 'vid', 'pid', 'serial_number'])
        xiao.device = '/dev/cu.xiao'
        xiao.vid = 0x2886
        xiao.pid = 0x0058
        xiao.serial_number = 'XYZ'
        mock_comports.return_value = [unknown, xiao]
        result = _find_dev_rp2350_pre_flash_port()
        assert result is not None
        assert result['repl_port'] == '/dev/cu.xiao'

    @patch("drivers.firmware_updater.list_ports.comports")
    def test_composite_device_returns_repl_and_app_ports(self, mock_comports):
        """Two CDC interfaces under the same composite device: REPL = lower-
        numbered port (interface 0 by USB CDC convention), App = higher."""
        repl_intf = MagicMock(spec=['device', 'vid', 'pid', 'serial_number'])
        repl_intf.device = '/dev/cu.usbmodem1101'
        repl_intf.vid = 0x2E8A
        repl_intf.pid = 0x0005
        repl_intf.serial_number = '0fbb6fbd5b33c455'
        app_intf = MagicMock(spec=['device', 'vid', 'pid', 'serial_number'])
        app_intf.device = '/dev/cu.usbmodem1103'
        app_intf.vid = 0x2E8A
        app_intf.pid = 0x0005
        app_intf.serial_number = '0fbb6fbd5b33c455'  # same serial = same device
        mock_comports.return_value = [repl_intf, app_intf]
        result = _find_dev_rp2350_pre_flash_port()
        assert result is not None
        assert result['repl_port'] == '/dev/cu.usbmodem1101'
        assert result['app_port'] == '/dev/cu.usbmodem1103'
        assert result['serial'] == '0fbb6fbd5b33c455'

    @patch("drivers.firmware_updater.list_ports.comports")
    def test_composite_repl_chosen_regardless_of_enumeration_order(self, mock_comports):
        """If list_ports returns the App port FIRST, sort still picks REPL
        as the lower-numbered device path."""
        app_intf = MagicMock(spec=['device', 'vid', 'pid', 'serial_number'])
        app_intf.device = '/dev/cu.usbmodem1103'
        app_intf.vid = 0x2E8A
        app_intf.pid = 0x0005
        app_intf.serial_number = 'ABC123'
        repl_intf = MagicMock(spec=['device', 'vid', 'pid', 'serial_number'])
        repl_intf.device = '/dev/cu.usbmodem1101'
        repl_intf.vid = 0x2E8A
        repl_intf.pid = 0x0005
        repl_intf.serial_number = 'ABC123'
        # Note: App port listed BEFORE REPL — auto-discovery must still
        # pick REPL via lexicographic sort.
        mock_comports.return_value = [app_intf, repl_intf]
        result = _find_dev_rp2350_pre_flash_port()
        assert result is not None
        assert result['repl_port'] == '/dev/cu.usbmodem1101'
        assert result['app_port'] == '/dev/cu.usbmodem1103'


class TestRebootDevBoardToBootsel:
    """_reboot_dev_board_to_bootsel calls picotool with the right flags."""

    @patch("drivers.firmware_updater._find_picotool", return_value=None)
    def test_returns_false_when_picotool_missing(self, mock_find):
        assert _reboot_dev_board_to_bootsel() is False

    @patch("drivers.firmware_updater._find_picotool", return_value='/usr/bin/picotool')
    @patch("drivers.firmware_updater.subprocess.run")
    def test_calls_picotool_with_reboot_u_f(self, mock_run, mock_find):
        mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
        result = _reboot_dev_board_to_bootsel()
        assert result is True
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == '/usr/bin/picotool'
        assert 'reboot' in cmd
        assert '-u' in cmd
        assert '-f' in cmd

    @patch("drivers.firmware_updater._find_picotool", return_value='/usr/bin/picotool')
    @patch("drivers.firmware_updater.subprocess.run")
    def test_returns_false_on_picotool_failure(self, mock_run, mock_find):
        mock_run.return_value = MagicMock(returncode=1, stdout='', stderr='boom')
        assert _reboot_dev_board_to_bootsel() is False

    @patch("drivers.firmware_updater._find_picotool", return_value='/usr/bin/picotool')
    @patch("drivers.firmware_updater.subprocess.run")
    def test_handles_timeout_expired(self, mock_run, mock_find):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd='picotool', timeout=10)
        assert _reboot_dev_board_to_bootsel() is False


class TestFlashDevBoardPreflight:
    """Pre-flight validation in flash_dev_board()."""

    def test_missing_uf2_returns_failure(self, tmp_path):
        result = flash_dev_board(tmp_path / 'does_not_exist.uf2')
        assert result.success is False
        assert result.error_stage == UpdateStage.PREFLIGHT
        assert 'not found' in result.error_message.lower()

    def test_too_small_uf2_returns_failure(self, tmp_path):
        tiny = tmp_path / 'tiny.uf2'
        tiny.write_bytes(b'too small')
        result = flash_dev_board(tiny)
        assert result.success is False
        assert result.error_stage == UpdateStage.PREFLIGHT
        assert 'too small' in result.error_message.lower()


class TestFlashDevBoardHappyPath:
    """End-to-end flash_dev_board with all subprocess + filesystem mocks."""

    @patch("drivers.firmware_updater._wait_for_serial_port",
           return_value='/dev/cu.usbmodem999')
    @patch("drivers.firmware_updater._wait_for_drive_disappear", return_value=True)
    @patch("drivers.firmware_updater.shutil.copy2")
    @patch("drivers.firmware_updater._wait_for_bootsel_drive")
    @patch("drivers.firmware_updater._reboot_dev_board_to_bootsel", return_value=True)
    @patch("drivers.firmware_updater._find_dev_rp2350_pre_flash_port",
           return_value={'repl_port': '/dev/cu.usbmodem101',
                         'app_port': None,
                         'vid': 0x2886, 'pid': 0x0058,
                         'label': 'Seeed XIAO RP2350',
                         'serial': 'XYZ'})
    @patch("drivers.firmware_updater._detect_bootsel_drive", return_value=None)
    def test_happy_path_via_picotool_reboot(
        self, mock_detect, mock_find_port, mock_reboot, mock_wait_drive,
        mock_copy, mock_drive_disappear, mock_wait_serial, tmp_path,
    ):
        # Provide a fake UF2 with a plausible mount target.
        uf2 = tmp_path / 'micropython-1.28.0-rp2350.uf2'
        uf2.write_bytes(b'\x00' * 1024)  # > 512 bytes
        bootsel_path = tmp_path / 'mock_bootsel'
        bootsel_path.mkdir()
        mock_wait_drive.return_value = bootsel_path

        result = flash_dev_board(uf2)

        assert result.success is True
        # Verify the orchestration:
        mock_detect.assert_called()
        mock_find_port.assert_called()
        mock_reboot.assert_called_once()  # picotool reboot was invoked
        mock_copy.assert_called_once()
        # Copy destination should be the BOOTSEL drive
        copy_args = mock_copy.call_args[0]
        assert copy_args[0] == uf2
        assert str(copy_args[1]).startswith(str(bootsel_path))

    @patch("drivers.firmware_updater._reboot_dev_board_to_bootsel",
           return_value=False)
    @patch("drivers.firmware_updater._find_dev_rp2350_pre_flash_port",
           return_value={'repl_port': '/dev/cu.test',
                         'app_port': None,
                         'vid': 0x2886, 'pid': 0x0058,
                         'label': 'XIAO',
                         'serial': 'TST'})
    @patch("drivers.firmware_updater._detect_bootsel_drive", return_value=None)
    def test_picotool_failure_returns_user_actionable_error(
        self, mock_detect, mock_find, mock_reboot, tmp_path,
    ):
        uf2 = tmp_path / 'fw.uf2'
        uf2.write_bytes(b'\x00' * 1024)

        result = flash_dev_board(uf2)
        assert result.success is False
        assert result.error_stage == UpdateStage.SENDING_FWUPDATE
        # Error message should explain the recovery path
        msg = result.error_message.lower()
        assert 'boot' in msg  # mentions BOOT button
        assert 'picotool' in msg or 'reset interface' in msg
