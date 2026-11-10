# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for src/_lumascope/drivers/firmware_updater.py.

Uses mock serial ports and mock filesystem -- no hardware needed.
Covers version parsing, board configs, BOOTSEL detection, serial port
lookup, FWUPDATE command, config backup/restore, UpdateResult, and
the top-level update_firmware orchestrator.
"""

import json
import time
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from _lumascope.drivers.firmware_updater import (
    _parse_uf2_version,
    _detect_bootsel_drive,
    _detect_bootsel_macos,
    _find_serial_port,
    _send_fwupdate_command,
    _backup_configs,
    _restore_configs,
    _report_progress,
    update_firmware,
    deploy_firmware_file,
    BoardType,
    BOARD_CONFIGS,
    UpdateError,
    UpdateResult,
    UpdateStage,
)
from _lumascope.drivers.serialboard import SerialBoard
from _lumascope.drivers.sim_wire.backend import MotorBoardSpec, SimWireBackend


def _motor_bus(dialect='field'):
    """A bus with one simulated motor board on *dialect*'s firmware, which the
    updater's SerialBoard finds by the motor board's VID and PID."""
    return SimWireBackend(MotorBoardSpec('LS850T', frozenset('XYZT'), dialect=dialect))


# The updater's own name for time, with sleeps that return at once. Patching
# time.sleep through it would replace the time module's sleep for every
# thread in the process.
_NO_WAIT_TIME = SimpleNamespace(monotonic=time.monotonic, sleep=lambda _s: None)


# ---------------------------------------------------------------------------
# 1. _parse_uf2_version
# ---------------------------------------------------------------------------


class TestParseUf2Version:
    def test_semantic_version(self):
        assert _parse_uf2_version(Path('led_firmware_v2.1.0.uf2')) == '2.1.0'

    def test_semantic_version_no_v(self):
        assert _parse_uf2_version(Path('motor_firmware_1.0.3.uf2')) == '1.0.3'

    def test_date_version(self):
        assert _parse_uf2_version(Path('motor_firmware_2026-03-09.uf2')) == '2026-03-09'

    def test_no_version_returns_stem(self):
        assert _parse_uf2_version(Path('custom_build.uf2')) == 'custom_build'

    def test_semantic_takes_precedence_over_date(self):
        # If both patterns appear, semantic (matched first) wins
        assert _parse_uf2_version(Path('fw_v1.2.3_2026-01-01.uf2')) == '1.2.3'

    def test_path_with_directory(self):
        p = Path('/some/dir/led_firmware_v3.0.0.uf2')
        assert _parse_uf2_version(p) == '3.0.0'


# ---------------------------------------------------------------------------
# 2. BoardConfig -- LED and MOTOR configs
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
# 3. _detect_bootsel_drive -- mock filesystem, macOS path
# ---------------------------------------------------------------------------


class TestDetectBootselDrive:
    @patch('_lumascope.drivers.firmware_updater.platform.system', return_value='Darwin')
    @patch('_lumascope.drivers.firmware_updater._detect_bootsel_macos')
    def test_macos_delegates(self, mock_macos, mock_sys):
        mock_macos.return_value = Path('/Volumes/RPI-RP2')
        result = _detect_bootsel_drive()
        assert result == Path('/Volumes/RPI-RP2')
        mock_macos.assert_called_once()

    @patch('_lumascope.drivers.firmware_updater.platform.system', return_value='Darwin')
    @patch('_lumascope.drivers.firmware_updater._detect_bootsel_macos', return_value=None)
    def test_macos_not_found(self, mock_macos, mock_sys):
        assert _detect_bootsel_drive() is None

    @patch('_lumascope.drivers.firmware_updater.platform.system', return_value='FreeBSD')
    def test_unsupported_platform(self, mock_sys):
        assert _detect_bootsel_drive() is None

    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    def test_macos_found(self, mock_is_dir, mock_exists):
        result = _detect_bootsel_macos()
        assert result == Path('/Volumes/RPI-RP2')

    @patch('pathlib.Path.is_dir', return_value=False)
    def test_macos_no_dir(self, mock_is_dir):
        result = _detect_bootsel_macos()
        assert result is None


# ---------------------------------------------------------------------------
# 4. _find_serial_port -- mock list_ports.comports()
# ---------------------------------------------------------------------------


class TestFindSerialPort:
    @patch('_lumascope.drivers.firmware_updater.list_ports.comports')
    def test_found(self, mock_comports):
        port = MagicMock()
        port.vid = 0x2E8A
        port.pid = 0x0005
        port.device = '/dev/ttyACM0'
        mock_comports.return_value = [port]
        assert _find_serial_port(0x2E8A, 0x0005) == '/dev/ttyACM0'

    @patch('_lumascope.drivers.firmware_updater.list_ports.comports')
    def test_not_found(self, mock_comports):
        port = MagicMock()
        port.vid = 0x1234
        port.pid = 0x5678
        mock_comports.return_value = [port]
        assert _find_serial_port(0x2E8A, 0x0005) is None

    @patch('_lumascope.drivers.firmware_updater.list_ports.comports')
    def test_empty_list(self, mock_comports):
        mock_comports.return_value = []
        assert _find_serial_port(0x2E8A, 0x0005) is None

    @patch('_lumascope.drivers.firmware_updater.list_ports.comports')
    def test_multiple_ports_returns_first_match(self, mock_comports):
        p1 = MagicMock(vid=0x1111, pid=0x2222, device='/dev/ttyACM0')
        p2 = MagicMock(vid=0x2E8A, pid=0x0005, device='/dev/ttyACM1')
        p3 = MagicMock(vid=0x2E8A, pid=0x0005, device='/dev/ttyACM2')
        mock_comports.return_value = [p1, p2, p3]
        assert _find_serial_port(0x2E8A, 0x0005) == '/dev/ttyACM1'


# ---------------------------------------------------------------------------
# 5-6. _send_fwupdate_command -- mock serial, verify bytes + error
# ---------------------------------------------------------------------------


class TestSendFwupdateCommand:
    def _make_motor_config(self):
        return BOARD_CONFIGS[BoardType.MOTOR]

    def _make_led_config(self):
        return BOARD_CONFIGS[BoardType.LED]

    @pytest.mark.firmware_tier
    def test_motor_sends_fwupdate(self):
        """Motor board on firmware with FWUPDATE (the 3.0 firmware, which
        answers 'Entering bootloader mode...'): the board leaves for its
        bootloader and the updater lets go of its port."""
        cfg = self._make_motor_config()
        backend = _motor_bus(dialect='3.0')
        board = SerialBoard(vid=cfg.vid, pid=cfg.pid, label='[FW-Motor]', backend=backend)
        board.connect()
        assert board.is_connected()

        _send_fwupdate_command(board, cfg)

        assert not board.is_connected()
        # The board's firmware has stopped: a fresh connect reads no version.
        after = SerialBoard(vid=cfg.vid, pid=cfg.pid, label='[FW-Motor]', backend=backend)
        after.connect()
        try:
            assert after.firmware_version is None
        finally:
            after.disconnect()

    def test_led_sends_fwupdate(self):
        """LED board: exchange_command('FWUPDATE') is called."""
        cfg = self._make_led_config()
        board = MagicMock()
        board.exchange_command.return_value = None
        _send_fwupdate_command(board, cfg)

        board.exchange_command.assert_called_once_with('FWUPDATE', timeout=3.0)
        board.disconnect.assert_called()

    def test_a_defect_in_the_exchange_leaves_as_itself(self):
        """Only the board's failures become UpdateError; our own leaves as itself."""
        cfg = self._make_motor_config()
        board = MagicMock()
        board.exchange_command.side_effect = TypeError('a defect')
        with pytest.raises(TypeError, match='a defect'):
            _send_fwupdate_command(board, cfg)

    @pytest.mark.firmware_tier
    def test_fallback_to_raw_repl_on_not_found(self):
        """Falls back to raw REPL when firmware doesn't support FWUPDATE:
        the field firmware's own not-found answer. The raw-REPL path is
        observed, not run: the emulated firmware has no raw REPL."""
        cfg = self._make_motor_config()
        board = SerialBoard(vid=cfg.vid, pid=cfg.pid, label='[FW-Motor]', backend=_motor_bus())
        board.connect()
        try:
            with patch(
                '_lumascope.drivers.firmware_updater._bootloader_via_raw_repl'
            ) as mock_fallback:
                _send_fwupdate_command(board, cfg)
                mock_fallback.assert_called_once_with(board)
        finally:
            board.disconnect()


# ---------------------------------------------------------------------------
# 7-9. _backup_configs
# ---------------------------------------------------------------------------


class TestBackupConfigs:
    def _make_board(self, files_on_board=None, file_contents=None, enter_repl_ok=True):
        """Create a mock board with SerialBoard-compatible methods."""
        board = MagicMock()
        board.enter_raw_repl.return_value = enter_repl_ok
        board.repl_list_files.return_value = files_on_board or []
        if file_contents is not None:
            board.repl_read_file.side_effect = file_contents
        return board

    def test_all_files_backed_up(self, tmp_path):
        """All config files are read and saved to disk."""
        cfg = BOARD_CONFIGS[BoardType.MOTOR]
        board = self._make_board(
            files_on_board=[
                'motorconfig.json',
                'xymotorconfig.ini',
                'ztmotorconfig.ini',
                'ztmotorconfig2.ini',
            ],
            file_contents=[
                b'{"motor": 1}',
                b'[xy]\nsteps=100',
                b'[zt]\nsteps=200',
                b'[zt2]\nsteps=300',
            ],
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

    def test_missing_file_skipped(self, tmp_path):
        """File not present on board is skipped (not an error)."""
        cfg = BOARD_CONFIGS[BoardType.MOTOR]
        board = self._make_board(
            files_on_board=['motorconfig.json'],
            file_contents=[b'{"motor": 1}'],
        )

        result = _backup_configs(board, cfg, tmp_path)

        assert len(result) == 1
        assert 'motorconfig.json' in result
        board.repl_read_file.assert_called_once()

    def test_read_failure_raises(self, tmp_path):
        """repl_read_file returning None raises UpdateError."""
        cfg = BOARD_CONFIGS[BoardType.LED]
        board = self._make_board(
            files_on_board=['cal.json'],
            file_contents=[None],
        )

        with pytest.raises(UpdateError) as exc_info:
            _backup_configs(board, cfg, tmp_path)
        assert exc_info.value.stage == UpdateStage.BACKING_UP_CONFIG

    def test_enter_repl_failure_raises(self, tmp_path):
        """Failure to enter raw REPL raises UpdateError."""
        cfg = BOARD_CONFIGS[BoardType.LED]
        board = self._make_board(enter_repl_ok=False)

        with pytest.raises(UpdateError) as exc_info:
            _backup_configs(board, cfg, tmp_path)
        assert exc_info.value.stage == UpdateStage.BACKING_UP_CONFIG


# ---------------------------------------------------------------------------
# 10-11. _restore_configs
# ---------------------------------------------------------------------------


class TestRestoreConfigs:
    def _make_board(
        self, files_on_board=None, file_contents=None, enter_repl_ok=True, write_ok=True
    ):
        """Create a mock board with SerialBoard-compatible methods."""
        board = MagicMock()
        board.enter_raw_repl.return_value = enter_repl_ok
        board.repl_list_files.return_value = files_on_board or []
        if file_contents is not None:
            board.repl_read_file.side_effect = file_contents
        else:
            board.repl_read_file.return_value = None
        board.repl_write_file.return_value = write_ok
        return board

    def test_surviving_file_skipped(self, tmp_path):
        """File that survived the update (matches backup) is not rewritten."""
        cfg = BOARD_CONFIGS[BoardType.MOTOR]
        data = b'{"motor": 1}'
        board = self._make_board(
            files_on_board=['motorconfig.json'],
            file_contents=[data],
        )

        result = _restore_configs(board, cfg, {'motorconfig.json': data}, tmp_path)

        assert result is True
        board.repl_write_file.assert_not_called()

    def test_missing_file_written(self, tmp_path):
        """File missing from board after update is restored."""
        cfg = BOARD_CONFIGS[BoardType.MOTOR]
        data = b'{"motor": 1}'
        board = self._make_board(files_on_board=[])

        result = _restore_configs(board, cfg, {'motorconfig.json': data}, tmp_path)

        assert result is True
        board.repl_write_file.assert_called_once()

    def test_changed_file_restored(self, tmp_path):
        """File that exists but differs from backup is overwritten."""
        cfg = BOARD_CONFIGS[BoardType.LED]
        backup_data = b'{"cal": "good"}'
        board = self._make_board(
            files_on_board=['cal.json'],
            file_contents=[b'{"cal": "corrupted"}'],
        )

        result = _restore_configs(board, cfg, {'cal.json': backup_data}, tmp_path)

        assert result is True
        board.repl_write_file.assert_called_once()

    def test_write_failure_raises_naming_the_backup_folder(self, tmp_path):
        """repl_write_file returning False raises UpdateError, naming the
        folder the backup is in, so the user can find the copy to restore."""
        cfg = BOARD_CONFIGS[BoardType.LED]
        board = self._make_board(files_on_board=[], write_ok=False)

        with pytest.raises(UpdateError) as exc_info:
            _restore_configs(board, cfg, {'cal.json': b'data'}, tmp_path)
        assert exc_info.value.stage == UpdateStage.RESTORING_CONFIG
        assert str(tmp_path / 'led') in str(exc_info.value)


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
            old_version='1.0.0',
            new_version='2.0.0',
            config_backup_path=Path('/tmp/backup'),
            error_message='something broke',
            error_stage=UpdateStage.COPYING_UF2,
            warnings=['warn1'],
        )
        assert r.success is False
        assert r.old_version == '1.0.0'
        assert r.new_version == '2.0.0'
        assert r.error_message == 'something broke'
        assert r.error_stage == UpdateStage.COPYING_UF2
        assert r.warnings == ['warn1']


# ---------------------------------------------------------------------------
# 13. update_firmware -- same version skips update
# ---------------------------------------------------------------------------


class TestUpdateFirmwareSameVersion:
    @pytest.mark.firmware_tier
    @patch('_lumascope.drivers.firmware_updater._detect_bootsel_drive', return_value=None)
    @patch('_lumascope.drivers.firmware_updater._parse_uf2_version', return_value='3.0.9')
    def test_same_version_returns_success(self, mock_parse, mock_bootsel, tmp_path, monkeypatch):
        """If board already has the target version, skip update: the
        updater's own board finds the 3.0 firmware (3.0.9) on the bus, and
        the firmware is left running."""
        import functools

        from _lumascope.drivers import firmware_updater

        backend = _motor_bus(dialect='3.0')
        monkeypatch.setattr(
            firmware_updater, 'SerialBoard', functools.partial(SerialBoard, backend=backend)
        )
        uf2 = tmp_path / 'motor_firmware_v3.0.9.uf2'
        uf2.write_bytes(b'\x00' * 1024)

        result = update_firmware(
            board_type=BoardType.MOTOR,
            uf2_path=uf2,
            backup_dir=tmp_path / 'backup',
        )
        assert result.success is True
        assert result.old_version == '3.0.9'
        assert result.new_version == '3.0.9'
        cfg = BOARD_CONFIGS[BoardType.MOTOR]
        after = SerialBoard(vid=cfg.vid, pid=cfg.pid, label='[FW-Motor]', backend=backend)
        after.connect()
        try:
            assert after.firmware_version == '3.0.9', 'the update was not skipped'
        finally:
            after.disconnect()


# ---------------------------------------------------------------------------
# 14. update_firmware -- pre-existing BOOTSEL drive causes abort
# ---------------------------------------------------------------------------


class TestUpdateFirmwareBootselAbort:
    @patch('_lumascope.drivers.firmware_updater.time', _NO_WAIT_TIME)
    @patch(
        '_lumascope.drivers.firmware_updater._detect_bootsel_drive',
        return_value=Path('/Volumes/RPI-RP2'),
    )
    def test_existing_bootsel_aborts(self, mock_bootsel, tmp_path):
        """Pre-existing BOOTSEL drive causes immediate abort."""
        uf2 = tmp_path / 'motor_firmware_v2.0.0.uf2'
        uf2.write_bytes(b'\x00' * 1024)

        result = update_firmware(
            board_type=BoardType.MOTOR,
            uf2_path=uf2,
            backup_dir=tmp_path / 'backup',
        )
        assert result.success is False
        assert result.error_stage == UpdateStage.PREFLIGHT
        assert 'already mounted' in result.error_message


# ---------------------------------------------------------------------------
# 15. Progress callback -- called at each stage, exception-safe
# ---------------------------------------------------------------------------


class TestProgressCallback:
    def test_report_progress_calls_callback(self):
        cb = Mock()
        _report_progress(cb, UpdateStage.PREFLIGHT, 'hello', 0.5)
        cb.assert_called_once_with(UpdateStage.PREFLIGHT, 'hello', 0.5)

    def test_report_progress_none_callback(self):
        """None callback does not raise."""
        _report_progress(None, UpdateStage.PREFLIGHT, 'hello', 0.5)

    def test_report_progress_exception_safe(self):
        """Callback raising exception does not propagate."""
        cb = Mock(side_effect=RuntimeError('boom'))
        # Should not raise
        _report_progress(cb, UpdateStage.PREFLIGHT, 'hello', 0.5)

    @patch('_lumascope.drivers.firmware_updater.time', _NO_WAIT_TIME)
    @patch(
        '_lumascope.drivers.firmware_updater._detect_bootsel_drive',
        return_value=Path('/Volumes/RPI-RP2'),
    )
    def test_callback_called_on_error(self, mock_bootsel, tmp_path):
        """Progress callback receives FAILED stage on error."""
        cb = Mock()
        uf2 = tmp_path / 'motor_firmware_v2.0.0.uf2'
        uf2.write_bytes(b'\x00' * 1024)

        update_firmware(
            board_type=BoardType.MOTOR,
            uf2_path=uf2,
            progress_callback=cb,
            backup_dir=tmp_path / 'backup',
        )
        # Last call should be FAILED
        stages = [c.args[0] for c in cb.call_args_list]
        assert UpdateStage.PREFLIGHT in stages
        assert UpdateStage.FAILED in stages

    @patch('_lumascope.drivers.firmware_updater.time', _NO_WAIT_TIME)
    @patch(
        '_lumascope.drivers.firmware_updater._detect_bootsel_drive',
        return_value=Path('/Volumes/RPI-RP2'),
    )
    def test_exception_in_callback_does_not_break_update(self, mock_bootsel, tmp_path):
        """Even if progress callback raises, update_firmware still returns result."""
        cb = Mock(side_effect=RuntimeError('callback broke'))
        uf2 = tmp_path / 'motor_firmware_v2.0.0.uf2'
        uf2.write_bytes(b'\x00' * 1024)

        result = update_firmware(
            board_type=BoardType.MOTOR,
            uf2_path=uf2,
            progress_callback=cb,
            backup_dir=tmp_path / 'backup',
        )
        # Should still get a result (the error from BOOTSEL pre-check)
        assert isinstance(result, UpdateResult)


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestUpdateErrorDataclass:
    def test_fields(self):
        e = UpdateError('msg', UpdateStage.COPYING_UF2, recoverable=False)
        assert str(e) == 'msg'
        assert e.stage == UpdateStage.COPYING_UF2
        assert e.recoverable is False

    def test_default_recoverable(self):
        e = UpdateError('msg', UpdateStage.PREFLIGHT)
        assert e.recoverable is True


class TestUpdateFirmwareMissingUf2:
    @patch('_lumascope.drivers.firmware_updater.time', _NO_WAIT_TIME)
    @patch('_lumascope.drivers.firmware_updater._detect_bootsel_drive', return_value=None)
    def test_missing_uf2_file(self, mock_bootsel, tmp_path):
        """Non-existent UF2 file returns error result."""
        result = update_firmware(
            board_type=BoardType.MOTOR,
            uf2_path=tmp_path / 'nonexistent.uf2',
            backup_dir=tmp_path / 'backup',
        )
        assert result.success is False
        assert result.error_stage == UpdateStage.PREFLIGHT


class TestUpdateFirmwareTooSmallUf2:
    @patch('_lumascope.drivers.firmware_updater.time', _NO_WAIT_TIME)
    @patch('_lumascope.drivers.firmware_updater._detect_bootsel_drive', return_value=None)
    def test_tiny_uf2(self, mock_bootsel, tmp_path):
        """UF2 file under 512 bytes is rejected."""
        uf2 = tmp_path / 'motor_firmware_v1.0.0.uf2'
        uf2.write_bytes(b'\x00' * 100)

        result = update_firmware(
            board_type=BoardType.MOTOR,
            uf2_path=uf2,
            backup_dir=tmp_path / 'backup',
        )
        assert result.success is False
        assert 'too small' in result.error_message


# ---------------------------------------------------------------------------
# The backup folder is the caller's: update_firmware and deploy_firmware_file
# ---------------------------------------------------------------------------


def _board_with_one_config(config_bytes):
    """A connected board holding motorconfig.json, whose raw REPL answers.

    A stand-in by design: the wire simulator has no raw REPL (entry fails
    on its firmware), so a backup cannot be taken from the simulated
    board; this is that simulator gap.
    """
    board = MagicMock()
    board.found = True
    board.firmware_version = '1.0.0'
    board.enter_raw_repl.return_value = True
    board.repl_list_files.return_value = ['motorconfig.json']
    board.repl_read_file.return_value = config_bytes
    board.repl_write_file.return_value = True
    board.verify_firmware_running.return_value = 'ok'
    return board


def _run_each_update(tmp_path, monkeypatch, board, backup_dir):
    """Run update_firmware and deploy_firmware_file on *board*, each with
    *backup_dir*, and return their results by name. The update stops at
    WAITING_BOOTSEL: a stand-in by design for the host's BOOTSEL drive and
    picotool, which the test host does not have."""
    from _lumascope.drivers import firmware_updater

    monkeypatch.setattr(firmware_updater, 'SerialBoard', lambda **_kw: board)
    monkeypatch.setattr(firmware_updater, 'time', _NO_WAIT_TIME)
    monkeypatch.setattr(firmware_updater, '_detect_bootsel_drive', lambda: None)
    monkeypatch.setattr(firmware_updater, '_wait_for_bootsel_drive', lambda timeout: None)
    monkeypatch.setattr(firmware_updater, '_find_picotool', lambda: None)
    uf2 = tmp_path / 'motor_firmware_v2.0.0.uf2'
    uf2.write_bytes(b'\x00' * 1024)
    main_py = tmp_path / 'main.py'
    main_py.write_bytes(b'# firmware\n' * 20)

    def folder(name):
        return None if backup_dir is None else backup_dir / name

    return {
        'update_firmware': update_firmware(
            BoardType.MOTOR, uf2, backup_dir=folder('update_firmware')
        ),
        'deploy_firmware_file': deploy_firmware_file(
            BoardType.MOTOR,
            main_py,
            backup_dir=folder('deploy_firmware_file'),
            skip_post_test=True,
        ),
    }


class TestTheBackupFolderIsTheCallers:
    def test_an_update_backs_up_into_the_folder_its_caller_names(self, tmp_path, monkeypatch):
        """Each update writes the board's config files and their manifest
        under the folder its caller named, and reports that folder."""
        config = b'{"motor": 1}'
        backups = tmp_path / 'backups'
        results = _run_each_update(tmp_path, monkeypatch, _board_with_one_config(config), backups)

        for name, result in results.items():
            board_dir = backups / name / 'motor'
            assert result.config_backup_path == backups / name, name
            assert (board_dir / 'motorconfig.json').read_bytes() == config, name
            manifest = json.loads((board_dir / 'backup_manifest.json').read_text())
            assert manifest['motorconfig.json']['size'] == len(config), name
        assert results['deploy_firmware_file'].success is True
        assert results['update_firmware'].error_stage == UpdateStage.WAITING_BOOTSEL

    def test_an_update_given_no_folder_takes_no_backup(self, tmp_path, monkeypatch):
        """None takes no backup: no folder is reported and no file is
        written anywhere the test can see."""
        results = _run_each_update(
            tmp_path, monkeypatch, _board_with_one_config(b'{"motor": 1}'), None
        )

        for name, result in results.items():
            assert result.config_backup_path is None, name
        assert results['deploy_firmware_file'].success is True
        assert results['update_firmware'].error_stage == UpdateStage.WAITING_BOOTSEL
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            'main.py',
            'motor_firmware_v2.0.0.uf2',
        ]

    @pytest.mark.firmware_tier
    def test_an_unmakeable_folder_fails_preflight_before_the_board_is_touched(
        self, tmp_path, monkeypatch
    ):
        """A folder that cannot be made fails each update at PREFLIGHT,
        naming the folder, before the simulated board is connected: no
        version was read from it."""
        import functools

        from _lumascope.drivers import firmware_updater

        monkeypatch.setattr(
            firmware_updater, 'SerialBoard', functools.partial(SerialBoard, backend=_motor_bus())
        )
        monkeypatch.setattr(firmware_updater, '_detect_bootsel_drive', lambda: None)
        a_file = tmp_path / 'a_file'
        a_file.write_bytes(b'')
        unmakeable = a_file / 'backups'
        uf2 = tmp_path / 'motor_firmware_v2.0.0.uf2'
        uf2.write_bytes(b'\x00' * 1024)
        main_py = tmp_path / 'main.py'
        main_py.write_bytes(b'# firmware\n' * 20)

        for result in (
            update_firmware(BoardType.MOTOR, uf2, backup_dir=unmakeable),
            deploy_firmware_file(BoardType.MOTOR, main_py, backup_dir=unmakeable),
        ):
            assert result.success is False
            assert result.error_stage == UpdateStage.PREFLIGHT
            assert str(unmakeable) in result.error_message
            assert result.old_version is None, 'the board was connected'
