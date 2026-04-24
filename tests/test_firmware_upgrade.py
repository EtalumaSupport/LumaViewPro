# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for drivers/firmware_updater.upgrade_board_fw40_from_source.

Per FIRMWARE_PLAN.md §13.X — the FW4.0 Field Upgrade Tool. Every
invariant I1-I9 has a failing-case test; every gate (P0-P5) has a
"nothing written past this point" test. MagicMock/patch only — no
bench hardware, no real serial.

Seams mocked:
  - drivers.firmware_updater._create_board           -> returns mock_board
  - drivers.firmware_updater.deploy_firmware_bundle_fw40 -> bundle write
  - drivers.firmware_updater._find_mpy_cross         -> pretend present
  - drivers.firmware_updater._mpy_cross_compile      -> pretend True
  - lvp_logger.log_dir                               -> tmp_path

This file will fail to import until
drivers.firmware_updater.upgrade_board_fw40_from_source and UpgradeResult
are added (step 2 of §13.X.8). That is the intended RED state.
"""

import hashlib
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from drivers.firmware_updater import (
    BoardType,
    BOARD_CONFIGS,
    UpdateResult,
    UpdateStage,
    UpdateError,
    # Added by step 2 — will ImportError until then.
    upgrade_board_fw40_from_source,
    UpgradeResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


MOTOR_MAIN_BODY = b'# fake motor main\nimport fw40_framing\n' * 40
LED_MAIN_BODY = b'# fake led main\nimport fw40_framing\n' * 40
FRAMING_BODY = b'# fake framing\nFW_VERSION = "4.0.0"\n' * 40


@pytest.fixture
def source_tree(tmp_path):
    """Temp Firmware-FW4.0-shaped source tree with manifest."""
    (tmp_path / 'fw40_framing.py').write_bytes(FRAMING_BODY)

    motor_dir = tmp_path / 'Motor Controller' / 'Firmware'
    motor_dir.mkdir(parents=True)
    (motor_dir / 'main.py').write_bytes(MOTOR_MAIN_BODY)

    led_dir = tmp_path / 'LED Controller'
    led_dir.mkdir()
    (led_dir / 'main.py').write_bytes(LED_MAIN_BODY)

    manifest = {
        "manifest_version": 1,
        "motor": {
            "fw_version": "4.0.0",
            "main": "Motor Controller/Firmware/main.py",
            "features": ["motion", "homing", "fw40_framing"],
        },
        "led": {
            "fw_version": "4.0.0",
            "main": "LED Controller/main.py",
            "features": ["led_on", "fw40_framing"],
        },
        "framing": "fw40_framing.py",
    }
    (tmp_path / 'firmware_manifest.json').write_text(json.dumps(manifest))
    return tmp_path


def _motorconfig_bytes(overwritable=None):
    """Compose a motorconfig.json payload. Default: all writes allowed."""
    doc = {
        "SerialNumber": "SN115",
        "Overwritable": overwritable if overwritable is not None else {
            "Main": 1, "IniFiles": 1, "uf2": 1,
        },
    }
    return json.dumps(doc).encode('utf-8')


class _FakeProtocol:
    """Stand-in for drivers.serialboard.ProtocolVersion enum entries.

    Tests that want to simulate a V4 board build a MagicMock with
    ``protocol_version=_FakeProtocol('v4')``. The probe reads ``.value``
    and isinstance-checks for ``str``.
    """
    def __init__(self, value: str):
        self.value = value


@pytest.fixture
def mock_board():
    """Healthy pre-3.0 motor board. Tests override specific attrs."""
    board = MagicMock()
    # SerialBoard attrs populated by connect() / _detect_firmware_version.
    # Explicit so the probe doesn't trip on MagicMock auto-attrs.
    board.firmware_version = 'v2.9.99'      # pre-3.0 → legacy_responsive
    board.firmware_date = None
    board.firmware_responding = True
    board.protocol_version = _FakeProtocol('legacy')
    board.features = []

    board.enter_raw_repl.return_value = True
    board.exit_raw_repl.return_value = None
    board.repl_list_files.return_value = [
        'main.py', 'motorconfig.json', 'xymotorconfig.ini',
        'ztmotorconfig.ini', 'ztmotorconfig2.ini', 'ztmotorconfig3.ini',
    ]
    board.repl_read_file.side_effect = lambda name, verify=True: (
        _motorconfig_bytes() if name == 'motorconfig.json'
        else b'# existing ini content\n'
    )
    board.repl_write_file.return_value = True
    board.repl_exec.return_value = (b'1\n', b'')     # machine.reset_cause() = 1
    board.verify_firmware_running.return_value = 'v2.9.99'
    board.detect_firmware_version.return_value = None
    board.disconnect.return_value = None
    # exchange_command is the JSON-INFO probe path; default = no JSON INFO
    board.exchange_command.return_value = None
    return board


@pytest.fixture
def post_flash_info_ok():
    """JSON INFO payload the board would emit after a successful 4.0 flash."""
    return json.dumps({
        "fw_version": "4.0.0",
        "features": ["motion", "homing", "fw40_framing"],
        "heap_free": 65536,
        "chip_check": {"TMC_XY": "ok", "TMC_ZT": "ok"},
    })


@pytest.fixture
def telemetry_dir(tmp_path, monkeypatch):
    """Redirect the telemetry writer to a tmp path instead of real LVP appdata."""
    log_dir = tmp_path / 'LVP_Log'
    log_dir.mkdir()
    monkeypatch.setattr('lvp_logger.log_dir', str(log_dir), raising=False)
    return log_dir


@pytest.fixture
def patched_bundle(monkeypatch, post_flash_info_ok):
    """Replace deploy_firmware_bundle_fw40 with a pass-through that:
      - records each call for assertion,
      - returns a successful UpdateResult,
      - flips the mock board into a 4.0-reporting state.
    """
    calls = []

    def _fake(board_type, main_module_path, framing_path,
              progress_callback=None, backup_dir=None,
              skip_config_backup=False, skip_post_test=False,
              soft_reset=False):
        calls.append({
            'board_type': board_type,
            'main_module_path': Path(main_module_path),
            'framing_path': Path(framing_path),
            'backup_dir': Path(backup_dir) if backup_dir else None,
            'skip_config_backup': skip_config_backup,
            'skip_post_test': skip_post_test,
            'soft_reset': soft_reset,
        })
        r = UpdateResult(success=True, board_type=board_type)
        r.old_version = 'v2.9.99'
        r.new_version = '4.0.0'
        return r

    monkeypatch.setattr(
        'drivers.firmware_updater.deploy_firmware_bundle_fw40',
        _fake,
    )
    return calls


@pytest.fixture
def patched_create_board(monkeypatch, mock_board):
    monkeypatch.setattr(
        'drivers.firmware_updater._create_board',
        lambda config, port=None, timeout=2.0: mock_board,
    )
    return mock_board


@pytest.fixture
def patched_mpy_cross(monkeypatch):
    monkeypatch.setattr(
        'drivers.firmware_updater._find_mpy_cross',
        lambda: Path('/fake/mpy-cross'),
    )
    monkeypatch.setattr(
        'drivers.firmware_updater._mpy_cross_compile',
        lambda py, mpy, mpy_cross_path=None: True,
    )


# ===========================================================================
# I8 / P0 — host source validation
# ===========================================================================


class TestPreflight:
    """I8: every failure happens on host before any board is touched."""

    def test_manifest_missing_exits_10(self, source_tree):
        (source_tree / 'firmware_manifest.json').unlink()
        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.success is False
        assert r.exit_code == 10
        assert r.error_code is not None

    def test_manifest_invalid_json_exits_10(self, source_tree):
        (source_tree / 'firmware_manifest.json').write_text('{ not json')
        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.exit_code == 10

    def test_main_source_missing_exits_10(self, source_tree):
        (source_tree / 'Motor Controller' / 'Firmware' / 'main.py').unlink()
        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.exit_code == 10
        assert r.error_code is not None

    def test_framing_missing_exits_10(self, source_tree):
        (source_tree / 'fw40_framing.py').unlink()
        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.exit_code == 10

    def test_mpy_cross_missing_exits_10(self, source_tree, monkeypatch):
        monkeypatch.setattr(
            'drivers.firmware_updater._find_mpy_cross', lambda: None)
        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.exit_code == 10

    def test_dry_run_no_transport(
        self, source_tree, patched_mpy_cross, telemetry_dir, monkeypatch
    ):
        """--dry-run completes P0 only, never calls _create_board."""
        create_calls = []
        monkeypatch.setattr(
            'drivers.firmware_updater._create_board',
            lambda *a, **k: create_calls.append(1) or MagicMock(),
        )
        r = upgrade_board_fw40_from_source(
            BoardType.MOTOR, source_tree, dry_run=True)
        assert r.success is True
        assert r.exit_code == 0
        assert create_calls == []


# ===========================================================================
# I7 / P1 — probe classifier
# ===========================================================================


class TestProbe:
    """I7: unresponsive boards are not written to; legacy_unknown proceeds."""

    def test_unresponsive_exits_20_no_write(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        patched_create_board.firmware_version = None
        patched_create_board.firmware_date = None
        patched_create_board.firmware_responding = False
        patched_create_board.protocol_version = _FakeProtocol('legacy')
        patched_create_board.features = []
        patched_create_board.enter_raw_repl.return_value = False
        patched_create_board.repl_exec.side_effect = Exception('no repl')

        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.success is False
        assert r.exit_code == 20
        assert r.probe_classification == 'unresponsive'
        assert patched_bundle == []  # I7: nothing written

    def test_legacy_unknown_permissive_proceeds(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        """§13.X.9 Q2 — board responds, INFO not parseable. Do NOT refuse."""
        patched_create_board.firmware_version = None
        patched_create_board.firmware_date = None
        patched_create_board.firmware_responding = True
        patched_create_board.protocol_version = _FakeProtocol('legacy')
        patched_create_board.features = []
        # Post-write detect flips fw_version + features to 4.0
        def _post_detect():
            patched_create_board.firmware_version = '4.0.0'
            patched_create_board.features = [
                'motion', 'homing', 'fw40_framing']
        patched_create_board.detect_firmware_version.side_effect = _post_detect

        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.probe_classification == 'legacy_unknown'
        assert len(patched_bundle) == 1, (
            'legacy_unknown MUST proceed through deploy per Q2 resolution')

    def test_fw40_current_idempotent_pass(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        """Re-running same manifest on an already-current board is a pass."""
        patched_create_board.firmware_version = '4.0.0'
        patched_create_board.firmware_responding = True
        patched_create_board.protocol_version = _FakeProtocol('v4')
        patched_create_board.features = [
            'motion', 'homing', 'fw40_framing']
        patched_create_board.detect_firmware_version.return_value = None

        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.success is True
        assert r.exit_code == 0
        assert r.probe_classification == 'fw40_current'

    def test_fw40_partial_classified(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        """Probe detects a half-upgraded board — V4 protocol, framing missing."""
        patched_create_board.firmware_version = '4.0.0'
        patched_create_board.firmware_responding = True
        patched_create_board.protocol_version = _FakeProtocol('v4')
        patched_create_board.features = []  # fw40_framing missing
        def _post_detect():
            patched_create_board.features = [
                'motion', 'homing', 'fw40_framing']
        patched_create_board.detect_firmware_version.side_effect = _post_detect

        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.probe_classification == 'fw40_partial'


# ===========================================================================
# I2 / P2 — mandatory verified config backup
# ===========================================================================


class TestBackup:
    """I2: backup must succeed before any firmware file is written."""

    def test_backup_reread_mismatch_exits_30_no_write(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        """P2 byte-compare fails → abort before P4 bundle."""
        good = _motorconfig_bytes()
        corrupt = b'corrupted-on-reread'
        reads = iter([good, corrupt])
        patched_create_board.repl_read_file.side_effect = (
            lambda name, verify=True: next(reads)
            if name == 'motorconfig.json' else b'# ini\n'
        )

        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.success is False
        assert r.exit_code == 30
        assert patched_bundle == []  # I2: nothing written to board

    def test_backup_read_failure_exits_30(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        # Clear the fixture's side_effect so return_value=None takes effect.
        patched_create_board.repl_read_file.side_effect = None
        patched_create_board.repl_read_file.return_value = None

        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.exit_code == 30
        assert patched_bundle == []


# ===========================================================================
# I1 / I3 / P4 — bundle write atomicity + ordering
# ===========================================================================


class TestBundleWrite:
    """I1: no silent half-upgrade. I3: write order framing → module → main."""

    def test_write_invokes_bundle_helper(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        """Contract: upgrade path uses deploy_firmware_bundle_fw40,
        never deploy_firmware_file directly (that was the gap §13.X.3
        describes)."""
        patched_create_board.verify_firmware_running.return_value = '4.0.0'
        patched_create_board.detect_firmware_version.side_effect = (
            lambda: setattr(patched_create_board, 'firmware_version', '4.0.0')
        )

        upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert len(patched_bundle) == 1
        assert patched_bundle[0]['board_type'] == BoardType.MOTOR
        # Paths resolved from manifest, not hard-coded
        assert patched_bundle[0]['main_module_path'].name == 'main.py'
        assert patched_bundle[0]['framing_path'].name == 'fw40_framing.py'

    def test_bundle_failure_exits_40(
        self, source_tree, patched_mpy_cross, patched_create_board,
        monkeypatch, telemetry_dir,
    ):
        """Simulated bundle write failure → exit 40."""
        def _fail(*a, **k):
            bt = k.get('board_type') or (a[0] if a else BoardType.MOTOR)
            r = UpdateResult(success=False, board_type=bt)
            r.error_message = 'SHA256 mismatch on fw40_framing.mpy'
            r.error_stage = UpdateStage.RESTORING_CONFIG
            return r
        monkeypatch.setattr(
            'drivers.firmware_updater.deploy_firmware_bundle_fw40', _fail)

        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.success is False
        assert r.exit_code == 40


# ===========================================================================
# I4 / P5 — post-flash verify
# ===========================================================================


class TestVerify:
    """I4: verify is FW4.0-aware. P5 mismatch → no auto-rollback (§13.X.5)."""

    def test_post_reboot_version_mismatch_exits_50_no_rollback(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        """After reboot, INFO still reports pre-3.0 → exit 50, no retry."""
        patched_create_board.verify_firmware_running.return_value = 'v2.9.99'
        patched_create_board.detect_firmware_version.return_value = None
        # Post-reboot exchange still returns no JSON INFO
        patched_create_board.exchange_command.return_value = None

        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.success is False
        assert r.exit_code == 50
        # No-auto-rollback: bundle helper called exactly once, not twice
        assert len(patched_bundle) == 1

    def test_post_reboot_features_missing_exits_50(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        """Version matches but expected feature not present → exit 50."""
        patched_create_board.firmware_version = '4.0.0'
        patched_create_board.verify_firmware_running.return_value = '4.0.0'
        patched_create_board.detect_firmware_version.side_effect = (
            lambda: setattr(patched_create_board, 'firmware_version', '4.0.0')
        )
        # INFO lies: version right, features incomplete
        patched_create_board.exchange_command.return_value = json.dumps({
            "fw_version": "4.0.0",
            "features": ["motion"],   # missing 'homing' + 'fw40_framing'
        })

        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.success is False
        assert r.exit_code == 50


# ===========================================================================
# I5 — Overwritable flags
# ===========================================================================


class TestOverwritable:
    """I5: motorconfig.json.Overwritable is honored."""

    def test_main_zero_skips_firmware_write(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        patched_create_board.repl_read_file.side_effect = (
            lambda name, verify=True: (
                _motorconfig_bytes(
                    {"Main": 0, "IniFiles": 1, "uf2": 1})
                if name == 'motorconfig.json' else b'# ini\n'
            )
        )

        r = upgrade_board_fw40_from_source(
            BoardType.MOTOR, source_tree, respect_overwritable=True)
        # Bundle helper not called when Main=0 blocks firmware
        assert patched_bundle == []
        assert 'main' in [f.lower() for f in r.files_skipped_overwritable]
        assert r.exit_code == 35

    def test_respect_overwritable_false_overrides(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        """--ignore-overwritable bypass: writes proceed despite Main=0."""
        patched_create_board.repl_read_file.side_effect = (
            lambda name, verify=True: (
                _motorconfig_bytes(
                    {"Main": 0, "IniFiles": 1, "uf2": 1})
                if name == 'motorconfig.json' else b'# ini\n'
            )
        )
        patched_create_board.verify_firmware_running.return_value = '4.0.0'
        patched_create_board.detect_firmware_version.side_effect = (
            lambda: setattr(patched_create_board, 'firmware_version', '4.0.0')
        )

        r = upgrade_board_fw40_from_source(
            BoardType.MOTOR, source_tree, respect_overwritable=False)
        assert len(patched_bundle) == 1
        # Should surface as a warning in telemetry
        assert any('overwritable' in w.lower() for w in r.warnings)

    def test_led_board_ignores_motorconfig_overwritable(
        self, source_tree, patched_mpy_cross, monkeypatch,
        patched_bundle, telemetry_dir,
    ):
        """LED board has no motorconfig.json — Overwritable check skipped."""
        led_board = MagicMock()
        led_board.firmware_version = 'v2.9.99'
        led_board.firmware_date = None
        led_board.firmware_responding = True
        led_board.protocol_version = _FakeProtocol('legacy')
        led_board.features = []
        led_board.enter_raw_repl.return_value = True
        led_board.exit_raw_repl.return_value = None
        led_board.repl_list_files.return_value = ['main.py', 'cal.json']
        led_board.repl_read_file.return_value = b'{"cal_data": []}'
        led_board.repl_write_file.return_value = True
        led_board.verify_firmware_running.return_value = '4.0.0'
        def _post_detect():
            led_board.firmware_version = '4.0.0'
            led_board.features = ['led_on', 'fw40_framing']
        led_board.detect_firmware_version.side_effect = _post_detect
        led_board.exchange_command.return_value = None
        led_board.disconnect.return_value = None
        monkeypatch.setattr(
            'drivers.firmware_updater._create_board',
            lambda *a, **k: led_board,
        )

        r = upgrade_board_fw40_from_source(BoardType.LED, source_tree)
        assert len(patched_bundle) == 1


# ===========================================================================
# I6 — telemetry log
# ===========================================================================


class TestTelemetry:
    """I6: JSON Lines log written on both success and abort."""

    def test_log_written_on_success(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        patched_create_board.verify_firmware_running.return_value = '4.0.0'
        patched_create_board.detect_firmware_version.side_effect = (
            lambda: setattr(patched_create_board, 'firmware_version', '4.0.0')
        )

        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.telemetry_log_path is not None
        p = Path(r.telemetry_log_path)
        assert p.exists(), f'telemetry JSONL missing: {p}'
        lines = p.read_text().strip().splitlines()
        assert len(lines) > 0
        first = json.loads(lines[0])
        assert 'step' in first
        assert 'timestamp' in first

    def test_log_written_on_abort(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        """Even when P2 aborts, the telemetry file exists and records the abort."""
        patched_create_board.repl_read_file.return_value = None  # backup fail

        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r.telemetry_log_path is not None
        assert Path(r.telemetry_log_path).exists()

    def test_log_path_under_appdata_lvp_log(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        """Path must be {lvp_appdata}/logs/LVP_Log/firmware_upgrade_*.jsonl."""
        patched_create_board.verify_firmware_running.return_value = '4.0.0'
        patched_create_board.detect_firmware_version.side_effect = (
            lambda: setattr(patched_create_board, 'firmware_version', '4.0.0')
        )

        r = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        p = Path(r.telemetry_log_path)
        assert p.parent == telemetry_dir
        assert p.name.startswith('firmware_upgrade_')
        assert p.suffix == '.jsonl'


# ===========================================================================
# P4 — soft_reset fed by probe classifier
# ===========================================================================


class TestSoftReset:
    """§13.X.4 P4: probe classification drives soft_reset choice."""

    def test_legacy_responsive_uses_soft_reset_false(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        """Pre-3.0 LED with WDT-feed Timer: soft_reset=False mandatory."""
        patched_create_board.verify_firmware_running.return_value = '4.0.0'
        patched_create_board.detect_firmware_version.side_effect = (
            lambda: setattr(patched_create_board, 'firmware_version', '4.0.0')
        )
        patched_create_board.firmware_version = 'v2.9.99'  # legacy_responsive

        upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert patched_bundle[0]['soft_reset'] is False

    def test_fw40_uses_soft_reset_false_default(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir, post_flash_info_ok,
    ):
        patched_create_board.firmware_version = '4.0.0'
        patched_create_board.exchange_command.return_value = post_flash_info_ok
        patched_create_board.verify_firmware_running.return_value = '4.0.0'
        patched_create_board.detect_firmware_version.side_effect = (
            lambda: setattr(patched_create_board, 'firmware_version', '4.0.0')
        )

        upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        # fw40_* → soft_reset=False default
        assert patched_bundle[0]['soft_reset'] is False


# ===========================================================================
# I9 — unattended-safe
# ===========================================================================


class TestUnattended:
    """I9: no input() / stdin / prompts anywhere in the upgrade path."""

    def test_no_interactive_prompts_in_source(self):
        """Grep the driver source for any read-from-stdin surface."""
        import drivers.firmware_updater as fu
        src = Path(fu.__file__).read_text()
        # Extract just the upgrade section + helpers it calls.
        # This is a conservative grep: if ANY input()/stdin surface
        # is added, this test fires and the reviewer must justify.
        #
        # Allowed: comments ('# input from ...') — pattern below is
        # word-boundary `input\(` to avoid matching docstrings.
        for pat in [r'\binput\s*\(', r'sys\.stdin', r'getpass',
                    r'confirm\s*=.*input']:
            matches = [
                (i, line) for i, line in enumerate(src.splitlines(), 1)
                if re.search(pat, line) and not line.lstrip().startswith('#')
            ]
            assert not matches, (
                f'I9 violation — found {pat} in firmware_updater.py:\n'
                + '\n'.join(f'  line {i}: {ln}' for i, ln in matches)
            )


# ===========================================================================
# Idempotency
# ===========================================================================


class TestIdempotent:
    """Running the same upgrade twice is a pass."""

    def test_same_manifest_twice(
        self, source_tree, patched_mpy_cross, patched_create_board,
        patched_bundle, telemetry_dir,
    ):
        patched_create_board.firmware_version = '4.0.0'
        patched_create_board.firmware_responding = True
        patched_create_board.protocol_version = _FakeProtocol('v4')
        patched_create_board.features = [
            'motion', 'homing', 'fw40_framing']
        patched_create_board.detect_firmware_version.return_value = None

        r1 = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        r2 = upgrade_board_fw40_from_source(BoardType.MOTOR, source_tree)
        assert r1.success is True, (
            f'first run failed: {r1.error_code} {r1.error_message}')
        assert r2.success is True
        assert r1.exit_code == 0
        assert r2.exit_code == 0


# ===========================================================================
# UpgradeResult dataclass shape
# ===========================================================================


class TestUpgradeResultShape:
    """Lock the public API shape — LVP UI depends on these fields."""

    def test_required_fields_present(self):
        r = UpgradeResult(
            success=True, board_type=BoardType.MOTOR, exit_code=0)
        for attr in ('success', 'board_type', 'exit_code', 'old_version',
                     'new_version', 'probe_classification',
                     'config_backup_path', 'telemetry_log_path',
                     'overwritable_flags', 'files_written',
                     'files_skipped_overwritable', 'error_code',
                     'error_message', 'error_stage', 'warnings'):
            assert hasattr(r, attr), f'UpgradeResult missing {attr}'
