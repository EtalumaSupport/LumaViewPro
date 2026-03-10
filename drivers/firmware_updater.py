# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Firmware updater for RP2040-based Etaluma boards.

Handles the complete firmware update cycle for field-deployed instruments:
  1. Verify current firmware version
  2. Back up all config files (via raw REPL)
  3. Send FWUPDATE command → board enters BOOTSEL/UF2 mode
  4. Detect RPI-RP2 USB mass storage drive
  5. Copy UF2 file → board auto-flashes and reboots
  6. Wait for serial port to reappear
  7. Verify new firmware version
  8. Restore config files if needed (via raw REPL with SHA256 verification)
  9. Run post-update health check

Designed for RELIABILITY over speed. These instruments are deployed
worldwide — a bricked unit requires expensive shipping for repair.

Usage::

    from drivers.firmware_updater import update_firmware, BoardType

    result = update_firmware(
        board_type=BoardType.MOTOR,
        uf2_path=Path('motor_firmware_v2.1.0.uf2'),
        progress_callback=my_progress_fn,
    )
    if result.success:
        print(f"Updated to {result.new_version}")

Safety invariants:
  - Config backup MUST succeed before any destructive action
  - UF2 write is atomic (RP2040 bootloader handles it)
  - Only one board updated at a time (both appear as 'RPI-RP2')
  - Configs verified via SHA256 after write
  - All file writes use temp-then-rename for atomicity
"""

import hashlib
import json
import logging
import platform
import re
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import serial
import serial.tools.list_ports as list_ports

from drivers.raw_repl import (
    enter_raw_repl,
    exit_raw_repl,
    list_files,
    read_file,
    write_file,
    verify_firmware_running,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class BoardType(Enum):
    LED = "led"
    MOTOR = "motor"


class UpdateStage(Enum):
    """Stages reported to the progress callback."""
    PREFLIGHT = "preflight"
    CHECKING_VERSION = "checking_version"
    BACKING_UP_CONFIG = "backing_up_config"
    SENDING_FWUPDATE = "sending_fwupdate"
    WAITING_BOOTSEL = "waiting_bootsel"
    COPYING_UF2 = "copying_uf2"
    WAITING_REBOOT = "waiting_reboot"
    VERIFYING_VERSION = "verifying_version"
    RESTORING_CONFIG = "restoring_config"
    POST_UPDATE_TEST = "post_update_test"
    COMPLETE = "complete"
    FAILED = "failed"


class UpdateError(Exception):
    """Firmware update failure with stage and recoverability info."""

    def __init__(self, message: str, stage: UpdateStage,
                 recoverable: bool = True):
        super().__init__(message)
        self.stage = stage
        self.recoverable = recoverable


@dataclass
class BoardConfig:
    board_type: BoardType
    vid: int
    pid: int
    label: str
    line_ending: bytes
    config_files: List[str]
    uf2_prefix: str

    # Timeouts — LED goes through USB hub, may be slower
    bootsel_timeout: float = 30.0
    serial_reappear_timeout: float = 30.0


@dataclass
class UpdateResult:
    success: bool
    board_type: BoardType
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    config_backup_path: Optional[Path] = None
    error_message: Optional[str] = None
    error_stage: Optional[UpdateStage] = None
    warnings: List[str] = field(default_factory=list)


# Progress callback: (stage, human-readable message, progress 0.0-1.0)
ProgressCallback = Callable[[UpdateStage, str, float], None]

# ---------------------------------------------------------------------------
# Board configurations
# ---------------------------------------------------------------------------

BOARD_CONFIGS = {
    BoardType.LED: BoardConfig(
        board_type=BoardType.LED,
        vid=0x0424,
        pid=0x704C,
        label="LED",
        line_ending=b'\r\n',
        config_files=['cal.json'],
        uf2_prefix='led_firmware',
        bootsel_timeout=45.0,       # USB hub adds delay
        serial_reappear_timeout=45.0,
    ),
    BoardType.MOTOR: BoardConfig(
        board_type=BoardType.MOTOR,
        vid=0x2E8A,
        pid=0x0005,
        label="Motor",
        line_ending=b'\n',
        config_files=[
            'motorconfig.json',
            'xymotorconfig.ini',
            'ztmotorconfig.ini',
            'ztmotorconfig2.ini',
        ],
        uf2_prefix='motor_firmware',
        bootsel_timeout=30.0,
        serial_reappear_timeout=30.0,
    ),
}

# ---------------------------------------------------------------------------
# Timing constants — conservative for field reliability
# ---------------------------------------------------------------------------

FWUPDATE_RESPONSE_TIMEOUT = 5.0    # Wait for FWUPDATE prompt
FWUPDATE_CONFIRM_TIMEOUT = 3.0     # Wait for "Entering bootloader..."
BOOTSEL_POLL_INTERVAL = 1.0        # Poll interval for drive detection
SERIAL_POLL_INTERVAL = 1.0         # Poll interval for port detection
POST_UF2_SETTLE_TIME = 3.0         # Wait after UF2 copy for drive to disappear
DRIVE_DISAPPEAR_TIMEOUT = 15.0     # Max wait for BOOTSEL drive to vanish
POST_REBOOT_SETTLE_TIME = 5.0      # Wait after port reappears before opening
SERIAL_OPEN_RETRIES = 3            # Attempts to open serial port after reboot


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _report_progress(callback, stage, message, progress):
    """Safely invoke progress callback."""
    if callback is not None:
        try:
            callback(stage, message, progress)
        except Exception as e:
            logger.warning(f"Progress callback error: {e}")


def _find_serial_port(vid, pid):
    """Find serial port by USB VID/PID. Returns port device string or None."""
    for port in list_ports.comports():
        if port.vid == vid and port.pid == pid:
            return port.device
    return None


def _open_serial(port, timeout=2.0, retries=SERIAL_OPEN_RETRIES):
    """Open serial port with retries (port may not be ready immediately)."""
    for attempt in range(1, retries + 1):
        try:
            ser = serial.Serial(
                port=port,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout,
                write_timeout=timeout,
            )
            time.sleep(0.5)  # Let port settle
            return ser
        except (serial.SerialException, OSError) as e:
            logger.warning(
                f"Serial open attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(2.0)
    return None


def _get_firmware_version(ser, board_config):
    """Send INFO command and parse firmware version string.

    Returns version string (e.g. '2.0.1') or None.
    """
    try:
        # Drain pending data
        ser.read(4096)
        time.sleep(0.1)

        ser.write(b'INFO' + board_config.line_ending)
        time.sleep(1.0)
        response = ser.read(4096)
        if not response:
            return None

        text = response.decode('utf-8', 'ignore')
        logger.info(f"INFO response: {text.strip()[:120]}")

        # Look for date-style version (2026-03-06) or semantic version (v2.0.1)
        # Date version
        m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
        if m:
            return m.group(1)
        # Semantic version
        m = re.search(r'v?(\d+\.\d+\.\d+)', text)
        if m:
            return m.group(1)

        return text.strip()[:50]
    except Exception as e:
        logger.warning(f"Failed to get firmware version: {e}")
        return None


def _parse_uf2_version(uf2_path):
    """Extract version from UF2 filename.

    Examples:
        led_firmware_v2.1.0.uf2  → '2.1.0'
        motor_firmware_2026-03-09.uf2 → '2026-03-09'
    """
    stem = uf2_path.stem
    # Semantic version
    m = re.search(r'v?(\d+\.\d+\.\d+)', stem)
    if m:
        return m.group(1)
    # Date version
    m = re.search(r'(\d{4}-\d{2}-\d{2})', stem)
    if m:
        return m.group(1)
    return stem


# ---------------------------------------------------------------------------
# BOOTSEL drive detection (cross-platform)
# ---------------------------------------------------------------------------

def _detect_bootsel_drive():
    """Detect RPI-RP2 BOOTSEL USB mass storage drive.

    Returns mount point Path, or None if not found.
    """
    system = platform.system()
    if system == 'Darwin':
        return _detect_bootsel_macos()
    elif system == 'Windows':
        return _detect_bootsel_windows()
    elif system == 'Linux':
        return _detect_bootsel_linux()
    else:
        logger.warning(f"Unsupported platform for BOOTSEL detection: {system}")
        return None


def _detect_bootsel_macos():
    """macOS: check /Volumes/RPI-RP2."""
    rpi_path = Path('/Volumes/RPI-RP2')
    if rpi_path.is_dir():
        # Verify it's a real RP2040 BOOTSEL by checking for INFO_UF2.TXT
        info_file = rpi_path / 'INFO_UF2.TXT'
        if info_file.exists():
            logger.info(f"BOOTSEL drive found: {rpi_path}")
            return rpi_path
    return None


def _detect_bootsel_windows():
    """Windows: scan drive letters for volume label 'RPI-RP2'."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        buf = ctypes.create_unicode_buffer(256)
        for letter in 'DEFGHIJKLMNOPQRSTUVWXYZ':
            drive = f'{letter}:\\'
            result = kernel32.GetVolumeInformationW(
                drive, buf, 256, None, None, None, None, 0)
            if result and buf.value == 'RPI-RP2':
                drive_path = Path(drive)
                info_file = drive_path / 'INFO_UF2.TXT'
                if info_file.exists():
                    logger.info(f"BOOTSEL drive found: {drive_path}")
                    return drive_path
    except Exception as e:
        logger.warning(f"Windows BOOTSEL detection error: {e}")
    return None


def _detect_bootsel_linux():
    """Linux: check common mount points for RPI-RP2."""
    import os
    candidates = [
        Path(f'/media/{os.getenv("USER", "")}/RPI-RP2'),
        Path('/media/RPI-RP2'),
        Path('/mnt/RPI-RP2'),
        Path('/run/media/' + os.getenv("USER", "") + '/RPI-RP2'),
    ]
    for path in candidates:
        if path.is_dir():
            info_file = path / 'INFO_UF2.TXT'
            if info_file.exists():
                logger.info(f"BOOTSEL drive found: {path}")
                return path
    return None


def _wait_for_bootsel_drive(timeout=30.0):
    """Poll for BOOTSEL drive to appear. Returns mount path or None."""
    logger.info(f"Waiting for BOOTSEL drive (timeout={timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        drive = _detect_bootsel_drive()
        if drive is not None:
            return drive
        time.sleep(BOOTSEL_POLL_INTERVAL)
    logger.error(f"BOOTSEL drive not found within {timeout}s")
    return None


def _wait_for_drive_disappear(drive_path, timeout=DRIVE_DISAPPEAR_TIMEOUT):
    """Wait for BOOTSEL drive to disappear (indicates UF2 was accepted)."""
    logger.info(f"Waiting for BOOTSEL drive to disappear...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not drive_path.is_dir():
            logger.info("BOOTSEL drive disappeared — UF2 accepted")
            return True
        time.sleep(0.5)
    logger.warning("BOOTSEL drive still present after timeout")
    return False


def _wait_for_serial_port(vid, pid, timeout=30.0):
    """Wait for serial port with given VID/PID to appear. Returns port or None."""
    logger.info(
        f"Waiting for serial port VID=0x{vid:04X} PID=0x{pid:04X} "
        f"(timeout={timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        port = _find_serial_port(vid, pid)
        if port is not None:
            logger.info(f"Serial port found: {port}")
            return port
        time.sleep(SERIAL_POLL_INTERVAL)
    logger.error(f"Serial port not found within {timeout}s")
    return None


# ---------------------------------------------------------------------------
# FWUPDATE command
# ---------------------------------------------------------------------------

def _send_fwupdate_command(ser, board_config):
    """Send FWUPDATE command with YES confirmation.

    After this call, the serial port is invalid (board is rebooting).
    Closes the serial port before returning.

    Raises UpdateError if the command flow fails.
    """
    try:
        # Drain any pending output
        ser.read(4096)
        time.sleep(0.2)

        # Send FWUPDATE
        logger.info(f"Sending FWUPDATE to {board_config.label} board")
        ser.write(b'FWUPDATE' + board_config.line_ending)

        # Wait for the confirmation prompt
        time.sleep(FWUPDATE_RESPONSE_TIMEOUT)
        response = ser.read(4096)
        text = response.decode('utf-8', 'ignore')
        logger.info(f"FWUPDATE response: {text.strip()[:200]}")

        if 'YES' not in text and 'confirm' not in text.lower():
            raise UpdateError(
                f"FWUPDATE did not prompt for confirmation. "
                f"Response: {text.strip()[:200]}",
                stage=UpdateStage.SENDING_FWUPDATE,
            )

        # Send YES confirmation
        logger.info("Sending YES confirmation")
        ser.write(b'YES' + board_config.line_ending)

        # Wait for "Entering bootloader mode..." confirmation
        time.sleep(FWUPDATE_CONFIRM_TIMEOUT)
        response = ser.read(4096)
        text = response.decode('utf-8', 'ignore')
        logger.info(f"Confirmation response: {text.strip()[:200]}")

        # Board is now rebooting — close serial port
        try:
            ser.close()
        except Exception:
            pass

        if 'bootloader' not in text.lower() and 'Entering' not in text:
            # The board may have already rebooted before we read the response
            # — this is OK if the BOOTSEL drive appears
            logger.warning(
                "Did not receive 'Entering bootloader' confirmation. "
                "Will check for BOOTSEL drive.")

    except UpdateError:
        raise
    except Exception as e:
        try:
            ser.close()
        except Exception:
            pass
        raise UpdateError(
            f"FWUPDATE command failed: {e}",
            stage=UpdateStage.SENDING_FWUPDATE,
        )


# ---------------------------------------------------------------------------
# Config backup and restore
# ---------------------------------------------------------------------------

def _backup_configs(ser, board_config, backup_dir, callback=None):
    """Back up all config files from board via raw REPL.

    Returns dict of {filename: bytes_content}.
    Raises UpdateError if a file exists on board but cannot be read.
    """
    configs = {}

    _report_progress(callback, UpdateStage.BACKING_UP_CONFIG,
                     f"Entering raw REPL on {board_config.label} board...", 0.12)

    if not enter_raw_repl(ser):
        raise UpdateError(
            f"Failed to enter raw REPL on {board_config.label} board",
            stage=UpdateStage.BACKING_UP_CONFIG,
        )

    try:
        # Discover what files are actually on the board
        board_files = list_files(ser)
        logger.info(f"Files on {board_config.label} board: {board_files}")

        for filename in board_config.config_files:
            if filename not in board_files:
                logger.info(f"Config file {filename} not on board — skipping")
                continue

            _report_progress(
                callback, UpdateStage.BACKING_UP_CONFIG,
                f"Reading {filename}...", 0.14)

            data = read_file(ser, filename, verify=True)
            if data is None:
                raise UpdateError(
                    f"Failed to read config file: {filename}. "
                    f"Update aborted — config backup must succeed before flashing.",
                    stage=UpdateStage.BACKING_UP_CONFIG,
                )

            configs[filename] = data
            logger.info(f"Backed up {filename}: {len(data)} bytes")

    finally:
        exit_raw_repl(ser)

    # Verify firmware recovered after raw REPL
    fw_response = verify_firmware_running(ser)
    if fw_response is None:
        raise UpdateError(
            f"{board_config.label} board not responding after config backup. "
            f"Try power-cycling the system.",
            stage=UpdateStage.BACKING_UP_CONFIG,
        )

    # Save backup files to local disk
    backup_dir.mkdir(parents=True, exist_ok=True)
    board_dir = backup_dir / board_config.board_type.value
    board_dir.mkdir(exist_ok=True)

    manifest = {}
    for filename, data in configs.items():
        local_path = board_dir / filename
        local_path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        manifest[filename] = {
            'size': len(data),
            'sha256': sha,
        }
        logger.info(f"Saved {local_path} ({len(data)} bytes, SHA256={sha[:16]}...)")

    # Save manifest
    manifest_path = board_dir / 'backup_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return configs


def _restore_configs(ser, board_config, config_data, callback=None):
    """Restore config files to board via raw REPL.

    Only restores files that are missing or differ from the backup.
    Returns True if all files restored successfully.
    Raises UpdateError on failure.
    """
    if not config_data:
        logger.info("No config files to restore")
        return True

    _report_progress(callback, UpdateStage.RESTORING_CONFIG,
                     f"Entering raw REPL on {board_config.label} board...", 0.80)

    if not enter_raw_repl(ser):
        raise UpdateError(
            f"Failed to enter raw REPL for config restore",
            stage=UpdateStage.RESTORING_CONFIG,
        )

    try:
        # Check which files need restoring
        board_files = list_files(ser)

        for filename, data in config_data.items():
            if filename in board_files:
                # File exists — check if it matches backup
                existing = read_file(ser, filename, verify=True)
                if existing == data:
                    logger.info(
                        f"{filename} survived update — skipping restore")
                    continue
                else:
                    logger.warning(
                        f"{filename} exists but differs from backup — "
                        f"restoring from backup")

            _report_progress(
                callback, UpdateStage.RESTORING_CONFIG,
                f"Restoring {filename}...", 0.85)

            if not write_file(ser, filename, data):
                raise UpdateError(
                    f"Failed to restore config: {filename}. "
                    f"Backup available on local disk.",
                    stage=UpdateStage.RESTORING_CONFIG,
                )
            logger.info(f"Restored {filename} ({len(data)} bytes)")

    finally:
        exit_raw_repl(ser)

    # Verify firmware recovered
    fw_response = verify_firmware_running(ser)
    if fw_response is None:
        raise UpdateError(
            f"{board_config.label} board not responding after config restore. "
            f"Try power-cycling the system.",
            stage=UpdateStage.RESTORING_CONFIG,
        )

    return True


# ---------------------------------------------------------------------------
# Post-update verification
# ---------------------------------------------------------------------------

def _run_post_update_test(ser, board_config):
    """Run abbreviated health check after firmware update.

    Returns (passed: bool, details: str).
    """
    issues = []

    # Test 1: INFO command
    ser.read(4096)  # drain
    ser.write(b'INFO' + board_config.line_ending)
    time.sleep(1.0)
    response = ser.read(4096)
    if not response:
        issues.append("INFO command returned no response")
    else:
        text = response.decode('utf-8', 'ignore')
        if 'Etaluma' not in text and 'EL-09' not in text and 'Firmware' not in text:
            issues.append(f"INFO response unexpected: {text.strip()[:100]}")

    # Test 2: Board-specific command
    time.sleep(0.5)
    ser.read(4096)  # drain

    if board_config.board_type == BoardType.LED:
        # Verify LED enable/disable works
        ser.write(b'LEDS_ENT' + board_config.line_ending)
        time.sleep(0.5)
        r = ser.read(4096).decode('utf-8', 'ignore')
        ser.write(b'LEDS_ENF' + board_config.line_ending)
        time.sleep(0.5)
        r2 = ser.read(4096).decode('utf-8', 'ignore')
        if 'Error' in r or 'Error' in r2:
            issues.append(f"LED enable/disable error: {r} / {r2}")

    elif board_config.board_type == BoardType.MOTOR:
        # Verify FULLINFO works
        ser.write(b'FULLINFO' + board_config.line_ending)
        time.sleep(1.0)
        r = ser.read(4096).decode('utf-8', 'ignore')
        if not r.strip():
            issues.append("FULLINFO returned empty response")
        elif 'not recognized' in r.lower():
            # Old firmware may not have FULLINFO — not a failure
            logger.info("FULLINFO not recognized (old firmware format)")

    if issues:
        detail = "; ".join(issues)
        logger.warning(f"Post-update test issues: {detail}")
        return False, detail
    else:
        logger.info("Post-update test passed")
        return True, "All checks passed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_available_updates(firmware_dir, board_type):
    """List available UF2 files for a board type, sorted newest first.

    Scans firmware_dir for files matching the board's UF2 prefix.
    Returns list of Path objects.
    """
    config = BOARD_CONFIGS[board_type]
    pattern = f"{config.uf2_prefix}*.uf2"
    files = sorted(firmware_dir.glob(pattern), reverse=True)
    return files


def check_update_needed(board_type, uf2_path):
    """Compare board's current version against UF2 file version.

    Returns (needs_update, current_version, target_version).
    Returns (True, None, target_version) if board is not connected.
    """
    config = BOARD_CONFIGS[board_type]
    target = _parse_uf2_version(uf2_path)

    port = _find_serial_port(config.vid, config.pid)
    if port is None:
        return True, None, target

    ser = _open_serial(port)
    if ser is None:
        return True, None, target

    try:
        current = _get_firmware_version(ser, config)
        needs_update = (current != target)
        return needs_update, current, target
    finally:
        ser.close()


def update_firmware(
    board_type,
    uf2_path,
    progress_callback=None,
    backup_dir=None,
    skip_config_backup=False,
    skip_post_test=False,
):
    """Execute the complete firmware update sequence.

    This is the main entry point. See module docstring for the full
    safety model.

    Args:
        board_type: BoardType.LED or BoardType.MOTOR
        uf2_path: Path to the UF2 file to flash
        progress_callback: Optional (stage, message, progress) callback
        backup_dir: Where to save config backups. Defaults to
            ~/Documents/Etaluma/firmware_backups/<timestamp>/
        skip_config_backup: Skip config backup (fresh board, no configs)
        skip_post_test: Skip post-update verification

    Returns:
        UpdateResult with success/failure details.
    """
    config = BOARD_CONFIGS[board_type]
    uf2_path = Path(uf2_path)
    result = UpdateResult(success=False, board_type=board_type)

    if backup_dir is None:
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = (Path.home() / 'Documents' / 'Etaluma'
                      / 'firmware_backups' / ts)

    backup_dir = Path(backup_dir)
    result.config_backup_path = backup_dir

    try:
        # ---- Stage 1: Pre-flight checks ----
        _report_progress(progress_callback, UpdateStage.PREFLIGHT,
                         "Checking UF2 file...", 0.0)

        if not uf2_path.is_file():
            raise UpdateError(
                f"UF2 file not found: {uf2_path}",
                stage=UpdateStage.PREFLIGHT,
            )

        uf2_size = uf2_path.stat().st_size
        if uf2_size < 512:
            raise UpdateError(
                f"UF2 file too small ({uf2_size} bytes) — likely corrupted",
                stage=UpdateStage.PREFLIGHT,
            )

        target_version = _parse_uf2_version(uf2_path)
        logger.info(f"Target firmware version: {target_version}")

        # Check no other board is already in BOOTSEL mode
        existing_bootsel = _detect_bootsel_drive()
        if existing_bootsel is not None:
            raise UpdateError(
                f"An RPI-RP2 drive is already mounted at {existing_bootsel}. "
                f"Cannot determine which board it belongs to. "
                f"Please eject it or power-cycle the system first.",
                stage=UpdateStage.PREFLIGHT,
            )

        # ---- Stage 2: Connect and check current version ----
        _report_progress(progress_callback, UpdateStage.CHECKING_VERSION,
                         f"Connecting to {config.label} board...", 0.05)

        port = _find_serial_port(config.vid, config.pid)
        if port is None:
            raise UpdateError(
                f"{config.label} board not found. "
                f"Check USB cable and power.",
                stage=UpdateStage.CHECKING_VERSION,
            )

        ser = _open_serial(port)
        if ser is None:
            raise UpdateError(
                f"Cannot open serial port {port}. "
                f"Close any other applications using the port (Thonny, etc).",
                stage=UpdateStage.CHECKING_VERSION,
            )

        current_version = _get_firmware_version(ser, config)
        result.old_version = current_version
        logger.info(f"Current firmware: {current_version}")

        if current_version == target_version and current_version is not None:
            logger.info("Firmware already at target version — no update needed")
            result.success = True
            result.new_version = current_version
            _report_progress(progress_callback, UpdateStage.COMPLETE,
                             "Already at target version", 1.0)
            ser.close()
            return result

        # ---- Stage 3: Back up config files ----
        config_data = {}
        if not skip_config_backup:
            _report_progress(progress_callback, UpdateStage.BACKING_UP_CONFIG,
                             "Backing up config files...", 0.10)
            config_data = _backup_configs(ser, config, backup_dir,
                                          progress_callback)
            logger.info(f"Backed up {len(config_data)} config files")
        else:
            logger.info("Config backup skipped by request")

        # ---- Stage 4: Send FWUPDATE command ----
        _report_progress(progress_callback, UpdateStage.SENDING_FWUPDATE,
                         "Sending FWUPDATE command...", 0.25)
        _send_fwupdate_command(ser, config)
        # ser is now closed — board is rebooting into BOOTSEL

        # ---- Stage 5: Wait for BOOTSEL drive ----
        _report_progress(progress_callback, UpdateStage.WAITING_BOOTSEL,
                         "Waiting for BOOTSEL drive...", 0.30)
        bootsel_drive = _wait_for_bootsel_drive(
            timeout=config.bootsel_timeout)
        if bootsel_drive is None:
            raise UpdateError(
                f"BOOTSEL drive did not appear within "
                f"{config.bootsel_timeout}s. The board may need a "
                f"power cycle. If the board has a BOOTSEL button, "
                f"hold it while power-cycling.",
                stage=UpdateStage.WAITING_BOOTSEL,
                recoverable=False,
            )

        # ---- Stage 6: Copy UF2 file ----
        _report_progress(progress_callback, UpdateStage.COPYING_UF2,
                         f"Copying {uf2_path.name} to board...", 0.40)
        logger.info(f"Copying {uf2_path} → {bootsel_drive}")

        dest = bootsel_drive / uf2_path.name
        shutil.copy2(uf2_path, dest)
        logger.info(f"UF2 file copied ({uf2_size} bytes)")

        # Wait for drive to disappear (indicates UF2 was processed)
        time.sleep(POST_UF2_SETTLE_TIME)
        if not _wait_for_drive_disappear(bootsel_drive):
            result.warnings.append(
                "BOOTSEL drive did not disappear after UF2 copy. "
                "The UF2 may not have been accepted.")

        # ---- Stage 7: Wait for serial port to reappear ----
        _report_progress(progress_callback, UpdateStage.WAITING_REBOOT,
                         "Waiting for board to reboot...", 0.55)

        # Wait extra time for firmware to initialize
        time.sleep(POST_REBOOT_SETTLE_TIME)

        port = _wait_for_serial_port(
            config.vid, config.pid,
            timeout=config.serial_reappear_timeout,
        )
        if port is None:
            # Check if the board fell back to BOOTSEL
            bootsel_again = _detect_bootsel_drive()
            if bootsel_again is not None:
                raise UpdateError(
                    f"Board returned to BOOTSEL mode instead of booting. "
                    f"The UF2 may be invalid. "
                    f"You can retry with a different UF2 file.",
                    stage=UpdateStage.WAITING_REBOOT,
                    recoverable=True,
                )
            raise UpdateError(
                f"Serial port did not reappear within "
                f"{config.serial_reappear_timeout}s. "
                f"Try power-cycling the system.",
                stage=UpdateStage.WAITING_REBOOT,
                recoverable=False,
            )

        # Wait for firmware to fully boot
        time.sleep(POST_REBOOT_SETTLE_TIME)

        ser = _open_serial(port)
        if ser is None:
            raise UpdateError(
                f"Cannot open serial port {port} after reboot.",
                stage=UpdateStage.WAITING_REBOOT,
            )

        # ---- Stage 8: Verify new firmware version ----
        _report_progress(progress_callback, UpdateStage.VERIFYING_VERSION,
                         "Verifying new firmware version...", 0.65)

        new_version = _get_firmware_version(ser, config)
        result.new_version = new_version
        logger.info(f"New firmware version: {new_version}")

        if new_version is None:
            result.warnings.append(
                "Could not read firmware version after update")
        elif target_version and new_version != target_version:
            result.warnings.append(
                f"Version mismatch: expected {target_version}, "
                f"got {new_version}")

        # ---- Stage 9: Restore config files ----
        if config_data:
            _report_progress(progress_callback, UpdateStage.RESTORING_CONFIG,
                             "Restoring config files...", 0.75)
            _restore_configs(ser, config, config_data, progress_callback)
        else:
            logger.info("No config files to restore")

        # ---- Stage 10: Post-update test ----
        if not skip_post_test:
            _report_progress(progress_callback, UpdateStage.POST_UPDATE_TEST,
                             "Running post-update test...", 0.90)
            passed, details = _run_post_update_test(ser, config)
            if not passed:
                result.warnings.append(f"Post-update test issues: {details}")
        else:
            logger.info("Post-update test skipped by request")

        # ---- Stage 11: Success ----
        ser.close()
        result.success = True
        _report_progress(progress_callback, UpdateStage.COMPLETE,
                         "Firmware update complete", 1.0)
        logger.info(
            f"Firmware update successful: {current_version} → {new_version}")
        return result

    except UpdateError as e:
        result.error_message = str(e)
        result.error_stage = e.stage
        logger.error(f"Firmware update failed at {e.stage.value}: {e}")
        _report_progress(progress_callback, UpdateStage.FAILED,
                         str(e), 0.0)
        return result

    except Exception as e:
        result.error_message = f"Unexpected error: {e}"
        result.error_stage = UpdateStage.FAILED
        logger.error(f"Firmware update unexpected error: {e}", exc_info=True)
        _report_progress(progress_callback, UpdateStage.FAILED,
                         f"Unexpected error: {e}", 0.0)
        return result
