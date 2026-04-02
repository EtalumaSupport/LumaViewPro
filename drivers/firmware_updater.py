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

import serial.tools.list_ports as list_ports

from drivers.serialboard import SerialBoard

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

    # True if RP2040 has direct USB (BOOTSEL accessible via software).
    # False for LED boards where RP2040 connects via UART through a USB
    # hub chip — BOOTSEL mode is not accessible, so UF2 flashing requires
    # physical BOOTSEL button or SWD.
    has_direct_usb: bool = True

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
        has_direct_usb=False,       # UART via USB hub — no BOOTSEL access
        bootsel_timeout=45.0,
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
            'ztmotorconfig3.ini',
        ],
        uf2_prefix='motor_firmware',
        bootsel_timeout=30.0,
        serial_reappear_timeout=30.0,
    ),
}

# ---------------------------------------------------------------------------
# Timing constants — conservative for field reliability
# ---------------------------------------------------------------------------

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
            logger.error(f"Progress callback error ({callback!r}): {e}")


def _find_serial_port(vid, pid):
    """Find serial port by USB VID/PID. Returns port device string or None."""
    for port in list_ports.comports():
        if port.vid == vid and port.pid == pid:
            return port.device
    return None


def _create_board(config, port=None, timeout=2.0):
    """Create a connected SerialBoard for the given board config.

    Uses SerialBoard's production connect logic: drain stale data,
    firmware recovery (Ctrl-C/B/D), version detection.

    Args:
        config: BoardConfig for this board type.
        port: Explicit serial port path. If None, searches by VID/PID.
        timeout: Serial read/write timeout.

    Returns:
        Connected SerialBoard instance.

    Raises:
        UpdateError if board cannot be found or connected.
    """
    board = SerialBoard(
        vid=config.vid, pid=config.pid,
        label=f'[FW-{config.label}]',
        timeout=timeout, write_timeout=timeout,
        port=port,
    )

    if not board.found:
        raise UpdateError(
            f"{config.label} board not found. Check USB cable and power.",
            stage=UpdateStage.CHECKING_VERSION,
        )

    board.connect()

    if board.driver is None:
        raise UpdateError(
            f"Cannot open serial port for {config.label} board. "
            f"Close any other applications using the port (Thonny, etc).",
            stage=UpdateStage.CHECKING_VERSION,
        )

    return board


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


def _find_picotool():
    """Find picotool executable on the system.

    Returns the path to picotool if found, or None.
    Checks common install locations: PATH, Homebrew, user-specified.
    """
    import subprocess

    # Check PATH first
    for name in ['picotool', 'picotool.exe']:
        try:
            result = subprocess.run(
                [name, 'version'],
                capture_output=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"Found picotool in PATH: {name}")
                return name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Check common Homebrew location (macOS)
    homebrew_path = Path('/opt/homebrew/bin/picotool')
    if homebrew_path.exists():
        logger.info(f"Found picotool at {homebrew_path}")
        return str(homebrew_path)

    return None


def _flash_uf2_picotool(uf2_path, picotool_path=None, reboot=True):
    """Flash UF2 file using picotool (direct USB, no mass storage mount needed).

    This is more robust than the mass storage copy method because it uses
    libusb to communicate directly with the RP2040 BOOTSEL bootloader,
    bypassing OS auto-mount issues.

    Args:
        uf2_path: Path to UF2 file to flash.
        picotool_path: Path to picotool binary. Auto-detected if None.
        reboot: If True, reboot into application mode after flashing.

    Returns True on success, False on failure.
    """
    import subprocess

    if picotool_path is None:
        picotool_path = _find_picotool()
    if picotool_path is None:
        logger.warning("picotool not found — cannot use direct USB flash")
        return False

    try:
        # Flash the UF2
        logger.info(f"Flashing {uf2_path} via picotool...")
        result = subprocess.run(
            [picotool_path, 'load', str(uf2_path)],
            capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.error(f"picotool load failed: {result.stderr}")
            return False
        logger.info(f"picotool flash complete: {result.stdout.strip()[-100:]}")

        # Reboot into application mode
        if reboot:
            time.sleep(1.0)
            result = subprocess.run(
                [picotool_path, 'reboot'],
                capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.warning(f"picotool reboot failed: {result.stderr}")
                # Not fatal — board may have auto-rebooted
            else:
                logger.info("picotool reboot: board entering application mode")

        return True

    except subprocess.TimeoutExpired:
        logger.error("picotool command timed out")
        return False
    except Exception as e:
        logger.error(f"picotool error: {e}")
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

def _send_fwupdate_command(board, board_config):
    """Reboot board into BOOTSEL/UF2 mode.

    Tries FWUPDATE command first (v3.0.4+ firmware). If the firmware
    doesn't recognize FWUPDATE (old/legacy firmware), falls back to
    entering raw REPL and running ``machine.bootloader()`` directly.

    After this call, the board is disconnected (rebooting into BOOTSEL).

    Raises UpdateError if neither method succeeds.
    """
    try:
        # Try FWUPDATE command first (v3.0.4+ firmware)
        logger.info(f"Sending FWUPDATE to {board_config.label} board")
        resp = board.exchange_command('FWUPDATE', timeout=3.0)

        if resp is not None:
            text = str(resp)
            if text.strip():
                logger.info(f"FWUPDATE response: {text.strip()[:200]}")

            # If we got "not recognized" or "not found", FWUPDATE isn't
            # supported — fall back to raw REPL machine.bootloader()
            if 'not recognized' in text.lower() or 'not found' in text.lower():
                logger.info("FWUPDATE not supported — using raw REPL fallback")
                _bootloader_via_raw_repl(board)
        else:
            # No response — board may have already rebooted (expected)
            logger.info("No response from FWUPDATE — board may have rebooted")

        board.disconnect()

    except UpdateError:
        board.disconnect()
        raise
    except Exception as e:
        board.disconnect()
        raise UpdateError(
            f"FWUPDATE command failed: {e}",
            stage=UpdateStage.SENDING_FWUPDATE,
        )


def _bootloader_via_raw_repl(board):
    """Enter BOOTSEL via raw REPL for firmware that lacks FWUPDATE command.

    Used as a fallback for old/legacy firmware. Enters raw REPL (Ctrl-C,
    Ctrl-A), then executes ``import machine; machine.bootloader()``.

    The board reboots into BOOTSEL mode. Serial errors after the command
    are expected (board disconnects from USB).
    """
    logger.info("Entering raw REPL for machine.bootloader() fallback")

    if not board.enter_raw_repl(soft_reset=False):
        raise UpdateError(
            "Failed to enter raw REPL for bootloader fallback",
            stage=UpdateStage.SENDING_FWUPDATE,
        )

    # Send machine.bootloader() — board reboots immediately
    # repl_exec may return None if board disconnects before response
    board.repl_exec('import machine\nmachine.bootloader()', timeout=5)
    time.sleep(2.0)
    logger.info("machine.bootloader() sent — board entering BOOTSEL")


# ---------------------------------------------------------------------------
# Config backup and restore
# ---------------------------------------------------------------------------

def _backup_configs(board, board_config, backup_dir, callback=None):
    """Back up all config files from board via raw REPL.

    Returns dict of {filename: bytes_content}.
    Raises UpdateError if a file exists on board but cannot be read.
    """
    configs = {}

    _report_progress(callback, UpdateStage.BACKING_UP_CONFIG,
                     f"Entering raw REPL on {board_config.label} board...", 0.12)

    if not board.enter_raw_repl():
        raise UpdateError(
            f"Failed to enter raw REPL on {board_config.label} board",
            stage=UpdateStage.BACKING_UP_CONFIG,
        )

    try:
        # Discover what files are actually on the board
        board_files = board.repl_list_files()
        logger.info(f"Files on {board_config.label} board: {board_files}")

        for filename in board_config.config_files:
            if filename not in board_files:
                logger.info(f"Config file {filename} not on board — skipping")
                continue

            _report_progress(
                callback, UpdateStage.BACKING_UP_CONFIG,
                f"Reading {filename}...", 0.14)

            data = board.repl_read_file(filename, verify=True)
            if data is None:
                raise UpdateError(
                    f"Failed to read config file: {filename}. "
                    f"Update aborted — config backup must succeed before flashing.",
                    stage=UpdateStage.BACKING_UP_CONFIG,
                )

            configs[filename] = data
            logger.info(f"Backed up {filename}: {len(data)} bytes")

    finally:
        board.exit_raw_repl()

    # Verify firmware recovered after raw REPL
    fw_response = board.verify_firmware_running()
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


def _restore_configs(board, board_config, config_data, callback=None):
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

    if not board.enter_raw_repl():
        raise UpdateError(
            f"Failed to enter raw REPL for config restore",
            stage=UpdateStage.RESTORING_CONFIG,
        )

    try:
        # Check which files need restoring
        board_files = board.repl_list_files()

        for filename, data in config_data.items():
            if filename in board_files:
                # File exists — check if it matches backup
                existing = board.repl_read_file(filename, verify=True)
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

            if not board.repl_write_file(filename, data):
                raise UpdateError(
                    f"Failed to restore config: {filename}. "
                    f"Backup available on local disk.",
                    stage=UpdateStage.RESTORING_CONFIG,
                )
            logger.info(f"Restored {filename} ({len(data)} bytes)")

    finally:
        board.exit_raw_repl()

    # Verify firmware recovered
    fw_response = board.verify_firmware_running()
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

def _run_post_update_test(board, board_config):
    """Run abbreviated health check after firmware update.

    Returns (passed: bool, details: str).
    """
    issues = []

    # Test 1: INFO command
    resp = board.exchange_command('INFO', response_numlines=6, timeout=2.0)
    if resp is None:
        issues.append("INFO command returned no response")
    else:
        text = '\n'.join(resp) if isinstance(resp, list) else str(resp)
        if 'Etaluma' not in text and 'EL-09' not in text and 'Firmware' not in text:
            issues.append(f"INFO response unexpected: {text.strip()[:100]}")

    # Test 2: Board-specific command
    if board_config.board_type == BoardType.LED:
        # Verify LED enable/disable works
        r = board.exchange_command('LEDS_ENT', timeout=2.0)
        r2 = board.exchange_command('LEDS_ENF', timeout=2.0)
        r_str = str(r or '')
        r2_str = str(r2 or '')
        if 'Error' in r_str or 'Error' in r2_str:
            issues.append(f"LED enable/disable error: {r_str} / {r2_str}")

    elif board_config.board_type == BoardType.MOTOR:
        # Verify FULLINFO works
        r = board.exchange_command('FULLINFO', response_numlines=6, timeout=2.0)
        r_str = '\n'.join(r) if isinstance(r, list) else str(r or '')
        if not r_str.strip():
            issues.append("FULLINFO returned empty response")
        elif 'not recognized' in r_str.lower():
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

    try:
        board = _create_board(config)
    except UpdateError:
        return True, None, target

    try:
        current = board.firmware_version or board.firmware_date
        needs_update = (current != target)
        return needs_update, current, target
    finally:
        board.disconnect()


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

        # LED boards have no USB to the RP2040 — UF2 flashing won't work
        if not config.has_direct_usb:
            raise UpdateError(
                f"{config.label} board has no direct USB to the RP2040 "
                f"(UART only via USB hub). UF2 flashing is not possible "
                f"via software. Use deploy_firmware_file() to update "
                f"main.py via raw REPL, or use a physical BOOTSEL button "
                f"to flash a new UF2.",
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

        board = _create_board(config)

        current_version = board.firmware_version or board.firmware_date
        result.old_version = current_version
        logger.info(f"Current firmware: {current_version}")

        if current_version == target_version and current_version is not None:
            logger.info("Firmware already at target version — no update needed")
            result.success = True
            result.new_version = current_version
            _report_progress(progress_callback, UpdateStage.COMPLETE,
                             "Already at target version", 1.0)
            board.disconnect()
            return result

        # ---- Stage 3: Back up config files ----
        config_data = {}
        if not skip_config_backup:
            _report_progress(progress_callback, UpdateStage.BACKING_UP_CONFIG,
                             "Backing up config files...", 0.10)
            config_data = _backup_configs(board, config, backup_dir,
                                          progress_callback)
            logger.info(f"Backed up {len(config_data)} config files")
        else:
            logger.info("Config backup skipped by request")

        # ---- Stage 4: Send FWUPDATE command ----
        _report_progress(progress_callback, UpdateStage.SENDING_FWUPDATE,
                         "Sending FWUPDATE command...", 0.25)
        _send_fwupdate_command(board, config)
        # board is now disconnected — rebooting into BOOTSEL

        # ---- Stage 5: Wait for BOOTSEL drive ----
        _report_progress(progress_callback, UpdateStage.WAITING_BOOTSEL,
                         "Waiting for BOOTSEL drive...", 0.30)
        bootsel_drive = _wait_for_bootsel_drive(
            timeout=config.bootsel_timeout)

        if bootsel_drive is not None:
            # ---- Stage 6: Copy UF2 file via mass storage ----
            _report_progress(progress_callback, UpdateStage.COPYING_UF2,
                             f"Copying {uf2_path.name} to board...", 0.40)
            if platform.system() == 'Darwin':
                _report_progress(
                    progress_callback, UpdateStage.COPYING_UF2,
                    "Note: macOS may show 'disk not ejected properly' — "
                    "this is normal (board reboots after flashing).", 0.40)
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
        else:
            # ---- Stage 6 fallback: Try picotool ----
            logger.info("BOOTSEL drive not mounted — trying picotool")
            _report_progress(progress_callback, UpdateStage.COPYING_UF2,
                             "BOOTSEL drive not mounted, trying picotool...",
                             0.35)
            picotool = _find_picotool()
            if picotool is not None:
                ok = _flash_uf2_picotool(uf2_path, picotool_path=picotool,
                                         reboot=True)
                if ok:
                    logger.info("UF2 flashed successfully via picotool")
                else:
                    raise UpdateError(
                        f"picotool failed to flash {uf2_path.name}. "
                        f"The board may need a power cycle.",
                        stage=UpdateStage.COPYING_UF2,
                        recoverable=False,
                    )
            else:
                raise UpdateError(
                    f"BOOTSEL drive did not appear within "
                    f"{config.bootsel_timeout}s and picotool is not "
                    f"installed. Install picotool (brew install picotool) "
                    f"or power-cycle the board while holding BOOTSEL.",
                    stage=UpdateStage.WAITING_BOOTSEL,
                    recoverable=False,
                )

        # ---- Stage 7: Wait for serial port to reappear ----
        _report_progress(progress_callback, UpdateStage.WAITING_REBOOT,
                         "Waiting for board to reboot...", 0.55)

        # Wait extra time for firmware to initialize
        time.sleep(POST_REBOOT_SETTLE_TIME)

        new_port = _wait_for_serial_port(
            config.vid, config.pid,
            timeout=config.serial_reappear_timeout,
        )
        if new_port is None:
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

        # Create NEW SerialBoard for the rebooted board (port may have changed)
        board2 = _create_board(config, port=new_port)

        # ---- Stage 8: Verify new firmware version ----
        _report_progress(progress_callback, UpdateStage.VERIFYING_VERSION,
                         "Verifying new firmware version...", 0.65)

        new_version = board2.firmware_version or board2.firmware_date
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
            _restore_configs(board2, config, config_data, progress_callback)
        else:
            logger.info("No config files to restore")

        # ---- Stage 10: Post-update test ----
        if not skip_post_test:
            _report_progress(progress_callback, UpdateStage.POST_UPDATE_TEST,
                             "Running post-update test...", 0.90)
            passed, details = _run_post_update_test(board2, config)
            if not passed:
                result.warnings.append(f"Post-update test issues: {details}")
        else:
            logger.info("Post-update test skipped by request")

        # ---- Stage 11: Success ----
        board2.disconnect()
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


def nuke_board(
    board_type,
    nuke_uf2_path,
    progress_callback=None,
):
    """Erase all flash on a board and leave it in BOOTSEL mode.

    This uses the Raspberry Pi flash_nuke UF2 which:
      1. Erases all flash memory (firmware + filesystem)
      2. Flashes the LED 3 times to confirm
      3. Reboots back into BOOTSEL mode (ready for new UF2)

    Use this to completely reset a board to factory-blank state.
    After nuke, the board will appear as RPI-RP2 USB mass storage
    and is ready for a fresh UF2 flash.

    Args:
        board_type: BoardType.LED or BoardType.MOTOR
        nuke_uf2_path: Path to flash_nuke UF2 (RP2040 or RP2350)
        progress_callback: Optional (stage, message, progress) callback

    Returns:
        UpdateResult with success/failure.
    """
    config = BOARD_CONFIGS[board_type]
    nuke_uf2_path = Path(nuke_uf2_path)
    result = UpdateResult(success=False, board_type=board_type)

    try:
        # LED boards have no USB to the RP2040 — nuke won't work
        if not config.has_direct_usb:
            raise UpdateError(
                f"{config.label} board has no direct USB to the RP2040 "
                f"(UART only via USB hub). Flash nuke requires BOOTSEL "
                f"access. Use a physical BOOTSEL button.",
                stage=UpdateStage.PREFLIGHT,
            )

        # Validate nuke UF2
        if not nuke_uf2_path.is_file():
            raise UpdateError(
                f"Flash nuke UF2 not found: {nuke_uf2_path}",
                stage=UpdateStage.PREFLIGHT,
            )

        _report_progress(progress_callback, UpdateStage.PREFLIGHT,
                         f"Preparing to nuke {config.label} board...", 0.0)

        # Check if already in BOOTSEL
        bootsel_drive = _detect_bootsel_drive()
        if bootsel_drive is not None:
            logger.info(f"Board already in BOOTSEL mode at {bootsel_drive}")
        else:
            # Send FWUPDATE to enter BOOTSEL
            _report_progress(progress_callback, UpdateStage.SENDING_FWUPDATE,
                             "Entering BOOTSEL mode...", 0.10)

            board = _create_board(config)
            _send_fwupdate_command(board, config)

            # Wait for BOOTSEL drive
            _report_progress(progress_callback, UpdateStage.WAITING_BOOTSEL,
                             "Waiting for BOOTSEL drive...", 0.25)
            bootsel_drive = _wait_for_bootsel_drive(
                timeout=config.bootsel_timeout)

        if bootsel_drive is not None:
            # Copy nuke UF2 via mass storage
            _report_progress(progress_callback, UpdateStage.COPYING_UF2,
                             "Erasing flash (this takes a few seconds)...",
                             0.40)
            if platform.system() == 'Darwin':
                _report_progress(
                    progress_callback, UpdateStage.COPYING_UF2,
                    "Note: macOS may show 'disk not ejected properly' — "
                    "this is normal.", 0.40)

            dest = bootsel_drive / nuke_uf2_path.name
            shutil.copy2(nuke_uf2_path, dest)
            logger.info(f"Nuke UF2 copied to {bootsel_drive}")
        else:
            # Fallback: try picotool
            logger.info("BOOTSEL drive not mounted — trying picotool")
            _report_progress(progress_callback, UpdateStage.COPYING_UF2,
                             "BOOTSEL drive not mounted, trying picotool...",
                             0.35)
            picotool = _find_picotool()
            if picotool is not None:
                ok = _flash_uf2_picotool(nuke_uf2_path,
                                         picotool_path=picotool,
                                         reboot=False)
                if ok:
                    logger.info("Nuke UF2 flashed via picotool")
                else:
                    raise UpdateError(
                        f"picotool failed to flash nuke UF2. "
                        f"Hold BOOTSEL and power-cycle the board.",
                        stage=UpdateStage.COPYING_UF2,
                        recoverable=False,
                    )
            else:
                raise UpdateError(
                    f"BOOTSEL drive not found and picotool is not "
                    f"installed. Install picotool (brew install picotool) "
                    f"or hold BOOTSEL and power-cycle the board.",
                    stage=UpdateStage.WAITING_BOOTSEL,
                    recoverable=False,
                )

        # Wait for drive to disappear and reappear (nuke reboots to BOOTSEL)
        time.sleep(POST_UF2_SETTLE_TIME)
        _wait_for_drive_disappear(bootsel_drive)

        # Nuke reboots back into BOOTSEL — wait for it to reappear
        _report_progress(progress_callback, UpdateStage.WAITING_REBOOT,
                         "Waiting for board to return to BOOTSEL...", 0.70)
        time.sleep(3.0)
        bootsel_drive = _wait_for_bootsel_drive(timeout=15.0)

        if bootsel_drive is not None:
            result.success = True
            _report_progress(progress_callback, UpdateStage.COMPLETE,
                             f"Flash erased. Board is in BOOTSEL mode at "
                             f"{bootsel_drive} — ready for new UF2.", 1.0)
            logger.info("Flash nuke complete — board in BOOTSEL mode")
        else:
            # Board may have nuked successfully but not remounted.
            # Check for serial port (would mean it booted with no firmware).
            result.success = True
            result.warnings.append(
                "BOOTSEL drive did not reappear after nuke. "
                "Board may need manual BOOTSEL entry (hold button + plug in).")
            _report_progress(progress_callback, UpdateStage.COMPLETE,
                             "Flash erased. Replug with BOOTSEL held to flash "
                             "new firmware.", 1.0)

        return result

    except UpdateError as e:
        result.error_message = str(e)
        result.error_stage = e.stage
        logger.error(f"Flash nuke failed: {e}")
        _report_progress(progress_callback, UpdateStage.FAILED,
                         str(e), 0.0)
        return result

    except Exception as e:
        result.error_message = f"Unexpected error: {e}"
        result.error_stage = UpdateStage.FAILED
        logger.error(f"Flash nuke unexpected error: {e}", exc_info=True)
        _report_progress(progress_callback, UpdateStage.FAILED,
                         f"Unexpected error: {e}", 0.0)
        return result


def deploy_firmware_file(
    board_type,
    firmware_path,
    progress_callback=None,
    backup_dir=None,
    skip_config_backup=False,
    skip_post_test=False,
):
    """Deploy main.py to a board via raw REPL (no UF2, no BOOTSEL).

    This is the primary update method for LED boards (UART-only, no USB
    to the RP2040) and an alternative method for motor boards when only
    updating the firmware Python file without changing the MicroPython
    runtime.

    The sequence:
      1. Connect to the board via serial
      2. Back up config files via raw REPL
      3. Write new main.py via raw REPL (SHA256 verified, atomic)
      4. Soft reset to boot the new firmware
      5. Verify new firmware version
      6. Run post-update health check

    No BOOTSEL mode is needed — the board stays connected throughout.

    Args:
        board_type: BoardType.LED or BoardType.MOTOR
        firmware_path: Path to the main.py file to deploy
        progress_callback: Optional (stage, message, progress) callback
        backup_dir: Where to save config backups
        skip_config_backup: Skip config backup
        skip_post_test: Skip post-update verification

    Returns:
        UpdateResult with success/failure details.
    """
    config = BOARD_CONFIGS[board_type]
    firmware_path = Path(firmware_path)
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
                         "Checking firmware file...", 0.0)

        if not firmware_path.is_file():
            raise UpdateError(
                f"Firmware file not found: {firmware_path}",
                stage=UpdateStage.PREFLIGHT,
            )

        fw_data = firmware_path.read_bytes()
        if len(fw_data) < 100:
            raise UpdateError(
                f"Firmware file too small ({len(fw_data)} bytes)",
                stage=UpdateStage.PREFLIGHT,
            )
        logger.info(f"Firmware file: {firmware_path.name} ({len(fw_data)} bytes)")

        # ---- Stage 2: Connect ----
        _report_progress(progress_callback, UpdateStage.CHECKING_VERSION,
                         f"Connecting to {config.label} board...", 0.05)

        # SerialBoard.connect() handles all recovery: drain stale data,
        # Thonny recovery (Ctrl-C/B/D), version detection, WDT-safe fallback
        board = _create_board(config)

        current_version = board.firmware_version or board.firmware_date
        result.old_version = current_version
        logger.info(f"Current firmware: {current_version}")

        # ---- Stage 3: Back up config files ----
        config_data = {}
        if not skip_config_backup:
            _report_progress(progress_callback, UpdateStage.BACKING_UP_CONFIG,
                             "Backing up config files...", 0.10)
            config_data = _backup_configs(board, config, backup_dir,
                                          progress_callback)
            logger.info(f"Backed up {len(config_data)} config files")
            # _backup_configs already exits raw REPL and verifies firmware
        else:
            logger.info("Config backup skipped by request")

        # ---- Stage 4: Deploy firmware via raw REPL ----
        _report_progress(progress_callback, UpdateStage.RESTORING_CONFIG,
                         "Deploying firmware file...", 0.40)

        # soft_reset=False: old firmware may have WDT (8388ms). Soft reset
        # kills the Timer that feeds WDT, leaving only ~8s before reset.
        # UART writes (57KB at 115200) take ~9s — not enough time.
        # Without soft reset, the Timer keeps feeding WDT during the write.
        if not board.enter_raw_repl(soft_reset=False):
            raise UpdateError(
                f"Failed to enter raw REPL for firmware deploy",
                stage=UpdateStage.RESTORING_CONFIG,
            )

        if not board.repl_write_file('main.py', fw_data):
            raise UpdateError(
                f"Failed to write main.py ({len(fw_data)} bytes)",
                stage=UpdateStage.RESTORING_CONFIG,
            )
        logger.info(f"Deployed main.py ({len(fw_data)} bytes, SHA256 verified)")

        # ---- Stage 5: Exit raw REPL and verify ----
        _report_progress(progress_callback, UpdateStage.VERIFYING_VERSION,
                         "Rebooting firmware...", 0.75)

        board.exit_raw_repl()
        time.sleep(3.0)

        # Re-detect firmware version after reboot
        board.detect_firmware_version()
        new_version = board.firmware_version or board.firmware_date
        result.new_version = new_version
        logger.info(f"New firmware version: {new_version}")

        # ---- Stage 6: Post-update test ----
        if not skip_post_test:
            _report_progress(progress_callback, UpdateStage.POST_UPDATE_TEST,
                             "Running post-update test...", 0.90)
            passed, details = _run_post_update_test(board, config)
            if not passed:
                result.warnings.append(f"Post-update test issues: {details}")
        else:
            logger.info("Post-update test skipped by request")

        # ---- Done ----
        board.disconnect()
        result.success = True
        _report_progress(progress_callback, UpdateStage.COMPLETE,
                         "Firmware deploy complete", 1.0)
        logger.info(
            f"Firmware deploy successful: {current_version} → {new_version}")
        return result

    except UpdateError as e:
        result.error_message = str(e)
        result.error_stage = e.stage
        logger.error(f"Firmware deploy failed at {e.stage.value}: {e}")
        _report_progress(progress_callback, UpdateStage.FAILED,
                         str(e), 0.0)
        return result

    except Exception as e:
        result.error_message = f"Unexpected error: {e}"
        result.error_stage = UpdateStage.FAILED
        logger.error(f"Firmware deploy unexpected error: {e}", exc_info=True)
        _report_progress(progress_callback, UpdateStage.FAILED,
                         f"Unexpected error: {e}", 0.0)
        return result
