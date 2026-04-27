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

import contextlib
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
# mpremote exceptions surface through SerialBoard's raw-REPL methods
# (drivers/mpremote_transport.py — plan §2 Phase 2). Wrap them in
# UpdateError with the calling stage per analysis §2 R7 so callers see
# consistent structured errors regardless of the transport backing.
from mpremote.transport import TransportError, TransportExecError

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


@dataclass
class UpgradeResult:
    """Result of an FW4.0 field upgrade (§13.X of FIRMWARE_PLAN.md).

    Distinct from ``UpdateResult`` — the upgrade orchestrates a
    per-run probe + backup + bundle-deploy + verify + telemetry
    sequence, and carries more structured state for the LVP UI and
    CLI exit code mapping.

    LVP surfaces this in a popup; CLI maps ``exit_code`` to its
    process exit. Both read the same error_code / error_message for
    user-facing text. Fields are populated progressively as the run
    advances — early-abort results may have many Nones.
    """
    success: bool
    board_type: BoardType
    exit_code: int
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    probe_classification: Optional[str] = None
    config_backup_path: Optional[Path] = None
    telemetry_log_path: Optional[Path] = None
    overwritable_flags: Optional[Dict[str, int]] = None
    files_written: List[str] = field(default_factory=list)
    files_skipped_overwritable: List[str] = field(default_factory=list)
    error_code: Optional[str] = None
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


@contextlib.contextmanager
def _wrap_mpremote_errors(stage: "UpdateStage", context: str = ""):
    """Translate mpremote Transport exceptions to UpdateError.

    Raw-REPL file I/O now routes through ``drivers.mpremote_transport``,
    which surfaces ``TransportError`` (link / protocol) and
    ``TransportExecError`` (device-side Python traceback). Neither maps
    cleanly onto ``UpdateResult.error_stage``, so callers wrap the
    raw-REPL block in this context manager. Per analysis §2 R7,
    ``TransportExecError.error_output`` (the device traceback) is
    preserved in the ``UpdateError.message`` for field debugging.
    """
    try:
        yield
    except TransportExecError as e:
        device_tb = (e.error_output or "").strip() if isinstance(
            e.error_output, str
        ) else (e.error_output or b"").decode("utf-8", errors="replace").strip()
        msg_prefix = f"Device error during {context}" if context else "Device error"
        raise UpdateError(
            f"{msg_prefix}: {device_tb or e}",
            stage=stage,
        ) from e
    except TransportError as e:
        msg_prefix = (
            f"Transport error during {context}" if context else "Transport error"
        )
        raise UpdateError(
            f"{msg_prefix}: {e}",
            stage=stage,
        ) from e


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


def _find_mpy_cross():
    """Locate the mpy-cross binary. Returns Path or None.

    mpy-cross is shipped with the micropython PyPI package (usually at
    $VIRTUAL_ENV/bin/mpy-cross or /usr/local/bin/mpy-cross).
    """
    import shutil as _shutil
    found = _shutil.which('mpy-cross')
    if found:
        return Path(found)
    return None


def _mpy_cross_compile(py_path, out_path, mpy_cross_path=None):
    """Compile `py_path` (a .py file) to `out_path` (a .mpy file) via
    mpy-cross. Returns True on success, False on failure."""
    import subprocess
    if mpy_cross_path is None:
        mpy_cross_path = _find_mpy_cross()
    if mpy_cross_path is None:
        raise UpdateError(
            "mpy-cross not found on PATH. Install the micropython "
            "PyPI package (pip install micropython) or set PATH to "
            "include your mpy-cross binary. Required for compile_mpy=True.",
            stage=UpdateStage.PREFLIGHT,
        )

    py_path = Path(py_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [str(mpy_cross_path), '-o', str(out_path), str(py_path)],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"mpy-cross timed out compiling {py_path}")
        return False
    if result.returncode != 0:
        logger.error(
            f"mpy-cross failed on {py_path}: {result.stderr.strip()}"
        )
        return False
    logger.info(
        f"mpy-cross: {py_path.name} "
        f"({py_path.stat().st_size}B) -> "
        f"{out_path.name} ({out_path.stat().st_size}B)"
    )
    return True


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

            # FW4.0 two-step: firmware asks for explicit CONFIRM before
            # actually rebooting. Response shape:
            #   {"ok":true,"cmd":"FWUPDATE",
            #    "msg":"send FWUPDATE CONFIRM to reboot into UF2 bootloader"}
            # See docs/FW40_COMMAND_REFERENCE.md §5 (FWUPDATE).
            # v3.0.x boards reboot silently on the first FWUPDATE and
            # never emit this phrase, so string detection is safe to
            # do unconditionally and keeps the helper protocol-agnostic
            # (works even if FW4.0 INFO malformed → LVP protocol-version
            # fell back to LEGACY, as observed on SN 11016 bench).
            elif 'FWUPDATE CONFIRM' in text.upper():
                logger.info(
                    "FW4.0 two-step FWUPDATE detected — sending CONFIRM"
                )
                try:
                    # No response expected — board reboots immediately.
                    # A None / timeout here is success.
                    board.exchange_command('FWUPDATE CONFIRM', timeout=3.0)
                except Exception as e:
                    logger.info(
                        f"FWUPDATE CONFIRM no response (expected on reboot): {e}"
                    )
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
        with _wrap_mpremote_errors(
            UpdateStage.BACKING_UP_CONFIG, context="config backup"
        ):
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
        with _wrap_mpremote_errors(
            UpdateStage.RESTORING_CONFIG, context="config restore"
        ):
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

    Recognizes all three INFO response shapes:
      - LEGACY (2024/2025 firmware): multi-line text, contains
        'Etaluma' / 'EL-09' / 'Firmware'.
      - v3.5 LED short-text: single line starting with 'INFO ' with
        'sub=LED' and 'proto=3.5'. Per FIRMWARE_PROTOCOL.md §3.4.
      - FW4.0 motor JSON: line starting with '{' containing
        '"cmd":"INFO"' or '"subsystem":"MOTOR"'. Per FIRMWARE_PROTOCOL.md §2.6.

    Returns (passed: bool, details: str).
    """
    issues = []

    # Test 1: INFO command. Use response_numlines=6 to handle the
    # LEGACY multi-line case; v3.5 single-line and FW4.0 JSON are
    # caught on the first line.
    resp = board.exchange_command('INFO', response_numlines=6, timeout=2.0)
    if resp is None:
        issues.append("INFO command returned no response")
    else:
        text = '\n'.join(resp) if isinstance(resp, list) else str(resp)
        first_line = text.lstrip().split('\n', 1)[0]
        recognized = (
            # LEGACY 2024/2025 multi-line text
            'Etaluma' in text or 'EL-09' in text or 'Firmware' in text
            # v3.5 LED single-line
            or (first_line.startswith('INFO ') and 'proto=3.5' in first_line)
            # FW4.0 motor JSON
            or (first_line.startswith('{') and (
                '"subsystem"' in first_line or '"cmd":"INFO"' in first_line
                or '"cmd": "INFO"' in first_line))
        )
        if not recognized:
            issues.append(f"INFO response unexpected: {text.strip()[:100]}")

    # Test 2: Board-specific command. Branch on the protocol the board
    # advertised post-update — the connect-time detector populates
    # `protocol_version` so we use it here rather than re-scraping the
    # INFO text.
    proto_value = getattr(getattr(board, 'protocol_version', None),
                          'value', '') or ''

    if board_config.board_type == BoardType.LED:
        if proto_value == 'v35':
            # v3.5: LED_ENABLE / LED_DISABLE replace LEDS_ENT / LEDS_ENF.
            # Two-line ack (RE: + OK); exchange_command auto-drains the
            # echo so resp is the OK line.
            r = board.exchange_command('LED_ENABLE ALL', timeout=2.0)
            r2 = board.exchange_command('LED_DISABLE ALL', timeout=2.0)
            if (r or '') != 'OK' or (r2 or '') != 'OK':
                issues.append(
                    f"LED enable/disable did not return OK: "
                    f"{r!r} / {r2!r}")
        else:
            # LEGACY 2024/2025 firmware (or pre-v3.5 dev firmware).
            r = board.exchange_command('LEDS_ENT', timeout=2.0)
            r2 = board.exchange_command('LEDS_ENF', timeout=2.0)
            r_str = str(r or '')
            r2_str = str(r2 or '')
            if 'Error' in r_str or 'Error' in r2_str:
                issues.append(
                    f"LED enable/disable error: {r_str} / {r2_str}")

    elif board_config.board_type == BoardType.MOTOR:
        if proto_value == 'v4':
            # FW4.0 motor JSON path. Use exchange_json so the response
            # is parsed as a dict; the legacy text-FULLINFO would
            # produce UNKNOWN_CMD on FW4.0 firmware.
            try:
                resp_json = board.exchange_json({'cmd': 'INFO'}, timeout=2.0)
            except Exception as e:
                resp_json = None
                issues.append(f"FW4.0 motor INFO JSON exchange raised: {e}")
            if resp_json is None or not isinstance(resp_json, dict):
                issues.append("FW4.0 motor INFO returned no/invalid JSON")
            elif resp_json.get('ok') is not True:
                issues.append(
                    f"FW4.0 motor INFO not ok: {resp_json}")
        else:
            # LEGACY motor firmware (pre-v3.0).
            r = board.exchange_command('FULLINFO', response_numlines=6, timeout=2.0)
            r_str = '\n'.join(r) if isinstance(r, list) else str(r or '')
            if not r_str.strip():
                issues.append("FULLINFO returned empty response")
            elif 'not recognized' in r_str.lower():
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


def flash_uf2_direct(
    board_type,
    uf2_path,
    progress_callback=None,
    bootsel_timeout=None,
):
    """Flash a UF2 to a board that is already in BOOTSEL mode.

    Use this when the board is bricked / non-responsive and was
    forced into BOOTSEL via physical means (BOOTSEL button short,
    SWD), so the normal update_firmware() flow — which expects a
    responsive board to back configs up and send FWUPDATE — cannot
    run.

    Flow:
      1. Verify the board's `has_direct_usb` (LED cannot enter
         BOOTSEL via software or hardware on a sealed unit; flashing
         an LED UF2 requires TP96 + TP8/TP11 on the bench).
      2. Detect the RPI-RP2 mass-storage drive (wait up to
         `bootsel_timeout` for it to appear).
      3. Copy the UF2. Fall back to picotool if drive detection
         fails (same fallback path update_firmware uses).
      4. Wait for drive to disappear (UF2 accepted), board to
         re-enumerate on USB CDC, and INFO to succeed.
      5. Report the new version.

    Configs are NOT restored — the board was bricked before this
    call, so there is nothing to back up, and the caller is
    responsible for restoring configs afterward (typically via
    `_restore_configs` against a previous `firmware_backups/`
    snapshot, or via `deploy_firmware_file`'s backup/restore cycle
    on a subsequent call).

    Args:
        board_type: BoardType.MOTOR (LED rejected — no BOOTSEL path).
        uf2_path: UF2 file to flash (MicroPython runtime build).
        progress_callback: Optional (stage, message, progress) callback.
        bootsel_timeout: Seconds to wait for BOOTSEL drive to appear.
            Defaults to the board's configured bootsel_timeout.

    Returns:
        UpdateResult with success/failure details. result.new_version
        populated on success.
    """
    config = BOARD_CONFIGS[board_type]
    uf2_path = Path(uf2_path)
    result = UpdateResult(success=False, board_type=board_type)

    if bootsel_timeout is None:
        bootsel_timeout = config.bootsel_timeout

    try:
        # ---- Pre-flight ----
        _report_progress(progress_callback, UpdateStage.PREFLIGHT,
                         f"Preparing direct UF2 flash for "
                         f"{config.label}...", 0.0)

        if not config.has_direct_usb:
            raise UpdateError(
                f"{config.label} board has no direct USB to the RP2040 "
                f"(UART only via USB hub). Direct UF2 flash via BOOTSEL "
                f"requires physical access to TP96 / TP8 / TP11 and is "
                f"out of scope for this method.",
                stage=UpdateStage.PREFLIGHT,
            )

        if not uf2_path.is_file():
            raise UpdateError(
                f"UF2 file not found: {uf2_path}",
                stage=UpdateStage.PREFLIGHT,
            )
        uf2_size = uf2_path.stat().st_size
        if uf2_size < 512:
            raise UpdateError(
                f"UF2 file too small ({uf2_size} bytes) — likely "
                f"corrupted",
                stage=UpdateStage.PREFLIGHT,
            )

        target_version = _parse_uf2_version(uf2_path)
        logger.info(
            f"Direct-flash target: {uf2_path.name} "
            f"(v{target_version})")

        # ---- Wait for BOOTSEL drive ----
        _report_progress(progress_callback, UpdateStage.WAITING_BOOTSEL,
                         "Waiting for BOOTSEL drive "
                         "(short BOOTSEL pin + power cycle)...", 0.10)
        bootsel_drive = _wait_for_bootsel_drive(timeout=bootsel_timeout)

        # ---- Flash ----
        if bootsel_drive is not None:
            _report_progress(progress_callback, UpdateStage.COPYING_UF2,
                             f"Copying {uf2_path.name} to "
                             f"{bootsel_drive}...", 0.30)
            if platform.system() == 'Darwin':
                _report_progress(
                    progress_callback, UpdateStage.COPYING_UF2,
                    "Note: macOS may show 'disk not ejected properly' — "
                    "this is normal (board reboots after flashing).",
                    0.30)
            dest = bootsel_drive / uf2_path.name
            shutil.copy2(uf2_path, dest)
            logger.info(f"UF2 copied ({uf2_size} bytes)")

            time.sleep(POST_UF2_SETTLE_TIME)
            if not _wait_for_drive_disappear(bootsel_drive):
                result.warnings.append(
                    "BOOTSEL drive did not disappear after UF2 copy. "
                    "The UF2 may not have been accepted.")
        else:
            # Same picotool fallback update_firmware uses.
            logger.info("BOOTSEL drive not mounted — trying picotool")
            _report_progress(progress_callback, UpdateStage.COPYING_UF2,
                             "BOOTSEL drive not mounted; trying "
                             "picotool...", 0.25)
            picotool = _find_picotool()
            if picotool is None:
                raise UpdateError(
                    f"BOOTSEL drive did not appear within "
                    f"{bootsel_timeout}s and picotool is not installed. "
                    f"Install picotool (brew install picotool) or verify "
                    f"the BOOTSEL pin short is making good contact "
                    f"during power-up.",
                    stage=UpdateStage.WAITING_BOOTSEL,
                    recoverable=False,
                )
            if not _flash_uf2_picotool(uf2_path, picotool_path=picotool,
                                       reboot=True):
                raise UpdateError(
                    f"picotool failed to flash {uf2_path.name}. The "
                    f"board may need another BOOTSEL + power-cycle.",
                    stage=UpdateStage.COPYING_UF2,
                    recoverable=False,
                )

        # ---- Wait for serial port ----
        _report_progress(progress_callback, UpdateStage.WAITING_REBOOT,
                         "Waiting for board to reboot on USB CDC...",
                         0.60)
        time.sleep(POST_REBOOT_SETTLE_TIME)

        new_port = _wait_for_serial_port(
            config.vid, config.pid,
            timeout=config.serial_reappear_timeout,
        )
        if new_port is None:
            bootsel_again = _detect_bootsel_drive()
            if bootsel_again is not None:
                raise UpdateError(
                    f"Board returned to BOOTSEL instead of booting. "
                    f"The UF2 may be invalid.",
                    stage=UpdateStage.WAITING_REBOOT,
                    recoverable=True,
                )
            raise UpdateError(
                f"Serial port did not reappear within "
                f"{config.serial_reappear_timeout}s. Power-cycle and "
                f"retry.",
                stage=UpdateStage.WAITING_REBOOT,
                recoverable=False,
            )
        time.sleep(POST_REBOOT_SETTLE_TIME)

        # ---- Verify version ----
        _report_progress(progress_callback, UpdateStage.VERIFYING_VERSION,
                         "Verifying new firmware version...", 0.85)
        board = _create_board(config, port=new_port)
        new_version = board.firmware_version or board.firmware_date
        result.new_version = new_version
        board.disconnect()
        logger.info(f"Direct-flash complete: v{new_version}")

        if new_version is None:
            # Bare MP runtime with no main.py prints no INFO — expected
            # after flashing mocon.uf2 to a freshly-nuked board.
            result.warnings.append(
                "No firmware version reported — expected after flashing "
                "a bare runtime with no main.py. Deploy main.py via "
                "deploy_firmware_file() next.")
        elif target_version and new_version != target_version:
            result.warnings.append(
                f"Version mismatch: expected {target_version}, "
                f"got {new_version}"
            )

        result.success = True
        _report_progress(progress_callback, UpdateStage.COMPLETE,
                         "Direct UF2 flash complete", 1.0)
        return result

    except UpdateError as e:
        result.error_message = str(e)
        result.error_stage = e.stage
        logger.error(f"flash_uf2_direct failed at {e.stage.value}: {e}")
        _report_progress(progress_callback, UpdateStage.FAILED,
                         str(e), 0.0)
        return result
    except Exception as e:
        result.error_message = f"Unexpected error: {e}"
        result.error_stage = UpdateStage.FAILED
        logger.error(
            f"flash_uf2_direct unexpected error: {e}", exc_info=True)
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


def restore_configs_from_backup(
    board_type,
    backup_dir,
    progress_callback=None,
    file_filter=None,
):
    """Restore config files from a local backup directory to a board.

    Symmetric counterpart of _backup_configs(). Takes a directory (e.g.
    one produced by an earlier deploy_firmware_file backup stage, or a
    hand-picked per-unit config set from bench archives) and pushes each
    config file that matches the board's BOARD_CONFIGS[board_type].
    config_files list onto the board via raw REPL (SHA256-verified by
    SerialBoard.repl_write_file).

    Use cases:
      - Re-provision a board after factory_reset_motor_board() nuked
        the filesystem.
      - Clone per-unit configs from a dev-bench board to a new production
        unit during bring-up.
      - Recover INI/motorconfig files from a backup after an operator
        edit gone wrong.

    Args:
        board_type: BoardType.LED or BoardType.MOTOR
        backup_dir: Directory containing backed-up config files. Only
            files whose names appear in BOARD_CONFIGS[board_type].
            config_files are considered (file_filter narrows further).
        progress_callback: Optional (stage, message, fraction) callback.
        file_filter: Optional list/set of filenames. If provided, only
            files in this list are restored (subject to config_files
            membership).

    Returns:
        UpdateResult. success=True iff every eligible file was pushed
        and verified.
    """
    config = BOARD_CONFIGS[board_type]
    backup_dir = Path(backup_dir)
    result = UpdateResult(success=False, board_type=board_type)
    result.config_backup_path = backup_dir

    try:
        _report_progress(progress_callback, UpdateStage.PREFLIGHT,
                         f"Loading backup from {backup_dir}...", 0.0)

        if not backup_dir.is_dir():
            raise UpdateError(
                f"Backup directory not found: {backup_dir}",
                stage=UpdateStage.PREFLIGHT,
            )

        # Build {filename: bytes} dict from the backup directory.
        # Only include files the BOARD_CONFIGS policy says belong on
        # this board type — avoids pushing stray files (README, main.py,
        # .bak files, etc.) that the backup directory may also hold.
        config_files = set(config.config_files)
        if file_filter is not None:
            config_files &= set(file_filter)

        config_data = {}
        for name in sorted(config_files):
            p = backup_dir / name
            if not p.is_file():
                logger.info(f"Skipping {name}: not present in backup")
                continue
            config_data[name] = p.read_bytes()

        if not config_data:
            raise UpdateError(
                f"No matching config files found in {backup_dir}. "
                f"Expected one or more of: {sorted(config.config_files)}",
                stage=UpdateStage.PREFLIGHT,
            )

        logger.info(
            f"Will restore {len(config_data)} file(s) to "
            f"{config.label} board: {sorted(config_data.keys())}"
        )

        _report_progress(progress_callback, UpdateStage.CHECKING_VERSION,
                         f"Connecting to {config.label} board...", 0.10)
        board = _create_board(config)
        try:
            current_version = board.firmware_version or board.firmware_date
            result.old_version = current_version
            result.new_version = current_version  # restore doesn't change FW

            # _restore_configs handles raw REPL + SHA256-verified writes.
            _restore_configs(board, config, config_data, progress_callback)
        finally:
            try:
                board.disconnect()
            except Exception:
                pass

        result.success = True
        _report_progress(progress_callback, UpdateStage.COMPLETE,
                         f"Restored {len(config_data)} file(s)", 1.0)
        logger.info(
            f"Config restore successful: {len(config_data)} file(s) on "
            f"{config.label} board"
        )
        return result

    except UpdateError as e:
        result.error_message = str(e)
        result.error_stage = e.stage
        logger.error(f"Config restore failed at {e.stage.value}: {e}")
        _report_progress(progress_callback, UpdateStage.FAILED, str(e), 0.0)
        return result

    except Exception as e:
        result.error_message = f"Unexpected error: {e}"
        result.error_stage = UpdateStage.FAILED
        logger.error(f"Config restore unexpected error: {e}", exc_info=True)
        _report_progress(progress_callback, UpdateStage.FAILED,
                         f"Unexpected error: {e}", 0.0)
        return result


def deploy_firmware_bundle_fw40(
    board_type,
    main_module_path,
    framing_path,
    progress_callback=None,
    backup_dir=None,
    skip_config_backup=False,
    skip_post_test=False,
    soft_reset=False,
):
    """Deploy the FW4.0 3-file bundle via raw REPL as precompiled .mpy.

    The FW4.0 stub + .mpy pattern (release gate §2.4, FIRMWARE_PLAN.md
    lines 340-375):

        main.py           -- stub, content: `import fw40_led` (or fw40_motor)
        fw40_led.mpy      -- precompiled LED firmware     (or fw40_motor.mpy)
        fw40_framing.mpy  -- precompiled shared framing

    MicroPython auto-selects `.mpy` over `.py` when both exist, and the
    stub `main.py` triggers the import. This retires:
      1. The MP 1.19 compile-OOM on LED's ~63KB main.py as plain .py
         (confirmed 2026-04-22 bench: MemoryError allocating 63296
         bytes — the reason this helper exists).
      2. The source-exposure risk of shipping readable firmware to
         sealed field units.
      3. The per-import compile latency on boot.

    On-device write order (same atomicity contract as deploy_firmware_file
    with extra_files): fw40_framing.mpy → fw40_<module>.mpy → main.py
    stub last. A partial failure leaves the old main.py + new
    framing/module files, which boot a crash-visible state rather than
    silently running the new main.py against a missing dependency.

    Args:
        board_type: BoardType.LED or BoardType.MOTOR.
        main_module_path: Path to the firmware source .py (e.g.
            'LED Controller/main.py'). The on-device import name is
            derived from the MODULE stem, not the filename — for a
            source named 'main.py' that must become 'fw40_led.mpy' on
            device, use main_module_remote_stem to override. Default:
            derive from board_type (LED → 'fw40_led', MOTOR → 'fw40_motor').
        framing_path: Path to fw40_framing.py.
        progress_callback, backup_dir, skip_config_backup, skip_post_test:
            same as deploy_firmware_file.

    Returns:
        UpdateResult with success/failure details.
    """
    module_stem = {
        BoardType.LED: 'fw40_led',
        BoardType.MOTOR: 'fw40_motor',
    }[board_type]
    stub_content = f'import {module_stem}\n'.encode('utf-8')

    import tempfile
    stub_dir = Path(tempfile.mkdtemp(prefix='fw40_stub_'))
    stub_main = stub_dir / 'main.py'
    stub_main.write_bytes(stub_content)

    # The stub is the firmware_path (written as main.py). The real
    # firmware + framing are extra_files (.py inputs, compiled to .mpy
    # by deploy_firmware_file when compile_mpy=True). This reuses
    # deploy_firmware_file's preflight / backup / atomic-write
    # machinery without any copy-paste.
    return deploy_firmware_file(
        board_type=board_type,
        firmware_path=stub_main,
        progress_callback=progress_callback,
        backup_dir=backup_dir,
        skip_config_backup=skip_config_backup,
        skip_post_test=skip_post_test,
        firmware_remote_name='main.py',  # stub IS main.py
        compile_mpy=True,  # compile the extra_files below
        soft_reset=soft_reset,
        extra_files=[
            (Path(framing_path), 'fw40_framing.mpy'),
            (Path(main_module_path), f'{module_stem}.mpy'),
        ],
    )


def factory_reset_motor_board(
    nuke_uf2_path,
    runtime_uf2_path,
    main_py_path,
    progress_callback=None,
    skip_post_test=False,
):
    """Full factory-reset recovery for a motor board whose firmware has
    left it in an unrecoverable state (e.g. main.py that blocks raw REPL
    entry via machine.disable_irq in the stdout path).

    Sequence:
      1. nuke_board(MOTOR, nuke_uf2)     — erase all flash incl. filesystem
      2. copy runtime_uf2 to BOOTSEL     — flash a clean MicroPython runtime
      3. wait for serial port to appear  — runtime boots with empty fs
      4. deploy_firmware_file(main.py)   — push known-good main.py via raw REPL

    Step 3+4 work because nuke clears the filesystem; no broken main.py
    blocks raw REPL after the runtime reboots.

    Motor-only — LED boards have no direct USB; factory reset there
    requires physical BOOTSEL access.

    Args:
        nuke_uf2_path: Path to flash_nuke_rp2040.uf2 (or rp2350 variant).
        runtime_uf2_path: Path to a clean MicroPython UF2 for the motor.
        main_py_path: Path to the main.py to restore after reflash.
        progress_callback: Optional (stage, message, fraction) callback.
        skip_post_test: Skip the final deploy_firmware_file post-update
            check.

    Returns:
        UpdateResult. success=True only if all three phases completed.
    """
    board_type = BoardType.MOTOR
    config = BOARD_CONFIGS[board_type]
    result = UpdateResult(success=False, board_type=board_type)

    nuke_uf2_path = Path(nuke_uf2_path)
    runtime_uf2_path = Path(runtime_uf2_path)
    main_py_path = Path(main_py_path)

    try:
        for p, label in ((nuke_uf2_path, 'nuke UF2'),
                         (runtime_uf2_path, 'runtime UF2'),
                         (main_py_path, 'main.py')):
            if not p.is_file():
                raise UpdateError(
                    f"{label} not found: {p}",
                    stage=UpdateStage.PREFLIGHT,
                )

        # ---- Phase 1: Nuke flash ----
        _report_progress(progress_callback, UpdateStage.PREFLIGHT,
                         "Phase 1/3: nuking flash...", 0.0)
        nuke_result = nuke_board(
            board_type, nuke_uf2_path,
            progress_callback=progress_callback,
        )
        if not nuke_result.success:
            raise UpdateError(
                f"Phase 1 (nuke) failed: {nuke_result.error_message}",
                stage=nuke_result.error_stage or UpdateStage.FAILED,
            )
        result.old_version = 'nuked'

        # After nuke the board is in BOOTSEL with empty flash. Copy the
        # runtime UF2 directly without invoking update_firmware (which
        # expects a serial-connected board + pre-flight BOOTSEL check
        # that would now fail because we're already in BOOTSEL).

        # ---- Phase 2: Flash runtime UF2 ----
        _report_progress(progress_callback, UpdateStage.COPYING_UF2,
                         "Phase 2/3: flashing runtime UF2...", 0.35)
        bootsel_drive = _wait_for_bootsel_drive(
            timeout=config.bootsel_timeout)
        if bootsel_drive is None:
            raise UpdateError(
                "BOOTSEL drive not found after nuke — cannot flash "
                "runtime UF2. Manual recovery required (hold BOOTSEL + "
                "replug USB).",
                stage=UpdateStage.WAITING_BOOTSEL,
                recoverable=False,
            )
        dest = bootsel_drive / runtime_uf2_path.name
        shutil.copy2(runtime_uf2_path, dest)
        logger.info(f"Runtime UF2 copied to {bootsel_drive}")

        # Wait for drive to disappear (UF2 accepted), then for the
        # board's serial port to reappear (runtime boots + enumerates).
        time.sleep(POST_UF2_SETTLE_TIME)
        _wait_for_drive_disappear(bootsel_drive)

        _report_progress(progress_callback, UpdateStage.WAITING_REBOOT,
                         "Phase 2/3: waiting for runtime to boot...", 0.55)
        time.sleep(POST_REBOOT_SETTLE_TIME)
        new_port = _wait_for_serial_port(
            config.vid, config.pid,
            timeout=config.serial_reappear_timeout,
        )
        if new_port is None:
            raise UpdateError(
                f"Serial port did not reappear within "
                f"{config.serial_reappear_timeout}s after runtime flash. "
                f"The runtime UF2 may be invalid or the board may need "
                f"power-cycling.",
                stage=UpdateStage.WAITING_REBOOT,
                recoverable=False,
            )
        time.sleep(POST_REBOOT_SETTLE_TIME)

        # ---- Phase 3: Push main.py via raw REPL ----
        _report_progress(progress_callback, UpdateStage.RESTORING_CONFIG,
                         "Phase 3/3: pushing main.py via raw REPL...", 0.75)

        # Filesystem is empty after nuke — no main.py is running, so
        # raw REPL entry works trivially (no command-loop to bypass).
        # deploy_firmware_file handles its own connection + backup +
        # write + reboot + post-test.
        repl_result = deploy_firmware_file(
            board_type, main_py_path,
            progress_callback=progress_callback,
            skip_config_backup=True,   # nuked filesystem — nothing to back up
            skip_post_test=skip_post_test,
        )
        if not repl_result.success:
            raise UpdateError(
                f"Phase 3 (deploy main.py) failed: "
                f"{repl_result.error_message}",
                stage=repl_result.error_stage or UpdateStage.FAILED,
            )

        result.success = True
        result.new_version = repl_result.new_version
        result.warnings.extend(nuke_result.warnings or [])
        result.warnings.extend(repl_result.warnings or [])
        _report_progress(progress_callback, UpdateStage.COMPLETE,
                         "Factory reset complete", 1.0)
        logger.info(
            f"Motor factory reset successful: runtime={runtime_uf2_path.name}, "
            f"main.py={main_py_path.name}, new_version={result.new_version}"
        )
        return result

    except UpdateError as e:
        result.error_message = str(e)
        result.error_stage = e.stage
        logger.error(f"Factory reset failed at {e.stage.value}: {e}")
        _report_progress(progress_callback, UpdateStage.FAILED, str(e), 0.0)
        return result

    except Exception as e:
        result.error_message = f"Unexpected error: {e}"
        result.error_stage = UpdateStage.FAILED
        logger.error(f"Factory reset unexpected error: {e}", exc_info=True)
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
    extra_files=None,
    firmware_remote_name='main.py',
    compile_mpy=False,
    soft_reset=False,
):
    """Deploy main.py (+ optional companion files) to a board via raw REPL.

    This is the primary update method for LED boards (UART-only, no USB
    to the RP2040) and an alternative method for motor boards when only
    updating the firmware Python file without changing the MicroPython
    runtime.

    The sequence:
      1. Connect to the board via serial
      2. Back up config files via raw REPL
      3. Write every `extra_files` entry first, then main.py last
         (each SHA256 verified, atomic). main.py is last so a partial
         failure leaves an old main.py + new framing — the new framing
         may crash a pre-FW4.0 main.py on boot, which is visible and
         recoverable; the reverse (new main.py needing framing that
         isn't there) would brick. Each file's write is atomic per-file
         (`.tmp` + SHA-256 + rename + `.bak`, with 3× transient-error
         retry). Cross-file atomicity is NOT guaranteed; if
         `fw40_framing.mpy` commits and `main.mpy` then exhausts its
         retries, the board boots with new framing + old main. This has
         not been observed on bench post-mpremote (raw-paste window-
         token flow control retires the exhaustion class); revisit with
         a two-phase-commit design if bench surfaces it.
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
        extra_files: Optional list of (local_path, remote_filename)
            companion files to deploy alongside main.py. FW4.0 requires
            (fw40_framing.py, 'fw40_framing.py') because both main.py
            files `import fw40_framing`. Each file is SHA256-verified by
            repl_write_file; any failure aborts the whole deploy before
            main.py is written, so we never leave the board in the
            'new main.py without its companion' state.
        firmware_remote_name: Filename to write the firmware under on
            the device. Defaults to 'main.py'. Set to e.g. 'fw40_led.mpy'
            for the stub-import + .mpy pattern (release gate §2.4).
        compile_mpy: If True, compile firmware_path and any .py entries
            in extra_files via mpy-cross before deploying. Compiled .mpy
            bytes are written under the caller-supplied remote names
            verbatim — the caller is responsible for making sure the
            remote name ends in .mpy (or makes sense as a .mpy import
            target). Required for LED-side FW4.0 on MP 1.19: main.py
            (~63KB) hits the on-device compile-OOM when sent as .py.
        soft_reset: If True, enter raw REPL via soft reset (Ctrl-D). Use
            this when the board has no WDT and no user code to preserve
            across the raw-REPL entry — e.g. a freshly-flashed bare MP
            runtime, or a board whose firmware has already been
            verified to not rely on a running Timer. Default is False
            for safety with v3.0.x LED firmware (pre-WDT-removal) where
            soft reset kills the WDT-feed Timer; leave False for that
            path. Bench-confirmed 2026-04-22: MP 1.27.0 bare runtime
            with soft_reset=False produces "Expected OK or \\x04 marker,
            got b'i'" write failures because Ctrl-C leaks into the
            interactive REPL mid-protocol; soft_reset=True avoids the
            race cleanly.

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

        # mpy-cross preflight — must resolve before we touch the board.
        mpy_cross_path = None
        if compile_mpy:
            mpy_cross_path = _find_mpy_cross()
            if mpy_cross_path is None:
                raise UpdateError(
                    "compile_mpy=True but mpy-cross was not found on "
                    "PATH. Install the micropython PyPI package or set "
                    "PATH to include mpy-cross.",
                    stage=UpdateStage.PREFLIGHT,
                )
            logger.info(f"mpy-cross found at {mpy_cross_path}")

        fw_data = firmware_path.read_bytes()
        if len(fw_data) < 10:
            # Stubs may be tiny (e.g. "import fw40_led\n" is ~16 bytes).
            # Hard floor at 10B guards against empty-file mishap.
            raise UpdateError(
                f"Firmware file too small ({len(fw_data)} bytes)",
                stage=UpdateStage.PREFLIGHT,
            )
        logger.info(f"Firmware file: {firmware_path.name} ({len(fw_data)} bytes)")

        # Compile firmware_path only when compile_mpy=True AND the REMOTE
        # name ends in .mpy. The remote name carries caller intent:
        #   firmware_remote_name='main.py'         -> source deploy, no compile
        #   firmware_remote_name='fw40_led.mpy'    -> compile .py source
        # This lets deploy_firmware_bundle_fw40 send a stub main.py verbatim
        # while compiling the module source to .mpy.
        if (compile_mpy
                and firmware_path.suffix == '.py'
                and firmware_remote_name.endswith('.mpy')):
            import tempfile
            tmp_mpy = Path(tempfile.mkdtemp(prefix='fw40_mpy_')) / (
                firmware_path.stem + '.mpy')
            if not _mpy_cross_compile(firmware_path, tmp_mpy,
                                      mpy_cross_path=mpy_cross_path):
                raise UpdateError(
                    f"mpy-cross failed on firmware_path {firmware_path}",
                    stage=UpdateStage.PREFLIGHT,
                )
            fw_data = tmp_mpy.read_bytes()
            logger.info(
                f"Compiled firmware: {firmware_path.name} "
                f"({firmware_path.stat().st_size}B) -> "
                f"{tmp_mpy.name} ({len(fw_data)}B)"
            )

        # Validate and read extra_files up-front so any missing file fails
        # preflight, not mid-deploy after we've already written something.
        extra_files_data = []  # list of (remote_name, bytes)
        if extra_files:
            for local_path, remote_name in extra_files:
                local_path = Path(local_path)
                if not local_path.is_file():
                    raise UpdateError(
                        f"extra_files entry not found: {local_path}",
                        stage=UpdateStage.PREFLIGHT,
                    )

                # Compile .py entries whose remote name ends in .mpy.
                # Same caller-intent rule as firmware_path above.
                if (compile_mpy
                        and local_path.suffix == '.py'
                        and str(remote_name).endswith('.mpy')):
                    import tempfile
                    tmp_mpy = Path(tempfile.mkdtemp(prefix='fw40_mpy_')) / (
                        local_path.stem + '.mpy')
                    if not _mpy_cross_compile(
                            local_path, tmp_mpy,
                            mpy_cross_path=mpy_cross_path):
                        raise UpdateError(
                            f"mpy-cross failed on extra_files entry "
                            f"{local_path}",
                            stage=UpdateStage.PREFLIGHT,
                        )
                    data = tmp_mpy.read_bytes()
                else:
                    data = local_path.read_bytes()

                if len(data) < 1:
                    raise UpdateError(
                        f"extra_files entry empty: {local_path}",
                        stage=UpdateStage.PREFLIGHT,
                    )
                extra_files_data.append((str(remote_name), data))
                logger.info(
                    f"Companion file: {local_path.name} -> "
                    f"{remote_name} ({len(data)} bytes)"
                )

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

        # Default soft_reset=False protects the v3.0.x LED WDT-feed Timer
        # during a 57KB @ 115200 UART write. Callers with a WDT-free
        # board (bare MP runtime, v3.0.4+ with WDT removed, or a post-
        # flash blank filesystem) pass soft_reset=True to avoid the
        # Ctrl-C-into-interactive-REPL race that breaks raw-paste writes
        # on MP 1.27.0 bare runtime (bench-confirmed 2026-04-22).
        if not board.enter_raw_repl(soft_reset=soft_reset):
            raise UpdateError(
                f"Failed to enter raw REPL for firmware deploy",
                stage=UpdateStage.RESTORING_CONFIG,
            )

        with _wrap_mpremote_errors(
            UpdateStage.RESTORING_CONFIG, context="firmware deploy"
        ):
            # Companion files first — see docstring note on ordering: if a
            # companion write fails, the old main.py still boots fine because
            # main.py hasn't been replaced yet.
            for remote_name, data in extra_files_data:
                if not board.repl_write_file(remote_name, data):
                    raise UpdateError(
                        f"Failed to write companion {remote_name} "
                        f"({len(data)} bytes)",
                        stage=UpdateStage.RESTORING_CONFIG,
                    )
                logger.info(
                    f"Deployed {remote_name} ({len(data)} bytes, "
                    f"SHA256 verified)"
                )

            if not board.repl_write_file(firmware_remote_name, fw_data):
                raise UpdateError(
                    f"Failed to write {firmware_remote_name} "
                    f"({len(fw_data)} bytes)",
                    stage=UpdateStage.RESTORING_CONFIG,
                )
            logger.info(
                f"Deployed {firmware_remote_name} ({len(fw_data)} bytes, "
                f"SHA256 verified)"
            )

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


# ---------------------------------------------------------------------------
# FW4.0 Field Upgrade Tool — §13.X of FIRMWARE_PLAN.md
# ---------------------------------------------------------------------------
#
# Orchestrates probe → backup → bundle-deploy → verify with telemetry per
# run. Mandate: "IT CANNOT FAIL" (Eric 2026-04-23). Every failure is a
# specific exit code + error_code; there is no silent half-upgrade (I1).
# Primary caller is LVP (Lumascope.upgrade_board_fw40); CLI is secondary.

UPGRADE_EXIT_OK = 0
UPGRADE_EXIT_P0_SOURCE = 10
UPGRADE_EXIT_P1_UNRESPONSIVE = 20
UPGRADE_EXIT_P2_BACKUP = 30
UPGRADE_EXIT_P2_OVERWRITABLE = 35
UPGRADE_EXIT_P4_BUNDLE = 40
UPGRADE_EXIT_P5_VERIFY = 50


def _load_firmware_manifest(source_tree: Path,
                            board_type: BoardType) -> Tuple[
                                Optional[dict], Optional[str]]:
    """Load + validate firmware_manifest.json for a board.

    Returns (manifest_entry_dict, error_message). On error the dict is
    None; error_message is a single human-readable line for the UI.
    """
    manifest_path = source_tree / 'firmware_manifest.json'
    if not manifest_path.is_file():
        return None, f"firmware_manifest.json not found at {manifest_path}"
    try:
        doc = json.loads(manifest_path.read_text())
    except Exception as e:
        return None, f"firmware_manifest.json is not valid JSON: {e}"

    board_key = board_type.value  # 'led' or 'motor'
    entry = doc.get(board_key)
    if not isinstance(entry, dict):
        return None, (
            f"firmware_manifest.json missing '{board_key}' entry")
    for required in ('fw_version', 'main', 'features'):
        if required not in entry:
            return None, (
                f"firmware_manifest.json '{board_key}' entry missing "
                f"'{required}'")

    framing = doc.get('framing')
    if not isinstance(framing, str):
        return None, "firmware_manifest.json missing 'framing' path"

    # Resolve paths relative to source_tree
    main_path = source_tree / entry['main']
    framing_path = source_tree / framing
    if not main_path.is_file():
        return None, f"manifest main path not on disk: {main_path}"
    if not framing_path.is_file():
        return None, f"manifest framing path not on disk: {framing_path}"

    resolved = {
        'fw_version': str(entry['fw_version']),
        'features': list(entry.get('features', [])),
        'main_path': main_path,
        'framing_path': framing_path,
    }
    return resolved, None


def _probe_board_state(board) -> str:
    """Classify the connected board from SerialBoard's cached detection.

    SerialBoard.connect() / _detect_firmware_version() populates
    firmware_version, firmware_date, firmware_responding, protocol_version,
    and features. We read those; no re-send of INFO here. For a board that
    didn't parse INFO we fall back to a raw-REPL liveness probe.

    Returns one of:
        'fw40_current'       — V4 firmware, required feature set present
        'fw40_partial'       — V4 firmware, missing expected features
        'legacy_responsive'  — pre-4.0 firmware, INFO parseable
        'legacy_unknown'     — responds but INFO not parseable, OR raw REPL alive
        'unresponsive'       — no INFO, no raw REPL — treat as bricked
    """
    # Defensive reads: SerialBoard sets these as concrete types
    # (str | None for firmware_version, bool for firmware_responding,
    # ProtocolVersion enum for protocol_version, list[str] for features).
    # MagicMock returns auto-Mocks for unset attrs; we isinstance-check
    # so a test fixture that forgets to set one doesn't get misclassified.
    proto = getattr(board, 'protocol_version', None)
    proto_value = getattr(proto, 'value', None) if proto is not None else None
    proto_str = proto_value.lower() if isinstance(proto_value, str) else ''

    fw_raw = getattr(board, 'firmware_version', None)
    fw = fw_raw if isinstance(fw_raw, str) and fw_raw else None

    features_raw = getattr(board, 'features', None)
    features = (
        [f for f in features_raw if isinstance(f, str)]
        if isinstance(features_raw, (list, tuple))
        else [])

    responding = getattr(board, 'firmware_responding', False) is True

    if proto_str == 'v4':
        if 'fw40_framing' in features:
            return 'fw40_current'
        return 'fw40_partial'

    if responding and fw:
        return 'legacy_responsive'

    if responding:
        return 'legacy_unknown'

    # Last resort: does raw REPL answer? If yes the board is running
    # SOME MicroPython — treat as legacy_unknown and let the permissive
    # deploy path run. Per §13.X.9 Q2 (Eric 2026-04-23).
    try:
        if board.enter_raw_repl(soft_reset=False):
            try:
                stdout, _ = board.repl_exec("print('fw40-probe')")
                if stdout and b'fw40-probe' in bytes(stdout):
                    return 'legacy_unknown'
            finally:
                try:
                    board.exit_raw_repl()
                except Exception:
                    pass
    except Exception:
        pass

    return 'unresponsive'


def _soft_reset_for_classification(classification: str) -> bool:
    """Per §13.X.4 P4 — probe classification drives soft_reset choice.

    Field-upgrade classifications map to soft_reset=False across the
    board: pre-3.0 may have a WDT-feed Timer (killed by Ctrl-D), FW4.0
    has a running main loop we don't need to tear down. ``bare_runtime``
    (factory bring-up, post-MP-flash blank filesystem) is a separate
    caller path and is not produced by this probe.
    """
    return False


def _parse_overwritable_flags(
        motorconfig_bytes: bytes) -> Optional[Dict[str, int]]:
    """Extract the Overwritable sub-object from a motorconfig.json payload.

    Returns None if the field is absent, unparseable, or malformed.
    """
    try:
        doc = json.loads(motorconfig_bytes)
    except Exception:
        return None
    ow = doc.get('Overwritable')
    if not isinstance(ow, dict):
        return None
    flags = {}
    for k in ('Main', 'IniFiles', 'uf2'):
        v = ow.get(k)
        if isinstance(v, bool):
            flags[k] = int(v)
        elif isinstance(v, (int, float)):
            flags[k] = int(v)
    return flags if flags else None


def _get_upgrade_log_dir() -> Path:
    """Return the directory where firmware_upgrade_*.jsonl is written.

    Resolved lazily from ``lvp_logger.log_dir`` so monkeypatch works in
    tests. Creates the directory if missing (matching lvp_logger's own
    behavior for the main log dir).
    """
    import lvp_logger
    p = Path(str(lvp_logger.log_dir))
    p.mkdir(parents=True, exist_ok=True)
    return p


class _UpgradeTelemetry:
    """JSON Lines writer for an upgrade run.

    One object per step written incrementally (flush + fsync per line) so
    a host crash preserves progress. Filename starts with a provisional
    ``_inprogress`` suffix and is renamed to include the final exit code
    when close() is called — matches §13.X.1 I6 path shape.
    """

    def __init__(self, board_type: BoardType):
        ts = time.strftime('%Y%m%dT%H%M%S', time.localtime())
        self.board = board_type.value
        log_dir = _get_upgrade_log_dir()
        self._in_progress_name = (
            f'firmware_upgrade_{ts}_{self.board}_inprogress.jsonl')
        self.path = log_dir / self._in_progress_name
        self._ts_stem = ts
        self._f = open(self.path, 'w', encoding='utf-8')

    def write(self, step: str, outcome: str, **fields) -> None:
        obj = {
            'timestamp': time.strftime(
                '%Y-%m-%dT%H:%M:%S', time.localtime()),
            'board': self.board,
            'step': step,
            'outcome': outcome,
        }
        obj.update(fields)
        self._f.write(json.dumps(obj) + '\n')
        self._f.flush()
        try:
            import os
            os.fsync(self._f.fileno())
        except OSError:
            # fsync is best-effort on some filesystems; never block the
            # upgrade on the telemetry layer.
            pass

    def close(self, exit_code: int) -> Path:
        try:
            self._f.close()
        except Exception:
            pass
        final_name = (
            f'firmware_upgrade_{self._ts_stem}_{self.board}'
            f'_{exit_code}.jsonl')
        final_path = self.path.parent / final_name
        try:
            self.path.rename(final_path)
            self.path = final_path
        except OSError as e:
            logger.warning(
                f'Telemetry rename {self.path} -> {final_path} failed: {e}')
        return self.path


def upgrade_board_fw40_from_source(
    board_type: BoardType,
    source_tree: Path,
    dry_run: bool = False,
    respect_overwritable: bool = True,
    port: Optional[str] = None,
    timeout: float = 2.0,
    progress_callback: Optional[ProgressCallback] = None,
) -> UpgradeResult:
    """Field-upgrade a pre-3.0 or FW4.0-partial board to FW4.0-current.

    Per FIRMWARE_PLAN.md §13.X. Orchestrates:

      P0  host-side source + manifest + mpy-cross validation
      P1  board probe (no writes — classify state)
      P2  mandatory verified config backup
      P3  existing-firmware snapshot
      P4  bundle write via ``deploy_firmware_bundle_fw40``
      P5  reboot + JSON INFO verify (version + features)
      P6  finalize + telemetry close

    Every gate failure produces a specific ``UpgradeResult.exit_code``
    (see UPGRADE_EXIT_* constants). No auto-rollback on P5 failure per
    §13.X.5. Telemetry JSONL is written on both success and abort.

    Args:
        board_type: BoardType.MOTOR or BoardType.LED.
        source_tree: Path to the Firmware-FW4.0 repo root (contains
            firmware_manifest.json).
        dry_run: If True, run P0 only and exit success without opening
            a serial transport.
        respect_overwritable: If True (default), Overwritable.Main=0
            aborts firmware writes with exit code 35. If False, writes
            proceed and a warning is recorded.
        port, timeout: Optional overrides passed to _create_board.
        progress_callback: Optional (stage, message, fraction) callback.

    Returns:
        UpgradeResult — always populated, even on abort.
    """
    source_tree = Path(source_tree)
    result = UpgradeResult(
        success=False, board_type=board_type,
        exit_code=UPGRADE_EXIT_P0_SOURCE,
    )
    telemetry: Optional[_UpgradeTelemetry] = None

    try:
        # ---- P0 — host source validation ----
        _report_progress(progress_callback, UpdateStage.PREFLIGHT,
                         'Validating source tree...', 0.0)

        manifest, err = _load_firmware_manifest(source_tree, board_type)
        if manifest is None:
            result.exit_code = UPGRADE_EXIT_P0_SOURCE
            result.error_code = 'P0_MANIFEST_INVALID'
            result.error_message = err
            result.error_stage = UpdateStage.PREFLIGHT
            return result

        if _find_mpy_cross() is None:
            result.exit_code = UPGRADE_EXIT_P0_SOURCE
            result.error_code = 'P0_MPY_CROSS_MISSING'
            result.error_message = (
                'mpy-cross not found on PATH. Install the micropython '
                'PyPI package or set PATH to include mpy-cross.')
            result.error_stage = UpdateStage.PREFLIGHT
            return result

        # Telemetry opens AFTER we know preflight passed — failing P0 is
        # a host-side misconfig, not useful field-log content.
        telemetry = _UpgradeTelemetry(board_type)
        result.telemetry_log_path = telemetry.path
        telemetry.write(
            'P0_source_validated', 'ok',
            fw_version=manifest['fw_version'],
            features=manifest['features'],
            main_size=manifest['main_path'].stat().st_size,
            framing_size=manifest['framing_path'].stat().st_size,
        )

        if dry_run:
            telemetry.write('dry_run', 'ok')
            result.success = True
            result.exit_code = UPGRADE_EXIT_OK
            result.telemetry_log_path = telemetry.close(UPGRADE_EXIT_OK)
            _report_progress(progress_callback, UpdateStage.COMPLETE,
                             'Dry run complete (P0 only).', 1.0)
            return result

        # ---- P1 — probe (transport open, read-only) ----
        _report_progress(progress_callback, UpdateStage.CHECKING_VERSION,
                         'Probing board...', 0.10)
        config = BOARD_CONFIGS[board_type]
        board = _create_board(config, port=port, timeout=timeout)
        result.old_version = (
            getattr(board, 'firmware_version', None)
            or getattr(board, 'firmware_date', None))

        classification = _probe_board_state(board)
        result.probe_classification = classification
        telemetry.write(
            'P1_probe', 'ok', classification=classification,
            old_version=result.old_version,
        )

        if classification == 'unresponsive':
            try:
                board.disconnect()
            except Exception:
                pass
            result.exit_code = UPGRADE_EXIT_P1_UNRESPONSIVE
            result.error_code = 'P1_UNRESPONSIVE'
            result.error_message = (
                f"{config.label} board not responding — reseat USB cable "
                f"and retry; if persistent, contact support.")
            result.error_stage = UpdateStage.CHECKING_VERSION
            telemetry.write(
                'P1_probe', 'abort', error_code=result.error_code)
            result.telemetry_log_path = telemetry.close(result.exit_code)
            return result

        # fw40_current is an idempotent-pass shortcut — board already at
        # target. We still run P5 verify so the caller knows the features
        # list is complete; but skip the write if everything matches.
        idempotent_pass = False
        if classification == 'fw40_current':
            current_features = set(
                getattr(board, 'features', None) or [])
            if current_features >= set(manifest['features']):
                idempotent_pass = True
                telemetry.write(
                    'P1_idempotent', 'ok',
                    features=sorted(current_features),
                )

        # ---- P2 — mandatory verified config backup ----
        _report_progress(progress_callback, UpdateStage.BACKING_UP_CONFIG,
                         'Backing up config files...', 0.20)
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = (Path.home() / 'Documents' / 'Etaluma'
                      / 'firmware_backups' / f'upgrade_{ts}_{config.label}')
        result.config_backup_path = backup_dir

        configs: Dict[str, bytes] = {}
        try:
            configs = _backup_with_reread_verify(
                board, config, backup_dir, progress_callback)
        except UpdateError as e:
            result.exit_code = UPGRADE_EXIT_P2_BACKUP
            result.error_code = (
                'P2_BACKUP_MISMATCH'
                if 'mismatch' in str(e).lower()
                else 'P2_BACKUP_FAILED')
            result.error_message = str(e)
            result.error_stage = e.stage
            telemetry.write(
                'P2_backup', 'abort', error_code=result.error_code,
                error_message=str(e),
            )
            try:
                board.disconnect()
            except Exception:
                pass
            result.telemetry_log_path = telemetry.close(result.exit_code)
            return result

        telemetry.write(
            'P2_backup', 'ok',
            files=list(configs.keys()),
            sizes={k: len(v) for k, v in configs.items()},
        )

        # I5 — parse Overwritable + gate firmware write if Main=0
        ow_flags = None
        if 'motorconfig.json' in configs:
            ow_flags = _parse_overwritable_flags(configs['motorconfig.json'])
        result.overwritable_flags = ow_flags
        if ow_flags is not None:
            telemetry.write('P2_overwritable', 'ok', flags=ow_flags)

        main_blocked = bool(
            ow_flags and ow_flags.get('Main') == 0)
        if main_blocked and respect_overwritable:
            result.files_skipped_overwritable.append('main')
            result.exit_code = UPGRADE_EXIT_P2_OVERWRITABLE
            result.error_code = 'P2_OVERWRITABLE_BLOCKED'
            result.error_message = (
                'Overwritable.Main=0 on board motorconfig.json — '
                'firmware write refused. Use --ignore-overwritable to '
                'force.')
            result.error_stage = UpdateStage.BACKING_UP_CONFIG
            telemetry.write(
                'P2_overwritable', 'abort',
                error_code=result.error_code, blocked=['main'],
            )
            try:
                board.disconnect()
            except Exception:
                pass
            result.telemetry_log_path = telemetry.close(result.exit_code)
            return result

        if main_blocked and not respect_overwritable:
            msg = (
                'Overwritable.Main=0 bypassed by respect_overwritable='
                'False — proceeding with firmware write.')
            result.warnings.append(msg)
            telemetry.write('P2_overwritable', 'bypass', warning=msg)

        # P3 — existing-firmware snapshot. Best-effort, not a gate.
        # For field boards the backed-up motorconfig + any main.py that
        # existed on device is already in backup_dir (via repl_read_file
        # during _backup_with_reread_verify). Explicit snapshot of the
        # pre-upgrade main.py would require another raw-REPL round; defer
        # to a future extension if bench shows value.

        # Close the raw-REPL session cleanly before handing off to the
        # bundle helper — it opens its own.
        try:
            board.exit_raw_repl()
        except Exception:
            pass
        try:
            board.disconnect()
        except Exception:
            pass

        # ---- P4 — bundle write (unless idempotent-pass) ----
        if idempotent_pass:
            telemetry.write('P4_bundle', 'skipped_idempotent')
            result.new_version = result.old_version
        else:
            _report_progress(progress_callback, UpdateStage.RESTORING_CONFIG,
                             'Writing FW4.0 bundle...', 0.50)
            soft_reset = _soft_reset_for_classification(classification)
            bundle_result = deploy_firmware_bundle_fw40(
                board_type=board_type,
                main_module_path=manifest['main_path'],
                framing_path=manifest['framing_path'],
                progress_callback=progress_callback,
                backup_dir=backup_dir,
                skip_config_backup=True,  # P2 already did it
                skip_post_test=True,      # P5 handles verify
                soft_reset=soft_reset,
            )
            if not bundle_result.success:
                result.exit_code = UPGRADE_EXIT_P4_BUNDLE
                result.error_code = 'P4_BUNDLE_WRITE_FAILED'
                result.error_message = (
                    bundle_result.error_message
                    or 'Bundle write failed')
                result.error_stage = (
                    bundle_result.error_stage or UpdateStage.FAILED)
                telemetry.write(
                    'P4_bundle', 'abort',
                    error_code=result.error_code,
                    error_message=result.error_message,
                )
                result.telemetry_log_path = telemetry.close(
                    result.exit_code)
                return result

            result.new_version = bundle_result.new_version
            result.files_written = [
                'fw40_framing.mpy',
                f'fw40_{board_type.value}.mpy',
                'main.py',
            ]
            telemetry.write(
                'P4_bundle', 'ok',
                files_written=result.files_written,
                soft_reset=soft_reset,
                old_version=bundle_result.old_version,
                new_version=bundle_result.new_version,
            )

        # ---- P5 — reboot + JSON INFO verify ----
        _report_progress(progress_callback, UpdateStage.VERIFYING_VERSION,
                         'Verifying new firmware...', 0.85)
        verify_board = _create_board(config, port=port, timeout=timeout)
        try:
            verify_board.detect_firmware_version()
            new_fw = getattr(verify_board, 'firmware_version', None)
            new_features = set(
                getattr(verify_board, 'features', None) or [])
            result.new_version = new_fw or result.new_version

            expected_version = manifest['fw_version']
            expected_features = set(manifest['features'])

            if str(new_fw or '') != str(expected_version):
                result.exit_code = UPGRADE_EXIT_P5_VERIFY
                result.error_code = 'P5_VERSION_MISMATCH'
                result.error_message = (
                    f"Post-flash version {new_fw!r} does not match "
                    f"manifest {expected_version!r}. Original config "
                    f"preserved; firmware flash incomplete. No "
                    f"auto-rollback (§13.X.5).")
                result.error_stage = UpdateStage.VERIFYING_VERSION
                telemetry.write(
                    'P5_verify', 'abort',
                    error_code=result.error_code,
                    expected=expected_version, actual=new_fw,
                )
                result.telemetry_log_path = telemetry.close(
                    result.exit_code)
                return result

            missing = expected_features - new_features
            if missing:
                result.exit_code = UPGRADE_EXIT_P5_VERIFY
                result.error_code = 'P5_FEATURES_MISSING'
                result.error_message = (
                    f"Post-flash firmware missing expected features: "
                    f"{sorted(missing)}. Original config preserved. No "
                    f"auto-rollback (§13.X.5).")
                result.error_stage = UpdateStage.VERIFYING_VERSION
                telemetry.write(
                    'P5_verify', 'abort',
                    error_code=result.error_code,
                    expected=sorted(expected_features),
                    actual=sorted(new_features),
                    missing=sorted(missing),
                )
                result.telemetry_log_path = telemetry.close(
                    result.exit_code)
                return result

            telemetry.write(
                'P5_verify', 'ok',
                fw_version=new_fw,
                features=sorted(new_features),
            )
        finally:
            try:
                verify_board.disconnect()
            except Exception:
                pass

        # ---- P6 — finalize ----
        result.success = True
        result.exit_code = UPGRADE_EXIT_OK
        telemetry.write('P6_finalize', 'ok')
        result.telemetry_log_path = telemetry.close(UPGRADE_EXIT_OK)
        _report_progress(progress_callback, UpdateStage.COMPLETE,
                         'Upgrade complete.', 1.0)
        logger.info(
            f"[UPGRADE] {config.label} {result.old_version} -> "
            f"{result.new_version} success (telemetry={result.telemetry_log_path})")
        return result

    except Exception as e:
        result.exit_code = result.exit_code or UPGRADE_EXIT_P0_SOURCE
        result.error_message = f"Unexpected error: {e}"
        result.error_stage = UpdateStage.FAILED
        logger.error(f"Upgrade unexpected error: {e}", exc_info=True)
        if telemetry is not None:
            try:
                telemetry.write(
                    'unexpected_error', 'abort', error=str(e))
                result.telemetry_log_path = telemetry.close(
                    result.exit_code)
            except Exception:
                pass
        _report_progress(progress_callback, UpdateStage.FAILED,
                         f"Unexpected error: {e}", 0.0)
        return result


def _backup_with_reread_verify(
    board,
    board_config: BoardConfig,
    backup_dir: Path,
    progress_callback=None,
) -> Dict[str, bytes]:
    """P2 — read every config file via raw REPL, save to disk, then
    re-read from board and byte-compare against the saved copy. Raises
    UpdateError if any file mismatches or cannot be read.

    Stronger than _backup_configs (which relies on per-read SHA256 only);
    this closes the race where the device filesystem could be mutating
    during the upgrade run.
    """
    if not board.enter_raw_repl():
        raise UpdateError(
            f"Failed to enter raw REPL for P2 backup",
            stage=UpdateStage.BACKING_UP_CONFIG,
        )
    configs: Dict[str, bytes] = {}
    try:
        with _wrap_mpremote_errors(
                UpdateStage.BACKING_UP_CONFIG, context="P2 backup"):
            board_files = board.repl_list_files()
            for filename in board_config.config_files:
                if filename not in board_files:
                    continue
                _report_progress(
                    progress_callback, UpdateStage.BACKING_UP_CONFIG,
                    f"Reading {filename}...", 0.22)
                data = board.repl_read_file(filename, verify=True)
                if data is None:
                    raise UpdateError(
                        f"Failed to read {filename}",
                        stage=UpdateStage.BACKING_UP_CONFIG,
                    )
                configs[filename] = data

            # Re-read + byte-compare — I2 P2 gate
            for filename, expected in configs.items():
                _report_progress(
                    progress_callback, UpdateStage.BACKING_UP_CONFIG,
                    f"Re-verifying {filename}...", 0.26)
                second = board.repl_read_file(filename, verify=True)
                if second is None:
                    raise UpdateError(
                        f"Re-read of {filename} returned no data",
                        stage=UpdateStage.BACKING_UP_CONFIG,
                    )
                if second != expected:
                    raise UpdateError(
                        f"P2 byte-compare mismatch on {filename} "
                        f"(first={len(expected)}B second={len(second)}B)",
                        stage=UpdateStage.BACKING_UP_CONFIG,
                    )
    finally:
        try:
            board.exit_raw_repl()
        except Exception:
            pass

    # Persist to disk
    backup_dir.mkdir(parents=True, exist_ok=True)
    board_dir = backup_dir / board_config.board_type.value
    board_dir.mkdir(exist_ok=True)
    manifest = {}
    for filename, data in configs.items():
        local_path = board_dir / filename
        local_path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        manifest[filename] = {'size': len(data), 'sha256': sha}
    (board_dir / 'backup_manifest.json').write_text(
        json.dumps(manifest, indent=2))
    return configs
