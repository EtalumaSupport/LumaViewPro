# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tech Support Report Generator for LumaViewPro.

Collects comprehensive diagnostic information and bundles it into a ZIP file
on the user's Desktop for emailing to techsupport@etaluma.com.

Two modes:
  1. Integrated: Called from the LumaViewPro GUI "Generate Support Report"
     button. Receives a Lumascope instance (or ScopeSession) from the running
     application.
  2. Standalone: ``python tech_support_report.py`` — connects to hardware
     directly, no LumaViewPro needed. Can also be frozen with PyInstaller
     into a standalone .exe (see build notes at bottom of file).

Config file retrieval:
  - Uses MicroPython raw REPL to read files directly from the RP2040
    filesystem (similar to Thonny). No custom firmware command needed.
    Temporarily interrupts firmware, reads files, then soft-resets.

Usage (standalone):
    python tech_support_report.py
    python tech_support_report.py --bandwidth-test
    python tech_support_report.py --no-firmware

Usage (integrated):
    from modules.tech_support_report import TechSupportReport
    report = TechSupportReport(scope=lumascope_instance)
    report.generate(callback=progress_callback)

Recent protocols list (reusable from GUI):
    from modules.tech_support_report import get_recent_protocols
    protocols = get_recent_protocols(10)
"""

import base64
import datetime
import json
import logging
import os
import pathlib
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile

from modules.path_utils import get_script_root, get_source_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_VERSION = "1.0.0"

# Config files to exclude when reading from board filesystem
FIRMWARE_EXCLUDE_FILES = {'main.py', 'boot.py'}
FIRMWARE_CONFIG_EXTENSIONS = {'.json', '.ini'}

RECENT_PROTOCOL_COUNT = 10

BACKLASH_FOLDER_PATTERNS = ['backlash', 'Backlash', 'BACKLASH']

LOG_DELIMITER = (
    "\n"
    "=" * 72 + "\n"
    "=== TECH SUPPORT REPORT GENERATION STARTED — {timestamp} ===\n"
    "=" * 72 + "\n"
)

# Camera bandwidth test defaults
BANDWIDTH_TEST_FRAMES = 5000
BANDWIDTH_TEST_TIMEOUT_S = 300  # 5 min — generous for slow cameras

# Disk speed test defaults
DISK_SPEED_TEST_MB = 256         # Write this many MB
DISK_SPEED_WARN_MBPS = 100      # Warn below this (video recording will lag)

# Voltage tolerance thresholds
VOLTAGE_NOMINAL = {'5V': 5.0, '3.3V': 3.3, '1.2V': 1.2}
VOLTAGE_WARN_PCT = 5.0          # ±5% = warning
VOLTAGE_FAIL_PCT = 10.0         # ±10% = fail

# LED leakage threshold (mA with all LEDs off)
LED_LEAKAGE_WARN_MA = 0.5

# Serial latency test
SERIAL_LATENCY_ITERATIONS = 100

# TMC5072 register addresses (read addresses = write addr | 0x00, but TMC
# uses bit 7 for R/W: addr & 0x7F to read).  These are per-motor within
# the TMC5072 dual driver; motor 0 and motor 1 offsets differ by 0x10.
# The DRVSTAT and SPI commands in firmware abstract this, but for a raw
# register dump we document the key diagnostic registers here.
TMC5072_DIAG_REGISTERS = {
    'GSTAT':       0x01,   # Global status (reset, driver error, UV)
    'IHOLD_IRUN0': 0x30,   # Motor 0 hold/run current
    'IHOLD_IRUN1': 0x50,   # Motor 1 hold/run current
    'CHOPCONF0':   0x6C,   # Motor 0 chopper config
    'CHOPCONF1':   0x7C,   # Motor 1 chopper config
    'DRV_STATUS0': 0x6F,   # Motor 0 driver status (open load, short, OT)
    'DRV_STATUS1': 0x7F,   # Motor 1 driver status
}


# ---------------------------------------------------------------------------
# Path helpers — mirrors userpaths.py conventions
# ---------------------------------------------------------------------------

def _get_app_root():
    """Return the LumaViewPro application root directory."""
    return get_script_root()


def _get_user_documents():
    """Return the user's Documents directory."""
    return pathlib.Path.home() / 'Documents'


def _get_lvp_data_dir():
    """Return the LVP data/ directory (settings.json, scopes.json, etc.)."""
    data_dir = get_source_root() / 'data'
    return data_dir if data_dir.is_dir() else _get_app_root()


def _get_lvp_logs_dir():
    """Return the LVP logs/ directory, or None."""
    logs_dir = get_source_root() / 'logs'
    return logs_dir if logs_dir.is_dir() else None


def _get_capture_dir():
    """Return the capture output directory (from settings.json or defaults)."""
    data_dir = _get_lvp_data_dir()
    settings_file = data_dir / 'settings.json'
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            capture_path = settings.get('capture_path', '')
            if capture_path and pathlib.Path(capture_path).is_dir():
                return pathlib.Path(capture_path)
        except (json.JSONDecodeError, OSError):
            pass
    # Fallback: common locations
    docs = _get_user_documents()
    for name in ['EtalumaCaptures', 'Etaluma', 'LumaViewPro']:
        candidate = docs / name
        if candidate.is_dir():
            return candidate
    return docs


def _get_protocol_dir():
    """Return the protocol files directory, or None."""
    for candidate in [
        _get_app_root() / 'protocols',
        _get_app_root() / 'Protocols',
        _get_app_root() / 'data' / 'protocols',
        _get_capture_dir() / 'protocols',
    ]:
        if candidate.is_dir():
            return candidate
    return None


def _get_desktop():
    """Return the Desktop path (fallback: home directory)."""
    desktop = pathlib.Path.home() / 'Desktop'
    return desktop if desktop.is_dir() else pathlib.Path.home()


# ---------------------------------------------------------------------------
# Recent Protocols — reusable from GUI "Recent Protocols" menu
# ---------------------------------------------------------------------------

def get_recent_protocols(n=RECENT_PROTOCOL_COUNT):
    """Return the N most recently modified protocol files.

    Returns list of dicts sorted by mtime descending::

        [{'path': Path, 'modified': datetime, 'name': str, 'size': int}, ...]

    This is decoupled from the report generator so the GUI can call it
    directly for a "Recent Protocols" menu.
    """
    search_dirs = set()
    for d in [_get_protocol_dir(), _get_lvp_data_dir(), _get_capture_dir()]:
        if d and d.is_dir():
            search_dirs.add(d)

    protocol_files = []
    seen = set()

    for search_dir in search_dirs:
        for json_file in search_dir.rglob('*.json'):
            real = json_file.resolve()
            if real in seen:
                continue
            seen.add(real)
            try:
                with open(json_file, 'r') as f:
                    head = f.read(2048)
                # Protocol files contain these keys
                if any(k in head for k in ('"steps"', '"sequences"', '"scan"',
                                           '"protocol"', '"Protocol"')):
                    stat = json_file.stat()
                    protocol_files.append({
                        'path': json_file,
                        'modified': datetime.datetime.fromtimestamp(stat.st_mtime),
                        'name': json_file.stem,
                        'size': stat.st_size,
                    })
            except (OSError, UnicodeDecodeError):
                continue

    protocol_files.sort(key=lambda x: x['modified'], reverse=True)
    return protocol_files[:n]


# ---------------------------------------------------------------------------
# System Information
# ---------------------------------------------------------------------------

def _collect_system_info():
    """Collect OS, CPU, RAM, disks, power/sleep config."""
    info = {
        'platform': platform.platform(),
        'os': platform.system(),
        'os_version': platform.version(),
        'os_release': platform.release(),
        'architecture': platform.machine(),
        'python_version': sys.version,
        'python_executable': sys.executable,
        'cpu': platform.processor() or 'Unknown',
    }

    def _run(args, timeout=10):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip()
        except Exception as e:
            return f'Error: {e}'

    is_win = platform.system() == 'Windows'
    is_mac = platform.system() == 'Darwin'

    # CPU detail
    if is_win:
        info['cpu_detail'] = _run([
            'wmic', 'cpu', 'get',
            'Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed',
            '/format:list'])
    elif is_mac:
        info['cpu_detail'] = _run(['sysctl', '-n', 'machdep.cpu.brand_string'])
        info['cpu_cores'] = _run(['sysctl', '-n', 'hw.ncpu'])
    else:
        try:
            with open('/proc/cpuinfo', 'r') as f:
                info['cpu_detail'] = f.read()[:2000]
        except OSError:
            pass

    # RAM
    if is_win:
        info['ram_detail'] = _run([
            'wmic', 'memorychip', 'get', 'Capacity,Speed,Manufacturer',
            '/format:list'])
        info['ram_total'] = _run([
            'wmic', 'computersystem', 'get', 'TotalPhysicalMemory',
            '/format:list'])
    elif is_mac:
        try:
            mem_bytes = int(_run(['sysctl', '-n', 'hw.memsize']))
            info['ram_total_gb'] = f'{mem_bytes / (1024**3):.1f} GB'
        except (ValueError, TypeError):
            pass
    else:
        try:
            with open('/proc/meminfo', 'r') as f:
                info['ram_detail'] = f.read()[:1000]
        except OSError:
            pass

    # Disks
    if is_win:
        info['disk_drives'] = _run([
            'wmic', 'diskdrive', 'get',
            'Model,Size,InterfaceType,MediaType,Status', '/format:list'])
        info['disk_partitions'] = _run([
            'wmic', 'logicaldisk', 'get',
            'DeviceID,Size,FreeSpace,FileSystem,VolumeName', '/format:list'])
    elif is_mac:
        info['disk_usage'] = _run(['df', '-h'])
        info['disk_drives'] = _run(['diskutil', 'list'], timeout=15)[:3000]
    else:
        info['disk_usage'] = _run(['df', '-h'])

    # Power / sleep configuration
    if is_win:
        # Targeted: just the sleep timeout and USB selective suspend
        info['power_sleep_ac'] = _run([
            'powercfg', '/query', 'SCHEME_CURRENT',
            '238c9fa8-0aad-41ed-83f4-97be242c8f20',  # Sleep subgroup
            '29f6c1db-86da-48c5-9fdb-f2b67b1f44da'],  # Sleep after (AC)
            timeout=5)
        info['power_sleep_dc'] = _run([
            'powercfg', '/query', 'SCHEME_CURRENT',
            '238c9fa8-0aad-41ed-83f4-97be242c8f20',
            '9d7815a6-7ee4-497e-8888-515a05f02364'],  # Sleep after (DC)
            timeout=5)
        info['usb_selective_suspend'] = _run([
            'powercfg', '/query', 'SCHEME_CURRENT',
            '2a737441-1930-4402-8d77-b2bebba308a3',  # USB subgroup
            '48e6b7a6-50f5-4782-a5d4-53bb8f07e226'],  # USB selective suspend
            timeout=5)
        # Also grab the human-readable active power scheme
        info['power_scheme'] = _run(['powercfg', '/getactivescheme'], timeout=5)
    elif is_mac:
        info['power_settings'] = _run(['pmset', '-g'])

    # Screen resolution and DPI scaling (Kivy rendering issues)
    if is_win:
        info['display'] = _run([
            'wmic', 'path', 'Win32_VideoController', 'get',
            'Name,CurrentHorizontalResolution,CurrentVerticalResolution,'
            'CurrentRefreshRate', '/format:list'])
        # DPI scaling — reg query is more reliable than wmic here
        info['dpi_scaling'] = _run([
            'reg', 'query',
            r'HKCU\Control Panel\Desktop\WindowMetrics',
            '/v', 'AppliedDPI'], timeout=5)
        # Also try the per-monitor DPI awareness setting
        info['dpi_awareness'] = _run([
            'reg', 'query',
            r'HKCU\Control Panel\Desktop',
            '/v', 'LogPixels'], timeout=5)
    elif is_mac:
        info['display'] = _run(['system_profiler', 'SPDisplaysDataType'], timeout=10)

    # Python package versions (critical dependencies)
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True, text=True, timeout=15)
        all_packages = result.stdout.strip()
        info['pip_freeze'] = all_packages
        # Also extract the critical ones for the summary
        critical = ['kivy', 'pypylon', 'ids-peak', 'ids_peak', 'numpy',
                     'numba', 'pyserial', 'Pillow', 'pillow', 'scipy',
                     'opencv', 'cv2', 'psutil', 'requests', 'fastapi']
        critical_pkgs = []
        for line in all_packages.split('\n'):
            pkg_lower = line.lower()
            if any(c in pkg_lower for c in critical):
                critical_pkgs.append(line.strip())
        info['critical_packages'] = critical_pkgs
    except Exception as e:
        info['pip_freeze'] = f'Error: {e}'

    # Camera SDK versions (separate from Python bindings)
    # Basler Pylon
    try:
        import pypylon.pylon as pylon
        info['pylon_version'] = pylon.GetPylonVersion()
    except Exception:
        # Try to find Pylon install from registry or filesystem
        if is_win:
            info['pylon_install'] = _run([
                'reg', 'query',
                r'HKLM\SOFTWARE\Basler\pylon',
                '/ve'], timeout=5)
        else:
            info['pylon_install'] = 'pypylon not importable'

    # IDS Peak
    try:
        import ids_peak
        info['ids_peak_version'] = getattr(ids_peak, '__version__', 'imported but no __version__')
    except Exception:
        if is_win:
            info['ids_peak_install'] = _run([
                'reg', 'query',
                r'HKLM\SOFTWARE\IDS\ids peak',
                '/ve'], timeout=5)
        else:
            info['ids_peak_install'] = 'ids_peak not importable'

    # Windows event log — recent USB/driver errors
    if is_win:
        info['recent_usb_events'] = _run([
            'wevtutil', 'qe', 'System',
            '/q:*[System[(EventID=219 or EventID=507 or EventID=112) '
            'and TimeCreated[timediff(@SystemTime) <= 604800000]]]',
            '/f:text', '/c:50'], timeout=10)

    return info


def _collect_usb_devices():
    """List all USB devices. Returns list of (label, text_content) tuples."""
    devices = []

    def _run(args, timeout=15):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            return r.stdout
        except Exception as e:
            return f'Error: {e}'

    is_win = platform.system() == 'Windows'

    if is_win:
        # PnP entities: USB devices, cameras, COM ports
        devices.append(('PnP_USB_Camera_Ports', _run([
            'wmic', 'path', 'Win32_PnPEntity', 'where',
            "PNPClass='USB' or PNPClass='USBDevice' or PNPClass='Camera' or PNPClass='Ports'",
            'get', 'Name,DeviceID,Status,PNPClass', '/format:list'])))
        devices.append(('USB_Hubs', _run([
            'wmic', 'path', 'Win32_USBHub', 'get',
            'Name,DeviceID,Status', '/format:list'])))
        devices.append(('USB_Controllers', _run([
            'wmic', 'path', 'Win32_USBController', 'get',
            'Name,DeviceID,Status', '/format:list'])))
    elif platform.system() == 'Darwin':
        devices.append(('SPUSBDataType', _run([
            'system_profiler', 'SPUSBDataType'])))
    else:
        devices.append(('lsusb', _run(['lsusb', '-v'])[:5000]))

    # pyserial port enumeration (always)
    try:
        from serial.tools import list_ports
        ports = []
        for port in list_ports.comports(include_links=True):
            ports.append({
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid,
                'vid': f'0x{port.vid:04X}' if port.vid else None,
                'pid': f'0x{port.pid:04X}' if port.pid else None,
                'serial_number': port.serial_number,
                'manufacturer': port.manufacturer,
                'product': port.product,
                'location': port.location,
            })
        devices.append(('pyserial_ports', json.dumps(ports, indent=2)))
    except Exception as e:
        devices.append(('pyserial_error', str(e)))

    return devices


def _collect_device_manager_full():
    """Full Device Manager dump (Windows only, CSV)."""
    if platform.system() != 'Windows':
        return None
    try:
        r = subprocess.run(
            ['wmic', 'path', 'Win32_PnPEntity', 'get',
             'Name,DeviceID,Status,PNPClass,Manufacturer', '/format:csv'],
            capture_output=True, text=True, timeout=30)
        return r.stdout
    except Exception as e:
        return f'Error: {e}'


# ---------------------------------------------------------------------------
# Raw REPL file transfer (Thonny-style) — delegated to drivers.raw_repl
# ---------------------------------------------------------------------------


# Raw REPL file operations are accessed through the board's production
# driver methods (board.enter_raw_repl(), board.repl_list_files(), etc.)
# rather than importing raw_repl functions directly.


# ---------------------------------------------------------------------------
# Motorconfig Validator
# ---------------------------------------------------------------------------

def validate_motorconfig(config_data, source_label=''):
    """Validate motorconfig.json for syntax and sanity.

    Args:
        config_data: bytes, str, or dict.
        source_label: description of where it came from (for the report).

    Returns dict: {'valid': bool, 'warnings': [], 'errors': [], 'parsed': dict|None}
    """
    result = {'valid': True, 'warnings': [], 'errors': [], 'parsed': None}

    # Parse
    if isinstance(config_data, bytes):
        config_data = config_data.decode('utf-8', 'replace')
    if isinstance(config_data, str):
        try:
            parsed = json.loads(config_data)
        except json.JSONDecodeError as e:
            result['valid'] = False
            result['errors'].append(f"Invalid JSON: {e}")
            return result
    elif isinstance(config_data, dict):
        parsed = config_data
    else:
        result['valid'] = False
        result['errors'].append(f"Unexpected type: {type(config_data).__name__}")
        return result

    result['parsed'] = parsed

    # Required keys
    for key in ['Model', 'Serial Number']:
        if key not in parsed:
            result['warnings'].append(f"Missing expected key: '{key}'")

    sn = parsed.get('Serial Number', '')
    if sn and not isinstance(sn, str):
        result['errors'].append(f"Serial Number should be string, got {type(sn).__name__}")
    elif isinstance(sn, str) and len(sn) < 2:
        result['warnings'].append(f"Serial Number seems too short: '{sn}'")

    # Axis configs
    for axis_name in ['X Axis', 'Y Axis', 'Z Axis', 'Turret']:
        axis = parsed.get(axis_name, {})
        if not isinstance(axis, dict):
            continue
        for field in ['Steps Per mm', 'Steps Per Rev', 'Travel mm',
                      'Initial Position after home']:
            val = axis.get(field)
            if val is not None:
                if not isinstance(val, (int, float)):
                    result['errors'].append(
                        f"{axis_name}.{field}: expected number, got {type(val).__name__}")
                elif val < 0 and field != 'Initial Position after home':
                    result['warnings'].append(
                        f"{axis_name}.{field}: negative ({val})")

        travel = axis.get('Travel mm')
        if isinstance(travel, (int, float)):
            if axis_name in ('X Axis', 'Y Axis') and travel > 200:
                result['warnings'].append(
                    f"{axis_name}: Travel {travel}mm seems very large")
            elif axis_name == 'Z Axis' and travel > 50:
                result['warnings'].append(
                    f"{axis_name}: Z Travel {travel}mm seems large")

    # Fan
    fan = parsed.get('Fan', {})
    if isinstance(fan, dict):
        fan_type = fan.get('Type', '')
        if fan_type and fan_type not in ('PWM', 'HILO', 'HiLo'):
            result['warnings'].append(f"Fan Type '{fan_type}' unrecognized")

    # Teststand should not be active on customer units
    ts = parsed.get('Teststand', {})
    if isinstance(ts, dict) and ts.get('Enabled'):
        result['warnings'].append(
            "Teststand mode is ENABLED — should not be active on customer units")

    return result


# ---------------------------------------------------------------------------
# Camera Bandwidth Test
# ---------------------------------------------------------------------------

class CameraBandwidthTest:
    """Stress-test USB camera bandwidth.

    Captures ``num_frames`` at full resolution, 12-bit (or whatever the
    camera is configured for), as fast as possible. Measures throughput,
    counts dropped/None frames, checks frame size consistency.

    Camera-independent: works through the abstract Camera interface
    (``get_image()``).
    """

    def __init__(self, camera, num_frames=BANDWIDTH_TEST_FRAMES):
        self.camera = camera
        self.num_frames = num_frames

    def run(self, progress_callback=None):
        """Run test. Returns results dict."""
        results = {
            'num_frames_requested': self.num_frames,
            'num_frames_received': 0,
            'num_frames_none': 0,
            'num_frames_error': 0,
            'total_bytes': 0,
            'elapsed_seconds': 0,
            'mb_per_second': 0.0,
            'fps_actual': 0.0,
            'frame_sizes': [],
            'errors': [],
            'passed': True,
        }

        # Grab camera info before test
        try:
            if hasattr(self.camera, 'get_image_width') and hasattr(self.camera, 'get_image_height'):
                results['resolution'] = (
                    f"{self.camera.get_image_width()}x{self.camera.get_image_height()}")
            if hasattr(self.camera, 'get_pixel_format'):
                results['pixel_format'] = str(self.camera.get_pixel_format())
            if hasattr(self.camera, 'get_frame_rate'):
                results['configured_fps'] = self.camera.get_frame_rate()
        except Exception:
            pass

        frame_size_set = set()
        start = time.monotonic()

        for i in range(self.num_frames):
            if progress_callback and i % 250 == 0:
                progress_callback(int(100 * i / self.num_frames),
                                  f"Frame {i}/{self.num_frames}")
            try:
                frame = self.camera.get_image()
                if frame is None:
                    results['num_frames_none'] += 1
                else:
                    results['num_frames_received'] += 1
                    nbytes = getattr(frame, 'nbytes', None) or len(frame)
                    results['total_bytes'] += nbytes
                    frame_size_set.add(nbytes)
            except Exception as e:
                results['num_frames_error'] += 1
                if len(results['errors']) < 20:
                    results['errors'].append(f"Frame {i}: {type(e).__name__}: {e}")

            # Hard timeout
            if time.monotonic() - start > BANDWIDTH_TEST_TIMEOUT_S:
                results['errors'].append(
                    f"Timeout at frame {i} after {BANDWIDTH_TEST_TIMEOUT_S}s")
                results['passed'] = False
                break

        elapsed = time.monotonic() - start
        results['elapsed_seconds'] = round(elapsed, 2)

        if elapsed > 0:
            results['mb_per_second'] = round(
                results['total_bytes'] / (1024 * 1024) / elapsed, 2)
            results['fps_actual'] = round(
                results['num_frames_received'] / elapsed, 1)

        results['frame_sizes'] = sorted(frame_size_set)

        # Pass/fail criteria
        if results['num_frames_none'] > 0:
            results['passed'] = False
            results['errors'].append(
                f"{results['num_frames_none']} frames returned None — "
                f"possible USB disconnect or bandwidth issue")
        if results['num_frames_error'] > 0:
            results['passed'] = False
        if len(frame_size_set) > 1:
            results['passed'] = False
            results['errors'].append(
                f"Inconsistent frame sizes: {sorted(frame_size_set)} — "
                f"possible data corruption or config change during test")

        return results


# ---------------------------------------------------------------------------
# Firmware Diagnostics
# ---------------------------------------------------------------------------

class FirmwareDiagnostics:
    """Talks to LED and motor boards to collect diagnostic data."""

    def __init__(self, scope=None):
        self._scope = scope
        self.led_board = getattr(scope, 'led', None) if scope else None
        self.motor_board = getattr(scope, 'motion', None) if scope else None

    def connect_standalone(self):
        """Auto-detect and connect to boards (standalone mode).

        Uses Lumascope.create_diagnostic() to connect through the proper
        API layer instead of importing drivers directly.
        """
        try:
            from modules.lumascope_api import Lumascope
            scope = Lumascope.create_diagnostic()
            self._scope = scope
            self.led_board = scope.led
            self.motor_board = scope.motion
        except Exception as e:
            logger.warning(f"Diagnostic scope creation failed: {e}")
            self._scope = None
            self.led_board = None
            self.motor_board = None

    def _led_ok(self):
        return self.led_board is not None and getattr(self.led_board, 'found', False)

    def _motor_ok(self):
        return self.motor_board is not None and getattr(self.motor_board, 'found', False)

    def _enter_engineering(self):
        """Enter LED engineering mode (send FACTORY + Y confirmation).

        Uses the production driver's exchange_command() for all serial I/O.
        """
        if not self._led_ok():
            return False
        board = self.led_board
        try:
            # Send FACTORY — firmware echoes prompt and waits for Y/N
            board.exchange_command('FACTORY', response_numlines=1, timeout=5)
            time.sleep(0.3)
            # Send Y confirmation — firmware enters engineering mode
            resp = board.exchange_command('Y', response_numlines=1, timeout=5)
            return True
        except Exception as e:
            logger.warning(f"Enter engineering mode failed: {e}")
            return False

    def _exit_engineering(self):
        """Exit LED engineering mode (send Q)."""
        if not self._led_ok():
            return
        self._cmd(self.led_board, 'Q')

    def _cmd(self, board, command, timeout=None):
        """Send command and return response string, or error string.

        Uses the production driver's exchange_command() which supports
        per-call timeout natively.
        """
        if board is None:
            return 'Board not connected'
        try:
            return board.exchange_command(command, timeout=timeout) or 'None'
        except Exception as e:
            return f'Error: {e}'

    def _read_multiline(self, board, command, timeout=60, end_markers=None):
        """Send command and read multi-line response (for SELFTEST etc.).

        Uses the production driver's exchange_multiline() for all serial I/O.
        """
        if board is None:
            return 'Board not connected'
        if end_markers is None:
            end_markers = ['PASS', 'FAIL', 'COMPLETE', 'DONE', 'ERROR']
        try:
            result = board.exchange_multiline(command, timeout=timeout, end_markers=end_markers)
            return result or 'No response'
        except Exception as e:
            return f'Error: {e}'

    # -- High-level collectors --

    def get_led_info(self):
        return self._read_multiline(
            self.led_board, 'INFO', timeout=5,
            end_markers=['RESET CAUSE', 'POWER-ON', 'HARD', 'WDT', 'CALIBRATION'],
        )

    def get_motor_info(self):
        return self._cmd(self.motor_board, 'INFO')

    def get_motor_fullinfo(self):
        return self._cmd(self.motor_board, 'FULLINFO')

    def get_serial_number(self):
        """Extract serial number from FULLINFO.

        Old firmware returns everything on one line:
          Etaluma Motor Controller Board EL-0923 Firmware: 2023-05-30 Model: LS850 Serial: 12006 X homed: True ...
        New firmware uses multi-line with 'Serial Number = ...'
        """
        fullinfo = self.get_motor_fullinfo()
        if not fullinfo or 'not connected' in fullinfo.lower() or 'error' in fullinfo.lower():
            return 'UNKNOWN'
        text = str(fullinfo)

        # Try "Serial Number = <value>" (new firmware, multi-line)
        m = re.search(r'Serial\s*Number\s*[=:]\s*(\S+)', text)
        if m:
            return m.group(1)

        # Try "Serial: <value>" or "Serial = <value>" (old firmware, one-line)
        m = re.search(r'Serial\s*[=:]\s*(\S+)', text, re.IGNORECASE)
        if m:
            return m.group(1)

        # Try "SN: <value>" or "SN = <value>"
        m = re.search(r'\bSN\s*[=:]\s*(\S+)', text, re.IGNORECASE)
        if m:
            return m.group(1)

        # Fallback: return cleaned first chunk
        clean = text.strip().split('\n')[0][:30]
        return clean if clean else 'UNKNOWN'

    def run_led_selftest(self):
        """Run SELFTEST on LED board (v2.0+). Returns full output."""
        if not self._led_ok():
            return 'LED board not connected'
        info = self.get_led_info()
        if not info or not re.search(r'v[2-9]\.\d', str(info)):
            return f'LED firmware too old for SELFTEST (info: {info})'
        self._enter_engineering()
        try:
            return self._read_multiline(self.led_board, 'SELFTEST', timeout=90)
        finally:
            self._exit_engineering()

    def get_led_readings(self):
        return self._read_multiline(
            self.led_board, 'LEDREADS', timeout=30,
            end_markers=['LED7 LED_K', 'AIN1)', 'ERROR'],
        )

    def get_voltages(self):
        return self._cmd(self.motor_board, 'VOLTAGE')

    def get_driver_status_all(self):
        """DRVSTAT for all 4 axes."""
        return {ax: self._cmd(self.motor_board, f'DRVSTAT_{ax}')
                for ax in 'XYZT'}

    def get_motor_positions_all(self):
        """Actual/target/status for all 4 axes."""
        result = {}
        for ax in 'XYZT':
            result[ax] = {
                'actual': self._cmd(self.motor_board, f'ACTUAL_R{ax}'),
                'target': self._cmd(self.motor_board, f'TARGET_R{ax}'),
                'status': self._cmd(self.motor_board, f'STATUS_R{ax}'),
            }
        return result

    def get_fan_status(self):
        return self._cmd(self.motor_board, 'FANSPEED')

    def get_i2c_scan(self):
        return self._cmd(self.led_board, 'I2CSCAN')

    def read_config_files(self, board, label=''):
        """Read config files from a board via raw REPL (Thonny-style).

        Uses the production driver's raw REPL methods (enter_raw_repl,
        repl_list_files, repl_read_file, exit_raw_repl) instead of
        accessing the serial port directly.

        Interrupts running firmware, reads config files, then soft-resets.
        Returns dict of {filename: bytes} or None.
        """
        if board is None:
            return None
        if not hasattr(board, 'enter_raw_repl'):
            return None

        result = {}
        try:
            if not board.enter_raw_repl():
                logger.warning(f"Could not enter raw REPL on {label} board")
                return None

            files = board.repl_list_files()
            logger.info(f"Files on {label} board: {files}")

            for fname in files:
                if fname in FIRMWARE_EXCLUDE_FILES:
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in FIRMWARE_CONFIG_EXTENSIONS:
                    continue
                content = board.repl_read_file(fname)
                if content is not None:
                    result[fname] = content
                    logger.info(f"  Read {label}/{fname} ({len(content)} bytes)")

        except Exception as e:
            logger.warning(f"Raw REPL file read error ({label}): {e}")
        finally:
            board.exit_raw_repl()
            # Verify firmware restarted after raw REPL exit — the serial
            # state may be dirty and normal commands (LED on/off) would
            # fail silently without this check.
            try:
                board.verify_firmware_running(timeout=10)
                logger.info(f"{label} board firmware verified after raw REPL")
            except Exception as e:
                logger.warning(f"{label} board firmware not responding after raw REPL: {e}")

        return result if result else None

    # -- New hardware diagnostics --

    def measure_serial_latency(self, board, command='INFO', iterations=SERIAL_LATENCY_ITERATIONS):
        """Send a command N times and measure round-trip latency.

        Returns dict with min/max/mean/std_dev in milliseconds, plus
        the raw timings list.
        """
        if board is None:
            return {'error': 'Board not connected'}
        timings = []
        errors = 0
        for _ in range(iterations):
            t0 = time.monotonic()
            resp = self._cmd(board, command)
            t1 = time.monotonic()
            if resp and 'Error' not in resp and resp != 'None':
                timings.append((t1 - t0) * 1000)  # ms
            else:
                errors += 1
        if not timings:
            return {'error': f'All {iterations} calls failed', 'errors': errors}
        import statistics
        return {
            'iterations': iterations,
            'errors': errors,
            'min_ms': round(min(timings), 2),
            'max_ms': round(max(timings), 2),
            'mean_ms': round(statistics.mean(timings), 2),
            'std_dev_ms': round(statistics.stdev(timings), 2) if len(timings) > 1 else 0,
            'timings_ms': [round(t, 2) for t in timings],
        }

    def read_tmc5072_registers(self):
        """Read key TMC5072 diagnostic registers via raw SPI commands.

        Returns dict per chip (XY, ZT) with register values.
        Uses the firmware's SPI<axis>0x<addr><payload> command.
        """
        if not self._motor_ok():
            return {'error': 'Motor board not connected'}
        results = {}
        # XY chip: use axis X (motor 0 = X, motor 1 = Y)
        # ZT chip: use axis Z (motor 0 = Z, motor 1 = T)
        for chip_label, axis in [('XY', 'X'), ('ZT', 'Z')]:
            chip = {}
            for reg_name, addr in TMC5072_DIAG_REGISTERS.items():
                # Read: send SPI command with read address (bit 7 = 0)
                # Format: SPI<axis>0x<addr>00000000 (32-bit read)
                read_addr = addr & 0x7F
                cmd = f'SPI{axis}0x{read_addr:02X}00000000'
                resp = self._cmd(self.motor_board, cmd)
                chip[reg_name] = resp
            results[chip_label] = chip
        return results

    def check_voltage_tolerance(self):
        """Read voltages and check against nominal ±tolerance.

        Returns dict with per-rail readings, status (PASS/WARN/FAIL),
        and deviation percentage.
        """
        raw = self.get_voltages()
        if not raw or 'not connected' in raw.lower() or 'Error' in raw:
            return {'error': raw, 'passed': False}

        results = {'raw_response': raw, 'rails': {}, 'passed': True}

        # Parse voltage response — format: "24V=OK  5V=N/A  3V3=N/A  1V2=N/A"
        # or "24V=OK  5V=5.18  3V3=3.31  1V2=1.24"
        for rail_name, nominal in VOLTAGE_NOMINAL.items():
            reading = None
            # Try to find this rail in the response
            if rail_name in raw:
                idx = raw.index(rail_name)
                after = raw[idx + len(rail_name):]
                after = after.lstrip('=: ')
                # Stop at next whitespace or rail name — don't read into next value
                token = after.split()[0] if after.split() else ''
                token = token.rstrip('V')  # Strip trailing 'V' unit
                if token and token not in ('N/A', 'OK', 'MISSING', 'ERROR'):
                    try:
                        reading = float(token)
                    except ValueError:
                        pass

            if reading is None:
                results['rails'][rail_name] = {
                    'nominal': nominal, 'reading': None,
                    'status': 'UNKNOWN', 'deviation_pct': None,
                }
                continue

            deviation_pct = abs(reading - nominal) / nominal * 100
            if deviation_pct > VOLTAGE_FAIL_PCT:
                status = 'FAIL'
                results['passed'] = False
            elif deviation_pct > VOLTAGE_WARN_PCT:
                status = 'WARN'
            else:
                status = 'PASS'

            results['rails'][rail_name] = {
                'nominal': nominal,
                'reading': round(reading, 3),
                'deviation_pct': round(deviation_pct, 2),
                'status': status,
            }

        return results

    def check_led_leakage(self):
        """Read all LED channels with LEDs off, check for leakage current.

        Returns dict with per-channel readings and pass/fail.
        Requires engineering mode (enters/exits automatically).
        """
        if not self._led_ok():
            return {'error': 'LED board not connected', 'passed': False}
        self._enter_engineering()
        try:
            raw = self.get_led_readings()
        finally:
            self._exit_engineering()
        if not raw or 'not connected' in raw.lower() or 'Error' in raw:
            return {'error': raw, 'passed': False}

        results = {'raw_response': raw, 'channels': {}, 'passed': True}

        # Parse LEDREADS response — v2.0+ format:
        # "LED0 I_SENS  (AIN14): 0.0234V  ->     0.3 mA"
        # "LED0 LED_K   (AIN15): 2.0833V"
        for ch in range(8):
            reading = None
            # Look for "LEDx I_SENS ... -> <value> mA"
            m = re.search(
                rf'LED{ch}\s+I_SENS\s+\(AIN\d+\):\s*[\d.]+V\s*->\s*([-\d.]+)\s*mA',
                raw,
            )
            if m:
                try:
                    reading = float(m.group(1))
                except ValueError:
                    pass

            status = 'PASS'
            if reading is not None and abs(reading) > LED_LEAKAGE_WARN_MA:
                status = 'WARN'
                results['passed'] = False

            results['channels'][f'CH{ch}'] = {
                'i_sens_mA': reading,
                'status': status,
            }

        return results

    def verify_fan_tachometer(self):
        """Set fan to known duty, wait, read tachometer.

        This is a non-critical / informational test. Many units in the
        field do not have a tachometer wire installed, so a zero RPM
        reading does not necessarily indicate a fault.

        Returns dict with RPM readings (never sets passed=False).
        """
        if not self._motor_ok():
            return {'error': 'Motor board not connected'}

        results = {'tests': [], 'note': 'Informational only — many units lack tachometer hardware'}

        # Test 1: Set fan to ~50% duty, read RPM
        self._cmd(self.motor_board, 'FAN:50')
        time.sleep(2.0)  # Wait for fan to spin up
        rpm_response = self._cmd(self.motor_board, 'FANSPEED')

        rpm_value = None
        if rpm_response and 'Error' not in rpm_response:
            for token in rpm_response.replace('=', ' ').replace(':', ' ').split():
                try:
                    val = float(token)
                    if val >= 0:
                        rpm_value = val
                        break
                except ValueError:
                    continue

        has_tach = rpm_value is not None and rpm_value > 100
        results['tests'].append({
            'duty_pct': 50,
            'rpm': rpm_value,
            'raw_response': rpm_response,
            'tachometer_detected': has_tach,
        })

        # Test 2: Fan off, read RPM
        self._cmd(self.motor_board, 'FAN:0')
        time.sleep(3.0)
        rpm_off = self._cmd(self.motor_board, 'FANSPEED')

        results['tests'].append({
            'duty_pct': 0,
            'raw_response': rpm_off,
        })

        results['tachometer_present'] = has_tach

        return results

    def run_homing_test(self):
        """Home all axes and verify positions match expected.

        Returns dict with per-axis results including final position
        and whether homing completed successfully.
        """
        if not self._motor_ok():
            return {'error': 'Motor board not connected'}

        results = {'axes': {}, 'passed': True}

        # Home Z first (safety — move Z up before XY)
        zhome_resp = self._cmd(self.motor_board, 'ZHOME', timeout=60)
        results['axes']['Z'] = {
            'home_response': zhome_resp,
            'actual_after': self._cmd(self.motor_board, 'ACTUAL_RZ'),
            'target_after': self._cmd(self.motor_board, 'TARGET_RZ'),
        }

        # Home turret
        thome_resp = self._cmd(self.motor_board, 'THOME', timeout=30)
        results['axes']['T'] = {
            'home_response': thome_resp,
            'actual_after': self._cmd(self.motor_board, 'ACTUAL_RT'),
            'target_after': self._cmd(self.motor_board, 'TARGET_RT'),
        }

        # Home XY
        home_resp = self._cmd(self.motor_board, 'HOME', timeout=60)
        for ax in 'XY':
            results['axes'][ax] = {
                'home_response': home_resp,
                'actual_after': self._cmd(self.motor_board, f'ACTUAL_R{ax}'),
                'target_after': self._cmd(self.motor_board, f'TARGET_R{ax}'),
            }

        # Check for errors in responses
        for ax, data in results['axes'].items():
            if 'Error' in str(data.get('home_response', '')):
                results['passed'] = False
                data['status'] = 'FAIL'
            else:
                data['status'] = 'OK'

        return results

    # NOTE: Motor repeatability testing requires optical feedback (e.g. a
    # test target on the stage imaged by the camera) because there are no
    # encoders — ACTUAL_R just reads the TMC5072 step counter which will
    # always agree with TARGET_R. True mechanical repeatability (backlash,
    # missed steps) must be measured optically. This is planned as a future
    # QC feature with a calibration target.


# ---------------------------------------------------------------------------
# Main Report Generator
# ---------------------------------------------------------------------------

class TechSupportReport:
    """Generate a comprehensive diagnostic ZIP for Etaluma tech support."""

    def __init__(self, scope=None, session=None,
                 led_board=None, motor_board=None, camera=None):
        # Store scope as primary interface — avoid extracting raw driver
        # objects at this level.  FirmwareDiagnostics handles board access.
        if scope is not None:
            self.scope = scope
        elif session is not None:
            # Legacy ScopeSession wrapper — build a minimal scope-like object
            self.scope = session
        else:
            self.scope = None

        # FirmwareDiagnostics owns the board references.  In integrated
        # mode it extracts them from scope; standalone callers pass raw
        # boards only when no scope is available.
        if self.scope is not None:
            self.diag = FirmwareDiagnostics(scope=self.scope)
        elif led_board is not None or motor_board is not None:
            # Standalone with explicit boards (no scope)
            diag = FirmwareDiagnostics()
            diag.led_board = led_board
            diag.motor_board = motor_board
            self.diag = diag
        else:
            # No scope, no boards — standalone will call diag.connect_standalone()
            self.diag = FirmwareDiagnostics()

        self._cancelled = False
        self._meta = {}

    @property
    def _camera(self):
        """Get camera through scope API (returns None if unavailable)."""
        return getattr(self.scope, 'camera', None)

    def cancel(self):
        self._cancelled = True

    def generate(self, callback=None, include_bandwidth_test=False,
                 output_dir=None):
        """Generate report. Returns path to ZIP, or None on failure."""
        cb = callback or (lambda pct, msg: None)
        try:
            return self._generate(cb, include_bandwidth_test, output_dir)
        except _Cancelled:
            cb(100, "Cancelled.")
            return None
        except Exception as e:
            logger.error(f"Report failed: {e}", exc_info=True)
            cb(100, f"Error: {e}")
            return None

    def _check_cancel(self):
        if self._cancelled:
            raise _Cancelled()

    def _generate(self, cb, include_bw, output_dir):
        cb(0, "Starting report generation...")
        self._write_log_delimiter()

        with tempfile.TemporaryDirectory(prefix='lvp_report_') as tmp:
            tmp = pathlib.Path(tmp)

            # 1. Firmware info + serial number  (0-5%)
            cb(1, "Querying firmware...")
            sn = self._step_firmware_info(tmp)
            self._check_cancel()

            # 2. Config files from both boards via raw REPL  (5-10%)
            cb(6, "Backing up firmware config files...")
            self._step_configbackup(tmp)
            self._check_cancel()

            # 3. LED selftest  (10-15%)
            cb(11, "Running LED selftest...")
            self._step_firmware_tests(tmp)
            self._check_cancel()

            # 4. Voltage tolerance + LED leakage checks  (15-18%)
            cb(16, "Checking voltages and LED leakage...")
            self._step_voltage_and_led_checks(tmp)
            self._check_cancel()

            # 5. TMC5072 register dump  (18-20%)
            cb(19, "Reading motor driver registers...")
            self._step_tmc_registers(tmp)
            self._check_cancel()

            # 6. Fan tachometer verification  (20-23%)
            cb(21, "Testing fan...")
            self._step_fan_test(tmp)
            self._check_cancel()

            # 7. Serial latency measurement  (23-27%)
            cb(24, "Measuring serial latency...")
            self._step_serial_latency(tmp)
            self._check_cancel()

            # 8. Homing test  (27-35%)
            cb(28, "Homing all axes...")
            self._step_homing_test(tmp)
            self._check_cancel()

            # 9. Camera diagnostics (temp)  (38-41%)
            cb(39, "Checking camera...")
            self._step_camera_diagnostics(tmp)
            self._check_cancel()

            # 11. System info  (48-52%)
            cb(49, "Collecting system information...")
            self._step_system_info(tmp)
            self._check_cancel()

            # 12. USB devices  (52-55%)
            cb(53, "Scanning USB devices...")
            self._step_usb_devices(tmp)
            self._check_cancel()

            # 13. Disk speed test  (55-60%)
            cb(56, "Testing disk write speed...")
            self._step_disk_speed(tmp)
            self._check_cancel()

            # 14. Data folder  (60-63%)
            cb(61, "Copying data folder...")
            self._step_data_folder(tmp)
            self._check_cancel()

            # 15. Logs  (63-66%)
            cb(64, "Copying log files...")
            self._step_logs(tmp)
            self._check_cancel()

            # 16. Backlash results  (66-69%)
            cb(67, "Collecting backlash test results...")
            self._step_backlash(tmp)
            self._check_cancel()

            # 17. Recent protocols  (69-72%)
            cb(70, "Collecting recent protocols...")
            self._step_protocols(tmp)
            self._check_cancel()

            # 18. Hardware serial tests (pytest)  (72-80%)
            cb(73, "Running hardware serial tests...")
            self._step_hardware_tests(tmp)
            self._check_cancel()

            # 19. Bandwidth test (optional)  (80-94%)
            if include_bw and self._camera is not None:
                cb(81, "Running camera bandwidth test (this takes a while)...")
                self._step_bandwidth(tmp, cb)
                self._check_cancel()

            # 20. Metadata + ZIP  (94-100%)
            cb(95, "Writing metadata...")
            self._step_metadata(tmp, sn)

            cb(97, "Creating ZIP file...")
            zip_path = self._create_zip(tmp, sn, output_dir)

            cb(100, f"Done — {zip_path.name}")
            return zip_path

    # -- Steps ---------------------------------------------------------------

    def _step_firmware_info(self, tmp):
        d = tmp / 'firmware_info'
        d.mkdir()

        led_info = self.diag.get_led_info()
        motor_info = self.diag.get_motor_info()
        fullinfo = self.diag.get_motor_fullinfo()
        sn = self.diag.get_serial_number()

        with open(d / 'led_info.txt', 'w') as f:
            f.write(f"LED Board INFO:\n{led_info}\n")

        with open(d / 'motor_info.txt', 'w') as f:
            f.write(f"Motor Board INFO:\n{motor_info}\n\n")
            f.write(f"Motor Board FULLINFO:\n{fullinfo}\n\n")
            f.write(f"Serial Number: {sn}\n")

        voltages = self.diag.get_voltages()
        with open(d / 'voltages.txt', 'w') as f:
            f.write(f"Power Rail Voltages:\n{voltages}\n")

        positions = self.diag.get_motor_positions_all()
        drvstat = self.diag.get_driver_status_all()
        with open(d / 'motor_status.txt', 'w') as f:
            f.write("Motor Positions:\n")
            for ax, data in positions.items():
                f.write(f"  {ax}: {json.dumps(data)}\n")
            f.write("\nTMC5072 Driver Status:\n")
            for ax, st in drvstat.items():
                f.write(f"  {ax}: {st}\n")

        fan = self.diag.get_fan_status()
        i2c = self.diag.get_i2c_scan()
        led_readings = self.diag.get_led_readings()
        with open(d / 'peripherals.txt', 'w') as f:
            f.write(f"Fan: {fan}\n\nI2C Scan: {i2c}\n\n")
            f.write(f"LED Readings (baseline, all off):\n{led_readings}\n")

        self._meta['serial_number'] = sn
        self._meta['led_info'] = str(led_info)
        self._meta['motor_info'] = str(motor_info)
        return sn

    def _step_configbackup(self, tmp):
        """Retrieve config files from BOTH boards via raw REPL."""
        d = tmp / 'firmware_configs'
        d.mkdir()

        for board, label in [(self.diag.led_board, 'led'), (self.diag.motor_board, 'motor')]:
            files = self.diag.read_config_files(board, label)
            if files is None:
                with open(d / f'{label}_config_UNAVAILABLE.txt', 'w') as f:
                    f.write(
                        f"Could not read config files from {label} board.\n"
                        f"Board may not be connected or raw REPL entry failed.\n"
                    )
                    if label == 'led':
                        f.write(
                            "Note: LED board raw REPL may require a power cycle "
                            "before config files can be read (SPI state issue).\n"
                        )
                continue

            board_dir = d / label
            board_dir.mkdir()
            for filename, content in files.items():
                # Sanitize filename (strip path components to prevent traversal)
                safe_name = pathlib.Path(filename).name
                if safe_name.lower() in ('main.py', 'boot.py'):
                    continue
                filepath = board_dir / safe_name
                with open(filepath, 'wb') as f:
                    f.write(content)
                logger.info(f"  Saved {label}/{safe_name} ({len(content)} bytes)")

                # Validate motorconfig.json
                if safe_name.lower() == 'motorconfig.json':
                    validation = validate_motorconfig(content, f'{label}/{safe_name}')
                    with open(d / f'{label}_motorconfig_validation.txt', 'w') as f:
                        f.write(f"Motorconfig Validation ({label}/{safe_name})\n")
                        f.write(f"Valid: {validation['valid']}\n\n")
                        if validation['errors']:
                            f.write("ERRORS:\n")
                            for e in validation['errors']:
                                f.write(f"  !! {e}\n")
                        if validation['warnings']:
                            f.write("WARNINGS:\n")
                            for w in validation['warnings']:
                                f.write(f"  -- {w}\n")
                        if not validation['errors'] and not validation['warnings']:
                            f.write("All checks passed.\n")

    def _step_firmware_tests(self, tmp):
        d = tmp / 'firmware_tests'
        d.mkdir()

        selftest = self.diag.run_led_selftest()
        with open(d / 'led_selftest.txt', 'w') as f:
            f.write(f"LED SELFTEST:\n\n{selftest}\n")

    def _step_voltage_and_led_checks(self, tmp):
        """Check voltage rails against tolerance and LED leakage."""
        d = tmp / 'hardware_checks'
        d.mkdir()

        # Voltage tolerance
        vtol = self.diag.check_voltage_tolerance()
        with open(d / 'voltage_tolerance.txt', 'w') as f:
            f.write("Power Rail Voltage Tolerance Check\n" + "=" * 45 + "\n\n")
            f.write(f"Raw response: {vtol.get('raw_response', 'N/A')}\n\n")
            if 'error' in vtol:
                f.write(f"Error: {vtol['error']}\n")
            else:
                for rail, data in vtol.get('rails', {}).items():
                    r = data.get('reading')
                    n = data.get('nominal')
                    d_pct = data.get('deviation_pct')
                    st = data.get('status', '?')
                    if r is not None:
                        f.write(f"  {rail:6s}  nominal={n}V  actual={r}V  "
                                f"deviation={d_pct}%  [{st}]\n")
                    else:
                        f.write(f"  {rail:6s}  could not parse reading  [{st}]\n")
                f.write(f"\nOverall: {'PASS' if vtol.get('passed') else 'FAIL/WARN'}\n")
                f.write(f"(Warn >{VOLTAGE_WARN_PCT}%, Fail >{VOLTAGE_FAIL_PCT}%)\n")
        with open(d / 'voltage_tolerance.json', 'w') as f:
            json.dump(vtol, f, indent=2, default=str)

        # LED leakage
        leakage = self.diag.check_led_leakage()
        with open(d / 'led_leakage.txt', 'w') as f:
            f.write("LED Leakage Check (all LEDs off)\n" + "=" * 40 + "\n\n")
            f.write(f"Raw response: {leakage.get('raw_response', 'N/A')}\n\n")
            if 'error' in leakage:
                f.write(f"Error: {leakage['error']}\n")
            else:
                for ch, data in leakage.get('channels', {}).items():
                    val = data.get('i_sens_mA')
                    st = data.get('status', '?')
                    if val is not None:
                        f.write(f"  {ch}: {val:7.3f} mA  [{st}]\n")
                    else:
                        f.write(f"  {ch}: could not parse  [{st}]\n")
                f.write(f"\nOverall: {'PASS' if leakage.get('passed') else 'WARN — leakage detected'}\n")
                f.write(f"(Threshold: {LED_LEAKAGE_WARN_MA} mA)\n")

    def _step_tmc_registers(self, tmp):
        """Dump key TMC5072 diagnostic registers."""
        d = tmp / 'hardware_checks'
        d.mkdir(exist_ok=True)

        regs = self.diag.read_tmc5072_registers()
        with open(d / 'tmc5072_registers.txt', 'w') as f:
            f.write("TMC5072 Register Dump\n" + "=" * 40 + "\n\n")
            if 'error' in regs:
                f.write(f"Error: {regs['error']}\n")
            else:
                for chip, registers in regs.items():
                    f.write(f"--- {chip} chip ---\n")
                    for reg_name, value in registers.items():
                        f.write(f"  {reg_name:16s}: {value}\n")
                    f.write("\n")
                f.write("Key flags in DRV_STATUS:\n")
                f.write("  Bit 0-1: SG_RESULT (StallGuard)\n")
                f.write("  Bit 24: s2ga (short to GND coil A)\n")
                f.write("  Bit 25: s2gb (short to GND coil B)\n")
                f.write("  Bit 26: ola (open load A — motor disconnected?)\n")
                f.write("  Bit 27: olb (open load B — motor disconnected?)\n")
                f.write("  Bit 25: ot (overtemperature shutdown)\n")
                f.write("  Bit 26: otpw (overtemperature pre-warning)\n")
                f.write("  Bit 31: stst (standstill indicator)\n")
        with open(d / 'tmc5072_registers.json', 'w') as f:
            json.dump(regs, f, indent=2, default=str)

    def _step_fan_test(self, tmp):
        """Test fan operation via tachometer (informational — many units lack tach)."""
        d = tmp / 'hardware_checks'
        d.mkdir(exist_ok=True)

        fan = self.diag.verify_fan_tachometer()
        with open(d / 'fan_test.txt', 'w') as f:
            f.write("Fan Test (informational)\n" + "=" * 40 + "\n")
            f.write("Note: Many units in the field do not have a tachometer\n")
            f.write("wire installed. Zero RPM does not necessarily mean the\n")
            f.write("fan is broken.\n\n")
            if 'error' in fan:
                f.write(f"Error: {fan['error']}\n")
            else:
                tach = fan.get('tachometer_present', False)
                f.write(f"Tachometer detected: {'Yes' if tach else 'No'}\n\n")
                for t in fan.get('tests', []):
                    f.write(f"  Duty {t.get('duty_pct', '?')}%: ")
                    if 'rpm' in t:
                        f.write(f"RPM={t.get('rpm', '?')}  ")
                    f.write(f"raw={t.get('raw_response', '?')}\n")

    def _step_serial_latency(self, tmp):
        """Measure serial round-trip latency on both boards."""
        d = tmp / 'hardware_checks'
        d.mkdir(exist_ok=True)

        # Run latency test once per board, write both text and JSON from same data
        results = {}
        for board, label in [(self.diag.led_board, 'LED'), (self.diag.motor_board, 'Motor')]:
            results[label] = self.diag.measure_serial_latency(board, 'INFO')

        with open(d / 'serial_latency.txt', 'w') as f:
            f.write("Serial Round-Trip Latency\n" + "=" * 40 + "\n\n")
            for label, latency in results.items():
                f.write(f"--- {label} Board ({SERIAL_LATENCY_ITERATIONS}x INFO) ---\n")
                if 'error' in latency:
                    f.write(f"  Error: {latency['error']}\n\n")
                else:
                    f.write(f"  Min:     {latency['min_ms']:7.2f} ms\n")
                    f.write(f"  Max:     {latency['max_ms']:7.2f} ms\n")
                    f.write(f"  Mean:    {latency['mean_ms']:7.2f} ms\n")
                    f.write(f"  Std dev: {latency['std_dev_ms']:7.2f} ms\n")
                    f.write(f"  Errors:  {latency['errors']}\n\n")
                    # Flag suspicious results
                    if latency['max_ms'] > 100:
                        f.write(f"  ** WARNING: max latency {latency['max_ms']}ms "
                                f"— possible USB suspend or contention **\n\n")
                    if latency['std_dev_ms'] > 20:
                        f.write(f"  ** WARNING: high variance (std={latency['std_dev_ms']}ms) "
                                f"— unstable USB connection **\n\n")

        with open(d / 'serial_latency.json', 'w') as f:
            summary = {label: {k: v for k, v in lat.items() if k != 'timings_ms'}
                       for label, lat in results.items()}
            json.dump(summary, f, indent=2, default=str)

    def _step_homing_test(self, tmp):
        """Home all axes and record results."""
        d = tmp / 'motion_tests'
        d.mkdir()

        homing = self.diag.run_homing_test()
        with open(d / 'homing_test.txt', 'w') as f:
            f.write("Homing Test\n" + "=" * 40 + "\n\n")
            if 'error' in homing:
                f.write(f"Error: {homing['error']}\n")
            else:
                f.write(f"Overall: {'PASS' if homing['passed'] else 'FAIL'}\n\n")
                for ax, data in homing.get('axes', {}).items():
                    f.write(f"  {ax} axis:\n")
                    f.write(f"    Home response: {data.get('home_response')}\n")
                    f.write(f"    Actual after:  {data.get('actual_after')}\n")
                    f.write(f"    Target after:  {data.get('target_after')}\n")
                    f.write(f"    Status:        {data.get('status')}\n\n")
        with open(d / 'homing_test.json', 'w') as f:
            json.dump(homing, f, indent=2, default=str)

    def _step_camera_diagnostics(self, tmp):
        """Read camera sensor temperature and basic info."""
        d = tmp / 'camera_info'
        d.mkdir()

        camera = self._camera
        if camera is None:
            (d / 'no_camera.txt').write_text("No camera available.\n")
            return

        info = {}
        # Try to read standard camera properties
        for attr in ['get_image_width', 'get_image_height', 'get_pixel_format',
                     'get_frame_rate', 'get_gain', 'get_exposure',
                     'get_temperature', 'get_sensor_temperature',
                     'get_device_temperature']:
            if hasattr(camera, attr):
                try:
                    info[attr] = getattr(camera, attr)()
                except Exception as e:
                    info[attr] = f'Error: {e}'

        # Read all temperature sensors through the camera driver API
        try:
            temps = camera.get_all_temperatures()
            for name, temp_c in temps.items():
                info[f'Temperature_{name}'] = temp_c
        except Exception as e:
            info['Temperature'] = f'Error: {e}'

        with open(d / 'camera_info.txt', 'w') as f:
            f.write("Camera Information\n" + "=" * 40 + "\n\n")
            f.write(f"Camera type: {type(camera).__name__}\n\n")
            for key, val in info.items():
                label = key.replace('get_', '').replace('_', ' ').title()
                f.write(f"  {label}: {val}\n")
                # Flag hot cameras
                if 'temp' in key.lower() and isinstance(val, (int, float)):
                    if val > 60:
                        f.write(f"    ** WARNING: sensor temperature {val}°C "
                                f"is high — check cooling/ventilation **\n")
                    elif val > 45:
                        f.write(f"    ** Note: sensor at {val}°C "
                                f"(elevated, may affect image noise) **\n")
        with open(d / 'camera_info.json', 'w') as f:
            json.dump(info, f, indent=2, default=str)

    def _step_disk_speed(self, tmp):
        """Write a test file to the capture drive, measure MB/s."""
        d = tmp / 'disk_speed'
        d.mkdir()

        capture_dir = _get_capture_dir()
        # Use the capture directory's drive for the test
        test_dir = capture_dir if capture_dir and capture_dir.is_dir() else _get_desktop()

        results = {
            'test_directory': str(test_dir),
            'test_size_mb': DISK_SPEED_TEST_MB,
        }

        test_file = test_dir / '.lvp_disk_speed_test.tmp'
        try:
            # Generate random-ish data (compressible data would give
            # misleadingly fast results on SSDs with compression)
            chunk = os.urandom(1024 * 1024)  # 1 MB of random data

            # Write test
            start = time.monotonic()
            with open(test_file, 'wb') as f:
                for _ in range(DISK_SPEED_TEST_MB):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            write_elapsed = time.monotonic() - start
            write_mbps = DISK_SPEED_TEST_MB / write_elapsed

            # Read test
            start = time.monotonic()
            with open(test_file, 'rb') as f:
                while f.read(1024 * 1024):
                    pass
            read_elapsed = time.monotonic() - start
            read_mbps = DISK_SPEED_TEST_MB / read_elapsed

            results['write_mbps'] = round(write_mbps, 1)
            results['write_elapsed_s'] = round(write_elapsed, 2)
            results['read_mbps'] = round(read_mbps, 1)
            results['read_elapsed_s'] = round(read_elapsed, 2)
            results['passed'] = write_mbps >= DISK_SPEED_WARN_MBPS

            # Check free space while we're at it
            try:
                usage = shutil.disk_usage(test_dir)
                results['disk_total_gb'] = round(usage.total / (1024**3), 1)
                results['disk_free_gb'] = round(usage.free / (1024**3), 1)
                results['disk_used_pct'] = round(usage.used / usage.total * 100, 1)
            except Exception:
                pass

        except Exception as e:
            results['error'] = str(e)
            results['passed'] = False
        finally:
            # Clean up test file
            try:
                test_file.unlink(missing_ok=True)
            except Exception:
                pass

        with open(d / 'disk_speed.txt', 'w') as f:
            f.write("Disk Speed Test\n" + "=" * 40 + "\n\n")
            f.write(f"Test directory: {results.get('test_directory')}\n")
            f.write(f"Test size:      {DISK_SPEED_TEST_MB} MB\n\n")
            if 'error' in results:
                f.write(f"Error: {results['error']}\n")
            else:
                f.write(f"Write speed: {results.get('write_mbps', '?')} MB/s "
                        f"({results.get('write_elapsed_s', '?')}s)\n")
                f.write(f"Read speed:  {results.get('read_mbps', '?')} MB/s "
                        f"({results.get('read_elapsed_s', '?')}s)\n\n")
                if 'disk_free_gb' in results:
                    f.write(f"Disk total:  {results['disk_total_gb']} GB\n")
                    f.write(f"Disk free:   {results['disk_free_gb']} GB\n")
                    f.write(f"Disk used:   {results['disk_used_pct']}%\n\n")
                f.write(f"Result: {'PASS' if results.get('passed') else 'FAIL'}\n")
                if not results.get('passed'):
                    f.write(f"** Write speed below {DISK_SPEED_WARN_MBPS} MB/s — "
                            f"video recording may drop frames **\n")
        with open(d / 'disk_speed.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def _step_system_info(self, tmp):
        d = tmp / 'system_info'
        d.mkdir()

        info = _collect_system_info()
        with open(d / 'system_info.json', 'w') as f:
            json.dump(info, f, indent=2, default=str)

        with open(d / 'summary.txt', 'w') as f:
            f.write(f"OS:      {info.get('platform')}\n")
            f.write(f"Python:  {info.get('python_version')}\n")
            f.write(f"CPU:     {info.get('cpu')}\n")
            if 'cpu_detail' in info:
                f.write(f"         {info['cpu_detail'][:200]}\n")
            for k in ['ram_total_gb', 'ram_total']:
                if k in info:
                    f.write(f"RAM:     {info[k]}\n")
                    break
            f.write(f"\nPower scheme: {info.get('power_scheme', 'N/A')}\n")
            if 'usb_selective_suspend' in info:
                f.write(f"USB selective suspend:\n{info['usb_selective_suspend']}\n")

            # Display / DPI
            if 'display' in info:
                f.write(f"\nDisplay:\n{info['display'][:500]}\n")
            if 'dpi_scaling' in info:
                f.write(f"DPI scaling: {info['dpi_scaling']}\n")

            # Camera SDKs
            if 'pylon_version' in info:
                f.write(f"\nBasler Pylon SDK: {info['pylon_version']}\n")
            elif 'pylon_install' in info:
                f.write(f"\nBasler Pylon: {info['pylon_install']}\n")
            if 'ids_peak_version' in info:
                f.write(f"IDS Peak SDK: {info['ids_peak_version']}\n")
            elif 'ids_peak_install' in info:
                f.write(f"IDS Peak: {info['ids_peak_install']}\n")

            # Critical Python packages
            critical = info.get('critical_packages', [])
            if critical:
                f.write(f"\nCritical Python packages:\n")
                for pkg in critical:
                    f.write(f"  {pkg}\n")

            # Recent USB events
            usb_events = info.get('recent_usb_events', '')
            if usb_events and 'Error' not in usb_events[:20]:
                f.write(f"\nRecent USB/driver events (last 7 days):\n")
                f.write(usb_events[:3000] if usb_events else '  None found\n')

        # Also write full pip freeze as separate file for easy diff
        pip_freeze = info.get('pip_freeze', '')
        if pip_freeze and 'Error' not in pip_freeze[:10]:
            with open(d / 'pip_freeze.txt', 'w') as f:
                f.write(pip_freeze)

    def _step_usb_devices(self, tmp):
        d = tmp / 'usb_devices'
        d.mkdir()

        for label, content in _collect_usb_devices():
            safe = label.replace(' ', '_').replace('/', '_')
            with open(d / f'{safe}.txt', 'w') as f:
                f.write(content if isinstance(content, str) else str(content))

        devmgr = _collect_device_manager_full()
        if devmgr and 'Error' not in str(devmgr)[:20]:
            with open(d / 'device_manager_full.csv', 'w') as f:
                f.write(devmgr)

    def _step_data_folder(self, tmp):
        data_dir = _get_lvp_data_dir()
        if not data_dir or not data_dir.is_dir():
            return
        dest = tmp / 'data'
        try:
            shutil.copytree(data_dir, dest, dirs_exist_ok=True,
                           ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        except Exception as e:
            dest.mkdir(exist_ok=True)
            (dest / 'ERROR.txt').write_text(f"Copy failed: {e}\n")

    def _step_logs(self, tmp):
        logs_dir = _get_lvp_logs_dir()
        if not logs_dir or not logs_dir.is_dir():
            return
        dest = tmp / 'logs'
        try:
            shutil.copytree(logs_dir, dest, dirs_exist_ok=True)
        except Exception as e:
            dest.mkdir(exist_ok=True)
            (dest / 'ERROR.txt').write_text(f"Copy failed: {e}\n")

    def _step_backlash(self, tmp):
        capture_dir = _get_capture_dir()
        if not capture_dir or not capture_dir.is_dir():
            return
        dest = tmp / 'backlash_tests'
        found = False
        for pattern in BACKLASH_FOLDER_PATTERNS:
            for match in capture_dir.glob(f'*{pattern}*'):
                if match.is_dir():
                    found = True
                    try:
                        shutil.copytree(match, dest / match.name, dirs_exist_ok=True)
                    except Exception as e:
                        dest.mkdir(exist_ok=True)
                        (dest / f'{match.name}_ERROR.txt').write_text(str(e))
        if not found:
            dest.mkdir(exist_ok=True)
            (dest / 'none_found.txt').write_text(
                "No backlash test folders found in capture directory.\n")

    def _step_protocols(self, tmp):
        d = tmp / 'recent_protocols'
        d.mkdir()

        protocols = get_recent_protocols(RECENT_PROTOCOL_COUNT)
        if not protocols:
            (d / 'none_found.txt').write_text("No protocol files found.\n")
            return

        with open(d / '_index.txt', 'w') as f:
            f.write(f"Most Recent {len(protocols)} Protocols\n{'=' * 40}\n\n")
            for i, p in enumerate(protocols, 1):
                f.write(f"{i:2d}. {p['name']}\n")
                f.write(f"    Modified: {p['modified']}\n")
                f.write(f"    Path:     {p['path']}\n")
                f.write(f"    Size:     {p['size']} bytes\n\n")

        for p in protocols:
            try:
                shutil.copy2(p['path'], d / f"{p['name']}.json")
            except Exception as e:
                logger.warning(f"Could not copy protocol {p['path']}: {e}")

        self._meta['recent_protocols'] = [
            {'name': p['name'], 'modified': p['modified'].isoformat()}
            for p in protocols
        ]

    def _step_hardware_tests(self, tmp):
        """Run test_hardware_serial.py with --run-hardware.

        This runs the real serial benchmarks: exchange_command latency,
        LED on/off cycles, position query throughput, rapid STATUS queries,
        INFO response validation, etc. These directly exercise the actual
        hardware and will reveal communication problems.

        We intentionally skip simulation tests (test_simulators,
        test_serial_safety, test_scope_api, etc.) because those verify
        the test infrastructure, not the customer's hardware.
        """
        d = tmp / 'test_results'
        d.mkdir()

        app_root = _get_app_root()
        tests_dir = app_root / 'tests'

        if not tests_dir.is_dir():
            (d / 'skipped.txt').write_text("Tests directory not found.\n")
            return

        test_file = tests_dir / 'test_hardware_serial.py'
        if not test_file.exists():
            (d / 'skipped.txt').write_text(
                "test_hardware_serial.py not found.\n")
            return

        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', str(test_file),
                 '--run-hardware', '-v', '--tb=short', '-q'],
                capture_output=True, text=True, timeout=180,
                cwd=str(app_root),
            )
            with open(d / 'test_hardware_serial.txt', 'w') as f:
                f.write(f"test_hardware_serial.py (--run-hardware)\n")
                f.write(f"Return code: {result.returncode}\n\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write(f"\nSTDERR:\n{result.stderr}")
        except subprocess.TimeoutExpired:
            (d / 'test_hardware_serial.txt').write_text(
                "TIMED OUT after 180s\n")
        except Exception as e:
            (d / 'test_hardware_serial.txt').write_text(f"Error: {e}\n")

    def _step_bandwidth(self, tmp, cb):
        d = tmp / 'bandwidth_test'
        d.mkdir()

        def bw_cb(pct, msg):
            cb(81 + int(pct * 0.13), f"Bandwidth: {msg}")

        bw = CameraBandwidthTest(self._camera)
        results = bw.run(progress_callback=bw_cb)

        with open(d / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        with open(d / 'summary.txt', 'w') as f:
            f.write("Camera Bandwidth Test\n" + "=" * 40 + "\n\n")
            f.write(f"Resolution:      {results.get('resolution', '?')}\n")
            f.write(f"Pixel format:    {results.get('pixel_format', '?')}\n")
            f.write(f"Configured FPS:  {results.get('configured_fps', '?')}\n\n")
            f.write(f"Frames requested: {results['num_frames_requested']}\n")
            f.write(f"Frames received:  {results['num_frames_received']}\n")
            f.write(f"Frames None:      {results['num_frames_none']}\n")
            f.write(f"Frames errored:   {results['num_frames_error']}\n\n")
            total_mb = results['total_bytes'] / (1024 * 1024)
            f.write(f"Total data:   {total_mb:.1f} MB\n")
            f.write(f"Elapsed:      {results['elapsed_seconds']:.1f} s\n")
            f.write(f"Throughput:   {results['mb_per_second']:.1f} MB/s\n")
            f.write(f"Actual FPS:   {results['fps_actual']:.1f}\n\n")
            f.write(f"Result: {'PASS' if results['passed'] else 'FAIL'}\n")
            if results['errors']:
                f.write(f"\nErrors ({len(results['errors'])}):\n")
                for err in results['errors']:
                    f.write(f"  - {err}\n")

    def _step_metadata(self, tmp, sn):
        meta = {
            'report_version': REPORT_VERSION,
            'generated_at': datetime.datetime.now().isoformat(),
            'serial_number': sn,
            'lvp_version': self._get_lvp_version(),
            'generator': 'tech_support_report.py',
            'contents': sorted(d.name for d in tmp.iterdir() if d.is_dir()),
        }
        meta.update(self._meta)
        with open(tmp / 'report_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    # -- Helpers -------------------------------------------------------------

    def _write_log_delimiter(self):
        ts = datetime.datetime.now().isoformat()
        delim = LOG_DELIMITER.format(timestamp=ts)
        logger.info(delim)
        logs_dir = _get_lvp_logs_dir()
        if logs_dir:
            try:
                logs = sorted(logs_dir.glob('*.log'), key=lambda p: p.stat().st_mtime)
                if logs:
                    with open(logs[-1], 'a') as f:
                        f.write(delim)
            except OSError:
                pass

    def _get_lvp_version(self):
        app_root = _get_app_root()
        for vf in ['VERSION', 'version.txt']:
            p = app_root / vf
            if p.exists():
                try:
                    return p.read_text().strip()
                except OSError:
                    pass
        main_py = app_root / 'lumaviewpro.py'
        if main_py.exists():
            try:
                with open(main_py, 'r') as f:
                    for line in f:
                        if '__version__' in line or 'VERSION' in line:
                            for part in line.split("'") + line.split('"'):
                                if '.' in part and any(c.isdigit() for c in part):
                                    return part.strip()
                        if line.startswith('class '):
                            break
            except OSError:
                pass
        return 'Unknown'

    def _create_zip(self, tmp, sn, output_dir=None):
        if output_dir is None:
            output_dir = _get_desktop()
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        clean_sn = ''.join(c for c in str(sn) if c.isalnum() or c in '-_') or 'UNKNOWN'
        ts = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        zip_path = output_dir / f"SN{clean_sn}-{ts}.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Include a privacy notice describing what data is collected
            zf.writestr('PRIVACY_NOTICE.txt', (
                'LumaViewPro Tech Support Report\n'
                '================================\n\n'
                'This ZIP contains diagnostic information to help Etaluma\n'
                'troubleshoot your microscope system. It includes:\n\n'
                '  - OS version, CPU model, RAM, disk info\n'
                '  - Connected USB devices and display configuration\n'
                '  - LumaViewPro settings, logs, and firmware versions\n'
                '  - Power/sleep configuration\n\n'
                'Please review the contents before sharing. Remove any\n'
                'files you are not comfortable sending.\n\n'
                'Contact: techsupport@etaluma.com\n'
            ))
            for fp in sorted(tmp.rglob('*')):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(tmp))

        logger.info(f"Report saved: {zip_path}")
        return zip_path


class _Cancelled(Exception):
    pass


# ---------------------------------------------------------------------------
# Kivy GUI Integration
# ---------------------------------------------------------------------------

KV_SNIPPET = """\
# Add inside the microscope settings panel in lumaviewpro.kv:
#
# BoxLayout:
#     size_hint_y: None
#     height: dp(48)
#     padding: dp(8)
#     RoundedButton:
#         id: btn_support_report
#         text: 'Generate Support Report'
#         on_release: app.generate_support_report()
"""

PYTHON_INTEGRATION = '''\
# --- Add these methods to the LumaViewPro App class in lumaviewpro.py ---

def generate_support_report(self):
    """Called when user clicks 'Generate Support Report'."""
    from ui.notification_popup import NotificationPopup

    popup = NotificationPopup(
        title='Tech Support Report',
        message=(
            'This will create a diagnostic report to send to\\n'
            'Etaluma Tech Support.\\n\\n'
            'The stage will be homed and moved during testing.\\n'
            'Please remove any samples from the stage.\\n\\n'
            'This may take a few minutes — please wait.'
        ),
        confirm_text='Generate',
        cancel_text='Cancel',
        on_confirm=self._start_support_report,
    )
    popup.open()

def _start_support_report(self):
    from ui.progress_popup import ProgressPopup
    from modules.tech_support_report import TechSupportReport
    import threading

    self._report_progress = ProgressPopup(
        title='Generating Support Report...', auto_dismiss=False)
    self._report_progress.open()

    def run():
        report = TechSupportReport(scope=self.scope)

        def progress(pct, msg):
            from kivy.clock import Clock
            Clock.schedule_once(
                lambda dt: self._update_report_progress(pct, msg), 0)

        path = report.generate(callback=progress, include_bandwidth_test=False)
        from kivy.clock import Clock
        Clock.schedule_once(lambda dt: self._report_done(path), 0)

    threading.Thread(target=run, daemon=True).start()

def _update_report_progress(self, pct, msg):
    if hasattr(self, '_report_progress') and self._report_progress:
        self._report_progress.progress = pct
        self._report_progress.message = msg

def _report_done(self, zip_path):
    if hasattr(self, '_report_progress') and self._report_progress:
        self._report_progress.dismiss()

    from ui.notification_popup import NotificationPopup
    if zip_path:
        popup = NotificationPopup(
            title='Report Complete',
            message=(
                f'Saved to Desktop:\\n{zip_path.name}\\n\\n'
                f'Please email this file to:\\n'
                f'techsupport@etaluma.com'
            ),
        )
    else:
        popup = NotificationPopup(
            title='Report Failed',
            message=(
                'Could not generate the report.\\n'
                'Check the log file for details and contact\\n'
                'techsupport@etaluma.com directly.'
            ),
        )
    popup.open()
'''


# ---------------------------------------------------------------------------
# PyInstaller Build Notes
# ---------------------------------------------------------------------------

PYINSTALLER_SPEC = """\
# --- Add to scripts/appBuild/build_win_release.ps1 ---
#
# After the main LumaViewPro build, add a second PyInstaller invocation
# to produce a standalone diagnostics executable:
#
#   pyinstaller --onefile --name "EtalumaDiagnostics" `
#       --icon "scripts/appBuild/config/etaluma_icon.ico" `
#       --add-data "data;data" `
#       --add-data "drivers;drivers" `
#       --add-data "modules;modules" `
#       --add-data "tests;tests" `
#       modules/tech_support_report.py
#
# This produces dist/EtalumaDiagnostics.exe which can be sent to customers
# independently of LumaViewPro. It connects to hardware directly and runs
# all the same diagnostics.
#
# Both executables (LumaViewPro.exe and EtalumaDiagnostics.exe) go into
# the final installer package.
"""


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    """Run diagnostics from command line without LumaViewPro."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Etaluma LumaViewPro — Tech Support Diagnostic Report',
    )
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (default: Desktop)')
    parser.add_argument('--bandwidth-test', action='store_true',
                        help='Include camera bandwidth test (~2-5 min)')
    parser.add_argument('--no-firmware', action='store_true',
                        help='Skip firmware communication (no hardware needed)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    logger.info("")
    logger.info("=" * 56)
    logger.info("  Etaluma Diagnostics — Tech Support Report Generator")
    logger.info("=" * 56)
    logger.info("")

    report = TechSupportReport()

    if not args.no_firmware:
        logger.info("Connecting to hardware...")
        report.diag.connect_standalone()
        led_ok = report.diag._led_ok()
        mot_ok = report.diag._motor_ok()
        logger.info(f"  LED board:   {'Connected' if led_ok else 'Not found'}")
        logger.info(f"  Motor board: {'Connected' if mot_ok else 'Not found'}")

        # If neither board found, prompt for power cycle before giving up
        if not led_ok and not mot_ok:
            logger.info("")
            logger.info("  ** No boards detected. **")
            logger.info("  Please try the following:")
            logger.info("    1. Check that the USB cable is connected")
            logger.info("    2. Power-cycle the system (turn off, wait 10 seconds, turn on)")
            logger.info("    3. Wait 30 seconds for the boards to boot")
            try:
                input("  Press Enter to retry (Ctrl-C to skip hardware)...")
            except KeyboardInterrupt:
                logger.info("  Skipping hardware.")
                led_ok = False
                mot_ok = False
            else:
                logger.info("  Retrying...")
                report.diag.connect_standalone()
                led_ok = report.diag._led_ok()
                mot_ok = report.diag._motor_ok()
                logger.info(f"  LED board:   {'Connected' if led_ok else 'Not found'}")
                logger.info(f"  Motor board: {'Connected' if mot_ok else 'Not found'}")
                if not led_ok and not mot_ok:
                    logger.info("")
                    logger.info("  Still no boards found. Generating report without hardware.")
                    logger.info("  Please include this report and contact techsupport@etaluma.com")
                    logger.info("")

        # Boards are owned by report.diag — no need to copy them to report
        if mot_ok:
            logger.info("")
            logger.info("  ** The stage will be homed and moved during testing.  **")
            logger.info("  ** This process may take 5-10 minutes to complete.    **")
            try:
                input("  Press Enter to continue (Ctrl-C to cancel)...")
            except KeyboardInterrupt:
                logger.info("  Cancelled.")
                return 1
    else:
        logger.info("Skipping firmware (--no-firmware)")

    logger.info("")

    def cli_progress(pct, msg):
        filled = int(30 * pct / 100)
        bar = '█' * filled + '░' * (30 - filled)
        # Progress bar uses carriage return — keep as print for CLI display
        print(f"\r  [{bar}] {pct:3d}%  {msg:<50s}", end='', flush=True)

    zip_path = report.generate(
        callback=cli_progress,
        include_bandwidth_test=args.bandwidth_test,
        output_dir=args.output,
    )

    print('\n')  # Newline after progress bar
    if zip_path:
        logger.info(f"  Report saved: {zip_path}")
        logger.info(f"  Please email to: techsupport@etaluma.com")
    else:
        logger.info("  Report generation failed.")
        logger.info("  Contact techsupport@etaluma.com directly.")
    logger.info("")

    return 0 if zip_path else 1


if __name__ == '__main__':
    sys.exit(main())
