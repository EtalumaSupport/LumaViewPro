# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tech Support Report Generator for LumaViewPro.

Collects comprehensive diagnostic information and bundles it into a ZIP file
on the user's Desktop for emailing to support@etaluma.com.

Two modes:
  1. Integrated: Called from the LumaViewPro GUI "Generate Support Report"
     button. Receives a Lumascope instance (or ScopeSession) from the running
     application.
  2. Standalone: ``python tech_support_report.py`` -- connects to hardware
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

import contextlib
import dataclasses
import datetime
import enum
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
from collections.abc import Callable

import platformdirs

from lvp_logger import collect_installed_packages
from modules import recording_frames, settings_init
from modules.exceptions import (
    DiagnosticRefusedError,
    HardwareCommandRefusedError,
    SUPPORT_ADDRESS,
    HomingFailedError,
    SupportReportNotSavedError,
)
from modules.lumascope_api.bring_up import CAUSE_PHRASES
from modules.lumascope_api.diagnostics import (
    LED_COMMANDS_V2,
    NOT_CONNECTED,
    is_board_reply,
)
from modules.path_utils import get_script_root, get_source_root
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_VERSION = '1.0.0'

# Config files to exclude when reading from board filesystem
FIRMWARE_EXCLUDE_FILES = {'main.py', 'boot.py'}
FIRMWARE_CONFIG_EXTENSIONS = {'.json', '.ini'}

RECENT_PROTOCOL_COUNT = 10

RECENT_VIDEO_RECEIPT_COUNT = 20

BACKLASH_FOLDER_PATTERNS = ['backlash', 'Backlash', 'BACKLASH']

LOG_DELIMITER = (
    '\n'
    + '=' * 72
    + '\n=== TECH SUPPORT REPORT GENERATION STARTED -- {timestamp} ===\n'
    + '=' * 72
    + '\n'
)

# Camera bandwidth test defaults
BANDWIDTH_TEST_FRAMES = 5000
BANDWIDTH_TEST_TIMEOUT_S = 300  # 5 min -- generous for slow cameras

# Disk speed test defaults
DISK_SPEED_TEST_MB = 256  # Write this many MB
DISK_SPEED_WARN_MBPS = 100  # Warn below this (video recording will lag)

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
    'GSTAT': 0x01,  # Global status (reset, driver error, UV)
    'IHOLD_IRUN0': 0x30,  # Motor 0 hold/run current
    'IHOLD_IRUN1': 0x50,  # Motor 1 hold/run current
    'CHOPCONF0': 0x6C,  # Motor 0 chopper config
    'CHOPCONF1': 0x7C,  # Motor 1 chopper config
    'DRV_STATUS0': 0x6F,  # Motor 0 driver status (open load, short, OT)
    'DRV_STATUS1': 0x7F,  # Motor 1 driver status
}


# ---------------------------------------------------------------------------
# Path helpers -- mirrors platformdirs conventions
# ---------------------------------------------------------------------------


def _get_app_root():
    """Return the LumaViewPro application root directory."""
    return get_script_root()


def _get_user_documents():
    """Return the user's Documents directory.

    Uses platformdirs to honor localized folder names (e.g. "Dokumente"
    on a German Windows install) and match the resolvers used by
    app_environment.init_environment and path_utils.get_source_root.
    """
    return pathlib.Path(platformdirs.user_documents_dir())


def _get_lvp_data_dir():
    """Return the LVP data/ directory (settings.json, scopes.json, etc.)."""
    data_dir = get_source_root() / 'data'
    return data_dir if data_dir.is_dir() else _get_app_root()


def _get_lvp_logs_dir():
    """Return the LVP logs/ directory, or None."""
    logs_dir = get_source_root() / 'logs'
    return logs_dir if logs_dir.is_dir() else None


def _get_capture_dir():
    """Return the capture output directory (from settings or defaults).

    The configured directory is the canonical live_folder key, resolved
    current.json-first via _resolve_settings_path -- current.json holds the
    absolute path microscope_settings resolved at runtime, while settings.json
    may still carry the unresolved './capture' default.
    """
    try:
        settings_file = settings_init._resolve_settings_path(str(_get_lvp_data_dir().parent))
        settings = settings_init.read_settings_json(settings_file)
        live_folder = settings.get('live_folder', '')
        if live_folder:
            resolved = pathlib.Path(live_folder).resolve()
            if resolved.is_dir():
                return resolved
    except (FileNotFoundError, settings_init.SettingsFileError, OSError):
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
    """Return the Desktop path (fallback: home directory).

    Uses platformdirs to honor localized folder names ("Schreibtisch"
    on German Windows, etc.); same rationale as _get_user_documents.
    """
    desktop = pathlib.Path(platformdirs.user_desktop_dir())
    return desktop if desktop.is_dir() else pathlib.Path.home()


# ---------------------------------------------------------------------------
# Recent Protocols -- reusable from GUI "Recent Protocols" menu
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
        for proto_file in (*search_dir.rglob('*.tsv'), *search_dir.rglob('*.json')):
            real = proto_file.resolve()
            if real in seen:
                continue
            seen.add(real)
            try:
                with open(proto_file) as f:
                    head = f.read(2048)
            except (OSError, UnicodeDecodeError):
                continue
            # LVP protocols are TSV-native: Protocol.to_file() writes a TSV
            # whose first row is the PROTOCOL_FILE_HEADER banner, and a run
            # in progress auto-writes its protocol (as unsaved_protocol.tsv)
            # into the run directory -- so TSV matching is what surfaces a
            # running protocol that was never explicitly saved. JSON is also
            # accepted for legacy / exported protocols.
            is_tsv_protocol = proto_file.suffix == '.tsv' and Protocol.PROTOCOL_FILE_HEADER in head
            is_json_protocol = proto_file.suffix == '.json' and any(
                k in head for k in ('"steps"', '"sequences"', '"scan"', '"protocol"', '"Protocol"')
            )
            if is_tsv_protocol or is_json_protocol:
                stat = proto_file.stat()
                protocol_files.append(
                    {
                        'path': proto_file,
                        'modified': datetime.datetime.fromtimestamp(stat.st_mtime),
                        'name': proto_file.stem,
                        'size': stat.st_size,
                    }
                )

    protocol_files.sort(key=lambda x: x['modified'], reverse=True)
    return protocol_files[:n]


# ---------------------------------------------------------------------------
# System Information
# ---------------------------------------------------------------------------


def _is_manifest_file(name: str) -> bool:
    """Matches every manifest generation and leg by ONE rule: the
    canonical recording_manifest.json, the legacy session_manifest.json,
    and the per-recording <name>_manifest.json the MP4 leg writes into
    the flat Manual folder."""
    return 'manifest' in name and name.endswith('.json')


def find_video_recording_dirs(capture_dir: pathlib.Path, limit: int) -> list:
    """Newest-first video recording folders under the capture tree.

    A folder qualifies by NAME (a protocol video step's own folder) or by
    CONTENT (an engine manifest or frame files). Content-matching keeps
    manual Video_* folders in the census and, deliberately, folders whose
    manifest was lost -- a recording that cannot prove its own delivery
    is exactly the one a support bundle must surface.
    """
    hits = []
    for path in capture_dir.rglob('*'):
        if not path.is_dir():
            continue
        try:
            files = [p.name for p in path.iterdir() if p.is_file()]
        except OSError:
            continue
        if (
            recording_frames.is_video_recording_dir_name(path.name)
            or any(_is_manifest_file(name) for name in files)
            or any(recording_frames.is_video_frame(name) for name in files)
        ):
            hits.append(path)
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[:limit]


def video_recording_inventory(path: pathlib.Path) -> dict:
    """Provable summary of one recording folder, no pixels shipped.

    Frame count plus min/max frame numbers let support see truncation and
    gaps directly (count < max-min+1 means dropped frames); manifests and
    MP4 sizes travel as names/bytes only.
    """
    frame_names = []
    frame_total_bytes = 0
    manifests = []
    mp4s = []
    other_file_count = 0
    for p in sorted(path.iterdir()):
        if not p.is_file():
            continue
        if recording_frames.is_video_frame(p.name):
            frame_names.append(p.name)
            frame_total_bytes += p.stat().st_size
        elif _is_manifest_file(p.name):
            manifests.append(p.name)
        elif p.name.lower().endswith('.mp4'):
            mp4s.append({'name': p.name, 'bytes': p.stat().st_size})
        else:
            other_file_count += 1
    frame_numbers = []
    for name in frame_names:
        try:
            frame_numbers.append(recording_frames.frame_number(name))
        except ValueError:
            pass
    return {
        'folder': str(path),
        'frame_count': len(frame_names),
        'frame_number_min': min(frame_numbers) if frame_numbers else None,
        'frame_number_max': max(frame_numbers) if frame_numbers else None,
        'frame_total_bytes': frame_total_bytes,
        'manifests': manifests,
        'mp4s': mp4s,
        'other_file_count': other_file_count,
    }


def find_execution_record(recording_dir: pathlib.Path, max_up: int = 3):
    """The owning protocol run's execution record, or None.

    The record lives at the run root; a protocol recording folder sits
    below it (typically run/<Color>/<step>_video), so walk up a bounded
    number of levels. Manual recordings have no owning run and return
    None.
    """
    node = recording_dir
    for _ in range(max_up):
        node = node.parent
        candidate = node / ProtocolExecutionRecord.DEFAULT_FILENAME
        if candidate.is_file():
            return candidate
    return None


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
        info['cpu_detail'] = _run(
            [
                'wmic',
                'cpu',
                'get',
                'Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed',
                '/format:list',
            ]
        )
    elif is_mac:
        info['cpu_detail'] = _run(['sysctl', '-n', 'machdep.cpu.brand_string'])
        info['cpu_cores'] = _run(['sysctl', '-n', 'hw.ncpu'])
    else:
        try:
            with open('/proc/cpuinfo') as f:
                info['cpu_detail'] = f.read()[:2000]
        except OSError:
            pass

    # RAM
    if is_win:
        info['ram_detail'] = _run(
            ['wmic', 'memorychip', 'get', 'Capacity,Speed,Manufacturer', '/format:list']
        )
        info['ram_total'] = _run(
            ['wmic', 'computersystem', 'get', 'TotalPhysicalMemory', '/format:list']
        )
    elif is_mac:
        try:
            mem_bytes = int(_run(['sysctl', '-n', 'hw.memsize']))
            info['ram_total_gb'] = f'{mem_bytes / (1024**3):.1f} GB'
        except (ValueError, TypeError):
            pass
    else:
        try:
            with open('/proc/meminfo') as f:
                info['ram_detail'] = f.read()[:1000]
        except OSError:
            pass

    # Disks
    if is_win:
        info['disk_drives'] = _run(
            [
                'wmic',
                'diskdrive',
                'get',
                'Model,Size,InterfaceType,MediaType,Status',
                '/format:list',
            ]
        )
        info['disk_partitions'] = _run(
            [
                'wmic',
                'logicaldisk',
                'get',
                'DeviceID,Size,FreeSpace,FileSystem,VolumeName',
                '/format:list',
            ]
        )
    elif is_mac:
        info['disk_usage'] = _run(['df', '-h'])
        info['disk_drives'] = _run(['diskutil', 'list'], timeout=15)[:3000]
    else:
        info['disk_usage'] = _run(['df', '-h'])

    # Power / sleep configuration
    if is_win:
        # Targeted: just the sleep timeout and USB selective suspend
        info['power_sleep_ac'] = _run(
            [
                'powercfg',
                '/query',
                'SCHEME_CURRENT',
                '238c9fa8-0aad-41ed-83f4-97be242c8f20',  # Sleep subgroup
                '29f6c1db-86da-48c5-9fdb-f2b67b1f44da',
            ],  # Sleep after (AC)
            timeout=5,
        )
        info['power_sleep_dc'] = _run(
            [
                'powercfg',
                '/query',
                'SCHEME_CURRENT',
                '238c9fa8-0aad-41ed-83f4-97be242c8f20',
                '9d7815a6-7ee4-497e-8888-515a05f02364',
            ],  # Sleep after (DC)
            timeout=5,
        )
        info['usb_selective_suspend'] = _run(
            [
                'powercfg',
                '/query',
                'SCHEME_CURRENT',
                '2a737441-1930-4402-8d77-b2bebba308a3',  # USB subgroup
                '48e6b7a6-50f5-4782-a5d4-53bb8f07e226',
            ],  # USB selective suspend
            timeout=5,
        )
        # Also grab the human-readable active power scheme
        info['power_scheme'] = _run(['powercfg', '/getactivescheme'], timeout=5)
    elif is_mac:
        info['power_settings'] = _run(['pmset', '-g'])

    # Screen resolution and DPI scaling (Kivy rendering issues)
    if is_win:
        info['display'] = _run(
            [
                'wmic',
                'path',
                'Win32_VideoController',
                'get',
                'Name,CurrentHorizontalResolution,CurrentVerticalResolution,CurrentRefreshRate',
                '/format:list',
            ]
        )
        # DPI scaling -- reg query is more reliable than wmic here
        info['dpi_scaling'] = _run(
            ['reg', 'query', r'HKCU\Control Panel\Desktop\WindowMetrics', '/v', 'AppliedDPI'],
            timeout=5,
        )
        # Also try the per-monitor DPI awareness setting
        info['dpi_awareness'] = _run(
            ['reg', 'query', r'HKCU\Control Panel\Desktop', '/v', 'LogPixels'], timeout=5
        )
    elif is_mac:
        info['display'] = _run(['system_profiler', 'SPDisplaysDataType'], timeout=10)

    # Python package versions (critical dependencies). Resolved through the
    # stdlib metadata API, never by shelling out to `pip freeze`: in a frozen
    # build sys.executable IS the application, so that subprocess either
    # re-launches LVP or burns its full timeout, and pip need not be
    # importable there at all. The launch banner already resolves the
    # inventory this way, so both records now come from one implementation.
    try:
        all_packages = '\n'.join(
            f'{name}=={version}' for name, version in collect_installed_packages().items()
        )
        info['pip_freeze'] = all_packages
        # Also extract the critical ones for the summary
        critical = [
            'kivy',
            'pypylon',
            'ids-peak',
            'ids_peak',
            'numpy',
            'pyserial',
            'Pillow',
            'pillow',
            'scipy',
            'opencv',
            'cv2',
            'psutil',
            'requests',
            'fastapi',
        ]
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
    except Exception as e:
        # The import failure text is the diagnostic (a bundling gap in a
        # frozen build names its missing module/DLL here); the registry
        # query only says whether the SDK installer ever ran.
        info['pylon_import_error'] = f'{type(e).__name__}: {e}'
        if is_win:
            info['pylon_install'] = _run(
                ['reg', 'query', r'HKLM\SOFTWARE\Basler\pylon', '/ve'], timeout=5
            )
        else:
            info['pylon_install'] = 'pypylon not importable'

    # IDS Peak
    try:
        import ids_peak

        info['ids_peak_version'] = getattr(ids_peak, '__version__', 'imported but no __version__')
    except Exception as e:
        info['ids_peak_import_error'] = f'{type(e).__name__}: {e}'
        if is_win:
            info['ids_peak_install'] = _run(
                ['reg', 'query', r'HKLM\SOFTWARE\IDS\ids peak', '/ve'], timeout=5
            )
        else:
            info['ids_peak_install'] = 'ids_peak not importable'

    # Windows event log -- recent USB/driver errors
    if is_win:
        info['recent_usb_events'] = _run(
            [
                'wevtutil',
                'qe',
                'System',
                '/q:*[System[(EventID=219 or EventID=507 or EventID=112) '
                'and TimeCreated[timediff(@SystemTime) <= 604800000]]]',
                '/f:text',
                '/c:50',
            ],
            timeout=10,
        )

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
        devices.append(
            (
                'PnP_USB_Camera_Ports',
                _run(
                    [
                        'wmic',
                        'path',
                        'Win32_PnPEntity',
                        'where',
                        "PNPClass='USB' or PNPClass='USBDevice' or PNPClass='Camera' or PNPClass='Ports'",
                        'get',
                        'Name,DeviceID,Status,PNPClass',
                        '/format:list',
                    ]
                ),
            )
        )
        devices.append(
            (
                'USB_Hubs',
                _run(
                    ['wmic', 'path', 'Win32_USBHub', 'get', 'Name,DeviceID,Status', '/format:list']
                ),
            )
        )
        devices.append(
            (
                'USB_Controllers',
                _run(
                    [
                        'wmic',
                        'path',
                        'Win32_USBController',
                        'get',
                        'Name,DeviceID,Status',
                        '/format:list',
                    ]
                ),
            )
        )
    elif platform.system() == 'Darwin':
        devices.append(('SPUSBDataType', _run(['system_profiler', 'SPUSBDataType'])))
    else:
        devices.append(('lsusb', _run(['lsusb', '-v'])[:5000]))

    # pyserial port enumeration (always)
    try:
        from serial.tools import list_ports

        ports = []
        for port in list_ports.comports(include_links=True):
            ports.append(
                {
                    'device': port.device,
                    'description': port.description,
                    'hwid': port.hwid,
                    'vid': f'0x{port.vid:04X}' if port.vid else None,
                    'pid': f'0x{port.pid:04X}' if port.pid else None,
                    'serial_number': port.serial_number,
                    'manufacturer': port.manufacturer,
                    'product': port.product,
                    'location': port.location,
                }
            )
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
            [
                'wmic',
                'path',
                'Win32_PnPEntity',
                'get',
                'Name,DeviceID,Status,PNPClass,Manufacturer',
                '/format:csv',
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return r.stdout
    except Exception as e:
        return f'Error: {e}'


# ---------------------------------------------------------------------------
# Raw REPL file transfer (Thonny-style) -- delegated to drivers.raw_repl
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
            result['errors'].append(f'Invalid JSON: {e}')
            return result
    elif isinstance(config_data, dict):
        parsed = config_data
    else:
        result['valid'] = False
        result['errors'].append(f'Unexpected type: {type(config_data).__name__}')
        return result

    result['parsed'] = parsed

    # Required keys
    for key in ['Model', 'Serial Number']:
        if key not in parsed:
            result['warnings'].append(f"Missing expected key: '{key}'")

    sn = parsed.get('Serial Number', '')
    if sn and not isinstance(sn, str):
        result['errors'].append(f'Serial Number should be string, got {type(sn).__name__}')
    elif isinstance(sn, str) and len(sn) < 2:
        result['warnings'].append(f"Serial Number seems too short: '{sn}'")

    # Axis configs
    for axis_name in ['X Axis', 'Y Axis', 'Z Axis', 'Turret']:
        axis = parsed.get(axis_name, {})
        if not isinstance(axis, dict):
            continue
        for field in ['Steps Per mm', 'Steps Per Rev', 'Travel mm', 'Initial Position after home']:
            val = axis.get(field)
            if val is not None:
                if not isinstance(val, (int, float)):
                    result['errors'].append(
                        f'{axis_name}.{field}: expected number, got {type(val).__name__}'
                    )
                elif val < 0 and field != 'Initial Position after home':
                    result['warnings'].append(f'{axis_name}.{field}: negative ({val})')

        travel = axis.get('Travel mm')
        if isinstance(travel, (int, float)):
            if axis_name in ('X Axis', 'Y Axis') and travel > 200:
                result['warnings'].append(f'{axis_name}: Travel {travel}mm seems very large')
            elif axis_name == 'Z Axis' and travel > 50:
                result['warnings'].append(f'{axis_name}: Z Travel {travel}mm seems large')

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
            'Teststand mode is ENABLED -- should not be active on customer units'
        )

    return result


# ---------------------------------------------------------------------------
# Camera Bandwidth Test
# ---------------------------------------------------------------------------


class CameraBandwidthTest:
    """Stress-test USB camera bandwidth.

    Thin wrapper around ``Lumascope.run_camera_bandwidth_test()`` for
    backward compat with existing report-generation call sites. The actual
    test loop lives at the API layer (LAYER-D / LV-23) so the bandwidth
    numbers reflect the production capture path.
    """

    def __init__(self, scope, num_frames=BANDWIDTH_TEST_FRAMES):
        self.scope = scope
        self.num_frames = num_frames

    def run(self, progress_callback=None):
        """Run test. Returns results dict."""
        if self.scope is None:
            return {
                'num_frames_requested': self.num_frames,
                'num_frames_received': 0,
                'num_frames_none': 0,
                'num_frames_error': 0,
                'total_bytes': 0,
                'elapsed_seconds': 0,
                'mb_per_second': 0.0,
                'fps_actual': 0.0,
                'frame_sizes': [],
                'errors': ['No scope available'],
                'passed': False,
            }
        return self.scope.diagnostics.run_camera_bandwidth_test(
            num_frames=self.num_frames,
            timeout_s=BANDWIDTH_TEST_TIMEOUT_S,
            progress_cb=progress_callback,
        )


# ---------------------------------------------------------------------------
# Firmware Diagnostics
# ---------------------------------------------------------------------------


class MotorBoardPresence(enum.Enum):
    """What the report can say about the motor board; the value is what it writes.

    "Not connected" and "this model has none" read the same at the driver
    (both are the null board), and a report that says the first about an
    LS620 sends support after a cable that does not exist.
    """

    CONNECTED = 'Motor board connected'
    MISSING = 'Motor board not connected'
    NOT_ON_THIS_MODEL = 'No motor board on this model'


def _unread_text(unread: dict) -> str:
    """The one statement an ``_unread_*`` entry makes, for a step that writes text."""
    (text,) = unread.values()
    return text


@dataclasses.dataclass(frozen=True)
class _SkippedStep:
    """Why the report's hardware steps did not run, and what that meant for each."""

    reason: str
    consequence: str


class FirmwareDiagnostics:
    """Talks to LED and motor boards to collect diagnostic data.

    All serial I/O is routed through the Lumascope API
    (``send_diagnostic_command`` / ``send_diagnostic_command_multiline``)
    so the API layer owns Rule-13 logging and Rule-14 error visibility.
    Driver objects are NOT held on this class (LAYER-D / LV-32 / LV-40).
    """

    # The command-line report builds its own scope with
    # ``connect_standalone()``; the GUI report is handed the running one.

    def __init__(self, scope=None):
        self._scope = scope
        # Why ``connect_standalone`` could not build a scope, or None.
        self.build_failure: Exception | None = None

    def connect_standalone(self) -> None:
        """Build a scope for the command-line report through the API.

        A scope that cannot be built (a missing or unreadable install file,
        say) leaves ``self._scope`` None and the cause in
        ``build_failure``, so the report can say in its own files why no
        hardware step ran instead of reporting every board as unplugged.
        """
        try:
            from modules.lumascope_api import Lumascope

            self._scope = Lumascope.create_diagnostic()
            self.build_failure = None
        except Exception as e:
            self._scope = None
            self.build_failure = e

    @property
    def scope(self):
        return self._scope

    @property
    def led_board(self):
        """Backward-compat accessor returning a stable per-scope reference
        used by ``_target_str`` identity matching only. New diagnostic
        code MUST go through ``send_diagnostic_command``. Returns the
        post-Wave-7 illumination sub-API namespace (was ``scope.led``
        pre-Wave-7 -- that attribute no longer exists)."""
        return getattr(self._scope, 'illumination', None) if self._scope else None

    @property
    def motor_board(self):
        """Backward-compat accessor returning a stable per-scope reference
        used by ``_target_str`` identity matching only. New diagnostic
        code MUST go through ``send_diagnostic_command``."""
        return getattr(self._scope, 'motion', None) if self._scope else None

    def _led_ok(self) -> bool:
        """True when a real LED board is connected to this scope.

        The probe is the scope's ``led_connected``: the sub-API namespace
        ``scope.illumination`` is non-None even when only a NullLEDBoard is
        installed, so its truthiness says nothing.
        """
        return self._scope is not None and self._scope.led_connected

    def motor_board_presence(self) -> MotorBoardPresence:
        """Whether this scope's motor board is connected, missing, or never fitted.

        The scope's ``motion_expected`` tells a manual scope from one whose
        board did not come up. The command-line report's scope is built
        without a model, so it always expects a board and can only answer
        connected or missing.
        """
        if self._scope is None:
            return MotorBoardPresence.MISSING
        if self._scope.motor_connected:
            return MotorBoardPresence.CONNECTED
        if not self._scope.motion_expected:
            return MotorBoardPresence.NOT_ON_THIS_MODEL
        return MotorBoardPresence.MISSING

    def _unread_motor_board(self) -> dict | None:
        """None when the motor board can be read; else the entry saying why it was not.

        A missing board is an ``error``; a model with none is
        ``not_applicable``, which the report writes as a statement.
        """
        presence = self.motor_board_presence()
        if presence is MotorBoardPresence.CONNECTED:
            return None
        if presence is MotorBoardPresence.NOT_ON_THIS_MODEL:
            return {'not_applicable': presence.value}
        return {'error': presence.value}

    def _unread_led(self, needs_v2: bool) -> dict | None:
        """None when an LED board command can be sent; else the entry saying why not.

        An unconnected board is an ``error``. A board that cannot carry the
        command -- no text command channel (an FX2 scope's LED peripheral),
        or firmware older than the command -- is ``not_applicable``: the
        check was not possible, which is neither a fault nor a pass.
        """
        if not self._led_ok():
            return {'error': 'LED board not connected'}
        command_set = self._scope.diagnostics.get_led_info()['command_set']
        if command_set is None:
            return {'not_applicable': 'Not supported on this LED board (no text command channel)'}
        if needs_v2 and command_set != LED_COMMANDS_V2:
            return {'not_applicable': 'Not supported by this LED firmware (needs v2 or later)'}
        return None

    def _enter_engineering(self):
        """Enter LED engineering mode via the diagnostics sub-API.

        Routes through ``scope.diagnostics.enter_led_engineering_mode``
        so the driver-canonical FACTORY + Y handshake (with end-marker
        detection + post-Y drain) is the single canonical implementation.
        """
        if not self._led_ok():
            return False
        return self._scope.diagnostics.enter_led_engineering_mode(timeout_s=5)

    def _exit_engineering(self):
        """Exit LED engineering mode via the diagnostics sub-API.

        Driver-canonical exit drains and sleeps after Q so the LED
        firmware actually transitions out of eng mode.
        """
        if not self._led_ok():
            return
        self._scope.diagnostics.exit_led_engineering_mode()

    def _cmd(self, target, command, timeout_s=None):
        """Send command and return response string, or error string.

        Args:
            target: 'led' or 'motor' (or, for backward compat, a board
                object -- used by older call sites; routed back to the
                target string by introspection).
            command: Firmware command string.
            timeout_s: Per-call serial timeout in seconds.

        Returns:
            str: Response, or the diagnostics channel's stand-in for one.
        """
        target_str = self._target_str(target)
        if target_str is None:
            return NOT_CONNECTED
        if self._scope is None:
            return NOT_CONNECTED
        return self._scope.diagnostics.send_diagnostic_command(
            target_str, command, timeout_s=timeout_s
        )

    def _read_multiline(self, target, command, timeout_s=60, end_markers=None):
        """Send command and read multi-line response (for SELFTEST etc.)."""
        target_str = self._target_str(target)
        if target_str is None:
            return NOT_CONNECTED
        if self._scope is None:
            return NOT_CONNECTED
        return self._scope.diagnostics.send_diagnostic_command_multiline(
            target_str, command, timeout_s=timeout_s, end_markers=end_markers
        )

    def _target_str(self, target):
        """Resolve a target (string or board object) to 'led' / 'motor'.

        Older call sites passed ``self.led_board`` or ``self.motor_board``
        as the first argument. Map those back to the canonical string.
        """
        if isinstance(target, str):
            return target
        if target is None:
            return None
        # Board object -- match by identity to the API's references.
        # Post-Wave-7 the LED namespace is ``scope.illumination``; the
        # legacy ``scope.led`` attribute was retired. Motor namespace
        # remained ``scope.motion``.
        if self._scope is None:
            return None
        if target is getattr(self._scope, 'illumination', None):
            return 'led'
        if target is getattr(self._scope, 'motion', None):
            return 'motor'
        return None

    # -- High-level collectors --

    def get_led_info(self) -> str | list[str]:
        unread = self._unread_led(needs_v2=False)
        if unread is not None:
            return _unread_text(unread)
        return self._read_multiline(
            self.led_board,
            'INFO',
            timeout_s=5,
            end_markers=['RESET CAUSE', 'POWER-ON', 'HARD', 'WDT', 'CALIBRATION'],
        )

    def get_motor_info(self):
        return self._cmd(self.motor_board, 'INFO')

    def get_motor_fullinfo(self):
        """Fetch motor-board FULLINFO with per-instance cache.

        FULLINFO is a static, multi-line dump of model/serial/firmware/
        homing status -- it doesn't change during a tech-support run.
        Pre-cache fix, the first sim run at 2026-05-03 showed FULLINFO
        firing twice ~2 ms apart during a single report generation:
        once via ``get_serial_number()`` (called for the report
        filename) and once via ``_step_motor_diagnostics()`` (called
        for the report body). Caching avoids the duplicate serial round
        trip and slightly speeds up tech-support runs.
        """
        if not hasattr(self, '_cached_motor_fullinfo'):
            self._cached_motor_fullinfo = self._cmd(self.motor_board, 'FULLINFO')
        return self._cached_motor_fullinfo

    def get_serial_number(self) -> str:
        """Extract serial number from FULLINFO.

        Old firmware returns everything on one line:
          Etaluma Motor Controller Board EL-0923 Firmware: 2023-05-30 Model: LS850 Serial: 12006 X homed: True ...
        New firmware uses multi-line with 'Serial Number = ...'
        """
        fullinfo = self.get_motor_fullinfo()
        if not is_board_reply(fullinfo):
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

    def run_led_selftest(self) -> str | list[str]:
        """Run SELFTEST on LED board (v2.0+). Returns full output."""
        unread = self._unread_led(needs_v2=True)
        if unread is not None:
            return _unread_text(unread)
        try:
            if not self._enter_engineering():
                return 'The LED board did not enter engineering mode; SELFTEST was not run'
            return self._read_multiline(self.led_board, 'SELFTEST', timeout_s=90)
        finally:
            self._exit_engineering()

    def get_led_readings(self) -> str | list[str]:
        unread = self._unread_led(needs_v2=True)
        if unread is not None:
            return _unread_text(unread)
        return self._read_multiline(
            self.led_board,
            'LEDREADS',
            timeout_s=30,
            end_markers=['LED7 LED_K', 'AIN1)', 'ERROR'],
        )

    def get_driver_status_all(self):
        """DRVSTAT for all 4 axes (raw 32-bit register values).

        Returns ``{axis: int | None}``. None means firmware does not
        support DRVSTAT_<axis> on this axis (legacy firmware).
        """
        if not self._scope:
            return dict.fromkeys('XYZT')
        return {ax: self._scope.diagnostics.read_motor_drv_status(ax) for ax in 'XYZT'}

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
        """Read fan tachometer RPM via the diagnostics sub-API.

        Returns int RPM, or None if firmware does not support FANSPEED.
        """
        if not self._scope:
            return None
        return self._scope.diagnostics.read_motor_fan_rpm()

    def get_i2c_scan(self) -> str:
        unread = self._unread_led(needs_v2=True)
        if unread is not None:
            return _unread_text(unread)
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
                logger.warning(f'Could not enter raw REPL on {label} board')
                return None

            files = board.repl_list_files()
            logger.info(f'Files on {label} board: {files}')

            for fname in files:
                if fname in FIRMWARE_EXCLUDE_FILES:
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in FIRMWARE_CONFIG_EXTENSIONS:
                    continue
                content = board.repl_read_file(fname)
                if content is not None:
                    result[fname] = content
                    logger.info(f'  Read {label}/{fname} ({len(content)} bytes)')

        except Exception as e:
            logger.warning(f'Raw REPL file read error ({label}): {e}')
        finally:
            board.exit_raw_repl()
            # Verify firmware restarted after raw REPL exit -- the serial
            # state may be dirty and normal commands (LED on/off) would
            # fail silently without this check.
            try:
                board.verify_firmware_running(timeout=10)
                logger.info(f'{label} board firmware verified after raw REPL')
            except Exception as e:
                logger.warning(f'{label} board firmware not responding after raw REPL: {e}')

        return result if result else None

    # -- New hardware diagnostics --

    def measure_serial_latency(
        self, target: str, command: str = 'INFO', iterations: int = SERIAL_LATENCY_ITERATIONS
    ) -> dict:
        """Send a command N times and measure round-trip latency.

        Only a board's reply is timed: the channel answers at once when no
        board is there or a read times out, and timing that answer would
        report a fast link that never carried a byte.

        Returns dict with min/max/mean/std_dev in milliseconds, plus
        the raw timings list.
        """
        timings = []
        errors = 0
        for _ in range(iterations):
            t0 = time.monotonic()
            resp = self._cmd(target, command)
            t1 = time.monotonic()
            if is_board_reply(resp):
                timings.append((t1 - t0) * 1000)  # ms
            else:
                errors += 1
        if not timings:
            return {'error': f'All {iterations} calls failed', 'errors': errors}
        return {
            'iterations': iterations,
            'errors': errors,
            'min_ms': round(min(timings), 2),
            'max_ms': round(max(timings), 2),
            'mean_ms': round(statistics.mean(timings), 2),
            'std_dev_ms': round(statistics.stdev(timings), 2) if len(timings) > 1 else 0,
            'timings_ms': [round(t, 2) for t in timings],
        }

    def read_tmc5072_registers(self) -> dict:
        """Read key TMC5072 diagnostic registers via raw SPI commands.

        Returns dict per chip (XY, ZT) with register values.
        Uses the firmware's SPI<axis>0x<addr><payload> command.
        """
        unread = self._unread_motor_board()
        if unread is not None:
            return unread
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

    def check_led_leakage(self) -> dict:
        """Read every LED channel's current with the LEDs off.

        Passes only when every channel was read and is under the threshold:
        a channel whose current could not be read is NOT MEASURED, and a
        board that could not carry the check is ``not_applicable``, never a
        pass. Requires engineering mode (enters/exits automatically).
        """
        unread = self._unread_led(needs_v2=True)
        if unread is not None:
            return {**unread, 'passed': False}
        try:
            if not self._enter_engineering():
                return {
                    'error': 'The LED board did not enter engineering mode; no current was read',
                    'passed': False,
                }
            currents = self._scope.diagnostics.read_led_currents_ma()
        finally:
            self._exit_engineering()
        if not currents:
            return {'error': 'The LED board reported no channels to read', 'passed': False}

        channels = {}
        for ch in sorted(currents):
            reading = currents[ch]
            if reading is None:
                status = 'NOT MEASURED'
            elif abs(reading) > LED_LEAKAGE_WARN_MA:
                status = 'WARN'
            else:
                status = 'PASS'
            channels[f'CH{ch}'] = {'i_sens_mA': reading, 'status': status}
        passed = all(c['status'] == 'PASS' for c in channels.values())
        return {'channels': channels, 'passed': passed}

    def verify_fan_tachometer(self) -> dict:
        """Set fan to known duty, wait, read tachometer.

        Informational test only -- many units lack a tachometer wire,
        so RPM=0 is not a fault. Returns ``supported=False`` (with no
        readings) when firmware does not implement FAN: / FANSPEED.
        """
        unread = self._unread_motor_board()
        if unread is not None:
            return {'supported': False, 'tests': [], **unread}

        # Probe first: if the driver rejects FAN:0 (always a safe
        # baseline) the firmware doesn't implement fan duty control,
        # and the rest of this test would just emit firmware errors.
        if not self._scope.diagnostics.set_motor_fan_duty(0):
            return {
                'supported': False,
                'message': (
                    'Firmware does not support fan duty control '
                    '(FAN:<duty> / FANSPEED). Upgrade motor firmware '
                    'to v3.1+ to enable this check.'
                ),
                'tests': [],
            }

        results = {
            'supported': True,
            'tests': [],
            'note': 'Informational only -- many units lack tachometer hardware',
        }

        # Test 1: Set fan to ~50% duty, read RPM
        self._scope.diagnostics.set_motor_fan_duty(50)
        time.sleep(2.0)
        rpm_50 = self._scope.diagnostics.read_motor_fan_rpm()

        has_tach = rpm_50 is not None and rpm_50 > 100
        results['tests'].append(
            {
                'duty_pct': 50,
                'rpm': rpm_50,
                'tachometer_detected': has_tach,
            }
        )

        # Test 2: Fan off, read RPM
        self._scope.diagnostics.set_motor_fan_duty(0)
        time.sleep(3.0)
        rpm_off = self._scope.diagnostics.read_motor_fan_rpm()

        results['tests'].append(
            {
                'duty_pct': 0,
                'rpm': rpm_off,
            }
        )

        results['tachometer_present'] = has_tach

        return results

    def run_homing_test(self) -> dict:
        """Home the axes this scope has and verify positions match expected.

        Returns dict with per-axis results including final position
        and whether homing completed successfully. An axis the scope does
        not have gets no row: a row there would pass a check that was
        never run.
        """
        unread = self._unread_motor_board()
        if unread is not None:
            return unread

        caps = self._scope.capabilities
        if not caps.axes:
            # With no rows the verdict would be a PASS for nothing homed.
            return {'error': 'The motor board is connected but reports no axes to home'}
        results = {'axes': {}, 'passed': True}

        # The homes go through the motion API, never as raw commands: a raw
        # home moves the hardware behind MotionAPI, which then reports axis
        # states and a turret slot that are no longer true -- and the turret
        # home also parks Z first, which a raw THOME does not.
        motion = self._scope.motion

        def _home(axis):
            try:
                motion.home(axis)
            except HomingFailedError as e:
                return f'Error: {e}'
            except HardwareCommandRefusedError as e:
                if e.reason != 'not_connected':
                    raise
                return f'Error: {e}'
            return 'OK'

        def _record(axis, home_response):
            results['axes'][axis] = {
                'home_response': home_response,
                'actual_after': self._cmd(self.motor_board, f'ACTUAL_R{axis}'),
                'target_after': self._cmd(self.motor_board, f'TARGET_R{axis}'),
            }

        # Home Z first (safety -- move Z up before XY)
        if caps.has_focus:
            _record('Z', _home('Z'))

        if caps.has_turret:
            _record('T', _home('T'))

        # Home XY (the firmware's full home, as the raw HOME was)
        if caps.has_xy_stage:
            home_resp = _home('ALL')
            for ax in 'XY':
                _record(ax, home_resp)

        # Check for errors in responses
        for _ax, data in results['axes'].items():
            if 'Error' in str(data.get('home_response', '')):
                results['passed'] = False
                data['status'] = 'FAIL'
            else:
                data['status'] = 'OK'

        return results

    # NOTE: Motor repeatability testing requires optical feedback (e.g. a
    # test target on the stage imaged by the camera) because there are no
    # encoders -- ACTUAL_R just reads the TMC5072 step counter which will
    # always agree with TARGET_R. True mechanical repeatability (backlash,
    # missed steps) must be measured optically. This is planned as a future
    # QC feature with a calibration target.


# ---------------------------------------------------------------------------
# Main Report Generator
# ---------------------------------------------------------------------------


_REPORT_TITLES = {'support report': 'Support Report Saved', 'logs zip': 'Logs Zip Saved'}


@dataclasses.dataclass(frozen=True)
class SupportReportSaved:
    """A support report or logs zip that was saved, and the words that say where.

    Attributes:
        path: The ZIP.
        report: ``'support report'`` or ``'logs zip'``.
    """

    path: pathlib.Path
    report: str

    def __post_init__(self):
        if self.report not in _REPORT_TITLES:
            raise ValueError(
                f'{self.report!r} is not a report; use one of {sorted(_REPORT_TITLES)}'
            )

    @property
    def title(self) -> str:
        return _REPORT_TITLES[self.report]

    @property
    def message(self) -> str:
        """Where the ZIP is and where to send it; the folder is the one it was written to."""
        return (
            f'Saved to {self.path.parent}:\n{self.path.name}\n\n'
            f'Email this file to {SUPPORT_ADDRESS}.'
        )


class TechSupportReport:
    """Generate a comprehensive diagnostic ZIP for Etaluma tech support."""

    def __init__(self, scope=None, session=None):
        # Store scope as primary interface -- avoid extracting raw driver
        # objects at this level.  FirmwareDiagnostics handles board access.
        #
        # A session is what lets the hardware steps hold the scope: the
        # report takes the session's diagnostic claim around them, so a
        # run or a recording cannot start underneath a homing or a fan
        # sweep, and a report started during one skips those steps instead
        # of driving the hardware out from under it.
        self._session = session
        if scope is not None:
            self.scope = scope
        elif session is not None:
            self.scope = session.scope
        else:
            self.scope = None

        # FirmwareDiagnostics routes all serial I/O through the
        # Lumascope API. In integrated mode it inherits the live scope; the
        # command-line report calls ``diag.connect_standalone()`` later.
        self.diag = FirmwareDiagnostics(scope=self.scope)

        self._meta = {}

    def _camera_active(self) -> bool:
        """True if a connected camera is reachable through the API.

        Replaces the old ``self._camera is not None`` check. Goes through
        the scope's diagnostic snapshot rather than reaching for the
        driver handle directly (LAYER-D / LV-32).
        """
        if self.scope is None:
            return False
        try:
            return bool(self.scope.diagnostics.get_camera_diagnostic_info().get('connected', False))
        except Exception:
            return False

    def generate(
        self,
        callback: Callable[[int, str], None] | None = None,
        include_bandwidth_test: bool = False,
        output_dir: str | pathlib.Path | None = None,
    ) -> pathlib.Path:
        """Make the full report and return the ZIP's path.

        Raises:
            SupportReportNotSavedError: no ZIP was saved; chained from the
                failure, whose words it carries. A step that fails inside
                the report is written into the report and does not raise.
        """
        cb = callback or (lambda pct, msg: None)
        try:
            return self._generate(cb, include_bandwidth_test, output_dir)
        except Exception as e:
            raise SupportReportNotSavedError('support report', e) from e

    def _generate(self, cb, include_bw, output_dir):
        cb(0, 'Starting report generation...')
        self._write_log_delimiter()

        with tempfile.TemporaryDirectory(prefix='lvp_report_') as tmp:
            tmp = pathlib.Path(tmp)

            # 1-9. Firmware, board, motion and camera steps  (0-41%)
            sn = self._run_scope_steps(tmp, cb)

            # 11. System info  (48-52%)
            cb(49, 'Collecting system information...')
            self._step_system_info(tmp)

            # 11b. Hardware-free diagnostics  (52%)
            self._run_hardware_free_steps(tmp, cb, 52)

            # 12. USB devices  (52-55%)
            cb(53, 'Scanning USB devices...')
            self._step_usb_devices(tmp)

            # 13. Disk speed test  (55-60%)
            cb(56, 'Testing disk write speed...')
            self._step_disk_speed(tmp)

            # 14. Data folder  (60-63%)
            cb(61, 'Copying data folder...')
            self._step_data_folder(tmp)

            # 15. Logs  (63-66%)
            cb(64, 'Copying log files...')
            self._step_logs(tmp)

            # 16. Backlash results  (66-69%)
            cb(67, 'Collecting backlash test results...')
            self._step_backlash(tmp)

            # 17. Recent protocols  (69-71%)
            cb(70, 'Collecting recent protocols...')
            self._step_protocols(tmp)

            # 17b. Video recording receipts  (71-72%)
            cb(71, 'Collecting video recording receipts...')
            self._step_video_receipts(tmp)

            # 18. Hardware serial tests (pytest)  (72-80%)
            cb(73, 'Running hardware serial tests...')
            self._step_hardware_tests(tmp)

            # 19. Bandwidth test (optional)  (80-94%)
            if include_bw and self._camera_active():
                cb(81, 'Running camera bandwidth test (this takes a while)...')
                self._step_bandwidth(tmp, cb)

            # 20. Metadata + ZIP  (94-100%)
            cb(95, 'Writing metadata...')
            self._step_metadata(tmp, sn)

            cb(97, 'Creating ZIP file...')
            zip_path = self._create_zip(tmp, sn, output_dir)

            cb(100, f'Done -- {zip_path.name}')
            return zip_path

    # -- Steps ---------------------------------------------------------------

    def _step_firmware_info(self, tmp, refusal=None):
        if refusal is not None:
            return self._step_firmware_info_cached(tmp, refusal)
        d = tmp / 'firmware_info'
        d.mkdir()

        led_info = self.diag.get_led_info()
        with open(d / 'led_info.txt', 'w') as f:
            f.write(f'LED Board INFO:\n{led_info}\n')

        if self._no_motor_board_on_this_model():
            motor_info = fan = MotorBoardPresence.NOT_ON_THIS_MODEL.value
            sn = 'UNKNOWN'
            self._write_no_motor_board(d, sn)
        else:
            motor_info = self.diag.get_motor_info()
            fullinfo = self.diag.get_motor_fullinfo()
            sn = self.diag.get_serial_number()
            with open(d / 'motor_info.txt', 'w') as f:
                f.write(f'Motor Board INFO:\n{motor_info}\n\n')
                f.write(f'Motor Board FULLINFO:\n{fullinfo}\n\n')
                f.write(f'Serial Number: {sn}\n')

            positions = self.diag.get_motor_positions_all()
            drvstat = self.diag.get_driver_status_all()
            with open(d / 'motor_status.txt', 'w') as f:
                f.write('Motor Positions:\n')
                for ax, data in positions.items():
                    f.write(f'  {ax}: {json.dumps(data)}\n')
                f.write('\nTMC5072 Driver Status:\n')
                for ax, st in drvstat.items():
                    f.write(f'  {ax}: {st}\n')
            fan = self.diag.get_fan_status()

        i2c = self.diag.get_i2c_scan()
        led_readings = self.diag.get_led_readings()
        with open(d / 'peripherals.txt', 'w') as f:
            f.write(f'Fan: {fan}\n\nI2C Scan: {i2c}\n\n')
            f.write(f'LED Readings (baseline, all off):\n{led_readings}\n')

        self._meta['serial_number'] = sn
        self._meta['led_info'] = str(led_info)
        self._meta['motor_info'] = str(motor_info)
        return sn

    def _step_firmware_info_cached(self, tmp, refusal):
        """Step 1 while another activity holds the scope.

        The board queries go through the raw command channel, which the
        lane refuses to anyone but the holder. What the drivers cached at
        connect -- model, serial number, firmware versions -- still goes in,
        and so do the typed driver-status and fan reads, which take the
        board's own lock and are not refused.
        """
        d = tmp / 'firmware_info'
        d.mkdir()
        led_info = self.scope.diagnostics.get_led_info()
        skipped = (
            f'SKIPPED board queries: {refusal.message}\n'
            'The microscope was in use; these are the values cached at connect.\n'
        )
        with open(d / 'led_info.txt', 'w') as f:
            f.write(f'LED Board (cached at connect):\n{led_info}\n\n{skipped}')
        if self._no_motor_board_on_this_model():
            motor_info = fan = MotorBoardPresence.NOT_ON_THIS_MODEL.value
            sn = 'UNKNOWN'
            self._write_no_motor_board(d, sn)
        else:
            motor_info = self.scope.diagnostics.get_motor_info()
            sn = motor_info.get('serial_number') or 'UNKNOWN'
            with open(d / 'motor_info.txt', 'w') as f:
                f.write(f'Motor Board (cached at connect):\n{motor_info}\n\n')
                f.write(f'Serial Number: {sn}\n\n{skipped}')
            drvstat = self.diag.get_driver_status_all()
            with open(d / 'motor_status.txt', 'w') as f:
                f.write('Motor Positions: not read.\n')
                f.write(skipped)
                f.write('\nTMC5072 Driver Status:\n')
                for ax, st in drvstat.items():
                    f.write(f'  {ax}: {st}\n')
            fan = self.diag.get_fan_status()
        with open(d / 'peripherals.txt', 'w') as f:
            f.write(f'Fan: {fan}\n\n')
            f.write('I2C Scan and LED Readings: not read.\n')
            f.write(skipped)
        self._meta['serial_number'] = sn
        self._meta['led_info'] = str(led_info)
        self._meta['motor_info'] = str(motor_info)
        return sn

    def _step_configbackup(self, tmp):
        """Retrieve config files from BOTH boards via raw REPL."""
        d = tmp / 'firmware_configs'
        d.mkdir()

        # A board that cannot carry the question says so, as the report's
        # other LED files do: an FX2 scope's LED peripheral has no REPL, and
        # a read attempted there fails as if the board were stuck.
        boards = []
        unread_led = self.diag._unread_led(needs_v2=False)
        if unread_led is not None and 'not_applicable' in unread_led:
            (d / 'led_config.txt').write_text(f'{unread_led["not_applicable"]}.\n')
        else:
            boards.append((self.diag.led_board, 'led'))
        if self._no_motor_board_on_this_model():
            (d / 'motor_config.txt').write_text(f'{MotorBoardPresence.NOT_ON_THIS_MODEL.value}.\n')
        else:
            boards.append((self.diag.motor_board, 'motor'))

        for board, label in boards:
            files = self.diag.read_config_files(board, label)
            if files is None:
                with open(d / f'{label}_config_UNAVAILABLE.txt', 'w') as f:
                    f.write(
                        f'Could not read config files from {label} board.\n'
                        f'Board may not be connected or raw REPL entry failed.\n'
                    )
                    if label == 'led':
                        f.write(
                            'Note: LED board raw REPL may require a power cycle '
                            'before config files can be read (SPI state issue).\n'
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
                logger.info(f'  Saved {label}/{safe_name} ({len(content)} bytes)')

                # Validate motorconfig.json
                if safe_name.lower() == 'motorconfig.json':
                    validation = validate_motorconfig(content, f'{label}/{safe_name}')
                    with open(d / f'{label}_motorconfig_validation.txt', 'w') as f:
                        f.write(f'Motorconfig Validation ({label}/{safe_name})\n')
                        f.write(f'Valid: {validation["valid"]}\n\n')
                        if validation['errors']:
                            f.write('ERRORS:\n')
                            for e in validation['errors']:
                                f.write(f'  !! {e}\n')
                        if validation['warnings']:
                            f.write('WARNINGS:\n')
                            for w in validation['warnings']:
                                f.write(f'  -- {w}\n')
                        if not validation['errors'] and not validation['warnings']:
                            f.write('All checks passed.\n')

    def _run_scope_steps(self, tmp, cb):
        """Steps 1-9: every step that talks to the scope. Returns the serial number.

        Every step that sends a board command runs under the session's
        diagnostic claim. When another activity holds the scope the claim is
        refused, and the lanes would refuse those commands too: each such
        step records that it was skipped and why, in the file its result
        would have gone to, so the report says what it did not do rather
        than leaving a gap. What does not need the lanes still goes in --
        the identity the drivers cached at connect, the typed driver-status
        and fan reads, and the camera temperatures.
        """
        with contextlib.ExitStack() as held:
            refusal = None
            skip = None
            not_built = self.diag.build_failure
            if not_built is not None:
                skip = _SkippedStep(
                    f'The scope could not be built: {type(not_built).__name__}: {not_built}',
                    'No board or camera was reached, so this step did not run.',
                )
                self._meta['scope_not_built'] = skip.reason
            # No session means the command-line report, which opens the
            # boards in a process of its own; nothing in that process can
            # contend for the scope, so there is no claim to take.
            elif self._session is not None:
                try:
                    held.enter_context(self._session.diagnostic_claim())
                except DiagnosticRefusedError as e:
                    refusal = e
                    skip = _SkippedStep(
                        e.message,
                        'The microscope was in use, so this step did not drive the hardware.',
                    )
                    logger.info(f'Report: hardware steps skipped -- {e.message}')

            # 1. Firmware info + serial number  (0-5%)
            cb(1, 'Querying firmware...')
            if not_built is not None:
                self._record_skipped(
                    tmp / 'firmware_info', 'firmware_info.txt', 'Firmware Info', skip
                )
                sn = 'UNKNOWN'
            else:
                sn = self._step_firmware_info(tmp, refusal)

            # 2. Config files from both boards via raw REPL  (5-10%)
            cb(6, 'Backing up firmware config files...')
            if skip is None:
                self._step_configbackup(tmp)
            else:
                self._record_skipped(
                    tmp / 'firmware_configs', 'config_backup.txt', 'Config Backup', skip
                )

            # 3. LED selftest  (10-15%)
            cb(11, 'Running LED selftest...')
            if skip is None:
                self._step_firmware_tests(tmp)
            else:
                self._record_skipped(
                    tmp / 'firmware_tests', 'led_selftest.txt', 'LED SELFTEST', skip
                )

            # 4. LED leakage check  (15-18%)
            cb(16, 'Checking LED leakage...')
            if skip is None:
                self._step_led_checks(tmp)
            else:
                self._record_skipped(
                    tmp / 'hardware_checks', 'led_leakage.txt', 'LED Leakage Check', skip
                )

            # 5. TMC5072 register dump  (18-20%)
            cb(19, 'Reading motor driver registers...')
            if skip is None:
                self._step_tmc_registers(tmp)
            else:
                self._record_skipped(
                    tmp / 'hardware_checks', 'tmc5072_registers.txt', 'TMC5072 Registers', skip
                )

            # 6. Fan tachometer verification  (20-23%)
            cb(21, 'Testing fan...')
            if skip is None:
                self._step_fan_test(tmp)
            else:
                self._record_skipped(tmp / 'hardware_checks', 'fan_test.txt', 'Fan Test', skip)

            # 7. Serial latency measurement  (23-27%)
            cb(24, 'Measuring serial latency...')
            if skip is None:
                self._step_serial_latency(tmp)
            else:
                self._record_skipped(
                    tmp / 'hardware_checks', 'serial_latency.txt', 'Serial Latency', skip
                )

            # 8. Homing test  (27-35%)
            cb(28, 'Homing all axes...')
            if skip is None:
                self._step_homing_test(tmp)
            else:
                self._record_skipped(tmp / 'motion_tests', 'homing_test.txt', 'Homing Test', skip)

            # 9. Camera diagnostics (temp)  (38-41%)
            cb(39, 'Checking camera...')
            if not_built is not None:
                self._record_skipped(tmp / 'camera_info', 'camera_info.txt', 'Camera', skip)
            else:
                self._step_camera_diagnostics(tmp)
        return sn

    @staticmethod
    def _record_skipped(directory, filename, title, skip):
        """Write a skipped step's file where its result would have gone."""
        directory.mkdir(exist_ok=True)
        with open(directory / filename, 'w') as f:
            f.write(f'{title}\n' + '=' * 40 + '\n\n')
            f.write(f'SKIPPED: {skip.reason}\n')
            f.write(f'{skip.consequence}\n')

    def _no_motor_board_on_this_model(self) -> bool:
        return self.diag.motor_board_presence() is MotorBoardPresence.NOT_ON_THIS_MODEL

    @staticmethod
    def _write_no_motor_board(directory, sn):
        """The motor files of a scope built without a motor board: one statement each."""
        absent = f'{MotorBoardPresence.NOT_ON_THIS_MODEL.value}.\n'
        with open(directory / 'motor_info.txt', 'w') as f:
            f.write(f'{absent}\nSerial Number: {sn}\n')
        with open(directory / 'motor_status.txt', 'w') as f:
            f.write(absent)

    def _step_firmware_tests(self, tmp):
        d = tmp / 'firmware_tests'
        d.mkdir()

        selftest = self.diag.run_led_selftest()
        with open(d / 'led_selftest.txt', 'w') as f:
            f.write(f'LED SELFTEST:\n\n{selftest}\n')

    def _step_led_checks(self, tmp):
        """Check LED leakage with all LEDs off."""
        d = tmp / 'hardware_checks'
        d.mkdir()

        # LED leakage
        leakage = self.diag.check_led_leakage()
        with open(d / 'led_leakage.txt', 'w') as f:
            f.write('LED Leakage Check (all LEDs off)\n' + '=' * 40 + '\n\n')
            if 'not_applicable' in leakage:
                f.write(f'{leakage["not_applicable"]}.\n')
            elif 'error' in leakage:
                f.write(f'Error: {leakage["error"]}\n')
            else:
                statuses = set()
                for ch, data in leakage['channels'].items():
                    val = data['i_sens_mA']
                    st = data['status']
                    statuses.add(st)
                    if val is not None:
                        f.write(f'  {ch}: {val:7.3f} mA  [{st}]\n')
                    else:
                        f.write(f'  {ch}: not measured  [{st}]\n')
                if leakage['passed']:
                    overall = 'PASS'
                elif 'WARN' in statuses:
                    overall = 'WARN -- leakage detected'
                else:
                    overall = 'INCOMPLETE -- a channel could not be read'
                f.write(f'\nOverall: {overall}\n')
                f.write(f'(Threshold: {LED_LEAKAGE_WARN_MA} mA)\n')

    def _step_tmc_registers(self, tmp):
        """Dump key TMC5072 diagnostic registers."""
        d = tmp / 'hardware_checks'
        d.mkdir(exist_ok=True)

        regs = self.diag.read_tmc5072_registers()
        with open(d / 'tmc5072_registers.txt', 'w') as f:
            f.write('TMC5072 Register Dump\n' + '=' * 40 + '\n\n')
            if 'not_applicable' in regs:
                f.write(f'{regs["not_applicable"]}.\n')
            elif 'error' in regs:
                f.write(f'Error: {regs["error"]}\n')
            else:
                for chip, registers in regs.items():
                    f.write(f'--- {chip} chip ---\n')
                    for reg_name, value in registers.items():
                        f.write(f'  {reg_name:16s}: {value}\n')
                    f.write('\n')
                f.write('Key flags in DRV_STATUS:\n')
                f.write('  Bit 0-1: SG_RESULT (StallGuard)\n')
                f.write('  Bit 24: s2ga (short to GND coil A)\n')
                f.write('  Bit 25: s2gb (short to GND coil B)\n')
                f.write('  Bit 26: ola (open load A -- motor disconnected?)\n')
                f.write('  Bit 27: olb (open load B -- motor disconnected?)\n')
                f.write('  Bit 25: ot (overtemperature shutdown)\n')
                f.write('  Bit 26: otpw (overtemperature pre-warning)\n')
                f.write('  Bit 31: stst (standstill indicator)\n')
        with open(d / 'tmc5072_registers.json', 'w') as f:
            json.dump(regs, f, indent=2, default=str)

    def _step_fan_test(self, tmp):
        """Test fan operation via tachometer (informational -- many units lack tach)."""
        d = tmp / 'hardware_checks'
        d.mkdir(exist_ok=True)

        fan = self.diag.verify_fan_tachometer()
        with open(d / 'fan_test.txt', 'w') as f:
            f.write('Fan Test (informational)\n' + '=' * 40 + '\n')
            f.write('Note: Many units in the field do not have a tachometer\n')
            f.write('wire installed. Zero RPM does not necessarily mean the\n')
            f.write('fan is broken.\n\n')
            if 'not_applicable' in fan:
                f.write(f'{fan["not_applicable"]}.\n')
            elif not fan.get('supported', True):
                msg = fan.get('message') or fan.get('error') or 'Fan diagnostic not available.'
                f.write(f'INCONCLUSIVE: {msg}\n')
            else:
                tach = fan.get('tachometer_present', False)
                f.write(f'Tachometer detected: {"Yes" if tach else "No"}\n\n')
                for t in fan.get('tests', []):
                    rpm = t.get('rpm')
                    rpm_str = 'unknown' if rpm is None else str(rpm)
                    f.write(f'  Duty {t.get("duty_pct", "?")}%:  RPM={rpm_str}\n')

    def _step_serial_latency(self, tmp):
        """Measure serial round-trip latency on both boards."""
        d = tmp / 'hardware_checks'
        d.mkdir(exist_ok=True)

        # Run latency test once per board, write both text and JSON from same data
        results = {
            'LED': self.diag._unread_led(needs_v2=False)
            or self.diag.measure_serial_latency('led', 'INFO')
        }
        results['Motor'] = self.diag._unread_motor_board() or self.diag.measure_serial_latency(
            'motor', 'INFO'
        )

        with open(d / 'serial_latency.txt', 'w') as f:
            f.write('Serial Round-Trip Latency\n' + '=' * 40 + '\n\n')
            for label, latency in results.items():
                f.write(f'--- {label} Board ({SERIAL_LATENCY_ITERATIONS}x INFO) ---\n')
                if 'not_applicable' in latency:
                    f.write(f'  {latency["not_applicable"]}.\n\n')
                elif 'error' in latency:
                    f.write(f'  Error: {latency["error"]}\n\n')
                else:
                    f.write(f'  Min:     {latency["min_ms"]:7.2f} ms\n')
                    f.write(f'  Max:     {latency["max_ms"]:7.2f} ms\n')
                    f.write(f'  Mean:    {latency["mean_ms"]:7.2f} ms\n')
                    f.write(f'  Std dev: {latency["std_dev_ms"]:7.2f} ms\n')
                    f.write(f'  Errors:  {latency["errors"]}\n\n')
                    # Flag suspicious results
                    if latency['max_ms'] > 100:
                        f.write(
                            f'  ** WARNING: max latency {latency["max_ms"]}ms '
                            f'-- possible USB suspend or contention **\n\n'
                        )
                    if latency['std_dev_ms'] > 20:
                        f.write(
                            f'  ** WARNING: high variance (std={latency["std_dev_ms"]}ms) '
                            f'-- unstable USB connection **\n\n'
                        )

        with open(d / 'serial_latency.json', 'w') as f:
            summary = {
                label: {k: v for k, v in lat.items() if k != 'timings_ms'}
                for label, lat in results.items()
            }
            json.dump(summary, f, indent=2, default=str)

    def _step_homing_test(self, tmp):
        """Home all axes and record results."""
        d = tmp / 'motion_tests'
        d.mkdir()

        homing = self.diag.run_homing_test()
        with open(d / 'homing_test.txt', 'w') as f:
            f.write('Homing Test\n' + '=' * 40 + '\n\n')
            if 'not_applicable' in homing:
                f.write(f'{homing["not_applicable"]}.\n')
            elif 'error' in homing:
                f.write(f'Error: {homing["error"]}\n')
            else:
                f.write(f'Overall: {"PASS" if homing["passed"] else "FAIL"}\n\n')
                for ax, data in homing.get('axes', {}).items():
                    f.write(f'  {ax} axis:\n')
                    f.write(f'    Home response: {data.get("home_response")}\n')
                    f.write(f'    Actual after:  {data.get("actual_after")}\n')
                    f.write(f'    Target after:  {data.get("target_after")}\n')
                    f.write(f'    Status:        {data.get("status")}\n\n')
        with open(d / 'homing_test.json', 'w') as f:
            json.dump(homing, f, indent=2, default=str)

    def _step_camera_diagnostics(self, tmp):
        """Read camera sensor temperature and basic info via the API."""
        d = tmp / 'camera_info'
        d.mkdir()

        if not self._camera_active():
            (d / 'no_camera.txt').write_text('No camera available.\n')
            return

        api_info = self.scope.diagnostics.get_camera_diagnostic_info()

        # Flatten temperatures into the top-level info block so the
        # output file format matches the historical layout (one
        # 'Temperature_<name>' entry per sensor) downstream consumers
        # may rely on.
        info: dict = {}
        for key in (
            'model',
            'resolution',
            'pixel_format',
            'gain_db',
            'exposure_ms',
            'max_gain_db',
            'max_exposure_ms',
        ):
            if key in api_info:
                info[key] = api_info[key]
        temperatures = api_info.get('temperatures', {})
        if isinstance(temperatures, str):
            # The snapshot's per-field error string: the read failed.
            info['Temperatures'] = temperatures
        else:
            for name, temp_c in temperatures.items():
                info[f'Temperature_{name}'] = temp_c

        with open(d / 'camera_info.txt', 'w') as f:
            f.write('Camera Information\n' + '=' * 40 + '\n\n')
            f.write(f'Camera model: {api_info.get("model", "?")}\n\n')
            for key, val in info.items():
                label = key.replace('get_', '').replace('_', ' ').title()
                f.write(f'  {label}: {val}\n')
                # Flag hot cameras
                if 'temperature' in key.lower() and isinstance(val, (int, float)):
                    if val > 60:
                        f.write(
                            f'    ** WARNING: sensor temperature {val}degC '
                            f'is high -- check cooling/ventilation **\n'
                        )
                    elif val > 45:
                        f.write(
                            f'    ** Note: sensor at {val}degC '
                            f'(elevated, may affect image noise) **\n'
                        )
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
            except Exception as e:
                logger.debug(
                    '[TSR] disk_speed_test: disk_usage(%s) failed; '
                    'disk space fields omitted from report: %s: %s',
                    test_dir,
                    type(e).__name__,
                    e,
                )

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
            f.write('Disk Speed Test\n' + '=' * 40 + '\n\n')
            f.write(f'Test directory: {results.get("test_directory")}\n')
            f.write(f'Test size:      {DISK_SPEED_TEST_MB} MB\n\n')
            if 'error' in results:
                f.write(f'Error: {results["error"]}\n')
            else:
                f.write(
                    f'Write speed: {results.get("write_mbps", "?")} MB/s '
                    f'({results.get("write_elapsed_s", "?")}s)\n'
                )
                f.write(
                    f'Read speed:  {results.get("read_mbps", "?")} MB/s '
                    f'({results.get("read_elapsed_s", "?")}s)\n\n'
                )
                if 'disk_free_gb' in results:
                    f.write(f'Disk total:  {results["disk_total_gb"]} GB\n')
                    f.write(f'Disk free:   {results["disk_free_gb"]} GB\n')
                    f.write(f'Disk used:   {results["disk_used_pct"]}%\n\n')
                f.write(f'Result: {"PASS" if results.get("passed") else "FAIL"}\n')
                if not results.get('passed'):
                    f.write(
                        f'** Write speed below {DISK_SPEED_WARN_MBPS} MB/s -- '
                        f'video recording may drop frames **\n'
                    )
        with open(d / 'disk_speed.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def _step_runtime_census(self, tmp):
        # A DLL-load failure is invisible after the fact unless the bundle
        # records which copy of each runtime DLL the process actually
        # loaded (an app-local file beside the exe shadows System32) and
        # what runtime files ship beside the exe. Previously this census
        # ran only when the IDS preload failed, so a working-but-degraded
        # process left no record at all.
        #
        # Best-effort by design: this section exists to diagnose loader
        # problems, so it must never be the reason a support bundle fails
        # -- generate()'s catch-all would discard the entire bundle over a
        # diagnostics section. A failure is written INTO the artifact,
        # where support reads it, instead of raised.
        d = tmp / 'runtime_census'
        d.mkdir()
        lines = []
        try:
            from modules.app_environment import loaded_module_census

            lines.append('Process-resident camera-stack / C-runtime modules')
            lines.append('(path shows WHICH copy won the loader search):')
            resident = loaded_module_census()
            lines.extend(f'  {p}' for p in resident)
            if not resident:
                lines.append('  (none matched, or not a Windows host)')
        except Exception as e:
            lines.append(f'  loaded-module census failed: {type(e).__name__}: {e}')
        try:
            exe_dir = pathlib.Path(sys.executable).resolve().parent
            lines.append('')
            lines.append(f'C-runtime files on disk under {exe_dir}:')
            from modules.app_environment import crt_dll_pattern

            crt_pattern = crt_dll_pattern()
            found_any = False
            for p in sorted(exe_dir.rglob('*.dll')):
                if crt_pattern.match(p.name):
                    lines.append(f'  {p.relative_to(exe_dir)}  ({p.stat().st_size} bytes)')
                    found_any = True
            if not found_any:
                lines.append('  (none)')
        except Exception as e:
            lines.append(f'  on-disk CRT listing failed: {type(e).__name__}: {e}')
        with open(d / 'runtime_census.txt', 'w') as f:
            f.write('\n'.join(lines) + '\n')

    def _step_system_info(self, tmp):
        d = tmp / 'system_info'
        d.mkdir()

        info = _collect_system_info()
        with open(d / 'system_info.json', 'w') as f:
            json.dump(info, f, indent=2, default=str)

        with open(d / 'summary.txt', 'w') as f:
            f.write(f'OS:      {info.get("platform")}\n')
            f.write(f'Python:  {info.get("python_version")}\n')
            f.write(f'CPU:     {info.get("cpu")}\n')
            if 'cpu_detail' in info:
                f.write(f'         {info["cpu_detail"][:200]}\n')
            for k in ['ram_total_gb', 'ram_total']:
                if k in info:
                    f.write(f'RAM:     {info[k]}\n')
                    break
            f.write(f'\nPower scheme: {info.get("power_scheme", "N/A")}\n')
            if 'usb_selective_suspend' in info:
                f.write(f'USB selective suspend:\n{info["usb_selective_suspend"]}\n')

            # Display / DPI
            if 'display' in info:
                f.write(f'\nDisplay:\n{info["display"][:500]}\n')
            if 'dpi_scaling' in info:
                f.write(f'DPI scaling: {info["dpi_scaling"]}\n')

            # Camera SDKs
            if 'pylon_version' in info:
                f.write(f'\nBasler Pylon SDK: {info["pylon_version"]}\n')
            elif 'pylon_install' in info:
                f.write(f'\nBasler Pylon: {info["pylon_install"]}\n')
            if 'ids_peak_version' in info:
                f.write(f'IDS Peak SDK: {info["ids_peak_version"]}\n')
            elif 'ids_peak_install' in info:
                f.write(f'IDS Peak: {info["ids_peak_install"]}\n')

            # Critical Python packages
            critical = info.get('critical_packages', [])
            if critical:
                f.write('\nCritical Python packages:\n')
                for pkg in critical:
                    f.write(f'  {pkg}\n')

            # Recent USB events
            usb_events = info.get('recent_usb_events', '')
            if usb_events and 'Error' not in usb_events[:20]:
                f.write('\nRecent USB/driver events (last 7 days):\n')
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
            shutil.copytree(
                data_dir,
                dest,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
            )
        except Exception as e:
            dest.mkdir(exist_ok=True)
            (dest / 'ERROR.txt').write_text(f'Copy failed: {e}\n')

    def _step_logs(self, tmp):
        logs_dir = _get_lvp_logs_dir()
        if not logs_dir or not logs_dir.is_dir():
            return
        dest = tmp / 'logs'
        try:
            shutil.copytree(logs_dir, dest, dirs_exist_ok=True)
        except Exception as e:
            dest.mkdir(exist_ok=True)
            (dest / 'ERROR.txt').write_text(f'Copy failed: {e}\n')

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
                'No backlash test folders found in capture directory.\n'
            )

    def _step_protocols(self, tmp):
        d = tmp / 'recent_protocols'
        d.mkdir()

        protocols = get_recent_protocols(RECENT_PROTOCOL_COUNT)
        if not protocols:
            (d / 'none_found.txt').write_text('No protocol files found.\n')
            return

        # Same-named protocols are the normal case (every run folder emits
        # protocol_record.tsv / protocol_post_record.tsv), so copying by bare
        # basename silently overwrote all but one. One numbered bundle name is
        # derived per entry and shared by the copied file, the index entry,
        # and the metadata mirror, so the three cannot disagree.
        for i, p in enumerate(protocols, 1):
            p['bundle_name'] = f'{i:02d}_{p["name"]}{p["path"].suffix}'

        with open(d / '_index.txt', 'w') as f:
            f.write(f'Most Recent {len(protocols)} Protocols\n{"=" * 40}\n\n')
            for i, p in enumerate(protocols, 1):
                f.write(f'{i:2d}. {p["bundle_name"]}\n')
                f.write(f'    Modified: {p["modified"]}\n')
                f.write(f'    Path:     {p["path"]}\n')
                f.write(f'    Size:     {p["size"]} bytes\n\n')

        for p in protocols:
            try:
                shutil.copy2(p['path'], d / p['bundle_name'])
            except Exception as e:
                logger.warning(f'Could not copy protocol {p["path"]}: {e}')

        self._meta['recent_protocols'] = [
            {'name': pathlib.Path(p['bundle_name']).stem, 'modified': p['modified'].isoformat()}
            for p in protocols
        ]

    def _step_video_receipts(self, tmp):
        """Video receipts: per recent recording, the manifests, a frames
        inventory, and the owning run's execution record -- enough for a
        bundle to prove delivered-vs-configured without shipping frames."""
        capture_dir = _get_capture_dir()
        if not capture_dir or not capture_dir.is_dir():
            return
        d = tmp / 'video_receipts'
        d.mkdir()
        recordings = find_video_recording_dirs(capture_dir, RECENT_VIDEO_RECEIPT_COUNT)
        if not recordings:
            (d / 'none_found.txt').write_text(
                'No video recording folders found in capture directory.\n'
            )
            return
        meta_rows = []
        index_lines = [f'Most Recent {len(recordings)} Video Recordings\n{"=" * 40}\n']
        for i, rec in enumerate(recordings, 1):
            bundle = d / f'{i:02d}_{rec.name}'
            try:
                bundle.mkdir()
                inventory = video_recording_inventory(rec)
                (bundle / 'inventory.json').write_text(json.dumps(inventory, indent=2))
                for name in inventory['manifests']:
                    shutil.copy2(rec / name, bundle / name)
                record = find_execution_record(rec)
                if record is not None:
                    shutil.copy2(record, bundle / record.name)
                index_lines.append(
                    f'{i:2d}. {rec.name}\n'
                    f'    Path:      {rec}\n'
                    f'    Frames:    {inventory["frame_count"]}'
                    f' (numbers {inventory["frame_number_min"]}..{inventory["frame_number_max"]},'
                    f' {inventory["frame_total_bytes"]} bytes)\n'
                    f'    Manifests: {", ".join(inventory["manifests"]) or "NONE"}\n'
                    f'    MP4s:      {len(inventory["mp4s"])}\n'
                    f'    Execution record: {record.name if record else "none"}\n'
                )
                meta_rows.append(
                    {
                        'folder': rec.name,
                        'frame_count': inventory['frame_count'],
                        'manifest_count': len(inventory['manifests']),
                        'has_execution_record': record is not None,
                    }
                )
            except Exception as e:
                (d / f'{i:02d}_{rec.name}_ERROR.txt').write_text(f'Receipt failed: {e}\n')
        (d / '_index.txt').write_text('\n'.join(index_lines))
        self._meta['video_receipts'] = meta_rows

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
            (d / 'skipped.txt').write_text('Tests directory not found.\n')
            return

        test_file = tests_dir / 'test_hardware_serial.py'
        if not test_file.exists():
            (d / 'skipped.txt').write_text('test_hardware_serial.py not found.\n')
            return

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'pytest',
                    str(test_file),
                    '--run-hardware',
                    '-v',
                    '--tb=short',
                    '-q',
                ],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(app_root),
            )
            with open(d / 'test_hardware_serial.txt', 'w') as f:
                f.write('test_hardware_serial.py (--run-hardware)\n')
                f.write(f'Return code: {result.returncode}\n\n')
                f.write(result.stdout)
                if result.stderr:
                    f.write(f'\nSTDERR:\n{result.stderr}')
        except subprocess.TimeoutExpired:
            (d / 'test_hardware_serial.txt').write_text('TIMED OUT after 180s\n')
        except Exception as e:
            (d / 'test_hardware_serial.txt').write_text(f'Error: {e}\n')

    def _step_bandwidth(self, tmp, cb):
        d = tmp / 'bandwidth_test'
        d.mkdir()

        def bw_cb(pct, msg):
            cb(81 + int(pct * 0.13), f'Bandwidth: {msg}')

        bw = CameraBandwidthTest(self.scope)
        results = bw.run(progress_callback=bw_cb)

        with open(d / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        with open(d / 'summary.txt', 'w') as f:
            f.write('Camera Bandwidth Test\n' + '=' * 40 + '\n\n')
            f.write(f'Resolution:      {results.get("resolution", "?")}\n')
            f.write(f'Pixel format:    {results.get("pixel_format", "?")}\n')
            f.write(f'Configured FPS:  {results.get("configured_fps", "?")}\n\n')
            f.write(f'Frames requested: {results["num_frames_requested"]}\n')
            f.write(f'Frames received:  {results["num_frames_received"]}\n')
            f.write(f'Frames None:      {results["num_frames_none"]}\n')
            f.write(f'Frames errored:   {results["num_frames_error"]}\n\n')
            total_mb = results['total_bytes'] / (1024 * 1024)
            f.write(f'Total data:   {total_mb:.1f} MB\n')
            f.write(f'Elapsed:      {results["elapsed_seconds"]:.1f} s\n')
            f.write(f'Throughput:   {results["mb_per_second"]:.1f} MB/s\n')
            f.write(f'Actual FPS:   {results["fps_actual"]:.1f}\n\n')
            f.write(f'Result: {"PASS" if results["passed"] else "FAIL"}\n')
            if results['errors']:
                f.write(f'\nErrors ({len(results["errors"])}):\n')
                for err in results['errors']:
                    f.write(f'  - {err}\n')

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
        # The logging call IS the write to the main log: the root logger
        # owns the lumaviewpro.log file handler and this module's logger
        # propagates to it. A direct file append here was a second
        # writer that landed in whichever log file happened to have the
        # newest mtime -- usually the camera.log firehose, never
        # reliably the main log the marker is meant for.
        ts = datetime.datetime.now().isoformat()
        logger.info(LOG_DELIMITER.format(timestamp=ts))

    def _get_lvp_version(self):
        from modules.path_utils import read_version

        version, build_timestamp = read_version()
        if version:
            return f'{version} ({build_timestamp})' if build_timestamp else version
        return 'Unknown'

    def _run_hardware_free_steps(self, tmp, cb, pct):
        """Diagnostics that touch no hardware -- run for EVERY bundle shape.

        The line that matters is hardware contact, not report size. A
        logs-only capture exists so support can ask for files without
        driving the stage, the fan or the camera; these steps drive none
        of them, so withholding them buys nothing and costs the reader
        the record. This member is the single home for that class: add a
        hardware-free step HERE and both bundle shapes get it, rather
        than to one entry point and not the other.

        Two neighbours deliberately do NOT qualify. `_step_system_info`
        duplicates the launch banner already present in every log.
        `_step_usb_devices` issues wmic calls that are timeouts rather
        than data on current Windows.

        Each step is guarded on its own: a raise out of either bundle path
        means no archive at all, so an unguarded diagnostic here could cost
        the user the very bundle it was added to enrich. A failure is recorded INTO the artifact,
        where support reads it.
        """
        cb(pct, 'Recording runtime census...')
        for step, name in (
            (self._step_runtime_census, 'runtime_census'),
            (self._step_bring_up, 'bring_up'),
            (self._step_plugins, 'plugins'),
        ):
            try:
                step(tmp)
            except Exception as e:
                try:
                    (tmp / f'{name}_ERROR.txt').write_text(f'{name} failed: {e}\n')
                except OSError:
                    pass

    def _step_bring_up(self, tmp):
        """What bring-up found: each part and why it is not up, the substitutions, the settings set aside.

        The session's record when there is one; the command-line report's
        diagnostic scope's otherwise, which holds the two boards and no
        camera and expects every board. No scope, no record: the file says
        why there is none.
        """
        if self._session is not None:
            record = self._session.bring_up_record()
        elif self.diag.scope is not None:
            record = self.diag.scope.bring_up_record()
        else:
            failure = self.diag.build_failure
            why = (
                f'the scope could not be built: {type(failure).__name__}: {failure}'
                if failure is not None
                else 'the report ran without connecting to the scope'
            )
            body = {'record': None, 'why': why}
            (tmp / 'bring_up.json').write_text(json.dumps(body, indent=2))
            return
        body = dataclasses.asdict(record)
        for part in body['parts']:
            part['cause_words'] = CAUSE_PHRASES.get(part['cause']) if part['cause'] else None
        (tmp / 'bring_up.json').write_text(json.dumps(body, indent=2, default=str))

    def _step_plugins(self, tmp):
        """The loaded plugins, the ones that did not load and why, and their runtime errors."""
        health = self._session.plugin_health() if self._session is not None else None
        if health is None:
            body = {'plugins': None, 'why': 'no plugin registry on this host'}
        else:
            body = dataclasses.asdict(health)
        (tmp / 'plugins.json').write_text(json.dumps(body, indent=2, default=str))

    def generate_logs_only(
        self,
        callback: Callable[[int, str], None] | None = None,
        output_dir: str | pathlib.Path | None = None,
    ) -> pathlib.Path:
        """Quick zip of logs + data + recent protocols + video receipts.
        No hardware tests.

        For sending diagnostic files to support when the issue is log-only
        (e.g. post-incident log review) and running the full report would
        exercise hardware needlessly. Video receipts ride along because
        they are small (manifests + inventories, no pixel data) and a
        video complaint usually arrives through this quick bundle, not
        the full report. Returns the ZIP's path.

        Raises:
            SupportReportNotSavedError: no ZIP was saved; chained from the
                failure, whose words it carries.
        """
        cb = callback or (lambda pct, msg: None)
        try:
            with tempfile.TemporaryDirectory(prefix='lvp_logs_') as tmp:
                tmp = pathlib.Path(tmp)

                cb(5, 'Copying log files...')
                self._step_logs(tmp)

                cb(35, 'Copying data folder...')
                self._step_data_folder(tmp)

                cb(70, 'Copying recent protocols...')
                self._step_protocols(tmp)

                cb(78, 'Collecting video recording receipts...')
                self._step_video_receipts(tmp)
                self._run_hardware_free_steps(tmp, cb, 80)

                cb(85, 'Writing metadata...')
                # SN lookup chain:
                # 1. motorconfig cache (populated at boot; no wire I/O).
                # 2. FULLINFO via diag.get_serial_number() (one wire round
                #    trip on first call per session; cached after). Same
                #    path the full TSR uses, so an SN reachable via
                #    FULLINFO surfaces here too.
                # 3. 'logs' fallback only when both are unavailable
                #    (standalone runs, motor not connected).
                sn_tag = None
                try:
                    mb = self.diag.motor_board
                    # A scope with no motor board has no motor configuration.
                    motorconfig = getattr(mb, 'motorconfig', None) if mb is not None else None
                    if motorconfig is not None:
                        sn = motorconfig.serial_number()
                        if sn and sn != 'Unknown':
                            sn_tag = sn
                except Exception:
                    # Log + continue. Silent swallow here masked the
                    # 2026-05-22 SNlogs regression for two days (TypeError
                    # from timeout=/timeout_s= kwarg drift propagated up
                    # via mb.motorconfig.serial_number's wire I/O caller
                    # and was lost). logger.exception preserves the
                    # traceback in the main log so future debuggers see
                    # WHY the SN couldn't be read instead of just the
                    # SNlogs- filename symptom.
                    logger.exception(
                        '[Report   ] SN lookup via motorconfig failed; '
                        'falling through to FULLINFO path'
                    )
                if not sn_tag:
                    try:
                        sn = self.diag.get_serial_number()
                        if sn and sn != 'UNKNOWN':
                            sn_tag = sn
                    except Exception:
                        # Same reasoning as above. If BOTH paths fail
                        # the bundle still ships under SNlogs-, but
                        # the cause is now visible in the log.
                        logger.exception(
                            '[Report   ] SN lookup via FULLINFO failed; '
                            "falling through to 'logs' last-resort"
                        )
                if not sn_tag:
                    sn_tag = 'logs'
                self._meta['report_type'] = 'logs_only'
                self._meta['report_generated_at'] = datetime.datetime.now().isoformat(
                    timespec='seconds'
                )
                (tmp / 'metadata.json').write_text(json.dumps(self._meta, indent=2))

                cb(95, 'Creating ZIP file...')
                zip_path = self._create_zip(
                    tmp,
                    sn_tag,
                    output_dir,
                    report_type='logs_only',
                )
                cb(100, f'Done -- {zip_path.name}')
                return zip_path
        except Exception as e:
            raise SupportReportNotSavedError('logs zip', e) from e

    def _create_zip(self, tmp, sn, output_dir=None, report_type='tsr'):
        if output_dir is None:
            output_dir = _get_desktop()
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        clean_sn = ''.join(c for c in str(sn) if c.isalnum() or c in '-_') or 'UNKNOWN'
        ts = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        # Full tech-support reports carry a "TSR" token so support
        # engineers can visually distinguish them from logs-only user
        # dumps (`SNlogs-<ts>.zip`) that ship with manual error reports.
        # Logs-only bundles keep the plain shape -- they ARE the
        # SNlogs/SN<sn>-<ts>.zip dumps.
        token = '-TSR-' if report_type == 'tsr' else '-'
        zip_path = output_dir / f'SN{clean_sn}{token}{ts}.zip'

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Include a privacy notice describing what data is collected
            zf.writestr(
                'PRIVACY_NOTICE.txt',
                (
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
                    f'Contact: {SUPPORT_ADDRESS}\n'
                ),
            )
            for fp in sorted(tmp.rglob('*')):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(tmp))

        logger.info(f'Report saved: {zip_path}')
        return zip_path


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Run diagnostics from command line without LumaViewPro."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Etaluma LumaViewPro -- Tech Support Diagnostic Report',
        epilog=(
            'The command-line report does not know the scope model, so it cannot tell '
            'a scope built without a motor board (LS560, LS620) from one whose board '
            'is not connected: both read "Motor board not connected". The report '
            'generated from LumaViewPro tells them apart.'
        ),
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None, help='Output directory (default: Desktop)'
    )
    parser.add_argument(
        '--bandwidth-test', action='store_true', help='Include camera bandwidth test (~2-5 min)'
    )
    parser.add_argument(
        '--no-firmware',
        action='store_true',
        help='Skip firmware communication (no hardware needed)',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    logger.info('')
    logger.info('=' * 56)
    logger.info('  Etaluma Diagnostics -- Tech Support Report Generator')
    logger.info('=' * 56)
    logger.info('')

    report = TechSupportReport()

    def connect():
        """Build the scope and say what came up: (LED ok, motor ok, built)."""
        report.diag.connect_standalone()
        not_built = report.diag.build_failure
        if not_built is not None:
            # Not a cable or power fault: power-cycle advice would send the
            # user after the wrong thing, and a retry fails the same way.
            logger.error(f'  The scope could not be built: {type(not_built).__name__}: {not_built}')
            logger.error('  The report will say so; no hardware test will run.')
            return False, False, False
        led_ok = report.diag._led_ok()
        mot_ok = report.diag.motor_board_presence() is MotorBoardPresence.CONNECTED
        logger.info(f'  LED board:   {"Connected" if led_ok else "Not found"}')
        logger.info(f'  Motor board: {"Connected" if mot_ok else "Not found"}')
        return led_ok, mot_ok, True

    if not args.no_firmware:
        logger.info('Connecting to hardware...')
        led_ok, mot_ok, built = connect()

        # If neither board found, prompt for power cycle before giving up
        if built and not led_ok and not mot_ok:
            logger.info('')
            logger.info('  ** No boards detected. **')
            logger.info('  Please try the following:')
            logger.info('    1. Check that the USB cable is connected')
            logger.info('    2. Power-cycle the system (turn off, wait 10 seconds, turn on)')
            logger.info('    3. Wait 30 seconds for the boards to boot')
            try:
                input('  Press Enter to retry (Ctrl-C to skip hardware)...')
            except KeyboardInterrupt:
                logger.info('  Skipping hardware.')
                led_ok = False
                mot_ok = False
            else:
                logger.info('  Retrying...')
                led_ok, mot_ok, built = connect()
                if built and not led_ok and not mot_ok:
                    logger.info('')
                    logger.info('  Still no boards found. Generating report without hardware.')
                    logger.info(f'  Please include this report and contact {SUPPORT_ADDRESS}')
                    logger.info('')

        # Boards are owned by report.diag -- no need to copy them to report
        if mot_ok:
            logger.info('')
            logger.info('  ** The stage will be homed and moved during testing.  **')
            logger.info('  ** This process may take 5-10 minutes to complete.    **')
            try:
                input('  Press Enter to continue (Ctrl-C to cancel)...')
            except KeyboardInterrupt:
                logger.info('  Cancelled.')
                return 1
    else:
        logger.info('Skipping firmware (--no-firmware)')

    logger.info('')

    def cli_progress(pct, msg):
        filled = int(30 * pct / 100)
        bar = '#' * filled + '-' * (30 - filled)
        # Progress bar uses carriage return -- keep as print for CLI display
        print(f'\r  [{bar}] {pct:3d}%  {msg:<50s}', end='', flush=True)

    try:
        zip_path = report.generate(
            callback=cli_progress,
            include_bandwidth_test=args.bandwidth_test,
            output_dir=args.output,
        )
    except SupportReportNotSavedError as e:
        print('\n')  # Newline after progress bar
        logger.error(f'  {e}', exc_info=e)
        logger.info('')
        return 1

    print('\n')  # Newline after progress bar
    logger.info(f'  {SupportReportSaved(zip_path, "support report").message}')
    logger.info('')

    return 0


if __name__ == '__main__':
    sys.exit(main())
