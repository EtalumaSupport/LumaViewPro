# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Application environment initialization -- paths, version, platform detection."""

import logging
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

_logger = logging.getLogger('LVP.app_environment')


@dataclass
class AppEnvironment:
    """Immutable snapshot of application environment determined at startup."""

    script_path: str
    source_path: str
    version: str
    build_timestamp: str
    windows_machine: bool
    num_cores: int
    lvp_installed: bool


def init_environment(main_file: str) -> AppEnvironment:
    """Determine paths, version, and platform. Called once at startup.

    Args:
        main_file: The ``__file__`` of the main script (lumaviewpro.py).

    Returns an AppEnvironment with all resolved values.
    """
    # Determine script location from the main entry point
    abspath = os.path.abspath(main_file)
    basename = os.path.basename(main_file)
    script_path = abspath[: -len(basename)]

    _logger.info(f'Script Location: {script_path}')

    # Recomputed independently of lvp_logger's identical os.name check by
    # design: this is a constant, not divergent state, and lvp_logger is the
    # lowest-level module (imported before this runs), so sharing one source
    # buys nothing and only adds an early-startup import coupling.
    windows_machine = os.name == 'nt'

    # Read version and build timestamp via shared reader
    from modules.path_utils import read_version

    version, build_timestamp = read_version(pathlib.Path(script_path))

    # Get git commit hash for build identification (dev mode only)
    if not build_timestamp:
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=script_path,
            )
            if result.returncode == 0:
                build_timestamp = result.stdout.strip()
        except Exception as e:
            _logger.debug(f'Failed to get git hash: {e}')

    # Check if running as installed application
    lvp_installed = False
    try:
        with open(os.path.join(script_path, 'marker.lvpinstalled')):
            lvp_installed = True
    except Exception:
        pass

    # Determine source_path (data directory)
    if windows_machine and lvp_installed:
        _logger.info('Machine-Type - WINDOWS')
        import platformdirs

        documents_folder = platformdirs.user_documents_dir()
        # Use base version (without hash) for folder name
        # version is already path-safe (no timestamp, no parens)
        lvp_appdata = os.path.join(documents_folder, f'LumaViewPro {version}')

        if not os.path.exists(lvp_appdata):
            os.mkdir(lvp_appdata)

        source_path = lvp_appdata
        _logger.info(f'Data Location: {source_path}')

        if not os.path.exists(os.path.join(lvp_appdata, 'data')):
            shutil.copytree(os.path.join(script_path, 'data'), os.path.join(lvp_appdata, 'data'))

        # Create logs directory if it doesn't exist. The source logs/ folder may not
        # exist in PyInstaller builds, so just create an empty directory structure.
        logs_dir = os.path.join(lvp_appdata, 'logs', 'LVP_Log')
        os.makedirs(logs_dir, exist_ok=True)

    elif windows_machine and not lvp_installed:
        _logger.info('Machine-Type - WINDOWS (not installed)')
        source_path = script_path
    else:
        _logger.info('Machine-Type - NON-WINDOWS')
        source_path = script_path

    num_cores = os.cpu_count()
    _logger.info(f'Num cores identified as {num_cores}')

    return AppEnvironment(
        script_path=script_path,
        source_path=source_path,
        version=version,
        build_timestamp=build_timestamp,
        windows_machine=windows_machine,
        num_cores=num_cores,
        lvp_installed=lvp_installed,
    )


def _dist_version(name: str) -> str | None:
    try:
        import importlib.metadata as imeta

        return imeta.version(name)
    except Exception:
        return None


def camera_sdk_probe() -> list[str]:
    """Describe the camera-SDK Python bindings by IMPORTING them.

    importlib.metadata is the wrong instrument here: frozen (installer)
    builds bundle the modules but almost none of the dist metadata, so a
    metadata read reports "not installed" for bindings that import fine --
    and says nothing useful when the binding genuinely cannot import.
    Probing by import answers the only question the driver layer cares
    about (can this SDK be used?) and carries the exact failure reason
    when it cannot.

    Returns:
        Human-readable one-line descriptions, one per SDK, suitable for
        the startup banner and support bundles.
    """
    lines = []

    try:
        from pypylon import pylon
    except Exception as e:
        lines.append(f'pypylon: not importable ({type(e).__name__}: {e})')
    else:
        import pypylon

        binding = (
            getattr(pypylon, '__version__', None)
            or _dist_version('pypylon')
            or 'importable (version unknown)'
        )
        try:
            sdk = pylon.GetPylonVersionString()
        except Exception:
            try:
                sdk = '.'.join(str(x) for x in pylon.GetPylonVersion())
            except Exception:
                sdk = 'unknown'
        lines.append(f'pypylon binding: {binding} / Pylon SDK: {sdk}')

    try:
        from ids_peak import ids_peak as ids_binding
    except Exception as e:
        lines.append(f'ids_peak: not importable ({type(e).__name__}: {e})')
    else:
        version = (
            getattr(ids_binding, '__version__', None)
            or _dist_version('ids_peak')
            or 'importable (version unknown)'
        )
        lines.append(f'ids_peak: {version}')

    lines.extend(_CAMERA_SDK_PRELOAD_REPORT)
    return lines


_CAMERA_SDK_PRELOAD_REPORT: list[str] = []


def _loaded_module_census() -> list[str]:
    """Full paths of process-resident DLLs relevant to the camera stacks."""
    if os.name != 'nt':
        return []
    import ctypes
    from ctypes import wintypes

    psapi = ctypes.WinDLL('psapi')
    kernel32 = ctypes.WinDLL('kernel32')
    process = kernel32.GetCurrentProcess()
    needed = wintypes.DWORD()
    module_handles = (wintypes.HMODULE * 2048)()
    ok = psapi.EnumProcessModulesEx(
        process, module_handles, ctypes.sizeof(module_handles), ctypes.byref(needed), 0x03
    )
    if not ok:
        return ['<module census unavailable>']
    count = min(needed.value // ctypes.sizeof(wintypes.HMODULE), len(module_handles))
    interesting = re.compile(
        r'ids_|tbb|genapi|gcbase|pylon|msvcp|vcruntime|nodemapdata'
        r'|xmlparser|mathparser|log4cpp|python3',
        re.IGNORECASE,
    )
    paths = []
    buffer = ctypes.create_unicode_buffer(1024)
    for handle in module_handles[:count]:
        if psapi.GetModuleFileNameExW(process, handle, buffer, len(buffer)) and interesting.search(
            os.path.basename(buffer.value)
        ):
            paths.append(buffer.value)
    return paths


INSTALLER_LOG_PATTERN = 'LumaViewPro*.log'
_MAX_CAPTURED_INSTALLER_LOGS = 10


def capture_installer_logs(
    log_dir: str | pathlib.Path,
    temp_dir: str | pathlib.Path | None = None,
    max_files: int = _MAX_CAPTURED_INSTALLER_LOGS,
) -> list[str]:
    """Copy the Windows installer's own logs into the application log folder.

    The installer writes to the user TEMP directory, so a support bundle
    never carries them: an install that silently failed to replace a
    binary is then indistinguishable from an application defect. Windows
    also sweeps TEMP on its own schedule, so the copy happens at every
    startup rather than on request.

    Args:
        log_dir: Application log folder; logs land in its ``install``
            subfolder.
        temp_dir: Directory to scan. Defaults to the system temp folder.
        max_files: Newest-first cap, so a long-lived TEMP cannot turn
            startup into a large copy.

    Returns:
        Names copied by THIS call. Files already captured at the same
        size are skipped, so repeated startups converge; a log that grew
        (an install still writing when the app started) is recaptured.
    """
    source_dir = (
        pathlib.Path(temp_dir) if temp_dir is not None else pathlib.Path(tempfile.gettempdir())
    )
    destination = pathlib.Path(log_dir) / 'install'
    copied: list[str] = []
    try:
        candidates = [path for path in source_dir.glob(INSTALLER_LOG_PATTERN) if path.is_file()]
    except OSError as e:
        _logger.warning(f'Could not scan {source_dir} for installer logs: {e}')
        return copied

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for source in candidates[:max_files]:
        target = destination / source.name
        try:
            if target.exists() and target.stat().st_size == source.stat().st_size:
                continue
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError as e:
            _logger.warning(f'Could not capture installer log {source.name}: {e}')
            continue
        copied.append(source.name)

    if copied:
        _logger.info(f'Captured {len(copied)} installer log(s) into {destination}')
    return copied


def preload_camera_sdks() -> None:
    """Import the IDS camera stack while the process is still nearly empty.

    The IDS image-processing library initializes cleanly in a bare process
    (on-machine loader probes passed in every import order and environment
    variant) but its DLL initialization routine fails once the
    application's full DLL population -- pylon, Kivy/SDL, numpy, cv2 -- is
    resident. Importing the stack here, before any of those load, gives it
    the process state it is known to survive. ids_peak_ipl goes first: its
    package __init__ registers the DLL directory the core binding and the
    extension bridge resolve against.

    Failures are recorded per stage with a resident-module census so a
    support bundle names the failing stage without another on-site
    round-trip; camera_sdk_probe() folds the report into the startup
    banner. Machines without the IDS wheels (dev Macs, sim boxes) record
    nothing -- the probe's own line already reports absence.
    """
    stages = (
        ('ids_peak_ipl', 'import ids_peak_ipl'),
        ('ids_peak', 'from ids_peak import ids_peak'),
        ('ids_peak_ipl_extension', 'from ids_peak import ids_peak_ipl_extension'),
    )
    for name, statement in stages:
        try:
            if name == 'ids_peak_ipl':
                import ids_peak_ipl  # noqa: F401
            elif name == 'ids_peak':
                from ids_peak import ids_peak  # noqa: F401
            else:
                from ids_peak import ids_peak_ipl_extension  # noqa: F401
        except ModuleNotFoundError:
            return
        except Exception as e:
            _CAMERA_SDK_PRELOAD_REPORT.append(
                f'ids preload FAILED at {name} ({statement}): {type(e).__name__}: {e}'
            )
            for path in _loaded_module_census():
                _CAMERA_SDK_PRELOAD_REPORT.append(f'  resident: {path}')
            return
    _CAMERA_SDK_PRELOAD_REPORT.append('ids preload: all stages imported in clean process state')
