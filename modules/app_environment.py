# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Application environment initialization -- paths, version, platform detection."""

import logging
import os
import pathlib
import re
import shutil
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

    windows_machine = os.name == 'nt'

    from modules.path_utils import AppRuntime, app_runtime, get_source_root, read_version

    version, build_timestamp = read_version(pathlib.Path(script_path))

    # The data root is path_utils'. An installed build's is in Documents and
    # starts empty, so the shipped data is copied into it on first launch;
    # this runs before the logger is imported, which reads debug_mode there.
    source_path = str(get_source_root())
    if app_runtime() is AppRuntime.INSTALLED:
        os.makedirs(source_path, exist_ok=True)
        if not os.path.exists(os.path.join(source_path, 'data')):
            shutil.copytree(os.path.join(script_path, 'data'), os.path.join(source_path, 'data'))

    num_cores = os.cpu_count()

    return AppEnvironment(
        script_path=script_path,
        source_path=source_path,
        version=version,
        build_timestamp=build_timestamp,
        windows_machine=windows_machine,
        num_cores=num_cores,
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


# C-runtime DLL families the censuses report, as bare stems so a version
# bump (msvcp140 -> msvcp150) cannot silently drop a family from view.
# A runtime that ships but is missing from this tuple is invisible in the
# exact report that exists to find shadowing, so the list is deliberately
# wider than the set the app ships today: vcomp140 (OpenMP) and the
# legacy msvcr* both shadow the same way msvcp does.
#
# api-ms-win-crt-* is deliberately EXCLUDED: ~40 OS-provided stubs that
# resolve from System32 on every process and never shadow, so listing
# them buries the entries that carry information.
C_RUNTIME_DLL_FAMILIES = (
    'concrt',
    'msvcp',
    'msvcr',
    'ucrtbase',
    'vcomp',
    'vcruntime',
)

# Camera-stack modules worth reporting alongside the C runtimes -- the
# SDKs whose native loads are what a shadowed runtime actually breaks.
_CAMERA_STACK_FAMILIES = (
    'gcbase',
    'genapi',
    'ids_',
    'log4cpp',
    'mathparser',
    'nodemapdata',
    'pylon',
    'python3',
    'tbb',
    'xmlparser',
)


def crt_dll_pattern() -> re.Pattern:
    """Filename matcher for C-runtime DLLs, version-agnostic.

    Matches the bare family plus any version/hash suffix, so both
    ``MSVCP140.dll`` and numpy's ``msvcp140-<hash>.dll`` are caught.
    """
    families = '|'.join(C_RUNTIME_DLL_FAMILIES)
    return re.compile(rf'^({families})[a-z0-9_\-]*\.dll$', re.IGNORECASE)


def loaded_module_census() -> list[str]:
    """Full paths of process-resident DLLs relevant to the camera stacks.

    The paths answer the question no log line otherwise can: WHICH copy of
    a DLL won the loader's search -- an app-local file beside the exe
    shadows System32 for the whole process, so two machines with identical
    installs can load different runtimes.
    """
    if os.name != 'nt':
        return []
    import ctypes
    from ctypes import wintypes

    psapi = ctypes.WinDLL('psapi')
    kernel32 = ctypes.WinDLL('kernel32')
    # Without declared argtypes, ctypes passes integer arguments as 32-bit
    # C ints -- an HMODULE is a 64-bit pointer, so any module mapped above
    # 4 GiB got a truncated handle, GetModuleFileNameExW failed on it, and
    # the module silently vanished from the census. ASLR routinely maps
    # DLLs that high, so the census could read as clean while missing the
    # very modules it exists to report.
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModulesEx.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
    ]
    psapi.EnumProcessModulesEx.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD

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
        '|'.join(_CAMERA_STACK_FAMILIES + C_RUNTIME_DLL_FAMILIES),
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

# MSI writes its verbose log as UTF-16, so every character costs two bytes
# for no added information -- a single install log runs ~6 MB where the
# same text is ~3 MB. These land in the user's Documents folder and stay
# there, so the copy transcodes to UTF-8 rather than storing the wide
# form. Byte-order marks, little- and big-endian.
_UTF16_BOMS = (b'\xff\xfe', b'\xfe\xff')


def _transcode_utf16_to_utf8(source: pathlib.Path, target: pathlib.Path) -> bool:
    """Write ``source`` to ``target`` as UTF-8 when it is UTF-16.

    Returns True when the transcode happened. A file that is not UTF-16,
    or that will not decode as the BOM claims, returns False so the
    caller copies the bytes verbatim -- a log that cannot be transcoded
    is still evidence, and halving its size is never worth losing it.
    """
    try:
        raw = source.read_bytes()
    except OSError:
        return False
    if raw[:2] not in _UTF16_BOMS:
        return False
    try:
        # 'utf-16' reads the BOM to pick the endianness and drops it.
        text = raw.decode('utf-16')
    except (UnicodeDecodeError, ValueError):
        return False
    target.write_bytes(text.encode('utf-8'))
    # Carry the source's timestamps across: the recapture check below
    # compares mtime, which is the only field that survives a transcode.
    shutil.copystat(source, target)
    return True


def capture_installer_logs(
    log_dir: str | pathlib.Path,
    *,
    temp_dir: str | pathlib.Path | None = None,
    max_files: int = _MAX_CAPTURED_INSTALLER_LOGS,
) -> list[str]:
    """Move the Windows installer's own logs into the application log folder.

    The installer writes to the user TEMP directory, so a support bundle
    never carries them: an install that silently failed to replace a
    binary is then indistinguishable from an application defect. Windows
    also sweeps TEMP on its own schedule, so the capture happens at every
    startup rather than on request.

    Each log is captured once: once its copy is complete the TEMP original
    is deleted, so a later start, or the next version's fresh data folder,
    never collects it again. A log that cannot be deleted (an install still
    writing it) stays, is named in a warning, and is taken at a later start.

    Only an installed build captures, and it asks the process's runtime
    itself rather than taking a caller's word. The logs describe installs,
    and any other run logs into the folder it was launched from: capturing
    there would take an install's log away from the installed build's
    folder.

    MSI's verbose log is UTF-16; it is transcoded to UTF-8 on the way in
    (see ``_transcode_utf16_to_utf8``), which halves what the user's
    Documents folder carries and loses nothing.

    Args:
        log_dir: Application log folder; logs land in its ``install``
            subfolder.
        temp_dir: Directory to scan. Defaults to the system temp folder.
        max_files: Newest-first cap, so a long-lived TEMP cannot turn
            startup into a large copy.

    Returns:
        Names copied by THIS call. A log still in TEMP that was already
        captured is recognised by modification time and only deleted; one
        that grew since (an install still writing when the app started) is
        recaptured. Timestamp rather than size, because a transcoded copy
        is never the size of its source -- and a same-size content change
        would slip past a size comparison anyway.
    """
    from modules import path_utils

    copied: list[str] = []
    if path_utils.app_runtime() is not path_utils.AppRuntime.INSTALLED:
        return copied
    source_dir = (
        pathlib.Path(temp_dir) if temp_dir is not None else pathlib.Path(tempfile.gettempdir())
    )
    destination = pathlib.Path(log_dir) / 'install'
    try:
        candidates = [path for path in source_dir.glob(INSTALLER_LOG_PATTERN) if path.is_file()]
    except OSError as e:
        _logger.warning(f'Could not scan {source_dir} for installer logs: {e}')
        return copied

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for source in candidates[:max_files]:
        target = destination / source.name
        try:
            if not (target.exists() and target.stat().st_mtime_ns == source.stat().st_mtime_ns):
                destination.mkdir(parents=True, exist_ok=True)
                if not _transcode_utf16_to_utf8(source, target):
                    shutil.copy2(source, target)
                copied.append(source.name)
        except OSError as e:
            _logger.warning(f'Could not capture installer log {source.name}: {e}')
            continue
        try:
            source.unlink()
        except OSError as e:
            _logger.warning(f'Installer log {source.name} captured but left in {source_dir}: {e}')

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
            for path in loaded_module_census():
                _CAMERA_SDK_PRELOAD_REPORT.append(f'  resident: {path}')
            return
    _CAMERA_SDK_PRELOAD_REPORT.append('ids preload: all stages imported in clean process state')
