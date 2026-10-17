# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Lumascope Classic's driver step on Windows: our WinUSB package bound,
stale ones cleared.

Run elevated, by the installer at install and uninstall (``lvp_fx2_driver_step
install`` / ``uninstall``). An installer action cannot show the person
anything, so the step only does and logs: every step and every Windows
answer goes to ``<install>\\logs\\driver_install_<version>_<time>.log``, and
the app's connect is where a binding that cannot work is told.

Install, in order:

1. Record each device of both identities and the driver bound to it.
2. Trust the certificate the package is signed with, when the build carries
   one, in ``LocalMachine\\TrustedPublisher``.
3. Add our package to the driver store. When it does not enter the store
   the step stops here and removes nothing, so a working Zadig or LumaView
   Classic binding is never stripped from a PC our package cannot replace
   it on.
4. For the running firmware's identity (ours alone), export and then remove
   every other vendor's package that names it, so uninstall can restore it.
   The bootloader's identity is every EEPROM-less Cypress FX2's, so nothing
   is removed for it: our newer package outranks an older one there.
5. Bind the present devices to our package, remove our older packages,
   remove the past devices of both identities so each is matched afresh on
   its next plug, and record the bindings again.

Uninstall removes our packages, restores what install exported, removes the
certificates install trusted, and logs what it did and what it left.

The app's Repair runs the same install, as an administrator, from the
installed build (``run_as_administrator``); ``repair_refusal`` says where it
cannot.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Protocol

from drivers.exceptions import CameraUsbLookupFailedError
from drivers.usb_driver_binding import (
    FX2_BOOTLOADER,
    FX2_FIRMWARE,
    DeviceBinding,
    StorePackage,
    read_package,
    store_inf_folder,
    store_packages,
)
from modules.exceptions import CameraDriverRepairFailedError
from modules.path_utils import AppRuntime, app_runtime, read_version

log = logging.getLogger('LVP.fx2_driver_step')

IDENTITIES = (FX2_BOOTLOADER, FX2_FIRMWARE)
PACKAGE_INF = 'Lumascope_Classic.inf'
SIGNING_CERTIFICATE = 'Etaluma_TrustedPublisher.cer'
# Install appends the thumbprint of each certificate it trusted, one per
# line, so uninstall removes those and no other: each renewal adds one.
TRUSTED_RECORD = 'trusted_publishers.txt'
REMOVED_FOLDER = 'removed'
# The installed name of this step's executable, beside the app's; the build
# names it in scripts/appBuild/config/fx2_bundle.py, and a test holds the two
# equal.
DRIVER_STEP_EXE = 'lvp_fx2_driver_step.exe'
# How long Repair waits for the step: its pnputil adds, exports and removals
# for two identities take seconds; the wait bounds a step held on a Windows
# prompt nobody can see.
REPAIR_WAIT_S = 300.0


class Machine(Protocol):
    """What the step acts on: ``pnputil``, ``certutil`` and the binding readers."""

    def run(self, *args: str) -> tuple[int, str]: ...

    def device_bindings(self, hardware_id: str) -> list[DeviceBinding]: ...

    def store_packages(self, hardware_id: str) -> list[StorePackage]: ...


class WindowsMachine:
    """The PC the step runs on."""

    def run(self, *args: str) -> tuple[int, str]:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        return result.returncode, (result.stdout + result.stderr).strip()

    def device_bindings(self, hardware_id: str) -> list[DeviceBinding]:
        # winusb_iso loads the Windows DLLs at import, so it is imported
        # where it is called, and this module imports on every host.
        from drivers.winusb_iso import device_bindings

        return device_bindings(hardware_id)

    def store_packages(self, hardware_id: str) -> list[StorePackage]:
        return store_packages(hardware_id, store_inf_folder())


def install(machine: Machine, package_dir: pathlib.Path) -> bool:
    """Bind both identities to our package. True when every step did.

    Args:
        machine: The PC (``WindowsMachine``).
        package_dir: ``<install>\\drivers\\fx2``: the stamped INF, its catalog,
            and the signing certificate when the build carries one.
    """
    inf = package_dir / PACKAGE_INF
    _record(machine, 'before install')
    trusted = _trust(machine, package_dir)
    code, _ = _pnputil(machine, '/add-driver', str(inf))
    ours = _our_current_package(machine, inf)
    if ours is None:
        log.error(
            f'{inf.name} did not enter the driver store (pnputil exit {code}); '
            'nothing removed, every existing binding left as it was'
        )
        return False
    log.info(f'{inf.name} is in the driver store as {ours.inf} ({ours.driver_ver})')
    exported = _export_and_remove_others(machine, package_dir / REMOVED_FOLDER)
    _pnputil(machine, '/add-driver', str(inf), '/install')
    older_removed = _remove_older_ours(machine, ours)
    past_removed = _remove_past_devices(machine)
    bound = _present_devices_on(machine, ours)
    _record(machine, 'after install')
    return all((trusted, exported, older_removed, past_removed, bound))


def uninstall(machine: Machine, package_dir: pathlib.Path) -> bool:
    """Remove our packages and restore what install removed. True when every step did."""
    _record(machine, 'before uninstall')
    complete = True
    for package in _our_packages(machine):
        code, _ = _pnputil(machine, '/delete-driver', package.inf, '/uninstall')
        complete = complete and code == 0
    complete = _restore_removed(machine, package_dir / REMOVED_FOLDER) and complete
    complete = _untrust(machine, package_dir) and complete
    _record(machine, 'after uninstall')
    left = [name for name in (REMOVED_FOLDER, TRUSTED_RECORD) if (package_dir / name).exists()]
    if left:
        log.warning(f'left in {package_dir}: {", ".join(left)}')
    return complete


def _pnputil(machine: Machine, *args: str) -> tuple[int, str]:
    code, out = machine.run('pnputil', *args)
    log.info(f'pnputil {" ".join(args)} -> exit {code}\n{out}')
    return code, out


def _record(machine: Machine, moment: str) -> None:
    for identity in IDENTITIES:
        try:
            bindings = machine.device_bindings(identity)
        except CameraUsbLookupFailedError as e:
            log.error(f'{moment}: the devices of {identity} could not be read: {e}')
            continue
        if not bindings:
            log.info(f'{moment}: no device of {identity} has been seen on this PC')
        for binding in bindings:
            log.info(f'{moment}: {binding.describe()}')
        for package in machine.store_packages(identity):
            whose = 'ours' if package.ours else 'not ours'
            log.info(
                f'{moment}: the store holds {package.inf} for {identity}, '
                f'{package.provider} {package.driver_ver} ({whose})'
            )


def _our_packages(machine: Machine) -> list[StorePackage]:
    found = {}
    for identity in IDENTITIES:
        for package in machine.store_packages(identity):
            if package.ours:
                found[package.inf] = package
    return list(found.values())


def _our_current_package(machine: Machine, inf: pathlib.Path) -> StorePackage | None:
    """Our package in the store with this build's ``DriverVer``, or None."""
    stamped = read_package(str(inf)).driver_ver
    for package in _our_packages(machine):
        if package.driver_ver == stamped:
            return package
    return None


def _trust(machine: Machine, package_dir: pathlib.Path) -> bool:
    certificate = package_dir / SIGNING_CERTIFICATE
    if not certificate.exists():
        log.warning(
            'this build carries no signing certificate; the package enters the store '
            'only if Windows already trusts its signer'
        )
        return True
    thumbprint = hashlib.sha1(certificate.read_bytes()).hexdigest().upper()
    code, out = machine.run('certutil', '-addstore', 'TrustedPublisher', str(certificate))
    log.info(f'certutil -addstore TrustedPublisher {certificate.name} -> exit {code}\n{out}')
    if code != 0:
        return False
    record = package_dir / TRUSTED_RECORD
    recorded = record.read_text(encoding='ascii').split() if record.exists() else []
    if thumbprint not in recorded:
        with open(record, 'a', encoding='ascii') as f:
            f.write(thumbprint + '\n')
    return True


def _untrust(machine: Machine, package_dir: pathlib.Path) -> bool:
    record = package_dir / TRUSTED_RECORD
    if not record.exists():
        return True
    kept = []
    for thumbprint in record.read_text(encoding='ascii').split():
        code, out = machine.run('certutil', '-delstore', 'TrustedPublisher', thumbprint)
        log.info(f'certutil -delstore TrustedPublisher {thumbprint} -> exit {code}\n{out}')
        if code != 0:
            kept.append(thumbprint)
    if kept:
        record.write_text(''.join(t + '\n' for t in kept), encoding='ascii')
        return False
    record.unlink()
    return True


def _export_and_remove_others(machine: Machine, removed_dir: pathlib.Path) -> bool:
    complete = True
    for package in machine.store_packages(FX2_FIRMWARE):
        if package.ours:
            continue
        target = removed_dir / package.inf
        target.mkdir(parents=True, exist_ok=True)
        code, _ = _pnputil(machine, '/export-driver', package.inf, str(target))
        if code != 0:
            log.error(f'{package.inf} could not be exported, so it is left in the store')
            complete = False
            continue
        code, _ = _pnputil(machine, '/delete-driver', package.inf, '/uninstall')
        complete = complete and code == 0
    return complete


def _remove_older_ours(machine: Machine, current: StorePackage) -> bool:
    complete = True
    for package in _our_packages(machine):
        if package.inf != current.inf:
            code, _ = _pnputil(machine, '/delete-driver', package.inf, '/uninstall')
            complete = complete and code == 0
    return complete


def _remove_past_devices(machine: Machine) -> bool:
    complete = True
    for identity in IDENTITIES:
        try:
            bindings = machine.device_bindings(identity)
        except CameraUsbLookupFailedError as e:
            log.error(f'the past devices of {identity} could not be read, so none removed: {e}')
            complete = False
            continue
        for binding in bindings:
            if not binding.present:
                code, _ = _pnputil(machine, '/remove-device', binding.instance_id)
                complete = complete and code == 0
    return complete


def _present_devices_on(machine: Machine, ours: StorePackage) -> bool:
    """Whether every present device of both identities is bound to ``ours``.

    The bind is read from what Windows bound, not from ``pnputil``'s exit
    code, which does not say which devices it moved.
    """
    complete = True
    for identity in IDENTITIES:
        try:
            bindings = machine.device_bindings(identity)
        except CameraUsbLookupFailedError as e:
            log.error(f'the devices of {identity} could not be read, so the bind is unknown: {e}')
            complete = False
            continue
        for binding in bindings:
            if binding.present and binding.inf != ours.inf:
                log.error(f'not bound to {ours.inf}: {binding.describe()}')
                complete = False
    return complete


def _restore_removed(machine: Machine, removed_dir: pathlib.Path) -> bool:
    if not removed_dir.exists():
        return True
    complete = True
    for exported in sorted(removed_dir.iterdir()):
        infs = sorted(exported.glob('*.inf'))
        restored = bool(infs)
        for inf in infs:
            code, _ = _pnputil(machine, '/add-driver', str(inf), '/install')
            restored = restored and code in (0, 3010)
        if restored:
            shutil.rmtree(exported)
        else:
            log.error(f'{exported.name} was not restored; its export is left in {exported}')
            complete = False
    if not any(removed_dir.iterdir()):
        removed_dir.rmdir()
    return complete


def repair_refusal() -> str | None:
    """Why this process cannot run Repair, as ``CameraDriverRepairRefusedError``'s
    reason; None where it can.

    Repair needs Windows, which binds a driver to each USB device, and an
    installed build, which carries this step's executable and the signed
    package beside it; a source run or a build run from ``dist`` has neither.
    """
    if sys.platform != 'win32':
        return 'repair_needs_windows'
    if app_runtime() is not AppRuntime.INSTALLED:
        return 'repair_needs_installed_build'
    return None


def run_as_administrator(executable: pathlib.Path, argument: str, wait_s: float) -> int | None:
    """Run ``executable argument`` as an administrator and wait for it.

    Windows asks for an administrator's approval once (UAC), on the desktop
    of the PC this runs on.

    Returns:
        int | None: The step's exit code; None when the approval was not
            given.

    Raises:
        CameraDriverRepairFailedError: Windows could not start it, it did not
            finish within ``wait_s``, or the wait or its exit code could not
            be read.
    """
    import ctypes
    from ctypes import wintypes

    class _ShellExecuteInfo(ctypes.Structure):
        _fields_ = [
            ('cbSize', wintypes.DWORD),
            ('fMask', wintypes.ULONG),
            ('hwnd', wintypes.HWND),
            ('lpVerb', wintypes.LPCWSTR),
            ('lpFile', wintypes.LPCWSTR),
            ('lpParameters', wintypes.LPCWSTR),
            ('lpDirectory', wintypes.LPCWSTR),
            ('nShow', ctypes.c_int),
            ('hInstApp', wintypes.HINSTANCE),
            ('lpIDList', ctypes.c_void_p),
            ('lpClass', wintypes.LPCWSTR),
            ('hkeyClass', wintypes.HKEY),
            ('dwHotKey', wintypes.DWORD),
            ('hIconOrMonitor', wintypes.HANDLE),
            ('hProcess', wintypes.HANDLE),
        ]

    see_mask_nocloseprocess = 0x40
    see_mask_noasync = 0x100
    sw_hide = 0
    error_cancelled = 1223
    wait_object_0 = 0
    wait_timeout = 0x102

    shell32 = ctypes.WinDLL('shell32', use_last_error=True)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    # Every call is declared: an undeclared return is a C int, which cuts a
    # 64-bit process handle short.
    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(_ShellExecuteInfo)]
    shell32.ShellExecuteExW.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    logs = executable.parent / 'logs'
    info = _ShellExecuteInfo()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = see_mask_nocloseprocess | see_mask_noasync
    info.lpVerb = 'runas'
    info.lpFile = str(executable)
    info.lpParameters = argument
    info.lpDirectory = str(executable.parent)
    info.nShow = sw_hide
    if not shell32.ShellExecuteExW(ctypes.byref(info)):
        code = ctypes.get_last_error()
        if code == error_cancelled:
            return None
        raise CameraDriverRepairFailedError(
            f'Windows could not start {executable.name} (error {code})', logs
        )
    if not info.hProcess:
        raise CameraDriverRepairFailedError(
            f'Windows started {executable.name} with no process to wait on', logs
        )
    try:
        waited = kernel32.WaitForSingleObject(info.hProcess, int(wait_s * 1000))
        if waited == wait_timeout:
            raise CameraDriverRepairFailedError(
                f'{executable.name} did not finish within {wait_s:.0f} s', logs
            )
        if waited != wait_object_0:
            raise CameraDriverRepairFailedError(
                f'waiting for {executable.name} failed (error {ctypes.get_last_error()})', logs
            )
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(exit_code)):
            raise CameraDriverRepairFailedError(
                f'its exit code could not be read (error {ctypes.get_last_error()})', logs
            )
        return exit_code.value
    finally:
        kernel32.CloseHandle(info.hProcess)


def main(argv: list[str]) -> int:
    """``lvp_fx2_driver_step install|uninstall``, run from the install folder."""
    if len(argv) != 1 or argv[0] not in ('install', 'uninstall'):
        print('usage: lvp_fx2_driver_step install|uninstall', file=sys.stderr)
        return 2
    install_dir = pathlib.Path(sys.executable).resolve().parent
    version, _ = read_version(install_dir)
    logs = install_dir / 'logs'
    logs.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    handler = logging.FileHandler(
        logs / f'driver_{argv[0]}_{version}_{stamp}.log', encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        log.info(
            f'LumaViewPro {version} FX2 driver step: {argv[0]}, from {install_dir}, '
            f'PID {os.getpid()}'
        )
        step = install if argv[0] == 'install' else uninstall
        try:
            complete = step(WindowsMachine(), install_dir / 'drivers' / 'fx2')
        except Exception:
            # The top of the step: the installer ignores the exit code and this
            # executable has no console, so the log is the only place it is told.
            log.exception(f'{argv[0]} stopped on an unexpected error')
            return 1
        log.info(f'{argv[0]} {"complete" if complete else "incomplete: see the errors above"}')
        return 0 if complete else 1
    finally:
        log.removeHandler(handler)
        handler.close()
