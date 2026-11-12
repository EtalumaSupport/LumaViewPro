# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Lumascope Classic's driver step on Windows: our WinUSB package takes
both of the scope's USB identities from every other package.

Run elevated, on the person's desktop, where Windows can ask whether to
install software from Etaluma: ``install`` by the app's Repair, and
``install-if-classic`` by the setup bundle at install, from a one-file build
of this step, which takes over only a PC that has had a Classic
(``install_if_classic``); ``uninstall`` by the installer. Every
step and every Windows answer goes to
``<install>\\logs\\driver_<step>_<version>_<time>.log``.

Install, in order:

1. Record each device of both identities and the driver bound to it.
2. Add our package to the driver store; Windows asks the person first,
   unless they chose to always trust Etaluma. When it does not enter the
   store -- the person declined, or Windows refused the package -- the step
   stops here and removes nothing, so a scope is never left with no driver
   on a PC our package cannot go onto.
3. Remove every other package that names either identity, whoever's
   (LumaView's libusbK, a Zadig binding, another vendor's), whole: a vendor
   package naming the bootloader beside its own device's ID loses that too,
   and the log names every ID each removed package named. Nothing is kept
   to restore: LumaView is not kept working once LumaViewPro takes over.
4. Bind the present devices to our package, remove our older packages,
   remove the past devices of both identities so each is matched afresh on
   its next plug, and record the bindings again.

Uninstall removes our packages and restores nothing, so the scope has no
driver until LumaViewPro is installed again; it logs what it did.

The app's Repair runs install as an administrator from the installed build
(``run_as_administrator``); ``repair_refusal`` says where it cannot.
"""

from __future__ import annotations

import enum
import logging
import os
import pathlib
import subprocess
import sys
from typing import Protocol

from _lumascope.drivers.exceptions import CameraUsbLookupFailedError
from _lumascope.drivers.usb_driver_binding import (
    FX2_BOOTLOADER,
    FX2_FIRMWARE,
    DeviceBinding,
    StorePackage,
    read_package,
    store_inf_folder,
    store_packages,
)
from _lumascope.modules.exceptions import CameraDriverRepairFailedError
from _lumascope.modules.log_stamp import make_formatter
import _lumascope.modules.wall_time as wall_time
from _lumascope.modules.path_utils import AppRuntime, app_runtime, launch_root, read_version

log = logging.getLogger('LVP.fx2_driver_step')

IDENTITIES = (FX2_BOOTLOADER, FX2_FIRMWARE)
PACKAGE_INF = 'Lumascope_Classic.inf'
# Where the installer puts the package, under the install folder.
PACKAGE_DIR = pathlib.Path('drivers', 'fx2')
# The installed name of this step's executable, beside the app's; the build
# names it in scripts/appBuild/config/fx2_bundle.py, and a test holds the two
# equal.
DRIVER_STEP_EXE = 'lvp_fx2_driver_step.exe'
# How long Repair waits for the step: its pnputil adds, exports and removals
# for two identities take seconds, and Windows' question to the person
# whether to install Etaluma's driver waits on their answer.
REPAIR_WAIT_S = 300.0
# The step's exit when Windows did not accept our package, so Repair tells
# it apart from a step that failed part way.
NOT_ACCEPTED_EXIT = 3
# The step's exit when it cannot write its log, before it changes anything:
# the log is its only record, and the error raised out of the windowed
# executable would be Windows' crash dialog on the person's desktop.
LOG_UNWRITABLE_EXIT = 4


class Installed(enum.Enum):
    """What install did: both identities bound, some step incomplete, our
    package not accepted into the driver store and nothing changed, or (the
    installer's run) nothing changed on a PC that has never had a Classic."""

    BOUND = 'bound'
    INCOMPLETE = 'incomplete'
    NOT_ACCEPTED = 'not_accepted'
    NO_CLASSIC = 'no_classic'


class Machine(Protocol):
    """What the step acts on: ``pnputil`` and the binding readers."""

    def run(self, *args: str) -> tuple[int, str]: ...

    def device_bindings(self, hardware_id: str) -> list[DeviceBinding]: ...

    def store_packages(self, hardware_id: str) -> list[StorePackage]: ...


class WindowsMachine:
    """The PC the step runs on."""

    def run(self, *args: str) -> tuple[int, str]:
        # No console window flashes on the person's desktop while Repair runs.
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        return result.returncode, (result.stdout + result.stderr).strip()

    def device_bindings(self, hardware_id: str) -> list[DeviceBinding]:
        # windows_usb loads the Windows DLLs at import, so it is imported
        # where it is called, and this module imports on every host.
        from _lumascope.drivers.windows_usb import device_bindings

        return device_bindings(hardware_id)

    def store_packages(self, hardware_id: str) -> list[StorePackage]:
        return store_packages(hardware_id, store_inf_folder())


def install(machine: Machine, package_dir: pathlib.Path) -> Installed:
    """Bind both identities to our package.

    Args:
        machine: The PC (``WindowsMachine``).
        package_dir: ``<install>\\drivers\\fx2``: the stamped INF and its
            signed catalog.
    """
    inf = package_dir / PACKAGE_INF
    _record(machine, 'before install')
    code, _ = _pnputil(machine, '/add-driver', str(inf))
    ours = _our_current_package(machine, inf)
    if ours is None:
        log.error(
            f'{inf.name} did not enter the driver store (pnputil exit {code}): installing '
            'it was declined, or Windows refused the package; nothing removed, every '
            'existing binding left as it was'
        )
        return Installed.NOT_ACCEPTED
    log.info(f'{inf.name} is in the driver store as {ours.inf} ({ours.driver_ver})')
    others_removed = _remove_others(machine)
    _pnputil(machine, '/add-driver', str(inf), '/install')
    older_removed = _remove_older_ours(machine, ours)
    past_removed = _remove_past_devices(machine)
    bound = _present_devices_on(machine, ours)
    _record(machine, 'after install')
    complete = all((others_removed, older_removed, past_removed, bound))
    return Installed.BOUND if complete else Installed.INCOMPLETE


def install_if_classic(machine: Machine, package_dir: pathlib.Path) -> Installed:
    """``install`` on a PC that has had a Lumascope Classic; elsewhere nothing.

    The installer runs on every PC, and most never see a Classic; on those
    the step changes nothing and Windows asks the person nothing. A PC has
    had one when it has seen a device of the running firmware's identity,
    present or past, or its store holds a package naming it (LumaView's, on
    every field PC). The bootloader's identity is every EEPROM-less Cypress
    FX2's, so a PC with only another vendor's FX2 is not taken over. A
    Classic first plugged in after the install is taken over by Repair.
    """
    if not machine.device_bindings(FX2_FIRMWARE) and not machine.store_packages(FX2_FIRMWARE):
        log.info(
            f'this PC has never had a Lumascope Classic: no device of {FX2_FIRMWARE} has '
            'been seen and no package names it; nothing changed'
        )
        return Installed.NO_CLASSIC
    return install(machine, package_dir)


def uninstall(machine: Machine) -> bool:
    """Remove our packages. True when every removal did."""
    _record(machine, 'before uninstall')
    complete = True
    for package in _our_packages(machine):
        code, _ = _pnputil(machine, '/delete-driver', package.inf, '/uninstall')
        complete = complete and code == 0
    _record(machine, 'after uninstall')
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


def _remove_others(machine: Machine) -> bool:
    others = {
        package.inf: package
        for identity in IDENTITIES
        for package in machine.store_packages(identity)
        if not package.ours
    }
    complete = True
    for package in others.values():
        log.info(
            f'removing {package.inf}, {package.provider} {package.driver_ver}, '
            f'which names {", ".join(package.hardware_ids)}'
        )
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
    """Whether every present device of both identities is bound to ``ours`` and started.

    The bind is read from what Windows bound, not from ``pnputil``'s exit
    code, which does not say which devices it moved; a device bound to
    ``ours`` that Windows did not start (a problem code) drives nothing, so
    it is not bound either.
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
            if not binding.present:
                continue
            if binding.inf != ours.inf:
                log.error(f'not bound to {ours.inf}: {binding.describe()}')
                complete = False
            elif not binding.started:
                log.error(f'bound to {ours.inf} and not started by Windows: {binding.describe()}')
                complete = False
    return complete


def repair_refusal() -> str | None:
    """Why this process cannot run Repair, as ``CameraDriverRepairRefusedError``'s
    reason; None where it can.

    Repair needs Windows, which binds a driver to each USB device, and an
    installed build that carries this step's executable and the signed
    package beside it: a source run or a build run from ``dist`` has
    neither, and a build made without a signing certificate leaves both out.
    """
    if sys.platform != 'win32':
        return 'repair_needs_windows'
    if app_runtime() != AppRuntime.INSTALLED:
        return 'repair_needs_installed_build'
    root = launch_root()
    if not (root / DRIVER_STEP_EXE).is_file() or not (root / PACKAGE_DIR / PACKAGE_INF).is_file():
        return 'repair_needs_driver_package'
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
    """``lvp_fx2_driver_step install|uninstall``, run from the install folder,
    or ``lvp_fx2_driver_step_setup install-if-classic <log folder>``.

    The setup bundle runs a one-file build of the step from its package
    cache, the INF and catalog beside it, and names the log folder in
    LumaViewPro's install folder; Repair and the installer's uninstall run
    the one in the install folder, its package under ``drivers\\fx2``.
    """
    here = launch_root()
    if len(argv) == 1 and argv[0] in ('install', 'uninstall'):
        package_dir, logs = here / PACKAGE_DIR, here / 'logs'
    elif len(argv) == 2 and argv[0] == 'install-if-classic':
        package_dir, logs = here, pathlib.Path(argv[1])
    else:
        print(
            'usage: lvp_fx2_driver_step install|uninstall, '
            'or lvp_fx2_driver_step_setup install-if-classic <log folder>',
            file=sys.stderr,
        )
        return 2
    version, _ = read_version()
    stamp = wall_time.now().strftime('%Y%m%d_%H%M%S')
    try:
        logs.mkdir(exist_ok=True)
        handler = logging.FileHandler(
            logs / f'driver_{argv[0]}_{version}_{stamp}.log', encoding='utf-8'
        )
    except OSError:
        return LOG_UNWRITABLE_EXIT
    handler.setFormatter(make_formatter('%(levelname)s %(message)s'))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        log.info(
            f'LumaViewPro {version} FX2 driver step: {argv[0]}, from {here}, PID {os.getpid()}'
        )
        installed = None
        try:
            if argv[0] == 'install':
                installed = install(WindowsMachine(), package_dir)
                complete = installed is Installed.BOUND
            elif argv[0] == 'install-if-classic':
                installed = install_if_classic(WindowsMachine(), package_dir)
                complete = installed in (Installed.BOUND, Installed.NO_CLASSIC)
            else:
                complete = uninstall(WindowsMachine())
        except Exception:
            # The top of the step: this executable has no console, and its exit
            # code says only that it failed, so the log is the only place the
            # cause is told.
            log.exception(f'{argv[0]} stopped on an unexpected error')
            return 1
        if installed is Installed.NOT_ACCEPTED:
            log.info(f'{argv[0]} not done: Windows did not accept the package')
            return NOT_ACCEPTED_EXIT
        log.info(f'{argv[0]} {"complete" if complete else "incomplete: see the errors above"}')
        return 0 if complete else 1
    finally:
        log.removeHandler(handler)
        handler.close()
