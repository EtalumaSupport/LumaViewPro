# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The FX2 driver step takes both of the Classic's USB identities for our
package, removing every other package that names either, and restores
nothing at uninstall; a package of ours that cannot enter the store removes
nothing.

``src/_lumascope/modules/fx2_driver_step.py`` is what the app's Repair runs elevated to
install, and the MSI to uninstall. The machine it acts on is stood in by ``_Machine``: a driver store
that is a folder of INF files read through the real ``store_packages``, and
``pnputil`` acting on it as Windows does for the commands the step runs
(add, add and install, delete, remove a device; any other command fails the
test). ``/install`` binds each present device to the newest package that
names it, the tie-break Windows applies between packages of equal signature.
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib.util
import pathlib
import re
import sys

from _lumascope.drivers.usb_driver_binding import (
    FX2_BOOTLOADER,
    FX2_FIRMWARE,
    DeviceBinding,
    store_packages,
)
from _lumascope.modules import fx2_driver_step, path_utils

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SHIPPED_INF = (
    REPO_ROOT / 'scripts' / 'appBuild' / 'build_exe' / 'fx2_driver' / 'Lumascope_Classic.inf'
)


def _inf(provider: str, catalog: str, date: str, *ids: str) -> str:
    models = ''.join(f'Dev = Install, {i}\n' for i in ids)
    return (
        f'[Version]\nProvider = "{provider}"\nCatalogFile = {catalog}\n'
        f'DriverVer = {date},1.0.0.0\n[Manufacturer]\nM = Models, NTamd64\n'
        f'[Models.NTamd64]\n{models}'
    )


# LumaView's unsigned libusbK package, on every field PC, names both identities.
LUMAVIEW_LIBUSBK = _inf('libusbK', 'LumaView.cat', '03/04/2019', FX2_BOOTLOADER, FX2_FIRMWARE)
ACME_DEVICE = 'USB\\VID_1234&PID_0001'
# Another vendor's package naming the FX2 bootloader beside its own device.
OTHER_VENDOR_BOOTLOADER = _inf('Acme', 'acme.cat', '05/06/2025', FX2_BOOTLOADER, ACME_DEVICE)


class _Machine:
    """A stand-in by design: the Windows driver store, which no test host has."""

    def __init__(
        self, store: pathlib.Path, *, accepts_ours: bool = True, refuses_delete: str | None = None
    ):
        self.store = store
        self.accepts_ours = accepts_ours
        # A published name whose delete pnputil refuses (a package in use).
        self.refuses_delete = refuses_delete
        self.devices: list[DeviceBinding] = []
        self.commands: list[tuple[str, ...]] = []

    def publish(self, text: str) -> str:
        names = {p.name for p in self.store.glob('oem*.inf')}
        number = 1
        while f'oem{number}.inf' in names:
            number += 1
        (self.store / f'oem{number}.inf').write_text(text, encoding='ascii')
        return f'oem{number}.inf'

    def plug(self, hardware_id: str, *, present: bool = True, inf: str | None = None):
        package = next((p for p in self.store_packages(hardware_id) if p.inf == inf), None)
        self.devices.append(
            DeviceBinding(
                instance_id=f'{hardware_id}\\{len(self.devices)}',
                present=present,
                service='WinUSB' if package else None,
                provider=package.provider if package else None,
                inf=inf,
                version=None,
                date=None,
                problem_code=None if package else 28,
            )
        )

    def device_bindings(self, hardware_id):
        return [d for d in self.devices if d.instance_id.startswith(hardware_id + '\\')]

    def store_packages(self, hardware_id):
        return store_packages(hardware_id, str(self.store))

    def run(self, *args):
        self.commands.append(args)
        command, rest = args[1], args[2:]
        if command == '/add-driver':
            text = pathlib.Path(rest[0]).read_text(encoding='ascii')
            ours = 'Lumascope_Classic.cat' in text
            if ours and not self.accepts_ours:
                return 1, 'INF_E_NO_CATALOG_FOR_OEM_INF'
            if not any((self.store / p).read_text(encoding='ascii') == text for p in self._oems()):
                self.publish(text)
            if '/install' in rest:
                self._bind_present_to_best()
            return 0, ''
        if command == '/delete-driver':
            if rest[0] == self.refuses_delete:
                return (
                    1,
                    'Failed to delete driver package: One or more devices are presently installed',
                )
            (self.store / rest[0]).unlink()
            self.devices = [
                dataclasses.replace(d, inf=None, service=None, provider=None)
                if d.inf == rest[0]
                else d
                for d in self.devices
            ]
            return 0, ''
        if command == '/remove-device':
            self.devices = [d for d in self.devices if d.instance_id != rest[0]]
            return 0, ''
        raise AssertionError(f'unexpected command {args}')

    def _oems(self):
        return sorted(p.name for p in self.store.glob('oem*.inf'))

    def _bind_present_to_best(self):
        def newest(identity):
            packages = self.store_packages(identity)
            return max(
                packages,
                key=lambda p: datetime.datetime.strptime(
                    p.driver_ver.split(',')[0].strip(), '%m/%d/%Y'
                ),
                default=None,
            )

        bound = []
        for device in self.devices:
            identity = device.instance_id.rsplit('\\', 1)[0]
            best = newest(identity)
            if device.present and best is not None:
                device = dataclasses.replace(
                    device,
                    inf=best.inf,
                    service='WinUSB',
                    provider=best.provider,
                    problem_code=None,
                )
            bound.append(device)
        self.devices = bound


def _package_dir(tmp_path) -> pathlib.Path:
    # build.ps1 runs the stamper from its folder by path; the test does the same.
    path = REPO_ROOT / 'scripts' / 'appBuild' / 'config' / 'fx2_driver_package.py'
    spec = importlib.util.spec_from_file_location('fx2_driver_package', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package_dir = tmp_path / 'install' / 'drivers' / 'fx2'
    module.stamp_driver_package(
        str(SHIPPED_INF), str(package_dir), '4.0.0.38', datetime.date(2026, 10, 17)
    )
    return package_dir


def _machine(tmp_path, **kwargs) -> _Machine:
    store = tmp_path / 'INF'
    store.mkdir()
    return _Machine(store, **kwargs)


def _bound(machine, hardware_id):
    return [(d.present, d.provider) for d in machine.device_bindings(hardware_id)]


def test_a_lumaview_bound_scope_ends_on_our_package_alone(tmp_path):
    machine = _machine(tmp_path)
    lumaview = machine.publish(LUMAVIEW_LIBUSBK)
    machine.plug(FX2_BOOTLOADER, inf=lumaview)
    machine.plug(FX2_FIRMWARE, inf=lumaview)

    installed = fx2_driver_step.install(machine, _package_dir(tmp_path))

    assert installed is fx2_driver_step.Installed.BOUND
    for identity in (FX2_BOOTLOADER, FX2_FIRMWARE):
        assert _bound(machine, identity) == [(True, 'Etaluma')]
        assert [p.provider for p in machine.store_packages(identity)] == ['Etaluma']


def test_a_package_that_cannot_enter_the_store_removes_nothing(tmp_path):
    machine = _machine(tmp_path, accepts_ours=False)
    lumaview = machine.publish(LUMAVIEW_LIBUSBK)
    machine.plug(FX2_FIRMWARE, inf=lumaview)

    installed = fx2_driver_step.install(machine, _package_dir(tmp_path))

    assert installed is fx2_driver_step.Installed.NOT_ACCEPTED
    assert _bound(machine, FX2_FIRMWARE) == [(True, 'libusbK')]
    assert [p.inf for p in machine.store_packages(FX2_FIRMWARE)] == [lumaview]


def test_another_vendors_bootloader_package_goes_whole_and_the_log_names_its_ids(tmp_path, caplog):
    machine = _machine(tmp_path)
    acme = machine.publish(OTHER_VENDOR_BOOTLOADER)

    with caplog.at_level('INFO', logger='LVP.fx2_driver_step'):
        installed = fx2_driver_step.install(machine, _package_dir(tmp_path))

    assert installed is fx2_driver_step.Installed.BOUND
    assert not (machine.store / acme).exists()
    # The step's log is the only record of what it removed; the vendor's own
    # device lost its package too, and the log says so.
    (removal,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith('removing')]
    assert acme in removal and FX2_BOOTLOADER in removal and ACME_DEVICE in removal


def test_a_newer_package_on_the_bootloader_is_removed_and_ours_bound(tmp_path):
    machine = _machine(tmp_path)
    # Dated after the build, so it would outrank ours if it were left.
    newer = machine.publish(_inf('libwdi', 'z.cat', '12/31/2026', FX2_BOOTLOADER))
    machine.plug(FX2_BOOTLOADER, inf=newer)

    installed = fx2_driver_step.install(machine, _package_dir(tmp_path))

    assert installed is fx2_driver_step.Installed.BOUND
    assert _bound(machine, FX2_BOOTLOADER) == [(True, 'Etaluma')]


def test_past_devices_and_our_older_package_are_removed(tmp_path):
    machine = _machine(tmp_path)
    older = machine.publish(
        _inf('Etaluma', 'Lumascope_Classic.cat', '01/02/2026', FX2_FIRMWARE, FX2_BOOTLOADER)
    )
    machine.plug(FX2_BOOTLOADER, present=False, inf=older)
    machine.plug(FX2_FIRMWARE, present=False, inf=older)

    installed = fx2_driver_step.install(machine, _package_dir(tmp_path))

    assert installed is fx2_driver_step.Installed.BOUND
    assert machine.devices == []
    assert [p.driver_ver for p in machine.store_packages(FX2_FIRMWARE)] == ['10/17/2026,4.0.0.38']


def test_a_package_windows_will_not_delete_makes_the_install_incomplete(tmp_path):
    machine = _machine(tmp_path)
    lumaview = machine.publish(LUMAVIEW_LIBUSBK)
    machine.refuses_delete = lumaview
    machine.plug(FX2_BOOTLOADER, inf=lumaview)

    installed = fx2_driver_step.install(machine, _package_dir(tmp_path))

    assert installed is fx2_driver_step.Installed.INCOMPLETE
    assert lumaview in [p.inf for p in machine.store_packages(FX2_BOOTLOADER)]


def test_uninstall_removes_ours_and_restores_nothing(tmp_path):
    machine = _machine(tmp_path)
    machine.plug(FX2_FIRMWARE, inf=machine.publish(LUMAVIEW_LIBUSBK))
    package_dir = _package_dir(tmp_path)
    fx2_driver_step.install(machine, package_dir)

    complete = fx2_driver_step.uninstall(machine)

    assert complete is True
    assert machine.store_packages(FX2_FIRMWARE) == []
    assert machine.store_packages(FX2_BOOTLOADER) == []
    assert _bound(machine, FX2_FIRMWARE) == [(True, None)]


def test_install_and_uninstall_never_touch_the_trust_store(tmp_path):
    # Pre-trusting our certificate would install the driver silently, which
    # Microsoft says never to do for a driver distributed outside the
    # organization; Windows asks the person instead.
    # A signing certificate beside the package is the file the step once
    # trusted; it is left alone now.
    machine = _machine(tmp_path)
    package_dir = _package_dir(tmp_path)
    (package_dir / 'Etaluma_TrustedPublisher.cer').write_bytes(b'DER bytes')

    fx2_driver_step.install(machine, package_dir)
    fx2_driver_step.uninstall(machine)

    assert [args[0] for args in machine.commands] == ['pnputil'] * len(machine.commands)


def _run_frozen_from(install_dir, monkeypatch) -> None:
    """The step as the installer runs it: frozen, from the install folder,
    which holds the build's files."""
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(install_dir / 'lvp_fx2_driver_step.exe'))
    monkeypatch.setattr(path_utils, 'build_root', lambda: install_dir)


def _run_main(tmp_path, monkeypatch, machine) -> tuple[int, str]:
    package_dir = _package_dir(tmp_path)
    install_dir = package_dir.parent.parent
    (install_dir / 'version.txt').write_text('4.0.0-beta38\n', encoding='ascii')
    _run_frozen_from(install_dir, monkeypatch)
    monkeypatch.setattr(fx2_driver_step, 'WindowsMachine', lambda: machine)

    code = fx2_driver_step.main(['install'])

    (log_file,) = (install_dir / 'logs').glob('driver_install_4.0.0-beta38_*.log')
    return code, log_file.read_text(encoding='utf-8')


def test_a_package_windows_did_not_accept_exits_with_its_own_code(tmp_path, monkeypatch):
    code, text = _run_main(tmp_path, monkeypatch, _machine(tmp_path, accepts_ours=False))

    assert code == fx2_driver_step.NOT_ACCEPTED_EXIT
    assert 'did not enter the driver store' in text
    assert text.rstrip().endswith('install not done: Windows did not accept the package')


def test_the_step_logs_to_its_install_folder_and_exits_1_when_incomplete(tmp_path, monkeypatch):
    machine = _machine(tmp_path)
    lumaview = machine.publish(LUMAVIEW_LIBUSBK)
    machine.refuses_delete = lumaview
    machine.plug(FX2_BOOTLOADER, inf=lumaview)

    code, text = _run_main(tmp_path, monkeypatch, machine)

    assert code == 1
    assert text.rstrip().endswith('install incomplete: see the errors above')
    # Every record starts with the full date and time LumaViewPro's logs share.
    assert re.match(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} INFO LumaViewPro 4\.0\.0-beta38 ', text)


def test_an_unexpected_error_is_logged_with_its_traceback_and_exits_1(tmp_path, monkeypatch):
    package_dir = _package_dir(tmp_path)
    install_dir = package_dir.parent.parent
    _run_frozen_from(install_dir, monkeypatch)

    class _Unreadable(_Machine):
        def store_packages(self, hardware_id):
            raise PermissionError('INF folder unreadable')

    monkeypatch.setattr(fx2_driver_step, 'WindowsMachine', lambda: _Unreadable(tmp_path))

    code = fx2_driver_step.main(['uninstall'])

    (log_file,) = (install_dir / 'logs').glob('driver_uninstall_*.log')
    text = log_file.read_text(encoding='utf-8')
    assert code == 1
    assert 'uninstall stopped on an unexpected error' in text
    assert 'PermissionError: INF folder unreadable' in text


def test_a_log_the_step_cannot_write_exits_with_its_own_code(tmp_path, monkeypatch):
    # Raised out of the windowed executable, the error would be Windows'
    # crash dialog on the person's desktop in the middle of an install.
    package_dir = _package_dir(tmp_path)
    install_dir = package_dir.parent.parent
    (install_dir / 'logs').write_text('a file where the log folder goes', encoding='ascii')
    _run_frozen_from(install_dir, monkeypatch)
    machine = _machine(tmp_path)
    monkeypatch.setattr(fx2_driver_step, 'WindowsMachine', lambda: machine)

    code = fx2_driver_step.main(['install'])

    assert code == fx2_driver_step.LOG_UNWRITABLE_EXIT
    assert machine.commands == []
