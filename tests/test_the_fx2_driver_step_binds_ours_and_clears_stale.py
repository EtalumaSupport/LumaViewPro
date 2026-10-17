# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The installer's FX2 driver step binds our package and clears stale ones,
and never strips a working binding it cannot replace.

``modules/fx2_driver_step.py`` is what the MSI runs elevated at install and
uninstall. The machine it acts on is stood in by ``_Machine``: a driver store
that is a folder of INF files read through the real ``store_packages``, and
``pnputil`` acting on it as Windows does for the commands the step runs
(add, add and install, delete, export, remove a device). ``/install`` binds
each present device to the newest package that names it, the tie-break
Windows applies between packages of equal signature.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import importlib.util
import pathlib
import shutil

from drivers.usb_driver_binding import (
    FX2_BOOTLOADER,
    FX2_FIRMWARE,
    DeviceBinding,
    store_packages,
)
from modules import fx2_driver_step

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


ZADIG_FIRMWARE = _inf('libwdi', 'LumaScope.cat', '03/04/2025', FX2_FIRMWARE)
OTHER_VENDOR_BOOTLOADER = _inf('Acme', 'acme.cat', '05/06/2025', FX2_BOOTLOADER)


class _Machine:
    """A stand-in by design: the Windows driver store, which no test host has."""

    def __init__(self, store: pathlib.Path, *, accepts_ours: bool = True):
        self.store = store
        self.accepts_ours = accepts_ours
        self.devices: list[DeviceBinding] = []
        self.commands: list[tuple[str, ...]] = []
        self.trusted: list[str] = []

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
        if args[0] == 'certutil':
            if args[1] == '-addstore':
                self.trusted.append(
                    hashlib.sha1(pathlib.Path(args[3]).read_bytes()).hexdigest().upper()
                )
            else:
                self.trusted = [t for t in self.trusted if t != args[3]]
            return 0, ''
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
            (self.store / rest[0]).unlink()
            self.devices = [
                dataclasses.replace(d, inf=None, service=None, provider=None)
                if d.inf == rest[0]
                else d
                for d in self.devices
            ]
            return 0, ''
        if command == '/export-driver':
            shutil.copy(self.store / rest[0], pathlib.Path(rest[1]) / rest[0])
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


def test_a_zadig_bound_scope_ends_on_our_package_with_zadigs_kept_for_uninstall(tmp_path):
    machine = _machine(tmp_path)
    zadig = machine.publish(ZADIG_FIRMWARE)
    machine.plug(FX2_FIRMWARE, inf=zadig)
    package_dir = _package_dir(tmp_path)

    complete = fx2_driver_step.install(machine, package_dir)

    assert complete is True
    assert _bound(machine, FX2_FIRMWARE) == [(True, 'Etaluma')]
    assert [p.provider for p in machine.store_packages(FX2_FIRMWARE)] == ['Etaluma']
    assert (package_dir / 'removed' / zadig / zadig).exists()


def test_a_package_that_cannot_enter_the_store_removes_nothing(tmp_path):
    machine = _machine(tmp_path, accepts_ours=False)
    zadig = machine.publish(ZADIG_FIRMWARE)
    machine.plug(FX2_FIRMWARE, inf=zadig)

    complete = fx2_driver_step.install(machine, _package_dir(tmp_path))

    assert complete is False
    assert _bound(machine, FX2_FIRMWARE) == [(True, 'libwdi')]
    assert [p.inf for p in machine.store_packages(FX2_FIRMWARE)] == [zadig]


def test_another_vendors_bootloader_package_is_left_in_the_store(tmp_path):
    machine = _machine(tmp_path)
    acme = machine.publish(OTHER_VENDOR_BOOTLOADER)

    fx2_driver_step.install(machine, _package_dir(tmp_path))

    assert acme in [p.inf for p in machine.store_packages(FX2_BOOTLOADER)]


def test_past_devices_and_our_older_package_are_removed(tmp_path):
    machine = _machine(tmp_path)
    older = machine.publish(
        _inf('Etaluma', 'Lumascope_Classic.cat', '01/02/2026', FX2_FIRMWARE, FX2_BOOTLOADER)
    )
    machine.plug(FX2_BOOTLOADER, present=False, inf=older)
    machine.plug(FX2_FIRMWARE, present=False, inf=older)

    complete = fx2_driver_step.install(machine, _package_dir(tmp_path))

    assert complete is True
    assert machine.devices == []
    assert [p.driver_ver for p in machine.store_packages(FX2_FIRMWARE)] == ['10/17/2026,4.0.0.38']


def test_a_present_device_left_on_another_package_makes_the_install_incomplete(tmp_path):
    machine = _machine(tmp_path)
    # A package dated after the build outranks ours on the bootloader, where
    # the step removes nothing.
    newer = machine.publish(_inf('libwdi', 'z.cat', '12/31/2026', FX2_BOOTLOADER))
    machine.plug(FX2_BOOTLOADER, inf=newer)

    complete = fx2_driver_step.install(machine, _package_dir(tmp_path))

    assert complete is False
    assert _bound(machine, FX2_BOOTLOADER) == [(True, 'libwdi')]


def test_uninstall_removes_ours_and_restores_what_install_removed(tmp_path):
    machine = _machine(tmp_path)
    machine.plug(FX2_FIRMWARE, inf=machine.publish(ZADIG_FIRMWARE))
    package_dir = _package_dir(tmp_path)
    fx2_driver_step.install(machine, package_dir)

    complete = fx2_driver_step.uninstall(machine, package_dir)

    assert complete is True
    assert [p.provider for p in machine.store_packages(FX2_FIRMWARE)] == ['libwdi']
    assert _bound(machine, FX2_FIRMWARE) == [(True, 'libwdi')]
    assert not (package_dir / 'removed').exists()


def test_the_signing_certificate_is_trusted_once_and_removed_at_uninstall(tmp_path):
    machine = _machine(tmp_path)
    package_dir = _package_dir(tmp_path)
    (package_dir / fx2_driver_step.SIGNING_CERTIFICATE).write_bytes(b'DER bytes')

    fx2_driver_step.install(machine, package_dir)
    fx2_driver_step.install(machine, package_dir)
    recorded = (package_dir / fx2_driver_step.TRUSTED_RECORD).read_text(encoding='ascii').split()
    fx2_driver_step.uninstall(machine, package_dir)

    assert len(recorded) == 1
    assert machine.trusted == []
    assert not (package_dir / fx2_driver_step.TRUSTED_RECORD).exists()
