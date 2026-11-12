# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Which driver Windows binds to a USB identity, and which packages could.

Windows binds each USB identity (``USB\\VID_04B4&PID_EA17``) on its own, to
the best-ranked package in its driver store that names it. Two readers, both
without elevation:

- ``_lumascope.drivers.winusb_iso.device_bindings`` reads each device of an identity,
  present or seen before, and the driver bound to it, through SetupAPI;
- ``store_packages`` here reads the store's published copies of every
  third-party INF (``%WINDIR%\\INF\\oem*.inf``) for the packages that name the
  identity, which is what the driver step removes, and needs no
  Windows call, so it runs on any host.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from _lumascope.drivers.exceptions import CameraUsbDriverUnusableError

FX2_BOOTLOADER = 'USB\\VID_04B4&PID_8613'
FX2_FIRMWARE = 'USB\\VID_04B4&PID_EA17'
# The names our INF gives each identity in Device Manager.
DEVICE_NAMES = {
    FX2_BOOTLOADER: 'Lumascope Classic (Bootloader)',
    FX2_FIRMWARE: 'Lumascope Classic',
}
# Ours is named by its catalog as well as its provider: an older Etaluma
# package (LumaView Classic's, or the unsigned LumaScope_WinUSB.inf the
# installer once carried) is another package to replace.
OUR_PROVIDER = 'Etaluma'
OUR_CATALOG = 'Lumascope_Classic.cat'


# Device Manager's problem codes, each in Microsoft's words ("Error codes in
# Device Manager in Windows"); a code not here is printed as its number.
PROBLEM_CODES = {
    10: 'This device cannot start.',
    14: 'This device cannot work properly until you restart your computer.',
    18: 'Reinstall the drivers for this device.',
    19: (
        'Windows cannot start this hardware device because its configuration information '
        '(in the registry) is incomplete or damaged.'
    ),
    21: 'Windows is removing this device.',
    22: 'This device is disabled.',
    24: (
        'This device is not present, is not working properly, or does not have all its '
        'drivers installed.'
    ),
    28: 'The drivers for this device are not installed.',
    31: (
        'This device is not working properly because Windows cannot load the drivers '
        'required for this device.'
    ),
    32: (
        'A driver (service) for this device has been disabled. An alternate driver may be '
        'providing this functionality.'
    ),
    37: 'Windows cannot initialize the device driver for this hardware.',
    39: (
        'Windows cannot load the device driver for this hardware. The driver may be '
        'corrupted or missing.'
    ),
    40: (
        'Windows cannot access this hardware because its service key information in the '
        'registry is missing or recorded incorrectly.'
    ),
    41: (
        'Windows successfully loaded the device driver for this hardware but cannot find '
        'the hardware device.'
    ),
    43: 'Windows has stopped this device because it has reported problems.',
    45: 'Currently, this hardware device is not connected to the computer.',
    47: (
        'Windows cannot use this hardware device because it has been prepared for safe '
        'removal, but it has not been removed from the computer.'
    ),
    48: (
        'The software for this device has been blocked from starting because it is known '
        'to have problems with Windows.'
    ),
    52: 'Windows cannot verify the digital signature for the drivers required for this device.',
}


@dataclass(frozen=True)
class DeviceBinding:
    """One device of a USB identity and the driver Windows has bound to it.

    ``service`` is the bound driver's service (``WinUSB``, ``libusbK``,
    ``libusb0``), None when no driver is bound; the package fields are None
    with it. ``present`` is False for a device seen before on another port or
    at another time, which keeps its binding until it is matched afresh.
    ``ours`` is whether the bound package is LumaViewPro's, as the store
    names it (``StorePackage.ours``). ``problem_code`` is Device Manager's
    problem code, None or 0 for none; ``problem_status`` the Windows status
    behind it. ``unreadable`` names each property Windows gave in a type
    this does not read, with that type: a fact for the log only, so the
    device is still read.
    """

    instance_id: str
    present: bool
    service: str | None
    provider: str | None
    inf: str | None
    version: str | None
    date: str | None
    problem_code: int | None
    problem_status: int | None
    ours: bool
    unreadable: tuple[str, ...]

    @property
    def started(self) -> bool:
        """Whether a driver is bound and Windows started it: no problem code.

        A package bound and not started (code 31, its driver not loaded)
        drives nothing, whoever's package it is.
        """
        return self.service is not None and not self.problem_code

    def describe(self) -> str:
        where = 'present' if self.present else 'not present'
        if self.service is None:
            driver = 'no driver'
        else:
            whose = "LumaViewPro's" if self.ours else "not LumaViewPro's"
            started = '' if self.started else ', not started by Windows'
            driver = (
                f'{self.service} from {self.provider} {self.inf} {self.version} dated {self.date}'
                f', {whose}{started}'
            )
        problem = ''
        if self.problem_code:
            meaning = PROBLEM_CODES.get(self.problem_code)
            problem = f', problem code {self.problem_code}' + (f' ({meaning})' if meaning else '')
        if self.problem_status:
            problem += f', problem status {self.problem_status:#010x}'
        if self.unreadable:
            problem += f', unreadable: {"; ".join(self.unreadable)}'
        return f'{self.instance_id} ({where}): {driver}{problem}'


@dataclass(frozen=True)
class StorePackage:
    """A package in the driver store whose INF names a USB identity.

    ``inf`` is the store's published name (``oem12.inf``), the name
    ``pnputil`` takes; ``driver_ver`` is the INF's ``DriverVer``, date then
    version, which ranks it against other packages of equal signature;
    ``hardware_ids`` is every ID its models name, upper case, each once:
    removing the package takes it from each of them.
    """

    inf: str
    provider: str
    catalog: str
    driver_ver: str
    hardware_ids: tuple[str, ...]

    @property
    def ours(self) -> bool:
        return self.provider == OUR_PROVIDER and self.catalog.lower() == OUR_CATALOG.lower()


def unusable(bindings: list[DeviceBinding]) -> list[DeviceBinding]:
    """The present devices not on LumaViewPro's package started by Windows.

    Ours alone, on both identities, whatever else could drive the device: a
    package installed after ours that takes an identity back is refused
    here and taken back by Repair, so ours always wins. A device with no
    driver is one, and so is one on ours that Windows did not start. A past
    device is not: it is matched afresh when it is plugged in.
    """
    return [b for b in bindings if b.present and not (b.ours and b.started)]


def refuse_unusable(bindings: dict[str, list[DeviceBinding]]) -> None:
    """Refuse the FX2 when a present device of either identity is not on our package, started.

    Args:
        bindings: Each identity's devices, as ``device_bindings`` reads them.

    Raises:
        CameraUsbDriverUnusableError: Naming each such device, the driver it
            is bound to, Windows' problem code, and the fix.
    """
    refused = [
        f'{DEVICE_NAMES[identity]} {b.describe()}'
        for identity, devices in bindings.items()
        for b in unusable(devices)
    ]
    if refused:
        raise CameraUsbDriverUnusableError(
            '; '.join(refused)
            + "; install LumaViewPro's Lumascope Classic driver, so Windows binds it and starts it"
        )


def store_inf_folder() -> str:
    return os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'INF')


def store_packages(hardware_id: str, inf_folder: str) -> list[StorePackage]:
    """Every published third-party package whose INF names ``hardware_id``.

    Args:
        hardware_id: ``USB\\VID_xxxx&PID_xxxx``, matched as Windows matches
            it, without regard to case.
        inf_folder: The store's INF folder (``store_inf_folder()``).
    """
    wanted = hardware_id.upper()
    packages = []
    for name in sorted(os.listdir(inf_folder)):
        if not re.fullmatch(r'oem\d+\.inf', name, re.IGNORECASE):
            continue
        package = _package(name, _inf_sections(_read_inf_text(os.path.join(inf_folder, name))))
        if wanted in package.hardware_ids:
            packages.append(package)
    return packages


def read_package(path: str) -> StorePackage:
    """The package an INF file describes, named for its file."""
    return _package(os.path.basename(path), _inf_sections(_read_inf_text(path)))


def _package(name: str, sections: dict[str, list[tuple[str, str]]]) -> StorePackage:
    version = _first_values(sections.get('version', []))
    strings = _first_values(sections.get('strings', []))
    catalog = version.get('catalogfile', version.get('catalogfile.ntamd64', ''))
    return StorePackage(
        inf=name,
        provider=_resolve(version.get('provider', ''), strings),
        catalog=catalog.strip('"'),
        driver_ver=version.get('driverver', ''),
        hardware_ids=_hardware_ids(sections),
    )


def _read_inf_text(path: str) -> str:
    """An INF is UTF-16 with a byte-order mark, or 8-bit text."""
    with open(path, 'rb') as f:
        raw = f.read()
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        return raw.decode('utf-16')
    if raw.startswith(b'\xef\xbb\xbf'):
        return raw[3:].decode('utf-8', errors='replace')
    return raw.decode('mbcs' if os.name == 'nt' else 'latin-1', errors='replace')


def _inf_sections(text: str) -> dict[str, list[tuple[str, str]]]:
    """Section name (lower case) to its ``key = value`` lines, in order.

    Every line is kept: a models section names one device for several IDs
    under the same key. A model line's value is
    ``install-section, hardware-id[, compatible-id...]``.
    """
    sections: dict[str, list[tuple[str, str]]] = {}
    current: list[tuple[str, str]] | None = None
    for line in text.splitlines():
        line = line.split(';', 1)[0].strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith(']'):
            current = sections.setdefault(line[1:-1].strip().lower(), [])
        elif current is not None and '=' in line:
            key, value = line.split('=', 1)
            current.append((key.strip().lower(), value.strip()))
    return sections


def _first_values(lines: list[tuple[str, str]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in lines:
        values.setdefault(key, value)
    return values


def _hardware_ids(sections: dict[str, list[tuple[str, str]]]) -> tuple[str, ...]:
    """Every ID the INF's models sections list, upper case, each once."""
    model_sections = set()
    for _, value in sections.get('manufacturer', []):
        parts = [part.strip() for part in value.split(',')]
        base = parts[0].lower()
        model_sections.add(base)
        model_sections.update(f'{base}.{decoration.lower()}' for decoration in parts[1:])
    ids: dict[str, None] = {}
    for section in sorted(model_sections):
        for _, value in sections.get(section, []):
            ids.update(dict.fromkeys(part.strip().upper() for part in value.split(',')[1:]))
    return tuple(ids)


def _resolve(value: str, strings: dict[str, str]) -> str:
    """``%Name%`` from the INF's [Strings], quotes dropped on both sides,
    since Zadig writes the reference itself in quotes."""
    value = value.strip().strip('"')
    if value.startswith('%') and value.endswith('%'):
        value = strings.get(value[1:-1].lower(), value)
    return value.strip('"')
