# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Which driver Windows binds to a USB identity, and which packages could.

Windows binds each USB identity (``USB\\VID_04B4&PID_EA17``) on its own, to
the best-ranked package in its driver store that names it. Two readers, both
without elevation:

- ``_lumascope.drivers.winusb_iso.device_bindings`` reads each device of an identity,
  present or seen before, and the driver bound to it, through SetupAPI's
  one walk of the USB devices, and ``binding_from`` here decides the
  binding from what the walk read, so it runs on any host;
- ``store_packages`` here reads the store's published copies of every
  third-party INF (``%WINDIR%\\INF\\oem*.inf``) for the packages that name the
  identity, which is what the driver step removes, and needs no
  Windows call, so it runs on any host.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar

from _lumascope.drivers.exceptions import CameraUsbDriverUnusableError, CameraUsbLookupFailedError
from _lumascope.lib.lumascope_error import RefusalCause

# The USB vendors a Lumascope is built from, by vendor ID as Windows writes
# it in a hardware ID: the Classic's FX2, the cameras, the EL-0940 boards.
CENSUS_VENDORS = {
    '04B4': 'Cypress (the Lumascope Classic)',
    '2676': 'Basler',
    '1409': 'IDS',
    '0424': 'Microchip (the EL-0940 LED board)',
    '2E8A': 'Raspberry Pi (the EL-0940 motor board)',
}
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
    behind it. ``location`` is the port and hub Windows names
    (``Port_#0002.Hub_#0001``), ``parent`` the instance ID of the hub it is
    on, and ``interface_guids`` the device interface GUIDs its bound
    package registered for it (a wrong one fails the bind with code 31);
    each None when Windows has none. ``unreadable`` names each property
    Windows gave in a type this does not read, with that type: a fact for
    the log only, so the device is still read.
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
    location: str | None
    parent: str | None
    interface_guids: tuple[str, ...] | None
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
        if self.location is not None:
            problem += f', at {self.location}'
        if self.parent is not None:
            problem += f' on {self.parent}'
        if self.interface_guids is not None:
            problem += f', interface GUIDs {" ".join(self.interface_guids) or "none"}'
        if self.unreadable:
            problem += f', unreadable: {"; ".join(self.unreadable)}'
        return f'{self.instance_id} ({where}): {driver}{problem}'


@dataclass(frozen=True)
class DeviceProperties:
    """One USB device as Windows' device list gives it, before any reader decides on it.

    ``values`` holds each property Windows gave, by name (``present``,
    ``service``, ``provider``, ``inf``, ``version``, ``date``,
    ``problem_code``, ``problem_status``, ``location``, ``parent``,
    ``address``, the port number on its hub, ``interface_guids``); a name
    absent is a property the device does not have, as a device with no
    driver has no service. ``unreadable`` holds
    each property Windows gave in a type LumaViewPro does not read, by name,
    with that type.
    """

    instance_id: str
    values: Mapping[str, object]
    unreadable: Mapping[str, str]


# The properties a binding is decided by: one unreadable fails the read,
# since whether the device is usable is then not known. Every other property
# is for the log, and one unreadable is recorded on its device, named by its
# words here or else by its name.
_DECIDING = ('present', 'service', 'provider', 'inf', 'version', 'date', 'problem_code')
_WORDS = {
    'problem_status': 'problem status',
    'interface_guids': 'interface GUIDs',
    'address': 'port number',
}


class _PropertyTypeUnreadError(CameraUsbLookupFailedError):
    """Windows gave a property a binding is decided by in a type LumaViewPro does not read."""

    causes: ClassVar[dict[str, RefusalCause | None]] = {'camera_usb_property_unread': None}
    title = 'Camera USB Property Unread'


def binding_from(device: DeviceProperties, ours: set[str]) -> DeviceBinding:
    """The device's binding, from the properties Windows gave.

    Args:
        device: The device, as the walk of Windows' device list read it.
        ours: The published INF names (``oem12.inf``), lower case, of our
            packages in the driver store.

    Raises:
        CameraUsbLookupFailedError: A property the binding is decided by came
            in a type this does not read.
    """
    deciding = {name: why for name, why in device.unreadable.items() if name in _DECIDING}
    if deciding:
        raise _PropertyTypeUnreadError(
            f'{device.instance_id}: SetupDiGetDevicePropertyW returned '
            + ', '.join(f'{name} as {why}' for name, why in deciding.items())
            + ', not a type this reads'
        )
    values = device.values
    inf = values.get('inf')
    return DeviceBinding(
        instance_id=device.instance_id,
        present=bool(values.get('present')),
        service=values.get('service'),
        provider=values.get('provider'),
        inf=inf,
        version=values.get('version'),
        date=values.get('date'),
        problem_code=values.get('problem_code'),
        problem_status=values.get('problem_status'),
        location=values.get('location'),
        parent=values.get('parent'),
        interface_guids=values.get('interface_guids'),
        ours=(inf or '').lower() in ours,
        unreadable=tuple(
            f'{_WORDS.get(name, name)} ({why})' for name, why in device.unreadable.items()
        ),
    )


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


# How much of the driver install log a support report carries: its newest
# sections that name a census vendor, each cut to its first lines, read from
# the log's last bytes (the log only grows, and holds every device's
# installs since Windows was installed).
SETUPAPI_LOG = 'setupapi.dev.log'
SETUPAPI_SECTIONS = 20
SETUPAPI_SECTION_LINES = 200
SETUPAPI_TAIL_BYTES = 16 * 1024 * 1024


def setupapi_log_excerpt(inf_folder: str) -> str:
    """The newest sections of Windows' driver install log that name a census vendor's device.

    Windows writes every driver install to ``%WINDIR%\\INF\\setupapi.dev.log``,
    one section per install, each opening ``>>>  [``: why a package bound,
    did not, or bound and failed to start is there and nowhere else.

    Args:
        inf_folder: The store's INF folder (``store_inf_folder()``).

    Returns:
        str: The sections, oldest first, or a line saying none names a
            census vendor, or why the log could not be read.
    """
    path = os.path.join(inf_folder, SETUPAPI_LOG)
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - SETUPAPI_TAIL_BYTES))
            text = f.read().decode('utf-8', errors='replace')
    except OSError as e:
        return f'{path} unreadable: {e}'
    sections = census_sections(text)
    if not sections:
        return f"No section of {path}'s last {SETUPAPI_TAIL_BYTES} bytes names a census vendor."
    return '\n'.join(sections)


def census_sections(text: str) -> list[str]:
    """The newest ``SETUPAPI_SECTIONS`` sections of a setupapi log naming a census vendor.

    A section naming ``VID_xxxx`` of any vendor in ``CENSUS_VENDORS``, in
    any case, is kept, each cut to ``SETUPAPI_SECTION_LINES`` lines with the
    cut said.
    """
    wanted = re.compile('|'.join(f'VID_{vid}' for vid in CENSUS_VENDORS), re.IGNORECASE)
    sections: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith('>>>  ['):
            sections.append([line])
        elif sections:
            sections[-1].append(line)
    kept = []
    for lines in sections:
        if not wanted.search('\n'.join(lines)):
            continue
        if len(lines) > SETUPAPI_SECTION_LINES:
            cut = len(lines) - SETUPAPI_SECTION_LINES
            lines = [*lines[:SETUPAPI_SECTION_LINES], f'     ... {cut} more lines of this section']
        kept.append('\n'.join(lines))
    return kept[-SETUPAPI_SECTIONS:]


# Where Windows states what it signs drivers by, under HKEY_LOCAL_MACHINE.
_CURRENT_VERSION = r'SOFTWARE\Microsoft\Windows NT\CurrentVersion'
_SECURE_BOOT = r'SYSTEM\CurrentControlSet\Control\SecureBoot\State'
_CONTROL = r'SYSTEM\CurrentControlSet\Control'


def signing_facts(read_value: Callable[[str, str], object]) -> dict[str, str]:
    """The facts Windows decides whether to load a driver by, each in words.

    The Windows build, whether Secure Boot is on, and whether test signing
    is on (a test-signed driver loads only with it). Each fact Windows did
    not give is stated unreadable with why.

    Args:
        read_value: Reads a value under HKEY_LOCAL_MACHINE by key path and
            name (``winreg.QueryValueEx`` on an opened key), raising
            ``OSError`` when the key or the value is absent.
    """

    def read(path: str, name: str) -> tuple[object, str | None]:
        try:
            return read_value(path, name), None
        except OSError as e:
            return None, f'unreadable ({path}\\{name}: {e})'

    facts = {}
    build = []
    for name in ('ProductName', 'DisplayVersion', 'CurrentBuild', 'UBR'):
        value, why = read(_CURRENT_VERSION, name)
        build.append(f'{name} {value}' if why is None else f'{name} {why}')
    facts['Windows'] = ', '.join(build)
    enabled, why = read(_SECURE_BOOT, 'UEFISecureBootEnabled')
    facts['Secure Boot'] = why or ('on' if enabled else 'off')
    options, why = read(_CONTROL, 'SystemStartOptions')
    facts['Test signing'] = why or ('on' if 'TESTSIGNING' in str(options).upper() else 'off')
    return facts
