# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Which driver Windows binds to a USB identity, and which packages could.

Windows binds each USB identity (``USB\\VID_04B4&PID_EA17``) on its own, to
the best-ranked package in its driver store that names it. Two readers, both
without elevation:

- ``drivers.winusb_iso.device_bindings`` reads each device of an identity,
  present or seen before, and the driver bound to it, through SetupAPI;
- ``store_packages`` here reads the store's published copies of every
  third-party INF (``%WINDIR%\\INF\\oem*.inf``) for the packages that name the
  identity, which is what the installer exports and removes, and needs no
  Windows call, so it runs on any host.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

FX2_BOOTLOADER = 'USB\\VID_04B4&PID_8613'
FX2_FIRMWARE = 'USB\\VID_04B4&PID_EA17'
# Ours is named by its catalog as well as its provider: an older Etaluma
# package (LumaView Classic's, or the unsigned LumaScope_WinUSB.inf the
# installer once carried) is another package to replace.
OUR_PROVIDER = 'Etaluma'
OUR_CATALOG = 'Lumascope_Classic.cat'


@dataclass(frozen=True)
class DeviceBinding:
    """One device of a USB identity and the driver Windows has bound to it.

    ``service`` is the bound driver's service (``WinUSB``, ``libusbK``,
    ``libusb0``), None when no driver is bound; the package fields are None
    with it. ``present`` is False for a device seen before on another port or
    at another time, which keeps its binding until it is matched afresh.
    """

    instance_id: str
    present: bool
    service: str | None
    provider: str | None
    inf: str | None
    version: str | None
    date: str | None
    problem_code: int | None

    def describe(self) -> str:
        where = 'present' if self.present else 'not present'
        if self.service is None:
            driver = 'no driver'
        else:
            driver = (
                f'{self.service} from {self.provider} {self.inf} {self.version} dated {self.date}'
            )
        problem = f', problem code {self.problem_code}' if self.problem_code else ''
        return f'{self.instance_id} ({where}): {driver}{problem}'


@dataclass(frozen=True)
class StorePackage:
    """A package in the driver store whose INF names a USB identity.

    ``inf`` is the store's published name (``oem12.inf``), the name
    ``pnputil`` takes; ``driver_ver`` is the INF's ``DriverVer``, date then
    version, which ranks it against other packages of equal signature.
    """

    inf: str
    provider: str
    catalog: str
    driver_ver: str

    @property
    def ours(self) -> bool:
        return self.provider == OUR_PROVIDER and self.catalog.lower() == OUR_CATALOG.lower()


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
        sections = _inf_sections(_read_inf_text(os.path.join(inf_folder, name)))
        if not _names_hardware_id(sections, wanted):
            continue
        version = _first_values(sections.get('version', []))
        packages.append(
            StorePackage(
                inf=name,
                provider=_resolve(
                    version.get('provider', ''), _first_values(sections.get('strings', []))
                ),
                catalog=version.get('catalogfile', version.get('catalogfile.ntamd64', '')),
                driver_ver=version.get('driverver', ''),
            )
        )
    return packages


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


def _names_hardware_id(sections: dict[str, list[tuple[str, str]]], wanted: str) -> bool:
    """Whether a models section of the INF lists ``wanted`` among its IDs."""
    model_sections = set()
    for _, value in sections.get('manufacturer', []):
        parts = [part.strip() for part in value.split(',')]
        base = parts[0].lower()
        model_sections.add(base)
        model_sections.update(f'{base}.{decoration.lower()}' for decoration in parts[1:])
    for section in model_sections:
        for _, value in sections.get(section, []):
            ids = [part.strip().upper() for part in value.split(',')[1:]]
            if wanted in ids:
                return True
    return False


def _resolve(value: str, strings: dict[str, str]) -> str:
    """``%Name%`` from the INF's [Strings], quotes dropped on both sides,
    since Zadig writes the reference itself in quotes."""
    value = value.strip().strip('"')
    if value.startswith('%') and value.endswith('%'):
        value = strings.get(value[1:-1].lower(), value)
    return value.strip('"')
