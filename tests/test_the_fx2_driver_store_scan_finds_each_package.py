# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The driver-store scan finds every package that names a USB identity, and
tells ours from every other.

Windows binds an identity to the best-ranked package in its store that names
it, so the installer's driver step removes the others for the FX2's running
firmware and keeps ours. The scan reads the store's published INF copies
(``oem*.inf``) as text, which needs no Windows call, so it runs here on real
INFs: the signed one every build carries, and one shaped as Zadig writes it
(UTF-16, the provider through a string, one device name for two IDs).
"""

from __future__ import annotations

import pathlib
import re

from _lumascope.drivers.usb_driver_binding import FX2_BOOTLOADER, FX2_FIRMWARE, store_packages

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SHIPPED_INF = (
    REPO_ROOT / 'scripts' / 'appBuild' / 'build_exe' / 'fx2_driver' / 'Lumascope_Classic.inf'
)

ZADIG_SHAPED_INF = """\
[Version]
Signature   = "$Windows NT$"
Class       = "Universal Serial Bus devices"
Provider    = "%ProviderName%"
CatalogFile = "LumaScope.cat"
DriverVer   = 03/04/2025, 6.1.7600.16385

[Manufacturer]
%ProviderName% = Devices, NTamd64

[Devices.NTamd64]
; One name for both IDs, as a single key repeated.
%DeviceName% = USB_Install, USB\\VID_04B4&PID_8613
%DeviceName% = USB_Install, USB\\vid_04b4&pid_ea17

[Strings]
ProviderName = "libwdi"
DeviceName   = "LumaScope"
"""


def _shipped_driver_ver() -> str:
    return re.search(r'^DriverVer\s*=\s*(\S+)', SHIPPED_INF.read_text('ascii'), re.MULTILINE)[1]


def _store(tmp_path) -> pathlib.Path:
    store = tmp_path / 'INF'
    store.mkdir()
    (store / 'oem3.inf').write_bytes(SHIPPED_INF.read_bytes())
    (store / 'oem9.inf').write_bytes(ZADIG_SHAPED_INF.encode('utf-16'))
    (store / 'oem12.inf').write_text(
        '[Version]\nProvider = Logitech\n[Manufacturer]\nLogi = M, NTamd64\n'
        '[M.NTamd64]\nMouse = I, USB\\VID_046D&PID_C52B\n',
        encoding='ascii',
    )
    # The store's own INFs, which are not published packages, are not read.
    (store / 'winusb.inf').write_text(ZADIG_SHAPED_INF, encoding='ascii')
    return store


def test_each_identity_finds_ours_and_the_zadig_package_and_no_other(tmp_path):
    store = _store(tmp_path)

    found = {
        identity: sorted(
            (p.inf, p.provider, p.driver_ver, p.ours) for p in store_packages(identity, str(store))
        )
        for identity in (FX2_FIRMWARE, FX2_BOOTLOADER)
    }

    expected = [
        ('oem3.inf', 'Etaluma', _shipped_driver_ver(), True),
        ('oem9.inf', 'libwdi', '03/04/2025, 6.1.7600.16385', False),
    ]
    assert found == {FX2_FIRMWARE: expected, FX2_BOOTLOADER: expected}


def test_an_older_etaluma_package_with_another_catalog_is_not_ours(tmp_path):
    store = tmp_path / 'INF'
    store.mkdir()
    # The unsigned INF the installer once carried: Etaluma's, another package.
    older = SHIPPED_INF.read_text(encoding='ascii').replace(
        'Lumascope_Classic.cat', 'LumaScope_WinUSB.cat'
    )
    (store / 'oem5.inf').write_text(older, encoding='ascii')

    (package,) = store_packages(FX2_FIRMWARE, str(store))

    assert (package.provider, package.ours) == ('Etaluma', False)


def test_a_quoted_catalog_name_is_read_without_its_quotes(tmp_path):
    store = tmp_path / 'INF'
    store.mkdir()
    quoted = SHIPPED_INF.read_text(encoding='ascii').replace(
        'CatalogFile = Lumascope_Classic.cat', 'CatalogFile = "Lumascope_Classic.cat"'
    )
    (store / 'oem4.inf').write_text(quoted, encoding='ascii')

    (package,) = store_packages(FX2_FIRMWARE, str(store))

    assert (package.catalog, package.ours) == ('Lumascope_Classic.cat', True)
