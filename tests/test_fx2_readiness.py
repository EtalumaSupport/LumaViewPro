# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The FX2 readiness report the installers print is the driver's own gate.

``scripts/install_mac.sh`` decides whether to install the native libusb,
and tells the user whether an LS560/LS620/LS720 will connect, from
``fx2driver.fx2_readiness()`` and ``fx2_readiness_line()``. The gate is
computed at import, so each case sets the module's term constants (and
the gate they produce) rather than the imports.
"""

from __future__ import annotations

import pytest

from drivers import fx2driver

TERMS = ('_HAS_USB', '_HAS_USB_BACKEND', '_HAS_USB1')


def _set_gate(monkeypatch, platform: str, **terms: bool) -> None:
    monkeypatch.setattr(fx2driver.sys, 'platform', platform)
    for name in TERMS:
        monkeypatch.setattr(fx2driver, name, terms.get(name, True))
    available = (
        fx2driver._HAS_USB
        and fx2driver._HAS_USB_BACKEND
        and (platform == 'win32' or fx2driver._HAS_USB1)
    )
    monkeypatch.setattr(fx2driver, '_FX2_AVAILABLE', available)


def test_all_present_on_macos_reads_ready(monkeypatch):
    _set_gate(monkeypatch, 'darwin')

    assert fx2driver.fx2_readiness() == {'pyusb': True, 'libusb-1.0': True, 'libusb1': True}
    assert fx2driver.fx2_readiness_line() == (
        'FX2 (LS560/LS620/LS720) support: ready -- '
        'pyusb present, libusb-1.0 present, libusb1 present'
    )


@pytest.mark.parametrize(
    ('term', 'key'),
    [('_HAS_USB', 'pyusb'), ('_HAS_USB_BACKEND', 'libusb-1.0'), ('_HAS_USB1', 'libusb1')],
)
def test_each_missing_term_is_named_and_not_ready_on_macos(monkeypatch, term, key):
    _set_gate(monkeypatch, 'darwin', **{term: False})

    readiness = fx2driver.fx2_readiness()
    assert readiness[key] is False
    assert all(present for name, present in readiness.items() if name != key)
    line = fx2driver.fx2_readiness_line()
    assert line.startswith('FX2 (LS560/LS620/LS720) support: NOT ready -- ')
    assert f'{key} missing' in line


def test_libusb1_is_not_needed_on_windows(monkeypatch):
    _set_gate(monkeypatch, 'win32', _HAS_USB1=False)

    assert fx2driver.fx2_readiness()['libusb1'] is None
    assert fx2driver.fx2_readiness_line() == (
        'FX2 (LS560/LS620/LS720) support: ready -- '
        'pyusb present, libusb-1.0 present, libusb1 not needed'
    )
