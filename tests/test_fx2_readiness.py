# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The FX2 readiness report the installers print is the driver's own gate.

``scripts/install_mac.sh`` tells the user whether an LS460/LS560/LS620/LS720
will connect, from
``fx2driver.fx2_readiness()`` and ``fx2_readiness_line()``. The gate is
decided at import, so each case sets the module's gate record, with the
terms it names, rather than the imports; ``_fx2_gate`` itself is pinned
in ``test_fx2_driver.py``.
"""

from __future__ import annotations

from _lumascope.drivers import fx2driver


def _set_gate(monkeypatch, refusal: str | None = None) -> None:
    """The gate as import would decide it: in use unless ``refusal`` says why not."""
    monkeypatch.setattr(
        fx2driver,
        '_FX2_GATE',
        fx2driver._Fx2Gate(
            library_path='/bundled/libusb-1.0.dylib' if refusal is None else None,
            library_refusal=refusal,
            library_version=None,
        ),
    )


def test_the_bundled_library_in_use_reads_ready(monkeypatch):
    _set_gate(monkeypatch)

    assert fx2driver.fx2_readiness() == {'libusb-package': True}
    assert fx2driver.fx2_readiness_line() == (
        'FX2 (LS460/LS560/LS620/LS720) support: ready -- libusb-package present'
    )


def test_a_refused_bundled_library_reads_not_ready_and_says_why(monkeypatch):
    reason = 'another libusb was loaded before the bundled one: /opt/homebrew/lib/libusb-1.0.dylib'
    _set_gate(monkeypatch, refusal=reason)

    assert fx2driver.fx2_readiness() == {'libusb-package': False}
    assert fx2driver.fx2_readiness_line() == (
        f'FX2 (LS460/LS560/LS620/LS720) support: NOT ready -- libusb-package missing ({reason})'
    )


def test_requirements_pin_the_bundled_libusb_on_exactly_the_hosts_it_has_wheels_for():
    # The pin is the FX2 driver's only source of libusb; the marker keeps
    # `pip install -r requirements.txt` working on hosts the wheel does not
    # cover (32-bit ARM Linux, Windows on ARM), where the gate reports why.
    from pathlib import Path

    # packaging is pytest's own dependency, so it is present wherever this runs.
    from packaging.requirements import Requirement

    lines = Path(fx2driver.__file__).resolve().parents[3].joinpath('requirements.txt').read_text()
    (line,) = [raw for raw in lines.splitlines() if raw.startswith('libusb-package')]
    requirement = Requirement(line)
    assert str(requirement.specifier) == '==1.0.30.0'
    covered = [
        ('darwin', 'arm64'),
        ('darwin', 'x86_64'),
        ('linux', 'x86_64'),
        ('linux', 'aarch64'),
        ('win32', 'AMD64'),
        ('win32', 'x86'),
    ]
    uncovered = [('linux', 'armv7l'), ('win32', 'ARM64')]
    for sys_platform, machine in covered + uncovered:
        env = {'sys_platform': sys_platform, 'platform_machine': machine}
        assert requirement.marker.evaluate(env) is ((sys_platform, machine) in covered), env
