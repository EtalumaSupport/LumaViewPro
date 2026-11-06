# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""On macOS the IDS driver's absence is a fact of the host, not a warning.

IDS peak ships Windows and Linux builds only. Every launch on a Mac, the
simulator included, logged `IDS camera driver unavailable: No module named
'ids_peak_ipl'` at WARNING, a line that can never be otherwise there and
that buried the warnings that matter; the environment banner now states
the line the registration hands it. On Windows and Linux the WARNING
stays: a frozen build missing the IDS library once made an IDS scope look
like it had no camera.
"""

import logging
import sys

from _lumascope.modules.lumascope_api import _lumascope


def _attempt(monkeypatch, platform):
    """The registration's answer and the (level, message) lines it hands the banner."""
    # The import fails the way it does on a host without the IDS SDK.
    monkeypatch.setitem(sys.modules, '_lumascope.drivers.idscamera', None)
    lines = []
    camera = _lumascope._register_ids_camera(platform, lines)
    return camera, lines


def test_on_macos_the_absence_is_info_and_the_driver_is_not_attempted(monkeypatch):
    camera, lines = _attempt(monkeypatch, 'darwin')
    assert camera is None
    assert [m for level, m in lines if level == logging.WARNING] == [], lines
    message = [m for level, m in lines if level == logging.INFO][-1]
    assert 'not supported on macOS' in message


def test_on_windows_and_linux_a_missing_driver_is_still_a_warning(monkeypatch):
    for platform in ('win32', 'linux'):
        camera, lines = _attempt(monkeypatch, platform)
        assert camera is None
        message = [m for level, m in lines if level == logging.WARNING][-1]
        assert 'IDS camera driver unavailable' in message


def test_requirements_pin_the_ids_libraries_on_exactly_the_hosts_they_have_wheels_for():
    # With no marker, `pip install -r requirements.txt` failed on every Mac
    # before installing anything: the install scripts and the README's macOS
    # steps. Each pin's wheels are Windows AMD64 and Linux x86_64 / aarch64.
    import pathlib

    from packaging.requirements import Requirement

    lines = pathlib.Path(__file__).resolve().parents[1].joinpath('requirements.txt').read_text()
    requirements = [Requirement(raw) for raw in lines.splitlines() if raw.startswith('ids-peak')]
    assert sorted(r.name for r in requirements) == [
        'ids-peak',
        'ids-peak-afl',
        'ids-peak-icv',
        'ids-peak-ipl',
    ]
    covered = [('win32', 'AMD64'), ('linux', 'x86_64'), ('linux', 'aarch64')]
    uncovered = [('darwin', 'arm64'), ('darwin', 'x86_64'), ('linux', 'armv7l'), ('win32', 'ARM64')]
    for requirement in requirements:
        for sys_platform, machine in covered + uncovered:
            env = {'sys_platform': sys_platform, 'platform_machine': machine}
            evaluated = requirement.marker is not None and requirement.marker.evaluate(env)
            assert evaluated is ((sys_platform, machine) in covered), (requirement.name, env)
