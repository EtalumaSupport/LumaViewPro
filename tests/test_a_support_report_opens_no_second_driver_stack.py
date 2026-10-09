# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A support report reaches the scope only through its own session.

The report's hardware-free steps ended by running
``tests/test_hardware_serial.py --run-hardware`` in a child pytest, which
built its own LED and motor boards on whatever ports were attached, outside
the scope's claim: beside a running LumaViewPro it shared the port (macOS) or
could not open it (Windows), and a report made on a simulated session drove
the real scope on the desk -- the test suite cycled a bench scope's LEDs that
way. The report's serial latency is its own step, on the session's drivers.
"""

from __future__ import annotations

import subprocess

import pytest

from modules import tech_support_report
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield s
    s.shutdown()


@pytest.mark.slow
def test_a_full_report_starts_no_pytest(session, tmp_path, monkeypatch):
    started = []
    real_run = subprocess.run

    def run(args, *a, **k):
        argv = [str(arg) for arg in args] if isinstance(args, (list, tuple)) else [str(args)]
        started.append(argv)
        # Recorded, never run: the child it would start opens real ports.
        if 'pytest' in argv:
            return subprocess.CompletedProcess(argv, 0, '', '')
        return real_run(args, *a, **k)

    monkeypatch.setattr(tech_support_report.subprocess, 'run', run)
    # The USB inventory lists the ports the OS knows and opens none; the
    # suite refuses even the listing, so it is answered empty here.
    monkeypatch.setattr(tech_support_report, '_collect_usb_devices', lambda: [])

    session.make_support_report(output_dir=tmp_path / 'out')

    assert not [args for args in started if 'pytest' in args or '--run-hardware' in args]
