# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The camera's uptime is told at bring-up, read over the API, and in the support report.

A camera fault that survives a reconnect clears only on a power cycle or a
reset, and nothing said whether the camera had one. The camera's own clock
counts from its power-up, and a reset restarts it, so its latched clock is
the camera's uptime: ``diagnostics.read_camera_uptime_s`` reads it, bring-up
logs it once, and the support report writes it. A camera without a latch has
no uptime; a camera that fails the latch at bring-up is reported once, and
bring-up goes on.
"""

from __future__ import annotations

import json
import logging

from _lumascope.drivers import host_clock
from _lumascope.drivers.exceptions import CameraReadFailedError
from _lumascope.drivers.simulated_camera import SimulatedCamera
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import build_scope
from tests.settings_fixtures import complete_settings


def test_the_uptime_is_the_time_since_the_camera_was_powered_up():
    before_ns = host_clock.steady_ns()
    scope = build_scope(simulate=True, sim_model='LS850')
    try:
        uptime_s = scope.diagnostics.read_camera_uptime_s()
        elapsed_s = (host_clock.steady_ns() - before_ns) / 1e9

        # The simulated camera powers up as it is built, inside the bracket.
        assert 0.0 <= uptime_s <= elapsed_s
        assert scope.diagnostics.read_camera_uptime_s() >= uptime_s
    finally:
        scope.disconnect()


def test_a_camera_without_a_clock_latch_has_no_uptime(monkeypatch):
    # A stand-in by design: the simulator has no camera without a latch (a
    # simulator gap), so its probe answers as an IDS body's does.
    monkeypatch.setattr(SimulatedCamera, 'supports_clock_latch', lambda self: False)
    scope = build_scope(simulate=True, sim_model='LS850')
    try:
        assert scope.capabilities.camera_has_clock_latch is False
        assert scope.diagnostics.read_camera_uptime_s() is None
    finally:
        scope.disconnect()


def test_bring_up_logs_the_cameras_uptime_once(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        lines = [r.getMessage() for r in caplog.records if 'camera uptime at bring-up' in r.message]
        assert len(lines) == 1, lines
        assert 'since its last power-up or reset' in lines[0]
    finally:
        session.shutdown()


def test_a_latch_that_fails_at_bring_up_is_reported_once_and_bring_up_goes_on(
    monkeypatch, tmp_path, centre_posts
):
    # A stand-in by design: the simulated camera's latch cannot be made to
    # fail (a simulator gap).
    def fail(self):
        raise CameraReadFailedError('TimestampLatch failed: the camera did not answer')

    monkeypatch.setattr(SimulatedCamera, 'latch_clock', fail)
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        failed = [n for n in centre_posts if n.reason == 'camera_read_failed']
        assert len(failed) == 1, centre_posts
        assert session.scope.camera_connected
    finally:
        session.shutdown()


def test_the_support_report_writes_the_cameras_uptime(sim_session, tmp_path):
    from _lumascope.modules.tech_support_report import TechSupportReport

    TechSupportReport(sim_session)._step_camera_diagnostics(tmp_path)

    info = json.loads((tmp_path / 'camera_info' / 'camera_info.json').read_text())
    assert isinstance(info['uptime_s'], float)
    assert info['uptime_s'] >= 0.0
