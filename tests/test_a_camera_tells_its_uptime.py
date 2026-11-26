# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The camera's uptime is told at bring-up, read over the API, and in the support report.

A camera fault that survives a reconnect clears only on a power cycle or a
reset, and nothing said whether the camera had one. The camera's own clock
counts from its power-up, and a reset restarts it, so its latched clock is
the camera's uptime: ``diagnostics.read_camera_uptime_s`` reads it, bring-up
logs it once, and the support report writes it. A camera without a latch has
no uptime; a camera that fails the latch at bring-up is reported once, and
bring-up goes on.

A Basler USB camera is reset at every connect, which restarts its clock, so
its driver reads the clock before the reset and keeps that reading as the
connect's; bring-up then tells the uptime the camera had when LumaViewPro
connected, and that LumaViewPro reset it.
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


def _reset_at_connect(monkeypatch, kept):
    # A stand-in by design: the simulated camera is never reset at connect (a
    # simulator gap: only a Basler USB body is), so its driver answers as a
    # reset camera's whose clock read ``kept`` before the reset; the connect's
    # own half is tests/test_a_basler_usb_camera_is_reset_at_connect.py.
    monkeypatch.setattr(SimulatedCamera, 'resets_at_connect', property(lambda self: True))
    monkeypatch.setattr(SimulatedCamera, 'clock_ticks_before_reset', property(lambda self: kept))


def _uptime_lines(caplog):
    return [r.getMessage() for r in caplog.records if 'camera uptime at' in r.getMessage()]


def test_bring_up_tells_the_uptime_read_before_the_reset_at_connect(monkeypatch, tmp_path, caplog):
    _reset_at_connect(monkeypatch, 7_200_000_000_000)
    with caplog.at_level(logging.INFO):
        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        lines = _uptime_lines(caplog)
        assert len(lines) == 1, lines
        assert 'camera uptime at connect: 7200.0 s' in lines[0]
        assert 'then reset by LumaViewPro' in lines[0]
    finally:
        session.shutdown()


def test_a_reading_before_the_reset_with_no_tick_rate_is_never_turned_into_seconds(
    monkeypatch, tmp_path, caplog
):
    _reset_at_connect(monkeypatch, 7_200_000_000_000)
    # A stand-in by design: a body whose Timestamp chunk does not turn on has
    # no tick rate, which the simulated camera always has (a simulator gap).
    monkeypatch.setattr(SimulatedCamera, 'TIMESTAMP_TICK_HZ', None)
    with caplog.at_level(logging.INFO):
        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        lines = _uptime_lines(caplog)
        assert len(lines) == 1, lines
        assert 'its tick rate is unknown' in lines[0]
        assert ' s (since' not in lines[0]
    finally:
        session.shutdown()


def test_a_clock_read_that_failed_before_the_reset_is_reported_once_and_bring_up_goes_on(
    monkeypatch, tmp_path, centre_posts
):
    _reset_at_connect(
        monkeypatch, CameraReadFailedError('TimestampLatch failed: the camera did not answer')
    )
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        failed = [n for n in centre_posts if n.reason == 'camera_read_failed']
        assert len(failed) == 1, centre_posts
        assert session.scope.camera_connected
    finally:
        session.shutdown()
