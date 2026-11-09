# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A grab-state read that fails is told as a failed read, never as "not streaming".

``is_grabbing`` answered False for any SDK error on Pylon, so a config write,
a capture's refusal, a run's admission and ``status.camera_streaming`` took a
read that failed for a camera that was not grabbing; on IDS the SDK's own
exception got out, and at bring-up's last line tore the session down. Now
the read raises ``CameraReadFailedError`` on both drivers, and each caller
answers it as itself: a config write is refused as not sent
(``'camera_read_failed'``), ``status`` carries ``None`` with the failure
reported once, the 1 Hz stream watch reads the grab the driver wants and
asks the camera nothing, and bring-up comes up saying the streaming state
is unknown.

Each case runs on the simulated scope through its ``grabbing`` read fault.
"""

from __future__ import annotations

import logging

import pytest

from _lumascope.drivers.exceptions import CameraReadFailedError
from _lumascope.drivers.simulated_camera import SimulatedCamera
from _lumascope.modules.exceptions import CameraSettingRejected, RunCheckFailedError
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

# Reads enough for any one case below: the grab state is read by every
# config write, the watch and each status ask.
_MANY = 1000


@pytest.fixture
def session():
    s = ScopeSession.create(complete_settings(), simulate=True, warn_pre_release=False)
    yield s
    s.shutdown()


@pytest.fixture
def reported(monkeypatch):
    calls = []
    monkeypatch.setattr(
        '_lumascope.modules.notification_center.notifications.report_outcome',
        lambda exc, **kw: calls.append((exc, kw)),
    )
    return calls


def _grabbing_fails(session, count=_MANY):
    session.scope.simulation.refuse_camera_read('grabbing', 'failed', count=count)


def test_status_carries_none_for_the_streaming_state_and_reports_the_read_once(session, reported):
    _grabbing_fails(session)

    status = session.status

    assert status.camera_streaming is None
    failed = [(exc, kw) for exc, kw in reported if isinstance(exc, CameraReadFailedError)]
    assert len(failed) == 1
    _exc, kw = failed[0]
    assert kw.get('log_only') is True and kw.get('solicited') is False


def test_rest_status_carries_null_for_a_grab_state_it_could_not_read(session):
    from fastapi.testclient import TestClient

    from _lumascope.rest.app import build_app

    _grabbing_fails(session)
    with TestClient(build_app(session)) as client:
        answer = client.get('/api/v1/status')

    assert answer.status_code == 200
    assert answer.json()['camera_streaming'] is None


def test_a_config_write_whose_grab_state_read_fails_is_refused_as_not_sent(session):
    driver = session.scope._camera_driver
    _grabbing_fails(session, count=1)

    with pytest.raises(CameraSettingRejected) as refused:
        session.scope.camera.set_binning(2)

    assert refused.value.reason == 'camera_read_failed', refused.value
    assert isinstance(refused.value.__cause__, CameraReadFailedError)
    assert driver.get_binning_size() == 1, 'the write was sent'
    assert session.scope.camera.binning == 1


def test_the_stream_watch_asks_the_camera_nothing_and_reports_nothing(session, caplog):
    _grabbing_fails(session)
    with caplog.at_level(logging.WARNING):
        session.scope.imaging._check_stream()

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_capture_whose_grab_state_read_fails_raises_the_read():
    """A pin: the capture's caller already met the read's raise as itself on
    a driver that raised; it is never the ``camera_not_streaming`` refusal."""
    s = ScopeSession.create(complete_settings(), simulate=True, warn_pre_release=False)
    try:
        s.scope.simulation.refuse_camera_read('grabbing', 'failed', count=_MANY)
        with pytest.raises(CameraReadFailedError):
            s.scope.imaging.capture_still(timeout_s=5.0)
    finally:
        s.scope._camera_driver.allow_read('grabbing')
        s.shutdown()


def test_a_run_whose_grab_state_read_fails_is_refused_as_state_unknown(tmp_path):
    """A pin: the run's admission already answered a raise from the streaming
    check as ``hardware_state_unknown``."""
    from tests.scope_fakes import home_sim_scope
    from tests.test_a_run_needs_every_axis_position import _run, _settings

    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        _grabbing_fails(s)
        with pytest.raises(RunCheckFailedError) as refused:
            _run(s, tmp_path)
        assert refused.value.reason == 'hardware_state_unknown'
    finally:
        s.scope._camera_driver.allow_read('grabbing')
        s.shutdown()


# --- bring-up --------------------------------------------------------------------------


def _camera_part(session):
    (camera,) = [p for p in session.bring_up_record().parts if p.part == 'camera']
    return camera


def test_bring_up_whose_grab_state_reads_fail_after_connect_comes_up_saying_so(monkeypatch, caplog):
    """The grab state fails from after the driver's connect until bring-up
    ends: the session comes up, its config writes refused as not sent and
    named on the camera's part, and its last line says the streaming state
    is unknown."""
    real_connect = SimulatedCamera.connect

    def connect_then_fail_the_grab_state(self):
        # a stand-in by design: the simulator cannot arm a read fault before
        # the scope it belongs to is built; armed once connect has run
        connected = real_connect(self)
        self.refuse_read('grabbing', 'failed', count=_MANY)
        return connected

    monkeypatch.setattr(SimulatedCamera, 'connect', connect_then_fail_the_grab_state)
    with caplog.at_level(logging.INFO):
        session = ScopeSession.create(
            complete_settings(binning={'size': '2x2'}), simulate=True, warn_pre_release=False
        )
    try:
        session.scope._camera_driver.allow_read('grabbing')
        part = _camera_part(session)
        assert part.cause == 'settings_not_applied'
        assert 'was not sent' in part.detail, part.detail
        assert any('camera streaming state unknown' in r.getMessage() for r in caplog.records)
    finally:
        session.shutdown()
