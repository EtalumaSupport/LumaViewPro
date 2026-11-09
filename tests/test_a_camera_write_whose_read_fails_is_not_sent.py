# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera write whose read fails is not sent, and is refused as one not sent.

Every write the camera API makes reads what it needs first: the driver's
frame grid, its format list, the binning a frame or binning write is planned
at, a feature's probe, the black level's range, a restore's compare reads.
A read that fails sends nothing, so the write is refused as
``CameraSettingRejected`` reason ``'camera_read_failed'``, chained to the
read: not the camera's refusal, not sent again as a write the camera did not
answer, and nothing held changes. A camera gone by then is ``not_connected``.

The callers that carry on past a refused write (a restore, the channel
apply, bring-up) take this refusal as they take any, so one failed read
never stops what a refused write would not. The reads the API makes for
itself, with no caller to raise to, are reported once by the one reporter
and shown to no one.

Each case runs on the simulated scope; where the simulator cannot fail a
read at that moment, the stand-in at the site says so.
"""

from __future__ import annotations

import logging

import pytest

from _lumascope.drivers.exceptions import CameraNoAnswerError, CameraReadFailedError
from _lumascope.drivers.simulated_camera import SimulatedCamera
from _lumascope.modules.exceptions import (
    CameraSettingRejected,
    HardwareCommandRefusedError,
    MissingPart,
)
from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session():
    s = ScopeSession.create(complete_settings(), simulate=True, warn_pre_release=False)
    yield s
    s.shutdown()


@pytest.fixture
def reported(monkeypatch):
    """Every outcome the one reporter is handed, with how it was asked to tell it."""
    calls = []
    monkeypatch.setattr(
        '_lumascope.modules.notification_center.notifications.report_outcome',
        lambda exc, **kw: calls.append((exc, kw)),
    )
    return calls


def _read_failed(raised) -> bool:
    return (
        isinstance(raised, CameraSettingRejected)
        and raised.reason == 'camera_read_failed'
        and isinstance(raised.__cause__, CameraReadFailedError)
    )


def _driver(session) -> SimulatedCamera:
    return session.scope._camera_driver


# --- a write whose own read fails ----------------------------------------------


def test_a_binning_write_whose_binning_read_fails_is_refused_and_not_sent(session):
    session.scope.simulation.refuse_camera_read('binning', 'failed')

    with pytest.raises(CameraSettingRejected) as refused:
        session.scope.camera.set_binning(2)

    assert _read_failed(refused.value), refused.value
    assert refused.value.setting == 'binning'
    assert _driver(session).get_binning_size() == 1, 'the write was sent'
    assert session.scope.camera.binning == 1


def test_a_frame_size_write_whose_grid_read_fails_keeps_the_window_it_held(session):
    camera = session.scope.camera
    camera.set_frame_size(800, 600)
    window = _driver(session)._frame_window
    session.scope.simulation.refuse_camera_read('frame_grid', 'failed')

    with pytest.raises(CameraSettingRejected) as refused:
        camera.set_frame_size(400, 300)

    assert _read_failed(refused.value), refused.value
    assert _driver(session)._frame_window == window
    assert (camera.frame_size.width, camera.frame_size.height) == (800, 600)


def test_a_pixel_format_write_whose_format_list_read_fails_is_refused_as_not_sent(session):
    camera = session.scope.camera
    format_before = camera.pixel_format
    session.scope.simulation.refuse_camera_read('pixel_formats', 'failed')

    with pytest.raises(CameraSettingRejected) as refused:
        camera.set_image_mode('8bit')

    assert _read_failed(refused.value), refused.value
    assert _driver(session).get_pixel_format() == format_before
    assert camera.pixel_format == format_before


def test_a_write_whose_read_fails_with_the_camera_gone_is_not_connected(session):
    """A pin of the answer the camera's absence already had: green before the
    read failure was its own refusal, and kept so by it."""
    session.scope.simulation.refuse_camera_read('binning', 'removed')

    with pytest.raises(HardwareCommandRefusedError) as refused:
        session.scope.camera.set_binning(2)

    assert (refused.value.reason, refused.value.missing) == ('not_connected', MissingPart.CAMERA)


def test_a_resend_whose_read_fails_is_the_unanswered_write(session):
    """The first try the camera did not answer may have taken the value, so a
    read failing on the resend leaves it unknown: the no-answer refusal, the
    binning held unknown."""
    simulation = session.scope.simulation
    driver = _driver(session)
    tries = []

    def unanswered_then_unreadable(size):
        # a stand-in by design: the simulator's write refusal stands in for the
        # driver's whole set_binning_size, so it cannot fail the resend's read
        # alone; the first try goes unanswered here, and the resend is the
        # real write, whose binning read is armed to fail
        tries.append(size)
        if len(tries) == 1:
            simulation.refuse_camera_read('binning', 'failed')
            raise CameraNoAnswerError('simulated binning write: the camera did not answer')
        return SimulatedCamera.set_binning_size(driver, size)

    driver.set_binning_size = unanswered_then_unreadable

    with pytest.raises(CameraSettingRejected) as refused:
        session.scope.camera.set_binning(2)

    assert refused.value.reason == 'camera_no_answer', refused.value
    assert len(tries) == 2
    assert session.scope.camera.binning is None


# --- the API's own reads before a write -------------------------------------------


def test_a_black_level_write_whose_range_read_fails_is_refused_as_not_sent(session, monkeypatch):
    def unreadable(self):
        # a stand-in by design: the simulator has no read fault on the range
        raise CameraReadFailedError('simulated black level range read failed')

    monkeypatch.setattr(SimulatedCamera, 'get_black_level_range', unreadable)
    if not session.scope.capabilities.camera_supports_black_level:
        pytest.skip('the simulated camera offers no black level here')

    with pytest.raises(CameraSettingRejected) as refused:
        session.scope.camera.set_black_level(3.0)

    assert _read_failed(refused.value), refused.value
    assert refused.value.setting == 'black_level'


def test_a_restore_whose_binning_compare_read_fails_stops_refused(session):
    camera = session.scope.camera
    snapshot = camera.save_camera_state('test')
    camera.set_binning(2)
    session.scope.simulation.refuse_camera_read('binning', 'failed')

    with pytest.raises(CameraSettingRejected) as refused:
        camera.restore_camera_state(snapshot)

    assert _read_failed(refused.value), refused.value
    assert refused.value.setting == 'binning'
    assert _driver(session).get_binning_size() == 2


def test_a_restore_whose_black_level_compare_read_fails_reports_it_and_goes_on(session, reported):
    camera = session.scope.camera
    if not session.scope.capabilities.camera_supports_black_level:
        pytest.skip('the simulated camera offers no black level here')
    camera.set_black_level(3.0)
    snapshot = camera.save_camera_state('test')
    camera.set_black_level(5.0)
    camera.set_gain_db(snapshot['gain_db'] + 2.0)
    camera.set_exposure_ms(snapshot['exposure_ms'] + 10.0)
    reported.clear()
    session.scope.simulation.refuse_camera_read('black_level', 'failed')

    camera.restore_camera_state(snapshot)

    assert camera.gain_db == pytest.approx(snapshot['gain_db'])
    assert camera.exposure_ms == pytest.approx(snapshot['exposure_ms'])
    refusals = [exc for exc, _kw in reported if isinstance(exc, CameraSettingRejected)]
    assert [(r.reason, r.setting) for r in refusals] == [('camera_read_failed', 'black_level')]


# --- the callers that carry on ---------------------------------------------------------


def _gain_read_fails_once(monkeypatch):
    """The gain write's first read fails, once.

    A stand-in by design: the read a gain write needs is the config guard's
    grab-state read, which the simulator cannot fail until it has the
    ``grabbing`` node.
    """
    real = SimulatedCamera.gain
    left = [1]

    def gain(self, value):
        if left[0]:
            left[0] -= 1
            raise CameraReadFailedError('simulated read before the gain write failed')
        return real(self, value)

    monkeypatch.setattr(SimulatedCamera, 'gain', gain)


def test_a_channel_apply_whose_gain_read_fails_still_writes_the_exposure(session, monkeypatch):
    camera = session.scope.camera
    _gain_read_fails_once(monkeypatch)

    with pytest.raises(CameraSettingRejected) as refused:
        camera.apply(gain_db=5.0, exposure_ms=40.0, channel='BF')

    assert _read_failed(refused.value), refused.value
    assert camera.exposure_ms == pytest.approx(40.0)


def test_a_restore_whose_gain_read_fails_still_writes_the_exposure(session, monkeypatch, reported):
    camera = session.scope.camera
    snapshot = camera.save_camera_state('test')
    camera.set_gain_db(snapshot['gain_db'] + 2.0)
    camera.set_exposure_ms(snapshot['exposure_ms'] + 10.0)
    reported.clear()
    _gain_read_fails_once(monkeypatch)

    camera.restore_camera_state(snapshot)

    assert camera.exposure_ms == pytest.approx(snapshot['exposure_ms'])
    refusals = [exc for exc, _kw in reported if isinstance(exc, CameraSettingRejected)]
    assert [r.reason for r in refusals] == ['camera_read_failed']


def test_two_parts_of_an_apply_whose_reads_fail_are_refused_as_read_failed(session, monkeypatch):
    camera = session.scope.camera
    _gain_read_fails_once(monkeypatch)
    real = SimulatedCamera.exposure_t

    def exposure_t(self, exposure_ms):
        # a stand-in by design, as _gain_read_fails_once's
        raise CameraReadFailedError('simulated read before the exposure write failed')

    monkeypatch.setattr(SimulatedCamera, 'exposure_t', exposure_t)

    with pytest.raises(CameraSettingRejected) as refused:
        camera.apply(gain_db=5.0, exposure_ms=40.0, channel='BF')

    assert refused.value.reason == 'camera_read_failed'
    assert refused.value.title == CameraReadFailedError.title
    monkeypatch.setattr(SimulatedCamera, 'exposure_t', real)


def test_an_unanswered_part_outranks_a_read_failed_one_in_a_combined_refusal(session, monkeypatch):
    """A pin: a no-answer part named the combined refusal before, and still
    outranks a part not sent."""
    _gain_read_fails_once(monkeypatch)
    session.scope.simulation.refuse_camera_write('exposure', 'no_answer', count=2)

    with pytest.raises(CameraSettingRejected) as refused:
        session.scope.camera.apply(gain_db=5.0, exposure_ms=40.0, channel='BF')

    assert refused.value.reason == 'camera_no_answer'


def test_two_refused_parts_stay_the_cameras_refusal(session):
    """A pin: two parts the camera refused are its refusal, as before."""
    simulation = session.scope.simulation
    simulation.refuse_camera_write('gain', 'false')
    simulation.refuse_camera_write('exposure', 'false')

    with pytest.raises(CameraSettingRejected) as refused:
        session.scope.camera.apply(gain_db=5.0, exposure_ms=40.0, channel='BF')

    assert refused.value.reason == 'camera_rejected'


# --- bring-up ----------------------------------------------------------------------------


def _camera_part(session):
    (camera,) = [p for p in session.bring_up_record().parts if p.part == 'camera']
    return camera


def test_bring_up_with_a_binning_write_whose_read_fails_comes_up_and_records_it(monkeypatch):
    def unreadable(self, size):
        # a stand-in by design: the simulator cannot arm a read fault before
        # the scope it belongs to is built; this is the binning write's read
        raise CameraReadFailedError('simulated binning read before the write failed')

    monkeypatch.setattr(SimulatedCamera, 'set_binning_size', unreadable)
    session = ScopeSession.create(
        complete_settings(binning={'size': '2x2'}), simulate=True, warn_pre_release=False
    )
    try:
        part = _camera_part(session)
        assert part.cause == 'settings_not_applied'
        assert 'binning change was not sent' in part.detail, part.detail
    finally:
        session.shutdown()


def test_bring_up_whose_binning_is_unoffered_and_unreadable_comes_up_and_records_it(
    monkeypatch,
):
    """Saved at a binning the camera does not offer, with the camera's own
    binning unknown and its read failing: bring-up records the read and goes
    on with the saved binning, where it once failed ``create``."""

    def unreadable(self):
        # a stand-in by design: the simulator cannot arm a read fault before
        # the scope it belongs to is built
        raise CameraReadFailedError('simulated binning node read failed')

    monkeypatch.setattr(SimulatedCamera, 'get_binning_size', unreadable)
    session = ScopeSession.create(
        complete_settings(binning={'size': '3x3'}), simulate=True, warn_pre_release=False
    )
    try:
        part = _camera_part(session)
        assert part.cause == 'settings_not_applied'
        assert 'the change needs first failed' in part.detail
    finally:
        session.shutdown()


# --- the reads the API makes for itself ---------------------------------------------------


def test_a_refresh_after_an_auto_write_whose_gain_read_fails_holds_it_unknown(session, reported):
    camera = session.scope.camera
    if not session.scope.capabilities.camera_supports_auto_gain:
        pytest.skip('the simulated camera has no auto gain here')
    camera.set_gain_db(4.0)
    reported.clear()
    session.scope.simulation.refuse_camera_read('gain', 'failed')

    camera.set_auto_gain(False, {'target_brightness': 0.5, 'min_gain_db': 0.0, 'max_gain_db': 20.0})

    assert camera.gain_db is None
    unread = [(exc, kw) for exc, kw in reported if isinstance(exc, CameraReadFailedError)]
    assert len(unread) == 1
    exc, kw = unread[0]
    assert 'gain' in str(exc)
    assert kw.get('log_only') is True and kw.get('solicited') is False


def test_bring_ups_populate_reports_a_read_that_fails_once_and_shows_nothing(monkeypatch, caplog):
    def unreadable(self):
        # a stand-in by design: the simulator cannot arm a read fault before
        # the scope it belongs to is built
        raise CameraReadFailedError('simulated exposure node read failed')

    monkeypatch.setattr(SimulatedCamera, 'get_exposure_t', unreadable)
    heard = []
    with caplog.at_level(logging.ERROR):
        session = ScopeSession.create(
            complete_settings(),
            simulate=True,
            warn_pre_release=False,
            simulated=SimulatedScope(),
            outcome_listener=heard.append,
        )
    try:
        lines = [
            r
            for r in caplog.records
            if r.name == 'LVP.outcomes' and 'simulated exposure node read failed' in r.getMessage()
        ]
        assert lines, 'the failed read is logged by the one reporter'
        assert all(r.levelno == logging.ERROR for r in lines)
        assert not [n for n in heard if 'exposure node read failed' in str(n)]
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and 'the value held is left' in r.getMessage()
        ]
        assert warnings == []
    finally:
        session.shutdown()
