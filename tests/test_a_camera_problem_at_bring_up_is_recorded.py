# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera problem met at bring-up is recorded on the camera's part, told once, and the scope comes up.

Bring-up writes to the camera twice: the scope's own geometry and feature
writes in ``initialize``, then the session's BF hold. A setting the camera
refuses or does not answer, a write that outlasts the camera lane's bound,
or a camera that leaves during either is kept on the camera's part of the
bring-up record (``settings_not_applied``, ``camera_left``), told once, and
bring-up goes on. A refusal from the BF hold used to raise out of
``create``, so LumaViewPro, the REST server and the support report did not
start; one from ``initialize`` was told and left nothing on the record.

The camera part keeps one cause, the one a person acts on first:
``camera_left`` over ``stream_not_started`` over ``settings_not_applied``;
every problem's words stay in its detail.

A camera not on the bus at bring-up is the cause before all of these: the
part is down, ``not_detected``, told once as a real scope's unplugged camera
is, and the scope comes up with every camera command refused.
"""

from __future__ import annotations

import logging
import threading

import pytest

from _lumascope.drivers.simulated_camera import SimulatedCamera
from _lumascope.modules.exceptions import (
    ArgumentRefusedError,
    HardwareCommandRefusedError,
    MissingPart,
)
from _lumascope.modules.scope_parts import Part
from _lumascope.modules.lumascope_api import camera as camera_api
from _lumascope.modules.lumascope_api.simulation import SimulatedCameraWriteRefusal as Refusal
from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from _lumascope.modules.notification_center import Severity
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

# BF's stored values differ from those the simulated camera starts at (its
# gain starts at 1.0, as the shipped BF's does), so the hold writes them.
_BF = {'gain_db': 7.0, 'exposure_ms': 33.0}


def _create(
    heard, *refusals, microscope='LS850', bf=None, refused_stream_starts=0, camera_present=True
):
    return ScopeSession.create(
        complete_settings(microscope=microscope, BF={**_BF, **(bf or {})}),
        simulate=True,
        warn_pre_release=False,
        simulated=SimulatedScope(
            refused_camera_writes=refusals,
            refused_stream_starts=refused_stream_starts,
            camera_present=camera_present,
        ),
        outcome_listener=heard.append,
    )


def _camera_part(session):
    (camera,) = [p for p in session.bring_up_record().parts if p.part == 'camera']
    return camera


def _told(heard, reason):
    return [n for n in heard if n.reason == reason and n.category == 'Camera']


@pytest.mark.parametrize(
    ('refusal', 'bf', 'reason', 'words'),
    [
        (Refusal('gain', 'false'), None, 'camera_rejected', 'gain'),
        (Refusal('exposure', 'false'), None, 'camera_rejected', 'exposure'),
        (Refusal('gain', 'no_answer', count=2), None, 'camera_no_answer', 'gain'),
        (Refusal('auto_gain', 'false'), {'auto_gain': True}, 'camera_rejected', 'auto'),
    ],
    ids=['refused_gain', 'refused_exposure', 'unanswered_gain', 'refused_auto_gain_arm'],
)
def test_a_bf_setting_the_camera_does_not_take_is_recorded_and_the_scope_comes_up(
    refusal, bf, reason, words
):
    # parametrize-justified: each node is its own write in the hold, with its own fixture
    heard = []
    session = _create(heard, refusal, bf=bf)
    try:
        camera = _camera_part(session)
        assert session.scope.camera_connected
        assert (camera.up, camera.cause) == (True, 'settings_not_applied')
        assert words in camera.detail.lower()
        assert session.status.parts == session.bring_up_record().parts
        (notice,) = _told(heard, reason)
        assert notice.solicited is False
    finally:
        session.shutdown()


def test_a_camera_that_leaves_during_the_bf_hold_is_recorded_as_left():
    heard = []
    session = _create(heard, Refusal('gain', 'removed'))
    try:
        assert _camera_part(session).cause == 'camera_left'
        assert len(_told(heard, 'not_connected')) == 1
    finally:
        session.shutdown()


def test_a_camera_that_leaves_during_the_binning_write_is_recorded_as_left():
    heard = []
    session = _create(heard, Refusal('binning', 'removed'))
    try:
        assert _camera_part(session).cause == 'camera_left'
        assert len(_told(heard, 'not_connected')) == 1
    finally:
        session.shutdown()


def test_a_refused_binning_is_recorded():
    heard = []
    session = _create(heard, Refusal('binning', 'false'))
    try:
        camera = _camera_part(session)
        assert camera.cause == 'settings_not_applied'
        assert 'binning' in camera.detail.lower()
        assert len(_told(heard, 'camera_rejected')) == 1
    finally:
        session.shutdown()


def test_a_camera_that_left_outranks_a_stream_that_did_not_start(monkeypatch):
    # a stand-in by design: the simulator arms no fault at the start of a
    # hold, so the departure is made there by wrapping the hold; the
    # departure itself is the simulated FX2's own unplug.
    real_hold = camera_api.CameraAPI._hold_channel_impl

    def leave_then_hold(self, layer, request):
        self._scope.simulation.remove_camera()
        return real_hold(self, layer, request)

    monkeypatch.setattr(camera_api.CameraAPI, '_hold_channel_impl', leave_then_hold)
    heard = []
    session = _create(heard, microscope='LS620', refused_stream_starts=1)
    try:
        camera = _camera_part(session)
        assert camera.cause == 'camera_left'
        assert 'CameraStreamStartFailedError' in camera.detail
        assert 'HardwareCommandRefusedError' in camera.detail
    finally:
        session.shutdown()


def test_a_bf_write_that_outlasts_the_lanes_bound_is_recorded(monkeypatch):
    # a stand-in by design: the simulator has no camera write that hangs, so
    # the simulated camera's gain write is held until the test lets it go.
    release = threading.Event()
    real_gain = SimulatedCamera.gain

    def held_gain(self, *args, **kwargs):
        release.wait(10.0)
        return real_gain(self, *args, **kwargs)

    monkeypatch.setattr(SimulatedCamera, 'gain', held_gain)
    monkeypatch.setattr(camera_api.CameraAPI, '_CAMERA_WRITE_TIMEOUT_S', 0.3)
    heard = []
    try:
        session = _create(heard)
    finally:
        release.set()
    try:
        camera = _camera_part(session)
        assert camera.cause == 'settings_not_applied'
        assert 'LaneCallStillRunningError' in camera.detail
        assert len(_told(heard, 'call_overran')) == 1
    finally:
        session.shutdown()


def test_after_a_refused_bf_gain_showing_bf_writes_it_again():
    heard = []
    session = _create(heard, Refusal('gain', 'false'))
    try:
        assert session.scope.camera.get_held_channel() is None

        session.show_channel('BF')

        assert session.scope.camera.get_held_channel() == 'BF'
    finally:
        session.shutdown()


def test_the_seam_is_refused_on_the_fx2_camera():
    with pytest.raises(ArgumentRefusedError) as refused:
        _create([], Refusal('gain', 'false'), microscope='LS620')
    assert refused.value.reason == 'not_injectable_on_board'


@pytest.mark.parametrize('microscope', ['LS850', 'LS820'])
def test_a_camera_not_on_the_bus_is_recorded_not_detected_and_told_once(microscope):
    # parametrize-justified: an EL-0940 scope with a motorized stage and one
    # without, each its own simulated scope
    heard = []
    session = _create(heard, microscope=microscope, camera_present=False)
    try:
        camera = _camera_part(session)
        assert (camera.up, camera.cause) == (False, 'not_detected')
        assert not session.scope.camera_connected
        told = [n for n in heard if n.category == 'Camera' and n.severity.level >= logging.WARNING]
        assert [(n.title, n.message) for n in told] == [
            ('Camera not detected', 'Camera not found. Check USB cable and power.')
        ]
        with pytest.raises(HardwareCommandRefusedError) as refused:
            session.scope.camera.read_gain_db()
        assert (refused.value.reason, refused.value.missing) == (
            'not_connected',
            MissingPart(Part.CAMERA),
        )
    finally:
        session.shutdown()


@pytest.mark.parametrize(
    ('microscope', 'motor_up', 'told'),
    [
        ('LS620', False, [('Hardware', 'No hardware detected')]),
        (
            'LS720',
            True,
            [('Camera', 'Camera not detected'), ('Illumination', 'LED Controller Unavailable')],
        ),
    ],
    ids=['manual', 'motorized'],
)
def test_an_fx2_not_on_the_bus_takes_its_camera_and_led_board_and_is_told_as_a_real_one(
    microscope, motor_up, told
):
    # parametrize-justified: a Classic with nothing else on the bus, and one
    # whose motor board still comes up, are told differently
    heard = []
    session = _create(heard, microscope=microscope, camera_present=False)
    try:
        parts = {p.part: p for p in session.bring_up_record().parts}
        assert {part: (parts[part].up, parts[part].cause) for part in ('camera', 'led')} == {
            'camera': (False, 'not_detected'),
            'led': (False, 'not_detected'),
        }
        assert parts['motor'].up is motor_up
        assert [
            (n.category, n.title) for n in heard if n.severity.level >= Severity.NOTICE.level
        ] == told
        with pytest.raises(HardwareCommandRefusedError) as refused:
            session.scope.illumination.led_on('BF', 10.0)
        assert refused.value.reason == 'not_connected'
    finally:
        session.shutdown()
