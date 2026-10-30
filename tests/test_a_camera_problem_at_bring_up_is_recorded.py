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
"""

from __future__ import annotations

import threading

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import ArgumentRefusedError
from modules.lumascope_api import camera as camera_api
from modules.lumascope_api.simulation import SimulatedCameraWriteRefusal as Refusal
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

# BF's stored values differ from those the simulated camera starts at (its
# gain starts at 1.0, as the shipped BF's does), so the hold writes them.
_BF = {'gain_db': 7.0, 'exposure_ms': 33.0}


def _create(heard, *refusals, microscope='LS850', bf=None, **kwargs):
    return ScopeSession.create(
        complete_settings(microscope=microscope, BF={**_BF, **(bf or {})}),
        simulate=True,
        warn_pre_release=False,
        sim_refused_camera_writes=refusals,
        outcome_listener=heard.append,
        **kwargs,
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
    real_hold = camera_api.CameraAPI._hold_layer_impl

    def leave_then_hold(self, layer, request):
        self._scope.simulation.remove_camera()
        return real_hold(self, layer, request)

    monkeypatch.setattr(camera_api.CameraAPI, '_hold_layer_impl', leave_then_hold)
    heard = []
    session = _create(heard, microscope='LS620', sim_refused_stream_starts=1)
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
        assert session.scope.camera.get_held_layer() is None

        session.show_channel('BF')

        assert session.scope.camera.get_held_layer() == 'BF'
    finally:
        session.shutdown()


def test_the_seam_is_refused_on_the_fx2_camera():
    with pytest.raises(ArgumentRefusedError) as refused:
        _create([], Refusal('gain', 'false'), microscope='LS620')
    assert refused.value.reason == 'not_injectable_on_board'
