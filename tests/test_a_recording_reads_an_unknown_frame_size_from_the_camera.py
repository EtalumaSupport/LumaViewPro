# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A manual recording started with the frame size unknown is made at the frame the camera delivers.

A manual recording makes its file at the held frame size,
``scope.camera.frame_size``, which is None after a frame write the camera
never answered. Then the size is read from the camera (``read_frame_size``)
rather than assumed, so the frames are recorded at the frame the camera is
delivering; a camera that does not answer that read refuses the recording
before anything is written.

A protocol video step reads its size the same way, but a run cannot reach
it with the size unknown: the run's start saves the camera's state, which
reads the geometry from the camera, so the size is held before any step.

Run on the simulated LS850 framed at 800 x 600, with the held frame size
left unknown by an unanswered write of 400 x 300.
"""

from __future__ import annotations

import threading

import pytest
import tifffile

import modules.manual_recording as manual_recording_module
from drivers.exceptions import CameraReadFailedError
from modules.exceptions import CameraSettingRejected
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT, _settings

_DELIVERED = (600, 800)


@pytest.fixture
def session(tmp_path, monkeypatch):
    # a stand-in by design: the disk probe would report this machine's free space
    monkeypatch.setattr(
        manual_recording_module, 'check_disk_space_ok', lambda *_: (True, 1_000_000.0)
    )
    s = ScopeSession.create(
        complete_settings(
            **_settings(
                tmp_path,
                video_as_frames=True,
                video={'max_fps': 0, 'max_duration_s': 60, 'timestamp_overlay': False},
            )
        ),
        simulate=True,
    )
    try:
        s.camera.set_frame_size(800, 600)
        # Unanswered twice: the write and the API's one resend. The camera
        # stays at 800 x 600 and the held frame size becomes unknown.
        s.scope.simulation.refuse_camera_write('frame_size', 'no_answer', count=2)
        with pytest.raises(CameraSettingRejected):
            s.camera.set_frame_size(400, 300)
        assert s.camera.frame_size is None
        yield s
    finally:
        s.shutdown()


def _frames_after_start(session, count):
    """An event set once ``count`` frames have arrived, and its listener's removal."""
    seen = []
    arrived = threading.Event()

    def listener(*_a):
        seen.append(1)
        if len(seen) >= count:
            arrived.set()

    session.scope.imaging.add_frame_listener(listener, name='test_frames_after_start')
    return arrived, lambda: session.scope.imaging.remove_frame_listener(listener)


def test_a_manual_recording_is_made_at_the_frame_the_camera_delivers(session):
    operation = session.manual_recording.start()
    arrived, remove = _frames_after_start(session, 5)
    try:
        assert arrived.wait(COMPLETION_TIMEOUT), 'the camera delivered no frames to record'
    finally:
        remove()
        operation.stop()
    operation.result(timeout=COMPLETION_TIMEOUT)

    frames = sorted(session.manual_recording.save_folder.glob('ManualVideo_Frame_*.tiff'))
    assert frames, 'the recording wrote no frame'
    assert tifffile.imread(frames[0]).shape == _DELIVERED


def test_a_manual_recording_whose_frame_size_cannot_be_read_does_not_start(session, tmp_path):
    session.scope.simulation.refuse_camera_read('frame_size', 'failed')

    with pytest.raises(CameraReadFailedError) as refused:
        session.manual_recording.start()

    assert refused.value.reason == 'camera_read_failed'
    assert 'frame size' in str(refused.value)
    assert not session.manual_recording.is_busy
    assert session.activity_claim.holder is None
    assert list(tmp_path.rglob('*.tiff')) == []
