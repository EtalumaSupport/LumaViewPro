# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A write or a record that needs the binning reads it from the camera when the held one is unknown.

``scope.camera.binning`` is None after a binning write the camera never
answered. A frame write (``set_frame_size``) and a capture's frame record
both need the binning the camera is at, so they ask the camera rather than
assume one; a camera that does not answer that read fails the write, and
nothing is saved.

Run on the simulated LS850, left at 2x binning with the held binning unknown
by an unanswered write of 1x, so an assumed 1x and the camera's 2x differ.
"""

from __future__ import annotations

import pytest

from drivers.exceptions import CameraReadFailedError
from modules.exceptions import CameraSettingRejected
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        s.camera.set_binning(2)
        # Unanswered twice: the write and the API's one resend. The camera
        # stays at 2x and the held binning becomes unknown.
        s.scope.simulation.refuse_camera_write('binning', 'no_answer', count=2)
        with pytest.raises(CameraSettingRejected):
            s.camera.set_binning(1)
        assert s.camera.binning is None
        yield s
    finally:
        s.shutdown()


def test_a_frame_written_with_the_binning_unknown_is_framed_at_the_cameras_binning(session):
    session.camera.set_frame_size(400, 300)

    frame = session.settings['frame']
    assert (frame['native_width'], frame['native_height']) == (800, 600)
    assert session.camera.binning == 2


def test_a_frame_written_with_the_binning_unknown_and_unreadable_is_refused_and_not_saved(session):
    saved = dict(session.settings['frame'])
    session.scope.simulation.refuse_camera_read('binning', 'failed')

    with pytest.raises(CameraReadFailedError) as refused:
        session.camera.set_frame_size(400, 300)

    assert refused.value.reason == 'camera_read_failed'
    assert 'binning' in str(refused.value)
    assert session.settings['frame'] == saved


def test_a_capture_with_the_binning_unknown_records_the_cameras_binning(session):
    session.scope.imaging.capture_and_wait(timeout_s=5.0)

    assert session.scope.imaging.last_capture_info['frame_record'].binning_size == 2
