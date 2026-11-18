# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Pylon camera whose device is gone answers no frame from ``grab``.

``Camera.grab`` checks the removal flag and the released handle before it
reads the handler's buffer, so a frame stored before the device left is not
reported as current. The Pylon guard once wrapped ``get_last_image`` and built
its own no-frame tuple, which went short each time the base tuple grew; the
frame is a record now and no frame is None. These tests pin that ``grab``
answers no frame for a released handle and for a removed device, quietly.
"""

import logging

import numpy as np
import pytest

from tests.camera_fakes import bare_image_handler
from tests.log_capture import records_of


@pytest.fixture
def detached_camera():
    """A Pylon camera whose handler holds a frame, then loses the device.

    The frame is stored first so the answer is distinguishable from "nothing
    was ever grabbed": with a frame present, only the removal check can be the
    reason the read reports no frame.
    """
    from _lumascope.drivers.camera import Camera

    handler, parent = bare_image_handler()
    handler._store_frame(
        np.zeros((4, 4), dtype=np.uint16),
        chunks=None,
        handed_over=handler.hand_over(),
        significant_bits=12,
        wire_bytes=0,
    )
    parent.cam_image_handler = handler
    # bare_pylon_camera stubs _mark_disconnected; the real one is what the
    # SDK removal callback runs, and its only state change is the flag.
    Camera._mark_disconnected(parent)
    assert parent._device_removed is True
    return handler, parent


class TestDetachedDeviceReportsNoFrame:
    def test_a_released_handle_answers_no_frame(self):
        """A frame is stored, then the handle is released: grab answers no
        frame rather than the frame stored before the release."""
        handler, parent = bare_image_handler()
        handler._store_frame(
            np.zeros((4, 4), dtype=np.uint16),
            chunks=None,
            handed_over=handler.hand_over(),
            significant_bits=12,
            wire_bytes=0,
        )
        parent.cam_image_handler = handler
        assert parent.grab() is not None
        parent.active = None
        assert parent.grab() is None

    def test_grab_answers_no_frame_quietly(self, detached_camera, caplog):
        """grab checks the removal flag before it reads the buffer, and
        answers no frame without logging an error."""
        from _lumascope.drivers import camera as camera_mod

        _handler, parent = detached_camera
        assert parent.grab() is None
        logged = [
            r for r in records_of(caplog, camera_mod) if r.levelno == logging.ERROR and r.exc_info
        ]
        assert logged == [], [r.getMessage() for r in logged]
