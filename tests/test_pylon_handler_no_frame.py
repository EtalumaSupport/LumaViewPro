# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Pylon handler whose device is gone answers no frame, as the base handler does.

``pyloncamera.ImageHandler`` inherits ``ImageHandlerBase`` and answers its
detached-device predicate, which the base's reader consults. The Pylon guard
once wrapped ``get_last_image`` and built its own no-frame tuple, which went
short each time the base tuple grew. The frame is a record now and no frame
is None, so there is no shape to copy; these tests pin that a frame stored
before the device left is not reported as current, and that ``grab`` stays
quiet on the way.
"""

import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.camera_fakes import bare_image_handler


@pytest.fixture
def detached_camera():
    """A Pylon camera whose handler holds a frame, then loses the device.

    The frame is stored first so the answer is distinguishable from "nothing
    was ever grabbed": with a frame present, only the guard can be the reason
    the read reports no frame.
    """
    from drivers.camera import Camera

    handler, parent = bare_image_handler()
    handler._store_frame(
        np.zeros((4, 4), dtype=np.uint16),
        datetime.datetime.now(),
        chunks=None,
        arrived_ns=0,
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
    def test_a_removed_device_answers_no_frame(self, detached_camera):
        handler, _parent = detached_camera
        assert handler.get_last_image() is None

    def test_a_released_handle_answers_no_frame(self):
        handler, parent = bare_image_handler()
        parent.active = None
        assert handler.get_last_image() is None

    def test_grab_answers_no_frame_quietly(self, detached_camera, monkeypatch):
        """grab pre-checks the removal flag, so it never reaches the guard;
        pinned here so the pre-check is not later removed on the strength
        of the guard alone."""
        from drivers import camera as camera_mod

        _handler, parent = detached_camera
        cam_log = MagicMock()
        monkeypatch.setattr(camera_mod, '_cam_log', cam_log)
        assert parent.grab() is None
        cam_log.exception.assert_not_called()
