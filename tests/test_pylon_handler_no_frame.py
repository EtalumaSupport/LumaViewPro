# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Pylon handler's no-frame answer is the base handler's no-frame answer.

``pyloncamera.ImageHandler`` composes ``ImageHandlerBase`` and wraps its
``get_last_image`` with a detached-device guard. The guard used to build its
own no-frame tuple, so each time the base tuple grew (the per-frame depth
stamp, then the arrival ordinal) the guard's copy stayed short. Every reader
unpacks the tuple positionally; ``grab`` and ``grab_latest`` catch the
resulting ValueError and log a traceback, but ``last_stamped_significant_bits``
has no net, so after a disconnect an L2 read of ``last_significant_bits``
raised instead of falling back.

The tuple now has one owner, ``ImageHandlerBase.NO_FRAME``, and these tests
pin the two contracts that owner exists for: the wrapper's guard answer IS the
base's answer, and the one unguarded reader survives a detached device.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.camera_fakes import bare_image_handler


@pytest.fixture
def detached_camera():
    """A Pylon camera whose handler holds a stamped frame, then loses the device.

    The frame is stored first so a fallback (None) is distinguishable from
    "nothing was ever grabbed": with a frame present, only the guard can be
    the reason the read reports no frame.
    """
    from drivers.camera import Camera

    handler, parent = bare_image_handler()
    handler._base._store_frame(
        np.zeros((4, 4), dtype=np.uint16), timestamp=1.0, chunks=None, significant_bits=12
    )
    parent.cam_image_handler = handler
    # bare_pylon_camera stubs _mark_disconnected; the real one is what the
    # SDK removal callback runs, and its only state change is the flag.
    Camera._mark_disconnected(parent)
    assert parent._device_removed is True
    return handler, parent


class TestDetachedDeviceReportsNoFrame:
    def test_removed_device_answers_with_the_base_no_frame(self, detached_camera):
        from drivers.camera import ImageHandlerBase

        handler, _parent = detached_camera
        assert handler.get_last_image() == ImageHandlerBase.NO_FRAME

    def test_released_handle_answers_with_the_base_no_frame(self):
        from drivers.camera import ImageHandlerBase

        handler, parent = bare_image_handler()
        parent.active = None
        assert handler.get_last_image() == ImageHandlerBase.NO_FRAME

    def test_no_frame_unpacks_like_a_delivered_frame(self):
        """The owner's shape is the delivered shape: five positional slots."""
        from drivers.camera import ImageHandlerBase

        base = ImageHandlerBase()
        base._store_frame(
            np.zeros((2, 2), dtype=np.uint8), timestamp=1.0, chunks=None, significant_bits=8
        )
        assert len(ImageHandlerBase.NO_FRAME) == len(base.get_last_image())
        assert ImageHandlerBase.NO_FRAME[0] is False


class TestUnguardedReaderSurvivesDetach:
    def test_stamp_read_falls_back_instead_of_raising(self, detached_camera):
        _handler, parent = detached_camera
        assert parent.last_stamped_significant_bits() is None

    def test_depth_read_falls_back_to_the_format_depth(self, detached_camera, monkeypatch):
        """``last_significant_bits`` is the L2-facing read; with no frame it
        reports the validated format depth, never an exception."""
        _handler, parent = detached_camera
        monkeypatch.setattr(type(parent), 'significant_bits', property(lambda self: 8))
        assert parent.last_significant_bits == 8

    def test_grab_paths_stay_quiet(self, detached_camera, monkeypatch):
        """grab / grab_latest pre-check the flag, so they never reach the
        guard; pinned here so the pre-check is not later removed on the
        strength of the guard alone."""
        from drivers import camera as camera_mod

        _handler, parent = detached_camera
        cam_log = MagicMock()
        monkeypatch.setattr(camera_mod, '_cam_log', cam_log)
        assert parent.grab() == (False, None, None)
        assert parent.grab_latest() == (False, None, None, None, None)
        cam_log.exception.assert_not_called()
