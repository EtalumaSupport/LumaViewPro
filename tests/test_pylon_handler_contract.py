# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every camera driver's image handler IS an ``ImageHandlerBase``.

``Camera`` reads its handler through the base surface -- ``frames_delivered``,
``get_last_chunks``, ``get_last_image``, ``NO_FRAME``. The Pylon handler once
held a base by composition and re-exposed four of its names by hand; each time
the base grew the copy did not, and the third such miss stopped LVP starting
on every Pylon scope (``Lumascope.initialize`` extinguishes the LEDs, the LED
write invalidates the frame, and the invalidation reads ``frames_delivered``
off a handler that lacked it).

These tests pin the contract at the class level, so a handler that stops
inheriting the base fails the suite rather than the first bench launch.
"""

import importlib

import numpy as np
import pytest

from drivers.camera import ImageHandlerBase
from tests.camera_fakes import bare_image_handler


def _store_one(handler, chunks=None):
    handler._store_frame(
        np.zeros((4, 4), dtype=np.uint16), timestamp=1.0, chunks=chunks, significant_bits=12
    )


class TestEveryDriverHandlerIsAnImageHandlerBase:
    @pytest.mark.parametrize(
        'module_name, class_name',
        [
            ('drivers.pyloncamera', 'ImageHandler'),
            ('drivers.idscamera', 'ImageHandler'),
            ('drivers.fx2driver', '_FX2ImageHandler'),
        ],
    )
    def test_handler_class_inherits_the_base(self, module_name, class_name):
        cls = getattr(importlib.import_module(module_name), class_name)
        assert issubclass(cls, ImageHandlerBase)


class TestPylonHandlerBaseSurface:
    def test_frames_delivered_counts_stored_frames(self):
        handler, parent = bare_image_handler()
        parent.cam_image_handler = handler
        assert handler.frames_delivered == 0
        assert parent.frames_delivered == 0
        _store_one(handler)
        assert handler.frames_delivered == 1
        assert parent.frames_delivered == 1

    def test_get_last_chunks_reports_the_stored_frames_chunks(self):
        handler, _parent = bare_image_handler()
        assert handler.get_last_chunks() is None
        _store_one(handler, chunks={'ExposureTime': 10000.0, 'Gain': 0.0})
        assert handler.get_last_chunks() == {'ExposureTime': 10000.0, 'Gain': 0.0}

    def test_detached_device_reports_no_chunks_and_no_frame(self):
        from drivers.camera import Camera

        handler, parent = bare_image_handler()
        _store_one(handler, chunks={'ExposureTime': 10000.0})
        Camera._mark_disconnected(parent)
        assert parent._device_removed is True
        assert handler.get_last_chunks() is None
        assert handler.get_last_image() == ImageHandlerBase.NO_FRAME
