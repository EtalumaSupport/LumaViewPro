# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A grabbed frame carries its own significant-bit depth.

The live/runtime counterpart of load_pixels' (pixels, significant_bits) pairing.
A camera frame is buffered together with the depth it was captured under, so a
consumer cannot pair a frame with a depth read separately from the driver's
current pixel format. That separate read is what crashed the app on an image-mode
switch: a Mono12 frame still in the buffer was downconverted against the driver's
already-switched Mono8 depth (an 8-bit, 256-entry LUT indexed by a 4095-range
value -> IndexError).

These pin the coupling at each layer:
  - ImageHandlerBase stores + returns the frame's depth (where Pylon/IDS buffer).
  - grab_latest carries it (the simulated camera, end to end).
  - get_image_from_buffer downconverts by the FRAME's depth, never a live driver
    re-query -- the direct regression for the mode-switch crash.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


class TestImageHandlerBaseCouplesDepth:
    """The frame buffer that Pylon/IDS use stores depth WITH the frame, so a
    buffered frame keeps its own depth even after the camera's format changes."""

    def test_store_frame_returns_its_significant_bits(self):
        from drivers.camera import ImageHandlerBase

        h = ImageHandlerBase()
        img = np.zeros((4, 4), dtype=np.uint16)
        h._store_frame(img, timestamp=1.0, chunks=None, significant_bits=12)
        result, _out_img, _out_ts, sig = h.get_last_image()
        assert result is True
        assert sig == 12

    def test_buffered_frame_keeps_depth_when_format_later_changes(self):
        from drivers.camera import ImageHandlerBase

        # A 12-bit frame is buffered; the camera then switches to Mono8. The
        # buffered frame must still report 12 (its own depth), not be re-derived
        # from whatever the camera reports now.
        h = ImageHandlerBase()
        h._store_frame(np.zeros((4, 4), dtype=np.uint16), timestamp=1.0, significant_bits=12)
        # (no new frame stored under the new format)
        _, _, _, sig = h.get_last_image()
        assert sig == 12


class TestGrabLatestCarriesDepth:
    """grab_latest returns the frame's depth alongside the frame."""

    def test_sim_grab_latest_returns_format_depth(self):
        from drivers.simulated_camera import SimulatedCamera

        cam = SimulatedCamera()
        cam.connect()
        assert cam.set_pixel_format('Mono12')
        cam.start_grabbing()
        result, _img, _ts, sig = cam.grab_latest()
        assert result is True
        assert sig == 12

        assert cam.set_pixel_format('Mono8')
        result, _img, _ts, sig = cam.grab_latest()
        assert sig == 8


class TestGetImageFromBufferUsesFrameDepth:
    """The preview downconvert uses the FRAME's depth, not a live driver query --
    the direct regression for the image-mode-switch crash."""

    def test_downconvert_uses_frame_depth_not_live_driver(self, monkeypatch):
        from drivers.simulated_camera import SimulatedCamera
        from modules.lumascope_api import Lumascope
        from modules.lumascope_api.imaging import ImagingAPI
        from modules.lumascope_api.runtime_state import RuntimeState

        cam = SimulatedCamera()
        cam.connect()
        cam.start_grabbing()

        # The switch race: the buffered frame is a 12-bit white frame (stamped 12
        # at capture), but the driver's LIVE significant_bits has already flipped
        # to 8. grab_latest hands back the frame WITH its own depth (12); the live
        # query (8) is the stale value the consumer must NOT use.
        mono12_white = np.full((64, 64), 4095, dtype=np.uint16)
        monkeypatch.setattr(cam, 'grab_latest', lambda: (True, mono12_white, 1.0, 12))
        monkeypatch.setattr(SimulatedCamera, 'significant_bits', property(lambda self: 8))

        scope = Lumascope.__new__(Lumascope)
        scope._camera_driver = cam
        scope.runtime_state = RuntimeState(scope)
        imaging = ImagingAPI(scope, cam)

        img, _ts = imaging.get_image_from_buffer(force_to_8bit=True)

        assert img is not None  # did not crash on the wide frame
        assert img.dtype == np.uint8
        # Scaled by the frame's own 12-bit depth, full-white maps to 255. Scaling
        # by the live 8-bit value would index a 256-entry LUT with 4095 and crash
        # (the original bug).
        assert int(img.max()) == 255


class TestLastSignificantBitsViaMethodContract:
    """Camera.last_significant_bits must read the buffered frame's depth through
    the handler's get_last_image() method, not the raw last_img_significant_bits
    attribute. The Pylon handler composes ImageHandlerBase (no raw attribute
    exposed), so reaching the attribute directly AttributeErrors on the Pylon
    camera; every other handler consumer already uses the method contract.
    """

    def test_reads_depth_from_composition_handler(self):
        from types import SimpleNamespace

        from drivers.camera import Camera

        # A composition-style handler (the Pylon shape): exposes get_last_image
        # but NOT the raw last_img_significant_bits attribute.
        handler = SimpleNamespace(get_last_image=lambda: (True, None, None, 12))
        assert not hasattr(handler, 'last_img_significant_bits')

        cam = SimpleNamespace(cam_image_handler=handler, significant_bits=16)
        # Reaching the raw attribute (the bug) would raise AttributeError here.
        assert Camera.last_significant_bits.fget(cam) == 12

    def test_falls_back_to_live_depth_when_no_buffered_frame(self):
        from types import SimpleNamespace

        from drivers.camera import Camera

        # No frame stored yet -> get_last_image reports failure -> live depth.
        handler = SimpleNamespace(get_last_image=lambda: (False, None, None, None))
        cam = SimpleNamespace(cam_image_handler=handler, significant_bits=16)
        assert Camera.last_significant_bits.fget(cam) == 16
