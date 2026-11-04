# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Preview 12->8 LUT reuses a caller-owned buffer instead of allocating
a fresh array every frame.

The 30 fps preview path converts each 12-bit frame to 8-bit via a LUT.
Without a reusable destination it allocated a fresh ~W*H array per frame
(~108 MB/s allocator churn on the display thread). get_image_from_buffer
now accepts out_8bit and threads it into convert_12bit_to_8bit(out=); the
preview owns a single buffer (the histogram, on another thread, passes
none, so there is no cross-thread sharing). tobytes() copies before the
next frame overwrites the buffer, so a single slot is safe.

Tests cover the convert reuse / fallback semantics directly and the
get_image_from_buffer wiring on the simulated scope.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np


REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from modules import image_utils


def test_convert_reuses_provided_out_buffer():
    img = np.arange(64, dtype=np.uint16).reshape(8, 8) * 60  # 0..3780, < 4096
    out = np.empty((8, 8), dtype=np.uint8)
    result = image_utils.convert_12bit_to_8bit(img, out=out)
    assert result is out, 'must write into the caller buffer, not allocate'
    fresh = image_utils.convert_12bit_to_8bit(img)
    assert np.array_equal(result, fresh), 'reused-buffer result must match fresh'


def test_convert_falls_back_on_shape_mismatch():
    img = np.zeros((8, 8), dtype=np.uint16)
    out = np.empty((4, 4), dtype=np.uint8)  # wrong shape
    result = image_utils.convert_12bit_to_8bit(img, out=out)
    assert result is not out, 'mismatched out must fall back to a fresh array'
    assert result.shape == (8, 8)


def test_convert_8bit_passthrough_ignores_out():
    img = np.zeros((8, 8), dtype=np.uint8)
    out = np.empty((8, 8), dtype=np.uint8)
    result = image_utils.convert_12bit_to_8bit(img, out=out)
    assert result is img, '8-bit input returns the input unchanged; out unused'


def test_get_image_from_buffer_reuses_caller_buffer(sim_scope):
    # The preview path must write each converted frame into the caller's
    # buffer instead of allocating a fresh array per frame.
    sim_scope.camera.set_image_mode('12bit_scientific')
    imaging = sim_scope.imaging
    imaging.get_image(force_new_capture=True)

    first, _ts = imaging.get_image_from_buffer(force_to_8bit=False)
    assert first is not None and first.dtype == np.uint16, (
        'precondition: the 12-bit sim frame must need conversion'
    )
    out = np.empty(first.shape, dtype=np.uint8)
    img1, _ = imaging.get_image_from_buffer(force_to_8bit=True, out_8bit=out)
    img2, _ = imaging.get_image_from_buffer(force_to_8bit=True, out_8bit=out)
    assert img1 is out and img2 is out, (
        'get_image_from_buffer must reuse the caller-owned out_8bit '
        'buffer on every frame, not allocate fresh arrays'
    )
