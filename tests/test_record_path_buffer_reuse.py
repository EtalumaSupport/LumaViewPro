# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Record-path conversions reuse caller-owned scratch buffers.

record_helper (the camera_executor task that writes one recorded video
frame) converts each frame's depth and, for mono fluorescence channels,
widens it to a false-color RGB array. Without reusable destinations each
frame allocated a fresh ~W*H (depth convert) plus a ~W*H*3 (false color)
array -- 3.6-10.8 MB of allocator churn per recorded frame. record_helper
now threads two scratch buffers (one per conversion) into the existing
out=/output= parameters; the buffers are owned by MainDisplay, sized
lazily, and freed at finalize.

Reuse is safe: record_helper runs on the single-threaded camera_executor
and copies its result into the memmap slot before the next call can
overwrite the scratch.

The depth-convert out= reuse is already covered (test_image_utils.py for
8-bit, test_audit_fixes.py PIW-5 for 16-bit). These tests cover the
add_false_color output= reuse semantics directly and lock the
record_helper wiring structurally (MainDisplay needs a full scope to
instantiate).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np


REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from modules import image_utils


def test_add_false_color_reuses_output_buffer():
    src = np.arange(48, dtype=np.uint8).reshape(6, 8)
    out = np.empty((6, 8, 3), dtype=np.uint8)
    result = image_utils.add_false_color(array=src, color='Green', output=out)
    assert result is out, 'must write into the caller buffer, not allocate'
    fresh = image_utils.add_false_color(array=src, color='Green')
    assert np.array_equal(result, fresh), 'reused-buffer result must match fresh'


def test_add_false_color_falls_back_on_shape_mismatch():
    src = np.zeros((6, 8), dtype=np.uint8)
    out = np.empty((3, 4, 3), dtype=np.uint8)  # wrong shape
    result = image_utils.add_false_color(array=src, color='Green', output=out)
    assert result is not out, 'mismatched output must fall back to a fresh array'
    assert result.shape == (6, 8, 3)


def test_add_false_color_falls_back_on_dtype_mismatch():
    src = np.zeros((6, 8), dtype=np.uint8)
    out = np.empty((6, 8, 3), dtype=np.uint16)  # wrong dtype
    result = image_utils.add_false_color(array=src, color='Green', output=out)
    assert result is not out, 'mismatched-dtype output must fall back to a fresh array'
    assert result.dtype == np.uint8


def test_record_helper_threads_scratch_buffers():
    # Structural lock: record_helper must pass MainDisplay-owned scratch
    # buffers AND the snapshotted capture depth through to the canonical depth
    # converters and the false-color widening. MainDisplay needs a full scope
    # to instantiate, so assert on source.
    src = (REPO / 'ui' / 'main_display.py').read_text()
    assert 'convert_to_8bit(' in src
    assert 'convert_to_16bit(' in src
    assert src.count('out=self._record_convert_buf') == 2
    assert 'self._record_capture_depth' in src
    assert 'output=self._record_color_buf' in src
