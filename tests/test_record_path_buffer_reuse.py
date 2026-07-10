# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Record-path conversions reuse caller-owned scratch buffers.

record_helper (the camera_executor task that writes one recorded video
frame) downconverts an 8-bit capture into a reusable scratch buffer;
without it each frame allocated a fresh ~W*H array of allocator churn.
record_helper threads that one scratch buffer into the converter's out=
parameter; the buffer is owned by MainDisplay, sized lazily, and freed at
finalize. False color is no longer applied in the record path -- the
memmap stays mono and colorization happens at the save edges -- so there
is no longer a second (RGB) scratch buffer here.

Reuse is safe: record_helper runs on the single-threaded camera_executor
and copies its result into the memmap slot before the next call can
overwrite the scratch.

The depth-convert out= reuse is already covered (test_image_utils.py for
8-bit). The add_false_color tests below cover the output= reuse semantics
directly (still used by the save-edge false-color path) and lock the
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
    # Structural lock: record_helper downconverts an 8-bit capture through the
    # canonical converter into a MainDisplay-owned scratch buffer, threaded with
    # the snapshotted capture depth. MainDisplay needs a full scope to
    # instantiate, so assert on source.
    src = (REPO / 'ui' / 'main_display.py').read_text()
    assert 'convert_to_8bit(' in src
    # A uint16 capture is stored in the memmap VERBATIM (right-aligned). The
    # record path must NOT left-justify it -- a prior convert_to_16bit here
    # double-encoded against the save edge (image_save.write_video_frame), the
    # single depth encoder. So only the 8-bit downconvert reuses a scratch.
    assert 'convert_to_16bit(' not in src
    assert src.count('out=self._record_convert_buf') == 1
    assert 'self._record_capture_depth' in src
    # Record-path false color is gone: the memmap stays mono and colorization
    # moved to the save edges, so the RGB color scratch buffer no longer exists.
    assert '_record_color_buf' not in src
