# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A mono-to-colour widening reuses the caller's output buffer.

The mono_to_rgb_falsecolor tests cover its output= reuse, which the
composite build passes so that each channel does not allocate a fresh
~W*H*3 array; a buffer of another shape or dtype is not used. The
record-path structural pin guards the depth contract at the recording
write edge.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np


REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from modules import image_utils
from modules.layer_record import release_color
from tests.stock_channels import LED1


def test_mono_to_rgb_falsecolor_reuses_output_buffer():
    src = np.arange(48, dtype=np.uint8).reshape(6, 8)
    out = np.empty((6, 8, 3), dtype=np.uint8)
    result = image_utils.mono_to_rgb_falsecolor(src, release_color(LED1), output=out)
    assert result is out, 'must write into the caller buffer, not allocate'
    fresh = image_utils.mono_to_rgb_falsecolor(src, release_color(LED1))
    assert np.array_equal(result, fresh), 'reused-buffer result must match fresh'


def test_mono_to_rgb_falsecolor_falls_back_on_shape_mismatch():
    src = np.zeros((6, 8), dtype=np.uint8)
    out = np.empty((3, 4, 3), dtype=np.uint8)  # wrong shape
    result = image_utils.mono_to_rgb_falsecolor(src, release_color(LED1), output=out)
    assert result is not out, 'mismatched output must fall back to a fresh array'
    assert result.shape == (6, 8, 3)


def test_mono_to_rgb_falsecolor_falls_back_on_dtype_mismatch():
    src = np.zeros((6, 8), dtype=np.uint8)
    out = np.empty((6, 8, 3), dtype=np.uint16)  # wrong dtype
    result = image_utils.mono_to_rgb_falsecolor(src, release_color(LED1), output=out)
    assert result is not out, 'mismatched-dtype output must fall back to a fresh array'
    assert result.dtype == np.uint8


def test_record_write_edge_does_not_left_justify():
    # A uint16 capture travels to the write edge VERBATIM (right-aligned).
    # The record path must NOT left-justify it -- a prior convert_to_16bit
    # double-encoded against the save edge (image_save.write_video_frame),
    # the single depth encoder. Structural source pin on the controller,
    # which owns the record write edge post-cutover.
    src = (REPO / 'modules' / 'manual_recording.py').read_text()
    assert 'convert_to_16bit(' not in src
