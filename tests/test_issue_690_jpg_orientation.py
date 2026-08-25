# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#690 regression: JPG export shares the TIFF save-orientation.

Bug
---
The TIFF save path vertically flips the camera array (inside
prepare_image_for_saving), but the JPG export skipped that prep and
encoded the raw array -- so JPGs saved upside-down relative to the
TIFFs and the tiling pipeline that reads them.

Fix
---
The flip is now a single shared helper, image_save._apply_save_orientation,
called by BOTH the TIFF prep and the JPG save branch. Bit depth, color
baking, and metadata stay format-specific (JPG is an 8-bit rendered
display image; TIFF / OME-TIFF carry the 16-bit data + metadata) -- only
the orientation is shared, so the two formats can never diverge again.

Note: orientation is ultimately a per-camera property (different sensors
deliver different native orientations). Normalizing it at the driver
layer is tracked separately; this helper is the single seam that change
will plug into.
"""

from __future__ import annotations

import pathlib

import numpy as np

from modules.image_save import _apply_save_orientation


REPO = pathlib.Path(__file__).resolve().parent.parent
IMAGE_SAVE_SRC = REPO / 'modules' / 'image_save.py'


def test_apply_save_orientation_is_vertical_flip():
    """Top row must end up at the bottom (np.flip axis 0)."""
    arr = np.arange(6, dtype=np.uint8).reshape(3, 2)  # rows 0,1,2 distinct
    out = _apply_save_orientation(arr)
    assert np.array_equal(out, np.flip(arr, 0)), 'orientation helper must flip vertically'
    # row order reversed
    assert np.array_equal(out[0], arr[-1]) and np.array_equal(out[-1], arr[0])


def test_tiff_and_jpg_save_identical_orientation(tmp_path, monkeypatch):
    """Both formats must land the SAME (vertically flipped) pixels on
    disk -- the bug was the JPG branch skipping the flip and saving
    upside-down relative to the TIFFs. (#690)"""
    import cv2
    import tifffile

    from modules import image_save

    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[0, :] = 255  # bright TOP edge in camera orientation
    stub_metadata = {
        'plate_pos_mm': {'x': 0.0, 'y': 0.0},
        'z_pos_um': 0.0,
        'objective': {'name': 'stub'},
        'exposure_time_ms': 10.0,
        'gain_db': 0.0,
        'illumination_ma': 0,
        'pixel_size_um': 1.0,
        'channel': 'BF',
        'datetime': '2026:06:12 00:00:00',
    }
    monkeypatch.setattr(
        image_save,
        'generate_image_metadata',
        lambda scope, channel, x, y, z: dict(stub_metadata),
    )
    from types import SimpleNamespace

    scope = SimpleNamespace(
        imaging=SimpleNamespace(capture_frame_depth=lambda array, sum_count=1: 8)
    )
    tiff_path = image_save.save_image(
        scope,
        arr.copy(),
        save_folder=str(tmp_path),
        file_root='o_',
        append='t',
        channel='BF',
        false_color_on=False,
        tail_id_mode=None,
        output_format='TIFF',
        save_encoding='8bit',
        significant_bits=8,
    )
    jpg_path = image_save.save_image(
        scope,
        arr.copy(),
        save_folder=str(tmp_path),
        file_root='o_',
        append='j',
        channel='BF',
        false_color_on=False,
        tail_id_mode=None,
        output_format='JPG',
        jpeg_quality=95,
        save_encoding='8bit',
        significant_bits=8,
    )
    tiff_px = tifffile.imread(tiff_path)
    jpg_px = cv2.imdecode(
        np.frombuffer(pathlib.Path(jpg_path).read_bytes(), np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    # The bright camera-top edge must land at the BOTTOM in both formats.
    assert tiff_px[-1].mean() > 200 and tiff_px[0].mean() < 50, 'TIFF must save vertically flipped'
    assert jpg_px[-1].mean() > 200 and jpg_px[0].mean() < 50, (
        'JPG must share the TIFF save orientation, not the raw array'
    )


def test_orientation_helper_is_single_definition():
    """Exactly one definition of the orientation convention."""
    # pin-justified: single-canonical-implementation guard; the behavioral
    # twin above proves the pixels, this proves there is only one transform.
    src = IMAGE_SAVE_SRC.read_text()
    assert src.count('def _apply_save_orientation(') == 1
    # The literal flip lives only inside the helper, nowhere else.
    assert src.count('np.flip(array, 0)') == 1, (
        'the np.flip orientation step must exist only inside _apply_save_orientation. (#690)'
    )
