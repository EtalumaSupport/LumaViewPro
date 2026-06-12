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

import ast
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


def _func_src(name: str) -> str:
    tree = ast.parse(IMAGE_SAVE_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f'{name} not found in {IMAGE_SAVE_SRC}')


def test_tiff_prep_applies_shared_orientation_helper():
    src = _func_src('prepare_image_for_saving')
    assert '_apply_save_orientation(' in src, (
        'prepare_image_for_saving (TIFF prep) must apply the shared '
        'orientation helper. (#690)'
    )


def test_jpg_branch_applies_shared_orientation_helper():
    """The JPG save branch must flip via the shared helper -- not encode
    the raw, unflipped array (the bug)."""
    src = _func_src('save_image')
    # encode_display_jpg must receive the oriented array, not a bare one.
    assert '_apply_save_orientation(array)' in src, (
        'save_image JPG branch must pass _apply_save_orientation(array) to '
        'the encoder so JPG shares the TIFF orientation. (#690)'
    )
    # The helper is the single definition of the convention: no other
    # standalone np.flip(array, 0) orientation step in save_image.
    assert 'np.flip(array, 0)' not in src, (
        'save_image must not re-implement the flip inline -- the shared '
        '_apply_save_orientation helper is the single source. (#690)'
    )


def test_orientation_helper_is_single_definition():
    """Exactly one definition of the orientation convention."""
    # pin-justified: single-canonical-implementation guard; the behavioral
    # twin above proves the pixels, this proves there is only one transform.
    src = IMAGE_SAVE_SRC.read_text()
    assert src.count('def _apply_save_orientation(') == 1
    # The literal flip lives only inside the helper, nowhere else.
    assert src.count('np.flip(array, 0)') == 1, (
        'the np.flip orientation step must exist only inside '
        '_apply_save_orientation. (#690)'
    )
