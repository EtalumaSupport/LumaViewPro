# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: BF/PC (black) scale bars are actually drawn.

Bright-background channels (BF, PC) use a black scale bar (value 0). The
overlay mask was built from nonzero canvas pixels -- but a black bar drawn at
value 0 onto a zeroed canvas leaves the mask empty, so add_scale_bar wrote
nothing and the BF/PC scale bar never appeared (white-bar darkfield /
fluorescence channels rendered fine, which is why only some channels showed a
bar). The geometry is now rendered with a nonzero sentinel so the mask captures
the bar+text location, and the real value (0 for black) is written there.

Scale-bar rendering is pure numpy/cv2; get_pixel_size falls back to default
optics when there is no app context, so this runs without the Kivy app.
"""

from __future__ import annotations

import numpy as np

import modules.image_utils as image_utils

_OBJECTIVE = {'focal_length': 10.0, 'magnification': 20}


def _add(image, color):
    image_utils._scale_bar_cache = {}  # module-global cache; isolate each call
    return image_utils.add_scale_bar(
        image=image, objective=_OBJECTIVE, binning_size=1, color=color, significant_bits=8
    )


def test_bf_black_bar_is_drawn():
    # Bright, uniform image (no zeros anywhere). A black bar must introduce
    # 0-valued pixels; before the fix the mask was empty and nothing changed.
    image = np.full((400, 400), 200, dtype=np.uint8)
    result = _add(image, color='BF')
    assert (result == 0).any(), 'BF (black) scale bar was not drawn at all'
    # Bar + text is a meaningful run of pixels, not a stray single zero.
    assert (result == 0).sum() > 20


def test_pc_black_bar_is_drawn():
    image = np.full((400, 400), 200, dtype=np.uint8)
    result = _add(image, color='PC')
    assert (result == 0).any(), 'PC (black) scale bar was not drawn'


def test_white_bar_still_drawn():
    # A non-bright channel on a dark image -> white bar (value 255) must
    # still appear (guards against regressing the path that already worked).
    image = np.zeros((400, 400), dtype=np.uint8)
    result = _add(image, color='Red')
    assert (result == 255).any(), 'white scale bar regressed'


def _add_depth(image, color, significant_bits):
    image_utils._scale_bar_cache = {}  # module-global cache; isolate each call
    return image_utils.add_scale_bar(
        image=image,
        objective=_OBJECTIVE,
        binning_size=1,
        color=color,
        significant_bits=significant_bits,
    )


def test_white_bar_value_tracks_significant_bits():
    # The white bar's pixel value is the payload max for the frame's depth, so
    # it maps to full white after the depth-keyed 8-bit downconvert. A summed
    # 16-bit-container frame must get 65535, not the 12-bit 4095 (which would
    # downconvert to a dim ~16/255 gray bar).
    r12 = _add_depth(np.zeros((400, 400), dtype=np.uint16), color='Red', significant_bits=12)
    assert int(r12.max()) == 4095

    r16 = _add_depth(np.zeros((400, 400), dtype=np.uint16), color='Red', significant_bits=16)
    assert int(r16.max()) == 65535


def test_uint8_bar_value_unchanged():
    r = _add_depth(np.zeros((400, 400), dtype=np.uint8), color='Red', significant_bits=8)
    assert int(r.max()) == 255
