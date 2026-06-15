"""Regression for #429: scale-bar color must key off background brightness,
not transmission mode.

The white scale bar is invisible on the bright field of brightfield / phase
contrast, so those get a black bar. Darkfield is a transmitted-light mode but
shows bright subjects on a DARK field, so its bar must stay white like the
fluorescence channels. The bug lumped DF in with BF/PC (via
get_transmitted_layers) and gave it a black, invisible bar.

_compute_scale_bar_overlay returns (overlay, mask, scale_bar_value); the third
element is the bar's grayscale value -- 0 = black, 255 = white (uint8).
"""

import numpy as np
import pytest

from modules.image_utils import _compute_scale_bar_overlay

_OBJECTIVE = {'focal_length': 45.0, 'magnification': 20}


@pytest.mark.parametrize(
    'color, expect_black',
    [
        ('BF', True),  # bright background -> black bar
        ('PC', True),  # bright background -> black bar
        ('DF', False),  # dark background -> white bar (the #429 fix)
        ('Blue', False),
        ('Green', False),
        ('Red', False),
    ],
)
def test_scale_bar_value_keys_off_background_brightness(color, expect_black):
    _, _, scale_bar_value = _compute_scale_bar_overlay(
        height=600,
        width=800,
        dtype=np.uint8,
        is_color=False,
        objective=_OBJECTIVE,
        binning_size=1,
        color=color,
    )
    if expect_black:
        assert scale_bar_value == 0, f'{color} (bright background) should get a black bar'
    else:
        assert scale_bar_value == 255, f'{color} (dark background) should get a white bar'
