"""Post-proc combiners must reject inputs not stitched to one geometry.

Compositing channels, projecting a Z-stack, and assembling a hyperstack all
require every input image to share one (H, W) canvas. When a stitcher produces
per-layer divergent canvases, the raw failure is a cryptic numpy broadcast /
stack error that hides which images disagree. These pin the contract: a
geometry violation raises a clear, actionable ValueError; matching inputs
combine normally.
"""

import numpy as np
import pytest

from modules import image_utils
from modules.composite_builder import build_composite
from modules.zprojection import ZProjectMethod, zproject


def test_require_uniform_geometry_rejects_mismatch():
    with pytest.raises(ValueError, match=r'not stitched to one geometry'):
        image_utils.require_uniform_geometry(
            [
                ('Blue', np.zeros((3829, 3644), np.uint8)),
                ('Green', np.zeros((3840, 3637), np.uint8)),
            ],
            operation='composite this well',
        )


def test_require_uniform_geometry_accepts_uniform():
    image_utils.require_uniform_geometry(
        [('Blue', np.zeros((16, 16), np.uint8)), ('Green', np.zeros((16, 16), np.uint8))],
        operation='composite this well',
    )


def test_build_composite_rejects_mismatched_channels():
    channels = {
        'Red': np.zeros((8, 8), np.uint8),
        'Green': np.zeros((9, 7), np.uint8),
    }
    with pytest.raises(ValueError, match=r'not stitched to one geometry'):
        build_composite(channel_images=channels)


def test_build_composite_accepts_uniform_channels():
    channels = {
        'Red': np.full((8, 8), 100, np.uint8),
        'Green': np.full((8, 8), 50, np.uint8),
    }
    out = build_composite(channel_images=channels)
    assert out.shape == (8, 8, 3)


def test_zproject_rejects_mismatched_slices():
    slices = [np.zeros((8, 8), np.uint8), np.zeros((8, 9), np.uint8)]
    with pytest.raises(ValueError, match=r'not stitched to one geometry'):
        zproject(images_data=slices, method=ZProjectMethod.Max)


def test_zproject_accepts_uniform_slices():
    slices = [np.full((8, 8), 10, np.uint8), np.full((8, 8), 200, np.uint8)]
    out = zproject(images_data=slices, method=ZProjectMethod.Max)
    assert out.shape == (8, 8)
    assert out.max() == 200
