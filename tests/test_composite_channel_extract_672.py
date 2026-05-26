"""Regression test: composite_generation extracts single-plane false-color
TIFFs without luminance attenuation (#672 round 4).

Bug shape: Chris's 2026-05-25 bundle showed 12-bit fluorescence
composites coming out "mostly green" + the 8-bit composite being
LARGER than the 12-bit (18 MB vs 8 MB). Root cause traced to two
``cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`` sites in
``modules/composite_generation.py`` that collapsed source TIFFs to
mono via luminance weighting (Y = 0.299R + 0.587G + 0.114B).

After the #669 cluster fix shipped (LVP `3e2c8b9`), per-channel
fluorescence TIFFs save 3-channel-RGB with one plane populated. The
RGB2GRAY luminance collapse on those TIFFs attenuates Blue by 88.6%,
Red by 70.1%, Green by 41.3% -- producing the "mostly green"
symptom and the file-size inversion (Red+Blue planes mostly zero
post-collapse compress better in LZW).

Fix: route both call sites through ``image_utils.rgb_image_to_gray``,
which detects the single-populated-plane case and uses
``np.amax(axis=2)`` to preserve full intensity. Mono 2D inputs pass
through unchanged.

The tests below exercise the composite generation by driving its
inner ``_combine_results`` (the build path that consumes the
read-from-disk source TIFFs) via the public ``_combine_results``
shape; or, more surgically here, exercise the underlying
``image_utils.rgb_image_to_gray`` helper that the fix routes through.
The full composite pipeline test is the integration shape; the unit
test below pins the contract the fix relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.image_utils import rgb_image_to_gray


def _single_plane_falsecolor(value, channel_index, shape=(8, 8), dtype=np.uint8):
    """Construct a 3-channel array with one populated plane (the
    false-color save shape) and zeros in the other two planes."""
    h, w = shape
    arr = np.zeros((h, w, 3), dtype=dtype)
    arr[:, :, channel_index] = value
    return arr


class TestRgbImageToGrayPreservesSinglePlaneFalseColor:
    """Pre-fix: ``cv2.cvtColor(..., RGB2GRAY)`` attenuated single-plane
    fluorescence data by 41-89% of its original intensity (the
    composite "mostly green" symptom). Post-fix: full value preserved
    via max-axis extraction."""

    @pytest.mark.parametrize(
        'color,channel_index',
        [
            ('Red', 0),
            ('Green', 1),
            ('Blue', 2),
        ],
    )
    def test_uint8_single_plane_value_preserved(self, color, channel_index):
        arr = _single_plane_falsecolor(value=200, channel_index=channel_index)
        out = rgb_image_to_gray(arr)
        assert out.shape == (8, 8), (
            f'{color} 3-channel input must collapse to 2D mono, got '
            f'shape {out.shape}'
        )
        assert (out == 200).all(), (
            f'{color} single-plane value 200 must be preserved at full '
            f'intensity (pre-fix RGB2GRAY would have produced '
            f'{int(200 * (0.299, 0.587, 0.114)[channel_index])} via '
            f'luminance weighting)'
        )

    @pytest.mark.parametrize(
        'color,channel_index',
        [
            ('Red', 0),
            ('Green', 1),
            ('Blue', 2),
        ],
    )
    def test_uint16_single_plane_value_preserved(self, color, channel_index):
        arr = _single_plane_falsecolor(
            value=3000, channel_index=channel_index, dtype=np.uint16
        )
        out = rgb_image_to_gray(arr)
        assert out.shape == (8, 8)
        assert (out == 3000).all(), (
            f'{color} 12-bit single-plane value 3000 must be preserved '
            f'(pre-fix produced the "mostly green" composite + file-size '
            f'inversion symptom for 12-bit data)'
        )


class TestRgbImageToGrayPassesGrayscaleThrough:
    """Mono 2D inputs (the common path after per-channel save) must
    return unchanged -- no false collapse, no shape change."""

    def test_uint8_grayscale_2d_passes_through(self):
        arr = np.full((8, 8), 100, dtype=np.uint8)
        out = rgb_image_to_gray(arr)
        assert out is arr or (out.shape == (8, 8) and (out == 100).all())

    def test_uint16_grayscale_2d_passes_through(self):
        arr = np.full((8, 8), 2500, dtype=np.uint16)
        out = rgb_image_to_gray(arr)
        assert out is arr or (out.shape == (8, 8) and (out == 2500).all())


class TestRgbImageToGrayPathDocumentedInCompositeGeneration:
    """Source-text contract: composite_generation.py must call
    rgb_image_to_gray (not cv2.cvtColor RGB2GRAY directly) to honor the
    single-plane preservation. Catches a future revert that swaps back
    to luminance collapse."""

    def test_no_cv2_cvtcolor_rgb2gray_in_composite_generation(self):
        import re
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / 'modules'
            / 'composite_generation.py'
        ).read_text(encoding='utf-8')
        # Allow the pattern inside comments / docstrings (the file
        # documents the prior bug shape in commit-message commentary,
        # but no live code path may use it).
        # Scrub comment lines:
        live = '\n'.join(
            line for line in src.splitlines() if not line.lstrip().startswith('#')
        )
        m = re.search(r'cv2\.cvtColor\([^)]*RGB2GRAY', live)
        assert m is None, (
            f'cv2.cvtColor(..., RGB2GRAY) reintroduced in '
            f'composite_generation.py at: {m.group(0) if m else "?"}. '
            f'Use image_utils.rgb_image_to_gray instead -- it preserves '
            f'single-plane false-color data via max-axis.'
        )

    def test_rgb_image_to_gray_called_in_composite_generation(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / 'modules'
            / 'composite_generation.py'
        ).read_text(encoding='utf-8')
        assert 'image_utils.rgb_image_to_gray' in src, (
            'composite_generation.py must collapse 3-channel TIFFs via '
            'image_utils.rgb_image_to_gray (preserves single-plane '
            'false-color data) rather than bare cv2.cvtColor.'
        )
