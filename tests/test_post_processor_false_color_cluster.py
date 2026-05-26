"""Regression tests: post-processor outputs apply false-color (#669, #678).

Bug shape: ``zprojector._zproject`` (#669 recurrence) and
``stitcher._simple_position_stitcher`` (#678) saved their outputs via
bare ``tifffile.imwrite``, skipping the false-color RGB widening that
``image_utils.write_tiff`` does for fluorescence captures. A user with
the ``false_color_16bit`` setting on would see colored per-slice TIFFs
but grayscale projections + stitches -- the most-reported symptom in
Chris's 2026-05-25 bench bundle.

Root cause: same shape at both sites. The false-color gate lived inline
inside ``write_tiff``; post-processors that didn't route through that
function got no gate. Fix extracts the gate to
``image_utils.maybe_apply_false_color`` and calls it from both sinks
before the bare imwrite.

Test shape mirrors the existing
``test_false_color_save_for_8bit_fluorescence.py`` regression matrix.
Each test synthesizes a mono uint8 fluorescence-shape stack on disk,
drives the post-processor, then reads the output via tifffile (not cv2)
and asserts a 3-channel RGB result with the layer's plane populated.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tifffile as tf

from modules import app_context as _app_ctx
from modules.stitcher import Stitcher
from modules.zprojector import ZProjector


@pytest.fixture
def false_color_setting_on():
    """Replace app_context.ctx with a stub whose settings have the
    false-color toggle on. Matches the helper's one-shot settings read
    path."""
    fake_ctx = MagicMock()
    fake_ctx.settings_lock = threading.Lock()
    fake_ctx.settings = {'false_color_16bit': True}
    orig = _app_ctx.ctx
    _app_ctx.ctx = fake_ctx
    try:
        yield fake_ctx
    finally:
        _app_ctx.ctx = orig


def _write_mono_tiff(path, value, shape=(8, 8)):
    """Write a single-channel uint8 grayscale tiff at the given path."""
    arr = np.full(shape, value, dtype=np.uint8)
    tf.imwrite(str(path), arr, compression='lzw')


class TestZProjectorAppliesFalseColorForFluorescence:
    """#669: 8-bit fluorescence z-projections must save as 3-channel RGB
    when the false-color setting is on."""

    @pytest.mark.parametrize(
        'color,expected_channel',
        [
            ('Red', 0),
            ('Green', 1),
            ('Blue', 2),
        ],
    )
    def test_zproject_blue_uint8_writes_rgb_with_false_color_on(
        self, tmp_path, false_color_setting_on, color, expected_channel
    ):
        # Build a 3-slice grayscale stack. The stubbed _ij_helper returns
        # a fixed mono uint8 projection (value 150); the test exercises
        # the save-side widening, not the projection math.
        slice_paths = []
        for i, val in enumerate((100, 150, 200)):
            p = tmp_path / f'slice_{i}.tiff'
            _write_mono_tiff(p, value=val)
            slice_paths.append(p.name)

        df = pd.DataFrame({'Filepath': slice_paths, 'Color': [color] * 3})
        # Bypass ZProjector.__init__ so ObjectiveLoader (which reads
        # _app_ctx.ctx.source_path) doesn't trip on the stubbed ctx.
        # Stub _ij_helper so the test doesn't depend on a Java/Maven
        # runtime; the production code under test is the save-side
        # widening, not the projection math itself.
        zproj = ZProjector.__new__(ZProjector)
        zproj._ij_helper = MagicMock()
        zproj._ij_helper.zproject.return_value = np.full((8, 8), 150, dtype=np.uint8)
        result = zproj._zproject(
            path=tmp_path,
            df=df,
            method='Average',
            output_file_loc=pd.Series(['out.tiff'])[0],
        )
        assert result['status'], f"_zproject failed: {result.get('error')}"

        out = tf.imread(str(tmp_path / 'out.tiff'))
        assert out.ndim == 3, (
            f'{color} z-projection with false-color on must save as '
            f'3-channel RGB, got shape {out.shape}. Pre-fix: bare '
            f'tf.imwrite on the 2D projection bypassed the gate.'
        )
        assert out.shape[2] == 3
        for ch in range(3):
            if ch == expected_channel:
                assert (out[..., ch] == 150).all(), (
                    f'{color} projection plane (index {ch}) must carry '
                    f'the (stubbed) projection value 150'
                )
            else:
                assert (out[..., ch] == 0).all(), (
                    f'non-{color} plane (index {ch}) must be zero'
                )


class TestStitcherAppliesFalseColorForFluorescence:
    """#678: 8-bit fluorescence stitched tiles must save as 3-channel RGB
    when the false-color setting is on."""

    @pytest.mark.parametrize(
        'color,expected_channel',
        [
            ('Red', 0),
            ('Green', 1),
            ('Blue', 2),
        ],
    )
    def test_stitch_uint8_tiles_writes_rgb_with_false_color_on(
        self, tmp_path, false_color_setting_on, color, expected_channel
    ):
        # 2x2 grid of mono uint8 tiles, all at value 200. The stitcher
        # concatenates with no overlap into an 8x8 array; the fix widens
        # the saved output to 3-channel RGB with one plane = 200.
        tile_paths = []
        rows = []
        for ix, x in enumerate((0.0, 1.0)):
            for iy, y in enumerate((0.0, 1.0)):
                p = tmp_path / f'tile_{ix}_{iy}.tiff'
                _write_mono_tiff(p, value=200, shape=(4, 4))
                tile_paths.append(p.name)
                rows.append({
                    'Filepath': p.name,
                    'Color': color,
                    'X': x,
                    'Y': y,
                })
        df = pd.DataFrame(rows)
        result = Stitcher._simple_position_stitcher(
            path=tmp_path,
            df=df,
            output_file_loc=pd.Series(['stitched.tiff'])[0],
        )
        assert result['status'], 'stitcher returned status=False'

        out = tf.imread(str(tmp_path / 'stitched.tiff'))
        assert out.ndim == 3, (
            f'{color} stitched output with false-color on must save as '
            f'3-channel RGB, got shape {out.shape}. Pre-fix: bare '
            f'tf.imwrite on the 2D stitched array bypassed the gate.'
        )
        assert out.shape[2] == 3
        assert (out[..., expected_channel] == 200).all(), (
            f'{color} stitched plane must carry the tile value 200'
        )
        for ch in range(3):
            if ch != expected_channel:
                assert (out[..., ch] == 0).all(), (
                    f'non-{color} plane (index {ch}) must be zero'
                )


class TestPostProcessorSkipsFalseColorForTransmitted:
    """Transmitted layers (BF/PC/DF) must stay grayscale -- they are not
    in ``common_utils.get_image_layers()`` so the gate must pass
    through. Regression guard against widening the gate too far."""

    def test_zproject_bf_stays_grayscale(self, tmp_path, false_color_setting_on):
        slice_paths = []
        for i, val in enumerate((100, 150, 200)):
            p = tmp_path / f'slice_{i}.tiff'
            _write_mono_tiff(p, value=val)
            slice_paths.append(p.name)

        df = pd.DataFrame({'Filepath': slice_paths, 'Color': ['BF'] * 3})
        zproj = ZProjector.__new__(ZProjector)
        zproj._ij_helper = MagicMock()
        zproj._ij_helper.zproject.return_value = np.full((8, 8), 150, dtype=np.uint8)
        result = zproj._zproject(
            path=tmp_path,
            df=df,
            method='Average',
            output_file_loc=pd.Series(['out.tiff'])[0],
        )
        assert result['status']
        out = tf.imread(str(tmp_path / 'out.tiff'))
        assert out.ndim == 2, (
            f'BF z-projection must stay grayscale, got shape {out.shape}'
        )

    def test_stitch_bf_stays_grayscale(self, tmp_path, false_color_setting_on):
        rows = []
        for ix, x in enumerate((0.0, 1.0)):
            for iy, y in enumerate((0.0, 1.0)):
                p = tmp_path / f'tile_{ix}_{iy}.tiff'
                _write_mono_tiff(p, value=200, shape=(4, 4))
                rows.append({'Filepath': p.name, 'Color': 'BF', 'X': x, 'Y': y})
        df = pd.DataFrame(rows)
        result = Stitcher._simple_position_stitcher(
            path=tmp_path,
            df=df,
            output_file_loc=pd.Series(['stitched.tiff'])[0],
        )
        assert result['status']
        out = tf.imread(str(tmp_path / 'stitched.tiff'))
        assert out.ndim == 2, (
            f'BF stitched output must stay grayscale, got shape {out.shape}'
        )
