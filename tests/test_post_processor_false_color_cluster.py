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


class TestStitcherGroupAlgorithmCarriesColor:
    """Production call path: _group_algorithm receives the full-shape df
    from protocol_post_processor.load_folder and slices it before passing
    to _simple_position_stitcher. The slice must include 'Color' so the
    downstream false-color gate has it to read.

    The earlier TestStitcherAppliesFalseColorForFluorescence tests skip
    _group_algorithm and call _simple_position_stitcher directly with a
    Color-bearing df, so they cannot catch a column-list narrowing in
    _group_algorithm. This class exercises the production-shape path."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            'Deferred to the 1d.5 follow-up commit that routes Stitcher through '
            'image_utils.write_tiff. Today Stitcher saves mono via bare tf.imwrite '
            "and narrows the df subset without 'Color' (the 'Color' carry-through "
            'is unnecessary until the write_tiff routing lands). Flips green when '
            'the 1d.5 follow-up commit migrates Stitcher to write_tiff(..., color=...).'
        ),
    )
    def test_group_algorithm_slice_preserves_color(self, tmp_path, false_color_setting_on):
        rows = []
        for ix, x in enumerate((0.0, 1.0)):
            for iy, y in enumerate((0.0, 1.0)):
                p = tmp_path / f'tile_{ix}_{iy}.tiff'
                _write_mono_tiff(p, value=200, shape=(4, 4))
                rows.append({
                    'Filepath': p.name,
                    'Color': 'Green',
                    'X': x,
                    'Y': y,
                    'Well': 'A1',
                    'Z-Slice': 0,
                    'Objective': '20x',
                    'Scan Count': 0,
                })
        df = pd.DataFrame(rows)
        stitcher = Stitcher.__new__(Stitcher)
        result = stitcher._group_algorithm(
            path=tmp_path,
            df=df,
            output_file_loc=pd.Series(['stitched.tiff'])[0],
        )
        assert result['status'], (
            f"_group_algorithm returned status=False: {result.get('error')}. "
            f'Pre-fix: narrowing the df to [Filepath, X, Y] dropped Color, '
            f"so _simple_position_stitcher raised KeyError on df['Color']."
        )
        out = tf.imread(str(tmp_path / 'stitched.tiff'))
        assert out.ndim == 3 and out.shape[2] == 3, (
            f'Stitched output via _group_algorithm with false-color on must '
            f'save as 3-channel RGB, got shape {out.shape}.'
        )
        assert (out[..., 1] == 200).all(), 'Green plane must carry tile value'


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
