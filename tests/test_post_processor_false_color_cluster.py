"""Regression tests: post-processor outputs apply false-color (#669, #678).

Bug shape: ``zprojector._zproject`` (#669 recurrence) and
``stitcher._simple_position_stitcher`` (#678) saved their outputs via
bare ``tifffile.imwrite``, skipping the false-color RGB widening that
``image_utils.write_tiff`` does for fluorescence captures. A user with
a false-color image mode selected would see colored per-slice TIFFs
but grayscale projections + stitches -- the most-reported symptom in
the 2026-05-25 bench bundle.

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
    """Replace app_context.ctx with a stub whose image_mode resolves to
    false-color RGB. Matches the helper's one-shot settings read path."""
    fake_ctx = MagicMock()
    fake_ctx.settings_lock = threading.Lock()
    fake_ctx.settings = {'image_mode': '12bit_false_color_rgb'}
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
    write_tiff routing can pass the layer through to the colormap gate.

    The earlier TestStitcherAppliesFalseColorForFluorescence tests skip
    _group_algorithm and call _simple_position_stitcher directly with a
    Color-bearing df, so they cannot catch a column-list narrowing in
    _group_algorithm. This class exercises the production-shape path."""

    def test_group_algorithm_slice_preserves_color(self, tmp_path, false_color_setting_on):
        rows = []
        for ix, x in enumerate((0.0, 1.0)):
            for iy, y in enumerate((0.0, 1.0)):
                p = tmp_path / f'tile_{ix}_{iy}.tiff'
                _write_mono_tiff(p, value=200, shape=(4, 4))
                rows.append(
                    {
                        'Filepath': p.name,
                        'Color': 'Green',
                        'X': x,
                        'Y': y,
                        'Well': 'A1',
                        'Z-Slice': 0,
                        'Objective': '20x',
                        'Scan Count': 0,
                    }
                )
        df = pd.DataFrame(rows)
        stitcher = Stitcher.__new__(Stitcher)
        result = stitcher._group_algorithm(
            path=tmp_path,
            df=df,
            output_file_loc=pd.Series(['stitched.tiff'])[0],
        )
        assert result.status, (
            f'_group_algorithm returned status=False: {result.error}. '
            f'Narrowing the df subset without Color drops the layer the '
            f'write_tiff routing needs for the colormap gate.'
        )
        with tf.TiffFile(str(tmp_path / 'stitched.tiff')) as tif:
            page = tif.pages[0]
            out = page.asarray()
            photometric = page.tags['PhotometricInterpretation'].value
            colormap_tag = page.tags.get('ColorMap')
        assert out.ndim == 2, (
            f'Stitched 8-bit fluorescence is mono on disk; layer color rides '
            f'as the TIFF colormap tag. Got shape {out.shape}.'
        )
        assert (out == 200).all(), 'Stitched pixels must carry the tile value'
        assert photometric == tf.PHOTOMETRIC.PALETTE, (
            f'8-bit fluorescence must save with PALETTE photometric so Windows '
            f'Preview and FIJI render the layer color. Got {photometric}.'
        )
        assert colormap_tag is not None, 'PALETTE photometric requires a ColorMap tag'


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
        result = zproj._zproject(
            path=tmp_path,
            df=df,
            method='Average',
            output_file_loc=pd.Series(['out.tiff'])[0],
        )
        assert result['status']
        out = tf.imread(str(tmp_path / 'out.tiff'))
        assert out.ndim == 2, f'BF z-projection must stay grayscale, got shape {out.shape}'

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
        assert out.ndim == 2, f'BF stitched output must stay grayscale, got shape {out.shape}'
