"""Regression: every post-processor's result carries its output significant_bits.

The base-class ``load_folder`` reads the output depth off each group's result to
report the input->output depth round-trip. A bench run crashed with
``KeyError('significant_bits')`` because the algorithms wrote the depth to the
output file but dropped it from the returned result -- structurally true for all
five subclasses (three omit the key, one nests it at the wrong level, stack
returns an empty dict). These tests drive the REAL algorithms to a written
artifact and assert the returned result exposes the output depth, which no dict
return satisfied. They replace the base-class stub that faked the contract.
"""

import threading
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tifffile as tf

import modules.app_context as _app_ctx
from modules.composite_generation import CompositeGeneration
from modules.protocol_post_processing_result import PostProcResult
from modules.stitcher import Stitcher
from modules.zprojector import ZProjector


@pytest.fixture
def neutral_ctx():
    """Stub app_context.ctx so the one-shot settings read in the false-color
    gate resolves; BF/mono inputs stay grayscale regardless of the mode."""
    fake_ctx = MagicMock()
    fake_ctx.settings_lock = threading.Lock()
    fake_ctx.settings = {'image_mode': '8bit_mono'}
    orig = _app_ctx.ctx
    _app_ctx.ctx = fake_ctx
    try:
        yield fake_ctx
    finally:
        _app_ctx.ctx = orig


def _write_mono_tiff(path, value, shape=(8, 8)):
    """Write a single-channel uint8 grayscale tiff (8 significant bits)."""
    arr = np.full(shape, value, dtype=np.uint8)
    tf.imwrite(str(path), arr, compression='lzw')


class TestPostProcResultContract:
    """The type's own guarantee: a success cannot exist without its depth."""

    def test_ok_requires_significant_bits(self):
        with pytest.raises(ValueError):
            PostProcResult.ok(significant_bits=None)

    def test_ok_carries_depth(self):
        result = PostProcResult.ok(significant_bits=12)
        assert result.status is True
        assert result.significant_bits == 12

    def test_failed_has_no_depth(self):
        result = PostProcResult.failed('boom')
        assert result.status is False
        assert result.significant_bits is None
        assert result.error == 'boom'


class TestGroupAlgorithmsCarryDepth:
    """Each real subclass, driven to a written artifact, must return a result
    whose significant_bits is the output depth (8 for these uint8 inputs)."""

    def test_zproject_result_carries_depth(self, tmp_path, neutral_ctx):
        slice_paths = []
        for i, val in enumerate((100, 150, 200)):
            p = tmp_path / f'slice_{i}.tiff'
            _write_mono_tiff(p, value=val)
            slice_paths.append(p.name)
        df = pd.DataFrame({'Filepath': slice_paths, 'Color': ['BF'] * 3})
        zproj = ZProjector.__new__(ZProjector)
        result = zproj._group_algorithm(
            path=tmp_path,
            df=df,
            method='Average',
            output_file_loc=pd.Series(['out.tiff'])[0],
        )
        assert isinstance(result, PostProcResult)
        assert result.status is True
        assert result.significant_bits == 8

    def test_stitch_result_carries_depth(self, tmp_path, neutral_ctx):
        rows = []
        for ix, x in enumerate((0.0, 1.0)):
            for iy, y in enumerate((0.0, 1.0)):
                p = tmp_path / f'tile_{ix}_{iy}.tiff'
                _write_mono_tiff(p, value=200, shape=(4, 4))
                rows.append({'Filepath': p.name, 'Color': 'BF', 'X': x, 'Y': y, 'Objective': '20x'})
        df = pd.DataFrame(rows)
        stitcher = Stitcher.__new__(Stitcher)
        result = stitcher._group_algorithm(
            path=tmp_path,
            df=df,
            output_file_loc=pd.Series(['stitched.tiff'])[0],
        )
        assert isinstance(result, PostProcResult)
        assert result.status is True
        assert result.significant_bits == 8

    def test_composite_result_carries_depth(self, tmp_path, neutral_ctx):
        rows = []
        for color, val in (('Red', 120), ('Green', 180)):
            p = tmp_path / f'{color}.tiff'
            _write_mono_tiff(p, value=val)
            rows.append({'Filepath': p.name, 'Color': color})
        df = pd.DataFrame(rows)
        composite = CompositeGeneration.__new__(CompositeGeneration)
        result = composite._group_algorithm(
            path=tmp_path,
            df=df,
            brightness_thresholds_percent={},
            output_file_loc=pd.Series(['composite.tiff'])[0],
        )
        assert isinstance(result, PostProcResult)
        assert result.status is True
        assert result.significant_bits == 8
