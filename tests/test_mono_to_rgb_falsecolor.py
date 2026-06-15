"""Tests for ``mono_to_rgb_falsecolor`` -- the Phase 1 boundary helper.

The single mono -> RGB widening entry point for encode boundaries (live
preview, MP4 / AVI). Save pipeline does NOT call this -- mono-native
save keeps 2D + layer metadata.

Per-layer + dtype matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.image_utils import mono_to_rgb_falsecolor


@pytest.fixture
def mono_uint8():
    return np.full((8, 8), 200, dtype=np.uint8)


@pytest.fixture
def mono_uint16():
    return np.full((8, 8), 42000, dtype=np.uint16)


@pytest.mark.parametrize('dtype_fixture', ['mono_uint8', 'mono_uint16'])
class TestFluorescenceLayers:
    """Red / Green / Blue / Lumi each place the source in the canonical
    RGB index. Other channels are zero."""

    def test_red_layer_populates_index_0(self, request, dtype_fixture):
        mono = request.getfixturevalue(dtype_fixture)
        rgb = mono_to_rgb_falsecolor(mono, 'Red')
        assert rgb.shape == (8, 8, 3)
        assert rgb.dtype == mono.dtype
        assert (rgb[:, :, 0] == mono).all()
        assert (rgb[:, :, 1] == 0).all()
        assert (rgb[:, :, 2] == 0).all()

    def test_green_layer_populates_index_1(self, request, dtype_fixture):
        mono = request.getfixturevalue(dtype_fixture)
        rgb = mono_to_rgb_falsecolor(mono, 'Green')
        assert (rgb[:, :, 0] == 0).all()
        assert (rgb[:, :, 1] == mono).all()
        assert (rgb[:, :, 2] == 0).all()

    def test_blue_layer_populates_index_2(self, request, dtype_fixture):
        mono = request.getfixturevalue(dtype_fixture)
        rgb = mono_to_rgb_falsecolor(mono, 'Blue')
        assert (rgb[:, :, 0] == 0).all()
        assert (rgb[:, :, 1] == 0).all()
        assert (rgb[:, :, 2] == mono).all()

    def test_lumi_layer_falls_into_blue_index(self, request, dtype_fixture):
        mono = request.getfixturevalue(dtype_fixture)
        rgb = mono_to_rgb_falsecolor(mono, 'Lumi')
        assert (rgb[:, :, 2] == mono).all()
        assert (rgb[:, :, 0] == 0).all()
        assert (rgb[:, :, 1] == 0).all()


class TestTransmittedAndUnknownLayers:
    """BF / PC / DF / unknown layers tile mono into all three channels --
    grayscale RGB so that downstream RGB-only consumers (cv2.VideoWriter)
    don't see a Red-channel-only signal."""

    @pytest.mark.parametrize('layer', ['BF', 'PC', 'DF', 'WidgetUnknown'])
    def test_layer_tiles_to_grayscale_rgb(self, layer):
        mono = np.full((8, 8), 100, dtype=np.uint8)
        rgb = mono_to_rgb_falsecolor(mono, layer)
        assert rgb.shape == (8, 8, 3)
        for ch in range(3):
            assert (rgb[:, :, ch] == 100).all(), (
                f'layer {layer}: channel {ch} should tile mono value, got {rgb[:, :, ch].flatten()}'
            )


class TestInputShapeContract:
    def test_3d_input_raises(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match='2D'):
            mono_to_rgb_falsecolor(rgb, 'Red')

    def test_1d_input_raises(self):
        flat = np.zeros(64, dtype=np.uint8)
        with pytest.raises(ValueError, match='2D'):
            mono_to_rgb_falsecolor(flat, 'Red')


class TestSourcePreservation:
    """Function returns a new array; the source mono is not modified."""

    def test_source_not_mutated(self):
        mono = np.full((4, 4), 50, dtype=np.uint16)
        mono_copy = mono.copy()
        _ = mono_to_rgb_falsecolor(mono, 'Red')
        assert (mono == mono_copy).all()
