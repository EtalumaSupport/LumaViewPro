# Copyright Etaluma, Inc.
"""Tests for modules/composite_builder.py — shared composite image builder."""

import numpy as np
import pytest

from modules.composite_builder import build_composite, CHANNEL_RGB_INDEX


class TestChannelMapping:
    """Verify the canonical RGB color mapping."""

    def test_red_maps_to_index_0(self):
        assert CHANNEL_RGB_INDEX['Red'] == 0

    def test_green_maps_to_index_1(self):
        assert CHANNEL_RGB_INDEX['Green'] == 1

    def test_blue_maps_to_index_2(self):
        assert CHANNEL_RGB_INDEX['Blue'] == 2

    def test_lumi_maps_to_blue_channel(self):
        assert CHANNEL_RGB_INDEX['Lumi'] == 2


class TestBuildCompositeNoTransmitted:
    """Composite without transmitted (BF/PC/DF) channel."""

    def test_single_red_channel(self):
        red = np.full((4, 4), 200, dtype=np.uint8)
        img = build_composite(channel_images={'Red': red})
        assert img.shape == (4, 4, 3)
        assert img.dtype == np.uint8
        np.testing.assert_array_equal(img[:, :, 0], 200)  # Red channel
        np.testing.assert_array_equal(img[:, :, 1], 0)    # Green empty
        np.testing.assert_array_equal(img[:, :, 2], 0)    # Blue empty

    def test_single_green_channel(self):
        green = np.full((4, 4), 150, dtype=np.uint8)
        img = build_composite(channel_images={'Green': green})
        np.testing.assert_array_equal(img[:, :, 0], 0)
        np.testing.assert_array_equal(img[:, :, 1], 150)
        np.testing.assert_array_equal(img[:, :, 2], 0)

    def test_single_blue_channel(self):
        blue = np.full((4, 4), 100, dtype=np.uint8)
        img = build_composite(channel_images={'Blue': blue})
        np.testing.assert_array_equal(img[:, :, 0], 0)
        np.testing.assert_array_equal(img[:, :, 1], 0)
        np.testing.assert_array_equal(img[:, :, 2], 100)

    def test_lumi_goes_to_blue_channel(self):
        lumi = np.full((4, 4), 80, dtype=np.uint8)
        img = build_composite(channel_images={'Lumi': lumi})
        np.testing.assert_array_equal(img[:, :, 2], 80)

    def test_all_three_channels(self):
        channels = {
            'Red': np.full((4, 4), 100, dtype=np.uint8),
            'Green': np.full((4, 4), 150, dtype=np.uint8),
            'Blue': np.full((4, 4), 200, dtype=np.uint8),
        }
        img = build_composite(channel_images=channels)
        np.testing.assert_array_equal(img[:, :, 0], 100)
        np.testing.assert_array_equal(img[:, :, 1], 150)
        np.testing.assert_array_equal(img[:, :, 2], 200)

    def test_16bit_dtype(self):
        red = np.full((4, 4), 3000, dtype=np.uint16)
        img = build_composite(channel_images={'Red': red}, dtype=np.uint16, max_value=4095)
        assert img.dtype == np.uint16
        np.testing.assert_array_equal(img[:, :, 0], 3000)

    def test_empty_channels_returns_black(self):
        """Empty channel dict with no transmitted image should raise."""
        with pytest.raises((StopIteration, ValueError)):
            build_composite(channel_images={}, dtype=np.uint8, max_value=255)


class TestBuildCompositeWithTransmitted:
    """Composite with transmitted (BF/PC/DF) base image."""

    def test_transmitted_only_replicates_to_rgb(self):
        bf = np.full((4, 4), 128, dtype=np.uint8)
        img = build_composite(channel_images={}, transmitted_image=bf)
        # Transmitted image replicated to all 3 channels
        np.testing.assert_array_equal(img[:, :, 0], 128)
        np.testing.assert_array_equal(img[:, :, 1], 128)
        np.testing.assert_array_equal(img[:, :, 2], 128)

    def test_fluorescence_above_threshold_replaces_transmitted(self):
        bf = np.full((4, 4), 100, dtype=np.uint8)
        red = np.full((4, 4), 200, dtype=np.uint8)
        img = build_composite(
            channel_images={'Red': red},
            transmitted_image=bf,
            brightness_thresholds={'Red': 50},
        )
        # All pixels above threshold: transmitted replaced with red channel
        np.testing.assert_array_equal(img[:, :, 0], 200)  # Red set
        np.testing.assert_array_equal(img[:, :, 1], 0)    # Others cleared
        np.testing.assert_array_equal(img[:, :, 2], 0)

    def test_fluorescence_below_threshold_keeps_transmitted(self):
        bf = np.full((4, 4), 100, dtype=np.uint8)
        red = np.full((4, 4), 30, dtype=np.uint8)
        img = build_composite(
            channel_images={'Red': red},
            transmitted_image=bf,
            brightness_thresholds={'Red': 50},
        )
        # All pixels below threshold: transmitted image unchanged
        np.testing.assert_array_equal(img[:, :, 0], 100)
        np.testing.assert_array_equal(img[:, :, 1], 100)
        np.testing.assert_array_equal(img[:, :, 2], 100)

    def test_mixed_above_below_threshold(self):
        bf = np.full((4, 4), 100, dtype=np.uint8)
        green = np.array([
            [200, 200, 10, 10],
            [200, 200, 10, 10],
            [10, 10, 200, 200],
            [10, 10, 200, 200],
        ], dtype=np.uint8)
        img = build_composite(
            channel_images={'Green': green},
            transmitted_image=bf,
            brightness_thresholds={'Green': 50},
        )
        # Top-left quadrant: above threshold → green channel set, others cleared
        assert img[0, 0, 0] == 0    # Red cleared
        assert img[0, 0, 1] == 200  # Green set
        assert img[0, 0, 2] == 0    # Blue cleared
        # Top-right: below threshold → transmitted preserved
        assert img[0, 2, 0] == 100
        assert img[0, 2, 1] == 100
        assert img[0, 2, 2] == 100

    def test_two_channels_additive_blending(self):
        bf = np.full((4, 4), 50, dtype=np.uint8)
        red = np.full((4, 4), 200, dtype=np.uint8)
        green = np.full((4, 4), 150, dtype=np.uint8)
        img = build_composite(
            channel_images={'Red': red, 'Green': green},
            transmitted_image=bf,
            brightness_thresholds={'Red': 10, 'Green': 10},
        )
        # Both above threshold: first channel clears and sets, second adds
        np.testing.assert_array_equal(img[:, :, 0], 200)  # Red
        np.testing.assert_array_equal(img[:, :, 1], 150)  # Green added
        np.testing.assert_array_equal(img[:, :, 2], 0)    # Blue cleared by first

    def test_default_threshold_is_zero(self):
        """When no thresholds provided, all pixels above 0 are composited."""
        bf = np.full((4, 4), 100, dtype=np.uint8)
        red = np.full((4, 4), 1, dtype=np.uint8)  # Very dim but > 0
        img = build_composite(
            channel_images={'Red': red},
            transmitted_image=bf,
        )
        # Default threshold 0: any pixel > 0 replaces transmitted
        np.testing.assert_array_equal(img[:, :, 0], 1)
        np.testing.assert_array_equal(img[:, :, 1], 0)
        np.testing.assert_array_equal(img[:, :, 2], 0)

    def test_unknown_channel_name_ignored(self):
        bf = np.full((4, 4), 100, dtype=np.uint8)
        img = build_composite(
            channel_images={'Unknown': np.full((4, 4), 200, dtype=np.uint8)},
            transmitted_image=bf,
        )
        # Unknown channel ignored — transmitted preserved
        np.testing.assert_array_equal(img[:, :, 0], 100)
        np.testing.assert_array_equal(img[:, :, 1], 100)
        np.testing.assert_array_equal(img[:, :, 2], 100)
