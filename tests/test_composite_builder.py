# Copyright Etaluma, Inc.
"""Tests for modules/composite_builder.py -- shared composite image builder."""

import numpy as np
import pytest

import modules.image_utils as image_utils
from modules.composite_builder import build_composite


def test_a_composite_of_pure_primaries_puts_each_channel_on_its_own_plane():
    """Red, Green and Blue, one each, build the composite they always have.

    Every pixel's planes come from the channels alone: with no transmitted
    base, each channel's 8-bit value on its own plane; with one, a pixel where
    any channel reaches its cutoff carries each channel at or above its cutoff
    on its plane and nothing else, and every other pixel keeps the base's grey.
    The 120 cases walk three input depths, a base or none, four cutoffs and
    five channel sets, each with a row of black and a row at full scale, and
    every case that differs is named.
    """
    plane_of = {'Red': 0, 'Green': 1, 'Blue': 2}
    rng = np.random.default_rng(0)
    differing = []
    cases = 0
    for bits, dtype in ((8, np.uint8), (12, np.uint16), (16, np.uint16)):
        full_scale = (1 << bits) - 1
        for with_base in (False, True):
            for cutoff in (1, 50, 128, 255):
                for keys in (
                    ('Red', 'Green', 'Blue'),
                    ('Red', 'Green'),
                    ('Green', 'Blue'),
                    ('Blue',),
                    ('Red', 'Blue'),
                ):
                    channels = {}
                    for key in keys:
                        image = rng.integers(0, full_scale + 1, (16, 16)).astype(dtype)
                        image[0, :] = 0
                        image[1, :] = full_scale
                        channels[key] = image
                    base = (
                        rng.integers(0, full_scale + 1, (16, 16)).astype(dtype)
                        if with_base
                        else None
                    )

                    eight_bit = {
                        key: image_utils.convert_to_8bit(image, bits)
                        for key, image in channels.items()
                    }
                    if base is None:
                        expected = np.zeros((16, 16, 3), np.uint8)
                        for key, image in eight_bit.items():
                            expected[:, :, plane_of[key]] = image
                    else:
                        base8 = image_utils.convert_to_8bit(base, bits)
                        expected = np.repeat(base8[:, :, None], 3, axis=2)
                        lit = np.zeros((16, 16), bool)
                        overlay = np.zeros((16, 16, 3), np.uint8)
                        for key, image in eight_bit.items():
                            reached = image >= cutoff
                            lit |= reached
                            overlay[reached, plane_of[key]] = image[reached]
                        expected[lit] = overlay[lit]

                    built = build_composite(
                        channel_images=channels,
                        significant_bits=bits,
                        transmitted_image=base,
                        brightness_thresholds=dict.fromkeys(keys, cutoff) if with_base else None,
                    )
                    cases += 1
                    if not np.array_equal(built, expected):
                        differing.append((bits, with_base, cutoff, keys))
    assert cases == 120
    assert differing == []


class TestBuildCompositeNoTransmitted:
    """Composite without transmitted (BF/PC/DF) channel."""

    def test_single_red_channel(self):
        red = np.full((4, 4), 200, dtype=np.uint8)
        img = build_composite(channel_images={'Red': red}, significant_bits=8)
        assert img.shape == (4, 4, 3)
        assert img.dtype == np.uint8
        np.testing.assert_array_equal(img[:, :, 0], 200)  # Red channel
        np.testing.assert_array_equal(img[:, :, 1], 0)  # Green empty
        np.testing.assert_array_equal(img[:, :, 2], 0)  # Blue empty

    def test_lumi_goes_to_blue_channel(self):
        lumi = np.full((4, 4), 80, dtype=np.uint8)
        img = build_composite(channel_images={'Lumi': lumi}, significant_bits=8)
        np.testing.assert_array_equal(img[:, :, 2], 80)

    def test_all_three_channels(self):
        channels = {
            'Red': np.full((4, 4), 100, dtype=np.uint8),
            'Green': np.full((4, 4), 150, dtype=np.uint8),
            'Blue': np.full((4, 4), 200, dtype=np.uint8),
        }
        img = build_composite(channel_images=channels, significant_bits=8)
        np.testing.assert_array_equal(img[:, :, 0], 100)
        np.testing.assert_array_equal(img[:, :, 1], 150)
        np.testing.assert_array_equal(img[:, :, 2], 200)

    def test_12bit_input_downconverts_to_8bit(self):
        """A composite is always 8-bit: a 12-bit input channel is downconverted,
        not stored right-aligned in a 16-bit container (which renders near-black
        in any viewer that ignores the depth tag)."""
        red = np.full((4, 4), 3000, dtype=np.uint16)
        img = build_composite(channel_images={'Red': red}, significant_bits=12)
        assert img.dtype == np.uint8
        # The red plane equals the canonical downconvert of the input -- not the
        # raw 12-bit value (3000), which the old build stored verbatim.
        expected = image_utils.convert_to_8bit(red, 12)
        np.testing.assert_array_equal(img[:, :, 0], expected)
        np.testing.assert_array_equal(img[:, :, 1], 0)
        np.testing.assert_array_equal(img[:, :, 2], 0)

    def test_full_scale_12bit_maps_to_full_8bit(self):
        """A full-scale 12-bit channel (4095) becomes a full-scale 8-bit channel
        (255) -- the composite is bright, not the ~6%-of-full-scale that made
        12-bit composites read as black."""
        red = np.full((4, 4), 4095, dtype=np.uint16)
        img = build_composite(channel_images={'Red': red}, significant_bits=12)
        assert img.dtype == np.uint8
        np.testing.assert_array_equal(img[:, :, 0], 255)

    def test_empty_channels_returns_black(self):
        """Empty channel dict with no transmitted image should raise."""
        with pytest.raises((StopIteration, ValueError)):
            build_composite(channel_images={}, significant_bits=8)


class TestBuildCompositeWithTransmitted:
    """Composite with transmitted (BF/PC/DF) base image."""

    def test_transmitted_only_replicates_to_rgb(self):
        bf = np.full((4, 4), 128, dtype=np.uint8)
        img = build_composite(channel_images={}, significant_bits=8, transmitted_image=bf)
        # Transmitted image replicated to all 3 channels
        np.testing.assert_array_equal(img[:, :, 0], 128)
        np.testing.assert_array_equal(img[:, :, 1], 128)
        np.testing.assert_array_equal(img[:, :, 2], 128)

    def test_mixed_above_below_threshold(self):
        bf = np.full((4, 4), 100, dtype=np.uint8)
        green = np.array(
            [
                [200, 200, 10, 10],
                [200, 200, 10, 10],
                [10, 10, 200, 200],
                [10, 10, 200, 200],
            ],
            dtype=np.uint8,
        )
        img = build_composite(
            channel_images={'Green': green},
            significant_bits=8,
            transmitted_image=bf,
            brightness_thresholds={'Green': 50},
        )
        # Top-left quadrant: above threshold -> green channel set, others cleared
        assert img[0, 0, 0] == 0  # Red cleared
        assert img[0, 0, 1] == 200  # Green set
        assert img[0, 0, 2] == 0  # Blue cleared
        # Top-right: below threshold -> transmitted preserved
        assert img[0, 2, 0] == 100
        assert img[0, 2, 1] == 100
        assert img[0, 2, 2] == 100

    def test_two_channels_additive_blending(self):
        bf = np.full((4, 4), 50, dtype=np.uint8)
        red = np.full((4, 4), 200, dtype=np.uint8)
        green = np.full((4, 4), 150, dtype=np.uint8)
        img = build_composite(
            channel_images={'Red': red, 'Green': green},
            significant_bits=8,
            transmitted_image=bf,
            brightness_thresholds={'Red': 10, 'Green': 10},
        )
        # Both above threshold: first channel clears and sets, second adds
        np.testing.assert_array_equal(img[:, :, 0], 200)  # Red
        np.testing.assert_array_equal(img[:, :, 1], 150)  # Green added
        np.testing.assert_array_equal(img[:, :, 2], 0)  # Blue cleared by first

    def test_default_threshold_is_zero(self):
        """When no thresholds provided, all pixels above 0 are composited."""
        bf = np.full((4, 4), 100, dtype=np.uint8)
        red = np.full((4, 4), 1, dtype=np.uint8)  # Very dim but > 0
        img = build_composite(
            channel_images={'Red': red},
            significant_bits=8,
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
            significant_bits=8,
            transmitted_image=bf,
        )
        # Unknown channel ignored -- transmitted preserved
        np.testing.assert_array_equal(img[:, :, 0], 100)
        np.testing.assert_array_equal(img[:, :, 1], 100)
        np.testing.assert_array_equal(img[:, :, 2], 100)
