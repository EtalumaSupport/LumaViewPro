# Copyright Etaluma, Inc.
"""Tests for modules/composite_builder.py -- shared composite image builder."""

import numpy as np
import pytest

import modules.image_utils as image_utils
import modules.layer_record as layer_record
from modules.composite_builder import CompositeChannel, TransmittedBase, build_composite
from tests.stock_channels import LED0, LED1, LED2


def _in_release_colours(images, bits=8):
    """Each image as a composite channel in its layer's release colour, at a
    depth of ``bits`` and stating no white."""
    return {
        key: CompositeChannel(image, layer_record.release_color(key), bits, None)
        for key, image in images.items()
    }


def test_a_composite_of_pure_primaries_puts_each_channel_on_its_own_plane():
    """LEDs 2, 1 and 0 (red, green and blue), one each, build the composite they always have.

    Every pixel's planes come from the channels alone: with no transmitted
    base, each channel's 8-bit value on its own plane; with one, a pixel where
    any channel reaches its cutoff carries each channel at or above its cutoff
    on its plane and nothing else, and every other pixel keeps the base's grey.
    The 120 cases walk three input depths, a base or none, four cutoffs and
    five channel sets, each with a row of black and a row at full scale, and
    every case that differs is named.
    """
    plane_of = {LED2: 0, LED1: 1, LED0: 2}
    rng = np.random.default_rng(0)
    differing = []
    cases = 0
    for bits, dtype in ((8, np.uint8), (12, np.uint16), (16, np.uint16)):
        full_scale = (1 << bits) - 1
        for with_base in (False, True):
            for cutoff in (1, 50, 128, 255):
                for keys in (
                    (LED2, LED1, LED0),
                    (LED2, LED1),
                    (LED1, LED0),
                    (LED0,),
                    (LED2, LED0),
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
                        channels=_in_release_colours(channels, bits),
                        transmitted=None if base is None else TransmittedBase(base, bits, None),
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
        img = build_composite(channels=_in_release_colours({LED2: red}, 8))
        assert img.shape == (4, 4, 3)
        assert img.dtype == np.uint8
        np.testing.assert_array_equal(img[:, :, 0], 200)  # Red channel
        np.testing.assert_array_equal(img[:, :, 1], 0)  # Green empty
        np.testing.assert_array_equal(img[:, :, 2], 0)  # Blue empty

    def test_lumi_goes_to_blue_channel(self):
        lumi = np.full((4, 4), 80, dtype=np.uint8)
        img = build_composite(channels=_in_release_colours({'Lumi': lumi}, 8))
        np.testing.assert_array_equal(img[:, :, 2], 80)

    def test_all_three_channels(self):
        channels = {
            LED2: np.full((4, 4), 100, dtype=np.uint8),
            LED1: np.full((4, 4), 150, dtype=np.uint8),
            LED0: np.full((4, 4), 200, dtype=np.uint8),
        }
        img = build_composite(channels=_in_release_colours(channels, 8))
        np.testing.assert_array_equal(img[:, :, 0], 100)
        np.testing.assert_array_equal(img[:, :, 1], 150)
        np.testing.assert_array_equal(img[:, :, 2], 200)

    def test_12bit_input_downconverts_to_8bit(self):
        """A composite is always 8-bit: a 12-bit input channel is downconverted,
        not stored right-aligned in a 16-bit container (which renders near-black
        in any viewer that ignores the depth tag)."""
        red = np.full((4, 4), 3000, dtype=np.uint16)
        img = build_composite(channels=_in_release_colours({LED2: red}, 12))
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
        img = build_composite(channels=_in_release_colours({LED2: red}, 12))
        assert img.dtype == np.uint8
        np.testing.assert_array_equal(img[:, :, 0], 255)

    def test_empty_channels_returns_black(self):
        """Empty channel dict with no transmitted image should raise."""
        with pytest.raises((StopIteration, ValueError)):
            build_composite(channels=_in_release_colours({}, 8))


class TestBuildCompositeWithTransmitted:
    """Composite with transmitted (BF/PC/DF) base image."""

    def test_transmitted_only_replicates_to_rgb(self):
        bf = np.full((4, 4), 128, dtype=np.uint8)
        img = build_composite(
            channels=_in_release_colours({}), transmitted=TransmittedBase(bf, 8, None)
        )
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
            channels=_in_release_colours({LED1: green}),
            transmitted=TransmittedBase(bf, 8, None),
            brightness_thresholds={LED1: 50},
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
            channels=_in_release_colours({LED2: red, LED1: green}),
            transmitted=TransmittedBase(bf, 8, None),
            brightness_thresholds={LED2: 10, LED1: 10},
        )
        # Both above threshold: the pixel is their sum in place of the base
        np.testing.assert_array_equal(img[:, :, 0], 200)  # Red
        np.testing.assert_array_equal(img[:, :, 1], 150)  # Green
        np.testing.assert_array_equal(img[:, :, 2], 0)  # no channel, no base

    def test_default_threshold_is_one(self):
        """When no thresholds provided, every pixel above 0 is composited."""
        bf = np.full((4, 4), 100, dtype=np.uint8)
        red = np.full((4, 4), 1, dtype=np.uint8)  # Very dim but > 0
        img = build_composite(
            channels=_in_release_colours({LED2: red}),
            transmitted=TransmittedBase(bf, 8, None),
        )
        # Default cutoff 1: any pixel > 0 replaces transmitted
        np.testing.assert_array_equal(img[:, :, 0], 1)
        np.testing.assert_array_equal(img[:, :, 1], 0)
        np.testing.assert_array_equal(img[:, :, 2], 0)


class TestChannelsAdd:
    """Each channel contributes its intensity times its colour, summed and
    clipped, so channels that share a colour add instead of overwriting."""

    def _pixel(self, channels, base=None):
        return build_composite(
            channels=channels,
            transmitted=(
                None if base is None else TransmittedBase(np.full((1, 1), base, np.uint8), 8, None)
            ),
            brightness_thresholds=None if base is None else dict.fromkeys(channels, 1),
        )[0, 0].tolist()

    def test_blue_and_lumi_add_in_either_order_with_a_base_or_none(self):
        blue = np.full((1, 1), 100, np.uint8)
        lumi = np.full((1, 1), 50, np.uint8)
        blue_first = _in_release_colours({LED0: blue, 'Lumi': lumi})
        lumi_first = _in_release_colours({'Lumi': lumi, LED0: blue})
        for base in (None, 30):
            assert self._pixel(blue_first, base) == [0, 0, 150]
            assert self._pixel(lumi_first, base) == [0, 0, 150]

    def test_a_sum_past_full_scale_saturates_rather_than_wrapping(self):
        channels = _in_release_colours(
            {LED0: np.full((1, 1), 200, np.uint8), 'Lumi': np.full((1, 1), 100, np.uint8)}
        )
        assert self._pixel(channels) == [0, 0, 255]

    def test_equal_red_and_green_are_equal_in_the_composite(self):
        # Green is not weaker than Red at the same intensity (#521).
        pair = _in_release_colours(
            {LED2: np.full((1, 1), 120, np.uint8), LED1: np.full((1, 1), 120, np.uint8)}
        )
        assert self._pixel(pair) == [120, 120, 0]
        assert self._pixel(pair, base=40) == [120, 120, 0]

    def test_a_colour_that_is_not_a_primary_lands_on_each_plane_by_its_weight(self):
        magenta = layer_record.parse_display_color('#ff00ff')
        orange = layer_record.parse_display_color('#ff8000')
        channels = {
            'Magenta': CompositeChannel(np.full((1, 1), 50, np.uint8), magenta, 8, None),
            'Orange': CompositeChannel(np.full((1, 1), 100, np.uint8), orange, 8, None),
        }
        # Orange's #80 is 128 / 255 of full weight: 100 of it rounds to 50.
        assert self._pixel(channels) == [150, 50, 50]

    def test_five_channels_sum_on_every_plane(self):
        def one(value):
            return np.full((1, 1), value, np.uint8)

        channels = _in_release_colours(
            {LED2: one(40), LED1: one(50), LED0: one(60), 'Lumi': one(70)}
        )
        channels['Magenta'] = CompositeChannel(
            one(20), layer_record.parse_display_color('#ff00ff'), 8, None
        )
        assert self._pixel(channels) == [60, 50, 150]
        assert self._pixel(channels, base=90) == [60, 50, 150]


def test_the_builds_peak_memory_does_not_grow_with_the_channel_count():
    """A stitched composite can be hundreds of megapixels, so the build holds
    one 8-bit total, one coloured channel and one channel's 8-bit copy at a
    time: five 12-bit channels peak under 10 bytes a pixel, as three do.

    The 16-bit total it replaced, with every channel converted up front,
    peaked at 16 bytes a pixel for three channels and grew with each one.

    The peak is measured in a Python of its own: tracemalloc counts every
    thread's allocations, so in a test worker a simulated camera another
    test left streaming added one frame (3.61 bytes a pixel) to the build's
    peak whenever a frame landed during it.
    """
    import pathlib
    import subprocess
    import sys

    script = (
        'import tracemalloc, numpy as np\n'
        'import modules.layer_record as layer_record\n'
        'from modules.composite_builder import CompositeChannel, build_composite\n'
        'rng = np.random.default_rng(0)\n'
        'side = 1000\n'
        'def peak_per_pixel(keys):\n'
        '    channels = {\n'
        '        key: CompositeChannel(\n'
        '            rng.integers(0, 4096, (side, side), dtype=np.uint16),\n'
        '            layer_record.release_color(key),\n'
        '            12,\n'
        '            None,\n'
        '        )\n'
        '        for key in keys\n'
        '    }\n'
        '    tracemalloc.start()\n'
        '    try:\n'
        '        build_composite(channels=channels)\n'
        '        return tracemalloc.get_traced_memory()[1] / side / side\n'
        '    finally:\n'
        '        tracemalloc.stop()\n'
        f'print(peak_per_pixel(({LED2!r}, {LED1!r}, {LED0!r})),\n'
        f"      peak_per_pixel(({LED2!r}, {LED1!r}, {LED0!r}, 'Lumi', 'BF')))\n"
    )
    measured = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).resolve().parents[1],
    )
    assert measured.returncode == 0, measured.stderr
    three, five = (float(peak) for peak in measured.stdout.split())
    assert five < 10, five
    assert five - three < 1, (three, five)
