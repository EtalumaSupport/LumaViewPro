# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An image's statistics are exact, in its own counts.

``image_utils.image_statistics`` counts each value once and reads every
statistic from the counts; a still's record carries them. The percentiles
are the lowest value at or above the fraction of the samples, numpy's
``inverted_cdf``.
"""

import numpy as np

from modules.image_utils import ImageStatistics, image_statistics


def _expected(image: np.ndarray) -> ImageStatistics:
    flat = image.ravel()
    return ImageStatistics(
        minimum=int(flat.min()),
        maximum=int(flat.max()),
        mean=float(flat.mean()),
        standard_deviation=float(flat.std()),
        percentile_1=int(np.percentile(flat, 1, method='inverted_cdf')),
        median=int(np.percentile(flat, 50, method='inverted_cdf')),
        percentile_99=int(np.percentile(flat, 99, method='inverted_cdf')),
    )


def _close(got: ImageStatistics, want: ImageStatistics) -> bool:
    return (
        (got.minimum, got.maximum, got.percentile_1, got.median, got.percentile_99)
        == (want.minimum, want.maximum, want.percentile_1, want.median, want.percentile_99)
        and np.isclose(got.mean, want.mean)
        and np.isclose(got.standard_deviation, want.standard_deviation)
    )


def test_a_12_bit_frame_a_sum_and_a_colour_frame_are_counted_exactly():
    rng = np.random.default_rng(7)
    frames = {
        '12-bit': rng.integers(100, 4096, (300, 400), dtype=np.uint16),
        'sum of three': rng.integers(0, 3 * 4096, (300, 400), dtype=np.uint16),
        'colour 8-bit': rng.integers(0, 256, (300, 400, 3), dtype=np.uint8),
    }
    wrong = {
        name: (image_statistics(image), _expected(image))
        for name, image in frames.items()
        if not _close(image_statistics(image), _expected(image))
    }
    assert not wrong, wrong


def test_a_flat_frame_has_one_value_everywhere():
    stats = image_statistics(np.full((10, 10), 37, dtype=np.uint16))

    assert stats == ImageStatistics(37, 37, 37.0, 0.0, 37, 37, 37)
