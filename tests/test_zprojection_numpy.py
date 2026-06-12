# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Hand-computed parity tests for the pure-numpy z-projection backend.

ImageJ is being removed, so these tests cannot diff numpy output against a
live ImageJ run. Instead they assert against reference values computed by hand
on tiny fixed stacks where the correct answer is obvious by inspection. These
values are the new source of truth for the projection backend.
"""

import numpy as np
import pytest

from modules.zprojection import ZProjectMethod, zproject


def _stack(frames, dtype):
    return [np.array(f, dtype=dtype) for f in frames]


def test_min_exact():
    frames = _stack([[[10, 20], [30, 40]],
                     [[5, 25], [35, 15]],
                     [[50, 1], [2, 45]]], np.uint8)
    result = zproject(frames, ZProjectMethod.Min)
    np.testing.assert_array_equal(result, np.array([[5, 1], [2, 15]], dtype=np.uint8))
    assert result.dtype == np.uint8


def test_max_exact():
    frames = _stack([[[10, 20], [30, 40]],
                     [[5, 25], [35, 15]],
                     [[50, 1], [2, 45]]], np.uint8)
    result = zproject(frames, ZProjectMethod.Max)
    np.testing.assert_array_equal(result, np.array([[50, 25], [35, 45]], dtype=np.uint8))
    assert result.dtype == np.uint8


def test_average_banker_rounding():
    # Two-frame stack with half-way means to pin round-half-to-even behavior:
    # 10.5 -> 10 (even), 11.5 -> 12 (even), 13.5 -> 14 (even), 12.5 -> 12 (even).
    frames = _stack([[[10, 11], [13, 12]],
                     [[11, 12], [14, 13]]], np.uint8)
    result = zproject(frames, ZProjectMethod.Average)
    np.testing.assert_array_equal(result, np.array([[10, 12], [14, 12]], dtype=np.uint8))
    assert result.dtype == np.uint8


def test_average_integer_mean():
    # 10, 11, 12 -> 11.0 exactly.
    frames = _stack([[[10]], [[11]], [[12]]], np.uint16)
    result = zproject(frames, ZProjectMethod.Average)
    np.testing.assert_array_equal(result, np.array([[11]], dtype=np.uint16))
    assert result.dtype == np.uint16


def test_median_odd_n():
    frames = _stack([[[1, 9], [4, 4]],
                     [[5, 9], [6, 4]],
                     [[3, 9], [2, 4]]], np.uint8)
    result = zproject(frames, ZProjectMethod.Median)
    np.testing.assert_array_equal(result, np.array([[3, 9], [4, 4]], dtype=np.uint8))


def test_median_even_n_rounds():
    # Even N: median is the mean of the two middle values, then round-then-cast.
    # [2,4] -> 3.0; [10,11] -> 10.5 -> 10 (even).
    frames = _stack([[[2, 10]], [[4, 11]]], np.uint8)
    result = zproject(frames, ZProjectMethod.Median)
    np.testing.assert_array_equal(result, np.array([[3, 10]], dtype=np.uint8))


def test_sum_in_range():
    frames = _stack([[[10, 20], [30, 40]],
                     [[10, 20], [30, 40]]], np.uint8)
    result = zproject(frames, ZProjectMethod.Sum)
    np.testing.assert_array_equal(result, np.array([[20, 40], [60, 80]], dtype=np.uint8))
    assert result.dtype == np.uint8


def test_sum_saturates_on_overflow():
    # uint16 max is 65535; 40000 + 40000 = 80000 must saturate, not wrap.
    frames = _stack([[[40000, 30000]], [[40000, 30000]]], np.uint16)
    result = zproject(frames, ZProjectMethod.Sum)
    np.testing.assert_array_equal(result, np.array([[65535, 60000]], dtype=np.uint16))
    assert result.dtype == np.uint16


def test_stddev_known_value():
    # Population std (ddof=0) of {2, 4} is 1.0.
    frames = _stack([[[2]], [[4]]], np.uint8)
    result = zproject(frames, ZProjectMethod.StdDev)
    np.testing.assert_array_equal(result, np.array([[1]], dtype=np.uint8))
    assert result.dtype == np.uint8


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('method', list(ZProjectMethod))
def test_output_dtype_matches_input(dtype, method):
    frames = _stack([[[10, 20], [30, 40]],
                     [[15, 25], [35, 45]]], dtype)
    result = zproject(frames, method)
    assert result.dtype == dtype


def test_empty_input_returns_none():
    assert zproject([], ZProjectMethod.Max) is None
