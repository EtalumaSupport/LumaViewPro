# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""set_autofocus_algorithm is an intentionally-kept selection seam.

It has no production caller yet -- focus_function always runs the module
default -- but it is retained as the dispatch point for future per-modality
focus selection (fluorescence vs brightfield, or a newer metric). This locks
it as present and functional so an orphan-code audit cannot delete it without
the test failing, and so the algorithm map does not silently rot.
"""

from __future__ import annotations

import pytest

from modules import autofocus_functions as af


@pytest.fixture(autouse=True)
def _restore_default_algorithm():
    saved = af._focus_function
    yield
    af._focus_function = saved


@pytest.mark.parametrize(
    'algorithm,expected_name',
    [
        ('vollath4', 'focus_vollath4_original'),
        ('vollath4_numba', 'focus_vollath4_original'),
        ('vollath4_original', 'focus_vollath4_original'),
        ('skew', 'focus_skew'),
        ('pixel_variation', 'focus_pixel_variation'),
    ],
)
def test_each_algorithm_selects_its_function(algorithm, expected_name):
    af.set_autofocus_algorithm(algorithm)
    assert af._focus_function.__name__ == expected_name


def test_unknown_algorithm_raises():
    with pytest.raises(NotImplementedError):
        af.set_autofocus_algorithm('does_not_exist')
