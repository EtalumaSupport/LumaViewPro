"""Correctness tests for the Vollath F4 autofocus metric.

focus_vollath4_original is the single numpy implementation after the numba
JIT variant was removed. These tests pin its value against an independent
ground-truth reference so the metric cannot silently drift -- the einsum
fused multiply-sum must equal the explicit definition exactly, and float64
accumulation must stay exact for integer pixels.
"""

import numpy as np
import pytest

import modules.autofocus_functions as af
from modules.autofocus_functions import focus_vollath4_original


def _reference_vollath4(image: np.ndarray) -> float:
    """Vollath F4 straight from the definition (explicit sums), float64."""
    img = image.astype(np.float64)
    w, h = img.shape
    s1 = sum(img[i, j] * img[i + 1, j] for i in range(w - 1) for j in range(h))
    s2 = sum(img[i, j] * img[i + 2, j] for i in range(w - 2) for j in range(h))
    return s1 - s2


def test_hand_computed_value():
    # image = [[1,2],[3,4],[5,6]]:
    #   s1 = 1*3 + 2*4 + 3*5 + 4*6 = 50
    #   s2 = 1*5 + 2*6           = 17
    #   F4 = 33
    image = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint16)
    assert focus_vollath4_original(image) == 33.0


@pytest.mark.parametrize('dtype,high', [(np.uint8, 256), (np.uint16, 65535)])
def test_matches_reference(dtype, high):
    rng = np.random.default_rng(1234)
    image = rng.integers(0, high, size=(40, 37), dtype=dtype)
    # Integer products summed in float64 are exact -> require an exact match,
    # not just np.isclose. A regression that changed accumulation dtype or the
    # index window would break this.
    assert focus_vollath4_original(image) == _reference_vollath4(image)


def test_einsum_equals_explicit_multiply():
    # The einsum form replaced `np.sum(np.multiply(...))`; document that the
    # swap is value-preserving so a future reader does not "simplify" it back
    # and reintroduce the per-call temp-array allocations.
    rng = np.random.default_rng(7)
    image = rng.integers(0, 65535, size=(128, 96), dtype=np.uint16).astype(np.float64)
    s1 = np.sum(np.multiply(image[:-1], image[1:]))
    s2 = np.sum(np.multiply(image[:-2], image[2:]))
    assert focus_vollath4_original(image) == s1 - s2


def test_a_frame_that_is_not_8_bit_is_refused_not_scored():
    # A 12-bit frame in its 16-bit container: its clipped pixels read 4095,
    # which the saturation mask (255 is full scale) would score as signal.
    rng = np.random.default_rng(3)
    native = rng.integers(0, 4096, size=(64, 64), dtype=np.uint16)
    native[:8] = 4095
    with pytest.raises(ValueError, match='8-bit frames, got uint16'):
        af.focus_function(native)


def test_autofocus_judges_a_frame_by_its_centre_quarter():
    # Detail outside the centre quarter changes nothing autofocus scores.
    rng = np.random.default_rng(5)
    textured = rng.integers(0, 200, size=(80, 120), dtype=np.uint8)
    framed = np.zeros_like(textured)
    framed[20:60, 30:90] = textured[20:60, 30:90]
    assert af.autofocus_score(framed) == af.autofocus_score(textured)
    assert af.autofocus_score(textured) == af.focus_function(textured[20:60, 30:90])


def test_default_focus_function_is_numpy():
    # The module default must be the numpy implementation (numba is gone).
    assert af._focus_function is focus_vollath4_original
