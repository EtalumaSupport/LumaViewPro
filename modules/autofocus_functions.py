# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import numpy as np

from lvp_logger import logger

_enable_af_score_logging = False


def enable_af_score_logging(enable: bool) -> None:
    global _enable_af_score_logging
    _enable_af_score_logging = enable


def set_autofocus_algorithm(algorithm: str) -> None:
    # Kept though nothing calls it yet: focus_function always runs the module
    # default. This is the selection seam for future per-modality focus -- a
    # different metric for fluorescence vs brightfield, or a newer algorithm --
    # wired to a modality/settings key when that lands. Deleting it now would
    # force re-deriving the dispatch then.
    global _focus_function

    if algorithm in ('vollath4', 'vollath4_numba', 'vollath4_original'):
        # All three legacy aliases collapse to the one numpy implementation.
        # 'vollath4_numba' is kept only so a persisted setting still resolves.
        _focus_function = focus_vollath4_original
    elif algorithm == 'skew':
        _focus_function = focus_skew
    elif algorithm == 'pixel_variation':
        _focus_function = focus_pixel_variation
    else:
        raise NotImplementedError(f'Focus algorithm {algorithm} not implemented.')


def _mask_saturated(image: np.ndarray, margin: int = 1) -> np.ndarray:
    """Zero out saturated pixels so they don't dominate focus scores.
    Pixels within `margin` of the dtype max are considered saturated."""
    max_val = np.iinfo(image.dtype).max
    threshold = max_val - margin
    mask = image >= threshold
    if np.any(mask):
        image = image.copy()
        image[mask] = 0
    return image


def focus_function(
    image: np.ndarray,
    skip_score_logging: bool = False,
) -> float:
    image = _mask_saturated(image)
    score = _focus_function(image=image)

    if _enable_af_score_logging and not skip_score_logging:
        logger.info(f'[SCOPE API ] Focus Score: {score}')

    return score


def focus_vollath4_original(image: np.ndarray) -> float:
    # Vollath F4 autocorrelation focus measure.
    # Journal of Microscopy, Vol. 188, Pt 3, December 1997, pp. 264-272.
    # einsum fuses the multiply-and-sum so neither shifted product is
    # materialized as a temp array -- on a 1000x1000 frame that is two fewer
    # whole-image allocations per focus score, which matters on the AF sweep
    # hot path. float64 accumulation is required: products of uint16 pixels
    # summed over ~1e6 elements overflow float32's mantissa.
    image = image.astype(np.float64, copy=False)
    sum_one = np.einsum('ij,ij->', image[:-1], image[1:])  # g(i, j).g(i+1, j)
    sum_two = np.einsum('ij,ij->', image[:-2], image[2:])  # g(i, j).g(i+2, j)
    return sum_one - sum_two


def focus_skew(image: np.ndarray) -> float:
    w, h = image.shape

    hist = np.histogram(image, bins=256, range=(0, 256))
    hist = np.asarray(hist[0], dtype='int')
    max_index = hist.argmax()

    edges = np.histogram_bin_edges(image, bins=1)
    white_edge = edges[1]

    skew = white_edge - max_index
    return skew


def focus_pixel_variation(image: np.ndarray) -> float:
    w, h = image.shape

    sum = np.sum(image)
    ssq = np.sum(np.square(image))
    var = ssq * w * h - sum**2
    return var


_focus_function = focus_vollath4_original
