
import numba as nb
import numpy as np

from lvp_logger import logger

_enable_af_score_logging = False


def enable_af_score_logging(enable: bool) -> None:
    global _enable_af_score_logging
    _enable_af_score_logging = enable


def set_autofocus_algorithm(algorithm: str) -> None:
    global _focus_function

    if algorithm in ('vollath4', 'vollath4_numba'):
        _focus_function = focus_vollath4_numba
    elif algorithm == 'vollath4_original':
        _focus_function = focus_vollath4_original
    elif algorithm == 'skew':
        _focus_function = focus_skew
    elif algorithm == 'pixel_variation':
        _focus_function = focus_pixel_variation
    else:
        raise NotImplementedError(f"Focus algorithm {algorithm} not implemented.")


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


# 5000 iterations on 1000x1000 uint16: 20s
# 96 well plate center focus takes 6m
def focus_vollath4_original(image: np.ndarray) -> float:
    # Journal of Microscopy, Vol. 188, Pt 3, December 1997, pp. 264–272
    # TODO the w/h seem swapped, but this is how the original code was written.
    # Needs further investigation to clarify.
    image = image.astype(np.float64, copy=False)
    w, h = image.shape

    sum_one = np.sum(np.multiply(image[:w-1,:h], image[1:w,:h])) # g(i, j).g(i+1, j)
    sum_two = np.sum(np.multiply(image[:w-2,:h], image[2:w,:h])) # g(i, j).g(i+2, j)
    return sum_one - sum_two


# 5000 iterations on 1000x1000 uint16: 0.375307 seconds
# 96 well plate center focus takes 5m
@nb.njit(fastmath=True)
def focus_vollath4_numba(image: np.ndarray) -> float:
    w, h = image.shape
    s1 = 0.0
    s2 = 0.0
    for i in range(w - 1):
        for j in range(h):
            s1 += image[i, j] * image[i+1, j]
    for i in range(w - 2):
        for j in range(h):
            s2 += image[i, j] * image[i+2, j]
    return s1 - s2


def focus_skew(image: np.ndarray) -> float:
    # TODO the w/h seem swapped, but this is how the original code was written.
    # Needs further investigation to clarify.
    w, h = image.shape

    hist = np.histogram(image, bins=256,range=(0,256))
    hist = np.asarray(hist[0], dtype='int')
    max_index = hist.argmax()

    edges = np.histogram_bin_edges(image, bins=1)
    white_edge = edges[1]

    skew = white_edge-max_index
    return skew


def focus_pixel_variation(image: np.ndarray) -> float:
    # TODO the w/h seem swapped, but this is how the original code was written.
    # Needs further investigation to clarify.
    w, h = image.shape

    sum = np.sum(image)
    ssq = np.sum(np.square(image))
    var = ssq*w*h-sum**2
    return var


_focus_function = focus_vollath4_numba


def warmup_jit():
    """Pre-compile numba JIT functions with a tiny array to avoid first-use lag."""
    try:
        dummy = np.zeros((4, 4), dtype=np.uint16)
        focus_vollath4_numba(dummy)
    except Exception:
        pass

warmup_jit()
