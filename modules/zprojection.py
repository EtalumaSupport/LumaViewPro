# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Pure-numpy z-projection backend.

Reduces a stack of equal-shape single-plane arrays along the stack axis using
one of six standard projection methods. This is the canonical z-projection
implementation, replacing the former ImageJ/JVM round-trip.

The function operates on a list of 2-D arrays -- the exact contract the
ZProjector post-processor feeds it (per-color-plane slices for color images,
whole frames for mono). Output dtype matches the input dtype, except that a
Sum of uint8 frames is uint16: its counts pass 255.
"""

import enum

import numpy as np

import modules.image_utils as image_utils

from lvp_logger import logger


class ZProjectMethod(enum.Enum):
    Min = 'min'
    Max = 'max'
    Average = 'avg'
    Median = 'median'
    Sum = 'sum'
    StdDev = 'sd'

    @classmethod
    def list(cls):
        return [c.name for c in cls]


def projected_dtype(method: ZProjectMethod, input_dtype: np.typing.DTypeLike) -> np.dtype:
    """The dtype a projection of frames of ``input_dtype`` comes out in.

    The input's, except a Sum of uint8 frames: a sum's counts pass 255, so it
    is stored in a 16-bit container, as a summed capture is.
    """
    input_dtype = np.dtype(input_dtype)
    if method == ZProjectMethod.Sum and input_dtype == np.uint8:
        return np.dtype(np.uint16)
    return input_dtype


def zproject(images_data: list[np.ndarray], method: ZProjectMethod) -> np.ndarray | None:
    """Project a stack of equal-shape arrays into a single array.

    Args:
        images_data: List of equal-shape, equal-dtype arrays (typically 2-D
            single-plane uint8 or uint16 frames). The stack is reduced along
            the list axis.
        method: Which reduction to apply.

    Returns:
        The projected array, or None if images_data is empty. Its dtype is the
        input frames', except a Sum of uint8 frames, which is uint16.

    Notes:
        Average/Median/StdDev finish with round-half-to-even then cast back to
        the input dtype -- byte-identical to the finishing step the ImageJ
        backend used. Min/Max are exact integer reductions. Sum accumulates in
        a wide integer type and is stored in a 16-bit container, a uint8 stack
        included, so it keeps its counts the way a summed capture does; it
        saturates at the container's ceiling rather than wrapping. Its depth
        tag is ``image_utils.summed_significant_bits`` of the slice count and
        the input depth.
    """
    if not images_data:
        logger.error('[ZProject] No images provided')
        return None

    # Every slice must share one canvas before it can be stacked; a per-slice
    # stitch divergence otherwise surfaces as a cryptic np.stack shape error.
    image_utils.require_uniform_geometry(
        [(f'slice {i}', img) for i, img in enumerate(images_data)],
        operation='z-project this stack',
    )

    orig_dtype = images_data[0].dtype
    out_dtype = projected_dtype(method, orig_dtype)
    stack = np.stack(images_data, axis=0)

    if method == ZProjectMethod.Min:
        result = stack.min(axis=0)
    elif method == ZProjectMethod.Max:
        result = stack.max(axis=0)
    elif method == ZProjectMethod.Average:
        result = stack.mean(axis=0, dtype=np.float64).round()
    elif method == ZProjectMethod.Median:
        result = np.median(stack, axis=0).round()
    elif method == ZProjectMethod.Sum:
        # Accumulate in a wide integer so the sum itself never overflows, then
        # saturate at the 16-bit container's ceiling. A uint16 stack would need
        # ~2.8e14 frames to overflow uint64.
        acc_dtype = np.uint64 if np.issubdtype(orig_dtype, np.unsignedinteger) else np.int64
        summed = stack.sum(axis=0, dtype=acc_dtype)
        result = np.clip(summed, 0, np.iinfo(out_dtype).max)
    elif method == ZProjectMethod.StdDev:
        result = stack.std(axis=0, dtype=np.float64).round()
    else:
        logger.error(f'[ZProject] Unknown method: {method!r}')
        return None

    return result.astype(out_dtype)
