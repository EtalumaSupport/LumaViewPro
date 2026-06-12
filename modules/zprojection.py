# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Pure-numpy z-projection backend.

Reduces a stack of equal-shape single-plane arrays along the stack axis using
one of six standard projection methods. This is the canonical z-projection
implementation, replacing the former ImageJ/JVM round-trip.

The function operates on a list of 2-D arrays -- the exact contract the
ZProjector post-processor feeds it (per-color-plane slices for color images,
whole frames for mono). Output dtype always matches the input dtype.
"""

import enum

import numpy as np

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


def zproject(images_data: list[np.ndarray], method: ZProjectMethod) -> np.ndarray | None:
    """Project a stack of equal-shape arrays into a single array.

    Args:
        images_data: List of equal-shape, equal-dtype arrays (typically 2-D
            single-plane uint8 or uint16 frames). The stack is reduced along
            the list axis.
        method: Which reduction to apply.

    Returns:
        The projected array with the same dtype as the input frames, or None
        if images_data is empty.

    Notes:
        Average/Median/StdDev finish with round-half-to-even then cast back to
        the input dtype -- byte-identical to the finishing step the ImageJ
        backend used. Min/Max are exact integer reductions. Sum accumulates in
        a wide integer type and saturates at the input dtype max rather than
        wrapping, so a deep bright stack rails to white instead of overflowing.
    """
    if not images_data:
        logger.error('[ZProject] No images provided')
        return None

    orig_dtype = images_data[0].dtype
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
        # saturate to the output dtype's range. A uint16 stack would need
        # ~2.8e14 frames to overflow uint64.
        acc_dtype = np.uint64 if np.issubdtype(orig_dtype, np.unsignedinteger) else np.int64
        summed = stack.sum(axis=0, dtype=acc_dtype)
        result = np.clip(summed, 0, np.iinfo(orig_dtype).max)
    elif method == ZProjectMethod.StdDev:
        result = stack.std(axis=0, dtype=np.float64).round()
    else:
        logger.error(f'[ZProject] Unknown method: {method!r}')
        return None

    return result.astype(orig_dtype)
