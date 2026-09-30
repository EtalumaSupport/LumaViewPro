# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated scene: one specimen field every simulated sensor renders.

Both simulated cameras -- the stand-in ``SimulatedCamera`` and the simulated
FX2's MT9P031 -- sample this one field, so a simulated LS620 and a simulated
LS850 look at the same sample and a change to the scene is made once.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def specimen_frames(
    h: int, w: int, count: int = 4, shift_px: int = 1, seed: int = 20260827
) -> list[np.ndarray]:
    """Build the specimen field as the frames a simulated sensor cycles through.

    ``SimulatedCamera`` falls back to these when no image dir exists; the
    simulated FX2 always renders them.

    These are ONE soft field sampled at ``count`` slightly different
    offsets, not independent patterns. The distinction is the whole point:
    the cycle advances once per generated frame, so anything that differs
    much between entries strobes at frame rate. Neighbouring crops of a
    single field read as gentle drift instead -- enough motion to tell a
    live stream from a frozen one, and to judge the rate by how smooth it
    looks, without a pattern change fighting for attention.

    The field is low-resolution noise upsampled and smoothed twice, which
    leaves rounded blobs with no hard edges. Grey values are held inside a
    mid band so no frame is ever fully black or blown out; the caller's
    brightness scaling then reads as exposure rather than as a switch.

    Args:
        h: Frame height in pixels.
        w: Frame width in pixels.
        count: How many frames the cycle should contain.
        shift_px: Sampling offset between consecutive frames. One pixel,
            set by watching the live display: the field is coherent, so a
            whole-frame shift is far more visible than its ~1.5 grey-level
            mean difference suggests, and larger values read as the sample
            being jostled rather than as the stream being alive.
        seed: Fixes the field so a run is reproducible.

    Returns:
        ``count`` grayscale uint8 frames of shape ``(h, w)``.
    """
    rng = np.random.RandomState(seed)
    margin = shift_px * count
    canvas_h = h + margin
    canvas_w = w + margin

    # Cell size sets the apparent feature scale -- roughly the blob width
    # in pixels before smoothing rounds them off.
    cell = 48
    low = rng.random((canvas_h // cell + 2, canvas_w // cell + 2))
    field = np.kron(low, np.ones((cell, cell)))[:canvas_h, :canvas_w]
    field = uniform_filter(field, size=cell // 2)
    field = uniform_filter(field, size=cell // 2)
    field -= field.min()
    field /= max(field.max(), 1e-9)
    canvas = 55.0 + field * 165.0

    frames = []
    # Sampling around a circle keeps the drift bounded and closes the loop,
    # so wrapping from the last frame back to the first is the same size
    # step as every other -- a linear walk would snap back on the wrap.
    for i in range(count):
        angle = 2.0 * np.pi * i / count
        y = margin // 2 + round(shift_px * np.sin(angle))
        x = margin // 2 + round(shift_px * np.cos(angle))
        frames.append(canvas[y : y + h, x : x + w].astype(np.uint8))
    return frames
