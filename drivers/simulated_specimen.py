# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated scene: one specimen field every simulated sensor renders.

Both simulated cameras -- the stand-in ``SimulatedCamera`` and the simulated
FX2's MT9P031 -- sample this one field, so a simulated LS620 and a simulated
LS850 look at the same sample and a change to the scene is made once.

The field is one sample, as a real one is, seen drifting one pixel around
a circle so a running stream looks alive. The four frames of the old cycle
were different crops with no focus, so what a focus metric scored at one Z
was whichever crop came up, and a sweep found its peak there.
"""

from __future__ import annotations

import cv2
import numpy as np


def specimen_field(h: int, w: int, seed: int = 20260827) -> np.ndarray:
    """The specimen as the full sensor sees it in focus: cells in a medium.

    Cells with a bright membrane, a fainter cytoplasm, a bright nucleus and
    small bright puncta, with edges at several scales, so a focus metric
    has a sharp peak at focus and falls away on either side. Grey values
    stay inside a mid band, so no frame is fully black or blown out and a
    caller's brightness scaling reads as exposure.

    Args:
        h: Field height in pixels.
        w: Field width in pixels.
        seed: Fixes the field, so a run is reproducible.

    Returns:
        A grayscale uint8 array of shape ``(h, w)``.
    """
    rng = np.random.default_rng(seed)
    # Gentle unevenness in the medium, as a real background has.
    low = rng.random((max(2, h // 160), max(2, w // 160))).astype(np.float32)
    field = (45.0 + cv2.resize(low, (w, h)) * 12.0).astype(np.uint8)

    n_cells = max(12, (h * w) // 4000)
    for _ in range(n_cells):
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        ry, rx = int(rng.integers(10, 26)), int(rng.integers(10, 26))
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(field, (cx, cy), (rx, ry), angle, 0, 360, 85, -1)
        cv2.ellipse(field, (cx, cy), (rx, ry), angle, 0, 360, 150, 2)
        nucleus = (max(3, rx // 3), max(3, ry // 3))
        cv2.ellipse(field, (cx, cy), nucleus, angle, 0, 360, 175, -1)
        for _ in range(int(rng.integers(2, 6))):
            py = cy + int(rng.integers(-(ry // 2), ry // 2 + 1))
            px = cx + int(rng.integers(-(rx // 2), rx // 2 + 1))
            cv2.circle(field, (px, py), 1, 215, -1)
    # The objective's own resolution: even in focus an edge is not a step.
    return cv2.GaussianBlur(field, (0, 0), 0.8)


# The drift: a step a frame around a circle one pixel across, which closes
# the loop so the wrap is the same size step as the others. One pixel was
# set by watching the live display: larger reads as the sample being
# jostled. It changes a focus score by about 0.03 %, against 4 % for a
# quarter of a depth of field off focus, so a sweep still finds the plane.
DRIFT_STEPS = tuple(
    (round(np.sin(2 * np.pi * i / 4)), round(np.cos(2 * np.pi * i / 4))) for i in range(4)
)


def drift_frames(sensor: np.ndarray, h: int, w: int) -> list[np.ndarray]:
    """The ``h`` x ``w`` frames, centred on ``sensor``, one per drift step.

    A frame the size of the sensor drifts into a one-pixel mirrored edge.
    The frames are views of one padded copy, not copies of their own.
    """
    padded = cv2.copyMakeBorder(sensor, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    y0 = (padded.shape[0] - h) // 2
    x0 = (padded.shape[1] - w) // 2
    return [padded[y0 + dy : y0 + dy + h, x0 + dx : x0 + dx + w] for dy, dx in DRIFT_STEPS]
