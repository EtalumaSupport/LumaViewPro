# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

from collections import deque

import numpy as np

# The most the stretch multiplies an image by. A near-flat frame (a field
# report's BF at 58-67) was stretched x28 and its sensor noise shown as
# flicker and moving bands; captures with content measure up to about x13.
MAX_GAIN = 10.0
_MIN_RANGE = 255.0 / MAX_GAIN


class ContrastStretcher:
    def __init__(
        self,
        window_len: int,
        bottom_pct: float,
        top_pct: float,
    ):
        self._window_len = window_len
        self._bottom_pct = bottom_pct
        self._top_pct = top_pct

        # deque(maxlen=window_len) is O(1) for both append and the
        # implicit drop of the oldest entry once the window is full.
        # Pre-deque code used a list with .pop(0), which is O(n) -- fine
        # at window_len=3 today, but this future-proofs against larger
        # window sizes without changing semantics.
        self._data = {
            'min': deque([0], maxlen=window_len),
            'range': deque([255], maxlen=window_len),
        }

        self._lut = np.arange(256, dtype=np.uint8)
        # Reused output scratch for the LUT apply in update(); sized lazily to
        # the frame on first use and whenever the frame shape changes.
        self._out_buf = None

    def update(self, image: np.ndarray) -> np.ndarray:

        # Subsample for faster percentile calculation (~16x fewer pixels)
        sampled = image[::4, ::4]

        min_val = np.percentile(sampled, self._bottom_pct)
        max_val = np.percentile(sampled, 100 - self._top_pct)
        range_diff = max_val - min_val

        self._data['min'].append(min_val)
        self._data['range'].append(range_diff)

        min_val_avg = np.average(self._data['min'])
        range_val_avg = np.average(self._data['range'])
        if range_val_avg < _MIN_RANGE:
            # Capped: the window sits on the frame's centre, held inside
            # 0-255, so a dark frame stays dark and a saturated one bright.
            centre = min_val_avg + range_val_avg / 2
            min_val_avg = min(max(centre - _MIN_RANGE / 2, 0.0), 255.0 - _MIN_RANGE)
            range_val_avg = _MIN_RANGE

        # Build a 256-entry uint8 LUT: avoids all per-pixel float math and
        # eliminates the float32 buffer + astype allocation entirely.
        scale = 255.0 / range_val_avg
        lut = self._lut
        vals = np.arange(256, dtype=np.float64)
        np.subtract(vals, min_val_avg, out=vals)
        np.multiply(vals, scale, out=vals)
        np.clip(vals, 0, 255, out=vals)
        lut[:] = vals.astype(np.uint8)

        # Apply the LUT into a reused scratch instead of allocating a fresh
        # full-frame array each display tick (~3.6 MB at 1900x1900, ~108 MB/s
        # at 30 fps when histogram-eq is on). Reuse is safe: the sole consumer
        # (scope_display) copies the result to bytes before the next update()
        # overwrites it, and update() runs on a single thread.
        if self._out_buf is None or self._out_buf.shape != image.shape:
            self._out_buf = np.empty(image.shape, dtype=np.uint8)
        return np.take(lut, image, out=self._out_buf)
