# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

from collections import deque

import numpy as np

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
        # Pre-deque code used a list with .pop(0), which is O(n) — fine
        # at window_len=3 today, but this future-proofs against larger
        # window sizes without changing semantics.
        self._data = {
            'min': deque([0], maxlen=window_len),
            'range': deque([255], maxlen=window_len),
        }

        self._lut = np.arange(256, dtype=np.uint8)


    def update(self, image: np.ndarray) -> np.ndarray:

        # Subsample for faster percentile calculation (~16x fewer pixels)
        sampled = image[::4, ::4]

        min_val = np.percentile(sampled, self._bottom_pct)
        max_val = np.percentile(sampled, 100-self._top_pct)
        range_diff = max_val - min_val

        self._data['min'].append(min_val)
        self._data['range'].append(range_diff)

        min_val_avg = np.average(self._data['min'])
        range_val_avg = np.average(self._data['range'])
        if range_val_avg == 0:
            return image

        # Build a 256-entry uint8 LUT: avoids all per-pixel float math and
        # eliminates the float32 buffer + astype allocation entirely.
        scale = 255.0 / range_val_avg
        lut = self._lut
        vals = np.arange(256, dtype=np.float64)
        np.subtract(vals, min_val_avg, out=vals)
        np.multiply(vals, scale, out=vals)
        np.clip(vals, 0, 255, out=vals)
        lut[:] = vals.astype(np.uint8)

        return lut[image]
