# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

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

        self._data = {
            'min': [0],
            'range': [255],
        }

        # Pre-allocated float32 buffer, lazily sized to match first frame
        self._buffer = None


    def update(self, image: np.ndarray) -> np.ndarray:

        # Subsample for faster percentile calculation (~16x fewer pixels)
        sampled = image[::4, ::4]

        # Calculate bottom % and top % of pixel values to use for contrast stretching
        min_val = np.percentile(sampled, self._bottom_pct)
        max_val = np.percentile(sampled, 100-self._top_pct)
        range_diff = max_val - min_val

        # Save the values to a list for calculating rolling averages
        self._data['min'].append(min_val)
        self._data['range'].append(range_diff)

        # Clamp the rolling average window length to last N values
        if len(self._data['min']) > self._window_len:
            self._data['min'].pop(0)
            self._data['range'].pop(0)

        # Calculate the rolling average values
        min_val_avg = np.average(self._data['min'])
        range_val_avg = np.average(self._data['range'])
        if range_val_avg == 0:
            return image

        # Apply contrast stretching using pre-allocated float32 buffer
        # to avoid float64 promotion (~4x less memory, ~1 GB/s GC reduction)
        if self._buffer is None or self._buffer.shape != image.shape:
            self._buffer = np.empty(image.shape, dtype=np.float32)

        scale = np.float32(255.0 / range_val_avg)
        offset = np.float32(min_val_avg)

        np.subtract(image, offset, out=self._buffer)
        np.multiply(self._buffer, scale, out=self._buffer)
        np.clip(self._buffer, 0, 255, out=self._buffer)
        return self._buffer.astype(np.uint8)
