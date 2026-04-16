# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for ContrastStretcher.

Regression coverage for the deque-window invariant: rolling average
must use only the last `window_len` samples; old samples must be
dropped automatically when the window is full. Pre-deque code used a
list with .pop(0) — fine functionally but O(n) per drop, and it's
easy to break the rolling-window invariant when refactoring.
"""

import numpy as np

from modules.contrast_stretcher import ContrastStretcher


class TestRollingWindow:

    def test_window_drops_oldest_when_full(self):
        cs = ContrastStretcher(window_len=3, bottom_pct=0, top_pct=0)
        # Seed with 4 known images; window=3 means only the last 3
        # samples (10/20/30 ranges) survive in `_data['range']`.
        for level in (5, 10, 20, 30):
            img = np.full((40, 40), 0, dtype=np.uint8)
            img[0, 0] = level  # Make max==level so range==level
            cs.update(img)
        # `range` deque should hold exactly 3 entries
        assert len(cs._data['range']) == 3
        # First-seeded value (5) should have been dropped
        assert 5 not in list(cs._data['range'])

    def test_window_average_uses_only_recent_samples(self):
        """If the window correctly drops old values, the rolling average
        of `min` should reflect only recent samples — not history."""
        cs = ContrastStretcher(window_len=2, bottom_pct=0, top_pct=0)
        # Two updates with min=100 (force via uniform image)
        for _ in range(2):
            img = np.full((40, 40), 100, dtype=np.uint8)
            cs.update(img)
        assert list(cs._data['min']) == [100.0, 100.0]
        # One update with min=200 — window slides, oldest 100 drops
        img = np.full((40, 40), 200, dtype=np.uint8)
        cs.update(img)
        assert list(cs._data['min']) == [100.0, 200.0]
        # One more — both 100s are gone
        img = np.full((40, 40), 200, dtype=np.uint8)
        cs.update(img)
        assert list(cs._data['min']) == [200.0, 200.0]


class TestUpdateReturnsArray:
    def test_update_returns_uint8_array_same_shape(self):
        cs = ContrastStretcher(window_len=3, bottom_pct=2, top_pct=2)
        img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        out = cs.update(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_zero_range_returns_input(self):
        """When range_val_avg == 0 (all-uniform image), .update() returns
        the original array (not a transformed copy)."""
        cs = ContrastStretcher(window_len=3, bottom_pct=0, top_pct=0)
        img = np.full((50, 50), 128, dtype=np.uint8)
        out = cs.update(img)
        # First call seeds with min=128, range=0; rolling avg range = (255+0)/2 nonzero
        # so this WILL transform. Second call: avg drops the seed 255,
        # range -> 0/0 = 0, returns input.
        out = cs.update(img)
        out = cs.update(img)
        assert out is img
