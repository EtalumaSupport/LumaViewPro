# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for ContrastStretcher.

Regression coverage for the deque-window invariant: rolling average
must use only the last `window_len` samples; old samples must be
dropped automatically when the window is full. Pre-deque code used a
list with .pop(0) -- fine functionally but O(n) per drop, and it's
easy to break the rolling-window invariant when refactoring.

And the gain cap: the live view's stretch multiplies an image by at most
x10, so a near-flat frame's sensor noise is not blown up into flicker.
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
        of `min` should reflect only recent samples -- not history."""
        cs = ContrastStretcher(window_len=2, bottom_pct=0, top_pct=0)
        # Two updates with min=100 (force via uniform image)
        for _ in range(2):
            img = np.full((40, 40), 100, dtype=np.uint8)
            cs.update(img)
        assert list(cs._data['min']) == [100.0, 100.0]
        # One update with min=200 -- window slides, oldest 100 drops
        img = np.full((40, 40), 200, dtype=np.uint8)
        cs.update(img)
        assert list(cs._data['min']) == [100.0, 200.0]
        # One more -- both 100s are gone
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


def _display_stretcher() -> ContrastStretcher:
    """The live view's own stretcher (``ui/scope_display.py``)."""
    return ContrastStretcher(window_len=3, bottom_pct=0.3, top_pct=0.3)


def _shown(levels: tuple[int, int], frames: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """(the last frame, what the display shows of it) after ``frames`` frames
    spread evenly over ``levels``, enough to fill the 3-frame window."""
    lo, hi = levels
    rng = np.random.default_rng(0)
    cs = _display_stretcher()
    for _ in range(frames):
        frame = rng.integers(lo, hi + 1, (200, 200), dtype=np.uint8)
        shown = cs.update(frame).copy()
    return frame, shown


def _gain(frame: np.ndarray, shown: np.ndarray) -> float:
    """The display levels per input level across the frame's spread."""
    lo, hi = int(frame.min()), int(frame.max())
    return (float(shown[frame == hi].mean()) - float(shown[frame == lo].mean())) / (hi - lo)


class TestGainCap:
    """The stretch multiplies by at most x10; a capped window sits on the
    frame's centre, held inside 0-255."""

    def test_a_near_flat_frame_is_stretched_by_at_most_ten(self):
        # The field report's frame: BF 2 mA / 1 ms, pixels 58-67, stretched
        # x28 before the cap.
        frame, shown = _shown((58, 67))
        assert _gain(frame, shown) <= 10.2

    def test_a_capped_frame_keeps_its_brightness_at_the_ends(self):
        # Mirror images of each other about mid-grey, shown mirrored.
        _, dark = _shown((0, 6))
        _, bright = _shown((249, 255))
        assert dark.mean() < 64
        assert bright.mean() > 191

    def test_a_capped_mid_grey_frame_is_shown_mid_grey(self):
        _, shown = _shown((58, 67))
        assert abs(float(shown.mean()) - 127.5) < 8

    def test_a_flat_frame_is_shown_through_the_capped_window(self):
        cs = _display_stretcher()
        frame = np.full((50, 50), 128, dtype=np.uint8)
        for _ in range(3):
            shown = cs.update(frame)
        assert shown is not frame
        assert set(np.unique(shown)) <= {127, 128}

    def test_a_frame_with_content_fills_the_display(self):
        # 33-199: a 4x BF capture, x1.5, far under the cap.
        frame, shown = _shown((33, 199))
        assert shown[frame <= 33].max() <= 1
        assert shown[frame >= 199].min() >= 254
        assert _gain(frame, shown) > 1.5


class TestLutApplyBufferReuse:
    """The LUT is applied into a reused scratch (np.take out=) instead of a
    fresh per-frame array. The result must still equal fancy-indexing the
    built LUT, and the scratch must resize when the frame shape changes."""

    def test_output_equals_lut_applied(self):
        cs = ContrastStretcher(window_len=3, bottom_pct=2, top_pct=2)
        img = np.random.default_rng(0).integers(0, 256, (64, 48), dtype=np.uint8)
        out = cs.update(img)
        assert np.array_equal(out, cs._lut[img])

    def test_scratch_resizes_on_shape_change(self):
        cs = ContrastStretcher(window_len=3, bottom_pct=2, top_pct=2)
        a = np.random.default_rng(1).integers(0, 256, (64, 48), dtype=np.uint8)
        b = np.random.default_rng(2).integers(0, 256, (80, 100), dtype=np.uint8)
        assert cs.update(a).shape == a.shape
        out_b = cs.update(b)
        assert out_b.shape == b.shape
        assert np.array_equal(out_b, cs._lut[b])
