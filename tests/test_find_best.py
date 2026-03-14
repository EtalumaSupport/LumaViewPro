# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for AutofocusExecutor._find_best() Gaussian peak fitting."""

import numpy as np
import pandas as pd
import pytest

from modules.autofocus_executor import AutofocusExecutor


def _make_gaussian_curve(center, sigma, z_min, z_max, step, noise_std=0.0):
    """Generate a Gaussian focus curve as a DataFrame."""
    z = np.arange(z_min, z_max + step / 2, step)
    scores = np.exp(-0.5 * ((z - center) / sigma) ** 2) * 1000
    if noise_std > 0:
        rng = np.random.default_rng(42)
        scores += rng.normal(0, noise_std, len(scores))
        scores = np.clip(scores, 0, None)
    return pd.DataFrame({'position': z, 'score': scores})


class TestFindBestRawMax:
    """Tests for the raw-max fallback behavior."""

    def test_single_point(self):
        df = pd.DataFrame({'position': [100.0], 'score': [500.0]})
        assert AutofocusExecutor._find_best(df) == 100.0

    def test_all_nan(self):
        df = pd.DataFrame({'position': [10.0, 20.0, 30.0],
                           'score': [np.nan, np.nan, np.nan]})
        result = AutofocusExecutor._find_best(df)
        assert result == 10.0  # returns first position

    def test_all_zero(self):
        df = pd.DataFrame({'position': [10.0, 20.0, 30.0],
                           'score': [0.0, 0.0, 0.0]})
        result = AutofocusExecutor._find_best(df)
        # With all zeros, peak_score=0 so fit is skipped, returns raw max (any)
        assert result in [10.0, 20.0, 30.0]

    def test_two_points_no_fit(self):
        """With fewer than 5 points above threshold, fit is skipped."""
        df = pd.DataFrame({'position': [10.0, 20.0],
                           'score': [100.0, 200.0]})
        assert AutofocusExecutor._find_best(df) == 20.0

    def test_flat_curve_no_fit(self):
        """Flat curve: all scores equal. Fit should degenerate (a=0), use raw max."""
        df = pd.DataFrame({'position': np.arange(0, 100, 10, dtype=float),
                           'score': np.full(10, 500.0)})
        result = AutofocusExecutor._find_best(df)
        assert 0 <= result <= 90


class TestFindBestGaussianFit:
    """Tests for Gaussian peak interpolation."""

    def test_symmetric_gaussian_peak_on_grid(self):
        """Peak exactly on a grid point — fit should match raw max closely."""
        df = _make_gaussian_curve(center=50.0, sigma=10.0,
                                  z_min=0, z_max=100, step=5.0)
        result = AutofocusExecutor._find_best(df)
        assert abs(result - 50.0) < 1.0

    def test_gaussian_peak_between_grid_points(self):
        """Peak at 53.0 with 10um steps — raw max is 50 or 60, fit should find ~53."""
        df = _make_gaussian_curve(center=53.0, sigma=15.0,
                                  z_min=0, z_max=100, step=10.0)
        result = AutofocusExecutor._find_best(df)
        # Fit should be closer to 53 than the nearest grid point
        assert abs(result - 53.0) < 5.0  # within half a step
        # And closer than raw max would be
        raw_max_pos = df.loc[df['score'].idxmax(), 'position']
        assert abs(result - 53.0) <= abs(raw_max_pos - 53.0)

    def test_gaussian_fit_with_noise(self):
        """Noisy Gaussian — fit should still be reasonable."""
        df = _make_gaussian_curve(center=50.0, sigma=12.0,
                                  z_min=0, z_max=100, step=5.0,
                                  noise_std=20.0)
        result = AutofocusExecutor._find_best(df)
        # With noise, allow wider tolerance
        assert abs(result - 50.0) < 10.0

    def test_10x_objective_typical(self):
        """Simulate 10x: DOF=8.5um, AF_min=8um step, range=200um."""
        df = _make_gaussian_curve(center=104.0, sigma=8.5,
                                  z_min=0, z_max=200, step=8.0)
        result = AutofocusExecutor._find_best(df)
        assert abs(result - 104.0) < 4.0  # sub-step accuracy

    def test_40x_objective_typical(self):
        """Simulate 40x: DOF=1.7um, AF_min=1um step, range=50um."""
        df = _make_gaussian_curve(center=26.3, sigma=1.7,
                                  z_min=0, z_max=50, step=1.0)
        result = AutofocusExecutor._find_best(df)
        assert abs(result - 26.3) < 0.5

    def test_fit_outside_range_uses_raw_max(self):
        """If fit peak is outside measured range, fall back to raw max."""
        # Create a curve that's mostly on one edge (peak off the measured range)
        z = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)
        # Monotonically increasing — peak is above 30
        scores = np.array([10, 20, 40, 80, 160, 320, 640], dtype=float)
        df = pd.DataFrame({'position': z, 'score': scores})
        result = AutofocusExecutor._find_best(df)
        # Should return raw max (30) since fit would be > 30
        assert result == 30.0

    def test_concave_up_uses_raw_max(self):
        """If fit gives concave-up parabola (a > 0), use raw max."""
        # U-shaped score curve (minimum, not maximum)
        z = np.arange(0, 100, 5, dtype=float)
        scores = (z - 50) ** 2 + 100  # parabola opening up
        df = pd.DataFrame({'position': z, 'score': scores})
        result = AutofocusExecutor._find_best(df)
        # Raw max is at the edges (0 or 95), fit would give minimum not maximum
        raw_max = df.loc[df['score'].idxmax(), 'position']
        assert result == raw_max


class TestFindBestEdgeCases:
    """Edge cases and robustness."""

    def test_inf_scores_filtered(self):
        df = pd.DataFrame({'position': [10.0, 20.0, 30.0],
                           'score': [np.inf, 100.0, 50.0]})
        result = AutofocusExecutor._find_best(df)
        assert result == 20.0  # inf filtered out, raw max of valid is 100 at 20

    def test_mixed_nan_and_valid(self):
        df = pd.DataFrame({'position': [10.0, 20.0, 30.0, 40.0, 50.0],
                           'score': [np.nan, 100.0, 200.0, 150.0, np.nan]})
        result = AutofocusExecutor._find_best(df)
        assert result == 30.0

    def test_negative_scores(self):
        """Negative scores (shouldn't happen, but handle gracefully)."""
        df = pd.DataFrame({'position': [10.0, 20.0, 30.0],
                           'score': [-5.0, -1.0, -10.0]})
        result = AutofocusExecutor._find_best(df)
        assert result == 20.0  # highest (least negative)
