# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for AutofocusRunner._find_best(): the focus a curve gives, and whether it is trusted.

``_find_best`` returns a ``FocusPick``: the position, and ``untrusted``,
None when the curve has a peak (a fit it trusts, or a peak too sharp to
fit, where the raw maximum is the focus) and otherwise why the raw maximum
stands in. The final pass refuses an untrusted pick
(``tests/test_autofocus_no_peak_is_a_failure.py``).
"""

import numpy as np
import pandas as pd

from modules.autofocus_runner import AutofocusRunner


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
        assert AutofocusRunner._find_best(df) == (100.0, None)

    def test_all_nan(self):
        df = pd.DataFrame({'position': [10.0, 20.0, 30.0], 'score': [np.nan, np.nan, np.nan]})
        # returns first position
        assert AutofocusRunner._find_best(df) == (10.0, 'no finite score')

    def test_all_zero(self):
        df = pd.DataFrame({'position': [10.0, 20.0, 30.0], 'score': [0.0, 0.0, 0.0]})
        pick = AutofocusRunner._find_best(df)
        # With all zeros, peak_score=0 so fit is skipped, returns raw max (any)
        assert pick.position in [10.0, 20.0, 30.0]
        assert pick.untrusted == 'no positive peak'

    def test_two_points_no_fit(self):
        """With fewer than 5 points above threshold, fit is skipped."""
        df = pd.DataFrame({'position': [10.0, 20.0], 'score': [100.0, 200.0]})
        assert AutofocusRunner._find_best(df) == (20.0, None)

    def test_flat_curve_no_fit(self):
        """Flat curve: all scores equal. The fit has no peak to find; raw max, untrusted."""
        df = pd.DataFrame(
            {'position': np.arange(0, 100, 10, dtype=float), 'score': np.full(10, 500.0)}
        )
        pick = AutofocusRunner._find_best(df)
        assert 0 <= pick.position <= 90
        assert pick.untrusted is not None


class TestFindBestGaussianFit:
    """Tests for Gaussian peak interpolation."""

    def test_symmetric_gaussian_peak_on_grid(self):
        """Peak exactly on a grid point -- fit should match raw max closely."""
        df = _make_gaussian_curve(center=50.0, sigma=10.0, z_min=0, z_max=100, step=5.0)
        pick = AutofocusRunner._find_best(df)
        assert abs(pick.position - 50.0) < 1.0
        assert pick.untrusted is None

    def test_gaussian_peak_between_grid_points(self):
        """Peak at 53.0 with 10um steps -- raw max is 50 or 60, fit should find ~53."""
        df = _make_gaussian_curve(center=53.0, sigma=15.0, z_min=0, z_max=100, step=10.0)
        pick = AutofocusRunner._find_best(df)
        # Fit should be closer to 53 than the nearest grid point
        assert abs(pick.position - 53.0) < 5.0  # within half a step
        # And closer than raw max would be
        raw_max_pos = df.loc[df['score'].idxmax(), 'position']
        assert abs(pick.position - 53.0) <= abs(raw_max_pos - 53.0)
        assert pick.untrusted is None

    def test_gaussian_fit_with_noise(self):
        """Noisy Gaussian -- fit should still be reasonable."""
        df = _make_gaussian_curve(
            center=50.0, sigma=12.0, z_min=0, z_max=100, step=5.0, noise_std=20.0
        )
        pick = AutofocusRunner._find_best(df)
        # With noise, allow wider tolerance
        assert abs(pick.position - 50.0) < 10.0
        assert pick.untrusted is None

    def test_10x_objective_typical(self):
        """Simulate 10x: DOF=8.5um, AF_min=8um step, range=200um."""
        df = _make_gaussian_curve(center=104.0, sigma=8.5, z_min=0, z_max=200, step=8.0)
        pick = AutofocusRunner._find_best(df)
        assert abs(pick.position - 104.0) < 4.0  # sub-step accuracy
        # A peak too sharp for five points above half is still a peak.
        assert pick.untrusted is None

    def test_40x_objective_typical(self):
        """Simulate 40x: DOF=1.7um, AF_min=1um step, range=50um."""
        df = _make_gaussian_curve(center=26.3, sigma=1.7, z_min=0, z_max=50, step=1.0)
        pick = AutofocusRunner._find_best(df)
        assert abs(pick.position - 26.3) < 0.5
        assert pick.untrusted is None

    def test_fit_outside_range_uses_raw_max(self):
        """A curve still rising at the edge of the measured range has no peak in it."""
        z = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40], dtype=float)
        # Rising slowly to the edge: every point above half the maximum, and
        # the fitted peak beyond the last point.
        scores = np.array([600, 650, 700, 740, 780, 820, 850, 880, 900], dtype=float)
        df = pd.DataFrame({'position': z, 'score': scores})
        assert AutofocusRunner._find_best(df) == (40.0, 'fit peak outside the measured range')

    def test_concave_up_with_contrast_uses_raw_max(self):
        """A concave-up fit on a curve with contrast: its raw max, as a peak."""
        # U-shaped score curve (minimum, not maximum)
        z = np.arange(0, 100, 5, dtype=float)
        scores = (z - 50) ** 2 + 100  # parabola opening up
        df = pd.DataFrame({'position': z, 'score': scores})
        pick = AutofocusRunner._find_best(df)
        # Raw max is at the edges (0 or 95), fit would give minimum not maximum
        raw_max = df.loc[df['score'].idxmax(), 'position']
        assert pick == (raw_max, None)

    def test_a_flat_topped_peak_with_contrast_is_its_raw_max(self):
        """A sharp peak whose top is flat fits no Gaussian, and is still a peak."""
        z = np.array([2850, 2886, 2922, 2958, 2994, 2998, 3006, 3014, 3030, 3066], dtype=float)
        scores = np.array([1.0, 1.0, 1.0, 1.1, 22.2, 20.2, 22.2, 4.2, 1.4, 1.0]) * 1e6
        df = pd.DataFrame({'position': z, 'score': scores})
        assert AutofocusRunner._find_best(df) == (2994.0, None)

    def test_a_concave_up_fit_with_no_contrast_has_no_peak(self):
        """Every point above half the maximum and no Gaussian in it: no peak."""
        z = np.arange(0, 100, 10, dtype=float)
        scores = (z - 45) ** 2 + 5000
        df = pd.DataFrame({'position': z, 'score': scores})
        assert AutofocusRunner._find_best(df).untrusted == 'fit not concave'


class TestFindBestEdgeCases:
    """Edge cases and robustness."""

    def test_inf_scores_filtered(self):
        df = pd.DataFrame({'position': [10.0, 20.0, 30.0], 'score': [np.inf, 100.0, 50.0]})
        # inf filtered out, raw max of valid is 100 at 20
        assert AutofocusRunner._find_best(df) == (20.0, None)

    def test_mixed_nan_and_valid(self):
        df = pd.DataFrame(
            {
                'position': [10.0, 20.0, 30.0, 40.0, 50.0],
                'score': [np.nan, 100.0, 200.0, 150.0, np.nan],
            }
        )
        assert AutofocusRunner._find_best(df) == (30.0, None)

    def test_negative_scores(self):
        """Negative scores (shouldn't happen, but handle gracefully)."""
        df = pd.DataFrame({'position': [10.0, 20.0, 30.0], 'score': [-5.0, -1.0, -10.0]})
        # highest (least negative)
        assert AutofocusRunner._find_best(df) == (20.0, 'no positive peak')
