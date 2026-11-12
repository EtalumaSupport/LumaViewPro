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

from _lumascope.modules.autofocus_runner import AutofocusRunner


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
        """One point is no curve: its spread is nothing, so it is no focus."""
        df = pd.DataFrame({'position': [100.0], 'score': [500.0]})
        assert AutofocusRunner._find_best(df) == (100.0, 'every score within 5% of the peak')

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


# SN12064, 60x Meiji, 2026-08-27 13:09:22: every point its four passes
# measured, as autofocus.log has them. Every score within 1.2% of the peak,
# and the fit's checks passed on it: the curve's shape is the noise's.
_A_FLAT_FIELD_SWEEP = [
    (5035.33, 2187569.0), (5039.13, 2190149.0), (5044.95, 2185717.0),
    (5050.88, 2195234.0), (5056.93, 2192667.0), (5062.79, 2180357.0),
    (5041.67, 2199455.0), (5043.65, 2190029.0), (5045.64, 2173232.0),
    (5047.64, 2192942.0), (5049.65, 2182662.0), (5051.64, 2196101.0),
    (5053.65, 2190076.0), (5039.67, 2193462.0), (5040.33, 2189893.0),
    (5040.99, 2187412.0), (5041.65, 2194907.0), (5042.32, 2189076.0),
    (5042.98, 2189702.0), (5043.64, 2197634.0), (5044.30, 2184215.0),
    (5044.97, 2199466.0), (5045.63, 2185819.0), (5046.29, 2189016.0),
    (5043.79, 2194466.0), (5044.28, 2199679.0), (5044.78, 2179614.0),
    (5045.28, 2188112.0),
]  # fmt: skip

# A faint real sample on a bright floor, 2026-10-22 14:02:24 (bench): every
# point above half the peak, its scores spread 36%, focused at 7109.
_A_FAINT_SAMPLE_SWEEP = [
    (7085.01, 7006595.0), (7094.01, 7001546.0), (7103.01, 7300816.0),
    (7112.02, 8450388.0), (7121.02, 7007971.0), (7130.03, 7028902.0),
    (7099.84, 7014168.0), (7102.84, 7290012.0), (7105.84, 8684397.0),
    (7108.84, 10845872.0), (7111.84, 8695513.0), (7114.85, 7046849.0),
    (7106.15, 8935812.0), (7107.16, 9945062.0), (7108.16, 10994517.0),
    (7109.17, 10664375.0), (7110.17, 9909426.0), (7111.17, 9232955.0),
]  # fmt: skip


def _curve(points):
    return pd.DataFrame(points, columns=['position', 'score'])


class TestAFlatCurveIsNoFocus:
    """A curve whose every score is within 5% of its peak has no focus,
    whatever a fit on it says; a curve with more spread is judged by the fit."""

    def test_a_flat_field_whose_fit_passes_is_no_focus(self):
        pick = AutofocusRunner._find_best(_curve(_A_FLAT_FIELD_SWEEP))

        assert pick.untrusted == 'every score within 5% of the peak'

    def test_a_faint_sample_with_more_spread_keeps_its_focus(self):
        pick = AutofocusRunner._find_best(_curve(_A_FAINT_SAMPLE_SWEEP))

        assert pick.untrusted is None
        assert abs(pick.position - 7109.0) < 0.5

    def test_a_few_flat_scores_among_dark_frames_are_no_focus(self):
        """A dark frame scores NaN and is dropped, so a sweep can keep only a
        few finite scores: too few to fit, and flat, they are no peak."""
        z = [6454.0, 6460.0, 6466.0, 6472.0, 6478.0, 6484.0, 6490.0]
        scores = [1.0e6, 1.001e6, np.nan, 1.002e6, np.nan, np.nan, 1.0005e6]

        pick = AutofocusRunner._find_best(pd.DataFrame({'position': z, 'score': scores}))

        assert pick.untrusted == 'every score within 5% of the peak'
