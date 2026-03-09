# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for ProtocolTimeEstimator — protocol imaging time estimation.
"""
import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from modules.protocol_time_estimator import (
    ProtocolTimeEstimator,
    StepTimeEstimate,
    ScanTimeEstimate,
    ProtocolTimeEstimate,
    XY_SPEED_UM_PER_S,
    Z_SPEED_UM_PER_S,
    LED_SETTLE_S,
    LED_SERIAL_S,
    STEP_OVERHEAD_S,
    AUTOGAIN_MAX_DURATION_S,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_protocol(steps_data, period_min=20.0, duration_hr=1.0):
    """Create a mock Protocol with given steps."""
    protocol = MagicMock()
    if steps_data:
        df = pd.DataFrame(steps_data)
    else:
        df = pd.DataFrame()
    protocol.steps.return_value = df
    protocol.num_steps.return_value = len(df)
    protocol.period.return_value = datetime.timedelta(minutes=period_min)
    protocol.duration.return_value = datetime.timedelta(hours=duration_hr)
    return protocol


def _valid_step(**overrides) -> dict:
    """Return a minimal valid step dict."""
    step = {
        'Name': 'A1_Blue',
        'X': 0.0, 'Y': 0.0, 'Z': 5000.0,
        'Auto_Focus': False,
        'Color': 'Blue',
        'False_Color': False,
        'Illumination': 100.0,
        'Gain': 1.0,
        'Auto_Gain': False,
        'Exposure': 50.0,  # ms
        'Sum': 1,
        'Objective': '10x Oly',
        'Acquire': 'image',
        'Video Config': {},
    }
    step.update(overrides)
    return step


OBJ_10X = {
    'magnification': 10,
    'focal_length': 18.0,
    'AF_min': 8.0,
    'AF_max': 36.0,
    'AF_range': 75.0,
}


@pytest.fixture(autouse=True)
def _mock_objectives():
    mock_loader = MagicMock()
    mock_loader.get_objective_info.return_value = OBJ_10X
    with patch('modules.protocol_time_estimator.ObjectiveLoader',
               return_value=mock_loader):
        yield mock_loader


# ---------------------------------------------------------------------------
# StepTimeEstimate tests
# ---------------------------------------------------------------------------

class TestStepTimeEstimate:
    def test_total_is_sum_of_components(self):
        est = StepTimeEstimate(
            step_index=0, step_name='test',
            move_time_s=0.1, led_time_s=0.01, autofocus_time_s=2.0,
            autogain_time_s=1.0, capture_time_s=0.5, overhead_s=0.02,
        )
        assert abs(est.total_s - 3.63) < 0.001

    def test_default_total_is_zero(self):
        est = StepTimeEstimate(step_index=0, step_name='test')
        assert est.total_s == 0.0


# ---------------------------------------------------------------------------
# Movement estimation tests
# ---------------------------------------------------------------------------

class TestMovementEstimation:
    def test_no_movement_first_step(self):
        estimator = ProtocolTimeEstimator()
        t = estimator._estimate_movement(_valid_step(), None)
        assert t == 0.0

    def test_xy_movement(self):
        estimator = ProtocolTimeEstimator()
        prev = _valid_step(X=0, Y=0, Z=5000)
        curr = _valid_step(X=1000, Y=500, Z=5000)
        t = estimator._estimate_movement(curr, prev)
        expected = 1000.0 / XY_SPEED_UM_PER_S
        assert abs(t - expected) < 0.0001

    def test_z_movement(self):
        estimator = ProtocolTimeEstimator()
        prev = _valid_step(X=0, Y=0, Z=5000)
        curr = _valid_step(X=0, Y=0, Z=6000)
        t = estimator._estimate_movement(curr, prev)
        expected = 1000.0 / Z_SPEED_UM_PER_S
        assert abs(t - expected) < 0.0001

    def test_xy_and_z_parallel(self):
        """XY and Z move concurrently — total is max, not sum."""
        estimator = ProtocolTimeEstimator()
        prev = _valid_step(X=0, Y=0, Z=5000)
        curr = _valid_step(X=50000, Y=0, Z=6000)  # 50mm X, 1mm Z
        t = estimator._estimate_movement(curr, prev)
        xy_time = 50000.0 / XY_SPEED_UM_PER_S  # 1.0s
        z_time = 1000.0 / Z_SPEED_UM_PER_S      # 0.2s
        assert abs(t - max(xy_time, z_time)) < 0.0001

    def test_z_downward_overshoot(self):
        """Downward Z moves add backlash penalty."""
        estimator = ProtocolTimeEstimator()
        prev = _valid_step(X=0, Y=0, Z=5000)
        curr = _valid_step(X=0, Y=0, Z=4000)
        t = estimator._estimate_movement(curr, prev)
        z_time = 1000.0 / Z_SPEED_UM_PER_S
        overshoot = 50.0 / Z_SPEED_UM_PER_S  # 2 * backlash
        assert t > z_time  # should include overshoot
        assert abs(t - (z_time + overshoot)) < 0.0001


# ---------------------------------------------------------------------------
# Autofocus estimation tests
# ---------------------------------------------------------------------------

class TestAutofocusEstimation:
    def test_autofocus_positive_time(self):
        estimator = ProtocolTimeEstimator()
        t = estimator._estimate_autofocus(OBJ_10X, exposure_s=0.05)
        assert t > 0
        # 10x AF should be in the 1-10 second range
        assert 0.5 < t < 15.0

    def test_higher_exposure_increases_af_time(self):
        estimator = ProtocolTimeEstimator()
        t_fast = estimator._estimate_autofocus(OBJ_10X, exposure_s=0.01)
        t_slow = estimator._estimate_autofocus(OBJ_10X, exposure_s=0.5)
        assert t_slow > t_fast

    def test_af_zero_range_returns_zero(self):
        estimator = ProtocolTimeEstimator()
        obj = {'AF_range': 0, 'AF_max': 0, 'AF_min': 0}
        t = estimator._estimate_autofocus(obj, exposure_s=0.05)
        assert t == 0.0


# ---------------------------------------------------------------------------
# Single step estimation tests
# ---------------------------------------------------------------------------

class TestStepEstimation:
    def test_basic_image_step(self):
        estimator = ProtocolTimeEstimator()
        step = _valid_step(Exposure=50)
        est = estimator._estimate_step(0, pd.Series(step), None, OBJ_10X)
        assert est.move_time_s == 0.0  # first step
        assert est.led_time_s == LED_SETTLE_S + LED_SERIAL_S
        assert est.autofocus_time_s == 0.0
        assert est.autogain_time_s == 0.0
        assert est.capture_time_s > 0
        assert est.overhead_s == STEP_OVERHEAD_S

    def test_autofocus_step(self):
        estimator = ProtocolTimeEstimator()
        step = _valid_step(Auto_Focus=True)
        est = estimator._estimate_step(0, pd.Series(step), None, OBJ_10X)
        assert est.autofocus_time_s > 0

    def test_autogain_step(self):
        estimator = ProtocolTimeEstimator()
        step = _valid_step(Auto_Gain=True)
        est = estimator._estimate_step(0, pd.Series(step), None, OBJ_10X)
        assert est.autogain_time_s == AUTOGAIN_MAX_DURATION_S

    def test_video_step(self):
        estimator = ProtocolTimeEstimator()
        vc = {'fps': 30, 'duration': 10.0}
        step = _valid_step(Acquire='video', **{'Video Config': vc})
        est = estimator._estimate_step(0, pd.Series(step), None, OBJ_10X)
        assert est.capture_time_s >= 10.0

    def test_sum_increases_capture_time(self):
        estimator = ProtocolTimeEstimator()
        est1 = estimator._estimate_step(
            0, pd.Series(_valid_step(Sum=1)), None, OBJ_10X)
        est4 = estimator._estimate_step(
            0, pd.Series(_valid_step(Sum=4)), None, OBJ_10X)
        assert est4.capture_time_s > est1.capture_time_s


# ---------------------------------------------------------------------------
# Full protocol estimation tests
# ---------------------------------------------------------------------------

class TestProtocolEstimation:
    def test_empty_protocol(self):
        estimator = ProtocolTimeEstimator()
        protocol = _make_protocol([])
        result = estimator.estimate(protocol)
        assert result.scan_estimate.total_s == 0.0
        assert result.num_scans >= 1

    def test_single_step_protocol(self):
        estimator = ProtocolTimeEstimator()
        protocol = _make_protocol([_valid_step()])
        result = estimator.estimate(protocol)
        assert result.scan_estimate.total_s > 0
        assert result.scan_estimate.num_steps == 1

    def test_multi_step_protocol(self):
        estimator = ProtocolTimeEstimator()
        protocol = _make_protocol([
            _valid_step(X=0, Y=0),
            _valid_step(X=1000, Y=0),
            _valid_step(X=2000, Y=0),
        ])
        result = estimator.estimate(protocol)
        assert result.scan_estimate.num_steps == 3
        # Steps 2 and 3 should have movement time
        assert result.scan_estimate.step_estimates[1].move_time_s > 0
        assert result.scan_estimate.step_estimates[2].move_time_s > 0

    def test_scan_within_period(self):
        estimator = ProtocolTimeEstimator()
        protocol = _make_protocol(
            [_valid_step()], period_min=20.0, duration_hr=1.0)
        result = estimator.estimate(protocol)
        assert result.scan_fits_in_period
        assert result.scan_overrun_s == 0.0

    def test_scan_exceeds_period(self):
        estimator = ProtocolTimeEstimator()
        # 100 AF steps with long exposure should exceed a 1-minute period
        steps = [_valid_step(Auto_Focus=True, Exposure=500,
                             X=i * 1000, Y=0)
                 for i in range(20)]
        protocol = _make_protocol(steps, period_min=1.0, duration_hr=1.0)
        result = estimator.estimate(protocol)
        assert not result.scan_fits_in_period
        assert result.scan_overrun_s > 0

    def test_num_scans(self):
        estimator = ProtocolTimeEstimator()
        protocol = _make_protocol(
            [_valid_step()], period_min=20.0, duration_hr=2.0)
        result = estimator.estimate(protocol)
        assert result.num_scans == 6  # 120min / 20min

    def test_summary_string(self):
        estimator = ProtocolTimeEstimator()
        protocol = _make_protocol([_valid_step()])
        result = estimator.estimate(protocol)
        summary = result.summary()
        assert 'Steps per scan' in summary
        assert 'Scan time' in summary
        assert 'Estimated total' in summary

    def test_estimated_completion_is_timedelta(self):
        estimator = ProtocolTimeEstimator()
        protocol = _make_protocol([_valid_step()])
        result = estimator.estimate(protocol)
        assert isinstance(result.estimated_completion, datetime.timedelta)
