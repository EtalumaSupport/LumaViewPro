# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for Protocol.validate_steps() — field-level validation of protocol steps.

Uses the real ObjectiveLoader loading the real objectives.json, so objective
names in test steps must match real entries in data/objectives.json.
"""
import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest

from modules.protocol import Protocol


# Real objective names from data/objectives.json — must match for validation
_VALID_OBJECTIVE = '4x Oly'
_INVALID_OBJECTIVE = '100x Oil Imm Fake'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_protocol(steps_data: list[dict], labware_id: str = '96 well microplate') -> Protocol:
    """Create a Protocol with given steps.

    Bypasses Protocol.__init__ because it requires a full valid config dict
    with a loaded DataFrame — validate_steps() only needs _config, so we
    set that directly. This is NOT a mock of Protocol's behavior; it just
    skips the constructor's file-loading step.
    """
    p = Protocol.__new__(Protocol)
    # Build the steps DataFrame
    dtypes = np.dtype(
        [
            ("Name", str), ("X", float), ("Y", float), ("Z", float),
            ("Auto_Focus", bool), ("Color", str), ("False_Color", bool),
            ("Illumination", float), ("Gain", float), ("Auto_Gain", bool),
            ("Exposure", float), ("Sum", int), ("Objective", str),
            ("Well", str), ("Tile", str), ("Z-Slice", int),
            ("Custom Step", bool), ("Tile Group ID", int),
            ("Z-Stack Group ID", int), ("Acquire", str),
            ("Video Config", object), ("Stim_Config", object),
        ]
    )
    if steps_data:
        df = pd.DataFrame(steps_data)
    else:
        df = pd.DataFrame(np.empty(0, dtype=dtypes))
    p._config = {
        'steps': df,
        'period': datetime.timedelta(minutes=1),
        'duration': datetime.timedelta(hours=1),
        'labware_id': labware_id,
    }
    # Real ObjectiveLoader — reads data/objectives.json
    from modules.objectives_loader import ObjectiveLoader
    p._objective_loader = ObjectiveLoader()
    return p


def _valid_step(**overrides) -> dict:
    """Return a minimal valid step dict, with optional field overrides."""
    step = {
        'Name': 'A1_Blue_T1',
        'X': 0.0, 'Y': 0.0, 'Z': 0.0,
        'Auto_Focus': False,
        'Color': 'Blue',
        'False_Color': False,
        'Illumination': 100.0,
        'Gain': 1.0,
        'Auto_Gain': False,
        'Exposure': 50.0,
        'Sum': 1,
        'Objective': _VALID_OBJECTIVE,
        'Well': 'A1',
        'Tile': '',
        'Z-Slice': 0,
        'Custom Step': False,
        'Tile Group ID': 0,
        'Z-Stack Group ID': 0,
        'Acquire': 'image',
        'Video Config': {},
        'Stim_Config': {},
    }
    step.update(overrides)
    return step


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidateStepsEmpty:
    def test_empty_protocol_returns_no_errors(self):
        p = _make_protocol([])
        assert p.validate_steps() == []

    def test_valid_single_step_returns_no_errors(self):
        p = _make_protocol([_valid_step()])
        assert p.validate_steps() == []


class TestValidateColor:
    def test_invalid_color(self):
        p = _make_protocol([_valid_step(Color='Purple')])
        errors = p.validate_steps()
        assert len(errors) == 1
        assert "Color 'Purple'" in errors[0]

    def test_all_valid_colors(self):
        for color in ('Blue', 'Green', 'Red', 'BF', 'PC', 'DF', 'Lumi'):
            p = _make_protocol([_valid_step(Color=color)])
            assert p.validate_steps() == [], f"Color {color} should be valid"


class TestValidateObjective:
    def test_invalid_objective(self):
        p = _make_protocol([_valid_step(Objective=_INVALID_OBJECTIVE)])
        errors = p.validate_steps()
        assert len(errors) == 1
        assert f"Objective '{_INVALID_OBJECTIVE}'" in errors[0]

    def test_valid_objective(self):
        p = _make_protocol([_valid_step(Objective='10x Oly')])
        assert p.validate_steps() == []


class TestValidateExposure:
    def test_zero_exposure_is_valid(self):
        """Exposure=0 is allowed for blank/placeholder steps."""
        p = _make_protocol([_valid_step(Exposure=0)])
        assert p.validate_steps() == []

    def test_negative_exposure(self):
        p = _make_protocol([_valid_step(Exposure=-10)])
        errors = p.validate_steps()
        assert any("Exposure must be >= 0" in e for e in errors)

    def test_valid_exposure(self):
        p = _make_protocol([_valid_step(Exposure=100.5)])
        assert p.validate_steps() == []


class TestValidateIllumination:
    def test_negative_illumination(self):
        p = _make_protocol([_valid_step(Illumination=-1)])
        errors = p.validate_steps()
        assert any("Illumination must be 0" in e for e in errors)

    def test_over_max_illumination(self):
        p = _make_protocol([_valid_step(Illumination=1001)])
        errors = p.validate_steps()
        assert any("Illumination must be 0" in e for e in errors)

    def test_zero_illumination_valid(self):
        p = _make_protocol([_valid_step(Illumination=0)])
        assert p.validate_steps() == []

    def test_max_illumination_valid(self):
        p = _make_protocol([_valid_step(Illumination=1000)])
        assert p.validate_steps() == []


class TestValidateGain:
    def test_negative_gain(self):
        p = _make_protocol([_valid_step(Gain=-1)])
        errors = p.validate_steps()
        assert any("Gain must be >= 0" in e for e in errors)

    def test_zero_gain_valid(self):
        p = _make_protocol([_valid_step(Gain=0)])
        assert p.validate_steps() == []


class TestValidateSum:
    def test_zero_sum(self):
        p = _make_protocol([_valid_step(Sum=0)])
        errors = p.validate_steps()
        assert any("Sum must be >= 1" in e for e in errors)

    def test_negative_sum(self):
        p = _make_protocol([_valid_step(Sum=-1)])
        errors = p.validate_steps()
        assert any("Sum must be >= 1" in e for e in errors)

    def test_valid_sum(self):
        p = _make_protocol([_valid_step(Sum=3)])
        assert p.validate_steps() == []


class TestValidateAcquireMode:
    def test_invalid_acquire_mode(self):
        p = _make_protocol([_valid_step(Acquire='timelapse')])
        errors = p.validate_steps()
        assert any("Acquire must be" in e for e in errors)

    def test_video_mode_valid(self):
        vc = {'fps': 30, 'duration': 10}
        p = _make_protocol([_valid_step(Acquire='video', **{'Video Config': vc})])
        assert p.validate_steps() == []


class TestValidateVideoConfig:
    def test_video_mode_zero_fps(self):
        vc = {'fps': 0, 'duration': 10}
        p = _make_protocol([_valid_step(Acquire='video', **{'Video Config': vc})])
        errors = p.validate_steps()
        assert any("fps must be > 0" in e for e in errors)

    def test_video_mode_zero_duration(self):
        vc = {'fps': 30, 'duration': 0}
        p = _make_protocol([_valid_step(Acquire='video', **{'Video Config': vc})])
        errors = p.validate_steps()
        assert any("duration must be > 0" in e for e in errors)

    def test_image_mode_ignores_video_config(self):
        p = _make_protocol([_valid_step(Acquire='image', **{'Video Config': 'garbage'})])
        assert p.validate_steps() == []


class TestValidateNameLength:
    def test_name_too_long(self):
        p = _make_protocol([_valid_step(Name='x' * 201)])
        errors = p.validate_steps()
        assert any("Name exceeds 200" in e for e in errors)

    def test_name_at_limit(self):
        p = _make_protocol([_valid_step(Name='x' * 200)])
        assert p.validate_steps() == []


class TestMultipleErrors:
    def test_multiple_fields_invalid(self):
        p = _make_protocol([_valid_step(Color='Bad', Exposure=-1, Sum=0)])
        errors = p.validate_steps()
        assert len(errors) == 3

    def test_multiple_steps_with_errors(self):
        p = _make_protocol([
            _valid_step(Color='Bad'),
            _valid_step(Exposure=-10),
        ])
        errors = p.validate_steps()
        assert len(errors) == 2
        assert "Step 1" in errors[0]
        assert "Step 2" in errors[1]


# ---------------------------------------------------------------------------
# validate_for_run() tests — pre-execution runtime validation
# ---------------------------------------------------------------------------

_DEFAULT_AXIS_LIMITS = {
    'X': {'min': 0, 'max': 120000},
    'Y': {'min': 0, 'max': 80000},
    'Z': {'min': 0, 'max': 14000},
}


# No WellPlateLoader fixture — tests use the real loader with real labware.json.
# Labware IDs in test steps must match real entries in data/labware.json.


class TestValidateForRunPositionBounds:
    def test_valid_positions_no_errors(self):
        p = _make_protocol([_valid_step(X=60000, Y=40000, Z=5000)])
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        assert errors == []

    def test_x_exceeds_max(self):
        p = _make_protocol([_valid_step(X=130000)])
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        assert any("X position 130000" in e and "outside travel limits" in e for e in errors)

    def test_y_exceeds_max(self):
        p = _make_protocol([_valid_step(Y=90000)])
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        assert any("Y position 90000" in e and "outside travel limits" in e for e in errors)

    def test_z_exceeds_max(self):
        p = _make_protocol([_valid_step(Z=15000)])
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        assert any("Z position 15000" in e and "outside travel limits" in e for e in errors)

    def test_negative_position(self):
        p = _make_protocol([_valid_step(X=-100)])
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        assert any("X position -100" in e and "outside travel limits" in e for e in errors)

    def test_position_at_max_boundary_valid(self):
        p = _make_protocol([_valid_step(X=120000, Y=80000, Z=14000)])
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        assert errors == []

    def test_position_at_min_boundary_valid(self):
        p = _make_protocol([_valid_step(X=0, Y=0, Z=0)])
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        assert errors == []

    def test_multiple_axes_out_of_range(self):
        p = _make_protocol([_valid_step(X=200000, Y=200000, Z=200000)])
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        position_errors = [e for e in errors if "outside travel limits" in e]
        assert len(position_errors) == 3

    def test_multiple_steps_one_out_of_range(self):
        p = _make_protocol([
            _valid_step(Name='A1_BF', X=60000, Y=40000, Z=5000),
            _valid_step(Name='B1_BF', X=130000, Y=40000, Z=5000),
        ])
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        position_errors = [e for e in errors if "outside travel limits" in e]
        assert len(position_errors) == 1
        assert "Step 2" in position_errors[0]


class TestValidateForRunNoLimits:
    def test_no_axis_limits_skips_position_check(self):
        p = _make_protocol([_valid_step(X=999999)])
        errors = p.validate_for_run(axis_limits=None)
        # Should only have validate_steps() errors, not position errors
        assert not any("outside travel limits" in e for e in errors)

    def test_empty_axis_limits_skips_position_check(self):
        p = _make_protocol([_valid_step(X=999999)])
        errors = p.validate_for_run(axis_limits={})
        assert not any("outside travel limits" in e for e in errors)

    def test_partial_axis_limits(self):
        """Only Z limits provided — X and Y should not be checked."""
        limits = {'Z': {'min': 0, 'max': 14000}}
        p = _make_protocol([_valid_step(X=999999, Z=15000)])
        errors = p.validate_for_run(axis_limits=limits)
        position_errors = [e for e in errors if "outside travel limits" in e]
        assert len(position_errors) == 1
        assert "Z position" in position_errors[0]


class TestValidateForRunLabware:
    def test_invalid_labware(self):
        p = _make_protocol([_valid_step()], labware_id='nonexistent plate')
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        assert any("Labware 'nonexistent plate' not found" in e for e in errors)

    def test_valid_labware(self):
        p = _make_protocol([_valid_step()], labware_id='96 well microplate')
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        assert not any("Labware" in e for e in errors)


class TestValidateForRunIncludesFieldValidation:
    def test_field_errors_included(self):
        """validate_for_run should include validate_steps errors too."""
        p = _make_protocol([_valid_step(Color='Bad', Z=15000)])
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        assert any("Color 'Bad'" in e for e in errors)
        assert any("Z position 15000" in e for e in errors)


class TestValidateForRunEmpty:
    def test_empty_protocol(self):
        p = _make_protocol([])
        errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS)
        assert errors == []
