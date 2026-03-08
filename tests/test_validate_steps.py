"""
Tests for Protocol.validate_steps() — field-level validation of protocol steps.
"""
import datetime
import pathlib
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from modules.protocol import Protocol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_protocol(steps_data: list[dict]) -> Protocol:
    """Create a Protocol with given steps, mocking file-dependent constructors."""
    with patch.object(Protocol, '__init__', lambda self, **kw: None):
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
        }
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
        'Objective': '4x Plan Apo',
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


# Patch ObjectiveLoader so validate_steps doesn't need objectives.json
@pytest.fixture(autouse=True)
def _mock_objective_loader():
    mock_loader = MagicMock()
    mock_loader.get_objectives_list.return_value = [
        '4x Plan Apo', '10x Plan Fluor', '20x Plan Apo', '40x Plan Apo',
    ]
    with patch('modules.protocol.ObjectiveLoader', return_value=mock_loader):
        yield


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
        p = _make_protocol([_valid_step(Objective='100x Oil')])
        errors = p.validate_steps()
        assert len(errors) == 1
        assert "Objective '100x Oil'" in errors[0]

    def test_valid_objective(self):
        p = _make_protocol([_valid_step(Objective='10x Plan Fluor')])
        assert p.validate_steps() == []


class TestValidateExposure:
    def test_zero_exposure(self):
        p = _make_protocol([_valid_step(Exposure=0)])
        errors = p.validate_steps()
        assert any("Exposure must be > 0" in e for e in errors)

    def test_negative_exposure(self):
        p = _make_protocol([_valid_step(Exposure=-10)])
        errors = p.validate_steps()
        assert any("Exposure must be > 0" in e for e in errors)

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
            _valid_step(Exposure=0),
        ])
        errors = p.validate_steps()
        assert len(errors) == 2
        assert "Step 1" in errors[0]
        assert "Step 2" in errors[1]
