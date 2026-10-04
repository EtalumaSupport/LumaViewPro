# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for Protocol.validate_steps() -- field-level validation of protocol steps.

Uses the real ObjectiveLoader loading the real objectives.json, so objective
names in test steps must match real entries in data/objectives.json.
"""

import datetime

import pandas as pd
import pytest

from modules.labware_loader import WellPlateLoader
from modules.objectives_loader import ObjectiveLoader
from modules.protocol import Protocol, ProtocolFormatError
from tests.test_protocol_roundtrip import TILING_CONFIGS


# Real objective names from data/objectives.json -- must match for validation
_VALID_OBJECTIVE = '4x Oly'
_INVALID_OBJECTIVE = '100x Oil Imm Fake'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_protocol(
    steps_data: list[dict],
    labware_id: str = '96 well microplate',
) -> Protocol:
    """A protocol over ``steps_data``, built as a caller's ``config=`` is."""
    return Protocol(
        tiling_configs_file_loc=TILING_CONFIGS,
        config={
            'steps': pd.DataFrame(steps_data),
            'period': datetime.timedelta(minutes=1),
            'duration': datetime.timedelta(hours=1),
            'labware_id': labware_id,
        },
    )


def _valid_step(**overrides) -> dict:
    """Return a minimal valid step dict, with optional field overrides."""
    step = {
        'Name': 'A1_Blue_T1',
        'X': 0.0,
        'Y': 0.0,
        'Z': 0.0,
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
        'Auto_Named': True,
        'Label': '',
    }
    step.update(overrides)
    return step


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateStepsEmpty:
    def test_empty_protocol_returns_no_errors(self):
        p = _make_protocol([])
        assert p.validate_steps(ObjectiveLoader(), led_max_ma=1000) == []

    def test_valid_single_step_returns_no_errors(self):
        p = _make_protocol([_valid_step()])
        assert p.validate_steps(ObjectiveLoader(), led_max_ma=1000) == []


class TestValidateColor:
    def test_invalid_color(self):
        p = _make_protocol([_valid_step(Color='Purple')])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert len(errors) == 1
        assert "Color 'Purple'" in errors[0]

    def test_all_valid_colors(self):
        for color in ('Blue', 'Green', 'Red', 'BF', 'PC', 'DF', 'Lumi'):
            p = _make_protocol([_valid_step(Color=color)])
            assert p.validate_steps(ObjectiveLoader(), led_max_ma=1000) == [], (
                f'Color {color} should be valid'
            )


class TestValidateObjective:
    def test_invalid_objective(self):
        p = _make_protocol([_valid_step(Objective=_INVALID_OBJECTIVE)])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert len(errors) == 1
        assert f"Objective '{_INVALID_OBJECTIVE}'" in errors[0]

    def test_valid_objective(self):
        p = _make_protocol([_valid_step(Objective='10x Oly')])
        assert p.validate_steps(ObjectiveLoader(), led_max_ma=1000) == []

    def test_the_catalogue_consulted_is_the_protocols_own(self):
        """The validator used to build a second loader over the shipped file,
        so a protocol whose own catalogue lacked the objective still validated
        clean. The check reads the catalogue its caller hands it."""
        from types import SimpleNamespace

        p = _make_protocol([_valid_step(Objective='10x Oly')])
        errors = p.validate_steps(
            SimpleNamespace(get_objectives_list=lambda: ['4x Oly']), led_max_ma=1000
        )
        assert len(errors) == 1
        assert "Objective '10x Oly'" in errors[0]

    def test_an_empty_catalogue_refuses_every_step_objective(self):
        """An empty catalogue used to skip the objective check entirely, so
        the protocol that could not run anywhere was the one that validated.
        No catalogue entry means no valid objective, so every step is flagged."""
        from types import SimpleNamespace

        p = _make_protocol([_valid_step(Objective='10x Oly'), _valid_step(Objective='4x Oly')])
        errors = p.validate_steps(SimpleNamespace(get_objectives_list=lambda: []), led_max_ma=1000)
        assert len(errors) == 2
        assert all('not found in objectives.json' in e for e in errors)


class TestValidateExposure:
    def test_zero_exposure_is_refused(self):
        """No camera takes 0 ms, so a 0 ms step is refused here rather than
        called valid and refused later by the run."""
        p = _make_protocol([_valid_step(Exposure=0)])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert any('Exposure must be more than 0 ms' in e for e in errors)

    def test_negative_exposure(self):
        p = _make_protocol([_valid_step(Exposure=-10)])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert any('Exposure must be more than 0 ms' in e for e in errors)

    def test_valid_exposure(self):
        p = _make_protocol([_valid_step(Exposure=100.5)])
        assert p.validate_steps(ObjectiveLoader(), led_max_ma=1000) == []


class TestValidateIllumination:
    def test_negative_illumination(self):
        p = _make_protocol([_valid_step(Illumination=-1)])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert any('Illumination must be 0' in e for e in errors)

    def test_over_max_illumination(self):
        p = _make_protocol([_valid_step(Illumination=1001)])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert any('Illumination must be 0-1000' in e for e in errors)

    def test_zero_illumination_valid(self):
        p = _make_protocol([_valid_step(Illumination=0)])
        assert p.validate_steps(ObjectiveLoader(), led_max_ma=1000) == []

    def test_max_illumination_valid(self):
        p = _make_protocol([_valid_step(Illumination=1000)])
        assert p.validate_steps(ObjectiveLoader(), led_max_ma=1000) == []


class TestValidateGain:
    def test_negative_gain(self):
        p = _make_protocol([_valid_step(Gain=-1)])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert any('Gain must be >= 0' in e for e in errors)

    def test_zero_gain_valid(self):
        p = _make_protocol([_valid_step(Gain=0)])
        assert p.validate_steps(ObjectiveLoader(), led_max_ma=1000) == []


class TestValidateSum:
    def test_zero_sum(self):
        p = _make_protocol([_valid_step(Sum=0)])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert any('Sum must be >= 1' in e for e in errors)

    def test_negative_sum(self):
        p = _make_protocol([_valid_step(Sum=-1)])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert any('Sum must be >= 1' in e for e in errors)

    def test_valid_sum(self):
        p = _make_protocol([_valid_step(Sum=3)])
        assert p.validate_steps(ObjectiveLoader(), led_max_ma=1000) == []


class TestValidateAcquireMode:
    def test_an_acquire_mode_that_is_not_image_or_video_is_refused(self):
        with pytest.raises(ProtocolFormatError, match='Acquire'):
            _make_protocol([_valid_step(Acquire='timelapse')])

    def test_video_mode_valid(self):
        vc = {'fps': 30, 'duration': 10}
        p = _make_protocol([_valid_step(Acquire='video', **{'Video Config': vc})])
        assert p.validate_steps(ObjectiveLoader(), led_max_ma=1000) == []


class TestValidateVideoConfig:
    def test_video_mode_zero_fps(self):
        vc = {'fps': 0, 'duration': 10}
        p = _make_protocol([_valid_step(Acquire='video', **{'Video Config': vc})])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert any('fps must be > 0' in e for e in errors)

    def test_video_mode_zero_duration(self):
        vc = {'fps': 30, 'duration': 0}
        p = _make_protocol([_valid_step(Acquire='video', **{'Video Config': vc})])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert any('duration must be > 0' in e for e in errors)

    def test_a_video_config_that_is_not_a_config_is_refused_on_any_step(self):
        # A cell of the wrong type, whatever the step acquires.
        with pytest.raises(ProtocolFormatError, match='Video Config'):
            _make_protocol([_valid_step(Acquire='image', **{'Video Config': 'garbage'})])


class TestValidateNameLength:
    def test_name_too_long(self):
        p = _make_protocol([_valid_step(Name='x' * 201)])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert any('Name exceeds 200' in e for e in errors)

    def test_name_at_limit(self):
        p = _make_protocol([_valid_step(Name='x' * 200)])
        assert p.validate_steps(ObjectiveLoader(), led_max_ma=1000) == []


class TestMultipleErrors:
    def test_multiple_fields_invalid(self):
        p = _make_protocol([_valid_step(Color='Bad', Exposure=-1, Sum=0)])
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert len(errors) == 3

    def test_multiple_steps_with_errors(self):
        p = _make_protocol(
            [
                _valid_step(Color='Bad'),
                _valid_step(Exposure=-10),
            ]
        )
        errors = p.validate_steps(ObjectiveLoader(), led_max_ma=1000)
        assert len(errors) == 2
        assert 'Step 1' in errors[0]
        assert 'Step 2' in errors[1]


# ---------------------------------------------------------------------------
# validate_for_run() tests -- pre-execution runtime validation
# ---------------------------------------------------------------------------

_DEFAULT_AXIS_LIMITS = {
    'X': {'min': 0, 'max': 120000},
    'Y': {'min': 0, 'max': 80000},
    'Z': {'min': 0, 'max': 14000},
}

_STAGE_OFFSET = {'x': 0.0, 'y': 0.0}


def _outside(position, limits=_DEFAULT_AXIS_LIMITS, labware_id='96 well microplate'):
    """The axes the one travel judgment finds outside, on a real plate."""
    from modules import labware_loader
    from modules.protocol import axes_outside_travel

    lw = labware_loader.WellPlateLoader().get_plate(plate_key=labware_id)
    return axes_outside_travel(position, limits, labware=lw, stage_offset=_STAGE_OFFSET)


# Labware IDs must match real entries in data/labware.json. Step X/Y are
# PLATE millimetres (a real protocol stores plate coordinates); the judgment
# converts them to stage micrometres before comparing with the travel limits.


class TestAxesOutsideTravel:
    def test_valid_positions_no_errors(self):
        # Plate center converts to a stage position inside all limits.
        assert _outside({'X': 60.0, 'Y': 40.0, 'Z': 5000.0}) == []

    def test_x_exceeds_max(self):
        # Plate X near the plate origin converts to a stage X beyond travel,
        # while Y stays in range -- only X should fault.
        assert _outside({'X': 0.0, 'Y': 40.0, 'Z': 5000.0}) == ['X']

    def test_y_exceeds_max(self):
        assert _outside({'X': 60.0, 'Y': 0.0, 'Z': 5000.0}) == ['Y']

    def test_z_exceeds_max(self):
        # Z is stage-um already -- compared directly, no conversion.
        assert _outside({'X': 60.0, 'Y': 40.0, 'Z': 15000.0}) == ['Z']

    def test_plate_origin_converts_out_of_range(self):
        # Regression for the mm-vs-um mismatch: a step stored at plate (0,0)
        # reads as 0/0, which a direct compare accepts, but converts to the
        # far stage corner -- outside both X and Y travel.
        assert _outside({'X': 0.0, 'Y': 0.0, 'Z': 0.0}) == ['X', 'Y']

    def test_negative_plate_position(self):
        # A negative plate coordinate converts even further past the stage edge.
        assert _outside({'X': -1.0, 'Y': 40.0, 'Z': 5000.0}) == ['X']

    def test_position_at_boundary_valid(self):
        # Plate (7.76, 5.48) converts to the (120000, 80000) max corner and
        # plate (127.76, 85.48) to (0, 0) min -- both inclusive-valid.
        assert _outside({'X': 7.76, 'Y': 5.48, 'Z': 14000.0}) == []
        assert _outside({'X': 127.76, 'Y': 85.48, 'Z': 0.0}) == []

    def test_multiple_axes_out_of_range(self):
        assert _outside({'X': 0.0, 'Y': 0.0, 'Z': 200000.0}) == ['X', 'Y', 'Z']

    def test_only_the_axes_with_limits_are_judged(self):
        """Only Z limits provided -- X and Y are not judged."""
        limits = {'Z': {'min': 0, 'max': 14000}}
        assert _outside({'X': 999999, 'Y': 0.0, 'Z': 15000}, limits) == ['Z']

    def test_no_limits_judges_nothing(self):
        assert _outside({'X': 999999, 'Y': 0.0, 'Z': 999999}, {}) == []


class TestAPositionIsANumber:
    @pytest.mark.parametrize('axis', ['X', 'Y', 'Z'])
    @pytest.mark.parametrize('value', ['', 'abc', None])
    def test_a_position_that_is_not_a_number_is_refused(self, axis, value):
        # An empty position too: "unknown" arrives with the Z-only row, which
        # fixes every reader of one first (ruled F2).
        with pytest.raises(ProtocolFormatError, match=axis):
            _make_protocol([_valid_step(**{axis: value})])

    def test_numeric_positions_are_not_reported(self):
        p = _make_protocol([_valid_step(X=999999, Y=-5, Z=200000)])
        errors = p.validate_for_run(
            objective_helper=ObjectiveLoader(),
            wellplate_loader=WellPlateLoader(),
            led_max_ma=1000,
        )
        assert not any('position' in e for e in errors), errors


class TestValidateForRunLabware:
    def test_invalid_labware(self):
        p = _make_protocol([_valid_step()], labware_id='nonexistent plate')
        errors = p.validate_for_run(
            objective_helper=ObjectiveLoader(),
            wellplate_loader=WellPlateLoader(),
            led_max_ma=1000,
        )
        assert any("Labware 'nonexistent plate' not found" in e for e in errors)

    def test_valid_labware(self):
        p = _make_protocol([_valid_step(X=60.0, Y=40.0)], labware_id='96 well microplate')
        errors = p.validate_for_run(
            objective_helper=ObjectiveLoader(),
            wellplate_loader=WellPlateLoader(),
            led_max_ma=1000,
        )
        assert not any('Labware' in e for e in errors)


class TestValidateForRunIncludesFieldValidation:
    def test_field_errors_included(self):
        """validate_for_run should include validate_steps errors too."""
        p = _make_protocol([_valid_step(X=60.0, Y=40.0, Color='Bad')])
        errors = p.validate_for_run(
            objective_helper=ObjectiveLoader(),
            wellplate_loader=WellPlateLoader(),
            led_max_ma=1000,
        )
        assert any("Color 'Bad'" in e for e in errors)


class TestValidateForRunEmpty:
    def test_empty_protocol(self):
        p = _make_protocol([])
        errors = p.validate_for_run(
            objective_helper=ObjectiveLoader(),
            wellplate_loader=WellPlateLoader(),
            led_max_ma=1000,
        )
        assert errors == []
