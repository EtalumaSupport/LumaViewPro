# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The LS720 stage's constants: the shipped section reads, converts, and refuses.

The TMCM-6110 section of the shipped motor defaults carries LumaView
Classic's production LS720 values. The config reads it once and refuses a
section with a missing or out-of-range value, naming the key, rather than
driving a stage with a value nobody chose.
"""

import copy

import pytest

from drivers.tmcm6110_config import SECTION, Tmcm6110Config, usteps_per_s, usteps_per_s2
from modules.scope_capabilities import _resolve_lens_focal_length_mm, _resolve_pixel_size_um
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS


def _defaults_with(edit):
    defaults = copy.deepcopy(SHIPPED_MOTOR_DEFAULTS)
    edit(defaults[SECTION])
    return defaults


def test_the_shipped_section_reads():
    Tmcm6110Config(SHIPPED_MOTOR_DEFAULTS)


def test_microsteps_per_mm_follow_classic():
    """StageController.cs: X/Y 6400 (200 steps x 32 on a 1 mm lead);
    Z 32 * 400 * 74 / 20 / 2.6 (the spiral cam through a 20:74 pulley)."""
    config = Tmcm6110Config(SHIPPED_MOTOR_DEFAULTS)
    assert config.usteps_per_mm('X') == 6400
    assert config.usteps_per_mm('Y') == 6400
    assert config.usteps_per_mm('Z') == pytest.approx(32 * 400 * 74 / 20 / 2.6)


def test_microsteps_per_mm_follow_the_microstep_resolution():
    """The resolution the board is told to use is the one the conversion uses."""
    config = Tmcm6110Config(
        _defaults_with(lambda s: s['Axis Parameters']['X'].update({'Microstep Resolution': 4}))
    )
    assert config.usteps_per_mm('X') == 3200


def test_the_axis_parameters_are_classics_production_values():
    """StageController.cs under PRODUCTION_720."""
    config = Tmcm6110Config(SHIPPED_MOTOR_DEFAULTS)
    shared = {
        'Microstep Resolution': 5,
        'Ramp Divisor': 9,
        'Pulse Divisor': 3,
        'Soft Stop Flag': 0,
        'Right Limit Switch Disable': 0,
        'Left Limit Switch Disable': 0,
    }
    assert dict(config.axis_parameters('X')) == {
        **shared,
        'Max Current': 16,
        'Standby Current': 2,
        'Max Positioning Speed': 1000,
        'Max Acceleration': 500,
    }
    assert dict(config.axis_parameters('Y')) == {
        **shared,
        'Max Current': 16,
        'Standby Current': 4,
        'Max Positioning Speed': 1000,
        'Max Acceleration': 500,
    }
    assert dict(config.axis_parameters('Z')) == {
        **shared,
        'Max Current': 48,
        'Standby Current': 8,
        'Max Positioning Speed': 250,
        'Max Acceleration': 2000,
    }


def test_the_speed_units_reproduce_the_manuals_worked_example():
    """TMCM-6110 TMCL firmware manual, 6.4: speed 1000 at pulse_div 1 is
    122070.31 microsteps/s; acceleration 1000 at pulse_div 1, ramp_div 1 is
    119.21 MHz/s, microsteps/s per second. The simulated board times its
    moves with these."""
    assert usteps_per_s(1000, 1) == pytest.approx(122070.31, abs=0.01)
    assert usteps_per_s2(1000, 1, 1) == pytest.approx(119.21e6, abs=0.01e6)


def test_the_travel_is_the_measured_far_switch_and_the_margin_sits_inside_it():
    """The bench LS720, 2026-10-05, row 6: each far switch's distance from
    the index, driven onto at a quarter of the axis's speed. The margin is
    for the units not measured."""
    config = Tmcm6110Config(SHIPPED_MOTOR_DEFAULTS)
    assert config.travel_limit_um('X') == pytest.approx(123_710)
    assert config.travel_limit_um('Y') == pytest.approx(79_790)
    assert config.travel_limit_um('Z') == pytest.approx(12_030)
    assert config.travel_margin_um() == pytest.approx(1_000)


def test_the_index_positions_are_the_bench_ls720s_and_place_each_axis_travel():
    """The bench LS720, 2026-10-06, row 4a: X 117.69 and Y 0.95 mm, from A1
    and H12 centred by eye; Z's is its switch. The limits are computed from
    the index position, the far switch and the margin, in one place: X,
    whose sign is +1, runs down from its index, Y and Z up from theirs."""
    config = Tmcm6110Config(SHIPPED_MOTOR_DEFAULTS)
    assert {axis: config.index_position_um(axis) for axis in 'XYZ'} == {
        'X': pytest.approx(117_690),
        'Y': pytest.approx(950),
        'Z': 0.0,
    }
    assert {axis: config.direction(axis) for axis in 'XYZ'} == {'X': 1, 'Y': -1, 'Z': -1}
    assert config.limits_um('X') == {
        'min': pytest.approx(117_690 - 123_710 + 1_000),
        'max': pytest.approx(117_690),
    }
    assert config.limits_um('Y') == {
        'min': pytest.approx(950),
        'max': pytest.approx(950 + 79_790 - 1_000),
    }
    # Z's travel is measured from its switch; its 0 sits 3643 microsteps
    # (200 um) above it.
    assert config.zero_above_reference_um('Z') == pytest.approx(200, abs=0.01)
    assert config.limits_um('Z') == {
        'min': 0.0,
        'max': pytest.approx(12_030 - 200 - 1_000, abs=0.01),
    }


def test_optics_and_led_block_come_from_the_models_row():
    """The 6110 carries neither, so the capability build falls through to
    the catalogue's optics, and the layer identity to the model's row."""
    config = Tmcm6110Config(SHIPPED_MOTOR_DEFAULTS)
    assert config.led_block() is None
    assert config.board_config_read_ok is True
    assert _resolve_lens_focal_length_mm(config, {'LensFocalLength': 47.8}) == 47.8
    assert _resolve_pixel_size_um(config, {'PixelSize': 2.2}, None) == 2.2


def test_the_homing_phases_are_handed_over_as_the_section_holds_them():
    """Checked whole at load; the shipped file's Index Search Max
    Acceleration of 50 is a homing value, not the axis's full value, so the
    limit's floor does not apply to it."""
    config = Tmcm6110Config(SHIPPED_MOTOR_DEFAULTS)
    assert dict(config.homing('Z')) == {
        'Switch Search': {
            'Reference Search Mode': 65,
            'Reference Search Speed': 500,
            'Back-off Microsteps': 3643,
        }
    }
    assert config.homing('X')['Switch Pre-move']['Max Positioning Speed'] == 2047
    assert config.homing('Y')['Index Search']['Max Acceleration'] == 50
    assert config.homing('Y')['Index Search']['Approach Speeds'] == (1000, 100)


def test_the_constants_cannot_be_edited_through_a_read():
    config = Tmcm6110Config(SHIPPED_MOTOR_DEFAULTS)
    with pytest.raises(TypeError):
        config.axis_parameters('X')['Max Current'] = 255
    with pytest.raises(TypeError):
        config.homing('X')['Index Search']['Reference Search Speed'] = 2047


def test_no_section_is_refused():
    defaults = copy.deepcopy(SHIPPED_MOTOR_DEFAULTS)
    del defaults[SECTION]
    with pytest.raises(ValueError, match="no 'TMCM-6110' section"):
        Tmcm6110Config(defaults)


@pytest.mark.parametrize(
    ('edit', 'named'),
    [
        (
            lambda s: s['Axis Parameters']['Y'].pop('Pulse Divisor'),
            r'Axis Parameters\.Y\.Pulse Divisor is missing',
        ),
        (
            lambda s: s['Axis Parameters']['X'].update({'Max Positioning Speed': 2048}),
            r'Max Positioning Speed = 2048 is outside 0\.\.2047',
        ),
        (
            lambda s: s['Axis Parameters']['Z'].update({'Max Current': 48.0}),
            r'Z\.Max Current is not an integer',
        ),
        (
            lambda s: s['Axis Drive']['Z'].update({'mm per Drive Revolution': 0}),
            r'Axis Drive\.Z\.mm per Drive Revolution = 0 is not a positive',
        ),
        (lambda s: s['Axis Direction'].update({'X': 0}), r'Axis Direction\.X = 0 is not 1 or -1'),
        (lambda s: s['Axis Travel Limit'].pop('Y'), r'Axis Travel Limit\.Y is missing'),
        (lambda s: s['Index Position'].pop('X'), r'Index Position\.X is missing'),
        (
            lambda s: s['Homing']['Z']['Switch Search'].pop('Back-off Microsteps'),
            r'Homing\.Z\.Switch Search\.Back-off Microsteps is missing',
        ),
        (
            lambda s: s['Index Position'].update({'Y': '0.95'}),
            r"Index Position\.Y = '0\.95' is not a number",
        ),
        (
            lambda s: s['Index Position'].update({'X': float('nan')}),
            r'Index Position\.X = nan is not a number',
        ),
        (lambda s: s.pop('Homing'), r'Homing is missing'),
        (
            lambda s: s['Homing']['Y']['Index Search'].pop('Back-off Microsteps'),
            r'Homing\.Y\.Index Search\.Back-off Microsteps is missing',
        ),
        (
            lambda s: s['Homing']['X']['Index Search'].update({'Max Acceleration': 99999}),
            r'Homing\.X\.Index Search\.Max Acceleration = 99999 is outside 0\.\.2047',
        ),
        (
            lambda s: s['Homing']['X']['Index Search'].update({'Approach Speeds': [1000]}),
            r'Approach Speeds is not two speeds',
        ),
        (
            lambda s: s['Homing']['Z']['Switch Search'].update({'Reference Search Speed': 5.0}),
            r'Homing\.Z\.Switch Search\.Reference Search Speed is not an integer',
        ),
        (
            lambda s: s['Axis Parameters']['X'].update({'Max Acceleration': 50}),
            r'Axis Parameters\.X\.Max Acceleration = 50 is below 51',
        ),
        (lambda s: s.pop('Travel Margin'), r'Travel Margin is missing'),
        (
            lambda s: s.update({'Travel Margin': 80}),
            r'Travel Margin = 80 is not below the Y travel \(79\.79 mm\)',
        ),
    ],
)
def test_a_bad_value_is_refused_naming_it(edit, named):
    with pytest.raises(ValueError, match=named):
        Tmcm6110Config(_defaults_with(edit))
