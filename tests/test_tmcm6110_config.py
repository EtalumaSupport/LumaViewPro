# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The LS720 stage's constants: the shipped section reads, converts, and refuses.

The TMCM-6110 section of the shipped motor defaults carries LumaView
Classic's production LS720 values. The config reads it once and refuses a
section with a missing or out-of-range value, naming the key, rather than
driving a stage with a value nobody chose.
"""

import copy

import pytest

from drivers.tmcm6110_config import SECTION, Tmcm6110Config
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


def test_ramp_params_reproduce_the_manuals_worked_example():
    """TMCM-6110 TMCL firmware manual, 6.4: speed 1000 at pulse_div 1 is
    122070.31 microsteps/s; acceleration 1000 at pulse_div 1, ramp_div 1 is
    119.21 MHz/s, microsteps/s per second. At 1000 microsteps per mm, um and
    microsteps are the same unit."""

    def edit(section):
        section['Axis Parameters']['X'].update(
            {
                'Microstep Resolution': 0,
                'Pulse Divisor': 1,
                'Ramp Divisor': 1,
                'Max Positioning Speed': 1000,
                'Max Acceleration': 1000,
            }
        )
        section['Axis Drive']['X'] = {
            'Full Steps per Motor Revolution': 1000,
            'Motor Revolutions per Drive Revolution': 1,
            'mm per Drive Revolution': 1.0,
        }

    ramp = Tmcm6110Config(_defaults_with(edit)).ramp_params('X')
    assert ramp['vmax'] == pytest.approx(122070.31, abs=0.01)
    assert ramp['amax'] == pytest.approx(119.21e6, abs=0.01e6)
    assert ramp['dmax'] == ramp['amax']


def test_the_travel_is_the_measured_far_switch_and_the_margin_sits_inside_it():
    """The bench LS720, 2026-10-05, row 6: each far switch's distance from
    the index, driven onto at a quarter of the axis's speed. The margin is
    for the units not measured."""
    config = Tmcm6110Config(SHIPPED_MOTOR_DEFAULTS)
    assert config.travel_limit_um('X') == pytest.approx(123_710)
    assert config.travel_limit_um('Y') == pytest.approx(79_790)
    assert config.travel_limit_um('Z') == pytest.approx(12_030)
    assert config.travel_margin_um() == pytest.approx(1_000)


def test_ramp_params_are_a_profile_for_every_axis():
    """The API builds a move profile only from a non-empty ramp."""
    config = Tmcm6110Config(SHIPPED_MOTOR_DEFAULTS)
    for axis in ('X', 'Y', 'Z'):
        ramp = config.ramp_params(axis)
        assert set(ramp) == {'vmax', 'amax', 'dmax'}
        assert all(v > 0 for v in ramp.values())


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
        'Switch Search': {'Reference Search Mode': 65, 'Reference Search Speed': 500}
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
