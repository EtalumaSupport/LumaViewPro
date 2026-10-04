# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Tmcm6110Config -- the LS720 stage's constants, read from the "TMCM-6110"
section of the shipped motor defaults.

The values are LumaView Classic's production LS720 build (StageController.cs,
PRODUCTION_720). The TMCM-6110 speaks TMCL's own units, not the TMC5072's
that MotorConfig converts, so it has its own config; it answers the members
the API asks of every motor driver's config the same way MotorConfig does.

The board holds no per-unit config, so nothing here is read from the board
and nothing is merged over the defaults.
"""

import types
from collections.abc import Mapping
from typing import ClassVar

SECTION = 'TMCM-6110'
AXES = ('X', 'Y', 'Z')

# The TMC429's clock on the TMCM-6110 (TMCM-6110 TMCL firmware manual,
# section 6.4, which also gives the two conversions in ramp_params).
_F_CLK_HZ = 16_000_000


def _frozen(value):
    """A read-only copy of a JSON value: mappings proxied, lists made tuples."""
    if isinstance(value, Mapping):
        return types.MappingProxyType({k: _frozen(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(v) for v in value)
    return value


class Tmcm6110Config:
    """The LS720 stage's constants, refused whole if any is missing or out of range.

    The homing phases are the exception: each axis's must be a mapping, and
    its contents are handed over as the section holds them.
    """

    # The TMCL axis parameters each axis is initialised with, by the name the
    # section uses, and the range the TMCM-6110 firmware manual gives each.
    _AXIS_PARAMETER_RANGES: ClassVar[dict] = {
        'Max Current': (0, 255),
        'Standby Current': (0, 255),
        'Microstep Resolution': (0, 8),
        'Ramp Divisor': (0, 13),
        'Pulse Divisor': (0, 13),
        'Max Positioning Speed': (0, 2047),
        'Max Acceleration': (0, 2047),
        'Soft Stop Flag': (0, 1),
        'Right Limit Switch Disable': (0, 1),
        'Left Limit Switch Disable': (0, 1),
    }

    _DRIVE_KEYS = (
        'Full Steps per Motor Revolution',
        'Motor Revolutions per Drive Revolution',
        'mm per Drive Revolution',
    )

    # The 6110 has no per-unit config to fail to read.
    board_config_read_ok: bool = True

    def __init__(self, defaults: Mapping):
        """Read and check the section.

        Raises:
            ValueError: the section, or a value in it, is missing, not a
                number where one is needed, or outside its range; the
                message names the key.
        """
        section = defaults.get(SECTION)
        if not isinstance(section, Mapping):
            raise ValueError(f'motor defaults have no {SECTION!r} section')
        self._section = section

        self._axis_parameters = {}
        self._usteps_per_mm = {}
        self._direction = {}
        self._travel_limit_mm = {}
        self._homing = {}
        for axis in AXES:
            params = {}
            for name, (low, high) in self._AXIS_PARAMETER_RANGES.items():
                value = self._value('Axis Parameters', axis, name)
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f'{SECTION}.Axis Parameters.{axis}.{name} is not an integer')
                if not low <= value <= high:
                    raise ValueError(
                        f'{SECTION}.Axis Parameters.{axis}.{name} = {value} '
                        f'is outside {low}..{high}'
                    )
                params[name] = value
            self._axis_parameters[axis] = _frozen(params)

            steps, reduction, mm_per_rev = (
                self._positive('Axis Drive', axis, key) for key in self._DRIVE_KEYS
            )
            # The microsteps per full step are what the board is told to use,
            # so the conversion follows them rather than being a second copy.
            microsteps = 2 ** params['Microstep Resolution']
            self._usteps_per_mm[axis] = microsteps * steps * reduction / mm_per_rev

            direction = self._value('Axis Direction', axis)
            if isinstance(direction, bool) or not isinstance(direction, int) or abs(direction) != 1:
                raise ValueError(f'{SECTION}.Axis Direction.{axis} = {direction!r} is not 1 or -1')
            self._direction[axis] = direction

            self._travel_limit_mm[axis] = self._positive('Axis Travel Limit', axis)

            homing = self._value('Homing', axis)
            if not isinstance(homing, Mapping):
                raise ValueError(f'{SECTION}.Homing.{axis} is not a mapping')
            self._homing[axis] = _frozen(homing)

        measured = section.get('Axis Travel Limit Measured')
        if not isinstance(measured, bool):
            raise ValueError(f'{SECTION}.Axis Travel Limit Measured is not true or false')
        # False until the ends are measured on an LS720.
        self.travel_limits_measured: bool = measured

    def _value(self, *path: str):
        node = self._section
        for i, key in enumerate(path):
            if not isinstance(node, Mapping) or key not in node:
                raise ValueError(f'{SECTION}.{".".join(path[: i + 1])} is missing')
            node = node[key]
        return node

    def _positive(self, *path: str) -> float:
        value = self._value(*path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f'{SECTION}.{".".join(path)} = {value!r} is not a positive number')
        return float(value)

    # --- Axis properties ---

    def usteps_per_mm(self, axis: str) -> float:
        """Microsteps per mm, from the drive train and the microstep resolution."""
        return self._usteps_per_mm[axis.upper()]

    def direction(self, axis: str) -> int:
        """+1 or -1: the sign that makes the board's position 0 at the
        reference and positive away from it."""
        return self._direction[axis.upper()]

    def travel_limit_um(self, axis: str) -> float:
        return self._travel_limit_mm[axis.upper()] * 1000.0

    def axis_parameters(self, axis: str) -> Mapping:
        """The TMCL axis parameters the axis is initialised with, by name."""
        return self._axis_parameters[axis.upper()]

    def homing(self, axis: str) -> Mapping:
        """The axis's homing phases and their parameters, as the section holds them."""
        return self._homing[axis.upper()]

    # --- What the API asks of every motor driver's config ---

    def ramp_params(self, axis: str) -> dict:
        """The axis's ramp in um/s and um/s^2: vmax, and amax = dmax.

        The TMC429 ramps are trapezoidal and symmetric. From the manual,
        microsteps/s = f_clk * speed / (2**pulse_div * 2048 * 32) and
        microsteps/s^2 = f_clk**2 * accel / 2**(pulse_div + ramp_div + 29).
        """
        params = self.axis_parameters(axis)
        pulse_div = params['Pulse Divisor']
        ramp_div = params['Ramp Divisor']
        um_per_ustep = 1000.0 / self.usteps_per_mm(axis)
        vmax = _F_CLK_HZ * params['Max Positioning Speed'] / (2**pulse_div * 2048 * 32)
        amax = _F_CLK_HZ**2 * params['Max Acceleration'] / 2 ** (pulse_div + ramp_div + 29)
        return {
            'vmax': vmax * um_per_ustep,
            'amax': amax * um_per_ustep,
            'dmax': amax * um_per_ustep,
        }

    def led_block(self) -> None:
        """None: the 6110 carries no LED/filterset block; the model's row does."""
        return None

    def pixel_size(self) -> None:
        """None: the 6110 carries no optics; the model's catalogue row does."""
        return None

    def lens_focal_length(self) -> None:
        """None: the 6110 carries no optics; the model's catalogue row does."""
        return None
