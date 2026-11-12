# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""MotorConfig -- reads per-unit hardware configuration from motorconfig.json.

Provides axis travel limits, microstep conversion factors, turret positions,
board identity (model/serial), and optics parameters. Falls back to defaults
for any missing keys.
"""

import copy
import math
import types
from collections.abc import Mapping
from typing import ClassVar

from _lumascope.drivers.exceptions import ConfigReadError
from _lumascope.lib.log_helpers import logger


# The axes whose travel bounds a move. The turret's T is a slot number, not
# a distance, and has no limit.
LIMITED_AXES = ('X', 'Y', 'Z')


def _is_number(value: object) -> bool:
    """Whether ``value`` is a finite int or float; a bool is not a number here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


TRAVEL = 'Axis Travel Limit'
MARGIN = 'Axis Travel Margin'


def _travel_problem(travel_mm: object) -> str | None:
    if not _is_number(travel_mm) or travel_mm <= 0:
        return f'travel {travel_mm!r} is not a positive number of mm'
    return None


def _margin_problem(margin_mm: object) -> str | None:
    if not _is_number(margin_mm) or margin_mm < 0:
        return f'margin {margin_mm!r} is not a number of mm of zero or more'
    return None


# What each figure of an axis's travel must be, by section.
_FIGURE_PROBLEM = {TRAVEL: _travel_problem, MARGIN: _margin_problem}


def unusable_travel(travel_mm: object, margin_mm: object) -> str | None:
    """Why an axis's stated travel and travel margin give no limit, or None.

    The limit is the travel less the margin, so the travel is a positive
    number and the margin a number of zero or more below it. The shipped
    defaults and a board's own figures are each judged by this where each
    enters, so the limit is never computed from a pair it cannot use.
    """
    problem = _travel_problem(travel_mm) or _margin_problem(margin_mm)
    if problem is not None:
        return problem
    if margin_mm >= travel_mm:
        return f'margin {margin_mm!r} mm leaves none of the travel {travel_mm!r} mm'
    return None


def read_only_axes_config(axes_config: dict) -> Mapping:
    """Freeze a motion driver's per-axis config, every level of it.

    The travel limits in here are the bound a move is refused against, and
    every read of them hands out this object. A plain dict let a caller
    that edited what it read move the bound itself, and the stage was then
    driven past its physical travel with nothing refusing it. Read-only,
    the edit raises TypeError at the line that tries it.
    """
    return types.MappingProxyType(
        {
            axis: types.MappingProxyType(
                {
                    key: types.MappingProxyType(dict(value)) if isinstance(value, dict) else value
                    for key, value in config.items()
                }
            )
            for axis, config in axes_config.items()
        }
    )


class MotorConfig:
    # True until a board config READ fails. False means the per-unit values
    # may exist on the board but are unavailable -- a different state from
    # "the board has none", and
    # consumers deciding between per-unit and fallback sources need the
    # difference or they silently serve the wrong unit's answer.
    board_config_read_ok: bool = True

    # Keys that describe ONE physical assembly (the channel/filterset block,
    # and the old Layers block it replaced). They are applied whole, never
    # deep-merged: a field-by-field merge of two sources could describe
    # hardware that does not exist.
    _WHOLESALE_KEYS: ClassVar[frozenset] = frozenset({'Channels', 'Layers', 'Filterset'})

    def __init__(self, defaults: Mapping):
        # A deep copy: the board merge edits nested sections in place, and
        # one loaded table is handed to every board built.
        self._config = copy.deepcopy(dict(defaults))

    def mark_board_read_failed(self) -> None:
        """Record that the board's config could not be read this session."""
        self.board_config_read_ok = False

    def led_block(self) -> dict | None:
        """The per-unit channel/filterset identity keys the config holds, or None when none.

        Whichever of ``Channels``, the old ``Layers`` and ``Filterset`` are
        present, returned whole because they describe one physical filterset
        assembly; the identity resolver decides what each means.
        """
        block = {key: self._config[key] for key in self._WHOLESALE_KEYS if key in self._config}
        return block or None

    def update_from_board(self, board_config: dict) -> None:
        """Merge per-unit config from the motor controller on top of defaults.

        Only keys explicitly present in board_config override defaults.
        Missing keys are left at their default values.  This is safe for
        boards with minimal configs (e.g., only serial number and model).

        Basic type validation is applied to numeric fields -- if a board
        sends a non-numeric value where a number is expected, the board
        value is rejected and the default is kept. A travel or travel margin
        the limit cannot be computed from is rejected the same way, alone
        and then as the pair the merge would give (`_keep_a_usable_travel`).

        The wholesale keys (the LED/filterset block) are assigned whole,
        never deep-merged, whatever shape either side carries.
        """
        validated = self._validate_board_config(board_config)
        self._keep_a_usable_travel(validated)
        for key in self._WHOLESALE_KEYS:
            if key in validated:
                self._config[key] = validated.pop(key)
        self._deep_merge(self._config, validated)

    def _keep_a_usable_travel(self, validated: dict) -> None:
        """Drop the board's travel figures that would leave an axis no limit.

        Each axis is judged on the pair the merge would give it: the board's
        figure where it states one, the one held where it does not. The
        board's margin is dropped first, then its travel, so what remains at
        worst is the shipped pair, judged usable where it was read.
        """
        for axis in LIMITED_AXES:
            for section in (MARGIN, TRAVEL):
                travel = validated.get(TRAVEL, {}).get(axis, self._config[TRAVEL][axis])
                margin = validated.get(MARGIN, {}).get(axis, self._config[MARGIN][axis])
                problem = unusable_travel(travel, margin)
                if problem is None:
                    break
                if axis in validated.get(section, {}):
                    logger.warning(
                        f'[MotorConfig] Rejecting board value {section}.{axis}='
                        f'{validated[section].pop(axis)!r}: {problem}'
                    )

    # Sections where values must be numeric (int or float).
    _NUMERIC_SECTIONS = frozenset(
        {
            'Axis Microsteps per mm / Objective',
            TRAVEL,
            MARGIN,
            'Axis Present',
            'Axis antibacklash',
            'Axis Offset',
            'Axis Timeout',
            'Initial Position after home',
        }
    )
    # Sections read by key: a board value that is not a mapping cannot be
    # read, so it is rejected and the default kept, as a non-numeric value is.
    _MAPPING_SECTIONS = _NUMERIC_SECTIONS | {'TurretPosition', 'Optics'}

    def _validate_board_config(self, board_config: dict) -> dict:
        """Return a copy of board_config with invalid values removed."""
        cleaned = {}
        for key, value in board_config.items():
            if key in self._MAPPING_SECTIONS and not isinstance(value, dict):
                logger.warning(
                    f'[MotorConfig] Rejecting board section {key}={value!r} '
                    f'(type={type(value).__name__}): not a mapping'
                )
                continue
            if key in self._NUMERIC_SECTIONS and isinstance(value, dict):
                valid_axes = {}
                for axis, val in value.items():
                    if not isinstance(val, (int, float)):
                        logger.warning(
                            f'[MotorConfig] Rejecting non-numeric board value '
                            f'{key}.{axis}={val!r} (type={type(val).__name__})'
                        )
                    elif key in _FIGURE_PROBLEM and (problem := _FIGURE_PROBLEM[key](val)):
                        logger.warning(
                            f'[MotorConfig] Rejecting board value {key}.{axis}={val!r}: {problem}'
                        )
                    else:
                        valid_axes[axis] = val
                cleaned[key] = valid_axes
            elif key == 'TurretPosition' and isinstance(value, dict):
                valid_pos = {}
                for pos, val in value.items():
                    if isinstance(val, (int, float)):
                        valid_pos[pos] = val
                    else:
                        logger.warning(
                            f'[MotorConfig] Rejecting non-numeric turret position {pos}={val!r}'
                        )
                cleaned[key] = valid_pos
            elif key == 'Optics' and isinstance(value, dict):
                valid_optics = {}
                for okey, oval in value.items():
                    if isinstance(oval, (int, float, str)):
                        valid_optics[okey] = oval
                    else:
                        logger.warning(
                            f'[MotorConfig] Rejecting invalid optics value {okey}={oval!r}'
                        )
                cleaned[key] = valid_optics
            else:
                cleaned[key] = value
        return cleaned

    @staticmethod
    def _deep_merge(base: dict, override: dict):
        """Recursively merge override into base (mutates base)."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                MotorConfig._deep_merge(base[key], value)
            else:
                base[key] = value

    def _axis_lookup(self, section: str, axis: str, default=0):
        sub = self._config.get(section)
        if sub is None:
            logger.warning(f"[MotorConfig] Section '{section}' not found")
            return default
        axis = axis.upper()
        val = sub.get(axis)
        if val is None:
            logger.warning(f"[MotorConfig] Axis '{axis}' not in section '{section}'")
            return default
        return val

    # --- Axis properties ---

    def usteps_per_mm(self, axis: str) -> int:
        return int(self._axis_lookup('Axis Microsteps per mm / Objective', axis, default=20157))

    def travel_limit_mm(self, axis: str) -> float:
        """The axis's stated travel, end to end, in mm.

        Raises:
            KeyError: the config states no travel for ``axis``.
        """
        return float(self._config[TRAVEL][axis.upper()])

    def travel_limit_um(self, axis: str) -> float:
        return self.travel_limit_mm(axis) * 1000.0

    def travel_margin_mm(self, axis: str) -> float:
        """How far inside the axis's far end its limit sits, in mm.

        Raises:
            KeyError: the config states no travel margin for ``axis``.
        """
        return float(self._config[MARGIN][axis.upper()])

    def limits_um(self, axis: str) -> dict[str, float]:
        """The travel a move of ``axis`` is held to, in um: from the home
        position to the margin inside the far end, so that no move the
        limits allow reaches the far switch or the stage's end."""
        return {
            'min': 0.0,
            'max': (self.travel_limit_mm(axis) - self.travel_margin_mm(axis)) * 1000.0,
        }

    def axis_present(self, axis: str) -> bool:
        return bool(self._axis_lookup('Axis Present', axis, default=True))

    def antibacklash_um(self, axis: str) -> int:
        return int(self._axis_lookup('Axis antibacklash', axis, default=0))

    def position_after_home_usteps(self, axis: str) -> int:
        """Where the firmware's HOME leaves ``axis``, in microsteps.

        The firmware homes, sets the new zero, then sets these targets, so
        a homed stage stands here, not at zero.

        Raises:
            KeyError: ``axis`` has no position after home (T: the turret
                stays where its home leaves it).
        """
        return int(self._config['Initial Position after home'][axis.upper()])

    # --- Motion ramp parameters (TMC5072 6-point ramp) ---

    # Default ramp parameters from INI files (usteps/sec and usteps/sec^2).
    # Keyed by axis. These are hardware constants set at firmware load time.
    # Future: query dynamically from board via SPI or v3.1 firmware command.
    _DEFAULT_RAMP: ClassVar[dict] = {
        'X': {
            'vstart': 0,
            'a1': 0,
            'v1': 0,
            'amax': 50000,
            'vmax': 800000,
            'dmax': 50000,
            'd1': 0,
            'vstop': 10,
        },
        'Y': {
            'vstart': 0,
            'a1': 0,
            'v1': 0,
            'amax': 50000,
            'vmax': 800000,
            'dmax': 50000,
            'd1': 0,
            'vstop': 10,
        },
        'Z': {
            'vstart': 0,
            'a1': 0,
            'v1': 0,
            'amax': 25000,
            'vmax': 400000,
            'dmax': 25000,
            'd1': 0,
            'vstop': 100,
        },
        'T': {
            'vstart': 0,
            'a1': 0,
            'v1': 0,
            'amax': 5000,
            'vmax': 128000,
            'dmax': 5000,
            'd1': 0,
            'vstop': 10,
        },
    }

    def ramp_params_usteps(self, axis: str) -> dict:
        """Return all TMC5072 ramp parameters in ustep units for an axis."""
        axis = axis.upper()
        return dict(self._DEFAULT_RAMP.get(axis, self._DEFAULT_RAMP['X']))

    # --- Board identity ---

    def model(self) -> str:
        return self._config.get('Microscope', 'LS850')

    def serial_number(self) -> str:
        return self._config.get('Serial Number', 'Unknown')

    def hardware_rev(self) -> str:
        return self._config.get('HardwareRev', 'Unknown')

    # --- Image center offset ---

    def image_center_offset(self) -> tuple[int, int]:
        """Return (X, Y) optical-center offset in sensor pixels.

        The collimator-measured offset of the sensor from the tube lens's
        optical axis, taken with no objective so it is objective-independent.
        Defaults to (0, 0) -- geometric center -- when the unit was never
        characterized.
        """
        section = self._config.get('ImageCenter')
        if section is None:
            return (0, 0)
        return (int(section.get('X', 0)), int(section.get('Y', 0)))

    # --- Turret ---

    def turret_position_usteps(self, position: int) -> int:
        """Get microstep position for turret slot (1-based)."""
        section = self._config.get('TurretPosition')
        if section is None:
            return 0
        val = section.get(str(position))
        if val is None:
            logger.warning(f'[MotorConfig] Turret position {position} not found')
            return 0
        return int(val)

    # --- Optics ---

    def lens_focal_length(self) -> float:
        """Tube-lens focal length (mm) from the loaded Optics block.

        Raises:
            HardwareError: the motorconfig declares no numeric
                Optics.LensFocalLength. A scope that has a motorconfig is
                expected to declare its optics; absence is a malformed
                config, not a normal state (the Classic, which has no optics
                here, has no motorconfig at all). The capability builder
                catches this and degrades the scale honestly.
        """
        try:
            return float(self._config['Optics']['LensFocalLength'])
        except (KeyError, TypeError, ValueError) as ex:
            raise ConfigReadError(
                f'motorconfig declares no numeric Optics.LensFocalLength: {ex}'
            ) from ex

    def pixel_size(self) -> float:
        """Camera pixel size (um/pixel) from the loaded Optics block.

        Raises:
            HardwareError: the motorconfig declares no numeric
                Optics.PixelSize. See lens_focal_length for why absence is a
                malformed config rather than a normal state.
        """
        try:
            return float(self._config['Optics']['PixelSize'])
        except (KeyError, TypeError, ValueError) as ex:
            raise ConfigReadError(f'motorconfig declares no numeric Optics.PixelSize: {ex}') from ex
