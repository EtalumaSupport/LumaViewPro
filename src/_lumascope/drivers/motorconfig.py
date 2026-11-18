# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""MotorConfig -- reads per-unit hardware configuration from motorconfig.json.

Provides axis travel limits, microstep conversion factors, turret positions,
board identity (model/serial), and optics parameters. Falls back to defaults
for any missing keys.
"""

import copy
import types
from collections.abc import Mapping
from typing import ClassVar

from _lumascope.drivers.exceptions import ConfigReadError, PartAbsentError
from _lumascope.lib.log_helpers import logger
from _lumascope.lib.real_numbers import is_count, is_finite_number


# The axes whose travel bounds a move. The turret's T is a slot number, not
# a distance, and has no limit.
LIMITED_AXES = ('X', 'Y', 'Z')


TRAVEL = 'Axis Travel Limit'
MARGIN = 'Axis Travel Margin'
MICROSTEPS = 'Axis Microsteps per mm / Objective'
ANTIBACKLASH = 'Axis antibacklash'
AFTER_HOME = 'Initial Position after home'
TURRET = 'TurretPosition'
OPTICS = 'Optics'


def _travel_problem(travel_mm: object) -> str | None:
    if not is_finite_number(travel_mm) or travel_mm <= 0:
        return f'travel {travel_mm!r} is not a positive number of mm'
    return None


def _margin_problem(margin_mm: object) -> str | None:
    if not is_finite_number(margin_mm) or margin_mm < 0:
        return f'margin {margin_mm!r} is not a number of mm of zero or more'
    return None


def _microsteps_problem(per_mm: object) -> str | None:
    # Every position read back is divided by it, and its sign is the motor's.
    if not is_count(per_mm) or per_mm < 1:
        return f'microsteps per mm {per_mm!r} is not a whole number of 1 or more'
    return None


def _antibacklash_problem(um: object) -> str | None:
    if not is_finite_number(um) or um < 0:
        return f'antibacklash {um!r} is not a number of um of zero or more'
    return None


def _position_problem(usteps: object) -> str | None:
    if not is_count(usteps) or usteps < 0:
        return f'position {usteps!r} is not a whole number of microsteps of zero or more'
    return None


def _optics_problem(figure: object) -> str | None:
    # The image scale in every saved file is computed from it.
    if not is_finite_number(figure) or figure <= 0:
        return f'optics figure {figure!r} is not a positive number'
    return None


# What every figure of each section must be, one rule a section: each entry
# of the section is judged by it (an axis's, a turret slot's, an optics
# key's). A figure is read only from a section with a row here
# (`MotorConfig._figure`), so no figure is read without a rule.
_FIGURE_PROBLEM = {
    TRAVEL: _travel_problem,
    MARGIN: _margin_problem,
    MICROSTEPS: _microsteps_problem,
    ANTIBACKLASH: _antibacklash_problem,
    AFTER_HOME: _position_problem,
    TURRET: _position_problem,
    OPTICS: _optics_problem,
}

JUDGED_SECTIONS = tuple(_FIGURE_PROBLEM)


def figure_problem(section: str, figure: object) -> str | None:
    """Why ``figure`` cannot be a figure of motor-config ``section``, or None.

    Raises:
        LookupError: ``section`` has no rule; only ``JUDGED_SECTIONS`` do.
    """
    return _rule(section)(figure)


def _rule(section: str):
    if section not in _FIGURE_PROBLEM:
        raise LookupError(f'motor-config section {section!r} has no rule for its figures')
    return _FIGURE_PROBLEM[section]


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
        An optics figure the table refuses is removed, default and all
        (`_merge_optics`).

        The wholesale keys (the LED/filterset block) are assigned whole,
        never deep-merged, whatever shape either side carries.
        """
        validated = self._validate_board_config(board_config)
        self._keep_a_usable_travel(validated)
        if OPTICS in validated:
            self._merge_optics(validated.pop(OPTICS))
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

    def _merge_optics(self, board_optics: object) -> None:
        """Merge the board's optics, removing each figure the table refuses.

        A refused figure is removed with the shipped one under it: the
        shipped focal length is the nominal lens, each unit's is measured,
        and a scale computed from the nominal figure would be wrong in every
        file the unit saves. With no figure here, the scale comes from the
        next source or is unknown. An ``Optics`` that is not a mapping
        states neither figure.
        """
        if not isinstance(board_optics, dict):
            logger.warning(
                f'[MotorConfig] Rejecting board section {OPTICS}={board_optics!r} '
                f'(type={type(board_optics).__name__}): not a mapping; '
                f'the unit has no pixel size or lens focal length'
            )
            self._config.pop(OPTICS, None)
            return
        held = self._config.setdefault(OPTICS, {})
        for key, figure in board_optics.items():
            problem = figure_problem(OPTICS, figure)
            if problem is None:
                held[key] = figure
            else:
                logger.warning(
                    f'[MotorConfig] Rejecting board value {OPTICS}.{key}={figure!r}: '
                    f'{problem}; the unit has no {key}'
                )
                held.pop(key, None)

    # Sections where values must be numeric (int or float).
    _NUMERIC_SECTIONS = frozenset({MICROSTEPS, TRAVEL, MARGIN, ANTIBACKLASH, AFTER_HOME})
    # Sections read by key: a board value that is not a mapping cannot be
    # read, so it is rejected and the default kept, as a non-numeric value is.
    _MAPPING_SECTIONS = _NUMERIC_SECTIONS | {TURRET}

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
                    if not is_finite_number(val):
                        logger.warning(
                            f'[MotorConfig] Rejecting non-numeric board value '
                            f'{key}.{axis}={val!r} (type={type(val).__name__})'
                        )
                    elif key in (TRAVEL, MARGIN) and (problem := figure_problem(key, val)):
                        logger.warning(
                            f'[MotorConfig] Rejecting board value {key}.{axis}={val!r}: {problem}'
                        )
                    else:
                        valid_axes[axis] = val
                cleaned[key] = valid_axes
            elif key == TURRET and isinstance(value, dict):
                valid_pos = {}
                for pos, val in value.items():
                    if is_finite_number(val):
                        valid_pos[pos] = val
                    else:
                        logger.warning(
                            f'[MotorConfig] Rejecting non-numeric turret position {pos}={val!r}'
                        )
                cleaned[key] = valid_pos
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

    def _figure(self, section: str, key: str) -> object:
        """The figure ``key`` of ``section``, as held.

        Every numeric read goes through here, so a section read without a
        row in the figure table fails on its first read.

        Raises:
            LookupError: ``section`` has no rule (``JUDGED_SECTIONS``).
            KeyError: the config states no such figure.
        """
        _rule(section)
        return self._config[section][key]

    # --- Axis properties ---

    def usteps_per_mm(self, axis: str) -> int:
        """The axis's microsteps per mm (T: per 90 degrees).

        Raises:
            KeyError: the config states none for ``axis``.
        """
        return int(self._figure(MICROSTEPS, axis.upper()))

    def travel_limit_mm(self, axis: str) -> float:
        """The axis's stated travel, end to end, in mm.

        Raises:
            KeyError: the config states no travel for ``axis``.
        """
        return float(self._figure(TRAVEL, axis.upper()))

    def travel_limit_um(self, axis: str) -> float:
        return self.travel_limit_mm(axis) * 1000.0

    def travel_margin_mm(self, axis: str) -> float:
        """How far inside the axis's far end its limit sits, in mm.

        Raises:
            KeyError: the config states no travel margin for ``axis``.
        """
        return float(self._figure(MARGIN, axis.upper()))

    def limits_um(self, axis: str) -> dict[str, float]:
        """The travel a move of ``axis`` is held to, in um: from the home
        position to the margin inside the far end, so that no move the
        limits allow reaches the far switch or the stage's end."""
        return {
            'min': 0.0,
            'max': (self.travel_limit_mm(axis) - self.travel_margin_mm(axis)) * 1000.0,
        }

    def antibacklash_um(self, axis: str) -> int:
        """The axis's antibacklash overshoot, in um.

        Raises:
            KeyError: the config states none for ``axis``.
        """
        return int(self._figure(ANTIBACKLASH, axis.upper()))

    def position_after_home_usteps(self, axis: str) -> int:
        """Where the firmware's HOME leaves ``axis``, in microsteps.

        The firmware homes, sets the new zero, then sets these targets, so
        a homed stage stands here, not at zero.

        Raises:
            KeyError: ``axis`` has no position after home (T: the turret
                stays where its home leaves it).
        """
        return int(self._figure(AFTER_HOME, axis.upper()))

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

    # --- Turret ---

    def turret_position_usteps(self, position: int) -> int:
        """The microstep position of turret slot ``position`` (1-based).

        Raises:
            HardwareError: the config lists no such slot.
        """
        try:
            return int(self._figure(TURRET, str(position)))
        except KeyError as ex:
            raise PartAbsentError(f'turret slot {position}: not in the motor config') from ex

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
            return float(self._figure(OPTICS, 'LensFocalLength'))
        except KeyError as ex:
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
            return float(self._figure(OPTICS, 'PixelSize'))
        except KeyError as ex:
            raise ConfigReadError(f'motorconfig declares no numeric Optics.PixelSize: {ex}') from ex
