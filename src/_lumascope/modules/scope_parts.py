# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The names of a scope's parts: the boards it is brought up from, and every part a refusal names.

A leaf module beside ``axes``, so the refusals (``exceptions``) and the
bring-up record name a part without importing each other. One table holds
each part's words -- how a sentence names it, what its absence is refused
as and what the person at the scope is told -- and one each board's label,
so a part is worded the same in a refusal, a run's ending, bring-up's
progress and its record.
"""

import enum
from dataclasses import dataclass

from _lumascope.modules.axes import StageAxis


class Board(enum.StrEnum):
    """A board a scope is brought up from, in the order bring-up connects them.

    The parts that connect, and so the ones that can be not connected: the
    bring-up record's parts, ``Lumascope.unconnected_parts()``, and the
    boards a simulated fault targets. Each is also a ``Part``, by the same
    value, so ``Part(board)`` is the board's part.
    """

    MOTOR = 'motor'
    LED = 'led'
    CAMERA = 'camera'

    @property
    def noun(self) -> str:
        """The board as a sentence names it: the LED controller is not connected."""
        return _WORDS[Part(self)].noun

    @property
    def label(self) -> str:
        """The board as a list labels it: ``Connecting the LED Controller``."""
        return _LABELS[self]


class Part(enum.StrEnum):
    """A part of a scope that a command can need and the scope can lack.

    A board, by its board's value; a motion part: the motors, a stage
    axis's motor (``AXIS``, its axis beside it), the turret, the XY stage;
    a channel and a channel's LED (their channel beside them); an LED board
    address (its address beside it); and the features a board may not
    have: fan control, LED engineering mode, text commands. A refusal names
    the one it lacked as a ``MissingPart``.
    """

    MOTOR = Board.MOTOR.value
    LED = Board.LED.value
    CAMERA = Board.CAMERA.value
    MOTORS = 'motors'
    AXIS = 'axis'
    TURRET = 'turret'
    XY_STAGE = 'xy_stage'
    CHANNEL = 'channel'
    CHANNEL_LED = 'channel_led'
    LED_ADDRESS = 'led_address'
    FAN_CONTROL = 'fan_control'
    LED_ENGINEERING_MODE = 'led_engineering_mode'
    TEXT_COMMANDS = 'text_commands'


@dataclass(frozen=True)
class _Words:
    """A part's words: ``noun`` and ``sentence`` name the ``axis``, ``channel`` or ``address`` the part ``takes``."""

    noun: str
    reason: str
    sentence: str
    takes: str | None = None


# A part that is not connected stays so until LumaViewPro restarts: nothing
# reconnects a board or a camera during a session, so each sentence names the
# restart, as the bring-up words for the same parts do.
_WORDS: dict[Part, _Words] = {
    Part.MOTOR: _Words(
        'motor controller',
        'not_connected',
        'The motor controller is not connected. Check the USB cable and that '
        'no other program is holding the port, then restart LumaViewPro.',
    ),
    Part.LED: _Words(
        'LED controller',
        'not_connected',
        'The LED controller is not connected. Check the USB cable and 24V power, '
        'then restart LumaViewPro.',
    ),
    Part.CAMERA: _Words(
        'camera',
        'not_connected',
        'The camera is not connected. Check its cable, then restart LumaViewPro.',
    ),
    Part.MOTORS: _Words('motors', 'axis_absent', 'This microscope has no motors.'),
    Part.AXIS: _Words(
        '{axis} motor', 'axis_absent', 'This microscope has no {axis} motor.', takes='axis'
    ),
    Part.TURRET: _Words('turret', 'axis_absent', 'This microscope has no turret.'),
    Part.XY_STAGE: _Words(
        'XY stage',
        'axis_absent',
        'This microscope has no XY stage, so it has one field and no wells to move '
        'between: its plate is Center Plate.',
    ),
    # A channel, not its LED, for a command that takes a channel rather than
    # lights one: a channel's settings, its focus, a still named for it. Lumi
    # has no LED, so "no Lumi LED" would be true and beside the point.
    Part.CHANNEL: _Words(
        '{channel} channel',
        'not_on_this_scope',
        'This microscope has no {channel} channel.',
        takes='channel',
    ),
    Part.CHANNEL_LED: _Words(
        '{channel} LED',
        'not_on_this_scope',
        'This microscope has no {channel} LED.',
        takes='channel',
    ),
    Part.LED_ADDRESS: _Words(
        'LED address {address}',
        'not_on_this_scope',
        'This microscope has no LED at address {address}.',
        takes='address',
    ),
    Part.FAN_CONTROL: _Words(
        'fan control',
        'not_on_this_scope',
        "This microscope's motor controller does not support fan control.",
    ),
    Part.LED_ENGINEERING_MODE: _Words(
        'LED engineering mode',
        'not_on_this_scope',
        "This microscope's LED controller has no engineering mode.",
    ),
    Part.TEXT_COMMANDS: _Words(
        'text commands',
        'not_on_this_scope',
        "This microscope's controller takes no text commands.",
    ),
}

# A board's label, as its noun capitalised for a list.
_LABELS: dict[Board, str] = {
    Board.MOTOR: 'Motor Controller',
    Board.LED: 'LED Controller',
    Board.CAMERA: 'Camera',
}

# Every reason a missing part is refused as.
MISSING_PART_REASONS = frozenset(words.reason for words in _WORDS.values())


def words_of(part: Part) -> _Words:
    """The table's row for ``part``."""
    return _WORDS[part]


def axis_part(axis: str) -> tuple[Part, StageAxis | None]:
    """The part a motion axis name names, and the stage axis beside it: ``'T'`` is the turret.

    Raises:
        ValueError: ``axis`` is not ``'X'``, ``'Y'``, ``'Z'`` or ``'T'``.
    """
    if axis == 'T':
        return Part.TURRET, None
    try:
        return Part.AXIS, StageAxis(axis)
    except ValueError:
        raise ValueError(f'{axis!r} is not a motion axis') from None
