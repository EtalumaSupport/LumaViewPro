# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The record of a scope's bring-up: what came up, what did not and why, what was substituted.

Held by the scope that was brought up and read through the session, so a
client that subscribes after bring-up -- a REST client, a script joining a
running instrument -- reads the same facts the client that heard bring-up
was told. Nothing here decides what is shown: the scope reports its outcomes
from these facts once, through the reporter, and this is what stays behind.
"""

from __future__ import annotations

import enum
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING
from modules.api_surface import api, api_fields
from modules.exceptions import MissingPart

if TYPE_CHECKING:
    from modules.settings_init import SettingValue


class Part(enum.StrEnum):
    """The parts a scope is made of, in the order the record lists them."""

    MOTOR = 'motor'
    LED = 'led'
    CAMERA = 'camera'

    @property
    def noun(self) -> str:
        """The part as a sentence names it: the LED controller is not connected."""
        return _PART_NOUNS[self]


# A part's noun is the one its refusal names (``MissingPart``), so a run's
# ending and a command's refusal word one part alike.
_PART_NOUNS = {
    Part.MOTOR: MissingPart.MOTOR_CONTROLLER.name,
    Part.LED: MissingPart.LED_CONTROLLER.name,
    Part.CAMERA: MissingPart.CAMERA.name,
}

PART_LABELS = {Part.MOTOR: 'Motor Controller', Part.LED: 'LED Board', Part.CAMERA: 'Camera'}

# A cause's words inside a list of parts: "Motor Controller (port in use)".
CAUSE_PHRASES = {
    'not_detected': 'not detected on USB',
    'port_in_use': 'port in use',
    'not_responding': 'not responding',
    'connect_failed': 'connect failed',
    'no_driver': 'no driver',
    'camera_driver_unavailable': 'not detected; a driver did not load',
    'camera_firmware_missing': 'firmware file missing',
    'camera_driver_unusable': 'on no driver, or one LumaViewPro cannot use',
    'camera_not_initialized': 'not initialized',
    'safety_off_failed': 'safety LEDs-off not confirmed',
    'stream_not_started': 'stream did not start',
}


@api_fields('cause', 'detail', 'expected', 'part', 'up')
@dataclass(frozen=True)
class PartStatus:
    """One part of the scope at bring-up.

    Attributes:
        part: Which part.
        up: Whether a real driver came up for it.
        expected: Whether this scope's model has one. A manual scope has no
            motor board, so its absence is not a failure; until the model is
            known every part is expected.
        cause: Why it is not up, as a code from ``CAUSE_PHRASES``; or, on a
            part that is up, a problem it met coming up (the LED board's
            safety LEDs-off not confirmed; the camera's stream that did not
            start, which a later start that succeeds does not clear). None
            when nothing is wrong.
        detail: The driver's or the error's own words, for the log and the
            record; empty when there are none.
    """

    part: Part
    up: bool
    expected: bool = True
    cause: str | None = None
    detail: str = ''

    @api
    @property
    def missing(self) -> bool:
        """Expected and not up: the part this scope should have and lacks.

        Raises:
            Nothing.
        """
        return self.expected and not self.up

    @api(in_process=True, reads=True)
    def describe(self) -> str:
        """The part's label with its cause: ``Motor Controller (port in use)``.

        Raises:
            Nothing.
        """
        label = PART_LABELS[self.part]
        if self.cause is None:
            return label
        return f'{label} ({CAUSE_PHRASES[self.cause]})'


@api_fields('saved', 'setting', 'used')
@dataclass(frozen=True)
class Substitution:
    """A saved setting the camera could not take, and what bring-up used instead.

    What becomes of the saved value is the setting's own rule, not this
    record's: the record says what was saved and what ran. The binning the
    camera delivered at is stored in the saved binning's place, with the
    frame it belongs to, and so is a frame refitted to the scope's maximum;
    the saved image mode stays saved.
    """

    setting: str
    saved: SettingValue
    used: SettingValue


@api_fields('path', 'reason')
@dataclass(frozen=True)
class SettingsSetAside:
    """The user's settings file that could not be used, and why.

    The app runs on the shipped template while this stands, and refuses to
    save over the file until the person decides what happens to it.
    """

    path: pathlib.Path
    reason: str


@api_fields('parts', 'settings_set_aside', 'substitutions')
@dataclass(frozen=True)
class BringUpRecord:
    """What bring-up found, substituted and set aside."""

    parts: tuple[PartStatus, ...]
    substitutions: tuple[Substitution, ...] = ()
    settings_set_aside: SettingsSetAside | None = None

    @api
    @property
    def missing(self) -> tuple[PartStatus, ...]:
        """The parts this scope should have and lacks.

        Raises:
            Nothing.
        """
        return tuple(status for status in self.parts if status.missing)
