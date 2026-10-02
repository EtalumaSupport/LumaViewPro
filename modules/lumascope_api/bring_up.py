# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The record of a scope's bring-up: what came up, what did not and why, what was substituted.

Held by the scope that was brought up and read through the session, so a
client that subscribes after bring-up -- a REST client, a script joining a
running instrument -- reads the same facts the client that heard bring-up
was told. Nothing here decides what is shown: the scope reports its outcomes
from these facts once, through the reporter, and this is what stays behind.
"""

from __future__ import annotations

from dataclasses import dataclass

# The parts a scope is made of, in the order the record lists them.
MOTOR = 'motor'
LED = 'led'
CAMERA = 'camera'

PART_LABELS = {MOTOR: 'Motor Controller', LED: 'LED Board', CAMERA: 'Camera'}

# A cause's words inside a list of parts: "Motor Controller (port in use)".
CAUSE_PHRASES = {
    'not_detected': 'not detected on USB',
    'port_in_use': 'port in use',
    'not_responding': 'not responding',
    'connect_failed': 'connect failed',
    'no_driver': 'no driver',
    'camera_in_use': 'in use by another application',
    'camera_port_in_use': 'port in use',
    'camera_not_detected': 'not detected',
    'camera_not_initialized': 'not initialized',
    'safety_off_failed': 'safety LEDs-off not confirmed',
}


@dataclass(frozen=True)
class PartStatus:
    """One part of the scope at bring-up.

    Attributes:
        part: ``MOTOR``, ``LED`` or ``CAMERA``.
        up: Whether a real driver came up for it.
        expected: Whether this scope's model has one. A manual scope has no
            motor board, so its absence is not a failure; until the model is
            known every part is expected.
        cause: Why it is not up, as a code from ``CAUSE_PHRASES``; or, on a
            part that is up, a problem it reported while connecting (the LED
            board's safety LEDs-off not confirmed). None when nothing is wrong.
        detail: The driver's or the error's own words, for the log and the
            record; empty when there are none.
    """

    part: str
    up: bool
    expected: bool = True
    cause: str | None = None
    detail: str = ''

    @property
    def missing(self) -> bool:
        """Expected and not up: the part this scope should have and lacks."""
        return self.expected and not self.up

    def describe(self) -> str:
        """The part's label with its cause: ``Motor Controller (port in use)``."""
        label = PART_LABELS[self.part]
        if self.cause is None:
            return label
        return f'{label} ({CAUSE_PHRASES[self.cause]})'


@dataclass(frozen=True)
class Substitution:
    """A saved setting the camera could not take, and what bring-up used instead.

    The saved value stays saved: it is the person's preference, and the
    camera that cannot take it may not be the one they set it on.
    """

    setting: str
    saved: object
    used: object


@dataclass(frozen=True)
class SettingsSetAside:
    """The user's settings file that could not be used, and why.

    The app runs on the shipped template while this stands, and refuses to
    save over the file until the person decides what happens to it.
    """

    path: str
    reason: str


@dataclass(frozen=True)
class BringUpRecord:
    """What bring-up found, substituted and set aside."""

    parts: tuple[PartStatus, ...]
    substitutions: tuple[Substitution, ...] = ()
    settings_set_aside: SettingsSetAside | None = None

    def part(self, name: str) -> PartStatus:
        """The status of ``name``; KeyError when this record has no such part."""
        for status in self.parts:
            if status.part == name:
                return status
        raise KeyError(name)

    @property
    def missing(self) -> tuple[PartStatus, ...]:
        """The parts this scope should have and lacks."""
        return tuple(status for status in self.parts if status.missing)

    def substitution(self, setting: str) -> Substitution | None:
        """What bring-up used for ``setting`` instead of the saved value, if anything."""
        for sub in self.substitutions:
            if sub.setting == setting:
                return sub
        return None
