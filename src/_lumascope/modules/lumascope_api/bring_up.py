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
from _lumascope.lvp_logger import logger
from _lumascope.modules.api_surface import ProgressCallback, api, api_fields
from _lumascope.modules.exceptions import Remedy
from _lumascope.modules.scope_parts import Board

if TYPE_CHECKING:
    from _lumascope.modules.settings_init import SettingValue


def connecting(board: Board) -> str:
    """The bring-up step that connects ``board``, as its progress report words it."""
    return f'Connecting the {board.label}'


# Bring-up's steps, in the order they run: each board connected, in the
# record's order, then the scope configured. Their words are the progress
# report's status lines, so a board is named one way.
CONFIGURING = 'Configuring the scope'
STEPS = (*(connecting(board) for board in Board), CONFIGURING)


def report_step(on_progress: ProgressCallback | None, step: str) -> None:
    """Tell the log, and the caller who asked, that bring-up starts ``step``.

    The percentage is the share of bring-up's steps already done, not a
    time estimate. Called before the step, on bring-up's own thread; a
    raise from ``on_progress`` is a raise from bring-up.
    """
    percent = 100 * STEPS.index(step) / len(STEPS)
    logger.info(f'[SCOPE API ] Bring-up step: {step} ({percent:.0f}%)')
    if on_progress is not None:
        on_progress(percent, step)


class PartReason(enum.StrEnum):
    """Why a part is not up, or the problem a part that is up met coming up.

    Each has the phrase a list of parts names it by: "Motor Controller
    (port in use)".
    """

    _phrase: str

    def __new__(cls, value: str, phrase: str) -> PartReason:
        member = str.__new__(cls, value)
        member._value_ = value
        member._phrase = phrase
        return member

    NOT_DETECTED = 'not_detected', 'not detected on USB'
    PORT_IN_USE = 'port_in_use', 'port in use'
    NOT_RESPONDING = 'not_responding', 'not responding'
    CONNECT_FAILED = 'connect_failed', 'connect failed'
    NO_DRIVER = 'no_driver', 'no driver'
    CAMERA_DRIVER_UNAVAILABLE = 'camera_driver_unavailable', 'not detected; a driver did not load'
    LED_DRIVER_UNAVAILABLE = 'led_driver_unavailable', 'not detected; a driver did not load'
    FX2_FAILED = 'fx2_failed', 'failed with the camera, the one USB device it is part of'
    CAMERA_FIRMWARE_MISSING = 'camera_firmware_missing', 'firmware file missing'
    CAMERA_DRIVER_UNUSABLE = (
        'camera_driver_unusable',
        "on no driver, one not LumaViewPro's, or ours not started",
    )
    CAMERA_NOT_INITIALIZED = 'camera_not_initialized', 'not opened'
    CAMERA_SETUP_FAILED = 'camera_setup_failed', 'opened, set-up failed'
    CAMERA_BRING_UP_FAILED = 'camera_bring_up_failed', 'LumaViewPro failed bringing it up'
    SAFETY_OFF_FAILED = 'safety_off_failed', 'safety LEDs-off not confirmed'
    STREAM_NOT_STARTED = 'stream_not_started', 'stream did not start'
    SETTINGS_NOT_APPLIED = 'settings_not_applied', 'settings not applied'
    CAMERA_LEFT = 'camera_left', 'left during bring-up'
    ACCELERATION_NOT_SET = 'acceleration_not_set', 'acceleration limit not set'


# A reason's words inside a list of parts.
REASON_PHRASES: dict[str, str] = {reason: reason._phrase for reason in PartReason}


@api_fields('detail', 'expected', 'part', 'reason', 'remedy', 'up')
@dataclass(frozen=True)
class PartStatus:
    """One part of the scope at bring-up.

    Attributes:
        part: Which board.
        up: Whether a real driver came up for it.
        expected: Whether this scope's model has one. A manual scope has no
            motor board, so its absence is not a failure; until the model is
            known every part is expected.
        reason: Why it is not up; or, on a
            part that is up, a problem it met coming up (the LED board's
            safety LEDs-off not confirmed; the camera's stream that did not
            start, which a later start that succeeds does not clear, a
            setting it did not take, or its leaving, the camera keeping the
            one of these a person acts on first; the motor controller's
            stored acceleration limit not taken). None when nothing is
            wrong.
        detail: The driver's or the error's own words, for the log and the
            record, each problem's joined with "; "; empty when there are
            none.
        remedy: The action LumaViewPro can take for the part, the one the
            notice of its reason offered, taken through
            ``ScopeSession.apply_remedy`` by its ``member``: Repair for a
            camera on ``camera_driver_unusable`` where this installation can
            run it. None when there is none.
    """

    part: Board
    up: bool
    expected: bool = True
    reason: PartReason | None = None
    detail: str = ''
    remedy: Remedy | None = None

    @property
    def missing(self) -> bool:
        """Expected and not up: the part this scope should have and lacks.

        Raises:
            Nothing.
        """
        return self.expected and not self.up

    def describe(self) -> str:
        """The part's label with its reason: ``Motor Controller (port in use)``.

        Raises:
            Nothing.
        """
        label = self.part.label
        if self.reason is None:
            return label
        return f'{label} ({REASON_PHRASES[self.reason]})'


@api_fields('saved', 'setting', 'used')
@dataclass(frozen=True)
class Substitution:
    """A saved setting the camera could not take, and what bring-up used instead.

    What becomes of the saved value is the setting's own rule, not this
    record's: the record says what was saved and what ran. The binning the
    camera delivered at is stored in the saved binning's place, with the
    frame it belongs to, and so is a saved frame the camera cannot run,
    refitted as the API fits one; the saved image mode stays saved.

    Attributes:
        setting: What was substituted: ``'binning'``, ``'frame'``, or the
            settings path of a camera feature the camera lacks
            (``'camera.high_conversion_gain'``, ``'camera.line_noise_reduction'``).
        saved: The value the settings held: the binning factor, the frame's
            (width, height) in pixels, or True for a feature.
        used: The value bring-up ran with in its place: the camera's
            binning factor, the fitted frame's (width, height), or False.
    """

    setting: str
    saved: SettingValue
    used: SettingValue


@api_fields('path', 'message')
@dataclass(frozen=True)
class SettingsSetAside:
    """The user's settings file that could not be used, and why.

    The app runs on the shipped template while this stands, and refuses to
    save over the file until the person decides what happens to it.

    Attributes:
        path: The settings file (``current.json``) that was set aside,
            untouched.
        message: Why it could not be used, in words a person reads.
    """

    path: pathlib.Path
    message: str


@api_fields('parts', 'settings_set_aside', 'substitutions')
@dataclass(frozen=True)
class BringUpRecord:
    """What bring-up found, substituted and set aside.

    Attributes:
        parts: Each part bring-up looked for, and whether it came up.
        substitutions: Each saved setting the camera could not take, with
            what ran in its place; empty when none was substituted.
        settings_set_aside: The user's settings file that could not be used,
            and why; None when it was used.
    """

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
