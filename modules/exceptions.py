# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Module/API-layer exception classes (raised from modules/, caught at UI).

For driver-layer hardware exceptions (HardwareError), see drivers/exceptions.py.
"""

from typing import ClassVar


class ProtocolError(Exception):
    """Protocol file parsing, validation, or execution error."""

    pass


class ConfigError(Exception):
    """Application configuration or settings error."""

    pass


class ObjectiveUnknownError(ConfigError):
    """No one can say which objective is in the light path.

    On a turreted scope the active objective is the objective assigned to
    the slot in the light path, so it is unknown when the slot is unknown
    (the turret has not been homed or moved since it was last lost), when
    the slot has no assignment, or when its assignment names nothing in
    the objective catalogue. Raised instead of answering with a stored
    objective, because an objective that is not in the light path puts a
    wrong scale into every image it names.

    On a scope with no turret it is unknown only before any objective was
    selected. On any scope it is unknown before bring-up has said whether
    the scope has a turret, since that decides where the objective comes
    from.

    Attributes:
        reason: ``'slot_unknown'``, ``'slot_unassigned'``,
            ``'not_in_catalogue'``, ``'none_selected'`` or
            ``'turret_undecided'``.
        slot: The slot in the light path, or None when that is what is
            unknown.
    """

    _SENTENCES: ClassVar[dict[str, str]] = {
        'slot_unknown': ('the turret is in no known slot -- home the turret or move it to a slot'),
        'slot_unassigned': (
            'turret slot {slot} has no objective assigned -- assign the objective installed there'
        ),
        'not_in_catalogue': (
            'turret slot {slot} is assigned an objective that is not in the catalogue'
            ' -- assign the objective installed there'
        ),
        'none_selected': 'no objective has been selected',
        'turret_undecided': (
            'the scope has not been configured, so whether it has a turret is not known'
            ' -- run initialize() (a ScopeSession does this at bring-up)'
        ),
    }

    def __init__(self, reason: str, slot: int | None = None):
        super().__init__(
            'The objective in the light path is unknown: '
            + self._SENTENCES[reason].format(slot=slot)
        )
        self.reason = reason
        self.slot = slot


class SettingsSaveRefusedError(ConfigError):
    """A settings save was refused: writing now would destroy real data.

    Raised by ``ScopeSession.save_settings`` instead of silently skipping
    the write -- a caller that cannot tell a refusal from a success reports
    success on a write that never happened, which is how a whole session's
    changes get lost with nothing said.

    Like the hardware-command refusal, this reaches an external API caller
    that no notification path serves, so it carries no user-facing strings;
    the caller that provoked it owns the response.

    Attributes:
        reason: Machine-readable refusal code for callers that map refusals
            to responses. ``'settings_provisional'`` -- the app is running
            on the shipped template because the user's file could not be
            read, and that file stays untouched until they decide
            (``force`` does not override). ``'no_hardware'`` -- no hardware
            was connected this session, so the in-memory per-channel values
            are slider defaults, not measurements (``force=True``
            overrides).
        file: The destination whose write was refused.
    """

    def __init__(self, reason: str, file: str):
        super().__init__(f'settings save to {file} refused: {reason}')
        self.reason = reason
        self.file = file


class CaptureError(Exception):
    """Image capture, save, or processing failure.

    Attributes:
        reason: Machine-readable failure code for callers that map
            failures to responses (REST status codes, UI branches).
            Without it a caller has to regex the prose to tell a timeout
            from a failed merge from a camera that returned nothing.

    ``reason`` is required rather than defaulted. A default would let a
    new raise site stay untyped while every caller still had to guess
    which raises carry a usable code and which carry a placeholder.
    """

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


class ProtocolRunRefusedError(ProtocolError):
    """A sequenced run was refused before any state was committed.

    Raised by SequencedCaptureRunner.prepare() when a run cannot start
    (already running, files still writing, empty protocol, validation
    errors, hardware not connected). Raised THERE, it has already been
    logged and notified to the user by the runner's refusal funnel, so
    callers reconcile their own state without re-notifying.

    Raised by the protocol BUILDER (Protocol.from_config, for a z-stack
    asked for with no range), it has not been: the builder runs before
    any run exists, so no funnel has seen it and those callers own the
    telling. A headless caller has the exception itself, which is the
    whole of what it needs; a widget renders title and message, never
    the joined str(e) form this class builds for debugging.

    Attributes:
        reason: Machine-readable refusal code for callers that map
            refusals to responses (REST status codes, UI branches).
        title: The notification title already shown to the user.
        message: The notification body already shown to the user.
        holder: What holds the microscope at refusal time
            ('protocol' or 'recording' for the exclusive-activity claim
            owner; 'autofocus' for a sweep in flight), or None when the
            refusal is not holder-shaped (validation, hardware, file
            drain).
        holder_trigger: Busy-with-what: the trigger of the run that
            holds the scope -- for a file-drain refusal the just-
            finished run's, for an autofocus sweep the run that
            dispatched it. None when no run is behind the holder -- a
            recording has no trigger; its kind IS the holder.
    """

    def __init__(
        self,
        reason: str,
        title: str,
        message: str,
        holder: 'str | None' = None,
        holder_trigger: 'str | None' = None,
    ):
        super().__init__(f'{reason}: {message}')
        self.reason = reason
        self.title = title
        self.message = message
        self.holder = holder
        self.holder_trigger = holder_trigger


class RunStartError(ProtocolError):
    """A sequenced run failed after it was committed but before it ran.

    The counterpart of ProtocolRunRefusedError on the other side of the
    commit line: a refusal means nothing started and nothing needs
    unwinding, while this means the run was committed, the terminal
    callback will fire and cleanup will run. Carries the same three
    fields so both sides deliver one shape to a caller, a REST handler
    and the popup.

    Attributes:
        reason: Machine-readable cause.
        title: Short heading for the user.
        message: The sentence a user reads -- never a raw exception
            string; those belong in the log.
    """

    def __init__(self, reason: str, title: str, message: str):
        super().__init__(f'{reason}: {message}')
        self.reason = reason
        self.title = title
        self.message = message


class RecordingRefusedError(CaptureError):
    """A video recording start was refused before any state was committed.

    Raised when a recording cannot begin: by VideoRecordingEngine.start()
    when an exclusive activity -- a protocol run or another recording --
    already holds the session's activity claim or the engine is still
    draining, and by the recording controllers for the caller-shaped
    refusals they own (a previous recording still finishing, an inactive
    camera, an unknown exposure, insufficient disk). Mirrors the
    ProtocolRunRefusedError shape so callers reconcile state the same way
    in both directions.

    Attributes:
        reason: Machine-readable refusal code for callers that map
            refusals to responses (REST status codes, UI branches).
        title: Short user-facing refusal title.
        message: One-sentence user-facing refusal body.
        holder: The exclusive-activity claim owner at refusal time, or
            None when the refusal is not claim-shaped.
        holder_trigger: The holding run's run_trigger_source when the
            holder is 'protocol'; a recording holder has no trigger.
    """

    def __init__(
        self,
        reason: str,
        title: str,
        message: str,
        holder: 'str | None' = None,
        holder_trigger: 'str | None' = None,
    ):
        super().__init__(f'{reason}: {message}', reason)
        self.title = title
        self.message = message
        self.holder = holder
        self.holder_trigger = holder_trigger


class HardwareCommandRefusedError(Exception):
    """A hardware command was refused: an exclusive activity holds the executor.

    Raised by the public hardware members (LED, camera and motion commands)
    when the executor that would carry the work will not accept it -- because
    a protocol run fenced it, or because the run disabled it outright. Both
    executor states make ``put()`` return None, and the caller cannot tell
    which one applies; asking whether work is accepted covers both, while
    asking why would need a list of reasons kept in sync with the executor.

    Distinct from the run and recording refusals, which are raised when an
    ACTIVITY is refused at start and which carry the title and body already
    shown to the user. This refusal reaches an external API caller that no
    notification path serves, so it carries no user-facing strings -- the
    caller that provoked it owns the response. Without it the command would
    be dropped silently, which is how a fenced write reaches no hardware and
    reports success.

    The Session's objective writers (select, slot assign and slot clear)
    raise it too while a run holds the scope, with the same reason: the run
    stamps the active objective's scale into each capture, so a change
    mid-run is a command against the run's hardware state.

    Attributes:
        reason: Machine-readable refusal code for callers that map refusals
            to responses (REST status codes, SDK branches).
        member: The public member that was refused, for the log and message.
    """

    def __init__(self, reason: str, member: str):
        super().__init__(f'{member} refused: {reason}')
        self.reason = reason
        self.member = member


class PositionOutOfRangeError(ValueError):
    """An absolute move was commanded beyond the axis's travel.

    The driver's own response to an out-of-travel target is to clamp it
    to the nearest limit and drive there, which reports success at a
    position nobody asked for: a protocol step saved beyond this scope's
    travel images the wrong place, and nothing in the log distinguishes
    that from a step that went where it was told. Refusing by name makes
    the substitution impossible rather than silent.

    Subclasses ValueError because an out-of-travel target is the same
    kind of bad argument as a non-numeric one, and callers already
    written to catch ValueError from this call keep working.

    The message reaches the user verbatim, so it names the axis, the
    request, and the range that refused it.

    ``bound`` names WHICH limit refused, because two of them can: the
    axis's own travel, and the coarse safety ceiling that rejects a
    nonsense magnitude before any axis is consulted. Telling someone
    their entry is "outside the travel range 0.0 to 80000.0" when it was
    really refused as absurd points them at the wrong number. ``quantity``
    likewise distinguishes a position from a relative distance. One
    optional argument each rather than a second exception class: the
    refusal is the same event, and only the sentence differs.
    """

    def __init__(
        self,
        axis: str,
        position: float,
        low: float,
        high: float,
        bound: str = 'travel range',
        quantity: str = 'position',
    ):
        super().__init__(f'{axis} {quantity} {position} is outside the {bound} {low} to {high}.')
        self.axis = axis
        self.position = position
        self.low = low
        self.high = high
        self.bound = bound
        self.quantity = quantity


# The axis-state value for an axis whose home is in progress. Spelled here
# rather than imported: AxisState lives in the lumascope_api package, whose
# import pulls in modules that import this one.
_AXIS_HOMING = 'homing'


def describe_unknown_positions(axes: dict[str, str]) -> str:
    """Say which axes do not know their position, in the words a user acts on.

    One wording for every refusal and ending that names an unknown
    position -- a run's start refusal, its mid-run ending, and a refused
    move or save -- so they never describe the same state differently. A
    homing axis is named apart from a lost one because the user does
    different things about them: wait for the one, home the other.

    Args:
        axes: Axis name to state, as ``MotionAPI.axes_without_position``
            answers it. Must not be empty.

    Returns:
        str: A clause such as "Z is still homing; the X and Y positions
            are unknown", with no leading capital or closing full stop, so
            each caller ends it with the action its own situation needs.
    """

    def _names(names: list[str]) -> str:
        return names[0] if len(names) == 1 else f'{", ".join(names[:-1])} and {names[-1]}'

    homing = [axis for axis, state in axes.items() if state == _AXIS_HOMING]
    lost = [axis for axis, state in axes.items() if state != _AXIS_HOMING]
    parts = []
    if homing:
        parts.append(f'{_names(homing)} {"is" if len(homing) == 1 else "are"} still homing')
    if lost:
        parts.append(
            f'the {_names(lost)} position{"" if len(lost) == 1 else "s"} '
            f'{"is" if len(lost) == 1 else "are"} unknown'
        )
    return '; '.join(parts)


def unknown_positions_sentence(axes: dict[str, str], then: str) -> str:
    """The whole refusal a user reads: what is unknown, and what to do about it.

    Args:
        axes: Axis name to state, as ``MotionAPI.axes_without_position``
            answers it. Must not be empty.
        then: What the user does once the scope knows its position, ending
            the sentence (e.g. ``'move it'``, ``'add the step'``).

    Returns:
        str: e.g. "The X and Y positions are unknown. Home the scope, then
            move it." -- or, when every axis named is still homing, "Wait
            for the home to finish" in place of "Home the scope".
    """
    clause = describe_unknown_positions(axes)
    waiting = all(state == _AXIS_HOMING for state in axes.values())
    remedy = 'Wait for the home to finish' if waiting else 'Home the scope'
    return f'{clause[0].upper()}{clause[1:]}. {remedy}, then {then}.'


class AxisStateUnknownError(Exception):
    """An axis whose position is not known was asked to move, or to be recorded.

    Raised by the motion pre-drive gate when the target axis is UNKNOWN:
    a home failed, the board vanished mid-move, or a move stalled out.
    An absolute move against an unknown reference frame is never a valid
    request -- there is no frame for it to be absolute in -- so refusing
    it discards nothing legitimate and makes the failure loud instead of
    letting the stage travel somewhere nobody asked for. Also raised when
    a position is about to be saved (a step, a focus, a bookmark) while
    an axis does not know where it is: the cached number is the last one
    the axis reported, real-looking and no longer true.

    The recovery paths that must move a still-unknown axis (lowering Z
    for turret safety, a deliberate re-home jog) pass ``force=True``
    rather than pre-checking state, so the gate can never deadlock the
    operation that would clear the state it guards.

    The message is written for the person at the scope, because the
    GUI's background lane shows a typed error's message as the popup
    body; it names every axis in one sentence so one refusal of a
    several-axis gesture reads as one.

    Attributes:
        axes: Every refused axis, mapped to its state, in the scope's axis
            order.
        axis: The first of them, for callers that map a refusal to a
            response by a single axis (REST status codes, SDK branches).
    """

    def __init__(self, axes: dict[str, str], then: str = 'move it'):
        """Build the refusal for ``axes``.

        Args:
            axes: Axis name to state for every axis refused. Must not be
                empty.
            then: What the user does once the scope knows its position,
                ending the sentence (e.g. ``'move it'``, ``'save the
                focus'``).
        """
        super().__init__(unknown_positions_sentence(axes, then))
        self.axes = dict(axes)
        self.axis = next(iter(axes))


class MoveNotCompletedError(Exception):
    """A waited move ended without its axis arriving at the target.

    The move was driven, so this is a failure, not a refusal: the motor
    may have travelled any part of the way. A waited move that returns
    means the axis arrived; this move could not say that, so it raises
    instead of reporting a position nobody reached.

    Distinct from ``AxisStateUnknownError``, which refuses a move before
    anything is driven and offers ``force=True`` -- advice that is wrong
    for a move that already happened.

    Attributes:
        axis: The axis whose move did not complete.
        reason: ``'faulted'`` -- the motion monitor gave the axis up during
            the wait (a stall, or the board lost); ``'timed_out'`` -- the
            wait's bound ran out before the axis arrived. Both leave the
            axis UNKNOWN. ``'stopped'`` -- a stop was issued while it
            moved; the axis is where the stop left it, which its position
            reports, and a turret is in no known slot.
    """

    _SENTENCES: ClassVar[dict[str, str]] = {
        'faulted': (
            'it stalled or the board was lost during the move. The {axis} position '
            'is now unknown -- home the scope before moving it again.'
        ),
        'timed_out': (
            'it did not arrive within the motion time limit. The {axis} position '
            'is now unknown -- home the scope before moving it again.'
        ),
        'stopped': 'the motors were stopped before it arrived.',
    }

    def __init__(self, axis: str, reason: str):
        super().__init__(
            f'The {axis} move did not complete: ' + self._SENTENCES[reason].format(axis=axis)
        )
        self.axis = axis
        self.reason = reason


class AutofocusAborted(Exception):  # noqa: N818 -- cancellation/abort signal, not an error; non-Error suffix is intentional
    """Autofocus run aborted by caller (e.g. user cancelled, protocol
    aborted, or app teardown)."""

    pass


class CameraSettingRejected(Exception):  # noqa: N818 -- named for the event it signals; one type covers the defect class
    """The camera driver rejected a state-changing setting apply.

    Raised by ImagingAPI setters (frame size, binning, pixel format) when
    a LIVE driver refuses the apply -- distinct from the camera-absent
    no-op, which stays a quiet sentinel per the missing-hardware contract.
    Success is observed by receiving the applied/delivered value, failure
    by this raise, so a caller cannot record a rejected apply as applied
    by forgetting to check a return code. The rejection has already been
    logged and notified to the user when this is raised.

    Attributes:
        setting: Machine-readable setting name (e.g. 'frame_size').
        requested: The value the caller asked for.
    """

    def __init__(self, setting: str, requested):
        super().__init__(f'{setting}: driver rejected {requested!r}')
        self.setting = setting
        self.requested = requested


class FrameDepthError(Exception):
    """A frame carries a payload value above its declared significant-bits depth.

    The downconvert to 8-bit scales against ``significant_bits`` -- the meaningful
    payload range the frame was captured under. A pixel larger than that range is
    a depth-contract violation: the significant-bits value is wrong for the data.
    Raised explicitly so the failure is loud and typed regardless of the
    downconvert arithmetic underneath -- a value-indexed LUT happens to raise
    IndexError today, but a scale-and-clip converter would silently map the
    over-range frame to white instead.

    Attributes:
        value: The offending pixel value.
        significant_bits: The declared depth it exceeded.
    """

    def __init__(self, value: int, significant_bits: int):
        super().__init__(
            f'frame value {value} exceeds the declared {significant_bits}-bit depth '
            f'(max {(1 << significant_bits) - 1}); the significant-bits contract is wrong'
        )
        self.value = value
        self.significant_bits = significant_bits
