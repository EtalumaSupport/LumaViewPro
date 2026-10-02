# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Module/API-layer exception classes (raised from modules/, caught at UI).

For driver-layer hardware exceptions (HardwareError), see drivers/exceptions.py.
"""

import pathlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Remedy:
    """The one action that answers a refusal, offered to whoever was refused.

    Data, not a callable: the layer that refuses cannot reach the member that
    answers (a run's engine refuses; the Session recovers), and a REST caller
    receives the remedy as a name it can send back. The Session is the one
    place a name becomes an action (``ScopeSession.apply_remedy``). The offer's
    prompt is the refusal's own title and message, so its words live in one
    place.

    Attributes:
        member: The Session member that answers the refusal.
        confirm_text: The words that take the remedy, naming what it costs.
        cancel_text: The words that decline it and leave things as they are.
    """

    member: str
    confirm_text: str
    cancel_text: str


class Refusal:
    """A request the scope declined: nothing broke, and the person who asked can act on it.

    Mixed into an exception whose message is written for that person. The
    background executor shows a refusal as a warning under ``title``, in the
    exception's own words, and logs one line for it with no traceback: an
    ERROR and a traceback say something went wrong, and a refusal is a
    designed outcome. Unmarked, a refusal raised inside a background task
    read as a crash -- "Background operation failed" over its message, and a
    traceback in the errors log.

    Attributes:
        title: The heading the person reads above the message.
        remedy: The action that answers this refusal, when one exists; the
            reporter then shows the refusal as an offer to take it. Set per
            instance: one reason of a refusal type can have a remedy that its
            others do not.
    """

    title: str
    remedy: Remedy | None = None


class RemedyUnknownError(Refusal, ValueError):
    """A remedy was asked for by a name the Session does not offer.

    A remedy names a Session member, and a name arrives as data -- from a
    REST caller, or from a refusal built by another layer -- so only the
    members the Session lists are reachable through it.

    Attributes:
        reason: ``'remedy_unknown'``.
        member: The name that was asked for.
    """

    title = 'Unknown Remedy'

    def __init__(self, member: str, offered: Iterable[str]):
        super().__init__(
            f"'{member}' is not an action the microscope offers as a remedy. "
            f'The remedies are: {", ".join(sorted(offered))}.'
        )
        self.reason = 'remedy_unknown'
        self.member = member


class Quiet:
    """An outcome that is recorded and never shown: nothing failed and nothing was declined.

    Mixed into an exception that tells a caller something it may need to act
    on -- a script learning its handle is stale -- but that the person at the
    instrument has no reason to see. It is logged at INFO in its own words and
    never becomes a notification. Whether an outcome is quiet belongs to its
    type, never to the code that raises or catches it, so every client reads
    the same answer. A message on a quiet exception is written for the log.
    """


class Notice:
    """An outcome that tells the person something: nothing failed and nothing was declined.

    Mixed into an exception whose message is written for the person -- a
    capture saved without its position, a long operation starting -- and
    reported, never raised: nothing waits on a notice, so the reporter is the
    whole of its flight. It is shown as a notice under ``title`` in its own
    words, and logged at NOTICE. Its kind belongs to its type, as a
    refusal's does, so every client that hears it reads the same answer.

    Attributes:
        title: The heading the person reads above the message.
        reason: The machine-readable code a client branches on, since two
            notices can share a title.
    """

    title: str
    reason: str
    remedy: Remedy | None = None


class ExposureAtMaximumNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """Auto-gain reached its exposure ceiling and the scene was still too dark.

    The setting keeps the ceiling; the person can add light or raise it.
    """

    title = 'Exposure at the maximum'
    reason = 'exposure_at_maximum'

    def __init__(self, ceiling_ms: float):
        super().__init__(
            f'Auto-exposure reached the {ceiling_ms:g} ms ceiling for this '
            'channel and the scene was still too dark. Add light or raise the '
            'auto-exposure ceiling in Advanced Settings.'
        )


class ExposureAtMinimumNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """Auto-gain settled at or below the channel's usable exposure floor.

    The setting keeps the floor, since it cannot hold the raw value.
    """

    title = 'Exposure at the minimum'
    reason = 'exposure_at_minimum'

    def __init__(self, exposure_ms: float, floor_ms: float):
        super().__init__(
            f'Auto-exposure settled at {exposure_ms:g} ms, at or below the '
            f'{floor_ms:g} ms usable floor for this channel; the setting keeps '
            'the floor. The scene is too bright: reduce the light.'
        )


class CapturePositionNotRecordedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A still was saved, without its well or position, because X or Y is unknown."""

    title = 'Position Not Recorded'
    reason = 'position_not_recorded'

    def __init__(self):
        super().__init__(
            'The stage position is unknown, so this image was saved without a well '
            'or position. Home the scope to record them.'
        )


class RecordingPositionNotRecordedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A frames recording started that cannot yet record where its frames are.

    An unknown axis is recorded from the moment it becomes known; with no
    labware or stage offset there is no plate frame to record X and Y in.

    Attributes:
        unknown_axes: The axes whose position is unknown at the start.
        has_plate: Whether a plate frame exists to record X and Y in.
    """

    title = 'Position Not Recorded'
    reason = 'position_not_recorded'

    def __init__(self, *, unknown_axes: list[str], has_plate: bool):
        sentences = []
        if unknown_axes:
            sentences.append(
                f'The scope does not know its {", ".join(unknown_axes)} position, so '
                'frames record it only once it is known. Home the scope to record it.'
            )
        if not has_plate:
            sentences.append(
                'No labware or stage offset is selected, so frames record no plate position.'
            )
        super().__init__(' '.join(sentences))
        self.unknown_axes = list(unknown_axes)
        self.has_plate = has_plate


class DuplicateCaptureFilenamesNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A loaded protocol has steps that would save to the same file name.

    The file still loads, so the steps can be renamed in the app; the run is
    refused at start until each step's file name is unique.
    """

    title = 'Duplicate filenames in protocol'
    reason = 'duplicate_capture_filenames'

    def __init__(self, *, colliding_steps: int, shared_names: int):
        super().__init__(
            f'Protocol has {colliding_steps} steps sharing {shared_names} capture '
            'filenames. The protocol can be edited, but running it will be refused '
            'until each step produces a unique filename -- rename the colliding '
            'steps first.'
        )


class SlowFileWritesNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A run spent long enough waiting on the save disk to say so at its end."""

    title = 'Very Slow File Writes'
    reason = 'slow_file_writes'

    def __init__(self):
        super().__init__(
            'Very slow writes are occurring on the save disk. '
            'Please confirm your computer and storage are OK.'
        )


class SingleScanNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A timed run will perform one scan, because of its period and duration."""

    title = 'Single Scan'
    reason = 'single_scan'

    def __init__(self, *, period, duration):
        if period.total_seconds() == 0:
            because = 'the capture period is 0'
        else:
            because = f'the duration ({duration}) is shorter than the capture period ({period})'
        super().__init__(f'This run performs a single scan because {because}.')


class HyperstacksSavingNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A run's hyperstacks are being built, which can take minutes."""

    title = 'Saving Hyperstacks'
    reason = 'hyperstacks_saving'

    def __init__(self):
        super().__init__(
            'Building hyperstacks from the run. This can take several minutes; '
            'a message will confirm completion.'
        )


class HyperstacksSavedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A run's hyperstacks were built: how many and where, or what was degraded."""

    title = 'Hyperstacks Saved'
    reason = 'hyperstacks_saved'

    def __init__(self, result: dict):
        if result.get('degraded'):
            message = result['message']
        else:
            message = (
                f'{result["new_count"]} hyperstack(s) saved to {result["output_root"]}.'
                f'{result["accounting_note"]}'
            )
        super().__init__(message)


class NoHardwareDetectedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """Nothing came up at bring-up: no LED board, no motor board, no camera.

    One notice for the whole scope, in place of one per part: the person
    with no instrument attached needs to be told once, not three times.
    """

    title = 'No hardware detected'
    reason = 'no_hardware'

    def __init__(self):
        super().__init__(
            'No microscope hardware was detected. You can continue in software-only '
            'mode (live view + protocol design will work; capture will not). To '
            'connect hardware, power on the scope and reconnect the USB cable, then '
            'restart LumaViewPro.'
        )


class BinningSubstitutedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """The saved binning is one this camera does not offer; bring-up ran at the camera's.

    The camera's binning is stored in the saved value's place: the binning
    is half of the frame's geometry, and the store holds the pair the
    camera delivered.
    """

    title = 'Saved binning not supported'
    reason = 'binning_substituted'

    def __init__(self, saved: int, used: int):
        super().__init__(
            f'The saved {saved}x{saved} binning is not supported by this camera; it '
            f'runs at {used}x{used}, which is now the saved binning.'
        )


class ImageModeSubstitutedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """The saved image mode needs a pixel depth this camera lacks; bring-up used one it has.

    Attributes:
        saved: The saved mode's label.
        used: The label of the mode bring-up started in.
    """

    title = 'Image mode not supported'
    reason = 'image_mode_substituted'

    def __init__(self, saved: str, used: str):
        super().__init__(
            f'This camera does not support the saved {saved} image mode; it runs in {used} instead.'
        )
        self.saved = saved
        self.used = used


class ProtocolError(Exception):
    """Protocol file parsing, validation, or execution error."""

    pass


class ProtocolNotSavedError(ProtocolError):
    """A protocol could not be written to its file.

    Raised by ``Protocol.to_file``, chained from the ``OSError`` of the write,
    so the words name the file and the operating system's own reason -- a
    missing or read-only folder, a full disk, a file another program holds --
    rather than guessing one. The write goes to a file beside the target
    first, so a file already there is unchanged.

    Attributes:
        file: The path the protocol was to be written to.
    """

    title = 'Protocol Not Saved'

    def __init__(self, file, cause: OSError):
        reason = cause.strerror or type(cause).__name__
        super().__init__(
            f'The protocol was not saved to {file} ({reason}). A file already there is unchanged.'
        )
        self.file = file


class ConfigError(Exception):
    """Application configuration or settings error."""

    pass


class InstallationFileError(Exception):
    """A file the installation ships is missing, unreadable, or not the shape its reader needs.

    The installation is at fault, not the user's settings, so this is
    deliberately not a ``ConfigError``: a host that answers a ``ConfigError``
    by falling back to the shipped settings template would replace good
    settings and still fail on the same file. The reader raises it and logs
    nothing; whoever catches it logs it once.

    Attributes:
        file_path: The file that could not be used.
    """

    title = 'Installation File Unusable'

    def __init__(self, file_path, problem: str):
        file_path = pathlib.Path(file_path)
        super().__init__(
            f'{file_path.name} in {file_path.parent} {problem}; reinstall LumaViewPro or '
            'restore the file'
        )
        self.file_path = file_path


class BringUpError(Exception):
    """A part of the scope did not come up as it should have; the rest of the scope runs.

    Reported, never raised: bring-up goes on without the part, and the
    person is told once what is missing and what to do about it. The record
    of the bring-up (``ScopeSession.bring_up_record``) keeps the fact for a
    client that asks later.

    Attributes:
        reason: The machine-readable code a client branches on.
    """

    title = 'Hardware Unavailable'

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


class CameraNotAvailableError(BringUpError):
    """The camera did not come up; the heading and the advice follow the cause.

    Attributes:
        reason: ``'camera_in_use'`` (another application holds it),
            ``'camera_port_in_use'``, ``'camera_not_detected'`` or
            ``'camera_not_initialized'`` (anything else).
    """

    _WORDS: ClassVar[dict[str, tuple[str, str]]] = {
        'camera_in_use': (
            'Camera in use',
            'Camera appears to be open in another application (Pylon Viewer, another '
            'LVP instance, etc.). Close it and restart LVP.',
        ),
        'camera_port_in_use': (
            'Camera port in use',
            'Camera port is in use by another program. Close the other program and restart LVP.',
        ),
        'camera_not_detected': (
            'Camera not detected',
            'Camera not found. Check USB cable and power.',
        ),
        'camera_not_initialized': (
            'Camera not initialized',
            'Could not connect to the camera. Check USB cable, power, and close other '
            'programs that may hold the camera.',
        ),
    }

    def __init__(self, reason: str):
        self.title, message = self._WORDS[reason]
        super().__init__(message, reason)


class LedBoardUnavailableError(BringUpError):
    """The LED board did not come up on a scope whose other parts did.

    Said once at bring-up rather than once per failed illumination command:
    without it the first symptom is a sample under a dark objective and
    controls that appear to do nothing. The advice follows the cause.

    Attributes:
        reason: The registry's fallback cause: ``'not_detected'``,
            ``'port_in_use'``, ``'not_responding'``, ``'connect_failed'`` or
            ``'no_driver'``.
    """

    title = 'LED Board Unavailable'

    _ADVICE: ClassVar[dict[str, str]] = {
        'not_detected': (
            'The LED control board was not found on USB, so illumination is not '
            'available this session. Check the USB cable and 24V power, then restart '
            'LumaViewPro.'
        ),
        'port_in_use': (
            'The LED control board was found but its port could not be opened, so '
            'illumination is not available this session. Close other programs holding '
            'the port (a serial monitor, Thonny), then restart LumaViewPro.'
        ),
        'not_responding': (
            'The LED control board did not respond, so illumination is not available '
            'this session. Power-cycle the microscope and restart LumaViewPro to '
            'restore illumination.'
        ),
        'connect_failed': (
            'Could not connect to the LED control board, so illumination is not '
            'available this session. Check the USB cable and 24V power, then restart '
            'LumaViewPro.'
        ),
        'no_driver': (
            'No LED control board driver is installed, so illumination is not available '
            'this session. Reinstall LumaViewPro.'
        ),
    }

    def __init__(self, reason: str):
        super().__init__(f'{self._ADVICE[reason]} The rest of the microscope is working.', reason)


class LedSafetyOffNotTakenError(BringUpError):
    """The LED board connected but did not confirm the LEDs-off sent on connect.

    Sample safety: firmware before the confirmation existed can leave channels
    on, photobleaching whatever is on the stage.

    Attributes:
        reason: ``'safety_off_failed'``.
    """

    title = 'LED Safety Off Not Confirmed'

    def __init__(self, detail: str):
        super().__init__(
            'The LED board connected but the safety LEDS_OFF command did not complete '
            f'({detail}). If the LEDs are stuck on, turn off illumination manually '
            'before placing a sample.',
            'safety_off_failed',
        )


class PartialHardwareError(BringUpError):
    """Parts this scope's model has did not come up; the rest of the scope runs.

    Attributes:
        reason: ``'partial_hardware'``.
        missing: Each missing part with its cause, as ``PartStatus.describe``
            writes it.
    """

    title = 'Partial Hardware Detected'

    def __init__(self, missing: Iterable[str]):
        self.missing = tuple(missing)
        super().__init__(
            f'Not connected: {", ".join(self.missing)}. Some features will be unavailable.',
            'partial_hardware',
        )


class ObjectiveUnknownError(Refusal, ConfigError):
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

    title = 'Objective Unknown'

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


class SettingsSaveRefusedError(Refusal, ConfigError):
    """A settings save was refused: writing now would destroy real data.

    Raised by ``ScopeSession.save_settings`` instead of silently skipping
    the write -- a caller that cannot tell a refusal from a success reports
    success on a write that never happened, which is how a whole session's
    changes get lost with nothing said.

    A refusal like any other: its message is the sentence a person reads
    and ``title`` its heading, so whichever caller ends up telling someone
    tells them the same thing.

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

    title = 'Settings Not Saved'

    _SENTENCES: ClassVar[dict[str, str]] = {
        'settings_provisional': (
            'The settings were not saved to {file}: LumaViewPro is running on its '
            'default settings because that file could not be read, and it is left '
            'as it is until you decide what to do with it.'
        ),
        'no_hardware': (
            'The settings were not saved to {file}: no microscope was connected '
            'this session, so the per-channel values are defaults, not values '
            'measured on this scope.'
        ),
    }

    def __init__(self, reason: str, file: str):
        super().__init__(self._SENTENCES[reason].format(file=file))
        self.reason = reason
        self.file = file


class SettingsFileNotReplacedError(ConfigError):
    """The unreadable settings file could not be moved aside after the user chose to start over.

    Raised by the retire, chained from the ``OSError`` the rename raised, so
    the words name the file and the operating system's own reason rather
    than guessing one: a lock by another program is the common cause on
    Windows, but a permissions or disk fault reads the same to a guess. The
    settings stay provisional, so nothing is lost and the question can be
    answered again.

    Attributes:
        file: The settings file that is still in place.
    """

    title = 'Settings File Not Replaced'

    def __init__(self, file: str, cause: OSError):
        reason = cause.strerror or type(cause).__name__
        super().__init__(
            f'{file} could not be moved aside ({reason}). Close any program that has '
            'it open, then choose again.'
        )
        self.file = file


class ScopeModelUnknownError(Refusal, ValueError):
    """A scope model was selected that this release's catalogue does not list.

    The selection is the whole identity of a scope that cannot report its
    own model, so an unlisted one would start the scope with no layers, no
    optics and no scale. Refused before it is saved, rather than saved and
    found at the next start.

    Subclasses ValueError because it is a bad argument, like an unknown
    axis name. The message reaches the person verbatim and names the
    models the catalogue does list.

    Attributes:
        model: The model that was refused.
    """

    title = 'Unknown Scope Model'

    def __init__(self, model: str, known: Iterable[str]):
        super().__init__(
            f'{model!r} is not a scope model this version of LumaViewPro knows. '
            f'Choose one of: {", ".join(sorted(known))}.'
        )
        self.model = model


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

    ``title`` is the heading the reporter shows above the message, which is
    written for the person. It is the type's, so every raise site reads the
    same; a subclass with its own heading sets its own.
    """

    title = 'Capture Failed'

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


class ProtocolRunRefusedError(Refusal, ProtocolError):
    """A sequenced run, or the protocol it would run, was refused before any state was committed.

    Raised by SequencedCaptureRunner.prepare() when a run cannot start
    (already running, files still writing, empty protocol, validation
    errors, hardware not connected), by the protocols API when a protocol
    or a step names something this scope cannot do, and by the protocol
    builder for a z-stack asked for with no range. Each of those reports it
    through the one reporter as it raises, so it has been logged and shown
    once already; any later report of the same exception is a no-op, and a
    caller reconciles its own state without telling anyone again.

    Its message is the sentence a person reads, so a headless caller that
    prints it prints what the GUI shows.

    Attributes:
        reason: Machine-readable refusal code for callers that map
            refusals to responses (REST status codes, UI branches).
        title: The heading shown above the sentence.
        message: The sentence; the same text as the exception's message.
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
        remedy: The action that answers this refusal, or None. A stalled
            file writer has one (recover it); a writer still making
            progress does not, since its files will land.
    """

    def __init__(
        self,
        reason: str,
        title: str,
        message: str,
        holder: 'str | None' = None,
        holder_trigger: 'str | None' = None,
        remedy: Remedy | None = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.title = title
        self.message = message
        self.holder = holder
        self.holder_trigger = holder_trigger
        self.remedy = remedy


class RunCheckFailedError(ProtocolError):
    """A run could not be checked before it started: a check itself crashed.

    Raised by SequencedCaptureRunner.prepare() when validating the protocol,
    or reading whether the hardware is connected, raised instead of
    answering. Not a refusal: nothing was declined -- the question could not
    be asked, and the crash that stopped it is chained as ``__cause__`` so
    its traceback is logged with this. Nothing is committed and nothing
    needs unwinding, as for a refusal. Not reported where it is raised: the
    caller that asked for the run reports it where its flight ends.

    Attributes:
        reason: Machine-readable cause ('validation_crashed',
            'hardware_state_unknown').
        title: Short heading for the user.
        message: The sentence a user reads; the same text as the
            exception's message.
    """

    def __init__(self, reason: str, title: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.title = title
        self.message = message


class RunAlreadyEndedError(Quiet, ProtocolError):
    """A stop named a run that has ended, and no run is live.

    Not a refusal: nothing was refused -- the run ended on its own, and a
    Stop that arrives after that has nothing to act on. Raised so a script
    learns its handle is stale; logged, never notified, because the person
    at the instrument pressed Stop on a run that has already stopped.
    """


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


class RecordingRefusedError(Refusal, CaptureError):
    """A video recording start was refused before any state was committed.

    Raised when a recording cannot begin: by VideoRecordingEngine.start()
    when an exclusive activity -- a protocol run or another recording --
    already holds the session's activity claim or the engine is still
    draining, and by the recording controllers for the caller-shaped
    refusals they own (a previous recording still finishing, an inactive
    camera, an unknown exposure, insufficient disk). Mirrors the
    ProtocolRunRefusedError shape so callers reconcile state the same way
    in both directions. Nothing reports it as it is raised: the caller that
    asked for the recording reports it where its flight ends.

    Attributes:
        reason: Machine-readable refusal code for callers that map
            refusals to responses (REST status codes, UI branches).
        title: Short user-facing refusal title.
        message: One-sentence user-facing refusal body; the same text as
            the exception's message.
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
        super().__init__(message, reason)
        self.title = title
        self.message = message
        self.holder = holder
        self.holder_trigger = holder_trigger


class HyperstackRefusedError(CaptureError):
    """The hyperstack builder declined to build a recording's frames into one file.

    Raised by the manual recording's finish when the builder answers
    status=False: the frames are on disk as recorded, and no file exists
    to announce. The builder's own sentence rides in ``message`` so the
    notification the user sees says why, in the builder's words, rather
    than telling them to read a log.

    Attributes:
        message: The builder's one-paragraph reason, user-facing.
    """

    title = 'Hyperstack Not Built'

    def __init__(self, message: str):
        super().__init__(message, 'hyperstack_refused')
        self.message = message


class PostProcessingRefusedError(Refusal, CaptureError):
    """A folder cannot yield the post-processed output asked of it.

    Nothing broke: the folder holds no images, no groups this operation can
    combine, only derived outputs, only groups whose outputs would share a
    name, source images in a format the operation cannot re-read, or
    protocol data that could not be loaded. The message says which, in words
    written for the person, and what to do.

    Attributes:
        operation: The operation's name as a person reads it ("Stitch").
        reason: Machine-readable refusal code.
    """

    # Refused for lack of Z-stack data, a z-projection names the thing the
    # folder is missing and where such a folder lives, rather than leaving
    # the person to guess which folder would have worked.
    _NO_ZSTACK_ADVICE = (
        ' Pick a folder that contains a Z-stack run -- look under '
        "'Manual/Z-Stacks/<timestamp>/' for a manual Z-stack, or a "
        "'ProtocolData/<timestamp>/' folder whose protocol included Z-stack steps."
    )

    def __init__(self, *, operation: str, reason: str, message: str):
        zstack_missing = operation == 'Z-Projection' and reason == 'no_data'
        if zstack_missing:
            message = f'{message}{self._NO_ZSTACK_ADVICE}'
        super().__init__(message, reason)
        self.operation = operation
        self.title = 'No Z-Stack Data Found' if zstack_missing else f'{operation} Not Possible'


class PostProcessingFailedError(CaptureError):
    """A post-processing build did not produce everything it was asked for.

    Raised whether nothing or only part was produced: a group that failed,
    a group refused because its output would share a name with another's,
    or frames a video could not add. What WAS produced rides along, so a
    caller that keeps artifacts keeps them, and no caller can read an
    incomplete build as a complete one.

    Attributes:
        operation: The operation's name as a person reads it ("Stitch").
        produced_paths: Every artifact that was written.
        output_root: The folder the artifacts went under, or None.
        errors: Every failed group's error, in order.
    """

    def __init__(
        self,
        *,
        operation: str,
        missing: str,
        produced_paths: Iterable[str] = (),
        output_root: 'str | None' = None,
        errors: Iterable[str] = (),
    ):
        produced_paths = tuple(produced_paths)
        if produced_paths:
            saved = f'{len(produced_paths)} output(s) were saved to {output_root}.'
        else:
            saved = 'Nothing was saved.'
        super().__init__(
            f'{missing} {saved} Check the log for the cause.',
            'post_processing_incomplete' if produced_paths else 'post_processing_failed',
        )
        self.operation = operation
        self.produced_paths = produced_paths
        self.output_root = output_root
        self.errors = tuple(errors)
        self.title = f'{operation} Incomplete' if produced_paths else f'{operation} Failed'


class RecordingStoppedError(CaptureError):
    """A manual recording was stopped before anyone asked it to stop.

    Built by the recording's own watch when the camera stops delivering
    frames or the free disk falls below the floor, and reported where it is
    detected: no caller is waiting on a recording that is running. The
    frames written so far are on disk.

    Attributes:
        reason: ``camera_disconnected``, ``camera_stalled`` or ``disk_floor``.
    """

    _TITLES: ClassVar[dict[str, str]] = {
        'camera_disconnected': 'Recording Stopped',
        'camera_stalled': 'Recording Stopped',
        'disk_floor': 'Recording Stopped -- Disk Almost Full',
    }
    _WORDS: ClassVar[dict[str, str]] = {
        'camera_disconnected': (
            'The camera stopped delivering frames, so the recording was stopped. '
            'Frames captured so far are saved; check the camera connection before '
            'recording again.'
        ),
        'disk_floor': (
            'Free disk space fell below the safety floor, so the recording was '
            'stopped early. Frames captured so far are saved; free up space before '
            'recording again.'
        ),
    }
    _WORDS['camera_stalled'] = _WORDS['camera_disconnected']

    def __init__(self, reason: str):
        super().__init__(self._WORDS[reason], reason)
        self.title = self._TITLES[reason]


class RecordingFinalizeError(CaptureError):
    """A recording finished but its output could not be fully assembled.

    Raised around what failed in the finish of a manual recording or of a
    protocol video step, and chained from it. The frames already written
    are on disk.

    Attributes:
        protocol_step: True for a protocol video step, False for a manual
            recording; each is named in its own words.
    """

    def __init__(self, *, protocol_step: bool):
        if protocol_step:
            message = (
                'A video step finished but its output could not be fully assembled. '
                'Frames already written are on disk; check the log.'
            )
            reason = 'video_step_finalize_failed'
            self.title = 'Video Finalize Failed'
        else:
            message = (
                'The recording finished but its output could not be fully assembled. '
                'Frames already written are on disk; check the log.'
            )
            reason = 'recording_finalize_failed'
            self.title = 'Recording Finalize Failed'
        super().__init__(message, reason)
        self.protocol_step = protocol_step


class VideoFramesDroppedError(CaptureError):
    """Frames a recording selected could not be written, so its video is short.

    Each dropped frame is a write that failed; the recording's manifest
    carries the counts too.

    Attributes:
        dropped: Frames that could not be written.
        selected: Frames the recording selected.
        protocol_step: True for a protocol video step, False for a manual
            recording.
    """

    title = 'Video Frames Dropped'

    def __init__(self, dropped: int, selected: int, *, protocol_step: bool):
        what = 'in a video step ' if protocol_step else ''
        short = (
            'that video is shorter than its'
            if protocol_step
            else 'the saved video is shorter than the'
        )
        super().__init__(
            f'{dropped} of {selected} frame(s) {what}could not be written, so {short} '
            'recording. Check the log for the cause.',
            'video_frames_dropped',
        )
        self.dropped = dropped
        self.selected = selected
        self.protocol_step = protocol_step


class VideoWriterFailedError(CaptureError):
    """The recording's video writer stopped working, so the recording ended.

    Chained from what escaped the writer. The frames already written are on
    disk. On a protocol video step the run is stopped too, and the run's
    ending carries these same words, so a caller reading the run's outcome
    reads what the person was shown.

    Attributes:
        protocol_step: True for a protocol video step, False for a manual
            recording; each is named in its own words.
    """

    title = 'Recording Failed'

    def __init__(self, *, protocol_step: bool):
        stopped = (
            ', so the recording and the run were stopped'
            if protocol_step
            else ' and the recording was aborted'
        )
        super().__init__(
            f'The video writer stopped working{stopped}. Frames already written '
            'are on disk; check the log for the cause.',
            'video_writer_died',
        )
        self.protocol_step = protocol_step


class RecordingDetailsNotSavedError(CaptureError):
    """A recording's details file could not be written; its frames are whole.

    Chained from the write's error. The details file is the only record of
    the recording's channel color and measured rate, so a video built from
    these frames later falls back to grayscale and a default rate.
    """

    title = 'Recording details not saved'

    def __init__(self):
        super().__init__(
            'The video frames are safe on disk, but the recording details file '
            'could not be written. Videos built from this recording may be '
            'grayscale and use a default frame rate; check disk space and the log.',
            'recording_details_not_saved',
        )


class ImageSaveError(CaptureError):
    """An image could not be written to disk.

    Raised by ``save_image`` from the ``OSError`` of the write -- a missing
    or read-only folder, a full disk, a denied permission -- so the words
    point at the disk. A failure before the write (encoding, metadata)
    is not this and propagates as itself.

    Attributes:
        file_loc: The path the image was to be written to.
    """

    title = 'Image Save Failed'

    def __init__(self, file_loc):
        super().__init__(
            f'Failed to save image to {file_loc}. Check disk space and permissions.',
            'image_save_failed',
        )
        self.file_loc = file_loc


class RunFilesNotWrittenError(CaptureError):
    """A build that reads a run's images back found them not all written.

    Raised by the wait a post-run build makes on its run's writes -- the
    composite merge and the hyperstack build -- so neither builds from a
    folder that is still filling or that lost images: the artifact would be
    silently incomplete. Nothing is built.

    Reasons:
        ``write_batch_timeout``: the run's writes did not finish within the
            build's bound.
        ``write_batch_abandoned``: some of the run's writes never ran -- the
            file writer was recovered, or the app shut down, while they were
            outstanding.
        ``write_batch_not_taken``: some of the run's images never reached
            the file writer -- it was stuck, or had stopped taking work,
            when they were handed over.
        ``write_batch_save_failed``: saving some of the run's images failed
            on disk.
        ``write_batch_disk_full``: some of the run's images were refused
            because the save drive was nearly full.
        ``write_batch_video_unfinished``: a video step's file did not
            finish.
    """

    title = 'Run Images Not Written'

    def __init__(self, reason: str, *, bound_s: float | None = None):
        if reason == 'write_batch_timeout':
            message = (
                f"The run's images did not finish writing within {bound_s:.0f} s, "
                'so nothing was built from them.'
            )
        elif reason == 'write_batch_abandoned':
            message = (
                "Some of the run's images were never written -- the file writer "
                'was recovered or the app shut down while they were waiting -- '
                'so nothing was built from the incomplete folder.'
            )
        elif reason == 'write_batch_not_taken':
            message = (
                "Some of the run's images never reached the file writer -- it was "
                'stuck, or had stopped taking work, when they were handed over -- '
                'so nothing was built from the incomplete folder.'
            )
        elif reason == 'write_batch_save_failed':
            message = (
                "Some of the run's images failed to save to disk, so nothing "
                'was built from the incomplete folder. Check that the save '
                'drive is connected and has space.'
            )
        elif reason == 'write_batch_disk_full':
            message = (
                "Some of the run's images were not saved because the save drive "
                'was nearly full, so nothing was built from the incomplete '
                'folder. Free space on the drive and run again.'
            )
        elif reason == 'write_batch_video_unfinished':
            message = (
                "A video step's file did not finish, so nothing was built from "
                'the incomplete folder. Check the log for why the video stopped.'
            )
        else:
            raise ValueError(f'unknown reason {reason!r}')
        super().__init__(message, reason)


class RunImagesNotSavedError(CaptureError):
    """Images a run captured are not on disk.

    Reported once, when the run's last write lands -- the one moment the
    count is known, since a run's images keep landing after it ends.

    Attributes:
        not_written: Images the run captured that are not on disk.
        written: Images that landed.
    """

    title = 'Run Images Not Saved'

    def __init__(self, *, written: int, not_written: int, reason: str):
        causes = {
            'write_batch_abandoned': 'the file writer was recovered or the app shut down',
            'write_batch_not_taken': 'the file writer was stuck',
            'write_batch_save_failed': 'saving failed on disk',
            'write_batch_disk_full': 'the save drive was nearly full',
            'write_batch_video_unfinished': "a video step's file did not finish",
        }
        super().__init__(
            f'{not_written} of the images this run captured '
            f'{"is" if not_written == 1 else "are"} not on disk ({causes[reason]}); '
            f'{written} were saved. Check the log for each one.',
            reason,
        )
        self.written = written
        self.not_written = not_written


class RunCleanupFailedError(CaptureError):
    """Steps that put the scope back after a run did not finish.

    The run's own ending stands; what failed is the restore after it, so
    the LEDs, the camera's gain and exposure or the stage may not be where
    the person expects. The run's outcome names the steps too.

    Attributes:
        steps: Each failed step's name, in the order they failed.
    """

    title = 'Protocol cleanup issues'

    def __init__(self, failures: list[tuple[str, str]]):
        lines = '\n'.join(f'  - {step}: {detail}' for step, detail in failures)
        # "ended", not "completed": a stopped or failed run is cleaned up too.
        super().__init__(
            f'Protocol ended but {len(failures)} cleanup step(s) failed:\n{lines}\n'
            'Check LED state, camera settings, and stage position.',
            'cleanup_failed',
        )
        self.steps = [step for step, _ in failures]


class RecordIncompleteError(CaptureError):
    """Captures a run made are missing from its execution record.

    Post-processing reads the record to find a run's images, so an image
    with no row is skipped by stitching and video builds.

    Attributes:
        missing: Captures with no row.
        attempted: Captures the run attempted.
    """

    title = 'Protocol Record Incomplete'

    def __init__(self, *, missing: int, attempted: int, record_name: str):
        super().__init__(
            f'{missing} of {attempted} captures were not written to the protocol '
            f'record ({record_name}). Those images, if saved, will be missing from '
            'stitching and video builds. Check the log for the cause.',
            'record_incomplete',
        )
        self.missing = missing
        self.attempted = attempted


class DiskSpaceCriticalError(CaptureError):
    """An image was not saved because the save drive is nearly full.

    Raised by the save that refused it, after the run has been stopped and
    the person told; it is what counts that image as not written.

    Attributes:
        free_mb: Space left on the drive, in MB.
    """

    title = 'Disk Space Critical'

    def __init__(self, free_mb: float):
        super().__init__(
            f'Only {free_mb:.0f} MB free on the save drive; the image was not saved.',
            'disk_space_critical',
        )
        self.free_mb = free_mb


class RunIncompleteError(CaptureError):
    """A run reached its end without every capture it was asked for.

    The run's outcome carries it as status ``'incomplete'``, reason
    ``'captures_failed'``, in these words; the person is told once when the
    run ends. The images that were captured are saved.

    Attributes:
        asked: Captures the run was asked for.
        captured: Captures that produced an image.
        failed_steps: The name of each step whose capture failed, in the
            order they failed.
    """

    title = 'Run Incomplete'
    _NAMED_AT_MOST = 5

    def __init__(self, *, asked: int, captured: int, failed_steps: list[str]):
        missing = asked - captured
        message = f'{missing} of the {asked} captures this run was asked for produced no image'
        if failed_steps:
            names = failed_steps[: self._NAMED_AT_MOST]
            more = len(failed_steps) - len(names)
            message += (
                ' (failed: ' + ', '.join(names) + (f', and {more} more' if more else '') + ')'
            )
        never_reached = missing - len(failed_steps)
        if never_reached:
            message += f'; {never_reached} were never reached'
        message += '. The images that were captured are saved; check the log for each failure.'
        super().__init__(message, 'captures_failed')
        self.asked = asked
        self.captured = captured
        self.failed_steps = list(failed_steps)


class RunFailedError(CaptureError):
    """The instrument ended a run: the fault that ended it, in its own words.

    Fatal: the person is told even during an unattended run, since the run
    they left is no longer running. The title and words are the run's
    ending's, so a client reading the run's outcome and one hearing this
    read the same.

    Attributes:
        title: The ending's heading.
        reason: The ending's machine-readable cause.
    """

    fatal = True

    def __init__(self, *, reason: str, title: str, message: str):
        super().__init__(message, reason)
        self.title = title


class RunFailedToStartError(CaptureError):
    """A run was committed to and did not start, in its ending's words.

    Not fatal: it is reported after the run's cleanup has lifted the
    unattended mute, so the person who started it sees it.

    Attributes:
        title: The ending's heading.
        reason: The ending's machine-readable cause.
    """

    def __init__(self, *, reason: str, title: str, message: str):
        super().__init__(message, reason)
        self.title = title


class CompositeFailedError(CaptureError):
    """A run's composite was not merged; the run's own images stand.

    Attributes:
        reason: Why the merge did not happen.
    """

    title = 'Composite Failed'


class AutoGainNotSettledError(CaptureError):
    """Auto-gain locked with no usable exposure or gain from the camera.

    The previous settings were kept, so a capture taken with it had no
    exposure check.
    """

    title = 'Auto-gain did not settle'

    def __init__(self):
        super().__init__(
            'The camera reported no usable exposure or gain when auto-gain was '
            'locked, so the previous settings were kept and any capture was taken '
            'without an exposure check. Check the live view, then try again.',
            'auto_gain_not_settled',
        )


class AutofocusFailedError(CaptureError):
    """An autofocus sweep chose no focus; the stage goes back to where it started.

    Attributes:
        reason: ``'flat_focus_curve'`` (every score zero or invalid) or
            ``'unexpected_error'`` (the sweep raised; its cause is chained).
    """

    title = 'Autofocus Failed'
    _WORDS: ClassVar[dict[str, str]] = {
        'flat_focus_curve': 'Focus curve is flat or invalid -- check sample and illumination',
        'unexpected_error': 'Autofocus stopped on an unexpected error; the log has the details.',
    }

    def __init__(self, reason: str):
        super().__init__(self._WORDS[reason], reason)


class AutofocusZNotRestoredError(CaptureError):
    """Autofocus stopped without a result and could not put Z back.

    What the person must do depends on how the restore failed: a move that
    faulted has lost Z, and no move is accepted on it until a home; a
    refused target left Z known, where the sweep parked it.

    Attributes:
        reason: ``'z_position_lost'`` or ``'z_left_at_search_position'``.
    """

    title = 'Z Position Not Restored'

    def __init__(self, *, z_lost: bool):
        if z_lost:
            reason = 'z_position_lost'
            what = 'The Z position is now unknown -- home the scope before moving it.'
        else:
            reason = 'z_left_at_search_position'
            what = 'Z was left at the last autofocus search position.'
        super().__init__(f'Could not restore Z position after autofocus stopped. {what}', reason)


class RunWriteRefusedError(CaptureError):
    """A write was handed to a run whose writes have ended.

    A run's writes end when its cleanup closes them, or when a writer
    recovery or a shutdown abandons them. A write arriving after that
    belongs to no run: taken, it would count toward the next run's files or
    land after the finished run said its files were written. The caller
    says what was not saved.

    Reasons: ``run_ended`` (closed by the run's cleanup),
    ``writes_abandoned`` (abandoned by a recovery or a shutdown) or
    ``writer_shut_down`` (the file lane itself no longer takes work).
    """

    title = 'Not Saved'

    _WHY: ClassVar[dict[str, str]] = {
        'run_ended': 'the run it belongs to has ended',
        'writes_abandoned': "the run's remaining writes were given up",
        'writer_shut_down': 'the file writer has shut down',
    }

    def __init__(self, reason: str, what: str):
        super().__init__(f'{what} was not saved: {self._WHY[reason]}.', reason)
        self.what = what


class FileWriterNotStuckError(Refusal, Exception):
    """File-writer recovery was asked for while nothing is stuck.

    Recovery discards the images still waiting to be written, so it runs
    only when the write in flight has stopped making progress. A writer that
    is behind but moving finishes on its own; discarding its images then
    would be data loss for nothing.

    Attributes:
        reason: ``'file_writer_not_stuck'``.
        pending: The writes still outstanding when asked.
    """

    title = 'File Writer Not Stuck'

    def __init__(self, pending: int):
        if pending:
            state = f'{pending} image(s) are still being written and the writer is making progress'
        else:
            state = 'nothing is waiting to be written'
        super().__init__(
            f'The file writer is not stuck: {state}. Recovery would discard images, '
            'so it is offered only when a write stops making progress.'
        )
        self.reason = 'file_writer_not_stuck'
        self.pending = pending


class CameraStreamStalledError(CaptureError):
    """The camera stopped delivering frames while it stayed connected and streaming.

    A camera's link or grab loop can stall without the device being removed:
    it reports itself connected and grabbing, and no frame arrives. Nothing
    is waiting on the stream when that happens, so the imaging API reports it
    when its stream check sees the frame count stand still for longer than a
    frame at the current exposure can take.

    Attributes:
        seconds: How long no new frame had arrived when the stall was seen.
    """

    title = 'Camera Not Delivering Frames'

    def __init__(self, seconds: float):
        super().__init__(
            f'The camera has delivered no new frame for {seconds:.0f} s, although it is '
            'connected and streaming. Check the USB cable and power connections; if '
            'frames do not resume, restart LumaViewPro.',
            'camera_stream_stalled',
        )
        self.seconds = seconds


class FrameListenerNotRegisteredError(CaptureError):
    """The camera driver would not take a frame listener, so it will receive no frames.

    Raised by ``add_frame_listener`` to whoever registered it -- a plugin, a
    recording, a script -- because each of them has something to undo or to
    refuse: a recording that started without its frames records nothing, and
    a plugin listed as loaded never runs. The driver's error is the cause.

    Attributes:
        name: The listener's display name.
    """

    title = 'Frame Listener Not Registered'

    def __init__(self, name: str) -> None:
        super().__init__(
            f"The camera did not accept the frame listener '{name}', so it will "
            'receive no frames. Check the log for the camera driver error.',
            'frame_listener_refused',
        )
        self.name = name


class FrameHandlerRemovedError(Refusal, Exception):
    """The API stopped calling a frame handler that kept failing or kept running too long.

    Nothing raises it: the frame thread that removes the handler has no
    caller to raise to, so it is built to be reported. A refusal -- the API
    declining to go on calling the handler, which the plugin's author can fix
    -- so it is shown as a warning and logged without a traceback.

    Attributes:
        name: The handler's display name.
        reason: ``'over_budget'`` -- it ran past the per-frame budget for the
            drop count of frames in a row; ``'raised'`` -- it raised for the
            drop count of frames in a row.
    """

    title = 'Plugin Removed'

    def __init__(
        self, name: str, reason: str, *, budget_ms: int, drop_k: int, last_ms: float = 0.0
    ) -> None:
        if reason == 'over_budget':
            cause = (
                f'exceeded the {budget_ms}ms budget for {drop_k} consecutive frames '
                f'(last: {last_ms:.0f}ms)'
            )
            fix = "Reduce the handler's per-frame cost"
        else:
            cause = f'raised an error on {drop_k} consecutive frames'
            fix = "Fix the handler's error (the first one is in the log)"
        super().__init__(
            f"Plugin '{name}': the frame handler {cause}. It has been disabled to "
            f'protect the imaging pipeline. {fix} and re-register, or restart the '
            'application.'
        )
        self.name = name
        self.reason = reason


class PluginError(Exception):
    """A plugin failed: it did not load, or it failed after loading.

    Nothing raises these: the plugin host catches the plugin's own failure
    and has no caller to raise to, so each is built to be reported, chained
    from the plugin's exception when there is one so the log record carries
    that traceback. A plugin is separately versioned and may not be ours,
    so its failure is a fault for the person to hear about while the rest
    of the application carries on.

    The title names the plugin, so two plugins failing together are two
    notices rather than one that hides the other.

    Attributes:
        plugin_name: The plugin that failed.
    """

    def __init__(self, plugin_name: str, title: str, message: str):
        super().__init__(message)
        self.plugin_name = plugin_name
        self.title = title


class PluginNotLoadedError(PluginError):
    """A plugin found in the plugin group did not load.

    Attributes:
        reason: Why, in words a person can act on.
    """

    def __init__(self, plugin_name: str, reason: str):
        super().__init__(
            plugin_name,
            f'Plugin Not Loaded: {plugin_name}',
            f'The "{plugin_name}" plugin did not load: {reason}. The rest of '
            'LumaViewPro is unaffected.',
        )
        self.reason = reason


class PluginFailedError(PluginError):
    """A loaded plugin failed while the host was calling it.

    Attributes:
        hook: What the host was calling the plugin for.
        detail: The plugin's own account of the failure, when it gave one.
    """

    def __init__(self, plugin_name: str, hook: str, detail: str = ''):
        said = f' It said: {detail}' if detail else ''
        super().__init__(
            plugin_name,
            f'Plugin Error: {plugin_name}',
            f'The "{plugin_name}" plugin failed ({hook}), so that action did not '
            f'complete. The rest of LumaViewPro is unaffected.{said}',
        )
        self.hook = hook
        self.detail = detail


class HardwareCommandRefusedError(Refusal, Exception):
    """A hardware command was refused: something else has the scope, the lane is closed, or nothing is connected to take it.

    Raised to whoever made the command -- a public hardware member (LED,
    camera and motion commands), a raw task on a lane, the Session's
    objective writers -- and never dropped: a command refused without a
    raise reaches no hardware and reports success. While a run or a
    diagnostic holds the scope, a lane refuses any task not made under
    the holder's taking with this; so do the run's own executor fences,
    which cannot say who closed them.

    The Session's objective writers (select, slot assign and slot clear)
    raise it too while a run or a diagnostic holds the scope: the run
    stamps the active objective's scale into each capture, so a change
    mid-run is a command against the run's hardware state.

    A home with no motor controller connected raises it as well
    (``'not_connected'``): nothing was driven, and the person's remedy is
    the cable, not a retry.

    A declined request, not a fault, so it is a ``Refusal``: the lane shows
    it as a warning in its own words and logs one line without a
    traceback. The message is written for the person at the scope and
    names the holder; ``reason`` and ``member`` are for code that maps a
    refusal to a response (REST status codes, SDK branches) and for the
    log.

    Attributes:
        reason: Machine-readable refusal code.
        member: The member or task that was refused, for the log.
        holder: The kind of activity holding the scope, when known.
        title: The heading shown with the sentence, which follows the reason:
            nothing connected is not a busy microscope.
    """

    def __init__(self, reason: str, member: str, holder: str | None = None):
        super().__init__(_command_refused_sentence(reason, holder))
        self.reason = reason
        self.member = member
        self.holder = holder
        self.title = (
            'Not Connected'
            if reason in ('not_connected', 'scope_disconnected')
            else 'Microscope Busy'
        )


_HOLDER_NOUNS = {'protocol': 'A run', 'diagnostic': 'A diagnostic', 'recording': 'A recording'}


def _command_refused_sentence(reason: str, holder: str | None) -> str:
    if reason == 'capture_in_flight':
        return 'A capture is still being saved. Try again in a moment.'
    if reason == 'activity_ended':
        return 'The activity that sent this command has ended, so the command was not sent.'
    if reason == 'scope_disconnected':
        return 'The microscope has been disconnected, so the command was not sent.'
    if reason == 'not_connected':
        return (
            'The motor controller is not connected. Check the USB cable and that '
            'no other program is holding the port.'
        )
    who = _HOLDER_NOUNS.get(holder, 'Another activity')
    return f'{who} is using the microscope. Try again when it ends.'


class DiagnosticRefusedError(Refusal, Exception):
    """A diagnostic could not take the scope: another activity holds it.

    Raised by ``ScopeSession.diagnostic_claim()`` when a run, a recording
    or another diagnostic already holds the session's activity claim. A
    diagnostic drives the hardware directly (homes, LED modes, forced
    grabs), so it runs only on a scope nothing else is using, and a caller
    told why can wait for the holder or stop it. Nothing was committed.

    Attributes:
        reason: Machine-readable refusal code for callers that map refusals
            to responses (REST status codes, SDK branches).
        title: Short user-facing refusal title.
        message: One-sentence user-facing refusal body.
        holder: The activity kind holding the claim at refusal time.
        holder_trigger: The holding run's run_trigger_source when the
            holder is a run; None otherwise.
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


class PositionOutOfRangeError(Refusal, ValueError):
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

    title = 'Position Out of Range'

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


class AxisStateUnknownError(Refusal, Exception):
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

    title = 'Scope Not Homed'

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
    """A move ended without its axis arriving at the target.

    The move was driven, so this is a failure, not a refusal: the motor
    may have travelled any part of the way. A waited move that returns
    means the axis arrived; this move could not say that, so it raises
    instead of reporting a position nobody reached.

    Distinct from ``AxisStateUnknownError``, which refuses a move before
    anything is driven and offers ``force=True`` -- advice that is wrong
    for a move that already happened.

    Attributes:
        axis: The axis whose move did not complete.
        reason: ``'driver_failed'`` -- the motor board did not take the
            command (chained from the driver's error). ``'stalled'`` --
            the motion monitor gave the axis up: it did not reach its
            target within the motion bound. ``'board_lost'`` -- the
            monitor lost the motor board while the axis moved.
            ``'faulted'`` -- something else set the axis UNKNOWN during
            the wait (a disconnect, a home). ``'timed_out'`` -- the wait's
            bound ran out before the axis arrived. Each of those leaves
            the axis UNKNOWN. ``'stopped'`` -- a stop was issued while it
            moved; the axis is where the stop left it, which its position
            reports, and a turret is in no known slot.
        title: The heading shown with the sentence, which follows the reason.
    """

    _SENTENCES: ClassVar[dict[str, str]] = {
        'driver_failed': (
            'the motor board did not take the command. The {axis} position is now '
            'unknown -- home the scope before moving it again.'
        ),
        'stalled': (
            'it did not reach its target within the motion time limit and was '
            'abandoned. Check for an obstruction, then home the axis and retry.'
        ),
        'board_lost': (
            'the motor board was lost while it moved, so the move was aborted. '
            'Reconnect the board and retry.'
        ),
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

    _TITLES: ClassVar[dict[str, str]] = {
        'stalled': 'Motor Axis Stalled',
        'board_lost': 'Motor Board Disconnected',
    }

    def __init__(self, axis: str, reason: str):
        super().__init__(
            f'The {axis} move did not complete: ' + self._SENTENCES[reason].format(axis=axis)
        )
        self.axis = axis
        self.reason = reason
        self.title = self._TITLES.get(reason, 'Move Did Not Complete')


class MotorStopFailedError(Exception):
    """The motor STOP command failed, so the stage may still be moving.

    Raised by ``stop_motion`` when the board did not take the STOP, chained
    from the driver's error. A failure, not a refusal: the STOP was sent
    and nothing vouches that it landed. The stop generation has already
    moved, so no waited move reads itself as arrived.
    """

    title = 'Motor Stop Failed'

    def __init__(self):
        super().__init__(
            'The motor STOP command failed. If the stage is still moving, '
            'power-cycle the microscope.'
        )


class ScopeDisconnectError(Exception):
    """One or more parts of the microscope did not shut down cleanly.

    Raised by ``Lumascope.disconnect`` only after every teardown step has
    run, so one part's failure never leaves the others connected. Names
    each part that failed, in the order the teardown reached it, and is
    chained from the first part's error; ``causes`` holds every part's
    error. A failure, not a refusal.

    Attributes:
        parts: The parts that failed, in teardown order.
        causes: Each failed part's error, keyed by part.
    """

    title = 'Disconnect Failed'

    _WORDS: ClassVar[dict[str, str]] = {
        'motor stop': (
            'The motor STOP command failed; if the stage is still moving, '
            'power-cycle the microscope.'
        ),
        'LED board': (
            'The LED board did not shut down cleanly; its serial port may be '
            'left open, and reconnecting may require a restart.'
        ),
        'motor board': (
            'The motor board did not shut down cleanly; its serial port may be '
            'left open, and reconnecting may require a restart.'
        ),
        'camera': (
            'The camera did not shut down cleanly; its USB resources may not be '
            'released until the app restarts.'
        ),
    }

    def __init__(self, causes: dict[str, BaseException]):
        self.causes = dict(causes)
        self.parts = tuple(self.causes)
        super().__init__(' '.join(self._WORDS[part] for part in self.parts))


class HomingFailedError(Exception):
    """A home was driven and did not establish a reference position.

    The driver answered that the home failed, the driver raised, or the
    home finished and a homed axis's position could not be read. Each
    leaves the axes UNKNOWN, so no caller may take the scope as knowing
    where it is. A failure, not a refusal: the motors may have moved.
    Chained from the driver's exception when there is one.

    Attributes:
        home: What was homed: ``'ALL'``, ``'Z'`` or ``'T'``.
        reason: ``'failed'`` -- the driver answered False; ``'error'`` --
            the home raised; ``'unread'`` -- homed, but ``axes`` could not
            be read.
        axes: The axes left without a known position.
    """

    title = 'Homing Failed'

    _SUBJECTS: ClassVar[dict[str, str]] = {
        'ALL': 'Homing',
        'Z': 'Z axis homing',
        'T': 'Turret homing',
    }

    def __init__(self, home: str, reason: str, axes: Iterable[str]):
        self.home = home
        self.reason = reason
        self.axes = tuple(axes)
        if reason == 'unread':
            sentence = (
                f'Homing finished but the position of {", ".join(self.axes)} '
                'could not be read. Position is unknown.'
            )
        elif reason == 'failed':
            sentence = f'{self._SUBJECTS[home]} failed. Position is unknown.'
        else:
            sentence = f'{self._SUBJECTS[home]} encountered an error. Position is unknown.'
        super().__init__(sentence)


class AutofocusAborted(Exception):  # noqa: N818 -- cancellation/abort signal, not an error; non-Error suffix is intentional
    """Autofocus run aborted by caller (e.g. user cancelled, protocol
    aborted, or app teardown)."""

    pass


class CameraSettingRejected(Exception):  # noqa: N818 -- named for the event it signals; one type covers the defect class
    """The camera driver rejected a state-changing setting apply.

    Raised by ImagingAPI setters (frame size, binning, pixel format, gain,
    exposure) when a LIVE driver refuses the apply -- distinct from the
    camera-absent no-op, which stays a quiet sentinel per the
    missing-hardware contract. Success is observed by receiving the
    applied/delivered value, failure by this raise, so a caller cannot
    record a rejected apply as applied by forgetting to check a return code.

    A fault, not a refusal: the camera did not take a value it should
    have, and the words say so. They are written for the person, so the
    one reporter shows them where the flight stops; no setter logs or
    notifies it.

    Attributes:
        setting: Machine-readable setting name (e.g. 'frame_size').
        requested: The value the caller asked for.
        title: The notification title the reporter shows.
    """

    def __init__(self, setting: str, requested, *, title: str, message: str):
        super().__init__(message)
        self.setting = setting
        self.requested = requested
        self.title = title


class CameraSettingUnsupportedError(Refusal, ValueError):
    """A camera setting this camera does not offer was asked for.

    Declined before anything reaches the camera: the value is outside the
    list the camera itself reports (its binning factors), so nothing
    broke and the person can pick one it offers. A ``ValueError`` too,
    for a caller that treats it as the bad argument it is.

    Attributes:
        reason: Machine-readable refusal code, ``'<setting>_unsupported'``.
        setting: Machine-readable setting name (e.g. 'binning').
        requested: The value asked for.
        offered: The values the camera reports it supports.
    """

    def __init__(self, setting: str, requested, offered, *, title: str, message: str):
        super().__init__(message)
        self.reason = f'{setting}_unsupported'
        self.setting = setting
        self.requested = requested
        self.offered = offered
        self.title = title


class CameraSettingOutOfRangeError(Refusal, ValueError):
    """A camera setting outside the range this camera declares was asked for.

    Declined before anything reaches the camera, the same on every camera: a
    body that would refuse the value and a body that would silently clamp it
    both answer this way, so a caller that does not read the return cannot
    record a value the camera never took. Nothing broke, and the caller can
    ask again inside the range, so a refusal rather than a fault. A
    ``ValueError`` too, for a caller that treats it as the bad argument it is.

    Attributes:
        reason: Machine-readable refusal code, ``'<setting>_out_of_range'``.
        setting: Machine-readable setting name ('gain_db', 'exposure_ms',
            'frame_width', 'frame_height').
        requested: The value asked for.
        minimum: The camera's declared floor, or None when it declares none
            (an undeclared floor is not checked).
        maximum: The camera's declared ceiling, or None when it declares none.
        title: The notification title the reporter shows.
    """

    def __init__(
        self,
        setting: str,
        requested: float,
        minimum: float | None,
        maximum: float | None,
        *,
        title: str,
        message: str,
    ):
        super().__init__(message)
        self.reason = f'{setting}_out_of_range'
        self.setting = setting
        self.requested = requested
        self.minimum = minimum
        self.maximum = maximum
        self.title = title


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
