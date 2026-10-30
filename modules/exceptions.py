# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Module/API-layer exception classes (raised from modules/, caught at UI).

For driver-layer hardware exceptions (HardwareError), see drivers/exceptions.py.
"""

import datetime
import pathlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from lib.lumascope_error import Fault, LumascopeError, RefusalCause, _Umbrella
from lib.stage_interlock import INTERLOCK_TITLES, REFUSED_BEFORE_MOTION
from drivers.exceptions import (
    CameraFirmwareMissingError,
    CameraNoAnswerError,
    CameraReadFailedError,
    MotionInterlockError,
)
from modules.api_surface import api_fields
from modules.axes import Axis
from modules.caller import Caller
from modules.live_work import SCOPE_HOLDERS
from modules.run_outcome import ENDING_TITLES, FAILURE_REASONS

if TYPE_CHECKING:
    from modules.frame_validity import FrameValiditySource
    from modules.plugins import PluginNotLoaded
    from modules.tech_support_report import SupportReportKind

# A camera that is connected and not delivering frames: a run refused at its
# start and a capture refused at its call carry the one heading.
CAMERA_NOT_STREAMING_TITLE = 'Camera Not Streaming'

# A held control session refuses a lane's task, a take of the scope and a
# guarded write alike: each carries the one heading.
UNDER_CONTROL_TITLE = 'Under Remote Control'

# A capture and a build never overlap: a run and a recording refused while a
# build is on the post-processing lane carry the one heading and sentence.
POST_PROCESSING_RUNNING_TITLE = 'Post-Processing Running'
POST_PROCESSING_RUNNING_MESSAGE = (
    'Post-processing is still running. A capture waits until it has finished, '
    'so the two do not compete for the computer. Please wait.'
)


# What holds the microscope when a refusal names its holder: the kinds that
# hold the scope, by their WorkKind words, and an autofocus sweep still in
# flight after its run ended. Built from SCOPE_HOLDERS, so the claim and the
# refusal spell the holding kinds once.
Holder = StrEnum(
    'Holder',
    [(kind.name, kind.value) for kind in sorted(SCOPE_HOLDERS)] + [('AUTOFOCUS', 'autofocus')],
    module=__name__,
)
Holder.__doc__ = 'What holds the microscope when a refusal names its holder.'


def _holder(kind: str | None) -> Holder | None:
    """The holder a refusal names, from the claim's kind or a member; None for none."""
    return None if kind is None else Holder(kind)


class RemedyName(StrEnum):
    """The Session members a refusal can name as its remedy (``ScopeSession.apply_remedy``)."""

    RECOVER_FILE_WRITER = 'recover_file_writer'
    REPAIR_CAMERA_DRIVER = 'repair_camera_driver'


@api_fields('member', 'confirm_text', 'cancel_text')
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

    member: RemedyName
    confirm_text: str
    cancel_text: str


class Refusal(LumascopeError):  # noqa: N818 -- a kind of outcome, not an error
    """A request the scope declined: nothing broke, and the person who asked can act on it.

    Mixed into an exception whose message is written for that person. The
    background executor shows a refusal as a warning under ``title``, in the
    exception's own words, and logs one line for it with no traceback: an
    ERROR and a traceback say something went wrong, and a refusal is a
    designed outcome. Unmarked, a refusal raised inside a background task
    read as a crash -- "Background operation failed" over its message, and a
    traceback in the errors log.

    Every reason a refusal carries maps to its ``RefusalCause`` in
    ``causes``: whether the same request could succeed later with the
    caller changing nothing.

    Attributes:
        title: The heading the person reads above the message.
        remedy: The action that answers this refusal, when one exists; the
            reporter then shows the refusal as an offer to take it. Set per
            instance: one reason of a refusal type can have a remedy that its
            others do not.
    """

    title: str
    remedy: Remedy | None = None
    _needs_cause = True


class LiveFolderPathRefusedError(Refusal, ValueError):
    """A path the caller gave was refused: it cannot be used as the member uses it.

    Raised by ``ScopeSession.live_folder_path`` for a wire caller's name,
    which may reach only the live folder: a name that is a path of its own,
    or that climbs out, reaches the rest of the machine. Raised by the
    ``path_utils`` checks for any caller's path that is missing or of the
    wrong kind, before the member reads or writes it.

    Attributes:
        reason: ``'outside_live_folder'`` -- the name is empty, absolute,
            carries a drive or a network share, or leads outside the live
            folder through ``..`` or a link. ``'capture_location_unusable'``
            -- the live folder itself is missing or not a folder.
            ``'path_not_found'`` -- nothing is at the path. ``'not_a_file'`` --
            a folder is where a file is expected. ``'not_a_folder'`` -- a
            file is where a folder is expected. ``'wrong_extension'`` -- a
            file to write is named without its kind's extension (a
            protocol's ``.tsv``); the name given is the name written, so
            nothing adds it. ``'file_exists'`` -- a file to create is
            already there; an upload never replaces a file.
        name: The name or path that was given.
    """

    title = 'Path Not Available'
    # A missing live folder is an unplugged drive, which comes back; a path
    # that is missing or of the wrong kind is answered again unchanged, so a
    # polling client must not wait on it. A file already there is answered
    # again unchanged too: an upload never replaces one, so the caller gives
    # a new name.
    causes: ClassVar[dict[str, RefusalCause]] = {
        'outside_live_folder': RefusalCause.REQUEST,
        'capture_location_unusable': RefusalCause.STATE,
        'path_not_found': RefusalCause.REQUEST,
        'not_a_file': RefusalCause.REQUEST,
        'not_a_folder': RefusalCause.REQUEST,
        'wrong_extension': RefusalCause.REQUEST,
        'file_exists': RefusalCause.REQUEST,
    }

    def __init__(self, reason: str, name: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.name = name


class LiveFolderFileNotWrittenError(Fault, Exception):
    """A file put into the live folder could not be written.

    Raised by ``ScopeSession.put_live_folder_file``, chained from the
    ``OSError`` of the write, so the words name the file and the operating
    system's own reason -- a read-only or full disk, a denied permission.
    Nothing is left at the name.

    Attributes:
        name: The name under the live folder the file was to be written at.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'live_folder_file_not_written': None}
    words_for_person = True
    title = 'File Not Written'

    def __init__(self, name: str, cause: OSError):
        reason = cause.strerror or type(cause).__name__
        super().__init__(f'{name} was not written to the live folder ({reason}).')
        self.name = name


class Quiet(LumascopeError):  # noqa: N818 -- a kind of outcome, not an error
    """An outcome that is recorded and never shown: nothing failed and nothing was declined.

    Mixed into an exception that tells a caller something it may need to act
    on -- a script learning its handle is stale -- but that the person at the
    instrument has no reason to see. It is logged at INFO in its own words and
    never becomes a notification. Whether an outcome is quiet belongs to its
    type, never to the code that raises or catches it, so every client reads
    the same answer. A message on a quiet exception is written for the log.

    Every reason maps to its ``RefusalCause`` in ``causes``, as a refusal's
    does, since a quiet outcome answers a caller's request too.
    """

    _needs_cause = True


class Notice(LumascopeError):  # noqa: N818 -- a kind of outcome, not an error
    """An outcome that tells the person something: nothing failed and nothing was declined.

    Mixed into an exception whose message is written for the person -- a
    capture saved without its position, a long operation starting, a move
    someone else stopped. Most are reported, never raised: nothing waits on
    them, so the reporter is the whole of their flight. One that ends a
    caller's request is raised, as a stopped move is, and reported where
    its flight stops. It is shown as a notice under ``title`` in its own
    words, and logged at NOTICE. Its kind belongs to its type, as a
    refusal's does, so every client that hears it reads the same answer.

    A notice that is raised maps its reason to whether the same request
    could succeed later, as a refusal's does; one that is only reported maps
    it to None.

    Attributes:
        title: The heading the person reads above the message.
    """

    title: str
    remedy: Remedy | None = None
    _needs_cause = None


class ExposureAtMaximumNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """Auto-gain reached its exposure ceiling and the scene was still too dark.

    The setting keeps the ceiling; the person can add light or raise it.
    """

    title = 'Exposure at the maximum'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'exposure_at_maximum': None}

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
    causes: ClassVar[dict[str, RefusalCause | None]] = {'exposure_at_minimum': None}

    def __init__(self, exposure_ms: float, floor_ms: float):
        super().__init__(
            f'Auto-exposure settled at {exposure_ms:g} ms, at or below the '
            f'{floor_ms:g} ms usable floor for this channel; the setting keeps '
            'the floor. The scene is too bright: reduce the light.'
        )


class CapturePositionNotRecordedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A still was saved, without its well or position, because X or Y is unknown."""

    title = 'Position Not Recorded'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'position_not_recorded': None}

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

    title = CapturePositionNotRecordedNotice.title
    causes: ClassVar[dict[str, RefusalCause | None]] = {'position_not_recorded': None}
    same_fact_as = (CapturePositionNotRecordedNotice,)

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
    causes: ClassVar[dict[str, RefusalCause | None]] = {'duplicate_capture_filenames': None}

    def __init__(self, *, colliding_steps: int, shared_names: int):
        super().__init__(
            f'Protocol has {colliding_steps} steps sharing {shared_names} capture '
            'filenames. The protocol can be edited, but running it will be refused '
            'until each step produces a unique filename -- rename the colliding '
            'steps first.'
        )


class ProtocolStepsInvalidNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A protocol holds a step the run will refuse, after a load or an edit.

    The load and the edit still succeed, so the step can be fixed in the
    app; the run is refused at start until every step is valid. The words
    are the validator's own lines, one per field.
    """

    title = 'Protocol has steps that cannot run'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'protocol_steps_invalid': None}

    def __init__(self, *, errors: list[str]):
        super().__init__(
            f'{len(errors)} step field(s) will be refused when the protocol is run. '
            'The protocol can be edited; fix these first:\n' + '\n'.join(errors)
        )


class SlowFileWritesNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A run spent long enough waiting on the save disk to say so at its end."""

    title = 'Very Slow File Writes'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'slow_file_writes': None}

    def __init__(self):
        super().__init__(
            'Very slow writes are occurring on the save disk. '
            'Please confirm your computer and storage are OK.'
        )


class SingleScanNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A timed run will perform one scan, because of its period and duration."""

    title = 'Single Scan'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'single_scan': None}

    def __init__(self, *, period_s: float, duration_s: float):
        if period_s == 0:
            because = 'the capture period is 0'
        else:
            # Shown as H:MM:SS, as a person reads a schedule.
            duration = datetime.timedelta(seconds=duration_s)
            period = datetime.timedelta(seconds=period_s)
            because = f'the duration ({duration}) is shorter than the capture period ({period})'
        super().__init__(f'This run performs a single scan because {because}.')


class HyperstacksSavingNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A run's hyperstacks are being built, which can take minutes."""

    title = 'Saving Hyperstacks'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'hyperstacks_saving': None}

    def __init__(self):
        super().__init__(
            'Building hyperstacks from the run. This can take several minutes; '
            'a message will confirm completion.'
        )


class HyperstacksSavedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A run's hyperstacks were built: how many and where, or what was degraded."""

    title = 'Hyperstacks Saved'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'hyperstacks_saved': None}

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
    causes: ClassVar[dict[str, RefusalCause | None]] = {'no_hardware': None}

    def __init__(self):
        super().__init__(
            'No microscope hardware was detected. You can continue in software-only '
            'mode (live view + protocol design will work; capture will not). To '
            'connect hardware, power on the scope and reconnect the USB cable, then '
            'restart LumaViewPro.'
        )


class CellCountScaleDroppedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A method file saved before the count read each image's own scale.

    Every such file carries the old default of 1.0 pixels per micron, which
    described no image, so the load drops it and the count measures each
    image at its own scale. A person who meant 1.0 types it again.
    """

    title = 'Cell-count scale'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'cell_count_scale_dropped': None}

    def __init__(self, path):
        super().__init__(
            f'The method in {path} was saved with a fixed scale of 1.0 pixels per '
            'micron, the old default. It now counts each image at the scale the '
            'image states, and in pixels when it states none. Type a scale to '
            'override it, and save the method to keep it.'
        )


class ScopeModelDeferredNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A scope model saved for the next start, not the one running.

    The running scope keeps its model until it is next brought up, so a
    saved selection changes nothing yet; and a microscope that reports its
    own model overrides the selection then.
    """

    title = 'Scope model saved'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'scope_model_deferred': None}

    def __init__(self, model: str):
        super().__init__(
            f'{model} is saved as the configured model and applies the next time '
            'LumaViewPro starts. A microscope that reports its own model overrides '
            'this selection.'
        )
        self.model = model


class BinningSubstitutedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """The saved binning is one this camera does not offer; bring-up ran at the camera's.

    The camera's binning is stored in the saved value's place: the binning
    is half of the frame's geometry, and the store holds the pair the
    camera delivered.
    """

    title = 'Saved binning not supported'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'binning_substituted': None}

    def __init__(self, saved: int, used: int):
        super().__init__(
            f'The saved {saved}x{saved} binning is not supported by this camera; it '
            f'runs at {used}x{used}, which is now the saved binning.'
        )


class CameraConfigNotAppliedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """The camera came up without some of its connect-time configuration.

    Each step of the configuration is tried whatever an earlier one did, so
    the camera runs with the rest applied; the ones it did not take are
    named, the image orientation among them when it is one.

    Attributes:
        not_applied: The configuration steps the camera did not take, by name.
    """

    title = 'Camera configuration incomplete'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'camera_config_not_applied': None}

    def __init__(self, not_applied: tuple[str, ...]):
        self.not_applied = tuple(not_applied)
        super().__init__(
            f'The camera came up without {", ".join(self.not_applied)}; the rest of its '
            f'configuration was applied. Reconnect the camera, then restart LumaViewPro.'
        )


class FrameRefittedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """The saved frame is larger than this scope delivers; bring-up ran at the largest it does.

    The refitted frame is stored in the saved frame's place, as the
    substituted binning is, so the next bring-up has nothing to refit.
    """

    title = 'Saved frame too large'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'frame_refitted': None}

    def __init__(self, saved: tuple[int, int], used: tuple[int, int], binning_size: int):
        super().__init__(
            f'The saved {saved[0]}x{saved[1]} frame is larger than this scope delivers at '
            f'{binning_size}x{binning_size} binning; it runs at {used[0]}x{used[1]}, which is '
            'now the saved frame.'
        )


class CameraFeatureAbsentNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """A camera feature saved on that this camera does not have; it is now saved off.

    A camera without high conversion gain or line-noise reduction is at its
    standard setting, which is known, so that setting is stored in the saved
    value's place, as a substituted binning is.

    Attributes:
        feature: The feature, in words (``'high conversion gain'``).
    """

    title = 'Saved camera feature not available'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'camera_feature_absent': None}

    def __init__(self, feature: str):
        self.feature = feature
        super().__init__(
            f'The saved settings turn on {feature}, which this camera does not have; it '
            'runs without it, which is now the saved setting.'
        )


class ChannelBlockSupersededNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """The motor board carries the old Layers block, which is no longer read.

    The unit's channels and filterset are its model's. A board's own channels
    are its ``Channels`` block, beside its ``Filterset``.
    """

    title = 'Board channel block not read'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'channel_block_superseded': None}

    def __init__(self, model: str | None):
        source = f"the {model} model's" if model else "its model's"
        super().__init__(
            "This unit's motor board describes its channels in the old Layers block, which "
            f'this LumaViewPro no longer reads; the scope uses {source} channels and filterset.'
        )


class ChannelBlockUnusableNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """The motor board's Channels block cannot be used, so the scope has no channels.

    The block describes one filter assembly, so a block that contradicts
    itself is not read in part: no row can be trusted over another.
    """

    title = 'Board channel block unusable'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'channel_block_unusable': None}

    def __init__(self, why: str):
        super().__init__(
            f"This unit's motor board describes its channels in a block LumaViewPro cannot use "
            f"({why}), so the scope has no channels. Correct the board's motorconfig.json."
        )


class ChannelRowsSkippedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """Rows of the motor board's Channels block that could not be read; the rest are the scope's.

    One notice names every skipped row, as one load's replaced settings are
    one notice.

    Attributes:
        skipped: Each skipped row's key and why it was not read.
    """

    title = 'Board channels not read'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'channel_rows_skipped': None}

    def __init__(self, skipped: dict[str, str]):
        self.skipped = dict(skipped)
        rows = '; '.join(f'{key}: {why}' for key, why in self.skipped.items())
        super().__init__(
            "This unit's motor board names channels LumaViewPro cannot read, so the scope "
            f'runs without them: {rows}.'
        )


class StoredSettingReplacedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """Saved settings no write could store, replaced by the shipped values at load.

    One notice names every value one load replaced: the notification centre
    shows one notice of a kind at a time, so a notice per value showed the
    first and hid the rest. Each value is replaced for its key alone, in the
    settings the app runs on and so in the file at its next save.

    Attributes:
        replacements: ``(path, saved, used)`` for each replaced setting: its
            dotted path, the value the settings file held, and the shipped
            value now in its place.
    """

    title = 'Saved settings replaced'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'stored_setting_replaced': None}

    def __init__(self, replacements: list[tuple[str, object, object]]):
        listed = '; '.join(f'{path} {saved!r} -> {used!r}' for path, saved, used in replacements)
        super().__init__(
            f'These saved settings cannot be used, so they now hold the shipped values: {listed}.'
        )
        self.replacements = replacements


class LayerSettingCappedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """Saved layer gains or exposures above this camera's range; each runs at the camera's limit.

    The saved value is the person's intent and is kept: a camera that can
    reach it applies it again. One notice names every capped layer value
    one bring-up found, as ``StoredSettingReplacedNotice`` names every
    replaced one, since the notification centre shows one notice of a kind
    at a time.

    Attributes:
        capped: ``(layer, setting, stored, applied)`` for each capped value:
            the layer, ``'gain_db'`` or ``'exposure_ms'``, the saved value
            and the camera's limit it runs at.
    """

    title = "Saved layer settings above the camera's range"
    causes: ClassVar[dict[str, RefusalCause | None]] = {'layer_setting_capped': None}

    def __init__(self, capped: list[tuple[str, str, float, float]]):
        listed = '; '.join(
            f'{layer} {setting} {stored!r} runs at {applied!r}'
            for layer, setting, stored, applied in capped
        )
        super().__init__(
            'This camera cannot reach these saved layer settings, so each runs at the '
            f"camera's limit; the saved value is kept for a camera that can: {listed}."
        )
        self.capped = capped


class SettingsNotSavedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """The session's accepted settings writes are kept in memory, not saved.

    Told once per session and reason, at the first save the session
    withholds: the write itself was accepted and is in effect, so nothing
    was refused, but a person who expects it at the next start must hear
    that it will not be there.

    Attributes:
        reason: ``'settings_provisional'`` -- the app is running on the
            shipped template because the user's file could not be read, and
            that file is left as it is until they decide. ``'settings_unmeasured'``
            -- no hardware came up this session, so the per-channel values
            are defaults, not values measured on this scope.
        file: The file the writes would have been saved to.
    """

    title = 'Settings not saved'

    _SENTENCES: ClassVar[dict[str, str]] = {
        'settings_provisional': (
            'Settings changes are kept for this session only, not saved to {file}: '
            'LumaViewPro is running on its default settings because that file could '
            'not be read, and it is left as it is until you decide what to do with it.'
        ),
        'settings_unmeasured': (
            'Settings changes are kept for this session only, not saved to {file}: no '
            'microscope was connected this session, so the per-channel values are '
            'defaults, not values measured on this scope.'
        ),
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_SENTENCES)

    def __init__(self, reason: str, file: str):
        super().__init__(self._SENTENCES[reason].format(file=file))
        self.reason = reason
        self.file = file


class ProtocolError(_Umbrella, Exception):
    """A protocol's file, steps or run: the subject, never raised itself.

    Its children's words are written for the person unless a child says
    otherwise.
    """

    words_for_person = True


class ProtocolNotLoadedError(Fault, ProtocolError):
    """A protocol file could not be read.

    Raised by ``Protocol.from_file``, chained from the ``OSError`` of the
    read, so the words name the file and the operating system's own reason
    -- a missing file, a denied permission, a drive that went away. A file
    that was read but is not a protocol LumaViewPro can take is
    ``ProtocolFormatError`` instead.

    Attributes:
        file: The path that could not be read.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'protocol_not_loaded': None}

    title = 'Protocol Not Loaded'

    def __init__(self, file, cause: OSError):
        reason = cause.strerror or type(cause).__name__
        super().__init__(f'The protocol at {file} could not be read ({reason}).')
        self.file = file


class ProtocolNotSavedError(Fault, ProtocolError):
    """A protocol could not be written to its file.

    Raised by ``Protocol.to_file``, chained from the ``OSError`` of the write,
    so the words name the file and the operating system's own reason -- a
    missing or read-only folder, a full disk, a file another program holds --
    rather than guessing one. The write goes to a file beside the target
    first, so a file already there is unchanged.

    Attributes:
        file: The path the protocol was to be written to.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'protocol_not_saved': None}

    title = 'Protocol Not Saved'

    def __init__(self, file, cause: OSError):
        reason = cause.strerror or type(cause).__name__
        super().__init__(
            f'The protocol was not saved to {file} ({reason}). A file already there is unchanged.'
        )
        self.file = file


class FocusNotWrittenError(Refusal, ProtocolError):
    """The focus a scan found was not written, because the protocol's steps changed under it.

    Raised by ``Protocol.adopt_focus_from`` when the protocol no longer has
    the steps the scan focused: a different number of them, or a step at a
    different position, channel or objective. Each focused Z belongs to the
    step it was found at, so writing it into a different step would give that
    step a focus nobody measured. The protocol is left unchanged; scanning
    again focuses the steps it has now.

    Attributes:
        reason: ``'focus_not_written'``.
        difference: What no longer matches, in the words the person reads.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'focus_not_written': RefusalCause.STATE}
    title = 'Focus Not Saved'

    def __init__(self, difference: str):
        super().__init__(
            f'The focus found by the scan was not saved: {difference} since the scan '
            'started. The protocol is unchanged; run the scan again to focus its steps.'
        )
        self.difference = difference


class ConfigError(_Umbrella, Exception):
    """The application's configuration or settings: the subject, never raised itself.

    Its children's words are written for the person unless a child says
    otherwise.
    """

    words_for_person = True


class ConfigurationDefectError(Fault, ConfigError):
    """The program, the installation or the scope's composition broke a configuration contract.

    Not the request's and not the scope's state: no caller can change what it
    sent to make it succeed, so it is a fault, never a refusal. Raised where
    a value every door has already judged arrives unjudged, where a scope is
    used before it is composed, or where the installation lacks what the
    release ships. Its words are a developer's, kept in the log, except where
    the reason's are written for the person.

    Attributes:
        reason: ``'settings_not_prepared'`` -- a capture configuration names
            an output format this build does not write, so the stored image
            output format was never judged by the settings' preparation.
            ``'composite_thresholds_missing'`` -- a composite was asked for
            without a brightness threshold for every fluorescence layer it
            blends. ``'composite_threshold_out_of_range'`` -- a composite's
            brightness threshold is not a number from 0 to 100 percent.
            ``'capture_config_inconsistent'`` -- a capture configuration's
            capture depth and save encoding do not match its image mode.
            ``'tail_id_mode_unknown'`` -- a file name was asked for with a
            tail id mode the namer does not implement.
            ``'labware_not_given'`` -- an image's record was written with no
            labware to state. ``'layers_unresolved'`` -- this scope's layers
            could not be resolved from its model, so no layer can be acted
            on. ``'scope_not_composed'`` -- a scope that no session has
            bound was used for work that reads or saves settings.
            ``'zstack_reference_unchecked'`` -- a Z-stack's position
            reference is none of top, center and bottom.
            ``'route_shadowed'`` -- a Session member has the name of one of
            the REST server's own routes, so a client could not reach it.
    """

    _TITLES: ClassVar[dict[str, str]] = {
        'settings_not_prepared': 'Configuration Fault',
        'composite_thresholds_missing': 'Configuration Fault',
        'composite_threshold_out_of_range': 'Configuration Fault',
        'capture_config_inconsistent': 'Configuration Fault',
        'tail_id_mode_unknown': 'Configuration Fault',
        'labware_not_given': 'Configuration Fault',
        'layers_unresolved': 'Scope Model Not Known',
        'scope_not_composed': 'Configuration Fault',
        'zstack_reference_unchecked': 'Configuration Fault',
        'route_shadowed': 'Configuration Fault',
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_TITLES)
    _FOR_PERSON: ClassVar[frozenset[str]] = frozenset({'layers_unresolved'})

    def __init__(self, reason: str, message: str):
        if reason not in self._TITLES:
            raise TypeError(f'ConfigurationDefectError has no reason {reason!r}')
        super().__init__(message)
        self.reason = reason
        self.title = self._TITLES[reason]

    @property
    def words_for_person(self) -> bool:
        return self.reason in self._FOR_PERSON


class InstallationFileError(Fault, Exception):
    """A file the installation ships is missing, unreadable, or not the shape its reader needs.

    The installation is at fault, not the user's settings, so this is
    deliberately not a ``ConfigError``: a host that answers a ``ConfigError``
    by falling back to the shipped settings template would replace good
    settings and still fail on the same file. The reader raises it and logs
    nothing; whoever catches it logs it once.

    Attributes:
        file_path: The file that could not be used.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'installation_file_unusable': None}

    words_for_person = True
    title = 'Installation File Unusable'

    def __init__(self, file_path, problem: str):
        file_path = pathlib.Path(file_path)
        super().__init__(
            f'{file_path.name} in {file_path.parent} {problem}; reinstall LumaViewPro or '
            'restore the file'
        )
        self.file_path = file_path


class ScopeInUseError(Refusal, Exception):
    """A real scope was asked for while another is driving the microscope.

    Raised by a real scope's construction, before any of its hardware is
    opened, when another process holds the installation's instance lock --
    a LumaViewPro, a REST server or a script -- or another real scope in
    this process holds it: a process drives one real scope. Nothing was
    opened. Unlike ``BringUpError``'s port notices, which report one part
    that did not come up beside the rest, this refuses the whole scope.

    Attributes:
        reason: ``'scope_in_use'``.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'scope_in_use': RefusalCause.STATE}
    title = 'Microscope In Use'

    def __init__(self):
        super().__init__(
            'The microscope was not opened, because another LumaViewPro, REST server '
            'or script is driving it on this computer. Close that one and try again.'
        )


class BringUpError(_Umbrella, Exception):
    """A part of the scope did not come up as it should have; the rest of the scope runs.

    Reported, never raised: bring-up goes on without the part, and the
    person is told once what is missing and what to do about it. The record
    of the bring-up (``ScopeSession.bring_up_record``) keeps the fact for a
    client that asks later. The subject its children share, each a fault
    naming what did not come up; never built itself.

    Attributes:
        reason: The machine-readable code a client branches on, the child's.
    """

    words_for_person = True
    title = 'Hardware Unavailable'

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


class CameraNotAvailableError(Fault, BringUpError):
    """The camera did not come up; the heading and the advice follow the cause.

    Attributes:
        reason: ``'port_in_use'`` -- the part was found and its port could
            not be opened, most often because another program holds it.
            ``'not_detected'`` -- no driver found the part on USB. These two
            are the hardware registry's own causes, the LED board's too.
            ``'camera_driver_unavailable'`` -- no camera was found, and a
            camera driver did not load on this installation.
            ``'camera_firmware_missing'`` -- the firmware file a Classic's
            driver uploads to the camera is not installed.
            ``'camera_driver_unusable'`` -- Windows has a Classic on no
            driver, or on one LumaViewPro cannot use.
            ``'camera_not_initialized'`` -- the camera did not come up for any
            other hardware cause, among them a camera that was found and could
            not be opened, one another application holds included.
            ``'camera_bring_up_failed'`` -- LumaViewPro's own code failed while
            bringing the camera up; nothing says the camera is at fault.
    """

    _WORDS: ClassVar[dict[str, tuple[str, str]]] = {
        'port_in_use': (
            'Camera port in use',
            'Camera port is in use by another program. Close the other program and restart LVP.',
        ),
        'not_detected': (
            'Camera not detected',
            'Camera not found. Check USB cable and power.',
        ),
        'camera_driver_unavailable': (
            'Camera not detected',
            'Camera not found, and a camera driver did not load on this installation '
            '(the log names it and why). Reinstall LumaViewPro if this scope needs that '
            'driver; otherwise check USB cable and power.',
        ),
        'camera_firmware_missing': (
            'Camera firmware missing',
            "The camera's firmware file is missing from this installation. Reinstall LumaViewPro.",
        ),
        'camera_driver_unusable': (
            'Camera driver not usable',
            'Windows has the camera on no driver, or on one LumaViewPro cannot use.',
        ),
        'camera_not_initialized': (
            'Camera not initialized',
            'Could not connect to the camera. Check USB cable, power, and close other '
            'programs that may hold the camera.',
        ),
        'camera_bring_up_failed': (
            'Camera not brought up',
            'LumaViewPro failed while bringing the camera up. The log has the details; '
            'restart LumaViewPro, and send a support report if it happens again.',
        ),
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_WORDS)
    same_fact_as = (CameraFirmwareMissingError,)

    # What 'camera_driver_unusable' tells the person to do, by whether the
    # driver can be installed from here.
    _UNUSABLE_ANSWER: ClassVar[dict[bool, str]] = {
        True: (
            ' Install its driver from here: Windows asks an administrator to approve it, then '
            'whether to install software from Etaluma. Then restart LumaViewPro.'
        ),
        False: (
            ' This LumaViewPro installation cannot install the Lumascope Classic driver: install '
            'a LumaViewPro build that carries it, then restart LumaViewPro.'
        ),
    }

    def __init__(self, reason: str, *, repairable: bool = False):
        """Build the notice for ``reason``.

        Args:
            reason: One of the class's reasons.
            repairable: Offer Repair (``repair()``) as the answer; only for
                ``'camera_driver_unusable'``.
        """
        if repairable and reason != 'camera_driver_unusable':
            raise TypeError(f'Repair answers camera_driver_unusable, not {reason!r}')
        self.title, message = self._WORDS[reason]
        if reason == 'camera_driver_unusable':
            message += self._UNUSABLE_ANSWER[repairable]
        super().__init__(message, reason)
        self.remedy = self.repair() if repairable else None

    @staticmethod
    def repair() -> Remedy:
        """The offer of Repair, on 'camera_driver_unusable' where the Session can take it.

        An installed Windows build that carries the Classic driver package
        (``fx2_driver_step.repair_refusal``). Built when it is offered, not
        when this module loads: a ``Remedy`` is judged as it is built, and
        the judge's refusal is defined in this module.
        """
        return Remedy(
            member=RemedyName.REPAIR_CAMERA_DRIVER,
            confirm_text='Install the driver (needs an administrator)',
            cancel_text='Not now',
        )


class CameraDriverRepairRefusedError(Refusal, Exception):
    """Repair of the Lumascope Classic's driver binding was asked for and cannot run.

    Repair runs the installed driver step as an administrator, so it needs
    Windows, an installed build that carries the signed package, an
    administrator who approves the prompt, the person's yes to Windows'
    question whether to install Etaluma's driver, and a binding that needs it.

    Attributes:
        reason: ``'repair_needs_windows'`` -- the computer is not Windows,
            which alone binds a USB device to a driver.
            ``'repair_needs_installed_build'`` -- LumaViewPro is running from
            source or from a build in ``dist``, not from an installation.
            ``'repair_needs_driver_package'`` -- the installation was built
            without the Classic driver package.
            ``'no_classic_bindings'`` -- the scope reads no Classic USB
            device, such as a simulated one without an FX2.
            ``'repair_declined'`` -- the administrator's prompt was not
            approved. ``'driver_not_accepted'`` -- Windows did not take the
            package: installing it was declined at its question, or Windows
            refused it. ``'driver_already_usable'`` -- a camera is up, or
            every present device is on a driver LumaViewPro can use.
    """

    title = 'Camera Driver Not Repaired'
    # Declined and not accepted are the ones a person can answer by asking
    # again; the others are answered the same until the host, the build or
    # the binding changes.
    causes: ClassVar[dict[str, RefusalCause]] = {
        'repair_needs_windows': RefusalCause.REQUEST,
        'repair_needs_installed_build': RefusalCause.REQUEST,
        'repair_needs_driver_package': RefusalCause.REQUEST,
        'no_classic_bindings': RefusalCause.REQUEST,
        'repair_declined': RefusalCause.STATE,
        'driver_not_accepted': RefusalCause.STATE,
        'driver_already_usable': RefusalCause.REQUEST,
    }
    _WORDS: ClassVar[dict[str, str]] = {
        'repair_needs_windows': (
            "Repairing the camera's driver is for Windows, which binds each USB device to a "
            'driver; this computer binds none.'
        ),
        'repair_needs_installed_build': (
            "LumaViewPro installs the camera's driver only from an installed build, which carries "
            "the signed driver package. Install LumaViewPro on the scope's PC with its installer, "
            'and install the driver from there.'
        ),
        'repair_needs_driver_package': (
            'This LumaViewPro installation does not carry the Lumascope Classic driver (it was '
            'built without a signing certificate). Install a LumaViewPro build that carries it.'
        ),
        'no_classic_bindings': (
            'This scope reads no Lumascope Classic USB device, so there is no driver to install.'
        ),
        'repair_declined': (
            "Windows asked for an administrator's approval to install the camera's driver, and it "
            'was not given. Try again and approve it (a standard account needs an '
            "administrator's password)."
        ),
        'driver_not_accepted': (
            "Windows did not take the camera's driver: installing it was declined at Windows' "
            "question, or Windows refused the package (the driver step's log, in the logs "
            'folder of the LumaViewPro installation, says which). Try again to be asked again.'
        ),
        'driver_already_usable': (
            'The camera is up, or its USB devices are already on drivers LumaViewPro can use; '
            'there is nothing to repair.'
        ),
    }

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(self._WORDS[reason])


class CameraDriverRepairFailedError(Fault, Exception):
    """The driver step Repair ran as an administrator did not bind the camera.

    The step logs every action and every Windows answer, so the words name
    its log folder.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'camera_driver_repair_failed': None}
    title = 'Camera Driver Repair Failed'

    words_for_person = True

    def __init__(self, what: str, logs: pathlib.Path):
        super().__init__(
            f"The camera's driver was not installed: {what}. The driver step's log is in {logs}. "
            'Try again, or send that log to Etaluma support.'
        )


class RestServerNotStartedError(Fault, Exception):
    """The REST server served beside the application could not take its port.

    Raised by ``rest.server.serve_beside``, chained from the ``OSError`` of
    the bind, so the words name the address and the operating system's own
    reason -- most often another program on the port. Nothing is served; the
    application runs on without it.

    Attributes:
        port: The port that could not be taken.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'rest_server_not_started': None}
    title = 'REST Server Not Started'

    words_for_person = True

    def __init__(self, host: str, port: int, cause: OSError):
        reason = cause.strerror or type(cause).__name__
        super().__init__(
            f'The REST server was not started on {host}:{port} ({reason}). LumaViewPro runs '
            'without it; if another program uses that port, set rest_api.port to a free one.'
        )
        self.port = port


@api_fields('loc', 'msg', 'type')
class UnfitArgument(NamedTuple):
    """One argument of a request that does not fit its member, as the request reader found it."""

    # Where the argument is: ``body``, ``query`` or ``path``, then its name
    # and, inside it, the key or index.
    loc: tuple[str | int, ...]
    # Why it does not fit, in the reader's words.
    msg: str
    # The reader's code for why.
    type: str


@api_fields('errors')
class ServerRefusedError(Refusal, Exception):
    """A request the REST server itself answers: it never reached a member.

    Raised by ``rest/`` for a request it answers itself: from its request
    reader, its handle and job stores, its idempotency keys, its live view
    and its close. What the refusal is lives here; how it is sent -- each
    reason's status, the ``Retry-After`` header -- is ``rest.problems``'s,
    read from this record.

    Attributes:
        reason: The machine-readable code. ``'not_found'`` -- the route, the
            handle or the job asked for is not there.
            ``'invalid_request'`` -- the request's arguments do not fit its
            member. ``'method_not_allowed'`` -- the route does not answer the
            method it was asked with. ``'unsupported_media_type'`` -- the
            request's body is not JSON. ``'handle_shared'`` -- the handle
            asked to be forgotten is one every client shares, and it is kept.
            ``'overloaded'`` -- the server holds as many handles or jobs as
            it keeps; asking again later may succeed. ``'server_closing'`` --
            the server is closing: it finishes what is under way, answers
            nothing new, and exits. ``'no_frame_yet'`` -- the live view has
            no frame to show, because the camera sent none in time; asking
            again later may succeed. ``'not_cancellable'`` -- the job is a
            call held as a job when it outlived its wait, and holds no
            operation to stop. ``'idempotency_key_reused'`` -- the
            ``Idempotency-Key`` was sent before with a different request; a
            key is sent again only to retry the same request.
        title: The heading, the reason's.
        errors: For ``'invalid_request'`` from the request reader, each
            argument that does not fit; None otherwise.
        retry_after_s: Seconds to wait before asking again, for a refusal
            that waiting may lift (``'overloaded'``, ``'no_frame_yet'``);
            None when no wait helps.
    """

    errors: tuple[UnfitArgument, ...] | None
    retry_after_s: float | None

    causes: ClassVar[dict[str, RefusalCause]] = {
        'not_found': RefusalCause.REQUEST,
        'invalid_request': RefusalCause.REQUEST,
        'method_not_allowed': RefusalCause.REQUEST,
        'unsupported_media_type': RefusalCause.REQUEST,
        'handle_shared': RefusalCause.REQUEST,
        'overloaded': RefusalCause.STATE,
        'server_closing': RefusalCause.RESTART,
        'no_frame_yet': RefusalCause.STATE,
        'not_cancellable': RefusalCause.STATE,
        'idempotency_key_reused': RefusalCause.REQUEST,
    }

    _TITLES: ClassVar[dict[str, str]] = {
        'not_found': 'Not Found',
        'invalid_request': 'Invalid Request',
        'method_not_allowed': 'Method Not Allowed',
        'unsupported_media_type': 'Unsupported Media Type',
        'handle_shared': 'Handle Shared',
        'overloaded': 'Overloaded',
        'server_closing': 'Server Closing',
        'no_frame_yet': 'No Frame Yet',
        'not_cancellable': 'Not Cancellable',
        'idempotency_key_reused': 'Idempotency Key Reused',
    }

    def __init__(
        self,
        reason: str,
        detail: str,
        *,
        errors: tuple[UnfitArgument, ...] | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(detail)
        self.reason = reason
        self.title = self._TITLES[reason]
        self.errors = errors
        self.retry_after_s = retry_after_s


class CameraDriverRepairedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """Repair bound the Lumascope Classic to LumaViewPro's driver; the camera connects at the next start."""

    causes: ClassVar[dict[str, RefusalCause | None]] = {'camera_driver_repaired': None}
    title = 'Camera Driver Repaired'

    def __init__(self):
        super().__init__(
            "The camera is on LumaViewPro's driver now. Restart LumaViewPro so the camera connects."
        )


class LedBoardUnavailableError(Fault, BringUpError):
    """The LED board did not come up on a scope whose other parts did.

    Said once at bring-up rather than once per failed illumination command:
    without it the first symptom is a sample under a dark objective and
    controls that appear to do nothing. The advice follows the cause.

    Attributes:
        reason: The hardware registry's cause, ``'not_detected'`` and
            ``'port_in_use'`` as ``CameraNotAvailableError`` explains them,
            or: ``'not_responding'`` -- the board's port opened and the board
            answered nothing. ``'connect_failed'`` -- the board's driver
            raised while connecting. ``'no_driver'`` -- no real LED board
            driver is registered on this installation.
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
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_ADVICE)
    same_fact_as = (CameraNotAvailableError,)

    def __init__(self, reason: str):
        super().__init__(f'{self._ADVICE[reason]} The rest of the microscope is working.', reason)


class LedSafetyOffNotTakenError(Fault, BringUpError):
    """The LED board connected but did not confirm the LEDs-off sent on connect.

    Sample safety: firmware before the confirmation existed can leave channels
    on, photobleaching whatever is on the stage.

    Attributes:
        reason: ``'safety_off_failed'``.
    """

    title = 'LED Safety Off Not Confirmed'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'safety_off_failed': None}

    def __init__(self, detail: str):
        super().__init__(
            'The LED board connected but the safety LEDS_OFF command did not complete '
            f'({detail}). If the LEDs are stuck on, turn off illumination manually '
            'before placing a sample.',
            'safety_off_failed',
        )


class PartialHardwareError(Fault, BringUpError):
    """Parts this scope's model has did not come up; the rest of the scope runs.

    Attributes:
        reason: ``'partial_hardware'``.
        missing: Each missing part with its cause, as ``PartStatus.describe``
            writes it.
    """

    title = 'Partial Hardware Detected'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'partial_hardware': None}

    def __init__(self, missing: Iterable[str]):
        self.missing = tuple(missing)
        super().__init__(
            f'Not connected: {", ".join(self.missing)}. Some features will be unavailable.',
            'partial_hardware',
        )


class FocusNotSavedError(Refusal, ConfigError):
    """A channel's saved focus was asked for, and none was ever saved.

    A channel with no saved focus is imaged at the stage's current Z when a
    step is built for it, so there is no height to go to; answering with a
    stored placeholder is what sent every such channel to one fixed height.

    Attributes:
        reason: ``'focus_not_saved'``.
        layer: The channel asked about.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'focus_not_saved': RefusalCause.STATE}
    title = 'No Focus Saved'

    def __init__(self, layer: str):
        super().__init__(
            f'No focus is saved for {layer}. Focus {layer} and press Save Focus to set one.'
        )
        self.layer = layer


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
    selected.

    Attributes:
        reason: ``'slot_unknown'`` -- the turret is in no known slot: it has
            not been homed or moved since its slot was last lost.
            ``'slot_unassigned'`` -- the slot in the light path has no
            objective assigned. ``'not_in_catalogue'`` -- the slot in the
            light path is assigned an objective that is not in the objective
            catalogue. ``'none_selected'`` -- no objective has been selected,
            as on a scope with no turret before its first selection.
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
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_SENTENCES, RefusalCause.STATE)

    def __init__(self, reason: str, slot: int | None = None):
        super().__init__(
            'The objective in the light path is unknown: '
            + self._SENTENCES[reason].format(slot=slot)
        )
        self.reason = reason
        self.slot = slot


def _offered_words(offered: Iterable[object]) -> str:
    # What a refusal offers, in its words. An empty offer is said, as
    # "nothing": joined, it left a sentence with a hole where the list goes.
    return ', '.join(str(o) for o in offered) or 'nothing'


@api_fields('argument', 'offered')
class CatalogueNameRefusedError(Refusal, ConfigError):
    """A labware or objective name the installation's catalogue does not hold.

    Raised by the catalogue's owner -- ``WellPlateLoader.resolve_plate_key``
    for a plate, ``ObjectiveLoader.get_objective_info`` for an objective --
    before anything is stored, so every door that names one is refused
    alike. A ``ConfigError``, because the same check reads a stored setting
    at bring-up and the callers that recover from a bad stored name catch
    that type.

    The catalogue is the installation's, fixed for the process, so the same
    name is refused every time it is sent.

    Attributes:
        reason: ``'labware_unknown'`` -- the labware name is no plate in the
            labware catalogue. ``'objective_not_in_catalogue'`` -- the
            objective name is no key of the objective catalogue.
        argument: The owner's parameter: ``labware_id`` or ``objective_id``.
        value: What was given, as given.
        offered: The catalogue's keys.

    ``argument`` and ``offered`` are published, so a REST problem carries
    them beside the words. The run's ``ProtocolRunRefusedError`` names the
    same fact for a protocol's steps; ``argument`` is on this type only.
    """

    argument: str
    offered: tuple[str, ...]
    _WORDS: ClassVar[dict[str, tuple[str, str]]] = {
        'labware_unknown': (
            'Labware Not Known',
            '{argument} must be a plate in the labware catalogue, one of {offered}; '
            '{value!r} is not one.',
        ),
        'objective_not_in_catalogue': (
            'Objective Not Known',
            '{argument} must be a key of the objective catalogue, one of {offered}; '
            '{value!r} is not one.',
        ),
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_WORDS, RefusalCause.REQUEST)

    def __init__(self, reason: str, *, argument: str, value: object, offered: tuple[str, ...]):
        if reason not in self._WORDS:
            raise TypeError(f'CatalogueNameRefusedError has no words for the reason {reason!r}')
        self.title, words = self._WORDS[reason]
        super().__init__(
            words.format(argument=argument, value=value, offered=_offered_words(offered))
        )
        self.reason = reason
        self.argument = argument
        self.value = value
        self.offered = offered


@api_fields('path', 'member', 'limits')
class SettingRefusedError(Refusal, ConfigError):
    """A settings write was refused; nothing was written.

    Raised by ``ScopeSession.update_settings`` and ``set_live_folder``, by
    the members that own a ranged setting (``set_channel_acquire``,
    ``set_image_mode``, through ``settings_paths.refuse_outside_range``) and
    for a settings dict handed to a session, at ``create``'s door
    (``settings_init.prepare_loaded``, through
    ``settings_paths.refuse_handed_values``). A caller that maps refusals to
    responses branches on ``reason``; a person reads the message.

    Attributes:
        reason: ``'has_member'`` -- the setting is changed by its own
            member, named in ``member`` by its path from the session (the
            camera's as ``scope.camera.<name>``). ``'host_only'`` -- that
            member is in-process: only the application hosting the scope
            changes the setting, and a REST client reads it. ``'not_a_setting'`` -- no
            setting has this path. ``'block'`` -- the path names a block of
            settings, which is written one setting at a time.
            ``'installation_only'`` -- the setting is read only from the
            installation's settings file. ``'derived'`` -- the setting is
            derived from the release when the template is read
            (``step_channel_order``); a handed dict's differing copy is
            refused for it too. ``'wrong_kind'`` -- the value is not the kind
            the setting holds.
            ``'out_of_range'`` -- the value is outside the setting's range.
            ``'not_offered'`` -- the setting holds a closed set (a
            ``StrEnum``, such as ``image_mode``'s ``ImageMode``) and the
            value is not one of its values.
        path: The dotted path that was written.
        detail: Why, as a clause about the value (``'it must be above 0,
            not 0'``), for a caller that names the setting its own way: a
            protocol step's column.
        member: The member that changes the setting, for ``'has_member'``
            and ``'host_only'``; otherwise None.
        limits: The lowest and highest value the setting takes, for an
            ``'out_of_range'`` setting with both ends; otherwise None.

    ``path``, ``member`` and ``limits`` are published, so a REST problem
    names the setting refused, the member to call instead and the range,
    beside the words.
    """

    path: str
    member: str | None
    limits: tuple[float, float] | None
    title = 'Setting Not Changed'
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(
        (
            'out_of_range',
            'wrong_kind',
            'not_offered',
            'not_a_setting',
            'block',
            'has_member',
            'host_only',
            'installation_only',
            'derived',
        ),
        RefusalCause.REQUEST,
    )

    def __init__(
        self,
        reason: str,
        path: str,
        detail: str,
        *,
        member: str | None = None,
        limits: tuple[float, float] | None = None,
    ):
        # A detail may be another refusal's whole sentence, its period included.
        super().__init__(f'{path} was not changed: {detail.removesuffix(".")}.')
        self.reason = reason
        self.path = path
        self.detail = detail
        self.member = member
        self.limits = limits


class SettingsFileNotReplacedError(Fault, ConfigError):
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

    causes: ClassVar[dict[str, RefusalCause | None]] = {'settings_file_not_replaced': None}

    title = 'Settings File Not Replaced'

    def __init__(self, file: str, cause: OSError):
        reason = cause.strerror or type(cause).__name__
        super().__init__(
            f'{file} could not be moved aside ({reason}). Close any program that has '
            'it open, then choose again.'
        )
        self.file = file


class SettingsFileNotWrittenError(Fault, ConfigError):
    """The session's settings could not be written to its settings file.

    Reported by the session's save, chained from the ``OSError`` of the
    write, so the words name the file and the operating system's own reason.
    The write goes beside the file first, so the file already there is
    unchanged; the next accepted write, and the close, save again.

    Attributes:
        file: The settings file that was to be written.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'settings_file_not_written': None}

    title = 'Settings File Not Written'

    def __init__(self, file: str, cause: OSError):
        reason = cause.strerror or type(cause).__name__
        super().__init__(
            f'The settings were not saved to {file} ({reason}). The file already there is '
            'unchanged; the next settings change saves again.'
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
        reason: ``'model_unknown'``.
        model: The model that was refused.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'model_unknown': RefusalCause.REQUEST}
    title = 'Unknown Scope Model'

    def __init__(self, model: str, known: Iterable[str]):
        super().__init__(
            f'{model!r} is not a scope model this version of LumaViewPro knows. '
            f'Choose one of: {", ".join(sorted(known))}.'
        )
        self.model = model


class CaptureError(_Umbrella, Exception):
    """Image capture, save or processing: the subject, never raised itself.

    Attributes:
        reason: Machine-readable failure code for callers that map
            failures to responses (REST status codes, UI branches).
            Without it a caller has to regex the prose to tell a timeout
            from a failed merge from a camera that returned nothing.

    ``reason`` is required rather than defaulted. A default would let a
    new raise site stay untyped while every caller still had to guess
    which raises carry a usable code and which carry a placeholder.

    ``title`` is the heading the reporter shows above the message, which is
    written for the person. It is each child's, read from its own table; the
    subject has none, as it has no kind.
    """

    words_for_person = True

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


@api_fields('source')
class CaptureFailedError(Fault, CaptureError):
    """A capture or a recording was attempted and produced nothing.

    Raised by ``capture_and_wait`` and ``get_image`` for a capture that got
    no frame, and by ``capture_and_wait`` for one whose frame's record could
    not be read; a camera that is not there is refused instead
    (``HardwareCommandRefusedError``, ``'not_connected'``).

    Attributes:
        reason: One of:

            - ``'no_frame'`` -- the camera delivered no new frame within its
              grab window, whether the capture was draining, grabbing or
              waiting for a sum's next frame.
            - ``'deadline_expired'`` -- the scope's state kept changing for
              longer than the capture's budget.
            - ``'chunk_rejected'`` -- frames came, but none was exposed at
              the ``source`` setting asked.
            - ``'no_fresh_frame'`` -- frames came, but none within the
              capture's bound counted for the ``source`` change: each was
              exposed, by its source's rule, before that change reached it.
            - ``'record_read_failed'`` -- a frame came, but the camera did
              not answer a read its record needs (the gain, exposure or
              black level), so it was not kept; chained from the
              ``CameraReadFailedError``.
            - ``'unknown_save_encoding'`` -- a frame was handed an encoding
              no writer knows, which the program, not the person, got wrong.
        source: The source the capture failed for: for
            ``'chunk_rejected'`` the setting no frame matched, ``EXPOSURE``
            or ``GAIN``; for ``'no_fresh_frame'`` the change no frame came
            after (``Z_MOVE``, ``LED``); None otherwise.
    """

    source: 'FrameValiditySource | None'

    # Each source's change as a person says it, in the words of a
    # ``'no_fresh_frame'`` failure. Keyed by each ``FrameValiditySource``'s
    # value, which its member finds: importing the set here closes an import
    # cycle through the camera driver.
    _CHANGE_WORDS: ClassVar[dict[str, str]] = {
        'led': 'the LED change',
        'gain': 'the gain change',
        'exposure': 'the exposure change',
        'xy_move': 'the stage move',
        'z_move': 'the focus move',
        'turret': 'the turret move',
        'auto_gain': 'auto gain turning on',
        'pixel_format': 'the pixel format change',
        'frame_size': 'the frame size change',
        'binning': 'the binning change',
        'black_level': 'the black level change',
        'conversion_gain_mode': 'the conversion gain mode change',
        'line_noise_reduction': 'the line noise reduction change',
    }

    _WORDS: ClassVar[dict[str, tuple[str, str]]] = {
        'no_frame': (
            'Capture Failed',
            'The camera delivered no new frame in time, so nothing was captured. '
            'Check that the live view is moving; if it has stopped, check the '
            "camera's USB cable.",
        ),
        'deadline_expired': (
            'Capture Failed',
            "The microscope's state kept changing for longer than the capture could "
            'wait, so nothing was captured. Try again once the stage and lights '
            'have settled.',
        ),
        'chunk_rejected': (
            'Capture Failed',
            'The camera kept delivering frames that were not exposed at the '
            '{source} asked for, so none was captured. Try again; if it repeats, '
            'the camera is holding a different {source} than was set.',
        ),
        'no_fresh_frame': (
            'Capture Failed',
            'No frame the camera delivered in time was exposed after {change}, '
            'so nothing was captured. Try again; if it repeats, check that the '
            'live view is moving.',
        ),
        'record_read_failed': (
            'Capture Failed',
            'The camera did not answer a read of a setting the frame is saved '
            'with (its gain, exposure or black level), so nothing was captured. '
            "Try again; if it repeats, check the camera's USB cable.",
        ),
        'unknown_save_encoding': (
            'Capture Failed',
            'A frame was handed the save encoding {encoding!r}, which no image '
            'writer knows, so it was not saved.',
        ),
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_WORDS)

    def __init__(
        self,
        reason: str,
        *,
        source: 'FrameValiditySource | None' = None,
        encoding: str | None = None,
    ):
        if reason not in self._WORDS:
            raise TypeError(f'CaptureFailedError has no reason {reason!r}')
        if (source is None) == (reason in ('chunk_rejected', 'no_fresh_frame')):
            raise TypeError(
                f'a {reason!r} capture failure takes source= exactly when '
                'chunk_rejected or no_fresh_frame'
            )
        if source is not None and source not in self._CHANGE_WORDS:
            raise TypeError(f'CaptureFailedError has no source {source!r}')
        if (encoding is None) == (reason == 'unknown_save_encoding'):
            raise TypeError(
                f'a {reason!r} capture failure takes encoding= exactly when unknown_save_encoding'
            )
        self.title, words = self._WORDS[reason]
        change = self._CHANGE_WORDS.get(source)
        super().__init__(words.format(source=source, change=change, encoding=encoding), reason)
        self.source = source

    @property
    def words_for_person(self) -> bool:
        return self.reason != 'unknown_save_encoding'


class RunWaitExpiredError(ProtocolError, Quiet, TimeoutError):
    """A run's waiting form ran out of its ``timeout_s`` with the run still going: nothing failed.

    Raised by ``ProtocolRunner.run_X``. A ``TimeoutError``, as a caller's own
    wait expiring always is, and quiet: the run goes on, and ``operation``
    is the caller's hold on it, to wait on again or to stop. ``TimeoutError``
    is the last base, so its ``OSError`` constructor never reads the words
    as an errno.

    Attributes:
        reason: ``'still_running'``.
        member: The waiting form that was called.
        operation: The run's ``Operation``, still running when this was raised.
    """

    title = 'Run Still Going'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'still_running': RefusalCause.STATE}

    def __init__(self, member: str, timeout_s: float, operation):
        super().__init__(
            f'{member}: the run was still going when the wait ran out after {timeout_s:g} s; '
            "it goes on, held by this error's operation"
        )
        self.reason = 'still_running'
        self.member = member
        self.operation = operation


@api_fields('holder', 'holder_trigger')
class ProtocolRunRefusedError(Refusal, ProtocolError):
    """A sequenced run, or the protocol it would run, was refused before any state was committed.

    Raised by SequencedCaptureRunner.prepare() when a run cannot start
    (already running, files still writing, empty protocol, validation
    errors, hardware not connected), by the protocols API when a protocol
    or a step names something this scope cannot do, and by the protocol
    builder for a z-stack asked for with no range. Each of those raises it
    without reporting it: the boundary the caller sits behind reports it
    once, so a REST caller is answered and a GUI user is shown it.

    Its message is the sentence a person reads, so a headless caller that
    prints it prints what the GUI shows.

    Attributes:
        reason: Machine-readable refusal code for callers that map
            refusals to responses (REST status codes, UI branches).
            ``'already_running'`` -- a run already holds the scope.
            ``'exclusive_activity_running'`` -- another exclusive activity,
            such as a diagnostic, a home or a run, holds the scope, and the
            work asked for is not made under it.
            ``'recording_active'`` -- a video recording is in progress, or
            a previous one is still finishing.
            ``'files_writing'`` -- the last run is still writing its files.
            ``'files_writing_stalled'`` -- the last run's file writer has
            stopped making progress, and has to be recovered before a new
            run starts. ``'post_processing_running'`` -- a build is on the
            post-processing lane, and a capture never overlaps one.
            ``'autofocus_running'`` -- an autofocus sweep from a run that
            has ended did not stop, and still occupies the autofocus
            thread. ``'not_connected'`` -- a part the work needs is not
            connected, and none reconnects during a session.
            ``'camera_not_streaming'`` -- the camera is connected and
            delivers no frames. ``'position_unknown'`` -- an axis this
            scope has does not know its position, so the scope has to be
            homed first. ``'lid_open_before_motion'`` -- the stage's lid is
            open, so the stage refuses to move.
            ``'no_acquiring_layer'`` -- no channel, or not the step's, is
            set to acquire, so there is no step to make.
            ``'channel_already_in_add'`` -- another step added with this one
            already takes the channel, and steps added together are one
            composite with one image per channel.
            ``'composite_two_transmitted_layers'`` -- a composite run has
            more than one transmitted channel set to capture.
            ``'turret_slot_ambiguous'`` -- a step recorded no turret slot,
            and its objective is assigned to two or more.
            ``'turret_slot_reassigned'`` -- the turret slot a step recorded
            now holds another objective. ``'step_position_unknown'`` -- a
            step cannot be added or updated while an axis does not know its
            position. ``'turret_objectives_unassigned'`` -- the protocol
            uses an objective no turret slot is assigned.
            ``'objective_not_mounted'`` -- on a scope with no turret, the
            protocol names an objective other than the selected one.
            ``'sequence_name_invalid'`` -- the run's name is a path: it
            holds a separator or a drive, or is ``..``.
            ``'empty_protocol'`` -- the protocol has no steps.
            ``'validation_failed'`` -- the protocol's steps fail the
            pre-run validation; the message names the errors.
            ``'composite_needs_two_channels'`` -- a composite run has fewer
            than two channels set to capture. ``'tiling_unknown'`` -- the
            tile grid asked for is not one this installation offers.
            ``'tiling_scale_unknown'`` -- the scope cannot report its pixel
            size, so a tile grid's spacing is unknown.
            ``'zstack_not_configured'`` -- a z-stack's range or step size is
            not greater than zero. ``'already_tiled'`` -- the protocol is
            tiled already, and a second grid cannot be laid over it.
            ``'tiles_outside_travel'`` -- tiles of the grid fall outside the
            stage's travel. ``'zslices_outside_travel'`` -- z-slices of the
            stack fall outside Z's travel.
            ``'positions_outside_travel'`` -- steps of the protocol fall
            outside the stage's travel.
            ``'camera_setting_out_of_range'`` -- a step's gain or exposure is
            outside this camera's range. ``'positions_unreachable'`` -- the
            work needs motion on an axis this scope has no motor for.
            ``'step_position_missing'`` -- steps have no position on an axis
            this scope moves, having been recorded on a scope without it.
            ``'layer_not_on_scope'`` -- steps take or stimulate a layer this
            scope does not have, or this scope's layers could not be
            resolved. ``'objectives_require_turret'`` -- the protocol
            changes objective between steps, and this scope has no turret
            to change it. ``'objective_not_given'`` -- a protocol with a
            channel set to acquire was built from a configuration that
            names no objective. ``'focus_not_given'`` -- a layer has no
            saved focus, and the configuration names no current Z to image
            it at. ``'overlap_out_of_range'`` -- the tile overlap is outside
            0 to 50 percent. ``'values_out_of_range'`` -- layer settings the
            protocol would store hold values no layer setting takes; the
            message names each.
        title: The heading shown above the sentence, the reason's.
        message: The sentence; the same text as the exception's message.
        holder: What holds the microscope at refusal time (the
            exclusive-activity claim's kind, or ``AUTOFOCUS`` for a sweep
            in flight), or None when the refusal is not holder-shaped
            (validation, hardware, file drain).
        holder_trigger: Who started the run that holds the scope --
            ``'gui'``, ``'rest'`` or ``'api'``
            (``modules.caller``); for a file-drain refusal the just-finished
            run's, for an autofocus sweep the run that dispatched it. None
            when no run is behind the holder -- a recording's kind IS the
            holder.
        remedy: The action that answers this refusal, or None. A stalled
            file writer has one (recover it); a writer still making
            progress does not, since its files will land.

    ``holder`` and ``holder_trigger`` are published, so a REST problem
    carries who holds the scope beside the words.
    """

    holder: Holder | None
    holder_trigger: Caller | None

    # What a protocol holds is the request's, whoever made it: a client
    # changes the protocol, and the scope's state never makes it runnable.
    # So are a stage's travel, a camera's range and hardware the scope does
    # not have, which no wait changes, and a stop naming an ended run.
    causes: ClassVar[dict[str, RefusalCause]] = {
        'already_running': RefusalCause.STATE,
        'exclusive_activity_running': RefusalCause.STATE,
        'recording_active': RefusalCause.STATE,
        'files_writing': RefusalCause.STATE,
        'files_writing_stalled': RefusalCause.STATE,
        'post_processing_running': RefusalCause.STATE,
        'autofocus_running': RefusalCause.STATE,
        'not_connected': RefusalCause.RESTART,
        'camera_not_streaming': RefusalCause.STATE,
        'position_unknown': RefusalCause.STATE,
        'lid_open_before_motion': RefusalCause.STATE,
        'capture_location_unusable': RefusalCause.STATE,
        'no_acquiring_layer': RefusalCause.STATE,
        'channel_already_in_add': RefusalCause.REQUEST,
        'composite_two_transmitted_layers': RefusalCause.REQUEST,
        'turret_slot_ambiguous': RefusalCause.STATE,
        'turret_slot_reassigned': RefusalCause.STATE,
        'step_position_unknown': RefusalCause.STATE,
        'turret_objectives_unassigned': RefusalCause.STATE,
        'objective_not_mounted': RefusalCause.STATE,
        'sequence_name_invalid': RefusalCause.REQUEST,
        'empty_protocol': RefusalCause.REQUEST,
        'validation_failed': RefusalCause.REQUEST,
        'composite_needs_two_channels': RefusalCause.REQUEST,
        'tiling_unknown': RefusalCause.REQUEST,
        'tiling_scale_unknown': RefusalCause.REQUEST,
        'zstack_not_configured': RefusalCause.REQUEST,
        'already_tiled': RefusalCause.REQUEST,
        'objective_not_in_catalogue': RefusalCause.REQUEST,
        'tiles_outside_travel': RefusalCause.REQUEST,
        'zslices_outside_travel': RefusalCause.REQUEST,
        'positions_outside_travel': RefusalCause.REQUEST,
        'camera_setting_out_of_range': RefusalCause.REQUEST,
        'positions_unreachable': RefusalCause.REQUEST,
        'step_position_missing': RefusalCause.REQUEST,
        'layer_not_on_scope': RefusalCause.REQUEST,
        'channel_unknown': RefusalCause.REQUEST,
        'objectives_require_turret': RefusalCause.REQUEST,
        'objective_not_given': RefusalCause.REQUEST,
        'focus_not_given': RefusalCause.REQUEST,
        'overlap_out_of_range': RefusalCause.REQUEST,
        'values_out_of_range': RefusalCause.REQUEST,
    }
    same_fact_as = (
        LiveFolderPathRefusedError,
        CatalogueNameRefusedError,
    )

    # Keyed as ``causes`` is. The three step refusals share one heading: the
    # step was not saved, whether it was being added or updated.
    _TITLES: ClassVar[dict[str, str]] = {
        'already_running': 'Already Running',
        'exclusive_activity_running': 'Another Activity Running',
        'recording_active': 'Recording Active',
        'files_writing': 'Files Still Writing',
        'files_writing_stalled': ENDING_TITLES['file_writer_stalled'],
        'post_processing_running': POST_PROCESSING_RUNNING_TITLE,
        'autofocus_running': 'Autofocus Running',
        'not_connected': 'Hardware Disconnected',
        'camera_not_streaming': CAMERA_NOT_STREAMING_TITLE,
        'position_unknown': 'Scope Not Homed',
        'lid_open_before_motion': INTERLOCK_TITLES['lid_open'],
        'capture_location_unusable': 'Save Location Unusable',
        'no_acquiring_layer': 'No Channel Set to Acquire',
        'channel_already_in_add': 'Step Not Saved',
        'composite_two_transmitted_layers': 'One Transmitted Channel',
        'turret_slot_ambiguous': 'Turret Slot Not Chosen',
        'turret_slot_reassigned': 'Turret Slot Reassigned',
        'step_position_unknown': 'Step Not Saved',
        'turret_objectives_unassigned': 'Turret Configuration Required',
        'objective_not_mounted': 'Objective Not Mounted',
        'sequence_name_invalid': 'Run Name Not Valid',
        'empty_protocol': 'No Steps',
        'validation_failed': 'Validation failed',
        'composite_needs_two_channels': 'Not Enough Channels',
        'tiling_unknown': 'Tiling Not Available',
        'tiling_scale_unknown': 'Tiling Not Available',
        'zstack_not_configured': 'Z-Stack Not Configured',
        'already_tiled': 'Protocol Already Tiled',
        'objective_not_in_catalogue': CatalogueNameRefusedError._WORDS[
            'objective_not_in_catalogue'
        ][0],
        'tiles_outside_travel': 'Tiles Outside Stage Travel',
        'zslices_outside_travel': 'Z-Stack Outside Z Travel',
        'positions_outside_travel': 'Positions Outside Stage Travel',
        'camera_setting_out_of_range': 'Camera Setting Out of Range',
        'positions_unreachable': 'Position Not Reachable',
        'step_position_missing': 'Step Has No Position',
        'layer_not_on_scope': 'Layer Not on This Scope',
        'channel_unknown': 'Channel Not Known',
        'objectives_require_turret': 'One Objective At A Time',
        'objective_not_given': 'No Objective Given',
        'focus_not_given': 'No Focus Given',
        'overlap_out_of_range': 'Tile Overlap Out of Range',
        'values_out_of_range': 'Protocol Values Out of Range',
    }

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        holder: 'Holder | None' = None,
        holder_trigger: Caller | None = None,
        remedy: Remedy | None = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.title = self._TITLES[reason]
        self.message = message
        self.holder = _holder(holder)
        self.holder_trigger = holder_trigger
        self.remedy = remedy


class RunCheckFailedError(Fault, ProtocolError):
    """A run could not be checked before it started: a check itself crashed.

    Raised by SequencedCaptureRunner.prepare() when validating the protocol,
    reading whether the camera is streaming, or reading the stage's
    interlocks, raised instead of answering. Not a refusal: nothing was
    declined -- the question could not be asked, and the crash that stopped
    it is chained as ``__cause__`` so its traceback is logged with this.
    Nothing is committed and nothing needs unwinding, as for a refusal. Not
    reported where it is raised: the caller that asked for the run reports
    it where its flight ends.

    Attributes:
        reason: Machine-readable cause. ``'validation_crashed'`` -- the
            protocol's pre-run validation raised before it produced an
            answer. ``'hardware_state_unknown'`` -- a hardware read the
            check needs, whether the camera is streaming or which of the
            stage's interlocks are open, raised instead of answering.
        title: Short heading for the user, the reason's.
        message: The sentence a user reads; the same text as the
            exception's message.
    """

    _TITLES: ClassVar[dict[str, str]] = {
        'validation_crashed': 'Cannot validate protocol',
        'hardware_state_unknown': 'Cannot verify hardware state',
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_TITLES)

    def __init__(self, reason: str, message: str):
        if reason not in self._TITLES:
            raise TypeError(f'RunCheckFailedError has no reason {reason!r}')
        super().__init__(message)
        self.reason = reason
        self.title = self._TITLES[reason]
        self.message = message


class RunStartError(Fault, ProtocolError):
    """A sequenced run failed after it was committed but before it ran.

    The counterpart of ProtocolRunRefusedError on the other side of the
    commit line: a refusal means nothing started and nothing needs
    unwinding, while this means the run was committed, the terminal
    callback will fire and cleanup will run. Carries the same three
    fields so both sides deliver one shape to a caller, a REST handler
    and the popup.

    Attributes:
        reason: Machine-readable cause, a run ending's reason, as
            ``RunFailedError`` explains it.
        title: Short heading for the user, the run ending's.
        message: The sentence a user reads -- never a raw exception
            string; those belong in the log.
    """

    _TITLES: ClassVar[dict[str, str]] = ENDING_TITLES
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(FAILURE_REASONS)
    same_fact_as = (CameraReadFailedError,)

    def __init__(self, reason: str, message: str):
        if reason not in self._TITLES:
            raise TypeError(f'RunStartError has no reason {reason!r}')
        super().__init__(f'{reason}: {message}')
        self.reason = reason
        self.title = self._TITLES[reason]
        self.message = message


@api_fields('holder', 'holder_trigger')
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
            refusals to responses (REST status codes, UI branches). A
            reason it shares with another refusal is explained there: with
            ``ProtocolRunRefusedError``, and ``'capture_location_unusable'``
            and ``'names_exhausted'`` with ``CaptureLocationError``. Its
            own: ``'camera_exposure_unknown'`` -- the camera has not
            reported its exposure time, from which the recording's
            dead-feed bound derives. ``'insufficient_disk'`` -- free disk
            at the save location is below the floor a recording needs to
            start. ``'falsifying_change_in_flight'`` -- an objective,
            camera frame or light change is under way, which a recording
            would capture through. ``'other_led_lit'`` -- an LED other than
            the shown layer's is lit, and the recording is named for the
            shown layer. ``'camera_holds_another_layer'`` -- the camera
            holds another layer's settings than the shown layer's, which
            the recording is named for.
        title: Short user-facing refusal title, the reason's.
        message: One-sentence user-facing refusal body; the same text as
            the exception's message.
        holder: The exclusive-activity claim owner at refusal time, or
            None when the refusal is not claim-shaped.
        holder_trigger: Who started the holding run (``'gui'``,
            ``'rest'``, ``'api'``) when the holder is 'protocol'; None for
            a recording holder.

    ``holder`` and ``holder_trigger`` are published.
    """

    holder: Holder | None
    holder_trigger: Caller | None

    _TITLES: ClassVar[dict[str, str]] = {
        'recording_active': ProtocolRunRefusedError._TITLES['recording_active'],
        'exclusive_activity_running': ProtocolRunRefusedError._TITLES['exclusive_activity_running'],
        'post_processing_running': POST_PROCESSING_RUNNING_TITLE,
        'camera_exposure_unknown': 'Camera Exposure Unavailable',
        'capture_location_unusable': 'Cannot Save Recording',
        'names_exhausted': 'Cannot Save Recording',
        'insufficient_disk': 'Insufficient Disk Space',
        'falsifying_change_in_flight': 'Microscope Changing',
        'other_led_lit': 'Another Light Is On',
        'camera_holds_another_layer': 'Camera On Another Layer',
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_TITLES, RefusalCause.STATE)
    same_fact_as = (ProtocolRunRefusedError,)

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        holder: 'Holder | None' = None,
        holder_trigger: Caller | None = None,
    ):
        if reason not in self._TITLES:
            raise TypeError(f'RecordingRefusedError has no reason {reason!r}')
        super().__init__(message, reason)
        self.title = self._TITLES[reason]
        self.message = message
        self.holder = _holder(holder)
        self.holder_trigger = holder_trigger


class HyperstackRefusedError(Fault, CaptureError):
    """The hyperstack builder declined to build a recording's frames into one file.

    Raised by the manual recording's finish when the builder answers
    status=False, or raises ``InputFramesUnusableError`` because a frame
    cannot be read or cannot share the stack: the frames are on disk as
    recorded.
    The builder's own sentence rides in ``message`` so the notification the
    user sees says why, in the builder's words, rather than telling them to
    read a log.

    Attributes:
        message: The builder's one-paragraph reason, user-facing.
        reason: ``'hyperstack_refused'`` -- the builder answered status=False,
            or the frames are not all one size;
            ``'unreadable_input_frame'`` -- a frame could not be read;
            ``'mixed_input_pixel_types'`` -- frames of different pixel types
            cannot share one stack.
    """

    title = 'Hyperstack Not Built'
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(
        ('hyperstack_refused', 'unreadable_input_frame', 'mixed_input_pixel_types')
    )

    def __init__(self, message: str, reason: str = 'hyperstack_refused'):
        super().__init__(message, reason)
        self.message = message


class InputFramesUnusableError(Refusal, CaptureError, ValueError):
    """Frames handed to an image operation cannot be used, naming each frame.

    Raised by the image reader for a frame that is missing or will not
    decode, by the canvas check for frames of different sizes, and by the
    hyperstack builder for frames whose pixel types cannot share one stack.
    A post-processing build answers it as the one group's failure and goes
    on with the others, so a damaged frame costs its group, not the run's
    other outputs. A ``ValueError``, so a caller that skips a frame it
    cannot read keeps doing so.

    Attributes:
        reason: ``'unreadable'`` -- a frame is missing or will not decode;
            ``'geometry'`` -- the frames are not all one size;
            ``'mixed_pixel_types'`` -- the frames' pixel types cannot share
            one output.
    """

    _TITLES: ClassVar[dict[str, str]] = {
        'unreadable': 'Image Not Readable',
        'geometry': 'Images Not One Size',
        'mixed_pixel_types': 'Images Not One Pixel Type',
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_TITLES, RefusalCause.REQUEST)

    def __init__(self, message: str, reason: str):
        if reason not in self._TITLES:
            raise TypeError(f'InputFramesUnusableError has no reason {reason!r}')
        super().__init__(message, reason)
        self.title = self._TITLES[reason]


class PostProcessingOperation(StrEnum):
    """Every post-processing operation, by the name a person reads.

    A post-processing outcome's title is keyed by the operation and its
    reason, so the operations are a closed set: an operation outside it fails
    where its outcome is built.
    """

    COMPOSITE = 'Composite'
    STITCH = 'Stitch'
    Z_PROJECTION = 'Z-Projection'
    VIDEO = 'Video'
    HYPERSTACK = 'Hyperstack'
    CELL_COUNT = 'Cell Count'
    GRAPHING = 'Graphing'
    TRENDLINE = 'Trendline'
    QUICK_ENHANCE = 'Quick Enhance'
    # Several builds in one call (PostProcessingAPI.build), for its own refusals.
    POST_PROCESSING = 'Post-Processing'


class PostProcessingRefusedError(Refusal, CaptureError):
    """What a post-processing operation was given cannot yield the output asked of it.

    Nothing broke: the folder holds no images, no groups this operation can
    combine, only derived outputs, only groups whose outputs would share a
    name, source images in a format the operation cannot re-read, protocol
    data that could not be loaded, or outputs the folder's protocol would
    name outside the folder; or the cell-count method it was
    given cannot be used. The message says which, in words written for the
    person, and what to do. Or the scope's state refused it: a build
    started while something holds the scope or a run's files are still
    being written would share the process's cores, interpreter lock and
    disk with that work (``'scope_held'``, ``'files_writing'``), and the
    same request may succeed once it ends.

    Attributes:
        operation: The operation's name as a person reads it ("Stitch").
        reason: Machine-readable refusal code.
            ``'axis_not_plottable'`` -- a graph's X or Y names a column of
            the results file that axis cannot take. ``'collision'`` -- every
            image group would write an output under a name another group's
            output shares, so nothing could be built.
            ``'excluded_inputs'`` -- the folder holds only derived outputs,
            which the build does not take as input. ``'fit_impossible'`` --
            the graphed values cannot give the trendline asked for.
            ``'graph_incomplete'`` -- a graph was asked for without a results
            file, an X axis and a Y axis. ``'invalid_path'`` -- no folder
            was given to process. ``'invalid_setting'`` -- a setting of the
            build is not one it can use, such as no build chosen, an image
            shape or scale that is not one, or a playback rate below 1 frame
            per second. ``'method_invalid'`` -- the cell-count method holds
            something the count cannot use. ``'method_unreadable'`` -- the
            cell-count method file cannot be read, is not JSON, or has no
            metadata naming its type and version. ``'no_data'`` -- the
            folder holds images, but not the data this operation needs: no
            Z-stack for a z-projection, no recording frames for a video,
            nothing a count could count, or a recording where stills are
            needed. ``'no_images'`` -- the folder holds no image files.
            ``'no_scale'`` -- the image states no scale, so it is counted in
            pixels, and the method filters a size in microns.
            ``'output_outside_folder'`` -- the folder's protocol names a
            step whose output would be written outside the folder.
            ``'protocol_data_unreadable'`` -- the folder's protocol data
            could not be loaded. ``'results_unreadable'`` -- the cell-count
            results file cannot be read as one LumaViewPro writes.
            ``'unsupported_source_format'`` -- the source images are not
            TIFF or OME-TIFF, which the build re-reads. ``'scope_held'`` --
            something holds the scope, and a build now would share the
            computer with its work. ``'files_writing'``, as
            ``ProtocolRunRefusedError`` explains it, and ``'unreadable'``, as
            ``InputFramesUnusableError`` explains it, are the same facts.
    """

    causes: ClassVar[dict[str, RefusalCause]] = {
        'axis_not_plottable': RefusalCause.REQUEST,
        'collision': RefusalCause.REQUEST,
        'excluded_inputs': RefusalCause.REQUEST,
        'fit_impossible': RefusalCause.REQUEST,
        'graph_incomplete': RefusalCause.REQUEST,
        'invalid_path': RefusalCause.REQUEST,
        'invalid_setting': RefusalCause.REQUEST,
        'method_invalid': RefusalCause.REQUEST,
        'method_unreadable': RefusalCause.REQUEST,
        'no_data': RefusalCause.REQUEST,
        'no_images': RefusalCause.REQUEST,
        'no_scale': RefusalCause.REQUEST,
        'output_outside_folder': RefusalCause.REQUEST,
        'protocol_data_unreadable': RefusalCause.REQUEST,
        'results_unreadable': RefusalCause.REQUEST,
        'unreadable': RefusalCause.REQUEST,
        'unsupported_source_format': RefusalCause.REQUEST,
        'scope_held': RefusalCause.STATE,
        'files_writing': RefusalCause.STATE,
    }
    same_fact_as = (
        InputFramesUnusableError,
        ProtocolRunRefusedError,
    )

    # Refused for lack of Z-stack data, a z-projection names the thing the
    # folder is missing and where such a folder lives, rather than leaving
    # the person to guess which folder would have worked.
    _NO_ZSTACK_ADVICE = (
        ' Pick a folder that contains a Z-stack run -- look under '
        "'Manual/Z-Stacks/<timestamp>/' for a manual Z-stack, or a "
        "'ProtocolData/<timestamp>/' folder whose protocol included Z-stack steps."
    )

    # Each operation's heading and the advice its words end with, and the one
    # reason that names its own: a z-projection refused for want of Z-stack
    # data says so, and where such a folder lives. A row carries its own
    # advice, so a heading added for another reason says only what it names.
    _WORDS: ClassVar[dict[object, tuple[str, str]]] = {
        **{operation: (f'{operation} Not Possible', '') for operation in PostProcessingOperation},
        (PostProcessingOperation.Z_PROJECTION, 'no_data'): (
            'No Z-Stack Data Found',
            _NO_ZSTACK_ADVICE,
        ),
    }

    def __init__(self, *, operation: str, reason: str, message: str):
        self.reason = reason
        operation = PostProcessingOperation(operation)
        key = (operation, reason) if (operation, reason) in self._WORDS else operation
        self.title, advice = self._WORDS[key]
        super().__init__(f'{message}{advice}', reason)
        self.operation = operation


@api_fields('kind', 'count')
@dataclass(frozen=True)
class ExcludedInput:
    """Files of one kind a post-processing build found in the folder and did not take.

    A build's own earlier outputs are not among them: those it has already
    made.

    Attributes:
        kind: The build that made the files: ``COMPOSITE``, ``STITCH``,
            ``Z_PROJECTION``, ``VIDEO`` or ``HYPERSTACK``. A recording's
            frames, which only a video or a hyperstack takes, are never
            among them.
        count: How many.
    """

    kind: PostProcessingOperation
    count: int


@api_fields('excluded_inputs')
class PostProcessingFailedError(Fault, CaptureError):
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
        excluded_inputs: The files of each kind the build did not take, as
            a complete build's result names them; empty for an operation
            that reads no protocol folder (Quick Enhance, a cell count).
        reason: ``'post_processing_failed'`` -- the build produced nothing.
            ``'post_processing_incomplete'`` -- the build produced part of
            what it was asked for, named in ``produced_paths``.
    """

    excluded_inputs: tuple[ExcludedInput, ...]

    _TITLES: ClassVar[dict[tuple[PostProcessingOperation, str], str]] = {
        (operation, reason): f'{operation} {heading}'
        for operation in PostProcessingOperation
        for reason, heading in (
            ('post_processing_failed', 'Failed'),
            ('post_processing_incomplete', 'Incomplete'),
        )
    }

    causes: ClassVar[dict[str, RefusalCause | None]] = {
        'post_processing_failed': None,
        'post_processing_incomplete': None,
    }

    def __init__(
        self,
        *,
        operation: str,
        missing: str,
        produced_paths: Iterable[str] = (),
        output_root: 'str | None' = None,
        errors: Iterable[str] = (),
        excluded_inputs: 'Iterable[ExcludedInput]' = (),
    ):
        produced_paths = tuple(produced_paths)
        if produced_paths:
            saved = f'{len(produced_paths)} output(s) were saved to {output_root}.'
        else:
            saved = 'Nothing was saved.'
        operation = PostProcessingOperation(operation)
        reason = 'post_processing_incomplete' if produced_paths else 'post_processing_failed'
        super().__init__(f'{missing} {saved} Check the log for the cause.', reason)
        self.operation = operation
        self.produced_paths = produced_paths
        self.output_root = output_root
        self.errors = tuple(errors)
        self.excluded_inputs = tuple(excluded_inputs)
        self.title = self._TITLES[(operation, reason)]


class RecordingStoppedError(Fault, CaptureError):
    """A manual recording was stopped before anyone asked it to stop.

    Built by the recording's own watch when the camera stops delivering
    frames or the free disk falls below the floor, and reported where it is
    detected: no caller is waiting on a recording that is running. The
    frames written so far are on disk.

    Attributes:
        reason: ``'camera_disconnected'`` -- the camera is no longer active.
            ``'camera_stalled'`` -- the camera stopped delivering frames for
            longer than the recording's exposure allows.
            ``'disk_floor'`` -- free disk fell below the safety floor.
    """

    _TITLES: ClassVar[dict[str, str]] = {
        'camera_disconnected': 'Recording Stopped',
        'camera_stalled': 'Recording Stopped',
        'disk_floor': 'Recording Stopped -- Disk Almost Full',
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_TITLES)
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


class RecordingFinalizeError(Fault, CaptureError):
    """A recording finished but its output could not be fully assembled.

    Raised around what failed in the finish of a manual recording or of a
    protocol video step, and chained from it. The frames already written
    are on disk.

    Attributes:
        reason: ``'video_step_finalize_failed'`` -- a protocol video step's
            output could not be fully assembled.
            ``'recording_finalize_failed'`` -- a manual recording's output
            could not be fully assembled.
        protocol_step: True for a protocol video step, False for a manual
            recording; each is named in its own words.
    """

    _TITLES: ClassVar[dict[str, str]] = {
        'video_step_finalize_failed': 'Video Finalize Failed',
        'recording_finalize_failed': 'Recording Finalize Failed',
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_TITLES)

    def __init__(self, *, protocol_step: bool):
        if protocol_step:
            message = (
                'A video step finished but its output could not be fully assembled. '
                'Frames already written are on disk; check the log.'
            )
            reason = 'video_step_finalize_failed'
        else:
            message = (
                'The recording finished but its output could not be fully assembled. '
                'Frames already written are on disk; check the log.'
            )
            reason = 'recording_finalize_failed'
        super().__init__(message, reason)
        self.title = self._TITLES[reason]
        self.protocol_step = protocol_step


class VideoFramesDroppedError(Fault, CaptureError):
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
    causes: ClassVar[dict[str, RefusalCause | None]] = {'video_frames_dropped': None}

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


class VideoWriterFailedError(Fault, CaptureError):
    """The recording's video writer stopped working, so the recording ended.

    Chained from what escaped the writer. The frames already written are on
    disk. On a protocol video step the run is stopped too, and the run's
    ending carries these same words, so a caller reading the run's outcome
    reads what the person was shown.

    Attributes:
        protocol_step: True for a protocol video step, False for a manual
            recording; each is named in its own words.
    """

    title = ENDING_TITLES['video_writer_died']
    causes: ClassVar[dict[str, RefusalCause | None]] = {'video_writer_died': None}
    same_fact_as = (RunStartError,)

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


class RecordingDetailsNotSavedError(Fault, CaptureError):
    """A recording's details file could not be written; its frames are whole.

    Chained from the write's error. The details file is the only record of
    the recording's channel color and measured rate, so a video built from
    these frames later falls back to grayscale and a default rate.
    """

    title = 'Recording details not saved'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'recording_details_not_saved': None}

    def __init__(self):
        super().__init__(
            'The video frames are safe on disk, but the recording details file '
            'could not be written. Videos built from this recording may be '
            'grayscale and use a default frame rate; check disk space and the log.',
            'recording_details_not_saved',
        )


class ImageSaveError(Fault, CaptureError):
    """An image could not be written to disk.

    Raised by ``save_image`` from the ``OSError`` of the write -- a missing
    or read-only folder, a full disk, a denied permission -- so the words
    point at the disk. A failure before the write (encoding, metadata)
    is not this and propagates as itself.

    Attributes:
        file_loc: The path the image was to be written to.
    """

    title = 'Image Save Failed'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'image_save_failed': None}

    def __init__(self, file_loc):
        super().__init__(
            f'Failed to save image to {file_loc}. Check disk space and permissions.',
            'image_save_failed',
        )
        self.file_loc = file_loc


class RunFilesNotWrittenError(Fault, CaptureError):
    """A build that reads a run's images back found them not all written.

    Raised by the wait a post-run build makes on its run's writes -- the
    composite merge and the hyperstack build -- so neither builds from a
    folder that is still filling or that lost images: the artifact would be
    silently incomplete. Nothing is built.

    Attributes:
        reason: ``'write_batch_timeout'`` -- the run's writes did not finish
            within the build's bound. ``'write_batch_abandoned'`` -- some of
            the run's writes never ran: the file writer was recovered, or
            the app shut down, while they were outstanding.
            ``'write_batch_not_taken'`` -- some of the run's images never
            reached the file writer: it was stuck, or had stopped taking
            work, when they were handed over. ``'write_batch_save_failed'``
            -- saving some of the run's images failed on disk.
            ``'write_batch_disk_full'`` -- some of the run's images were
            refused because the save drive was nearly full.
            ``'write_batch_video_unfinished'`` -- a video step's file did
            not finish. ``RunImagesNotSavedError`` carries all but the
            first as the same facts.
    """

    title = 'Run Images Not Written'
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(
        (
            'write_batch_timeout',
            'write_batch_abandoned',
            'write_batch_not_taken',
            'write_batch_save_failed',
            'write_batch_disk_full',
            'write_batch_video_unfinished',
        )
    )

    def __init__(self, reason: str, *, bound_s: float | None = None):
        self.reason = reason
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
        else:
            message = (
                "A video step's file did not finish, so nothing was built from "
                'the incomplete folder. Check the log for why the video stopped.'
            )
        super().__init__(message, reason)


class RunImagesNotSavedError(Fault, CaptureError):
    """Images a run captured are not on disk.

    Reported once, when the run's last write lands -- the one moment the
    count is known, since a run's images keep landing after it ends.

    Attributes:
        reason: Why images are not on disk, as ``RunFilesNotWrittenError``
            explains it.
        not_written: Images the run captured that are not on disk.
        written: Images that landed.
    """

    title = 'Run Images Not Saved'
    _WORDS: ClassVar[dict[str, str]] = {
        'write_batch_abandoned': 'the file writer was recovered or the app shut down',
        'write_batch_not_taken': 'the file writer was stuck',
        'write_batch_save_failed': 'saving failed on disk',
        'write_batch_disk_full': 'the save drive was nearly full',
        'write_batch_video_unfinished': "a video step's file did not finish",
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_WORDS)
    same_fact_as = (RunFilesNotWrittenError,)

    def __init__(self, *, written: int, not_written: int, reason: str):
        self.reason = reason
        super().__init__(
            f'{not_written} of the images this run captured '
            f'{"is" if not_written == 1 else "are"} not on disk ({self._WORDS[reason]}); '
            f'{written} were saved.',
            reason,
        )
        self.written = written
        self.not_written = not_written


class RunCleanupFailedError(Fault, CaptureError):
    """Steps that put the scope back after a run did not finish.

    The run's own ending stands; what failed is the restore after it, so
    the LEDs, the camera's gain and exposure or the stage may not be where
    the person expects. The run's outcome names the steps too.

    Attributes:
        steps: Each failed step's name, in the order they failed.
    """

    title = 'Protocol cleanup issues'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'cleanup_failed': None}

    def __init__(self, failures: list[tuple[str, str]]):
        lines = '\n'.join(f'  - {step}: {detail}' for step, detail in failures)
        # "ended", not "completed": a stopped or failed run is cleaned up too.
        super().__init__(
            f'Protocol ended but {len(failures)} cleanup step(s) failed:\n{lines}\n'
            'Check LED state, camera settings, and stage position.',
            'cleanup_failed',
        )
        self.steps = [step for step, _ in failures]


class RecordIncompleteError(Fault, CaptureError):
    """Captures a run made are missing from its execution record.

    Post-processing reads the record to find a run's images, so an image
    with no row is skipped by stitching and video builds.

    Attributes:
        missing: Captures with no row.
        attempted: Captures the run attempted.
    """

    title = 'Protocol Record Incomplete'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'record_incomplete': None}

    def __init__(self, *, missing: int, attempted: int, record_name: str):
        super().__init__(
            f'{missing} of {attempted} captures were not written to the protocol '
            f'record ({record_name}). Those images, if saved, will be missing from '
            'stitching and video builds. Check the log for the cause.',
            'record_incomplete',
        )
        self.missing = missing
        self.attempted = attempted


class RunEndedByStopNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
    """STOP was pressed while a run was live, so the run ended.

    The run's outcome carries it as status ``'aborted'``, reason
    ``'stage_stopped'``, in these words, and the person is told once when the
    run ends: the run's owner may not be the one who pressed STOP (a REST
    client, the GUI's button). The words say what was pressed, which is true
    whether or not the board took the STOP; a STOP it did not take is
    reported as that failure, where it was pressed.
    """

    title = ENDING_TITLES['stage_stopped']
    causes: ClassVar[dict[str, RefusalCause | None]] = {'stage_stopped': None}

    def __init__(self):
        super().__init__('STOP was pressed, so the run ended.')


class RunIncompleteError(Fault, CaptureError):
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

    title = ENDING_TITLES['captures_failed']
    causes: ClassVar[dict[str, RefusalCause | None]] = {'captures_failed': None}
    same_fact_as = (RunStartError,)
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


class RunFailedError(Fault, CaptureError):
    """The instrument ended a run: the fault that ended it, in its own words.

    Fatal: the person is told even during an unattended run, since the run
    they left is no longer running. The title and words are the run's
    ending's, so a client reading the run's outcome and one hearing this
    read the same.

    Attributes:
        title: The ending's heading, its reason's (``ENDING_TITLES``).
        reason: The ending's machine-readable cause, a run ending's reason;
            ``RunStartError`` and ``RunFailedToStartError`` carry the same.
            ``'captures_failed'`` -- the run reached its end without every
            capture it was asked for. ``'start_failed'`` -- the run's setup
            raised a fault of no other reason after the run was committed.
            ``'dispatch_refused'`` -- the run loop was refused when it was
            dispatched, so it never ran. ``'capture_location_failed'`` --
            the run's folder could not be made at the capture location.
            ``'run_dir_init_failed'`` -- the run's folder was made and could
            not be initialized. ``'camera_lane_stalled'`` -- a camera
            command did not finish, so the run could not take the camera.
            ``'camera_read_failed'`` -- the camera's settings could not be
            read when the run took its snapshot of them.
            ``'run_loop_crashed'`` -- the run loop raised, or stopped
            without saying how the run ended.
            ``'hardware_disconnected'`` -- a part the run needs was found
            disconnected. ``'consecutive_scan_failures'`` -- the scan failed
            too many times in a row with the hardware still present.
            ``'interlock'`` -- the stage's interlock refused a move during a
            scan. ``'position_lost'`` -- an axis lost its position during a
            scan, and nothing in a run homes it again.
            ``'disk_space_critical'`` -- free disk at the save location fell
            below what the run needs, so the run stopped to save what it
            has. ``'file_writer_stalled'`` -- saving stopped making progress,
            so the run stopped before it lost more captures.
            ``'video_writer_died'`` -- the writer of a video step failed.
            ``'led_channel_unavailable'`` -- steps of the run take an LED
            channel this scope does not have, and their captures kept
            failing. ``'camera_failure'`` -- the camera failed too many steps
            in a row. ``'autofocus_failed'`` -- a single-autofocus run's
            autofocus did not find focus. ``'autofocus_not_started'`` -- a
            single-autofocus run's autofocus was refused before the stage
            moved, its first window reaching past Z's travel.
    """

    fatal = True

    _TITLES: ClassVar[dict[str, str]] = ENDING_TITLES
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(FAILURE_REASONS)
    same_fact_as = (RunStartError,)

    def __init__(self, *, reason: str, message: str):
        super().__init__(message, reason)
        self.title = self._TITLES[reason]


class RunFailedToStartError(Fault, CaptureError):
    """A run was committed to and did not start, in its ending's words.

    Not fatal: it is reported after the run's cleanup has lifted the
    unattended mute, so the person who started it sees it.

    Attributes:
        title: The ending's heading, its reason's (``ENDING_TITLES``).
        reason: The ending's machine-readable cause, as ``RunFailedError``
            explains it.
    """

    _TITLES: ClassVar[dict[str, str]] = ENDING_TITLES
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(FAILURE_REASONS)
    same_fact_as = (RunStartError,)

    def __init__(self, *, reason: str, message: str):
        super().__init__(message, reason)
        self.title = self._TITLES[reason]


@api_fields('run_ending', 'merge_failure')
class CompositeFailedError(Fault, CaptureError):
    """A run's composite was not merged; the run's own images stand.

    Its reasons are its own: what the run ended with, or what the merge
    raised, is a field, since each is another vocabulary's string.

    Attributes:
        reason: ``'run_not_completed'`` -- the run ended before a composite
            was made. ``'no_run_dir'`` -- the run has no folder to merge
            from. ``'no_composite_config'`` -- the run carries no composite
            thresholds. ``'merge_failed'`` -- the merge finished without
            writing a file. ``'merge_error'`` -- the merge failed.
        run_ending: The run's ending reason, for ``'run_not_completed'``;
            otherwise None.
        merge_failure: The reason of the outcome the merge failed with, for
            ``'merge_error'`` when it had one; otherwise None.
    """

    run_ending: str | None
    merge_failure: str | None
    title = 'Composite Failed'
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(
        ('run_not_completed', 'no_run_dir', 'no_composite_config', 'merge_failed', 'merge_error')
    )

    def __init__(
        self,
        message: str,
        reason: str,
        *,
        run_ending: str | None = None,
        merge_failure: str | None = None,
    ):
        super().__init__(message, reason)
        self.run_ending = run_ending
        self.merge_failure = merge_failure


class AutoGainNotSettledError(Fault, CaptureError):
    """Auto-gain locked with no usable exposure or gain from the camera.

    The previous settings were kept, so a capture taken with it had no
    exposure check.
    """

    title = 'Auto-gain did not settle'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'auto_gain_not_settled': None}

    def __init__(self):
        super().__init__(
            'The camera reported no usable exposure or gain when auto-gain was '
            'locked, so the previous settings were kept and any capture was taken '
            'without an exposure check. Check the live view, then try again.',
            'auto_gain_not_settled',
        )


class AutofocusFailedError(Fault, CaptureError):
    """An autofocus sweep chose no focus; the stage goes back to where it started.

    Attributes:
        reason: ``'flat_focus_curve'`` -- the final pass's curve has no peak:
            every score zero or invalid, a fit that cannot be made, or a
            curve with no contrast (every point above half its maximum)
            whose fit is not concave, peaks outside the measured range, or
            shifts past twice the step spacing. ``'travel_exhausted'`` -- a
            window the sweep reached after it moved the stage reaches past
            Z's travel; refused before the stage moves there.
            ``'unexpected_error'`` -- the sweep raised; its cause is
            chained.
    """

    title = ENDING_TITLES['autofocus_failed']
    _WORDS: ClassVar[dict[str, str]] = {
        'flat_focus_curve': 'Focus curve is flat or invalid -- check sample and illumination',
        'travel_exhausted': (
            'Autofocus would search past the end of Z travel -- start it further from the '
            "travel limit, or narrow the objective's autofocus range."
        ),
        'unexpected_error': 'Autofocus stopped on an unexpected error; the log has the details.',
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_WORDS)

    def __init__(self, reason: str):
        super().__init__(self._WORDS[reason], reason)


class AutofocusOutOfTravelError(Refusal, CaptureError):
    """An autofocus whose first window reaches past Z's travel, refused before the stage moves.

    The window is centred on where Z stands, so the same autofocus started
    further from the travel limit runs. A window the sweep reaches later,
    after the stage has moved, is ``AutofocusFailedError('travel_exhausted')``.

    Attributes:
        reason: ``'out_of_travel'``.
    """

    title = ENDING_TITLES['autofocus_not_started']
    causes: ClassVar[dict[str, RefusalCause | None]] = {'out_of_travel': RefusalCause.STATE}

    def __init__(self):
        super().__init__(
            'Autofocus would search past the end of Z travel -- start it further from the '
            "travel limit, or narrow the objective's autofocus range.",
            reason='out_of_travel',
        )


class AutofocusZNotRestoredError(Fault, CaptureError):
    """Autofocus stopped without a result and could not put a known Z back.

    The restore's target was refused, so Z is known, where the sweep parked
    it. A restore on a lost Z is not this: the fault that lost Z is told
    where it was found, and the run ends on the lost position.

    Attributes:
        reason: ``'z_left_at_search_position'``.
    """

    title = 'Z Position Not Restored'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'z_left_at_search_position': None}

    def __init__(self):
        super().__init__(
            'Could not restore Z position after autofocus stopped. '
            'Z was left at the last autofocus search position.',
            'z_left_at_search_position',
        )


class RunWriteRefusedError(Refusal, CaptureError):
    """A write was handed to a run whose writes have ended.

    A run's writes end when its cleanup closes them, or when a writer
    recovery or a shutdown abandons them. A write arriving after that
    belongs to no run: taken, it would count toward the next run's files or
    land after the finished run said its files were written. The caller
    says what was not saved. Nothing broke in declining it, and the same
    write into a live run is taken, so its cause is the state.

    Attributes:
        reason: ``'run_writes_ended'`` -- the run's cleanup closed its
            writes. ``'writes_abandoned'`` -- a writer recovery or a
            shutdown abandoned the run's remaining writes.
            ``'writer_shut_down'`` -- the file lane itself no longer takes
            work.
    """

    title = 'Not Saved'

    _WHY: ClassVar[dict[str, str]] = {
        'run_writes_ended': 'the run it belongs to has ended',
        'writes_abandoned': "the run's remaining writes were given up",
        'writer_shut_down': 'the file writer has shut down',
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_WHY, RefusalCause.STATE)

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

    causes: ClassVar[dict[str, RefusalCause | None]] = {'file_writer_not_stuck': RefusalCause.STATE}
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
        self.pending = pending


class CameraStreamStalledError(Fault, CaptureError):
    """The camera stopped delivering frames while it was not known to be removed.

    A camera's link or grab loop can stall without the device being removed:
    it reports itself connected and grabbing, and no frame arrives. Nothing
    is waiting on the stream when that happens, so the imaging API reports it
    when its stream check sees the frame count stand still for ten times the
    longest exposure the camera may be using (the sensor's maximum while the
    exposure is unknown), and never less than 5 s. A pulled cable meets the same
    check first when its driver reports the removal later than that (the IDS
    SDK reports it seconds after the pull), so the words never say the camera
    is still connected.

    Attributes:
        seconds: How long no new frame had arrived when the stall was seen.
    """

    title = 'Camera Not Delivering Frames'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'camera_stream_stalled': None}

    def __init__(self, seconds: float):
        super().__init__(
            f'The camera has delivered no new frame for {seconds:.0f} s. Check the USB '
            'cable and power connections; if frames do not resume, restart LumaViewPro.',
            'camera_stream_stalled',
        )
        self.seconds = seconds


class FileWriterStalledError(Fault, CaptureError):
    """A finished run's file writer stopped making progress on one of its files.

    Nobody is waiting on a run's files once it has ended, so the run engine
    reports the stall when it sees it, offering the recovery that answers it.
    The run refusal for the same stall makes the same offer: the sentences
    and the remedy below are the one wording of both.

    Attributes:
        remedy: Recovering the file writer, which gives up on the unsaved images.
    """

    title = ENDING_TITLES['file_writer_stalled']
    causes: ClassVar[dict[str, RefusalCause | None]] = {'file_writer_stalled': None}
    same_fact_as = (RunStartError,)

    def __init__(self, run: str, stuck: str, unsaved: int):
        super().__init__(
            f'{self.stalled_sentence(run, stuck)} Recovering the file writer unlocks '
            f'the app: {self.cost_sentence(unsaved)}',
            'file_writer_stalled',
        )
        self.remedy = self.recovery(unsaved)

    @staticmethod
    def stalled_sentence(run: str, stuck: str) -> str:
        """What stopped: *run* names the run, *stuck* the write in flight."""
        return f'{run} has stopped writing its files ({stuck}).'

    @staticmethod
    def cost_sentence(unsaved: int) -> str:
        """What recovering loses."""
        return (
            f'its {unsaved} unsaved image(s) will be lost, and a partial file from the '
            'stuck write may remain on disk, locked until that write releases it.'
        )

    @staticmethod
    def recovery(unsaved: int) -> Remedy:
        """The offer that answers the stall."""
        return Remedy(
            member=RemedyName.RECOVER_FILE_WRITER,
            confirm_text=f'Discard {unsaved} unsaved and unlock',
            cancel_text='Keep waiting',
        )


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
    causes: ClassVar[dict[str, RefusalCause | None]] = {
        'over_budget': RefusalCause.STATE,
        'raised': RefusalCause.STATE,
    }

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


class PluginError(_Umbrella, Exception):
    """A plugin failed: it did not load, or it failed after loading. The subject, never raised itself.

    Nothing raises these: the plugin host catches the plugin's own failure
    and has no caller to raise to, so each is built to be reported, chained
    from the plugin's exception when there is one so the log record carries
    that traceback. A plugin is separately versioned and may not be ours,
    so its failure is a fault for the person to hear about while the rest
    of the application carries on.

    Each kind's title is its class's; the words name the plugin.
    """

    words_for_person = True


class PluginNotLoadedError(Fault, PluginError):
    """Plugins found at discovery did not load, every one of them in one outcome.

    Discovery reports once, when it ends: the reporter merges notices that
    share a title within its window, so one outcome per plugin would show
    only the first of several failing together. The host chains every
    plugin's own exception, as an ``ExceptionGroup``, so the one log record
    carries each traceback.

    Attributes:
        failures: What did not load, each with its name and why, in the
            order discovery found them.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'plugin_not_loaded': None}
    title = 'Plugin Not Loaded'

    def __init__(self, failures: 'Sequence[PluginNotLoaded]'):
        super().__init__(
            ' '.join(f'The "{f.name}" plugin did not load: {f.reason}.' for f in failures)
            + ' The rest of LumaViewPro is unaffected.'
        )
        self.failures = tuple(failures)


class PluginFailedError(Fault, PluginError):
    """A loaded plugin failed while the host was calling it.

    Attributes:
        plugin_name: The plugin that failed.
        hook: What the host was calling the plugin for.
        detail: The plugin's own account of the failure, when it gave one.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'plugin_failed': None}
    title = 'Plugin Error'

    def __init__(self, plugin_name: str, hook: str, detail: str = ''):
        said = f' It said: {detail}' if detail else ''
        super().__init__(
            f'The "{plugin_name}" plugin failed ({hook}), so that action did not '
            f'complete. The rest of LumaViewPro is unaffected.{said}'
        )
        self.plugin_name = plugin_name
        self.hook = hook
        self.detail = detail


class PluginProcessorSkippedError(Fault, PluginError):
    """A run's post-processing by a plugin did not run: the plugin was unloaded first.

    The session's close unloads plugins before it waits for the work under
    way, because a plugin's ``unregister`` is what stops its own work; a
    run that finishes during that wait -- or after a host unloaded the
    plugins itself -- then hands its folder to processors no longer loaded. The folder and its images are complete; only the
    plugin's processing of it is missing, so the person is told which run
    to process again.

    Attributes:
        run: The run whose folder was not processed, by its protocol name
            or its folder.
        folder: The run's folder.
        plugin_name: The plugin whose processing did not run.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'processor_skipped': None}
    title = 'Post-Processing Skipped'

    def __init__(self, plugin_name: str, run: str, folder: str):
        super().__init__(
            f'The "{plugin_name}" plugin did not process the run "{run}": it had been '
            f"unloaded before the run finished, as LumaViewPro's close does first. The "
            f"run's images are all in {folder}; process them with the plugin when it is "
            'next loaded.',
        )
        self.plugin_name = plugin_name
        self.run = run
        self.folder = folder


@api_fields('holder', 'holder_trigger')
class HardwareCommandRefusedError(Refusal, Exception):
    """A hardware command was refused: something else has the scope, the lane is closed, or nothing is connected to take it.

    Raised to whoever made the command -- a public hardware member (LED,
    camera and motion commands), a raw task on a lane, the Session's
    objective writers -- and never dropped: a command refused without a
    raise reaches no hardware and reports success. While a run, a
    diagnostic or a home holds the scope, a lane refuses any task not made under
    the holder's taking with this; so do the run's own executor fences,
    which cannot say who closed them. While a control session is held, a
    lane refuses any task not made for its holder with
    ``ControlSessionRefusedError``, as a guarded write is refused.

    The Session's objective writers (select, slot assign and slot clear)
    raise it too while a run, a diagnostic or a home holds the scope: the run
    stamps the active objective's scale into each capture, so a change
    mid-run is a command against the run's hardware state.

    A motion or LED command for hardware the scope does not have raises it
    too, and ``missing`` names the part: ``'not_connected'`` when the model
    has a motor controller and none is connected, whose remedy is the cable,
    or when no LED controller is connected; ``'axis_absent'`` when the
    scope does not have the motion part -- a Z-only scope asked for X, a
    scope with no turret asked for one, a manual scope asked for any
    motion; ``'not_on_this_scope'`` when it does not have another part -- an
    LS560 asked to light Ex589, a motor controller without fan control asked
    to drive the fan. Nothing was driven. So does a move whose drive needs a
    position the controller did not report (``'position_unread'``): the
    relative base, or Z for the backlash approach. Nothing was driven and
    the axis keeps its state.
    So does a move or a home the stage's own interlock refused before
    anything moved (``INTERLOCK_REASONS``): the lid is open, or the stage
    has no power; the refused axis is where it was, its position known.
    A capture asked of a camera that is connected but not streaming is
    refused ``'camera_not_streaming'``: nothing can deliver its frame.

    A declined request, not a fault, so it is a ``Refusal``: the lane shows
    it as a warning in its own words and logs one line without a
    traceback. The message is written for the person at the scope and
    names the holder; ``reason`` and ``member`` are for code that maps a
    refusal to a response (REST status codes, SDK branches) and for the
    log.

    Attributes:
        reason: Machine-readable refusal code.
            ``'exclusive_activity_running'``, ``'not_connected'``,
            ``'camera_not_streaming'`` and ``'lid_open_before_motion'`` are
            explained by ``ProtocolRunRefusedError``, which carries them as
            the same fact. ``'capture_in_flight'`` -- an earlier still has
            not finished. ``'home_in_flight'`` -- another home holds the
            scope. ``'home_not_stoppable'`` -- a home drives on a motor
            controller that cannot stop one. ``'scope_disconnected'`` -- the
            scope's lanes are closed, so it takes no command until the
            application restarts. ``'monitor_ended'`` -- the motion monitor
            has ended, so nothing would judge the move.
            ``'position_unread'`` -- the move's drive needs a position the
            motor controller did not report: the relative base, or Z for the
            backlash approach. ``'stage_unpowered_before_motion'`` -- the
            stage has no power, so its interlock refuses to move it.
            ``'axis_absent'`` -- the scope does not have the motion part
            asked for. ``'not_on_this_scope'`` -- the scope does not have
            another part asked for, such as an LED channel or fan control.
            ``'activity_ended'`` -- the command was made under an activity
            that has ended. ``'other_led_lit'`` -- a capture held at a layer
            would be lit by another LED, which ``RecordingRefusedError``
            explains as the same fact. ``'channel_not_shown'`` -- a still was
            asked of a layer other than the shown one.
        member: The member or task that was refused, for the log.
        holder: The kind of activity holding the scope, when known.
        holder_trigger: The holding run's starter, for a run (``'gui'``,
            ``'rest'``, ``'api'``); None otherwise.
        missing: The part the command needed and the scope does not have,
            for ``'not_connected'``, ``'axis_absent'`` and
            ``'not_on_this_scope'``; None otherwise.
        title: The heading shown with the sentence, which follows the reason:
            nothing connected is not a busy microscope.

    ``holder`` and ``holder_trigger`` are published.
    """

    holder: Holder | None
    holder_trigger: Caller | None

    # A part the scope does not have is never found by asking again, and a
    # task made under a taking that has ended is never sent under it. A
    # scope whose lanes are shut, or whose motion monitor has ended, takes
    # no command again until the application restarts, and nor does a part
    # that is not connected: nothing reconnects one during a session.
    causes: ClassVar[dict[str, RefusalCause]] = {
        'exclusive_activity_running': RefusalCause.STATE,
        'capture_in_flight': RefusalCause.STATE,
        'home_in_flight': RefusalCause.STATE,
        'home_not_stoppable': RefusalCause.STATE,
        'scope_disconnected': RefusalCause.RESTART,
        'monitor_ended': RefusalCause.RESTART,
        'position_unread': RefusalCause.STATE,
        'lid_open_before_motion': RefusalCause.STATE,
        'stage_unpowered_before_motion': RefusalCause.STATE,
        'not_connected': RefusalCause.RESTART,
        'camera_not_streaming': RefusalCause.STATE,
        'axis_absent': RefusalCause.REQUEST,
        'not_on_this_scope': RefusalCause.REQUEST,
        'activity_ended': RefusalCause.REQUEST,
        'other_led_lit': RefusalCause.STATE,
        'channel_not_shown': RefusalCause.STATE,
    }
    same_fact_as = (ProtocolRunRefusedError, RecordingRefusedError)

    # Keyed as ``causes`` is: a refusal for who holds the scope reads as the
    # microscope being busy; one for the hardware's state names that state.
    _TITLES: ClassVar[dict[str, str]] = {
        'activity_ended': 'Microscope Busy',
        'capture_in_flight': 'Microscope Busy',
        'exclusive_activity_running': 'Microscope Busy',
        'home_in_flight': 'Microscope Busy',
        'home_not_stoppable': 'Microscope Busy',
        'not_connected': 'Not Connected',
        'scope_disconnected': 'Not Connected',
        'camera_not_streaming': CAMERA_NOT_STREAMING_TITLE,
        'monitor_ended': 'Motion Monitor Stopped',
        'axis_absent': 'Not on This Microscope',
        'not_on_this_scope': 'Not on This Microscope',
        'position_unread': 'Motor Controller Not Responding',
        'lid_open_before_motion': INTERLOCK_TITLES['lid_open'],
        'stage_unpowered_before_motion': INTERLOCK_TITLES['stage_unpowered'],
        'other_led_lit': RecordingRefusedError._TITLES['other_led_lit'],
        'channel_not_shown': 'Another Layer Is Shown',
    }

    def __init__(
        self,
        reason: str,
        member: str,
        holder: Holder | None = None,
        *,
        missing: 'MissingPart | None' = None,
        holder_trigger: Caller | None = None,
    ):
        if (missing is None) != (reason not in _MISSING_PART_REASONS):
            raise TypeError(f'a {reason!r} refusal takes missing= exactly when it names a part')
        if missing is not None and missing.reason != reason:
            raise TypeError(f'{missing} is refused as {missing.reason!r}, not {reason!r}')
        if missing is not None:
            sentence = missing.sentence
        else:
            sentence = _command_refused_sentence(reason, holder)
        super().__init__(sentence)
        self.reason = reason
        self.member = member
        self.holder = _holder(holder)
        self.holder_trigger = holder_trigger
        self.missing = missing
        self.title = self._TITLES[reason]


# The refusals a stage's own interlock raises: the hardware is in a state
# that does not let it move, and the person's remedy is at the scope. The
# interlocks' one list names them.
INTERLOCK_REASONS = frozenset(REFUSED_BEFORE_MOTION.values())

# The refusals that describe the hardware's state rather than who holds the
# scope: no board, a board that did not report a position (the turret home's
# Z park), an interlock. A home refused for one of them is reported where it
# was asked for, as a missing board is; any other refusal of a home is a
# caller's defect.
HARDWARE_STATE_REASONS = frozenset({'not_connected', 'position_unread'}) | INTERLOCK_REASONS


@dataclass(frozen=True)
class MissingPart:
    """The hardware a refused command needed that the scope does not have.

    The refusal's reason and its sentence are the part's, so no raise words
    its own: a controller the model has is not connected, and every other
    part -- a motor, the turret, an LED -- is not on this scope. The motion
    parts, the LED controller and the camera are the class's constants; an LED is named
    when it is refused, by the channel's name (``MissingPart.led``) or by its
    board address (``MissingPart.led_address``) as the command named it, and
    a layer by name (``MissingPart.layer``).

    Attributes:
        name: The part, for the log.
        reason: ``'not_connected'`` for a controller, ``'axis_absent'``
            for a motion part (the motors, an axis, the turret, the XY
            stage), ``'not_on_this_scope'`` for any other part (a layer, an LED,
            fan control, LED engineering mode, a controller's text commands).
        sentence: What the person at the scope is told.
    """

    name: str
    reason: str
    sentence: str

    MOTOR_CONTROLLER: ClassVar['MissingPart']
    MOTORS: ClassVar['MissingPart']
    X: ClassVar['MissingPart']
    Y: ClassVar['MissingPart']
    Z: ClassVar['MissingPart']
    TURRET: ClassVar['MissingPart']
    XY_STAGE: ClassVar['MissingPart']
    LED_CONTROLLER: ClassVar['MissingPart']
    CAMERA: ClassVar['MissingPart']
    FAN_CONTROL: ClassVar['MissingPart']
    LED_ENGINEERING_MODE: ClassVar['MissingPart']
    TEXT_COMMANDS: ClassVar['MissingPart']

    @classmethod
    def axis(cls, axis: str) -> 'MissingPart':
        """The part for a motion axis name, ``'X'``, ``'Y'``, ``'Z'`` or ``'T'``.

        Raises:
            ValueError: ``axis`` is not one of them.
        """
        try:
            return _AXIS_PARTS[axis]
        except KeyError:
            raise ValueError(f'{axis!r} is not a motion axis') from None

    @classmethod
    def layer(cls, layer: str) -> 'MissingPart':
        """The part for a layer of this release that this scope's model does not have.

        For a command that takes a layer rather than lights an LED: a
        layer's settings, its focus, a still named for it. Lumi has no LED,
        so "no Lumi LED" would be true and beside the point.
        """
        return cls(f'{layer} layer', 'not_on_this_scope', f'This microscope has no {layer} layer.')

    @classmethod
    def led(cls, channel: str) -> 'MissingPart':
        """The part for a channel's LED that this scope's model does not have."""
        return cls(f'{channel} LED', 'not_on_this_scope', f'This microscope has no {channel} LED.')

    @classmethod
    def led_address(cls, address: int) -> 'MissingPart':
        """The part for an LED board address no channel of this scope's model drives."""
        return cls(
            f'LED address {address}',
            'not_on_this_scope',
            f'This microscope has no LED at address {address}.',
        )


# A part that is not connected stays so until LumaViewPro restarts: nothing
# reconnects a board or a camera during a session, so each sentence names the
# restart, as the bring-up words for the same parts do.
MissingPart.MOTOR_CONTROLLER = MissingPart(
    'motor controller',
    'not_connected',
    'The motor controller is not connected. Check the USB cable and that '
    'no other program is holding the port, then restart LumaViewPro.',
)
MissingPart.MOTORS = MissingPart('motors', 'axis_absent', 'This microscope has no motors.')
MissingPart.X = MissingPart('X', 'axis_absent', 'This microscope has no X motor.')
MissingPart.Y = MissingPart('Y', 'axis_absent', 'This microscope has no Y motor.')
MissingPart.Z = MissingPart('Z', 'axis_absent', 'This microscope has no Z motor.')
MissingPart.TURRET = MissingPart('T', 'axis_absent', 'This microscope has no turret.')
MissingPart.XY_STAGE = MissingPart(
    'XY stage',
    'axis_absent',
    'This microscope has no XY stage, so it has one field and no wells to move '
    'between: its plate is Center Plate.',
)
MissingPart.LED_CONTROLLER = MissingPart(
    'LED controller',
    'not_connected',
    'The LED controller is not connected. Check the USB cable and 24V power, '
    'then restart LumaViewPro.',
)
MissingPart.CAMERA = MissingPart(
    'camera',
    'not_connected',
    'The camera is not connected. Check its cable, then restart LumaViewPro.',
)
MissingPart.FAN_CONTROL = MissingPart(
    'fan control',
    'not_on_this_scope',
    "This microscope's motor controller does not support fan control.",
)
MissingPart.LED_ENGINEERING_MODE = MissingPart(
    'LED engineering mode',
    'not_on_this_scope',
    "This microscope's LED controller has no engineering mode.",
)
MissingPart.TEXT_COMMANDS = MissingPart(
    'text commands',
    'not_on_this_scope',
    "This microscope's controller takes no text commands.",
)
_AXIS_PARTS = {
    'X': MissingPart.X,
    'Y': MissingPart.Y,
    'Z': MissingPart.Z,
    'T': MissingPart.TURRET,
}


_MISSING_PART_REASONS = frozenset({'not_connected', 'axis_absent', 'not_on_this_scope'})


_HOLDER_NOUNS = {
    'protocol': 'A run',
    'diagnostic': 'A diagnostic',
    'recording': 'A recording',
    'home': 'A home',
    'control': 'Remote control',
}


def the_activity_named(kind: str | None) -> str:
    """Name an activity holding the scope by its kind, to open a sentence.

    The one table every refusal and lockout sentence reads, so a kind is
    worded the same wherever it is named. 'Another activity' when the kind
    is unknown -- a holder that released before it could be read.
    """
    return _HOLDER_NOUNS.get(kind, 'Another activity')


def _command_refused_sentence(reason: str, holder: Holder | None) -> str:
    if reason == 'capture_in_flight':
        return 'A capture is still being saved. Try again in a moment.'
    if reason == 'home_in_flight':
        return 'The microscope is already homing. Wait for the home to finish.'
    if reason == 'home_not_stoppable':
        return 'This microscope cannot stop a home; the home runs to its end.'
    if reason == 'activity_ended':
        return 'The activity that sent this command has ended, so the command was not sent.'
    if reason == 'scope_disconnected':
        return 'The microscope has been disconnected, so the command was not sent.'
    if reason == 'monitor_ended':
        return (
            'The software that watches moves has stopped, so nothing was moved. '
            'Restart the application.'
        )
    if reason == 'position_unread':
        return (
            'The motor controller did not report the stage position, so the move '
            'was not sent. Try again; if it repeats, check the USB cable.'
        )
    if reason == 'lid_open_before_motion':
        return "The microscope's lid is open. Close it to move or home the stage."
    if reason == 'stage_unpowered_before_motion':
        return "The stage has no power. Check the stage's power supply, then home."
    if reason == 'other_led_lit':
        return (
            "An LED other than the shown layer's is lit, so nothing was captured. "
            'Turn it off, or show its layer, then capture.'
        )
    if reason == 'channel_not_shown':
        return 'A still is taken of the shown layer. Show the layer, then capture.'
    if reason == 'camera_not_streaming':
        return (
            'The camera is connected but is not delivering frames, so nothing was '
            'captured. Start its live feed and try again.'
        )
    return f'{the_activity_named(holder)} is using the microscope. Try again when it ends.'


@api_fields('holder', 'holder_trigger')
class DiagnosticRefusedError(Refusal, Exception):
    """A diagnostic could not take the scope: another activity holds it.

    Raised by ``ScopeSession.diagnostic_claim()`` when a run, a recording, a
    home or another diagnostic already holds the session's activity claim;
    a control session held by someone else refuses it with
    ``ControlSessionRefusedError``, as it does every lane. A diagnostic drives the
    hardware directly (homes, LED modes, forced grabs), so it runs only on a
    scope nothing else is using, and a caller told why can wait for the
    holder or stop it. Nothing was committed.

    Attributes:
        reason: Machine-readable refusal code for callers that map refusals
            to responses (REST status codes, SDK branches):
            ``'exclusive_activity_running'``, as ``ProtocolRunRefusedError``
            explains it.
        title: Short user-facing refusal title, the reason's.
        message: One-sentence user-facing refusal body.
        holder: The activity kind holding the claim at refusal time.
        holder_trigger: Who started the holding run (``'gui'``, ``'rest'``,
            ``'api'``) when the holder is a run; None otherwise.

    ``holder`` and ``holder_trigger`` are published.
    """

    holder: Holder | None
    holder_trigger: Caller | None

    _TITLES: ClassVar[dict[str, str]] = {
        'exclusive_activity_running': ProtocolRunRefusedError._TITLES['exclusive_activity_running'],
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_TITLES, RefusalCause.STATE)
    same_fact_as = (HardwareCommandRefusedError,)

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        holder: 'Holder | None' = None,
        holder_trigger: Caller | None = None,
    ):
        if reason not in self._TITLES:
            raise TypeError(f'DiagnosticRefusedError has no reason {reason!r}')
        super().__init__(f'{reason}: {message}')
        self.reason = reason
        self.title = self._TITLES[reason]
        self.message = message
        self.holder = _holder(holder)
        self.holder_trigger = holder_trigger


class SessionClosingError(Refusal, Exception):
    """An activity was asked to take the scope while the session is closing.

    Raised by the activity claim for a run, a recording, a home or a
    diagnostic while ``ScopeSession.shutdown`` runs: the close finishes the
    work already under way, then releases the hardware, so nothing new may
    start in between. Once the close has finished, nothing takes the scope
    again: a start whose own checks come first is refused by them, finding
    the scope disconnected, and any other is refused here when it takes the
    scope -- a home, or a run that passed its checks before the close
    finished. Work inside an activity already under way -- a run's video
    step under its run -- is not refused. Nothing was taken.

    Attributes:
        reason: ``'session_closing'``.
        activity: The kind of activity that was refused.
    """

    # A closing session never takes the scope again.
    title = 'Closing'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'session_closing': RefusalCause.RESTART}

    def __init__(self, activity: str):
        super().__init__(
            f'{the_activity_named(activity)} was not started: LumaViewPro is closing, and finishes '
            'what is already under way before it lets go of the microscope.'
        )
        self.reason = 'session_closing'
        self.activity = activity


@api_fields('holder', 'holder_trigger', 'control_session')
class ControlSessionRefusedError(Refusal, Exception):
    """Control of the scope was not taken or used: someone else's work holds it.

    Raised by the activity claim when a control session is asked for while
    another one is held, while an activity holds the scope, while a setting
    or file write is still under way, or while work that has let the scope
    go is still finishing -- so a client never takes the scope over the
    operator's work, or over a write half done. While a session is held,
    it is the one refusal of anything asked for by anyone but the
    holder (``'control_session_held'``): a guarded write (a setting, a
    file), a command on a hardware lane, and the start of a run, a
    recording, a diagnostic or a home. Raised too for a request naming a control
    session that is not the live one: it was ended, or never was. Nothing
    was taken or written.

    Attributes:
        reason: ``'control_session_held'`` -- a control session is held, and
            the request is not its holder's. ``'write_in_flight'`` -- a
            setting or a file is still being written, so control is not
            taken over it half done. ``'work_finishing'`` -- a run or a
            recording that has ended is still finishing.
            ``'control_session_ended'`` -- the request names a control
            session that is not the live one: it has ended, or was never
            issued. ``'exclusive_activity_running'``, an activity holding
            the scope, is ``ProtocolRunRefusedError``'s to explain.
        title: The heading the person reads, the reason's.
        holder: The kind of activity holding the scope, for
            ``'exclusive_activity_running'``; None otherwise.
        holder_trigger: Who holds the scope: the control session's taker,
            or the holding run's starter (``'gui'``, ``'rest'``, ``'api'``);
            None when neither is known.
        control_session: The name of the control session that holds, for
            ``'control_session_held'``; None otherwise.

    ``holder``, ``holder_trigger`` and ``control_session`` are published.
    """

    holder: Holder | None
    holder_trigger: Caller | None
    control_session: str | None

    # An id that is not the live session's never becomes it again: the
    # caller takes control anew.
    causes: ClassVar[dict[str, RefusalCause]] = {
        'control_session_held': RefusalCause.STATE,
        'exclusive_activity_running': RefusalCause.STATE,
        'write_in_flight': RefusalCause.STATE,
        'work_finishing': RefusalCause.STATE,
        'control_session_ended': RefusalCause.REQUEST,
    }
    same_fact_as = (HardwareCommandRefusedError,)

    _TITLES: ClassVar[dict[str, str]] = {
        'control_session_held': UNDER_CONTROL_TITLE,
        'exclusive_activity_running': 'Microscope Busy',
        'write_in_flight': 'Microscope Busy',
        'work_finishing': 'Microscope Busy',
        'control_session_ended': 'Control Session Ended',
    }

    def __init__(
        self,
        reason: str,
        *,
        holder: Holder | None = None,
        holder_trigger: Caller | None = None,
        control_session: str | None = None,
    ):
        self.reason = reason
        super().__init__(_control_refused_sentence(reason, holder, holder_trigger, control_session))
        self.title = self._TITLES[reason]
        self.holder = _holder(holder)
        self.holder_trigger = holder_trigger
        self.control_session = control_session


def _control_refused_sentence(
    reason: str, holder: Holder | None, holder_trigger: Caller | None, control_session: str | None
) -> str:
    if reason == 'control_session_held':
        return under_control_sentence(holder_trigger, control_session)
    if reason == 'exclusive_activity_running':
        return f'{the_activity_named(holder)} is using the microscope. Take control when it ends.'
    if reason == 'write_in_flight':
        return 'A setting or a file is being written. Take control when it finishes.'
    if reason == 'work_finishing':
        return (
            'A run or a recording that has ended is still finishing. Take control when it is done.'
        )
    return (
        'No live control session has that id: it has ended, or was never issued. '
        'Take control again to go on.'
    )


def under_control_sentence(holder_trigger: Caller | None, control_session: str | None) -> str:
    """The one sentence for every refusal a held control session makes: who holds, and the way out."""
    who = f'"{control_session}"' if control_session else 'A client'
    through = f' ({holder_trigger.words})' if holder_trigger is not None else ''
    return f'{who}{through} has control of the microscope. Ask it to end, or take back control.'


@api_fields('argument', 'board', 'limits', 'member', 'offered', 'plate', 'takes')
class ArgumentRefusedError(Refusal, ValueError):
    """An argument the scope cannot act on: it breaks a fixed rule, whatever the scope's state.

    Raised by the member that owns the rule, before anything is commanded
    or stored, so every caller -- REST, the SDK, the GUI, a run's own
    engine -- is refused alike. A ``ValueError``, as the bare raises it
    replaces were, so a caller catching a bad argument keeps working.

    Attributes:
        reason: One of:

            - ``'not_offered'`` -- a value of a closed set (a ``StrEnum``)
              that is not one of ``offered``: not a member, nor a member's
              plain string, or a member the taker does not take. Values
              are exact: ``'off'`` is not ``'Off'``.
            - ``'not_a_number'`` -- a number was needed and the value is
              not a finite one: NaN, an infinity, a ``bool`` or not a
              number at all. Every comparison with NaN is False, so a
              range check alone passes it.
            - ``'plate_frame_axis'`` -- a plate coordinate for an axis the
              plate frame does not define (``offered``: X and Y).
            - ``'well_unknown'`` -- a well label the selected plate
              (``plate``) does not have.
            - ``'out_of_range'`` -- a number outside the ``limits`` its
              member takes.
            - ``'fan_duty_out_of_range'`` -- a fan duty outside 0 to 100
              percent.
            - ``'channel_unknown'`` -- the name is not one of this release's
              channels (``offered``, the catalogue). Names are exact:
              ``'bf'`` is not ``'BF'``. A channel this model lacks is a
              name, and is refused as absent hardware, not here.
            - ``'no_channel_selected'`` -- the member needs a channel and was
              given None (``offered``, the catalogue).
            - ``'led_address_unknown'`` -- an LED address that is not one of
              the attached LED board's addresses (``offered``). A ``bool`` is
              not an address.
            - ``'illumination_out_of_range'`` -- an LED current outside
              ``limits``, 0 to the attached LED board's maximum, in mA.
            - ``'missing_key'`` -- a dictionary argument lacks a key its
              member reads; ``argument`` names the key's path.
            - ``'wrong_kind'`` -- an argument, or a value in a dictionary
              argument, is not of the kind its member reads (``takes``, in
              words).
            - ``'protocol_source_ambiguous'`` -- not exactly one of the
              protocol sources ``offered`` was given.
            - ``'not_a_setting'`` -- a dotted path that names no setting.
              The same fact a write is refused for as
              ``SettingRefusedError('not_a_setting')``; this type says a
              read asked it.
            - ``'has_member'`` -- a dotted path to a setting that has a read
              of its own, named in ``member`` by its path from the session,
              so one setting has one read.
            - ``'needs_simulated_scope'`` -- an argument that exists only
              on a simulated scope, given for a real one.
            - ``'needs_simulated_camera'`` -- an argument that exists only
              on the simulated camera, given for a model simulated with
              an FX2.
            - ``'exposure_not_recorded'`` -- a frame the simulated camera
              holds no exposure record for: one it did not deliver, or one
              older than the records it keeps.
            - ``'not_an_installation'`` -- a folder that holds neither
              settings file, so it is not an LVP installation root.
            - ``'binning_size_invalid'`` -- a binning label that is not a
              positive square ``NxN``.
            - ``'fault_unknown'`` -- a simulated fault's board, or a
              device's hardware ID, that is not one of ``offered``: sets
              this scope's boards and this machine's devices decide.
            - ``'fault_bound'`` -- a simulated fault's count, fraction or
              time outside what it takes (``takes``, in words).
            - ``'not_injectable_on_board'`` -- a simulated fault the board
              that would present it cannot produce; ``argument`` names the
              ``scope.simulation`` member, ``board`` the board.
            - ``'argument_not_on_board'`` -- an argument of a simulated
              fault the board cannot take; ``board`` names the board,
              ``takes`` what it takes for that argument.
            - ``'slot_not_holding_objective'`` -- a turret slot given for
              steps naming an objective the slot does not hold now
              (``offered``: the slots that hold it).
            - ``'objective_in_several_slots'`` -- an objective to turn
              the turret to that two or more slots hold (``offered``: the
              slots), so no one slot is meant.
            - ``'objective_not_on_turret'`` -- an objective to turn the
              turret to that no slot holds (``offered``: the objectives
              the slots hold, in slot order).
            - ``'slot_required'`` -- an objective confirmed on
              a turret scope with no slot named; the confirmation records
              the glass in a slot.
            - ``'gain_range_inverted'`` -- an auto-gain settings dictionary
              whose ``min_gain_db`` is above its ``max_gain_db``; ``value``
              holds the two.
            - ``'key_unknown'`` -- a key a dictionary argument does not
              take (``offered``: the keys it takes).
        argument: The name of the argument refused.
        value: What was given, as given.
        offered: The values the argument takes, where it takes a list;
            None otherwise.
        limits: The lowest and highest value the argument takes, where it
            takes a range that depends on the scope; None otherwise.
        takes: The value the argument takes, in words, for
            ``'wrong_kind'``, ``'fault_bound'`` and
            ``'argument_not_on_board'``; None otherwise.
        board: The simulated board, in words, for
            ``'not_injectable_on_board'`` and ``'argument_not_on_board'``;
            None otherwise.
        plate: The name of the plate looked on, for ``'well_unknown'``;
            None otherwise.
        member: The member to read instead, for ``'has_member'``; None
            otherwise.

    ``argument``, ``offered``, ``limits``, ``takes``, ``board``, ``plate``
    and ``member`` are published, so a REST problem carries them beside the
    words; the client already holds the value it sent.
    """

    argument: str
    offered: tuple[str | int, ...] | None
    limits: tuple[float, float] | None
    takes: str | None
    board: str | None
    plate: str | None
    member: str | None
    _WORDS: ClassVar[dict[str, tuple[str, str]]] = {
        'not_offered': (
            'Not One of the Values Offered',
            '{argument} must be one of {offered}; {value!r} is not one.',
        ),
        'not_a_number': (
            'Not a Number',
            '{argument} must be a finite number; {value!r} is not one.',
        ),
        'plate_frame_axis': (
            'Not a Plate Axis',
            'Plate coordinates are defined for {offered} only; {argument} {value!r} is not one of them.',
        ),
        'well_unknown': (
            'Not a Well',
            '{plate} has no well {value!r}.',
        ),
        'out_of_range': (
            'Value Out of Range',
            '{argument} must be from {low:g} to {high:g}; {value!r} is outside it.',
        ),
        'fan_duty_out_of_range': (
            'Fan Duty Not Changed',
            '{argument} is a percentage from 0 to 100; {value!r} is outside it.',
        ),
        'channel_unknown': (
            'Not a Channel',
            '{argument} must be one of {offered}; {value!r} is not one.',
        ),
        'no_channel_selected': (
            'No Channel Selected',
            'No channel is selected; {argument} must be one of {offered}.',
        ),
        'led_address_unknown': (
            'Not an LED Address',
            "{argument} must be one of the LED board's addresses, {offered}; {value!r} is not one.",
        ),
        'illumination_out_of_range': (
            'LED Current Not Set',
            '{argument} must be from {low:g} to {high:g} mA on this LED board; {value!r} is outside it.',
        ),
        'missing_key': (
            'Value Missing',
            '{argument} is required and was not given.',
        ),
        'wrong_kind': (
            'Wrong Kind of Value',
            '{argument} must be {takes}; {value!r} is not.',
        ),
        'protocol_source_ambiguous': (
            'Protocol Source Ambiguous',
            'Pass exactly one of {offered}; {value} were given.',
        ),
        'binning_size_invalid': (
            'Not a Binning Size',
            '{argument} must be NxN, with N a whole number of at least 1, such as 2x2; '
            '{value!r} is not.',
        ),
        'not_a_setting': (
            'Not a Setting',
            '{argument} names no setting; {value!r} is not a dotted path to one.',
        ),
        'has_member': (
            'Setting Has Its Own Read',
            "{value!r} is read by {member}, the setting's own read; {argument} reads a "
            'setting that has none.',
        ),
        'needs_simulated_scope': (
            'Needs a Simulated Scope',
            '{argument} needs a simulated scope (simulate=True).',
        ),
        'needs_simulated_camera': (
            'Needs the Simulated Camera',
            '{argument} needs the simulated camera; this model is simulated with an FX2.',
        ),
        'exposure_not_recorded': (
            'No Exposure Record',
            '{argument} is not a frame the simulated camera holds a record for: one it did '
            'not deliver, or one older than the records it keeps.',
        ),
        'not_an_installation': (
            'Not an Installation',
            '{argument} is not an LVP installation root: {value!r} holds neither '
            'data/current.json nor data/settings.json.',
        ),
        'fault_unknown': (
            'Not a Simulated Fault',
            '{argument} must be one of {offered}; {value!r} is not one.',
        ),
        'fault_bound': (
            'Not a Simulated Fault',
            '{argument} must be {takes}; {value!r} is not.',
        ),
        'not_injectable_on_board': (
            'Fault Not Injectable',
            'On {board}, {argument} is not a fault that can be injected.',
        ),
        'argument_not_on_board': (
            'Fault Not Injectable',
            '{argument} {value!r} is not taken by {board}; it takes {takes}.',
        ),
        'slot_not_holding_objective': (
            'Slot Does Not Hold That Objective',
            '{argument} must be a slot holding the objective, one of {offered}; {value!r} is not.',
        ),
        'objective_in_several_slots': (
            'Objective in Several Slots',
            '{value!r} is in turret slots {offered}; turn the turret to the slot to use.',
        ),
        'objective_not_on_turret': (
            'Objective Not on the Turret',
            '{value!r} is in no turret slot; the turret holds {offered}. Assign it to the '
            'slot it is installed in.',
        ),
        'gain_range_inverted': (
            'Auto-Gain Range Inverted',
            '{argument} sets a minimum gain above its maximum: {value!r}.',
        ),
        'key_unknown': (
            'Not a Key Taken Here',
            '{argument} takes only the keys {offered}; {value!r} is not one.',
        ),
        'slot_required': (
            'Turret Slot Not Named',
            '{argument} is needed on a scope with a turret: the confirmation records the '
            'objective in a slot.',
        ),
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_WORDS, RefusalCause.REQUEST)
    same_fact_as = (
        SettingRefusedError,
        ProtocolRunRefusedError,
    )

    def __init__(
        self,
        reason: str,
        *,
        argument: str,
        value: object,
        offered: tuple[str | int, ...] | None = None,
        limits: tuple[float, float] | None = None,
        takes: str | None = None,
        board: str | None = None,
        plate: str | None = None,
        member: str | None = None,
    ):
        if reason not in self._WORDS:
            raise TypeError(f'ArgumentRefusedError has no words for the reason {reason!r}')
        self.title, words = self._WORDS[reason]
        if ('{offered}' in words) != (offered is not None):
            raise TypeError(
                f'ArgumentRefusedError({reason!r}) takes offered values exactly when its words name them'
            )
        if ('{low' in words) != (limits is not None):
            raise TypeError(
                f'ArgumentRefusedError({reason!r}) takes limits exactly when its words name them'
            )
        for name, given in (('takes', takes), ('board', board), ('plate', plate)):
            if ('{' + name + '}' in words) != (given is not None):
                raise TypeError(
                    f'ArgumentRefusedError({reason!r}) takes {name} exactly when its words name it'
                )
        if ('{member}' in words) != (member is not None):
            raise TypeError(
                f'ArgumentRefusedError({reason!r}) takes a member exactly when its words name it'
            )
        low, high = limits or (None, None)
        super().__init__(
            words.format(
                argument=argument,
                # A member of a closed set is named by its word, as a caller
                # writes it; its repr is the class's, not the caller's.
                value=str(value) if isinstance(value, StrEnum) else value,
                offered=_offered_words(offered or ()),
                low=low,
                high=high,
                takes=takes,
                board=board,
                plate=plate,
                member=member,
            )
        )
        self.reason = reason
        self.argument = argument
        self.value = value
        self.offered = offered
        self.limits = limits
        self.takes = takes
        self.board = board
        self.plate = plate
        self.member = member


@api_fields('member', 'argument', 'declared')
class ArgumentTypeRefusedError(Refusal, TypeError):
    """An argument whose type its member does not take, refused before the member runs.

    Raised by the ``@api`` door for every caller alike -- the GUI, a
    script, a plugin, a member calling another -- from the
    member's own annotation, so the body never meets a value of another
    type. A ``TypeError``, as Python raises for a call it cannot take. The
    door judges the type only: a number's range and finiteness, a name's
    membership and a dictionary's contents stay their member's.

    Attributes:
        reason: ``'wrong_argument_type'``.
        member: The member called, as ``Class.member``.
        argument: The name of the argument refused.
        declared: The type the member takes, as its annotation reads.
        value: What was given, as given.
    """

    member: str
    argument: str
    declared: str
    causes: ClassVar[dict[str, RefusalCause | None]] = {'wrong_argument_type': RefusalCause.REQUEST}
    title = 'Wrong Type of Argument'

    def __init__(self, *, member: str, argument: str, declared: str, value: object):
        shown = repr(value)
        if len(shown) > 80:
            shown = shown[:77] + '...'
        super().__init__(
            f'{member} takes {argument} as {declared}; {shown} is a {type(value).__name__}.'
        )
        self.member = member
        self.argument = argument
        self.declared = declared
        self.value = value


@api_fields('limits')
class AccelerationLimitRefusedError(Refusal, ValueError):
    """An acceleration limit no board may be given was refused; nothing was commanded or stored.

    A refusal, not a fault: the person typed or a caller passed a number
    outside the range, and the words tell them the range. Raised as a bare
    ValueError it was reported as an operation that failed, with a
    traceback, over "Check the main log for details". Subclasses ValueError
    so callers that catch a bad argument keep working.

    Attributes:
        reason: ``'acceleration_out_of_range'``.
        value: What was given, as given.
        limits: The lowest and highest limit a board takes, in percent.

    ``limits`` is published, so a client corrects the value without reading
    the words.
    """

    limits: tuple[int, int]

    causes: ClassVar[dict[str, RefusalCause | None]] = {
        'acceleration_out_of_range': RefusalCause.REQUEST
    }
    title = 'Acceleration Limit Not Changed'

    def __init__(self, value: object, low: int, high: int):
        super().__init__(
            f'An acceleration limit of {value!r}% cannot be set: it is a whole number from {low} to {high}.'
        )
        self.value = value
        self.limits = (low, high)


@api_fields('axis', 'limits')
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

    Attributes:
        axis: The axis whose position or distance was refused.
        position: What was asked for.
        limits: The lowest and highest value ``bound`` allows, in the
            quantity's own unit.
        bound: Which limit refused it.
        quantity: What was refused: a position, a distance, a plate position.

    ``axis`` and ``limits`` are published, so a client corrects the move
    without reading the words.
    """

    axis: Axis
    limits: tuple[float, float]

    # The travel is the model's, fixed for the scope's life, so a target
    # outside it is refused every time it is sent.
    causes: ClassVar[dict[str, RefusalCause | None]] = {
        'position_out_of_range': RefusalCause.REQUEST
    }
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
        self.axis = Axis(axis)
        self.position = position
        self.limits = (low, high)
        self.bound = bound
        self.quantity = quantity


# The axis-state value for an axis whose home is in progress. Spelled here
# rather than imported: AxisState lives in the lumascope_api package, whose
# import pulls in modules that import this one.
_AXIS_HOMING = 'homing'


def _names(names: list[str]) -> str:
    """Axis names as a person lists them: "Z", "X and Y", "X, Y and Z"."""
    return names[0] if len(names) == 1 else f'{", ".join(names[:-1])} and {names[-1]}'


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
        reason: ``'position_unknown'``.
        axes: Every refused axis, mapped to its state, in the scope's axis
            order.
        axis: The first of them, for callers that map a refusal to a
            response by a single axis (REST status codes, SDK branches).
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'position_unknown': RefusalCause.STATE}
    same_fact_as = (ProtocolRunRefusedError,)
    title = ProtocolRunRefusedError._TITLES['position_unknown']

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


class AxisMovingError(Refusal, Exception):
    """A position was asked for to be kept while an axis it records is moving.

    While an axis moves, the number it answers is the motion monitor's
    last poll, behind the board by up to a poll, so a focus, a bookmark or
    a step saved then records a place the axis was passing through, real-
    looking and never where it stopped. The state and the number are read
    in one snapshot, so a move that ends between the two cannot pass a
    number taken while it ran.

    Distinct from ``AxisStateUnknownError``: a moving axis knows its frame
    and needs only to stop; the user waits, never homes.

    Attributes:
        reason: ``'axis_moving'``.
        axes: Every moving axis asked for, in the order asked.
        axis: The first of them, for callers that map a refusal to a
            response by a single axis (REST status codes, SDK branches).
    """

    title = 'Axis Moving'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'axis_moving': RefusalCause.STATE}

    def __init__(self, axes: Iterable[str]):
        """Build the refusal for ``axes``.

        Args:
            axes: Every moving axis. Must not be empty.
        """
        axes = list(axes)
        moving = 'is' if len(axes) == 1 else 'are'
        them = 'it' if len(axes) == 1 else 'them'
        super().__init__(
            f'{_names(axes)} {moving} still moving, so there is no position to keep. '
            f'Wait for {them} to stop, then try again.'
        )
        self.reason = 'axis_moving'
        self.axes = axes
        self.axis = axes[0]


class MoveNotCompletedError(_Umbrella, Exception):
    """A move ended without its axis arriving at the target: the subject, never raised itself.

    A waited move that returns means the axis arrived; this move could not
    say that, so it raises instead of reporting a position nobody reached.
    ``except MoveNotCompletedError`` catches every way a move ends short;
    the child raised says which, and its kind with it: ``MoveFaultError``
    when the move failed, ``MoveStoppedError`` when a stop or a later
    command ended it.

    Distinct from ``AxisStateUnknownError``, which refuses a move before
    anything is driven and offers ``force=True`` -- advice that is wrong
    for a move that already happened.

    Attributes:
        axis: The axis whose move did not complete.
        reason: The child's code for how the move ended.
        title: The heading shown with the sentence.
    """

    _SENTENCES: ClassVar[dict[str, str]]

    def __init__(self, axis: str, reason: str):
        self._refuse_if_bare()
        super().__init__(
            f'The {axis} move did not complete: ' + self._SENTENCES[reason].format(axis=axis)
        )
        self.axis = axis
        self.reason = reason


class MoveFaultError(Fault, MoveNotCompletedError):
    """A move failed: the motor may have travelled any part of the way.

    Attributes:
        reason: ``'driver_failed'`` -- the motor board did not take the
            command (chained from the driver's error). ``'stalled'`` --
            the motion monitor gave the axis up: its position stopped
            changing short of its target, and the stage was halted.
            ``'monitor_ended_mid_move'`` -- the motion monitor ended while the
            axis moved, so nothing could judge the move. ``'board_lost'`` -- the
            monitor lost the motor board while the axis moved.
            ``'faulted'`` -- something else set the axis UNKNOWN during
            the move, carrying no fault of its own, with no stop before it.
            ``'position_unread_after_move'`` -- the board did not report
            where the axis is: within the stall window after it reported the
            axis arrived, or when a home read every axis's position.
            ``'status_unread'`` -- the board did not report whether the
            axis arrived, within the stall window. Each leaves the axis
            UNKNOWN.
        title: The heading shown with the sentence, which follows the reason.
    """

    words_for_person = True
    _SENTENCES: ClassVar[dict[str, str]] = {
        'driver_failed': (
            'the motor board did not take the command. The {axis} position is now '
            'unknown -- home the scope before moving it again.'
        ),
        'stalled': (
            'it stopped moving before it reached its target, and the stage was '
            'halted. Check for an obstruction, then home the axis and retry.'
        ),
        'monitor_ended_mid_move': (
            'the software that watches moves stopped while it moved. The {axis} '
            'position is now unknown -- restart the application before moving it again.'
        ),
        'board_lost': (
            'the motor board was lost while it moved, so the move was aborted. '
            'Reconnect the board and retry.'
        ),
        'faulted': (
            'it stalled or the board was lost during the move. The {axis} position '
            'is now unknown -- home the scope before moving it again.'
        ),
        'position_unread_after_move': (
            'it stopped, but the motor board did not report where. The {axis} '
            'position is now unknown -- home the scope before moving it again.'
        ),
        'status_unread': (
            'the motor board did not report whether it arrived. The {axis} '
            'position is now unknown -- home the scope before moving it again.'
        ),
    }

    _TITLES: ClassVar[dict[str, str]] = {
        'driver_failed': 'Move Did Not Complete',
        'faulted': 'Move Did Not Complete',
        'stalled': 'Motor Axis Stalled',
        'monitor_ended_mid_move': 'Motion Monitor Stopped',
        'board_lost': 'Motor Board Disconnected',
        'position_unread_after_move': 'Motor Position Unknown',
        'status_unread': 'Motor Position Unknown',
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_TITLES)

    def __init__(self, axis: str, reason: str):
        super().__init__(axis, reason)
        self.title = self._TITLES[reason]


class MoveStoppedError(Notice, MoveNotCompletedError):
    """A move was ended before it arrived by a stop or a later command: nothing failed.

    A notice: the person whose move someone else stopped is told so, as
    information, and a REST caller is answered 409. The axis is where the
    stop or the later command left it, which its position reports; after
    the scope is closed its position is unknown.

    Attributes:
        reason: ``'stopped'`` -- a stop sent while it moved ended it:
            ``stop_motion``, the stage's own stop in refusing a move, or
            the close's, even one the board did not take (that failure is
            reported once, as itself); a turret is then in no known slot.
            ``'superseded'`` -- another move or a home took the axis before
            this one arrived; that command's own outcome says where the axis
            went.
    """

    title = 'Move Stopped'
    _SENTENCES: ClassVar[dict[str, str]] = {
        'stopped': 'the motors were stopped before it arrived.',
        'superseded': 'another command on the {axis} axis was given before it arrived.',
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_SENTENCES, RefusalCause.STATE)


class MoveWaitExpiredError(MoveNotCompletedError, Quiet, TimeoutError):
    """The caller's own wait ran out with the axis still moving: nothing failed.

    A ``TimeoutError``, as a caller's own wait expiring always is, and quiet:
    the move goes on, and the axis keeps the state its own move gives it.
    ``TimeoutError`` is the last base, so its ``OSError`` constructor never
    reads the axis and reason as an errno.

    Attributes:
        reason: ``'still_moving'``.
    """

    title = 'Still Moving'
    _SENTENCES: ClassVar[dict[str, str]] = {
        'still_moving': 'the {axis} axis was still moving when the wait ran out of time.',
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_SENTENCES, RefusalCause.STATE)


class LaneCallStillRunningError(Fault, Exception):
    """The API's own bound on a call ran out before the call finished.

    The hardware has stopped answering or is slower than the bound allows,
    and the call may still finish: nothing says it failed, nor that it
    will not change the scope after this. A fault, never a ``TimeoutError``,
    which means only that the caller's own wait ran out.

    Attributes:
        reason: ``'call_overran'``.
        member: The API member whose call is still running.
        timeout_s: The bound that ran out.
    """

    words_for_person = True
    title = 'Still Running'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'call_overran': None}

    def __init__(self, member: str, timeout_s: float):
        super().__init__(
            f'{member} did not finish within {timeout_s:g} s and may still finish. '
            'Check the scope before asking again.'
        )
        self.member = member
        self.timeout_s = timeout_s


class MotorStopFailedError(Fault, Exception):
    """The motor STOP command failed, so the stage may still be moving.

    Raised by ``stop_motion`` when the board did not take the STOP, chained
    from the driver's error. A failure, not a refusal: the STOP was sent
    and nothing vouches that it landed. The stop generation has already
    moved, so no waited move reads itself as arrived.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'motor_stop_failed': None}

    words_for_person = True
    title = 'Motor Stop Failed'

    def __init__(self):
        super().__init__(
            'The motor STOP command failed. If the stage is still moving, '
            'power-cycle the microscope.'
        )


class DisconnectPart(StrEnum):
    """A part of the microscope ``Lumascope.disconnect`` shuts down, in teardown order."""

    LEDS_OFF = 'LEDs off'
    MOTOR_STOP = 'motor stop'
    LED_BOARD = 'LED board'
    MOTOR_BOARD = 'motor board'
    CAMERA = 'camera'


@api_fields('parts', 'part_reasons')
class ScopeDisconnectError(Fault, Exception):
    """One or more parts of the microscope did not shut down cleanly.

    Raised by ``Lumascope.disconnect`` only after every teardown step has
    run, so one part's failure never leaves the others connected. Names
    each part that failed, in the order the teardown reached it, and is
    chained from the first part's error. Each part's error is kept private
    and logged where it was caught: an exception has no wire form, so what
    a client reads is each part's reason. A failure, not a refusal.

    Attributes:
        reason: ``'disconnect_failed'``.
        parts: The parts that failed, in teardown order.
        part_reasons: Each failed part's reason, keyed by part; None for an
            error that carries none.
    """

    parts: tuple[DisconnectPart, ...]
    part_reasons: dict[DisconnectPart, str | None]
    words_for_person = True
    title = 'Disconnect Failed'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'disconnect_failed': None}

    _PART_WORDS: ClassVar[dict[DisconnectPart, str]] = {
        DisconnectPart.LEDS_OFF: (
            'The LEDs were not confirmed off, so the sample may still be lit; '
            'power-cycle the microscope to turn them off.'
        ),
        DisconnectPart.MOTOR_STOP: (
            'The motor STOP command failed; if the stage is still moving, '
            'power-cycle the microscope.'
        ),
        DisconnectPart.LED_BOARD: (
            'The LED board did not shut down cleanly; its serial port may be '
            'left open, and reconnecting may require a restart.'
        ),
        DisconnectPart.MOTOR_BOARD: (
            'The motor board did not shut down cleanly; its serial port may be '
            'left open, and reconnecting may require a restart.'
        ),
        DisconnectPart.CAMERA: (
            'The camera did not shut down cleanly; its USB resources may not be '
            'released until the app restarts.'
        ),
    }

    def __init__(self, errors: dict[DisconnectPart, BaseException]):
        self._part_errors = {DisconnectPart(part): error for part, error in errors.items()}
        self.parts = tuple(self._part_errors)
        self.part_reasons = {
            part: getattr(error, 'reason', None) for part, error in self._part_errors.items()
        }
        super().__init__(' '.join(self._PART_WORDS[part] for part in self.parts))


class HomingFailedError(_Umbrella, Exception):
    """A home ended without establishing a reference position: the subject, never raised itself.

    ``except HomingFailedError`` catches every way a home ends short; the
    child raised says which, and its kind with it: ``HomingFaultError``
    when the home failed, ``HomingStoppedError`` when a Stop ended it.

    Attributes:
        home: What was homed: ``'ALL'``, ``'Z'`` or ``'T'``.
        reason: The child's code for how the home ended.
        axes: The axes left without a known position.
    """

    _SUBJECTS: ClassVar[dict[str, str]] = {
        'ALL': 'Homing',
        'Z': 'Z axis homing',
        'T': 'Turret homing',
    }

    def __init__(self, home: str, reason: str, axes: Iterable[str]):
        self._refuse_if_bare()
        self.home = home
        self.reason = reason
        self.axes = tuple(axes)
        super().__init__(self._sentence(self._SUBJECTS[home]))

    def _sentence(self, subject: str) -> str:
        raise NotImplementedError


class HomingFaultError(Fault, HomingFailedError):
    """A home failed: the driver raised, or a homed axis's position could not be read.

    Each leaves the axes UNKNOWN, so no caller may take the scope as
    knowing where it is. The motors may have moved. Chained from the
    driver's exception when there is one.

    Attributes:
        reason: ``'error'`` -- the home raised; ``'unread'`` -- homed, but
            ``axes`` could not be read; ``'lid_open'`` -- the lid was
            opened while the home was moving.
    """

    words_for_person = True
    title = 'Homing Failed'
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(
        ('error', 'unread', *INTERLOCK_TITLES)
    )
    same_fact_as = (MotionInterlockError,)

    def _sentence(self, subject: str) -> str:
        if self.reason == 'unread':
            return (
                f'Homing finished but the position of {", ".join(self.axes)} '
                'could not be read. Position is unknown.'
            )
        if self.reason == 'lid_open':
            return f"{subject} stopped: the microscope's lid was opened. Position is unknown."
        return f'{subject} encountered an error. Position is unknown.'


class HomingStoppedError(Notice, HomingFailedError):
    """A home was ended by a Stop: nothing failed.

    A notice: the person whose home someone else stopped is told so, as
    information, and a REST caller is answered 409. A Stop that landed
    before the driver was asked to home names no axis: nothing it homes
    had moved.

    Attributes:
        reason: ``'stopped'`` -- a Stop ended the home.
        axes: The axes left without a known position; empty for a home a
            Stop ended before it was driven.
    """

    title = 'Homing Stopped'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'stopped': RefusalCause.STATE}
    same_fact_as = (MoveStoppedError,)

    def _sentence(self, subject: str) -> str:
        if not self.axes:
            return f'{subject} was stopped before it began. The position is as it was.'
        return f'{subject} was stopped. Position is unknown.'


class AutofocusAborted(_Umbrella, Exception):  # noqa: N818 -- cancellation/abort signal, not an error; non-Error suffix is intentional
    """An autofocus sweep cut short before it chose a focus: the subject, never raised itself.

    ``except AutofocusAborted`` catches every way a sweep is cut short; the
    child raised says which, and its kind with it.
    """


class AutofocusStopped(Quiet, AutofocusAborted):
    """The sweep was stopped by its caller: a Stop, the run's cleanup, the app closing.

    Recorded, never shown: the run's own end is the report.

    Attributes:
        reason: ``'sweep_stopped'``.
    """

    title = 'Autofocus Stopped'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'sweep_stopped': RefusalCause.STATE}

    def __init__(self):
        super().__init__('Autofocus was stopped before it chose a focus.')
        self.reason = 'sweep_stopped'


class AutofocusRunEnded(Quiet, AutofocusAborted):
    """The run a sweep belongs to had ended when the sweep went to take its LEDs.

    Recorded, never shown: whatever holds the LEDs now is not that run, and
    the run's own end is the report.

    Attributes:
        reason: ``'sweep_run_ended'``.
    """

    title = 'Autofocus Not Started'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'sweep_run_ended': RefusalCause.STATE}

    def __init__(self):
        super().__init__('Autofocus did not start: the run it belongs to had ended.')
        self.reason = 'sweep_run_ended'


class AutofocusLedsHeldError(Refusal, AutofocusAborted):
    """The sweep's run is live, but another lease on top of the run's holds the LEDs.

    Without them the sweep would scan an unlit field and commit a wrong Z,
    so it does not start. The run goes on and captures at its fallback Z.

    Attributes:
        reason: ``'leds_held'``.
        holder: The purpose of the lease that holds the LEDs.
    """

    title = 'LEDs In Use'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'leds_held': RefusalCause.STATE}

    def __init__(self, holder: str):
        super().__init__(
            f'Autofocus did not start: {holder!r} holds the LEDs. '
            'The step is captured at its fallback focus.'
        )
        self.reason = 'leds_held'
        self.holder = holder


class CameraSettingRejected(Fault, Exception):  # noqa: N818 -- named for the event it signals; one type covers the defect class
    """The camera driver rejected a state-changing setting apply.

    Raised by CameraAPI setters (frame size, binning, image mode, gain,
    exposure) when a LIVE driver refuses the apply -- distinct from a
    camera that is not there, before or during the write, which is refused
    as ``HardwareCommandRefusedError('not_connected')``. Success is observed by receiving the
    applied/delivered value, failure by this raise, so a caller cannot
    record a rejected apply as applied by forgetting to check a return code.

    A fault, not a refusal: the camera did not take a value it should
    have, and the words say so. They are written for the person, so the
    one reporter shows them where the flight stops; no setter logs or
    notifies it.

    Attributes:
        reason: ``'camera_rejected'`` -- the camera refused the write, or
            its driver raised from it. ``'camera_no_answer'`` -- the camera
            did not answer the write, sent twice, so whether it took it is
            not known. The setting is in ``setting`` and in the words.
        setting: Machine-readable setting name (e.g. 'frame_size').
        requested: The value the caller asked for.
    """

    words_for_person = True
    _TITLES: ClassVar[dict[str, str]] = {
        'camera_rejected': 'Camera Setting Not Applied',
        'camera_no_answer': 'Camera Did Not Answer',
    }
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(_TITLES)
    same_fact_as = (CameraNoAnswerError,)

    def __init__(self, setting: str, requested, *, message: str, reason: str = 'camera_rejected'):
        if reason not in self._TITLES:
            raise TypeError(f'CameraSettingRejected has no reason {reason!r}')
        super().__init__(message)
        self.reason = reason
        self.title = self._TITLES[reason]
        self.setting = setting
        self.requested = requested


class CameraSettingUnsupportedError(Refusal, ValueError):
    """A camera setting this camera does not offer was asked for.

    Declined before anything reaches the camera: the value is outside the
    list the camera itself reports (its binning factors), so nothing
    broke and the person can pick one it offers. A ``ValueError`` too,
    for a caller that treats it as the bad argument it is.

    Attributes:
        reason: Machine-readable refusal code, ``'<setting>_unsupported'``.
            ``'auto_gain_unsupported'`` -- this camera has no automatic
            gain to turn on. ``'auto_exposure_unsupported'`` -- this camera
            has no automatic exposure to turn on.
            ``'auto_gain_target_brightness_unsupported'`` -- this camera has
            no automatic gain, so it takes no target brightness.
            ``'binning_unsupported'`` -- the binning size is not one this
            camera reports. ``'black_level_unsupported'`` -- this camera has
            no black level setting. ``'conversion_gain_mode_unsupported'``
            -- this camera has no high conversion gain, or the mode asked
            for is neither High nor Low.
            ``'line_noise_reduction_unsupported'`` -- this camera has no
            line noise reduction to turn on.
        setting: Machine-readable setting name (e.g. 'binning').
        requested: The value asked for.
        offered: The values the camera reports it supports.
    """

    # Hardware this camera does not have is the request's: asking again,
    # unchanged, never finds it.
    title = 'Not Available on This Camera'
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(
        (
            'auto_gain_unsupported',
            'auto_exposure_unsupported',
            'auto_gain_target_brightness_unsupported',
            'binning_unsupported',
            'black_level_unsupported',
            'conversion_gain_mode_unsupported',
            'line_noise_reduction_unsupported',
        ),
        RefusalCause.REQUEST,
    )

    def __init__(self, setting: str, requested, offered, *, message: str):
        super().__init__(message)
        self.reason = f'{setting}_unsupported'
        self.setting = setting
        self.requested = requested
        self.offered = offered


@api_fields('limits')
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
            ``'gain_db_out_of_range'`` -- the gain is outside the camera's
            declared range. ``'exposure_ms_out_of_range'`` -- the exposure
            is outside the camera's declared range.
            ``'frame_width_out_of_range'`` -- the frame width is outside the
            widths the camera declares. ``'frame_height_out_of_range'`` --
            the frame height is outside the heights the camera declares.
            ``'black_level_out_of_range'`` -- the black level is outside the
            camera's declared range.
        setting: Machine-readable setting name ('gain_db', 'exposure_ms',
            'frame_width', 'frame_height', 'black_level').
        requested: The value asked for.
        limits: The camera's declared floor and ceiling, in the setting's
            unit; either is None when the camera declares none, and is not
            checked.

    ``limits`` is published, so a client corrects the value without reading
    the words.
    """

    limits: tuple[float | None, float | None]

    # The range is the attached camera's, fixed for its life, so a value
    # outside it is refused every time it is sent.
    title = ProtocolRunRefusedError._TITLES['camera_setting_out_of_range']
    causes: ClassVar[dict[str, RefusalCause | None]] = dict.fromkeys(
        (
            'gain_db_out_of_range',
            'exposure_ms_out_of_range',
            'frame_width_out_of_range',
            'frame_height_out_of_range',
            'black_level_out_of_range',
        ),
        RefusalCause.REQUEST,
    )

    def __init__(
        self,
        setting: str,
        requested: float,
        minimum: float | None,
        maximum: float | None,
        *,
        message: str,
    ):
        super().__init__(message)
        self.reason = f'{setting}_out_of_range'
        self.setting = setting
        self.requested = requested
        self.limits = (minimum, maximum)


class FrameDepthError(Fault, Exception):
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

    causes: ClassVar[dict[str, RefusalCause | None]] = {'frame_depth_exceeded': None}
    title = 'Frame Depth Exceeded'

    def __init__(self, value: int, significant_bits: int):
        super().__init__(
            f'frame value {value} exceeds the declared {significant_bits}-bit depth '
            f'(max {(1 << significant_bits) - 1}); the significant-bits contract is wrong'
        )
        self.value = value
        self.significant_bits = significant_bits


# Where a support report or a logs zip is sent. The one copy: the saved
# report's words and the failed report's words both name it.
SUPPORT_ADDRESS = 'support@etaluma.com'


class SupportReportNotSavedError(Fault, Exception):
    """A support report or a logs zip was not saved.

    A fault, not a refusal: the report ran and something under it failed.
    Chained from that failure, whose words the message carries, so the
    person can say what went wrong when they write to support without the
    report.

    Attributes:
        report: Which ZIP was not saved.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'support_report_not_saved': None}

    words_for_person = True
    title = 'Support Report Not Saved'

    def __init__(self, report: 'SupportReportKind', cause: BaseException):
        words = str(cause) or type(cause).__name__
        super().__init__(
            f'The {report} was not saved: {words}\n\nContact {SUPPORT_ADDRESS} directly.'
        )
        self.report = report


class AlreadyStartedError(Refusal, RuntimeError):
    """A start a session makes once was asked of a session that has made it.

    A host's lifecycle call, refused before anything is started twice: a
    second plugin load would register every plugin again, and a second
    metrics start would orphan the first schedule as untracked,
    forever-ticking events. Still a ``RuntimeError``, as it was documented.

    Attributes:
        reason: ``'plugins_loaded'`` -- this session already loaded its
            plugins; it never loads them again. ``'metrics_running'`` --
            metrics are already running; ``disable_metrics`` lets them start
            again.
    """

    title = 'Already Started'
    causes: ClassVar[dict[str, RefusalCause]] = {
        'plugins_loaded': RefusalCause.REQUEST,
        'metrics_running': RefusalCause.STATE,
    }

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class OperationSettleRefusedError(Refusal, RuntimeError):
    """An operation's ``set_result`` or ``set_exception`` was called by code holding it.

    Only its producer settles an operation: a holder's settle would forge
    the outcome every other holder reads, and make the producer's own settle
    fail where it is made -- on a run's delivery, a recording's finish, a
    move's monitor. It refuses the accident and the casual forge, not a
    determined caller: code in the same process can settle through the base
    class, as the producers do. Still a ``RuntimeError``, as a Future's own
    refusal of a second settle is.

    Attributes:
        reason: ``'settled_by_its_producer'``.
        member: The member that was called.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {
        'settled_by_its_producer': RefusalCause.REQUEST
    }
    title = 'Operation Settle Refused'

    def __init__(self, member: str):
        super().__init__(
            f'{member}: an operation is settled by the work that produced it, not by its holder'
        )
        self.member = member


class BlockingCallRefusedError(Refusal, RuntimeError):
    """A blocking call was made on a thread it could end up waiting on.

    Refused before any wait is made, by where the call was made: its
    non-blocking form is the one that thread uses. Still a ``RuntimeError``,
    as it was documented.

    Attributes:
        reason: ``'blocking_on_worker'`` -- made on an executor's worker,
            whose own work it may queue behind. ``'wait_on_ui_thread'`` --
            a wait on the UI thread for an operation not yet done, which
            would hold the window until the operation ends; a done-callback
            is the UI thread's form. ``'blocking_inline'`` -- made inside
            the GUI's inline boundary, whose thread draws the window;
            ``submit_reported`` is that call's form. ``'swap_in_delivery'`` --
            a UI dispatcher swap from a callback that dispatcher is
            delivering, which the swap would wait on.
        member: The member that was called.
    """

    title = 'Blocking Call Refused'
    causes: ClassVar[dict[str, RefusalCause]] = {
        'blocking_on_worker': RefusalCause.REQUEST,
        'wait_on_ui_thread': RefusalCause.REQUEST,
        'blocking_inline': RefusalCause.REQUEST,
        'swap_in_delivery': RefusalCause.REQUEST,
    }

    def __init__(self, reason: str, member: str, *, worker: str | None = None):
        self.reason = reason
        if reason == 'blocking_on_worker':
            words = (
                f'a blocking call from the {worker} worker, which its own work may queue '
                'behind; use the non-blocking form'
            )
        elif reason == 'wait_on_ui_thread':
            words = (
                'a wait on the UI thread for an operation that has not ended, which would '
                'hold the window until it does; add a done-callback instead'
            )
        elif reason == 'swap_in_delivery':
            words = (
                'a UI dispatcher swap from a callback it is delivering, which would wait '
                'on that delivery; swap it from outside its deliveries'
            )
        else:
            words = 'a blocking dispatch from the inline run_reported; use submit_reported'
        super().__init__(f'{member}: {words}')
        self.reason = reason
        self.member = member


class ClaimStillLentError(Fault, RuntimeError):
    """A diagnostic's block ended with a run it lent its claim to still live.

    The diagnostic's end waited for the run and the run outlived the wait,
    so the claim stays with the run and is released when the run ends: the
    scope is not free yet. Raised at the block's end when the block ended
    cleanly; reported once when the block raised, so the block's own
    exception is the one its caller receives. Still a ``RuntimeError``, as
    it was documented.

    Attributes:
        timeout_s: How long the end waited for the run.
    """

    words_for_person = True
    title = 'Run Still Holds the Microscope'
    causes: ClassVar[dict[str, RefusalCause | None]] = {'claim_still_lent': None}

    def __init__(self, timeout_s: float):
        super().__init__(
            f'A run the diagnostic started was still running {timeout_s:g} s after the '
            'diagnostic ended. The microscope is released when that run ends.'
        )
        self.timeout_s = timeout_s
