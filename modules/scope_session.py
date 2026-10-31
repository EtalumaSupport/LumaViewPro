# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
ScopeSession -- GUI-independent state container for a microscope session.

Consolidates the shared state that was previously scattered across module-level
globals in lumaviewpro.py.  LumaViewPro, the REST API, and standalone scripts
can each create (or share) a ScopeSession instance and pass it to the
functions in config_helpers and Lumascope's executor-backed command API.

Usage
-----
    from modules.scope_session import ScopeSession

    session = ScopeSession.create(settings=settings, source_path=source_path)
    # or, with no GUI, from the user's configuration on disk:
    session = ScopeSession.create(ScopeSession.load_user_settings(source_path), simulate=True)
"""

import concurrent.futures
import contextlib
import copy
import dataclasses
import datetime
import enum
import os
import pathlib
import stat
import threading
import time
import typing
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

import modules.settings_init as settings_init
from drivers.simulated_camera import SimulatedStall
from lvp_logger import logger
from modules import (
    common_utils,
    kivy_utils,
    live_work,
    path_utils,
    settings_paths,
)
from modules.layer_record import refuse_unknown_layer
from modules.activity_claim import (
    SCOPE_HOLDING_KINDS,
    ActivityClaim,
    ControlSession,
    HeldClaim,
    a_guarded_write,
    acting,
    command_refused,
    control_refused,
    current_taking,
    guarded_write,
    the_holder_named,
)
from modules.caller import Caller
from modules.exceptions import (
    AlreadyStartedError,
    ArgumentRefusedError,
    ClaimStillLentError,
    CameraDriverRepairedNotice,
    CameraDriverRepairFailedError,
    CameraDriverRepairRefusedError,
    ConfigError,
    DiagnosticRefusedError,
    FileWriterNotStuckError,
    HARDWARE_STATE_REASONS,
    HardwareCommandRefusedError,
    HomingFailedError,
    LiveFolderFileNotWrittenError,
    LiveFolderPathRefusedError,
    MissingPart,
    MoveStoppedError,
    ObjectiveUnknownError,
    ProtocolNotLoadedError,
    ProtocolRunRefusedError,
    Remedy,
    RemedyName,
    ScopeDisconnectError,
    ScopeModelDeferredNotice,
    ScopeModelUnknownError,
    SettingRefusedError,
    SettingsFileNotWrittenError,
    SettingsNotSavedNotice,
)
from modules.kivy_utils import UiDispatcher
from modules.live_work import LiveWork, WorkItem, WorkKind
from modules.lumascope_api import AxisPosition, TurretStatus
from modules.lumascope_api._constants import refuse_unless_turret_slot
from modules.lumascope_api.bring_up import PartStatus
from modules.lumascope_api.illumination import (
    LedState,
    LedTransition,
    LedTransitionCtx,
    lit_other_than,
)
from modules.lumascope_api.motion import HomeTarget
from modules.finite_number import refuse_unless_wait_s
from modules.frame_validity import FrameValiditySource
from modules.lumascope_api.camera import CameraAPI, LayerCameraRequest, LayerHold
from modules.lumascope_api.frame_record import CapturedStill
from modules.lumascope_api.runtime_state import slots_holding
from modules.manual_capture import ManualCaptureController
from modules.notification_center import Notification
from modules.operation import Operation, refuse_waiting_on_ui_thread
from modules.manual_recording import ManualRecordingController
from modules.config_helpers import DisplayRates, log_environment_once
from modules.coord_transformations import PlateLocation
from modules.acquire_mode import AcquireMode
from modules.axes import StageAxis
from modules.zstack_config import ZStackReference
from modules.metrics_logger import ENGINEERING_METRICS_INTERVAL_S, MetricsLogger
from modules.plugins import PLUGIN_API_LEVEL, PluginRegistry
from modules.python_values import python_values
from modules.scheduler import Scheduler, ThreadingTimerScheduler
from modules.settings_saver import SettingsSaver, write_whole
from modules.sequential_io_executor import IOTask, refuse_blocking_on_a_worker, slow_task_budget
from modules.api_surface import (
    IN_PROCESS,
    Admission,
    FilePath,
    ProgressCallback,
    api,
    api_fields,
    mark_of,
    sets_of,
)

# How long a diagnostic's end waits for a run it lent its claim to. The
# window of one autofocus inside a characterization. Per
# PERFORMANCE_BUDGETS.md row diagnostic_exit_run_idle_wait_s.
DIAGNOSTIC_EXIT_RUN_IDLE_WAIT_S = 120.0

# How long the full support report may run before "Slow task" means
# anything: it homes, sweeps the fan, times the serial link and copies the
# files, which the report itself tells the person takes 5-10 minutes.
# Module-level because the decorator runs at class-body time. Budget row:
# support_report_slow_task_s in PERFORMANCE_BUDGETS.md.
SUPPORT_REPORT_SLOW_TASK_S = 600.0

# How long the startup motion may run before "Slow task" means anything: the
# home's declared 120 s and the turret move's 15 s, the two motions it waits
# for in turn (derived from those declarations, not measured whole).
STARTUP_MOTION_SLOW_TASK_S = 135.0

# A Session member that writes settings or files, admitted and counted by the
# session's claim (``guarded_write``).
_guarded_write = guarded_write(lambda session: session.activity_claim)

# The work a session counts while it runs, by its live-work kind: work that
# holds no claim for all of its length, so no claim names it.
_COUNTED_WORK_NAMES = {
    live_work.WorkKind.SUPPORT_REPORT: 'A support report',
    live_work.WorkKind.LOGS_ZIP: 'A logs zip',
    live_work.WorkKind.PLUGIN_RUN: 'An engineering plugin run',
}

# A protocol's Layer Settings column it is saved as, the ChannelSettings
# field it is read back as, and the layer-settings key it is saved from and
# restored into; Acquire and Stim_Enabled are handled apart.
_LAYER_SETTINGS_KEYS = (
    ('Illumination', 'illumination_ma', 'illumination_ma'),
    ('Gain', 'gain_db', 'gain_db'),
    ('Auto_Gain', 'auto_gain', 'auto_gain'),
    ('Exposure', 'exposure_ms', 'exposure_ms'),
    ('False_Color', 'false_color', 'false_color'),
    ('Sum', 'frames_summed', 'sum'),
)

# ProtocolRunner is referenced only in a return annotation; it is
# imported function-locally to avoid a circular import. Declare it here
# for the annotation without a runtime import.
if TYPE_CHECKING:
    from modules.lumascope_api import Lumascope
    from modules.lumascope_api.bring_up import BringUpRecord
    from modules.lumascope_api.diagnostics import UsbIdentityBindings
    from modules.lumascope_api.camera import AutoGainLock
    from modules.lumascope_api.simulation import SimulatedCameraWriteRefusal, SimulatedModelAnswer
    from modules.protocol import Protocol, ProtocolSizeAdvisory
    from modules.protocol_image_writer import RunWriteBatch
    from modules.lumascope_api.protocols import StepTargets, TurretSlotQuestion
    from modules.protocol_runner import ProtocolRunner
    from modules.sequential_io_executor import SequentialIOExecutor
    from modules.plugins import PluginHealth
    from modules.post_processing_api import PostProcessingAPI
    from modules.settings_init import SettingValue
    from modules.tech_support_report import SupportReportSaved
    from modules.tiling_config import TilingConfig


def simulated_stuck_write(seconds: float) -> None:
    """A write to a simulated drive that has stopped answering: it returns after *seconds*."""
    time.sleep(seconds)


def _scheduler_callback_error(exc: BaseException) -> None:
    """A scheduled callback died on its timer thread; say so loudly.

    The scheduler's default is to swallow the exception, which is the
    wrong default here: the callbacks the session schedules include the
    recording health check, whose entire purpose is loud failure. No caller
    waits on a timer, so this is where the fault's flight stops.
    """
    from modules.notification_center import notifications

    notifications.report_outcome(exc, solicited=False, category='Scheduler')


def _apply_logging(settings: dict) -> None:
    """Put the settings' log level on the main log, and the autofocus score log on or off.

    Logging is process-wide, so the settings of the session a host builds
    decide it, on every host: applied when the session is created, before
    its scope is brought up, and again when either setting is written, so
    the logger never disagrees with the store. The level goes on both
    loggers that write the main log -- the 'LVP' parent of every module's
    logger, and lvp_logger's own. ``debug_mode`` is the installation's
    switch for a detailed log: on, the level is DEBUG, as lvp_logger set it
    from the first line; off, it is ``logging.default.level``.
    """
    import logging

    import modules.autofocus_functions as autofocus_functions

    level = 'DEBUG' if settings['debug_mode'] else settings['logging']['default']['level'].upper()
    for name in ('LVP', 'lvp_logger'):
        logging.getLogger(name).setLevel(level)
    logger.info(f'[Session  ] Log level {level}')
    autofocus_functions.enable_af_score_logging(enable=bool(settings['logging']['autofocus']))


def _owner_of(member: settings_paths.SettingMember) -> type:
    """The class an owned setting's member is declared on, which the table names by its name."""
    return {'ScopeSession': ScopeSession, 'CameraAPI': CameraAPI}[member.owner]


def _tell_stored_replacements(pending: list[tuple[str, object, object]]) -> None:
    """Tell, as one notice, the stored settings a load replaced, and empty ``pending``.

    One notice per load: the notification centre shows one notice of a kind
    at a time and drops a repeat within its window, so a second notice from
    the same load would hide the first's values. Emptied as it is told, so
    each way out of a bring-up can tell it and the load is told once.
    """
    if not pending:
        return
    from modules.exceptions import StoredSettingReplacedNotice
    from modules.notification_center import notifications

    told = list(pending)
    pending.clear()
    notifications.report_outcome(
        StoredSettingReplacedNotice(told), solicited=False, category='Settings'
    )


@api_fields('choices', 'proposed', 'turret_slot')
@dataclasses.dataclass(frozen=True)
class ObjectiveQuestion:
    """Which objective is in the light path; what to ask the user.

    Returned by ``ScopeSession.objective_question`` when no one has ever
    confirmed the objective on this install, or the turret sits on a slot
    with no assignment; and by ``current_slot_objective_question`` whenever
    a person asks to say what the current turret slot holds, or on a scope
    without a turret which objective is installed. The pixel size
    derived from the objective is stamped into the scale bar and every
    saved image's metadata, and a wrong scale cannot be told from a
    measured one afterwards -- so the session exposes the question instead
    of assuming silently, and a host renders it however it likes. The answer goes back through
    ``ScopeSession.confirm_objective``.

    Attributes:
        turret_slot: The slot the answer binds to, or None on a
            non-turret model.
        proposed: The catalogue id to offer as the default.
        choices: The catalogue, in its shipped order.
    """

    turret_slot: int | None
    proposed: str
    choices: tuple[str, ...]


@api_fields('z_um', 'step_index')
@dataclasses.dataclass(frozen=True)
class SavedFocus:
    """What ``ScopeSession.save_focus`` wrote.

    Attributes:
        z_um: The Z saved as the layer's focus.
        step_index: The step that took it as its Z, or None when no step did.
    """

    z_um: float
    step_index: int | None


@api_fields('run_named', 'control_named', 'recording')
@dataclasses.dataclass(frozen=True)
class ControlsLock:
    """Which of a person's controls the Session locks now, and why.

    The Session's editing policy for the controls a person works, not an
    API refusal: a REST client may still make any call the API accepts,
    and what the API refuses it refuses whatever this says.

    Attributes:
        run_named: What holds the run surface -- a run, a diagnostic, a
            home, or a finished run's files still writing -- as a sentence
            a person reads; None while nothing does. A display that pauses
            for that work reads this, not ``run_lockout``, so it keeps
            showing the scope under an idle control session.
        control_named: The control session held, as a sentence a person
            reads; None while none is. It holds the run surface too, the
            holder's own run included.
        recording: Whether a manual recording is live. Its drain is not: a
            draining recording frees the controls while its claim still
            refuses new runs.
    """

    run_named: str | None
    control_named: str | None
    recording: bool

    @api
    @property
    def run_lockout(self) -> bool:
        """True while something holds the run surface: ``run_named`` or ``control_named``.

        A finished run's files hold it until the last one is written,
        though the run has freed its claim: the controls stay locked until
        the queue empties.

        Raises:
            Nothing.
        """
        return self.run_named is not None or self.control_named is not None

    @api
    @property
    def named(self) -> str | None:
        """The sentence for what locks the controls now, None while nothing does.

        The control session's while one is held, the holder's own run
        included; else what holds the run surface; else a live recording.

        Raises:
            Nothing.
        """
        if self.control_named is not None:
            return self.control_named
        if self.run_named is not None:
            return self.run_named
        return 'A recording is in progress.' if self.recording else None

    @api
    @property
    def controls_locked(self) -> bool:
        """True while the whole control surface locks: the run surface is held, or a recording is live.

        Raises:
            Nothing.
        """
        return self.run_lockout or self.recording


@api_fields(
    'live_work', 'lock', 'control', 'run_refused', 'axes', 'turret', 'parts', 'camera_streaming'
)
@dataclasses.dataclass(frozen=True)
class Status:
    """What the scope is doing, in one read, for a client that has just connected.

    Each field is read from its owner, one after another rather than under
    one lock, so two fields can describe slightly different moments: an
    axis can read HOMING in ``axes`` a moment before the home appears in
    ``live_work``.

    Attributes:
        live_work: Everything the session is still doing
            (``ScopeSession.live_work``).
        lock: Which of a person's controls the Session locks, and why
            (``ScopeSession.controls_lock``): its editing policy, not a
            refusal.
        control: The control session held, None when none is
            (``ScopeSession.control``).
        run_refused: The refusal a run would get now for the scope's state
            alone, None when the state refuses none
            (``SequencedCaptureRunner.state_refusal``). None does not
            promise a run starts: the protocol, an axis's position, the
            camera feed and the interlocks are judged when the run is asked.
        axes: Each stage axis's state and position, None where the
            position is not known (``MotionAPI.axis_positions``): X, Y and
            Z, in um.
        turret: The turret's state and slot (``MotionAPI.turret_status``),
            None on a scope with no turret.
        parts: Which parts came up at bring-up, and why each that did not
            (``ScopeSession.bring_up_record``).
        camera_streaming: Whether the camera is grabbing frames, asked of
            the camera itself (``ImagingAPI.is_streaming``).
    """

    live_work: LiveWork
    lock: ControlsLock
    control: ControlSession | None
    run_refused: ProtocolRunRefusedError | None
    axes: dict[StageAxis, AxisPosition]
    turret: TurretStatus | None
    parts: tuple[PartStatus, ...]
    camera_streaming: bool


@api_fields('paths')
@dataclasses.dataclass(frozen=True)
class SettingsChanged:
    """Settings one write stored, told once the write has let go of the store.

    Attributes:
        paths: The dotted paths stored, each once, in the order stored.
    """

    paths: tuple[str, ...]


class SettingsLock:
    """The settings store's lock: what a hold stored is told once the hold is released.

    Every write to the store is made under it, and ``_store_setting`` records
    each path it stores, so the paths a hold stored -- two to five for some
    members -- are told together after the release, outside the lock: never
    one by one mid-batch, never to a listener that would read under it. A
    hold that stored nothing, a reader's, tells nothing.
    """

    def __init__(self, tell: Callable[[tuple[str, ...]], None]) -> None:
        self._lock = threading.Lock()
        self._stored: list[str] = []
        self._tell = tell

    def __enter__(self) -> 'SettingsLock':
        self._lock.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        # Told even when the hold raised part-way: what it stored is stored.
        stored, self._stored = self._stored, []
        self._lock.release()
        if stored:
            self._tell(tuple(dict.fromkeys(stored)))

    def locked(self) -> bool:
        return self._lock.locked()

    def stored(self, path: str) -> None:
        """Record *path* as stored under this hold; called under the lock."""
        self._stored.append(path)


@api_fields('channel', 'lit')
@dataclasses.dataclass(frozen=True)
class ShownChannel:
    """The channel the scope shows: the channel whose camera settings are on
    the camera, and whether its LED is lit with every other one dark.

    Attributes:
        channel: The channel's key name.
        lit: Whether its LED reads lit in the LED store, whoever lit it. An
            LED whose last write went unanswered reads unknown there, and is
            not lit here; the ``led`` event carries its unknown state.
    """

    channel: str
    lit: bool


class EntryKind(enum.StrEnum):
    """What a live-folder listing's entry is."""

    FILE = 'file'
    FOLDER = 'folder'


@api_fields('name', 'kind', 'size_bytes', 'modified')
@dataclasses.dataclass(frozen=True)
class LiveFolderEntry:
    """One file or folder in a live-folder listing.

    Attributes:
        name: Its full name under the live folder, ``/``-separated on every
            host: what a path parameter, a download and a further listing
            take.
        kind: Whether it is a file or a folder.
        size_bytes: Its size in bytes, as the file system reports it.
        modified: When it was last modified, in local time with its offset.
    """

    name: str
    kind: EntryKind
    size_bytes: int
    modified: datetime.datetime


@api_fields('engineering_mode', 'manual_capture', 'manual_recording', 'post_processing', 'scope')
class ScopeSession:
    """Owns the shared, GUI-independent state for one microscope session.

    Attributes:
        engineering_mode: Whether the session runs in engineering mode: a
            saved still or run image on a scope with a turret names the
            slot it was taken at, and the system metrics are sampled more
            often unless the settings name an interval. It begins as
            ``create`` was given; a plugin may turn it on when the host
            loads plugins.
    """

    # What the session holds, set during construction.
    engineering_mode: bool
    manual_capture: ManualCaptureController
    manual_recording: ManualRecordingController
    post_processing: 'PostProcessingAPI'
    scope: 'Lumascope'

    def __init__(
        self,
        settings: dict,
        scope,
        executor_bundle,
        protocol_thread=None,
        autofocus_runner=None,
        autofocus_thread=None,
        owns_scope: bool = False,
        scheduler: Scheduler | None = None,
        engineering_mode: bool = False,
        no_engineering: bool = False,
    ):
        self.settings = settings
        # The lock lives with the dict it guards. Every host hands the same
        # dict to whatever else it composes, so a lock held anywhere else
        # can only guard one of the aliases -- which is no guard at all.
        # Readers on other threads take a snapshot; writers use
        # update_settings.
        self._settings_listeners: list = []
        self.settings_lock = SettingsLock(self._tell_settings_changed)
        # The plugins this session hosts, or None until its host asks for
        # them (load_plugins): a session nobody asked plugins for has none,
        # whatever is installed, so a script or a test is not changed by the
        # machine it runs on.
        self.plugins: PluginRegistry | None = None
        self.scope = scope
        # One scheduler per session, owned here and shared by every
        # periodic consumer -- metrics, camera-temp logging, and the
        # recording health check. Plain daemon timers on every host,
        # never a UI clock: a safety bound armed on a UI loop stops
        # when the UI freezes, and one timebase means the GUI, tests,
        # and headless all exercise the same path. Injectable so tests
        # fire callbacks by hand. Metrics stay opt-in through
        # enable_metrics; the scheduler existing does not start them.
        if scheduler is None:
            scheduler = ThreadingTimerScheduler(
                name_prefix='LVP-SessionTimer',
                on_callback_error=_scheduler_callback_error,
            )
        self._scheduler = scheduler
        # The one store for "metrics are running"; enable_metrics /
        # disable_metrics are its only writers. Host-serialized (main
        # thread in the GUI): a threaded host must serialize
        # enable_metrics / disable_metrics itself.
        self._metrics_started = False
        # Whether a shutdown() pass has COMPLETED. Written True as the last
        # statement of that pass, so a pass that raised part-way leaves a
        # retry possible; read at entry to make the second call a logged
        # no-op. Host-serialized like the metrics flag.
        self._shut_down = False
        # A diagnostic's claim whose lent run outlived the diagnostic's wait:
        # released when that run ends, so a claim is never left with nobody
        # to release it -- a close would wait on it for ever.
        self._release_at_run_idle: list[HeldClaim] = []
        self._release_at_run_idle_lock = threading.Lock()
        # Held for the whole of a close, so a close asked on another thread
        # meanwhile waits for it and then finds it done; re-entrant, so a
        # close asked from inside the close (a listener it calls) returns
        # rather than waiting on itself.
        self._close_lock = threading.RLock()
        self._close_running = False
        # Support reports, logs zips and plugin runs in flight, by kind: each
        # runs on its caller's thread -- the GUI's diagnostics lane, a
        # script's own, a plugin's -- so the members count them, and the
        # live-work read sees a script's report as it sees the GUI's.
        self._counted_work = dict.fromkeys(_COUNTED_WORK_NAMES, 0)
        self._counted_lock = threading.Lock()
        # The file every accepted settings write is saved to, and the thread
        # that saves it: both set by create when its host names a file, and
        # None otherwise, when the session saves nothing.
        self._settings_file: str | None = None
        self._settings_saver: SettingsSaver | None = None
        # The reasons a save was withheld that the session has told,
        # so each is said once (_save_settings_file).
        self._unsaved_told: set[str] = set()
        # The one store for engineering mode. Set from what the host was
        # built in; the engineering plugin turns it on when it loads.
        # no_engineering is the host's word that the engineering plugin is
        # not loaded at all (load_plugins leaves it out). Every reader --
        # a run, a still, the metrics cadence, the GUI -- reads it here.
        self.engineering_mode = engineering_mode
        self.no_engineering = no_engineering
        # The session stops the bundle it holds at shutdown(), whoever built
        # it: a caller that hands one to the constructor hands it over. The
        # IO and CAMERA lanes the bundle holds are the scope's, and stop
        # when the scope disconnects.
        self.executor_bundle = executor_bundle
        # True only when the factory BUILT the scope, False for a scope
        # handed to the constructor. Decides whether shutdown() runs the hardware half (LEDs
        # off, disconnect). Coupled to the object here, at construction,
        # because shutdown() can run before a factory returns (a bring-up
        # that raises) and a directly constructed session calls it too -- a
        # flag patched on afterwards would miss both.
        self._owns_scope = owns_scope
        # The one FILE lane, from the bundle: the run engine and
        # ProtocolRunner write through it, and the session reads the
        # file-drain facts from it.
        self.file_io_executor = executor_bundle.file_io_executor
        # The post-processing builds, on their own lane from the bundle, so
        # a build never queues in front of a run's writes on the file lane.
        from modules.post_processing_api import PostProcessingAPI

        self.post_processing = PostProcessingAPI(
            lane=executor_bundle.post_processing_executor,
            tiling_grids=scope.protocols.tiling_grids(),
            has_turret=lambda: scope.capabilities.has_turret,
            settings_snapshot=self.get_settings_snapshot,
            channel_colors=self.channel_colors,
            # Read as each build starts; the claim is made further down.
            scope_holder=lambda: self.activity_claim.holder,
            files_draining=lambda: self._draining_batch() is not None,
        )
        # Run-state listeners: zero-argument callables notified on every
        # run-state transition edge (claim grant/release, a run's return
        # to IDLE after its cleanup, file-drain exit). They fire on the TRANSITIONING thread,
        # possibly under engine locks, so a listener must only schedule
        # or re-read the level-derivation properties below -- never
        # acquire engine locks or trust edge context.
        self._run_state_listeners: list = []
        # The layer the scope shows, the one store of it: show_channel,
        # go_to_step and a run's steps write it. Whether it is lit is the
        # LED store's, read at each read, never kept here. The GUI opens on
        # BF, and so does the store.
        self._shown_layer = common_utils.DEFAULT_LAYER
        self._shown_channel_listeners: list = []
        # The layer shown as a run's first lit step found it, shown again
        # once the run's cleanup has restored the camera; None outside a run
        # whose steps have lit. Written and read on the protocol thread.
        self._shown_before_run: str | None = None
        # The step the last go_to_step previewed, as (protocol, index): a
        # repeat of it is a re-click, whose preview would put out a channel
        # the person lit in between. Forgotten at every run transition, so
        # the first click after a run previews whatever the run left.
        self._last_step_gone_to: tuple | None = None
        # The lock each layer's last turning off of auto-gain took, by
        # layer: what the camera reached, readable after the setter has
        # answered (a setting's URL answers only its read). A record of that
        # turning off, not of the layer now, so no later write forgets it.
        # In memory only. Written by set_channel_auto_gain under settings_lock.
        self._auto_gain_locks: dict[str, AutoGainLock] = {}
        # Outcome listeners this session registered on the notification
        # centre, so its shutdown takes back exactly what it gave.
        self._outcome_listeners: list = []
        # The single arbitration point for exclusive activities: a
        # protocol run and a video recording each claim here before
        # committing, so the two can never run concurrently. Enforcement
        # lives with the claimants (the sequenced-capture runner's
        # refusal gate and the recording engine's start), which take
        # this handle by injection.
        self.activity_claim = ActivityClaim(on_transition=self.notify_run_state)
        # The device lanes ask the claim before running work, so while a run,
        # a diagnostic or a home holds the scope only its own work reaches the
        # hardware, whoever submits. The IO key is kept for the one named
        # override on that lane, shutdown's LED drain; the camera key goes to
        # the scope for its temperature read, so the lanes ask before the
        # scope is serviced.
        self._io_override_key = self.io_executor.ask_claim(self.activity_claim)
        self._camera_override_key = self.camera_executor.ask_claim(self.activity_claim)
        # The scope reads the configuration it acts on -- labware, stage
        # offset, turret map, objective, scale bar -- from these settings,
        # their one store, so it holds no copy to keep in step. After the
        # claim: the lanes refuse a second session over this scope there,
        # before it could point the scope at its own settings.
        scope.bind_settings(self._read_stored_setting, self._store_settings)
        # Service the scope NOW, before any collaborator is composed: the
        # temperature log reads the camera under the key, and the protocol
        # constructors resolve their data files from the source path.
        self._register_scope_services(scope)
        # The session's periodic metrics: it holds the scheduler that starts
        # them, the bundle the watchdog snapshots and the settings the system
        # tick reads, so it builds the logger with all three at once.
        self.metrics_logger = MetricsLogger(
            scope=scope, executor_bundle=executor_bundle, settings=settings
        )
        # Manual video recording, composed with the session claim so a
        # recording and a protocol run are mutually exclusive for every
        # caller tier (GUI, L2, REST).
        self.manual_recording = ManualRecordingController(
            scope=scope,
            settings=settings,
            shown_layer=lambda: self._shown_layer,
            channel_colors=self.channel_colors,
            activity_claim=self.activity_claim,
            scheduler=self._scheduler,
            post_processing=self.post_processing,
        )
        # Manual stills: one at a time, on the camera lane, which already
        # refuses them while a run holds the camera. No activity claim -- a
        # still must not refuse Record, a run or a reconnect.
        self.manual_capture = ManualCaptureController(
            scope=scope,
            settings_snapshot=self.get_settings_snapshot,
            shown_layer=lambda: self._shown_layer,
            layer_camera_request=self._layer_camera_request,
            engineering_mode=lambda: self.engineering_mode,
            channel_colors=self.channel_colors,
        )

        # The run engine and its autofocus pair are SESSION-composed:
        # one SequencedCaptureRunner per session, shared by the GUI,
        # ProtocolRunner, and (later) REST -- a second engine instance
        # would duplicate run state beside the shared claim. Hosts
        # inject their own AF pair / protocol thread; bundle-building
        # factories construct real ones; a bare session composes an
        # engine with what it has (an AF-bearing run then refuses or
        # fails loudly at the producer site).
        self.protocol_thread = protocol_thread or executor_bundle.protocol_thread
        self.autofocus_runner = autofocus_runner
        self.autofocus_thread = autofocus_thread
        from modules.sequenced_capture_runner import SequencedCaptureRunner

        self.sequenced_capture_runner = SequencedCaptureRunner(
            scope=scope,
            protocol_thread=self.protocol_thread,
            file_io_executor=self.file_io_executor,
            autofocus_thread=autofocus_thread,
            autofocus_runner=autofocus_runner,
            activity_claim=self.activity_claim,
            channel_colors=self.channel_colors,
            post_processing=self.post_processing,
            on_run_idle=self._run_went_idle,
            on_protocol_files_written=self._run_protocol_complete_processors,
            on_step_lit=self._step_lit_by_run,
            on_run_ending=self._show_the_layer_shown_before_the_run,
        )
        self._protocol_runner = None

    @property
    def io_executor(self) -> 'SequentialIOExecutor':
        """The scope's IO lane. Read from the scope, never held as a copy.

        Composition wiring for the host that builds its context and its run
        engine around the session -- not part of the L2 API surface.
        """
        return self.scope.io_lane()

    @property
    def camera_executor(self) -> 'SequentialIOExecutor':
        """The scope's CAMERA lane. Read from the scope, never held as a copy.

        Composition wiring for the host that builds its context and its run
        engine around the session -- not part of the L2 API surface.
        """
        return self.scope.camera_lane()

    @api
    @property
    def source_path(self) -> pathlib.Path:
        """The data folder this session runs on: its scope's, never a copy.

        Raises:
            Nothing.
        """
        return self.scope.source_path

    def _register_scope_services(self, scope) -> None:
        """Register the session's services on a scope (the one bring-up).

        The camera override key and the activity claim its homes take live
        on the scope but belong to the session's composition. Construction
        comes through here so no scope the session drives can be left
        un-serviced -- the bring-up steps are spelled out exactly once.
        """
        scope.set_camera_override_key(self._camera_override_key)
        scope.set_activity_claim(self.activity_claim)

    @api(in_process=True)
    @contextlib.contextmanager
    def diagnostic_claim(self) -> Iterator[HeldClaim]:
        """Hold the scope for a diagnostic for the length of a ``with`` block.

        A diagnostic -- a characterization, the support report's hardware
        steps -- drives every axis, the LEDs and the camera directly. While
        it holds the claim, a run or a recording start is refused, the
        controls lock and the objective cannot change, exactly as during a
        run. The claim is released when the block ends, including on a
        raise, so no caller owns the release path.

        A run the diagnostic lent its claim to (``start_autofocus(claim=...)``)
        may still be live when the block ends -- a caller that raised
        between starting it and waiting on it. The release waits for that
        run to go idle first: its LED lease lives under this claim, and
        releasing underneath a live run hands the scope to the next taker
        mid-sweep. If the run is still live after the wait, this raises and
        the claim is kept with the run, released when the run ends.

        Yields:
            The held claim.

        Raises:
            DiagnosticRefusedError: ``'exclusive_activity_running'``, a
                run, a recording, a home or another diagnostic holds the
                scope. Nothing was taken.
            ControlSessionRefusedError: ``'control_session_held'``, a
                control session is held by someone else. Nothing was taken.
            ClaimStillLentError: ``'claim_still_lent'``, at the end of a
                block that ended cleanly, a run under this claim was still
                live after the wait; the claim is released when the run ends.
                When the block raised, its own exception is raised and this is
                reported once instead. Still a ``RuntimeError``.
        """
        held = self.activity_claim.try_claim(WorkKind.DIAGNOSTIC)
        if held is None:
            holder = self.activity_claim.holder
            kind = holder.kind if holder is not None else None
            # The holder can release between the failed take and this read;
            # the refusal still stands, it just cannot name who refused it.
            named = the_holder_named(holder)
            raise DiagnosticRefusedError(
                reason='exclusive_activity_running',
                message=f'{named} is using the microscope. Let it finish, then start the diagnostic.',
                holder=kind,
                holder_trigger=(holder.run_started_by if holder is not None else None),
            )
        # Set by the block's own raise, never read from sys.exc_info(): a
        # clean block opened inside a caller's except would read as raised.
        block_raised = False
        try:
            with acting(held):
                yield held
        except BaseException:
            block_raised = True
            raise
        finally:
            if not self.sequenced_capture_runner.wait_for_run_idle(DIAGNOSTIC_EXIT_RUN_IDLE_WAIT_S):
                with self._release_at_run_idle_lock:
                    self._release_at_run_idle.append(held)
                # The run can have ended between the wait and the hand-over,
                # its idle already told: then nobody else will release it.
                if not self.sequenced_capture_runner.run_in_progress():
                    self._run_went_idle()
                kept = ClaimStillLentError(DIAGNOSTIC_EXIT_RUN_IDLE_WAIT_S)
                if not block_raised:
                    raise kept
                # A raise here would replace the block's own exception, the
                # one its caller is owed; the kept claim is told once instead.
                from modules.notification_center import notifications

                notifications.report_outcome(kept, solicited=False, category='Diagnostic')
            else:
                held.release()

    @api
    def take_control(self, name: str) -> str:
        """Take the one control session: from now on only work in it writes to the scope.

        While it is held every other caller's write -- a lane command, a
        setting, a file, a run, a recording, a home, a diagnostic -- is
        refused, the display preferences aside; a stop never is. Work in the
        session is a call made inside ``acting_in_control`` with its id, or
        a REST request naming it. Taken only while nothing of anyone else's
        is under way, so no one's work is ever put under a client's control:
        no other session, no run, recording, home or diagnostic, no setting
        or file being written, and no run or recording that has ended still
        finishing its files.

        Args:
            name: What the person at the scope is told holds it.

        Returns:
            The session's id, told to its taker and to no one else: what a
            call in the session names, and what ends it. ``control`` reads
            the session without it.

        Raises:
            ControlSessionRefusedError: ``'control_session_held'``, another
                session is held; ``'exclusive_activity_running'``, a run, a
                recording, a home or a diagnostic holds the scope;
                ``'write_in_flight'``, a setting or a file is being written;
                ``'work_finishing'``, a run or recording that has ended is
                still finishing, or a build is on the post-processing lane,
                which a capture in the session would overlap. Nothing was
                taken.
            SessionClosingError: ``'session_closing'``, the session's close
                is running or has finished.
        """
        runner = self.sequenced_capture_runner
        controller = self.activity_claim.take_control(
            name,
            finishing=lambda: (
                runner.run_live or bool(self._finishing_work()) or self.post_processing.has_work()
            ),
        )
        return controller.id

    @api
    def end_control(self, session_id: str) -> ControlSession:
        """End the control session named by its id: its taker ends it, or the person at the scope.

        The session ends at once, and the run or recording taken in it is
        stopped: it unwinds under its own claim and holds the scope until it
        has, as any stopped run does. A run or recording begun after it is
        not stopped. A home or a diagnostic taken in it ends on its own. A
        write of the session's already under way finishes whole, and no new
        session is taken until it has. A request naming the ended session
        is refused from now on.

        Args:
            session_id: The ``id`` the session was taken with.

        Returns:
            The session ended.

        Raises:
            ControlSessionRefusedError: ``'control_session_ended'``, no
                session is held, or the live one has another id.
        """
        ended = self.activity_claim.end_control(session_id)
        self._stop_the_work_taken_for(ended)
        return ended

    @api(in_process=True)
    def take_back_control(self) -> ControlSession | None:
        """End whatever control session is held, and stop the run or recording taken in it, as ``end_control``.

        The person at the scope taking it back without naming it -- the
        GUI's Take back control press and its close, which hold no id. Kept
        off the wire: a client ends only the session whose id it was told.

        Returns:
            The session ended, or None when none was held.

        Raises:
            Nothing.
        """
        ended = self.activity_claim.end_live_control()
        if ended is not None:
            self._stop_the_work_taken_for(ended)
        return ended

    def _stop_the_work_taken_for(self, ended: ControlSession) -> None:
        """Stop the run or recording taken for the session just ended; nothing begun after it."""
        # Ended before its work is stopped: the run or recording still holds
        # the slot until it unwinds, so nothing is taken in between.
        self.sequenced_capture_runner.stop_the_run_taken_for(ended)
        self.manual_recording.stop_the_recording_taken_for(ended)

    @api(in_process=True)
    @contextlib.contextmanager
    def acting_in_control(self, session_id: str) -> Iterator[ControlSession]:
        """Act in the control session named by its id for the length of a ``with`` block.

        Every call the block makes on this thread is the session's, and so
        is the lane work it hands on. A callback the API runs on a thread of
        its own -- an operation's done-callback -- runs outside the block,
        and is refused as any non-holder's call while the session is held.

        Args:
            session_id: The ``id`` the session was taken with.

        Yields:
            The session.

        Raises:
            ControlSessionRefusedError: ``'control_session_ended'``, no
                session is held, or the live one has another id. The block
                did not run.
        """
        controller = self.activity_claim.controller(session_id)
        with acting(controller):
            yield controller.session

    @_guarded_write
    def as_a_guarded_write[R](self, write: Callable[[], R]) -> R:
        """Run ``write`` as one of this session's guarded writes; returns what it returns.

        For a host's own write that no Session member makes: the REST
        server letting go of a handle a caller was given. It is refused
        while a control session is held by someone else, as every Session
        write is, and before ``write`` runs.

        It is not part of the L2 API surface.

        Raises:
            ControlSessionRefusedError: ``'control_session_held'``. ``write``
                did not run.
        """
        return write()

    @api
    @property
    def control(self) -> ControlSession | None:
        """The control session held now, None when none is.

        Raises:
            Nothing.
        """
        return self.activity_claim.control

    @api(reads=True)
    def held_by_other(self, run: 'Operation | None' = None) -> bool:
        """Whether the scope is held by anything but *run*, the operation a run member returned.

        What a run control greys on while leaving its own run's Stop live
        (``live_work`` names what holds the scope).

        Args:
            run: The operation a run member returned, whose own hold is not
                counted. None, the default, asks whether anything holds the
                scope at all.

        Raises:
            Nothing.
        """
        return self.sequenced_capture_runner.held_by_other(run)

    @api
    def last_run(self) -> 'Operation | None':
        """The last run's operation, live or ended, whoever started it; None before any run.

        Every ``ProtocolRunner`` start is a run: a scan, a protocol, a
        composite, an autofocus, a z-stack. A client that learns of a run
        from the ``status`` edges -- one the GUI, a script or the SDK
        started -- reads it here, and on the wire it is that run's one job,
        whoever reads it: its progress, its outcome and its ``job_ended``,
        which a run already ended tells at the read. Live or ended, since a
        run goes idle before its operation settles, and one short against
        the client's round trip would otherwise end unread. Its per-step
        events stay with its starter.

        A POST, not a read: it hands out a running call. A finished job a
        client deleted answers 404 on its id.

        Raises:
            Nothing.
        """
        return self.sequenced_capture_runner.last_run_operation()

    # ------------------------------------------------------------------
    # Run-state facts and derivations
    #
    # Each FACT has exactly one owner (the claim, the recording engine,
    # the file writer, the scope config); everything a consumer needs is
    # a synchronous DERIVATION over them. All reads are lock-free
    # attribute/queue reads, so these properties are safe from any
    # thread, including inside a transition listener.
    # ------------------------------------------------------------------

    @api(in_process=True)
    @property
    def live_work(self) -> LiveWork:
        """Everything the session is still doing, each piece from its owner.

        What holds the scope (a run, a recording, a home, a diagnostic and
        any run acting under its lent claim -- and a run starting or
        unwinding with no holder yet or any more);
        then what finishes after it lets go: a recording's file, a run's
        video steps, a finished run's images and its post-run builds;
        post-processing builds running and queued; a support report or logs
        zip; a plugin's run, from its press to its end; a still
        being saved. Each item says how much it has left, and
        a build how far it has got. Read lock-free, owner by owner, from any
        thread; ``closing`` while the close runs, ``closed`` once the
        session has shut down. A wire client reads it in ``status``.

        Raises:
            Nothing.
        """
        work: list[WorkItem] = []
        runner = self.sequenced_capture_runner
        holder = self.activity_claim.holder
        if holder is not None:
            work.append(WorkItem(holder.kind, the_holder_named(holder)))
            # A run lent the holder's claim (an autofocus inside a
            # diagnostic) holds no claim of its own, so it is named here
            # beside the activity it runs inside.
            lent_run = self.activity_claim.run_holder
            if lent_run is not None and lent_run is not holder:
                work.append(WorkItem(live_work.WorkKind.PROTOCOL, the_holder_named(lent_run)))
        elif runner.run_live:
            work.append(WorkItem(live_work.WorkKind.PROTOCOL, 'A run, starting or ending'))
        work.extend(self._finishing_work())
        work.extend(self.post_processing.work())
        for kind, count in self._counted_work_now().items():
            if count:
                work.append(WorkItem(kind, _COUNTED_WORK_NAMES[kind], left=count))
        if self.manual_capture.in_flight:
            work.append(WorkItem(live_work.WorkKind.STILL, 'A still being saved'))
        return LiveWork(
            work=tuple(work), closing=self.activity_claim.closing, closed=self._shut_down
        )

    def _finishing_work(self) -> list[WorkItem]:
        """What finishes after its activity let the scope go: a recording's file, a run's video and images.

        Read lock-free, and never through the activity claim, so the claim
        can ask it under its own lock (``take_control``).
        """
        work: list[WorkItem] = []
        runner = self.sequenced_capture_runner
        recording = self.manual_recording
        if recording.is_busy and not recording.is_recording:
            work.append(
                WorkItem(
                    live_work.WorkKind.RECORDING_FINISH,
                    "A recording's file",
                    left=recording.pending_writes,
                    discardable=True,
                )
            )
        if runner.video_drain_busy:
            work.append(
                WorkItem(
                    live_work.WorkKind.RUN_VIDEO_FINISH,
                    "A run's video",
                    left=runner.video_pending_writes,
                    discardable=True,
                )
            )
        files = self._draining_batch()
        if files is not None:
            work.append(
                WorkItem(live_work.WorkKind.RUN_FILES, "A run's images", left=files.pending)
            )
        return work

    @api
    @property
    def status(self) -> Status:
        """What the scope is doing: its work, its axes, its parts and its camera, in one read.

        The read a client makes first and polls as its health check. Each
        field comes from its owner, one after another rather than under one
        lock (``Status``); no field does serial I/O.

        Raises:
            RunCheckFailedError: ``'hardware_state_unknown'``, the
                connection read behind ``run_refused`` crashed instead of
                answering.
        """
        return self.status_at_edge()()

    def status_at_edge(self) -> Callable[[], Status]:
        """The status as of now but for the camera's answer, which the returned call reads.

        For a run-state listener, which is told on the thread making the
        transition and must not wait there: every field but
        ``camera_streaming`` is read on the calling thread, at the edge, and
        the camera is asked when the call is made, on a thread that may
        wait (a grab stop holds the camera SDK for as long as the camera
        takes to stop). ``status`` makes the call at once. Composition
        wiring for a host's event stream, not part of the L2 API surface.

        Raises:
            RunCheckFailedError: ``'hardware_state_unknown'``, the
                connection read behind ``run_refused`` crashed instead of
                answering.
        """
        at_edge = {
            'live_work': self.live_work,
            'lock': self.controls_lock,
            'control': self.activity_claim.control,
            'run_refused': self.sequenced_capture_runner.state_refusal(self.activity_claim),
            'axes': self.scope.motion.axis_positions(),
            'turret': self.scope.motion.turret_status(),
            'parts': self.scope.bring_up_record().parts,
        }
        return lambda: Status(**at_edge, camera_streaming=self.scope.imaging.is_streaming())

    @contextlib.contextmanager
    def _count_in_flight(self, kind: str):
        """Count a support report, logs zip or plugin run from its entry to its exit."""
        with self._counted_lock:
            self._counted_work[kind] += 1
        try:
            yield
        finally:
            with self._counted_lock:
                self._counted_work[kind] -= 1

    def _counted_work_now(self) -> dict[str, int]:
        """The counted work in flight, by kind, read at once."""
        with self._counted_lock:
            return dict(self._counted_work)

    @api(in_process=True)
    @contextlib.contextmanager
    def plugin_run_in_flight(self) -> Iterator[None]:
        """Count a plugin's run as live work from entry to exit, entered and left on any threads.

        For a plugin whose run starts on one thread and ends on a worker it
        starts: entered at the press, before the worker, and left as the
        worker ends, so ``live_work`` names the run
        across the gap before its own claim and after it -- a confirmation
        asked first, the worker's end. A run that holds a claim takes it as
        it always did; this holds no claim and refuses nothing, and the
        close waits for it as for any live work, after the plugins are told
        to stop.

        Raises:
            Nothing.
        """
        with self._count_in_flight(live_work.WorkKind.PLUGIN_RUN):
            yield

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def discard_close_drain(self) -> None:
        """Drop every video frame still queued in either drain, loudly.

        The one escape from waiting on the video frames ``live_work`` lists
        -- a recording's file, a run's video -- for a close or any caller;
        each file is finished with what was written, and frames already on
        disk stay.

        Raises:
            Nothing.
        """
        self.manual_recording.discard_pending()
        self.sequenced_capture_runner.discard_video_pending()

    def _draining_batch(self) -> 'RunWriteBatch | None':
        """A finished run's write batch from the run's end until its last file is written; else None.

        None while the run is live -- the run's own state answers then --
        and once its files are all on disk or given up on.
        """
        batch = self.sequenced_capture_runner.write_batch()
        return batch if batch is not None and batch.draining else None

    @api
    @property
    def protocol_files_stalled(self) -> bool:
        """True while a run's files are draining and the write in flight has
        stopped making progress -- by the same stall threshold the run
        refusal uses, so what a display says of a stuck writer and what a
        new run is refused for cannot disagree.

        Raises:
            Nothing.
        """
        from modules.protocol_image_writer import WRITE_STALL_FATAL_S

        batch = self._draining_batch()
        return batch is not None and batch.stalled(WRITE_STALL_FATAL_S)

    @api
    @property
    def protocol_files_stuck_write(self) -> str:
        """The write in flight on the file lane, named for a stall report.

        Raises:
            Nothing.
        """
        return self.file_io_executor.describe_running_task()

    @api(in_process=True)
    @property
    def controls_lock(self) -> ControlsLock:
        """Which of a person's controls the Session locks now, and why: its editing policy, not a refusal.

        The run surface is held by a run by its kind, a diagnostic as one (a
        characterization and the support report share its claim), a home,
        a finished run's files still writing, or a control session; the
        whole surface also by a live manual recording. The one rule every
        lock read reads; a wire client reads it in ``status``. It locks for
        the holder of a control session too: it is the GUI's editing
        policy, and the GUI is never the holder.

        Raises:
            Nothing.
        """
        holder = self.activity_claim.holder
        if holder is not None and holder.kind in SCOPE_HOLDING_KINDS:
            run_named = f'{the_holder_named(holder)} is in progress.'
        elif self._draining_batch() is not None:
            run_named = "A protocol's files are still being written."
        else:
            run_named = None
        control = self.activity_claim.control
        return ControlsLock(
            run_named=run_named,
            control_named=f'{control.name} has remote control.' if control is not None else None,
            recording=self.manual_recording.is_recording,
        )

    def in_flight_named(self) -> list[str]:
        """What a gesture started that is still running, each as a sentence; empty when nothing is.

        The read a sim walk's ``wait`` makes before its next step, and the
        one place the answer is composed, so the GUI reads nothing below the
        Session: what holds the scope (``controls_lock``); each queued
        lane with a task running or queued, since a turret pick or a build
        blocks on its lane; an axis still travelling, since a jog, a go-to
        or a stage click starts its move on the lane and the lane is idle
        while the stage moves; an autofocus sweep, which holds no scope
        claim; a plugin run from its press to its end
        (``plugin_run_in_flight``), which holds its claim for only part of
        that. Composition wiring for the walk host, not part of the L2 API
        surface.
        """
        still = []
        holding = self.controls_lock.run_named
        if holding is not None:
            still.append(holding)
        for name, lane in self.executor_bundle.queued_executors():
            if lane.is_busy():
                still.append(f'the {name} lane ({lane.describe_running_task()})')
        if self.scope.motion.is_any_axis_moving():
            still.append('an axis moving')
        if self.autofocus_thread.is_running:
            still.append('an autofocus sweep')
        if self._counted_work_now()[live_work.WorkKind.PLUGIN_RUN]:
            still.append(_COUNTED_WORK_NAMES[live_work.WorkKind.PLUGIN_RUN].lower())
        return still

    @api(in_process=True, event=Status)
    def add_run_state_listener(self, listener: Callable[[], None]) -> None:
        """Register a run-state transition listener and level-sync it.

        The immediate call is the level republish: transitions are
        edges, and a listener registered after a grant would otherwise
        never see it. A wire client is sent the session's ``status`` at
        each edge, in the order the edges happen: every field read at the
        edge but the camera's, which is asked just after
        (``status_at_edge``); a field that changes with no edge (an axis,
        a part, the camera) is as of that edge.

        Args:
            listener: Called with no arguments at each run-state transition,
                and once now; it re-reads whatever state it shows.

        Raises:
            Nothing.
        """
        self._run_state_listeners.append(listener)
        listener()

    @api(in_process=True)
    def remove_run_state_listener(self, listener: Callable[[], None]) -> None:
        """Stop telling ``listener`` of run-state transitions; a listener never added is a no-op.

        Args:
            listener: The callable ``add_run_state_listener`` registered.

        Raises:
            Nothing.
        """
        with contextlib.suppress(ValueError):
            self._run_state_listeners.remove(listener)

    def _run_went_idle(self) -> None:
        """The engine's run has ended: release any claim waiting on it, then tell the listeners."""
        with self._release_at_run_idle_lock:
            waiting, self._release_at_run_idle = self._release_at_run_idle, []
        for held in waiting:
            held.release()
        self.notify_run_state()

    def notify_run_state(self) -> None:
        """Notify every run-state listener (level semantics: listeners
        re-read the derivations; an extra notification is harmless).

        Each listener's raise is reported and the rest are still told: no
        caller waits on a notification, so this is where its fault stops.
        """
        from modules.notification_center import notifications

        self._last_step_gone_to = None
        notifications.tell_listeners(list(self._run_state_listeners), category='Run State')

    @api(in_process=True, event=Notification)
    def add_outcome_listener(self, listener: Callable[['Notification'], None]) -> None:
        """Hear every outcome the scope reports from now on.

        Each ``Notification`` carries its ``kind``, words, ``reason``,
        ``remedy``, ``outcome_id`` and ``shown``: an outcome muted when it
        happened (an unattended run, the dedup window, shutdown) arrives with
        ``shown`` False, and arrives again with the same ``outcome_id`` and
        ``shown`` True if it is later shown.

        Bring-up has happened by the time a session exists; pass the listener
        to ``create(outcome_listener=...)`` to hear it. The outcomes heard are
        every session's in this process, as the notification centre is one
        per process. ``shutdown()`` removes the listener.

        Args:
            listener: Called as ``listener(notification)`` once per delivery,
                on the thread that reported it, so it must return promptly
                and must not wait on the scope.

        Raises:
            Nothing.
        """
        from modules.notification_center import Severity, notifications

        notifications.add_listener(listener, min_severity=Severity.DEBUG)
        self._outcome_listeners.append(listener)

    @api(in_process=True)
    def remove_outcome_listener(self, listener: Callable[['Notification'], None]) -> None:
        """Stop ``listener`` hearing outcomes; a listener never added is a no-op.

        Args:
            listener: The callable ``add_outcome_listener`` (or ``create``'s
                ``outcome_listener``) registered.

        Raises:
            Nothing.
        """
        from modules.notification_center import notifications

        notifications.remove_listener(listener)
        self._outcome_listeners = [cb for cb in self._outcome_listeners if cb != listener]

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @api(in_process=True)
    @staticmethod
    def set_ui_dispatcher(dispatcher: UiDispatcher | None) -> None:
        """Set how the process hands a callback to its UI thread.

        A process has one UI thread, so this is one setting for the process,
        not for a session: every lane's completion callback, every run
        delivery and every listener the GUI marshals reads it at the moment
        it dispatches, whichever session or scope it belongs to. A GUI host
        sets it once, before it builds a session; a host with no UI thread
        (a script, a REST server) never sets it. Callbacks still waiting for
        the UI thread go to the new dispatcher in their order, or with None
        run here, each raise reported.

        Args:
            dispatcher: What hands each callback to the UI thread:
                ``dispatcher.thread`` is the thread ``schedule`` delivers on,
                and a run's ``wait()`` on that thread is refused, since what
                it waits for is delivered there. None (the default) calls
                each callback directly on the thread that dispatches it, and
                a callback's raise is reported rather than raised.

        Raises:
            BlockingCallRefusedError: ``'swap_in_delivery'``, called from a
                callback the dispatcher is delivering, since the swap waits
                for that delivery to finish.
        """
        kivy_utils._set_ui_dispatcher(dispatcher)

    @api(in_process=True)
    @classmethod
    def create(
        cls,
        settings: dict,
        source_path: FilePath | None = None,
        *,
        simulate: bool = False,
        warn_pre_release: bool = True,
        engineering_mode: bool = False,
        no_engineering: bool = False,
        sim_model: str | None = None,
        sim_model_answer: 'SimulatedModelAnswer | None' = None,
        sim_refused_stream_starts: int = 0,
        sim_refused_camera_writes: 'tuple[SimulatedCameraWriteRefusal, ...]' = (),
        sim_camera_stall: 'SimulatedStall | None' = None,
        sim_file_stall: 'SimulatedStall | None' = None,
        sim_focal_spread_um: float = 0.0,
        outcome_listener: Callable[['Notification'], None] | None = None,
        settings_file: FilePath | None = None,
    ) -> 'ScopeSession':
        """Create a session, constructing defaults for any missing components.

        This is the one composition path: the GUI, REST and scripts all
        build their session here, passing what only a host knows as the
        keyword arguments below. The factory builds the scope, so every
        session's scope is brought up the one way. The session's data folder
        and catalogues are always its scope's.

        The executor bundle is always built, around the scope's own IO and
        CAMERA lanes, via executor_registry.create_default, so every caller
        gets the topology the GUI runs: FILE + WORKER_POOL executors plus
        protocol_thread (started) and scope_display_thread (constructed,
        not started).

        The settings' log level and autofocus score log are applied to the
        process's logging first, so every host logs as its settings say,
        bring-up included.

        The scope is brought up before this returns (``configure_scope``,
        then the camera start gate released) and is torn down by
        ``shutdown()``.

        Args:
            settings: The user configuration the session holds and configures
                the scope from (``load_user_settings`` reads the
                installation's). It is completed from the shipped template
                as a stored file is, and every value in it is held to the
                ranges a write is.
            source_path: The data folder the factory builds the scope on;
                None (the default) is the installation's own folder.
            simulate: build a simulated scope (the model from
                ``settings['microscope']``, the motor board's tier from
                ``settings['simulator_tier']``).
            warn_pre_release: whether this construction fires the
                pre-release FutureWarning -- the factory's own call and the
                scope constructor's. A host that ships with the API passes
                False; a separately shipped caller leaves the default.
            engineering_mode: the session's engineering mode to begin with;
                a plugin may turn it on when the host loads plugins.
            no_engineering: ``load_plugins`` leaves out the installed
                plugin named ``etaluma_engineering`` before it is imported,
                as if it were not installed.
            sim_model: the model the simulated scope is, when it is not the
                one ``settings`` declares: a stored file that disagrees
                with the scope attached, which bring-up settles by the
                scope's answer. None (the default) is the declared model.
                Refused unless ``simulate``.
            sim_model_answer: the model the simulated motor board names as
                its own at bring-up, whatever it was built as
                (``scope.simulation.answer_model``), until the scope's
                faults are cleared. Refused unless ``simulate``, and by the
                scope on any board but the fast motor board.
            sim_refused_stream_starts: how many of the camera's stream
                starts it refuses from bring-up on
                (``scope.simulation.refuse_stream_start``), so bring-up's
                own start is refused; 0 (the default) refuses none. Refused
                unless ``simulate``, and by the scope on any camera but the
                simulated FX2.
            sim_refused_camera_writes: camera writes the simulated camera
                refuses from bring-up on (``scope.simulation.refuse_camera_write``,
                each in turn), so bring-up's own writes meet them; empty (the
                default) refuses none. Refused unless ``simulate``, and by
                the scope on the FX2 camera.
            sim_camera_stall: a stall for the simulated camera's stream, so a
                simulated scope shows a stream that stops delivering; refused
                by the scope itself unless it is simulated with the
                simulated camera.
            sim_file_stall: a stall for the session's file lane, so a
                simulated scope shows a save drive that stops answering:
                ``after_s`` seconds after bring-up the lane's worker is held
                for ``for_s`` seconds. Refused unless ``simulate``.
            sim_focal_spread_um: how far the simulated sample's focus may
                lie from where bring-up places the stage, drawn once per
                session, so every session needs a real autofocus; 0 (the
                default) puts it in focus there. Refused as
                ``sim_camera_stall`` is.
            outcome_listener: heard from before the scope is built, so it
                hears what bring-up reports; the session's
                ``add_outcome_listener`` says what it receives. A factory
                that raises takes it back; otherwise the session's
                ``shutdown()`` does.
            settings_file: the file the session saves its settings to,
                one second after the first unsaved write and once at
                ``shutdown()``, whole or not at all; the hosts name the
                installation's ``user_settings_file(source_path)``. None (the default)
                saves nothing: a script's or a test's handed dict stays in
                memory. Writes are kept unsaved, and the session says so
                once (``SettingsNotSavedNotice``), while the settings are
                provisional or when the scope found no hardware at
                bring-up. Saving starts once the scope is up, so a factory
                that raises writes no file.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``, a ``sim_``
                argument without ``simulate``: a real scope's file lane
                writes to a real drive, and only a simulated scope's boards
                and camera take a fault. ``'needs_simulated_camera'``,
                ``sim_camera_stall`` for a model simulated with an FX2.
                ``'not_injectable_on_board'``, ``sim_model_answer``,
                ``sim_refused_stream_starts`` or ``sim_refused_camera_writes``
                for a scope whose board or camera cannot show it;
                ``'fault_bound'``, a negative ``sim_refused_stream_starts``
                or a refusal's count below 1; ``'not_offered'``, a node or an
                answer the camera's write does not give. ``'wrong_kind'``, a block of
                ``settings`` that is not the kind the shipped template holds
                (``argument`` names it).
            SettingRefusedError: ``'wrong_kind'`` or ``'out_of_range'``, a
                value in ``settings`` the writer refuses: a dict handed to a
                session is a write of every key at once, held to the same
                ranges.
            ScopeInUseError: ``'scope_in_use'``, a session with
                ``simulate=False`` asked for while another LumaViewPro, REST
                server or script drives the microscope, or while another
                real scope in this process is open. Nothing was opened.
        """
        from modules.lumascope_api._lumascope import _fire_pre_release_warning
        from modules.path_utils import get_source_root

        if not simulate:
            for argument, value in (
                ('sim_model', sim_model),
                ('sim_model_answer', sim_model_answer),
                ('sim_refused_stream_starts', sim_refused_stream_starts or None),
                ('sim_refused_camera_writes', sim_refused_camera_writes or None),
                ('sim_file_stall', sim_file_stall),
            ):
                if value is not None:
                    raise ArgumentRefusedError(
                        'needs_simulated_scope', argument=argument, value=value
                    )
        # The one preparation every settings dict gets: a dict a caller
        # hands in is completed from the shipped template as the file is,
        # so nothing below hedges for a key it lacks, and is held to the
        # writer's ranges as a write is.
        root = get_source_root(source_path)
        template = settings_init.load_template(logger, str(root / 'data' / 'settings.json'))
        settings = settings_init.prepare_loaded(
            logger, settings, template, str(root), source='settings', handed=True
        )
        # Taken before the dict is any session's store: the preparation's
        # record of the old channel blocks it renamed, for the bring-up to
        # place by the unit.
        old_channel_blocks = settings_init.take_old_channel_blocks(settings)
        _apply_logging(settings)
        if warn_pre_release:
            _fire_pre_release_warning()

        from modules.notification_center import Severity, notifications

        if outcome_listener is not None:
            notifications.add_listener(outcome_listener, min_severity=Severity.DEBUG)
        # What the settings preparation replaced, taken now so a second
        # create (the GUI's fallback on the template) never tells this load's;
        # told once, with what bring-up replaced, however this ends.
        replaced = [
            replacement
            for notice in settings_init.take_stored_replacements()
            for replacement in notice.replacements
        ]
        try:
            import modules.lumascope_api as lumascope_api

            scope = lumascope_api.Lumascope(
                simulate=simulate,
                warn_pre_release=warn_pre_release,
                configured_model=settings.get('microscope'),
                sim_model=sim_model,
                sim_tier=cls._simulator_tier(settings) if simulate else 'fast',
                fx2_debug_wire=settings['fx2_debug_wire_enabled'],
                source_path=root,
                sim_camera_stall=sim_camera_stall,
                sim_focal_spread_um=sim_focal_spread_um,
            )
            # Armed before the bring-up below, which is what they are for.
            try:
                if sim_model_answer is not None:
                    scope.simulation.answer_model(sim_model_answer.model)
                if sim_refused_stream_starts != 0:
                    scope.simulation.refuse_stream_start(sim_refused_stream_starts)
                for refusal in sim_refused_camera_writes:
                    scope.simulation.refuse_camera_write(
                        refusal.node, refusal.answer, refusal.count
                    )
            except BaseException:
                cls._report_teardown_failure(scope.disconnect)
                raise

            from modules.executor_registry import create_default

            executor_bundle = create_default(
                scope.io_lane(),
                scope.camera_lane(),
            )

            # Service registration (the camera override key) happens in
            # __init__ for every session-composed scope -- nothing here.

            autofocus_runner, autofocus_thread = cls._build_autofocus_pair(scope=scope)

            # The ownership fact goes in HERE, before _bring_up can call
            # shutdown on a refusal: a session torn down mid-factory must
            # already know the scope is its own.
            try:
                session = cls(
                    settings=settings,
                    scope=scope,
                    executor_bundle=executor_bundle,
                    autofocus_runner=autofocus_runner,
                    autofocus_thread=autofocus_thread,
                    owns_scope=True,
                    engineering_mode=engineering_mode,
                    no_engineering=no_engineering,
                )
            except BaseException:
                # No session exists to tear down, so stop what this factory
                # started.
                autofocus_thread.stop(timeout=2.0)
                executor_bundle.shutdown()
                cls._report_teardown_failure(scope.disconnect)
                raise
            if outcome_listener is not None:
                session._outcome_listeners.append(outcome_listener)
            cls._bring_up(session, replaced, old_channel_blocks)
            if sim_file_stall is not None:
                session._hold_the_file_lane(sim_file_stall)
            if settings_file is not None:
                session._start_saving_to(settings_file)
        except BaseException:
            _tell_stored_replacements(replaced)
            # Give the listener back: a host that composes again after this
            # raise (the GUI's fallback to the shipped defaults) would
            # otherwise hear every outcome twice.
            if outcome_listener is not None:
                notifications.remove_listener(outcome_listener)
            raise
        _tell_stored_replacements(replaced)
        return session

    def _hold_the_file_lane(self, stall: 'SimulatedStall') -> None:
        """At ``stall.after_s``, hold the file lane's worker for ``stall.for_s``.

        A simulated save drive that stops answering: the lane's one worker
        sits in a task that does not return for the stall's length, and the
        writes behind it wait, as they would behind a write to an
        unresponsive drive. The write path itself is not told; the lane
        judges the stall as it judges any other.
        """
        from modules.sequential_io_executor import IOTask

        held = threading.Event()
        handles = []

        def _hold() -> None:
            if held.is_set():
                return
            held.set()
            self.file_io_executor.put(
                IOTask(action=simulated_stuck_write, kwargs={'seconds': stall.for_s})
            )
            for handle in handles:
                self._scheduler.unschedule(handle)

        handles.append(self._scheduler.schedule_interval(_hold, stall.after_s))
        if held.is_set():
            # Fired before its handle was kept (a stall with no delay).
            self._scheduler.unschedule(handles[0])

    @staticmethod
    def _simulator_tier(settings: dict) -> str:
        """The simulated motor board's tier from the settings, resolved for
        this machine.

        The setting is the user's choice and is refused when it names no
        tier. The firmware tier needs a MicroPython runtime on this
        machine; where there is none -- an unsupported platform, or a Linux
        machine that has not built it -- the session runs the fast tier and
        says why, because a simulated scope that cannot start on a
        developer's machine is worse than one on the lighter tier. A
        runtime that is present but broken raises further down.
        """
        from drivers.sim_wire.backend import DEFAULT_DIALECT, runtime_missing

        tier = settings['simulator_tier']
        missing = runtime_missing(DEFAULT_DIALECT) if tier != 'fast' else None
        if missing is not None:
            logger.warning(
                f'[Session  ] simulator_tier is {tier}, but there is no MicroPython runtime '
                f'here ({missing}): running the fast tier'
            )
            return 'fast'
        return tier

    @api(in_process=True)
    @staticmethod
    def load_user_settings(source_path: FilePath) -> dict:
        """The user's configuration, as the GUI would configure a scope from it.

        For a host with no GUI -- a script, a server -- to pass to
        ``create``: ``create(ScopeSession.load_user_settings(root),
        simulate=...)``. Reading is its own call rather than a default of
        ``create`` so a caller that meant to pass settings and forgot is
        refused for the missing argument instead of quietly configured
        from whatever is on disk.

        Args:
            source_path: The installation folder whose settings are read:
                its ``current.json`` first, else its shipped ``settings.json``.
                In a process that has already prepared its settings (the
                GUI's), those are answered and ``source_path`` is not read.

        Raises:
            ArgumentRefusedError: ``'not_an_installation'``, ``source_path``
                holds neither settings file.
            SettingsFileError: the user's ``current.json`` is unusable.
            InstallationFileError: the installation's own ``scopes.json``,
                whose layer vocabulary the settings are checked against, or
                its ``settings.json`` template, is missing or unusable.
        """
        from modules.settings_init import settings as default_settings

        if default_settings is not None:
            return default_settings.copy()
        # Settings not loaded yet (e.g. headless/test usage) -- resolve
        # the same file the GUI reads (current.json first, then
        # settings.json) so headless state matches the running app,
        # instead of hardcoding settings.json and ignoring live state.
        # The same preparation the GUI runs -- shape check, folds,
        # repairs, default merge -- not just the file read. Reading
        # alone yields a dict that parses and is silently missing
        # whatever newer releases added to the template.
        #
        # A directory with no shipped template is not an installation:
        # a session configured from an empty dict would have no frame
        # and no objective, so it refuses here, naming the root. An
        # unusable current.json surfaces the same way: the GUI answers
        # that by asking the user, and there is nobody to ask here.
        try:
            settings, _rejected = settings_init.prepare_settings(
                logger, source_path, fall_back_to_template=False
            )
        except FileNotFoundError as e:
            raise ArgumentRefusedError(
                'not_an_installation', argument='source_path', value=source_path
            ) from e
        return settings

    @api(in_process=True)
    @staticmethod
    def user_settings_file(source_path: FilePath) -> str:
        """The file the installation at ``source_path`` keeps its user configuration in.

        For a host that saves accepted writes, to pass to ``create``:
        ``create(settings, source_path=root,
        settings_file=ScopeSession.user_settings_file(root))``, as the GUI
        and the REST server do. ``load_user_settings`` reads this file
        first. Nothing is read or created here.

        Args:
            source_path: The installation folder, as ``create`` and
                ``load_user_settings`` take it.

        Raises:
            Nothing.
        """
        return settings_init.current_json_path(source_path)

    @staticmethod
    def _report_teardown_failure(teardown: typing.Callable[[], None]) -> None:
        """Run a teardown on a path that is already failing, and report a
        part that did not shut down instead of letting it out.

        The fault that brought the caller here is the one its own caller
        must see; a disconnect failure raised from inside the handler would
        replace it and survive only as its context.
        """
        from modules.notification_center import notifications

        try:
            teardown()
        except ScopeDisconnectError as e:
            notifications.report_outcome(e, solicited=False, category='Hardware')

    @classmethod
    def _bring_up(
        cls, session: 'ScopeSession', replaced: list, old_channel_blocks: dict[str, str]
    ) -> None:
        """Configure the scope a factory built; ``initialize`` releases the
        camera start gate last, after the capture pixel format. A raise
        anywhere in here leaves the caller with no session object to tear
        down, so this tears down what the factory started before it lets
        the raise out. What the bring-up replaces is added to ``replaced``,
        the load's list, for the factory to tell once.

        The old channel settings are placed first, so the configuration and
        the absent-layer check judge each where this unit holds it."""
        try:
            session._place_old_channel_blocks(old_channel_blocks)
            session._configure_scope(replaced)
        except BaseException:
            # Before the teardown, which takes back the listeners that hear it.
            _tell_stored_replacements(replaced)
            cls._report_teardown_failure(session.shutdown)
            raise
        session._stop_acquiring_absent_layers()
        # The one marker for "the session is up": a host measures its own
        # consumer's start against it. It says whether the camera is
        # grabbing, which a launch without a camera does not.
        streaming = session.scope.imaging.is_streaming()
        logger.info(
            '[Session  ] bring-up complete: scope configured, '
            f'camera {"streaming" if streaming else "not streaming"}'
        )

    @staticmethod
    def _build_autofocus_pair(*, scope):
        """Real AF runner + started AF thread for a factory-built session,
        so every host gets the same wiring."""
        from modules.autofocus_runner import AutofocusRunner
        from modules.autofocus_thread import AutofocusThread

        autofocus_runner = AutofocusRunner(scope=scope)
        autofocus_thread = AutofocusThread(afe=autofocus_runner)
        autofocus_thread.start()
        return autofocus_runner, autofocus_thread

    # ------------------------------------------------------------------
    # Convenience wrappers (delegate to config_helpers / scope_commands)
    # ------------------------------------------------------------------

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def recover_file_writer(self) -> int:
        """Give up on a finished run's unwritten images and unlock a stuck writer.

        When a protocol run's file writer stops making progress, the run
        engine reports the stall once (``FileWriterStalledError``) and
        every subsequent run is refused with the ``files_writing_stalled``
        reason until the writer is recovered or the app restarts; both
        carry this method as their remedy, and any caller may call it by
        name. The run's outstanding images are given up on and counted,
        and the worker stuck mid-write is abandoned and replaced. Nothing
        else queued on the lane is discarded.

        Returns:
            How many of the run's images were given up on.

        Raises:
            HardwareCommandRefusedError: a run, a diagnostic or a home holds the
                scope: the writer is recovered only while the scope is
                held by nothing, so no live run's captures are given up on.
            FileWriterNotStuckError: no write has stopped making progress;
                the writer will finish on its own, and recovering would
                lose images for nothing.
        """
        from modules.protocol_image_writer import WRITE_STALL_FATAL_S

        holder = self.activity_claim.owner
        if holder in SCOPE_HOLDING_KINDS:
            raise HardwareCommandRefusedError(
                'exclusive_activity_running', 'recover_file_writer', holder
            )
        batch = self.sequenced_capture_runner.write_batch()
        if batch is None or not batch.stalled(WRITE_STALL_FATAL_S):
            raise FileWriterNotStuckError(batch.pending if batch is not None else 0)
        # Given up on first, so the stuck write returning later counts
        # nothing and the writes queued behind it skip their turn.
        abandoned = batch.abandon('File writer recovery')
        self.file_io_executor.replace_stuck_worker()
        return abandoned

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def repair_camera_driver(self) -> 'tuple[UsbIdentityBindings, ...]':
        """Install LumaViewPro's driver for a Lumascope Classic and bind its USB identities to it.

        Offered as the remedy of the bring-up's ``camera_driver_unusable``
        on an installed Windows build that carries the Classic driver, and
        callable by name from any client. It runs the installed driver step
        as an administrator, on the scope PC's desktop: Windows asks there
        for an administrator's approval, then whether to install software
        from Etaluma (unless the person chose to always trust it), so someone
        at the scope answers whoever asked. The step logs every action to the
        install folder's ``logs``. A camera that is up is never touched:
        Repair runs only while no camera is connected and a present device is
        on a driver LumaViewPro cannot use, and the camera connects on its new
        driver at the next start, which the notice it reports says.

        Returns:
            tuple[UsbIdentityBindings, ...]: The bindings read after the
                step, as ``scope.diagnostics.get_camera_usb_bindings``.

        Raises:
            CameraDriverRepairRefusedError: ``'repair_needs_windows'``,
                ``'repair_needs_installed_build'``,
                ``'repair_needs_driver_package'`` (built without the Classic
                driver), ``'no_classic_bindings'``, ``'driver_already_usable'``
                (a camera is up, or every present device is on a driver
                LumaViewPro can use), ``'repair_declined'`` when the
                administrator's approval was not given, or
                ``'driver_not_accepted'`` when Windows did not take the
                package (declined at its question, or refused).
            CameraDriverRepairFailedError: The step could not start, did not
                finish, ended unsuccessfully, or left a present device on a
                driver LumaViewPro cannot use; the words name its log.
            CameraUsbLookupFailedError: The bindings could not be read.
        """
        from modules import fx2_driver_step
        from modules.notification_center import notifications
        from modules.path_utils import launch_root

        reason = fx2_driver_step.repair_refusal()
        if reason is not None:
            raise CameraDriverRepairRefusedError(reason)
        # A camera that is up is on a driver that works, and the step
        # reinstalls every present device, so it would be pulled from under
        # whatever is using it; an unbound bootloader identity beside it is
        # another vendor's board, not ours to take.
        bindings = self.scope.diagnostics.get_camera_usb_bindings()
        if bindings is None:
            raise CameraDriverRepairRefusedError('no_classic_bindings')
        if self.scope.camera_connected or all(identity.usable for identity in bindings):
            raise CameraDriverRepairRefusedError('driver_already_usable')
        step = launch_root() / fx2_driver_step.DRIVER_STEP_EXE
        exit_code = fx2_driver_step.run_as_administrator(
            step, 'install', fx2_driver_step.REPAIR_WAIT_S
        )
        if exit_code is None:
            raise CameraDriverRepairRefusedError('repair_declined')
        if exit_code == fx2_driver_step.NOT_ACCEPTED_EXIT:
            raise CameraDriverRepairRefusedError('driver_not_accepted')
        logs = step.parent / 'logs'
        if exit_code != 0:
            raise CameraDriverRepairFailedError(
                f'the driver step ended with exit code {exit_code}', logs
            )
        after = self.scope.diagnostics.get_camera_usb_bindings()
        if not all(identity.usable for identity in after):
            raise CameraDriverRepairFailedError(
                'the driver step finished and a device is still on a driver LumaViewPro cannot use',
                logs,
            )
        notifications.report_outcome(
            CameraDriverRepairedNotice(), solicited=True, category='Camera'
        )
        return after

    @api
    def apply_remedy(self, remedy: Remedy) -> 'int | tuple[UsbIdentityBindings, ...]':
        """Take the action a refusal named as its remedy, and return its answer.

        The one place a remedy's name becomes an action: the GUI's offer and
        a REST caller answer a refusal through this, never by resolving the
        name themselves. A name arrives as data -- from a refusal another
        layer built, or from a caller sending it back -- and a ``Remedy``
        holds only a ``RemedyName``, so only the members listed here are
        reachable through it. The member still decides whether the remedy
        applies now, and refuses if not.

        Args:
            remedy: The remedy the refusal carried; its ``member`` names the
                member taken: ``recover_file_writer`` or
                ``repair_camera_driver``.

        Returns:
            The remedy's own answer: ``recover_file_writer`` answers how many
            of the run's images it gave up on, ``repair_camera_driver`` the
            bindings after the repair.

        Raises:
            HardwareCommandRefusedError: ``'exclusive_activity_running'``,
                as ``recover_file_writer``.
            FileWriterNotStuckError: as ``recover_file_writer``.
            CameraDriverRepairRefusedError: ``'repair_needs_windows'`` and the
                other reasons ``repair_camera_driver`` names.
            CameraDriverRepairFailedError: as ``repair_camera_driver``.
            CameraUsbLookupFailedError: as ``repair_camera_driver``.
        """
        remedies: dict[RemedyName, Callable[[], int | tuple[UsbIdentityBindings, ...]]] = {
            RemedyName.RECOVER_FILE_WRITER: self.recover_file_writer,
            RemedyName.REPAIR_CAMERA_DRIVER: self.repair_camera_driver,
        }
        return remedies[remedy.member]()

    def get_layer_configs(self, specific_layers: list | None = None) -> dict:
        """The stored settings of each layer, or of ``specific_layers``.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'`` or
                ``'no_channel_selected'``, a name in ``specific_layers`` is
                not a layer.
        """
        import modules.config_helpers as config_helpers

        for layer in specific_layers or ():
            refuse_unknown_layer(layer)

        layer_configs = config_helpers.get_layer_configs(self.settings, specific_layers)
        self._read_autofocus_as_off_without_z(layer_configs.values())
        return layer_configs

    def _read_autofocus_as_off_without_z(self, layer_entries: Iterable[dict]) -> None:
        """Turn off the autofocus switch in copies read from a scope with no Z.

        Autofocus moves Z, so a run that asks for it on this scope is
        refused, and the GUI shows no autofocus control here. A switch
        saved on from a scope with Z is then one the user can neither see
        nor clear; read as it stands, every step and new protocol would
        carry it and every run would be refused. The entries are copies;
        this writes nothing back to the saved settings.
        """
        if self.scope.capabilities.has_focus:
            return
        for entry in layer_entries:
            entry['autofocus'] = False

    @api(reads=True)
    def saved_focus_z_um(self, channel: str) -> float:
        """The Z saved as ``channel``'s focus, in micrometres.

        Args:
            channel: The channel (layer) whose saved focus is read.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a channel.
            FocusNotSavedError: no focus was ever saved for ``channel``; a step
                built for it takes the current Z instead.
        """
        from modules.exceptions import FocusNotSavedError

        refuse_unknown_layer(channel)
        focus = self.settings[channel]['focus']
        if focus is None:
            raise FocusNotSavedError(channel)
        return focus

    @api(in_process=True, reads=True)
    def get_zstack_params(self) -> dict:
        """The stored z-stack, what New and Add stack with.

        ``range`` and ``step_size`` in micrometres, and ``z_reference``, where
        a step's Z sits in its stack, a ``ZStackReference``. The store holds
        each to its writer's rule at load and at every write.

        Raises:
            Nothing.
        """
        import modules.config_helpers as config_helpers

        return config_helpers.get_zstack_params_from_settings(self.settings)

    @api(reads=True)
    def channel_colors(self) -> dict[str, tuple[float, float, float]]:
        """Every layer of the release and the colour it is shown and written
        in, as red, green and blue weights, each 0 to 1.

        The one read of a channel's colour: every capture, recording, build
        and view takes it from here, the run and each build as it starts.
        A layer's ``display_color`` setting, a ``'#rrggbb'`` colour at full
        brightness, overrides its colour; null is no override. Without one,
        a layer this unit carries takes the unit's colour (its record's,
        which a board's Channels block may give), and any other layer the
        release's, so a folder captured on another unit still has one.

        Raises:
            InstallationFileError: the installation's ``scopes.json`` is
                unusable (``layer_record.release_layers``).
        """
        from modules.layer_record import parse_display_color, release_colors

        colors = release_colors()
        for record in self.scope.channel_identity.channels:
            colors[record.key_name] = parse_display_color(record.display_color)
        with self.settings_lock:
            overrides = {key: self.settings[key]['display_color'] for key in colors}
        for key, override in overrides.items():
            if override is not None:
                colors[key] = parse_display_color(override)
        return colors

    def get_stim_configs(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_stim_configs(self.settings)

    def get_auto_gain_settings(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_auto_gain_settings(self.settings)

    def get_sequenced_capture_config(
        self,
        *,
        tiling: str = '1x1',
        use_zstacking: bool = False,
    ) -> dict:
        """The sequenced capture config for this session's settings.

        What ``new_protocol`` builds from, in-process; a caller of the API
        builds through ``new_protocol``. Tiling and z-stacking are arguments
        rather than stored settings: neither survives a restart, so there is
        nothing for a session to read them from and a caller states what it
        wants.

        Its ``current_z`` is Z's position, or None while Z has none, so a
        config never carries a number nobody read; ``new_protocol`` refuses
        that before a step would be saved at it. While Z moves it is refused,
        so a config never carries a place Z was passing through. A scope
        with no Z motor takes the plate read's answer.

        Raises:
            ObjectiveUnknownError: the objective in the light path is unknown:
                ``'slot_unknown'``, ``'slot_unassigned'`` or
                ``'not_in_catalogue'`` on a turret scope; ``'none_selected'``
                with no turret.
            ProtocolScheduleRefusedError: the stored period or duration is one
                no protocol can run.
            ConfigError: the stored z-stack will not parse (as
                ``get_zstack_params``).
            AxisStateUnknownError: a scope with no Z motor, and an axis it has
                does not know its position (as ``get_current_plate_position``).
            AxisMovingError: ``'axis_moving'``, Z is moving, or, on a scope
                with no Z motor, an axis the plate read reads.
            HardwareCommandRefusedError: ``'not_connected'``, the model's
                motor controller is not connected.
        """
        import modules.config_helpers as config_helpers

        z = self.scope.motion.axis_positions().get('Z')
        if z is None:
            current_z = self.get_current_plate_position().z_um
        elif z.position_um is None:
            current_z = None
        else:
            current_z = self.scope.motion.settled_positions(('Z',))['Z']
        return config_helpers.get_sequenced_capture_config_from_settings(
            self.capture_settings_snapshot(),
            objective_helper=self.scope.objective_helper,
            wellplate_loader=self.scope.wellplate_loader,
            current_z=current_z,
            tiling=tiling,
            use_zstacking=use_zstacking,
        )

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def load_protocol(self, file_path: FilePath) -> 'Protocol':
        """Load the protocol at ``file_path`` and put the scope on its plate.

        A protocol's positions are stated against the plate it names, so it
        is handed back only with the scope on that plate: the plate is
        selected through ``select_labware``, and a refused selection refuses
        the load. On a scope with no XY stage there is no plate to be on; the
        protocol and the scope both take "Center Plate".

        Its contract is the plate. The protocol's period, duration and
        per-layer settings are the protocol's own and are not applied to
        anything else.

        Args:
            file_path: The protocol file (``.tsv``) to read.

        Raises:
            LiveFolderPathRefusedError: ``'path_not_found'``, nothing is at
                ``file_path``; ``'not_a_file'``, a folder is.
            ProtocolNotLoadedError: As ``ProtocolsAPI.load_protocol``.
            ProtocolFormatError: As ``ProtocolsAPI.load_protocol``.
            ProtocolRunRefusedError: ``'objective_not_mounted'``,
                ``'objectives_require_turret'``,
                ``'turret_objectives_unassigned'`` or
                ``'layer_not_on_scope'``: the file holds a step this scope
                cannot perform; ``'channel_unknown'``, a step names a
                channel no release has; ``'values_out_of_range'``, its Layer
                Settings hold a value no layer setting takes; each as
                ``ProtocolsAPI.load_protocol``.
            HardwareCommandRefusedError: ``'exclusive_activity_running'``,
                as ``select_labware``; the scope stays on the plate it had.
        """
        protocol = self.scope.protocols.load_protocol(file_path=file_path)
        self.scope.protocols.set_labware(protocol, protocol.labware_id())
        self.select_labware(protocol.labware_id())
        return protocol

    def _layers_on_scope(self) -> set[str]:
        """The layers this scope has, by key name: none when unresolved."""
        return {record.key_name for record in self.scope.channel_identity.channels}

    def _place_old_channel_blocks(self, moved: dict[str, str]) -> None:
        """Place the settings preparation renamed from old channel names by this unit's LEDs.

        An old name meant an LED: preparation, before any scope existed,
        moved each old block to its stock key (``settings_init.
        fold_old_channel_blocks``; ``moved`` is its record, each old name
        and its stock key), and that is this unit's key for the LED
        only where its LEDs are the stock filterset's. Where the unit's key
        differs (``ChannelIdentity.unit_key``: a unit whose own block puts
        488 nm on LED 0), the block the stock key holds becomes the unit's
        key's, every source read before any is written, and a stock key no
        LED of this unit now claims takes the template's block. Logged, not
        told: no value is lost, and the moves happen only on a unit whose
        block is its own.
        """
        if not moved:
            return
        identity = self.scope.channel_identity
        placed = {stock: identity.unit_key(old) for old, stock in moved.items()}
        if all(unit == stock for stock, unit in placed.items()):
            return
        template = self.scope.settings_template
        with self.settings_lock:
            blocks = {stock: copy.deepcopy(self.settings[stock]) for stock in placed}
            for stock, unit in placed.items():
                self._store_setting(unit, blocks[stock])
            for stock in set(placed) - set(placed.values()):
                self._store_setting(stock, template[stock])
        logger.info(
            "[Session   ] Placed the old channel settings on this unit's LEDs: "
            + ', '.join(
                f'{old} -> {placed[stock]}'
                for old, stock in moved.items()
                if placed[stock] != stock
            )
        )

    def _stop_acquiring_absent_layers(self) -> None:
        """Turn acquiring off on every layer this scope does not have.

        A saved setting from another scope (a configuration carried between
        machines, or one written before this scope's layers were known) can
        have such a layer acquiring, and New, Add and a composite would then
        build steps for it. The stored value is replaced by what this scope
        can deliver, and each replacement is logged.
        """
        present = self._layers_on_scope()
        with self.settings_lock:
            for name in common_utils.get_layers():
                if name in present or self.settings[name].get('acquire') is None:
                    continue
                logger.warning(
                    f'[Session   ] {name} was set to acquire '
                    f'{self.settings[name]["acquire"]!r}, but this scope '
                    f'({self.scope.channel_identity.model}) has no {name} layer; '
                    'it is set to acquire nothing.'
                )
                self._store_setting(f'{name}.acquire', None)

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def apply_channel_settings(self, protocol: 'Protocol') -> None:
        """Put a protocol's Layer Settings into this session's layer controls.

        The other half of ``save_protocol``, and what the GUI's Load does
        once ``load_protocol`` has accepted the plate. Every layer stops
        acquiring and stimulating; each layer the protocol names then takes
        its acquire mode and every value its row holds. A blank value leaves
        that layer's control as it was. A layer this release does not know
        (not in ``common_utils.get_layers()``), or this scope does not have,
        is logged and dropped.

        Args:
            protocol: The protocol whose Layer Settings are applied, as
                ``load_protocol`` returned it or as built in memory.

        Raises:
            ProtocolFormatError: A protocol built in memory carries a
                Layer Settings cell that is not of its column's type; no
                layer is changed. A loaded protocol was refused at load.
            ProtocolRunRefusedError: ``'values_out_of_range'``, a row it
                would apply holds a value outside its layer setting's range,
                each named by layer, column and bound; no layer is changed.
                A loaded protocol was refused at load.
        """
        self.scope.protocols.refuse_layer_settings_out_of_range(protocol)
        rows = protocol.channel_settings()
        layers = common_utils.get_layers()
        present = self._layers_on_scope()
        with self.settings_lock:
            for name in layers:
                self._store_setting(f'{name}.acquire', None)
                if self.settings[name].get('stim_config') is not None:
                    self._store_setting(f'{name}.stim_config.enabled', False)
            for name, row in rows.items():
                if name not in layers:
                    logger.warning(
                        f'[Session   ] Protocol carries settings for unknown layer '
                        f'{name!r}; that layer is dropped on load.'
                    )
                    continue
                if name not in present:
                    logger.warning(
                        f'[Session   ] Protocol carries settings for {name}, which this '
                        f'scope ({self.scope.channel_identity.model}) does not have; '
                        'that layer is dropped on load.'
                    )
                    continue
                self._store_setting(f'{name}.acquire', row.acquire)
                for _, field, key in _LAYER_SETTINGS_KEYS:
                    if getattr(row, field) is not None:
                        self._store_setting(f'{name}.{key}', getattr(row, field))
                stim = self.settings[name].get('stim_config')
                if row.stim_enabled is not None and isinstance(stim, dict):
                    self._store_setting(f'{name}.stim_config.enabled', row.stim_enabled)

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def save_protocol(self, protocol: 'Protocol', file_path: FilePath) -> pathlib.Path:
        """Write a protocol to a file, with this session's Layer Settings.

        The block holds each layer set to acquire an image or a video, with
        the values its controls hold now; ``apply_channel_settings`` puts them
        back. What the GUI opens at its next start-up is not changed: a
        script's scratch save is not the person's protocol.

        Args:
            protocol: The protocol whose steps are written.
            file_path: The file written, exactly as given: it ends in
                ``.tsv`` (in any case), and nothing is added to it.

        Returns:
            The path written; a missing folder is made.

        Raises:
            LiveFolderPathRefusedError: ``'not_a_file'``, a folder is at
                ``file_path``; ``'wrong_extension'``, its name does not end
                in ``.tsv``; ``'not_a_folder'``, a file is where its folder
                would be. Nothing is written.
            ProtocolNotSavedError: The file could not be written; a file
                already at the path is unchanged.
        """
        path = pathlib.Path(file_path)
        layer_settings = {}
        with self.settings_lock:
            for name in common_utils.get_layers():
                layer = self.settings[name]
                if layer['acquire'] is None:
                    continue
                stim = layer.get('stim_config')
                layer_settings[name] = {
                    'Layer': name,
                    'Acquire': layer['acquire'],
                    **{column: layer[key] for column, _, key in _LAYER_SETTINGS_KEYS},
                    'Stim_Enabled': (
                        stim['enabled'] if isinstance(stim, dict) and 'enabled' in stim else ''
                    ),
                }
        protocol.to_file(file_path=path, layer_settings=layer_settings)
        return path

    @api(sets='channel_acquire', admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def set_channel_acquire(self, channel: str, mode: AcquireMode | None) -> None:
        """Set what a layer captures: an ``AcquireMode``, or None (nothing).

        The layers set to acquire are the ones ``new_protocol`` and
        ``add_step`` build steps for and a composite merges. A layer set to
        acquire stops stimulating: one layer does not capture and
        stimulate at once.

        Nothing is changed on any refusal. Setting a layer to acquire
        nothing is always admitted.

        Args:
            channel: The layer set.
            mode: What it captures, an ``AcquireMode``; None sets it to
                capture nothing.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a
                channel; ``'not_offered'``, ``mode`` is not an
                ``AcquireMode`` or None.
            ConfigurationDefectError: ``'layers_unresolved'``, this scope's layers
                could not be resolved.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, ``mode`` is not
                None and this scope does not have ``channel``.
        """
        refuse_unknown_layer(channel)
        if mode is not None:
            self.scope.channel_identity.refuse_unless_on_scope(
                channel, 'set_channel_acquire', then='be set to acquire'
            )
        with self.settings_lock:
            self._write_layer_acquire(channel, mode)

    @api(reads=True)
    def channel_acquire(self, channel: str) -> AcquireMode | None:
        """What ``channel`` captures: an ``AcquireMode``, or None (nothing).

        Args:
            channel: The layer read.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a channel.
        """
        refuse_unknown_layer(channel)
        mode = self.settings[channel]['acquire']
        return None if mode is None else AcquireMode(mode)

    def _write_layer_acquire(self, layer: str, mode: AcquireMode | None) -> None:
        """Under ``settings_lock``: ``layer`` acquires ``mode``, and stops stimulating if it does."""
        self._store_setting(f'{layer}.acquire', mode)
        if mode is not None and self.settings[layer].get('stim_config') is not None:
            self._store_setting(f'{layer}.stim_config.enabled', False)

    @api(sets='channel_auto_gain', admitted_by=(Admission.LANE_CALL, Admission.GUARDED_WRITE))
    @_guarded_write
    def set_channel_auto_gain(self, channel: str, enabled: bool) -> 'AutoGainLock | None':
        """Turn a layer's auto-gain on or off, as the GUI's Auto Gain/Exp box does.

        For the shown layer the camera follows when this returns: turning it
        on arms the camera's auto loop (``apply_layer_camera``); turning it
        off locks the standing arm (``scope.camera.lock_auto_gain``) and
        stores what the camera reached as the layer's manual setting: the
        lock's ``gain_db`` and its ``stored_exposure_ms``, rounded to 0.1 dB
        and 0.01 ms, the resolution the stored settings carry, both when the
        lock reached them; with no arm standing, or a lock that found
        nothing usable, they are left as they were. Either waits on the
        camera lane. For a layer the camera does not hold, only the
        preference is stored: the camera's arm is another layer's, and its
        values are not this layer's to take. While a run, a diagnostic or a
        home holds the scope, turning auto-gain on is stored only, and
        turning the shown layer's off is refused.

        Nothing is changed on any refusal.

        Args:
            channel: The layer whose auto-gain is set.
            enabled: True turns it on, False off.

        Returns:
            The lock when turning off (its ``state`` is None when no arm
            stood), so a caller can read the state the camera reached, and
            ``channel_auto_gain_lock`` answers it from then on; None when
            turning on.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a channel.
            ConfigurationDefectError: ``'layers_unresolved'``, this scope's layers
                could not be resolved.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, ``enabled`` is
                True and this scope does not have ``channel``; or the lock's
                ``'exclusive_activity_running'``, when a run, a diagnostic
                or a home holds the scope.
        """
        from modules.lumascope_api.camera import AutoGainLock

        refuse_unknown_layer(channel)
        lock = None
        holds = self.scope.camera.get_held_layer() == channel
        if enabled:
            self.scope.channel_identity.refuse_unless_on_scope(
                channel, 'set_channel_auto_gain', then='run auto-gain'
            )
        elif holds:
            lock = self.scope.camera.lock_auto_gain()
        else:
            lock = AutoGainLock(state=None)
        with self.settings_lock:
            if lock is not None and lock.gain_db is not None:
                self._store_setting(f'{channel}.gain_db', round(lock.gain_db, 1))
                self._store_setting(f'{channel}.exposure_ms', round(lock.stored_exposure_ms, 2))
            self._store_setting(f'{channel}.auto_gain', enabled)
            if lock is not None:
                self._auto_gain_locks[channel] = lock
        if (
            channel == self._shown_layer
            and (enabled or holds)
            and not self._an_activity_holds_the_scope()
            and self.scope.camera_connected
        ):
            # Armed for the shown layer; or, after the lock, the camera
            # recorded as holding the layer at the values just stored.
            self.apply_layer_camera(channel)
        return lock

    @api(reads=True)
    def channel_auto_gain(self, channel: str) -> bool:
        """Whether ``channel``'s auto-gain is on, as ``set_channel_auto_gain`` set it.

        Args:
            channel: The layer read.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a channel.
        """
        refuse_unknown_layer(channel)
        return self.settings[channel]['auto_gain']

    @api(reads=True)
    def channel_auto_gain_lock(self, channel: str) -> 'AutoGainLock | None':
        """What the camera reached the last time ``channel``'s auto-gain was turned off.

        The lock ``set_channel_auto_gain(channel, False)`` returned, the last
        time it was called: its ``state`` says whether the camera converged,
        hit the exposure ceiling or floor, or reported nothing usable, and is
        None when no auto-gain was running or the camera did not hold the
        layer. It describes that turning off, not the layer now: a later
        turning on, a step's values or a stored gain leave it as it was.
        None when the layer's auto-gain has not been turned off since the
        session started.

        Args:
            channel: The layer read.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a channel.
        """
        refuse_unknown_layer(channel)
        with self.settings_lock:
            return self._auto_gain_locks.get(channel)

    def apply_layer_camera(self, layer: str) -> dict | None:
        """Put ``layer``'s stored exposure, gain and auto-gain on the camera, and wait.

        Nothing is written when the camera already holds ``layer`` at its
        stored settings (``scope.camera.hold_layer``).

        The one way a layer's settings reach the camera outside a run,
        in-process: ``show_channel`` (the public way), bring-up's BF,
        ``go_to_step``'s step layer, a write to the shown layer and a run's
        end each call it. A stored auto-gain
        arms the live auto loop, capped to the layer's channel class and the
        installation's override, as the GUI's Auto Gain/Exp box does; a
        camera without hardware auto-gain applies the layer manually.

        Returns:
            What ``scope.camera.hold_layer`` returns: the gain and exposure
            now in effect.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, ``layer`` is not a layer.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, this scope has
                no ``layer``.
            ConfigError: this scope's layers could not be resolved. Nothing is applied.
            HardwareCommandRefusedError: a run, a diagnostic or a home holds the
                scope (an autofocus is a run), or ``'not_connected'``, naming
                the camera, with none connected; nothing is applied.
            CameraSettingRejected: the camera refused a setting; the others
                were applied.
        """
        self.scope.channel_identity.refuse_unless_on_scope(
            layer, 'apply_layer_camera', then='be applied to the camera'
        )
        return self.scope.camera.hold_layer(layer, self._layer_camera_request(layer))

    @api(in_process=True)
    def capture_still(
        self,
        channel: str | None = None,
        *,
        force_to_8bit: bool = True,
        accept_dark: bool = False,
        exclude_sources: tuple[FrameValiditySource, ...] = (),
        all_ones_check: bool = False,
        timeout_s: float = 0.0,
        sum_count: int = 1,
        sum_delay_s: float = 0.0,
        sum_iteration_callback: Callable[[], None] | None = None,
    ) -> CapturedStill:
        """Capture a frame-valid image and hand it back with its record.

        In-process: the image does not travel the wire.

        Args:
            channel: The shown layer, whose stored exposure, gain and
                auto-gain the camera holds for the grab, put on in the
                capture's own camera-lane task; no other layer is taken, so
                the camera holds the shown layer after it. After a raw camera
                write (``CameraAPI.set_gain_db``, ``set_exposure_ms``,
                ``apply``) the camera holds no layer's settings, and none are
                written: the raw values stand, and the record's
                ``held_channel`` is None. None (the default) captures with the
                camera as set.
            force_to_8bit: Convert the image to 8 bits; False keeps the
                camera's depth (``CapturedStill.significant_bits``).
            accept_dark: Skip the dark-frame judgment: a frame with
                essentially no lit pixel while a channel is lit is kept at
                once, its record's ``dark`` None. False (the default) retries
                such a frame until ``timeout_s`` and then keeps it, its
                record's ``dark`` True.
            exclude_sources: Hardware changes to ignore for validity, e.g.
                ``(FrameValiditySource.Z_MOVE,)``. The frame returned can
                predate an excluded change, so a frame that is measured or
                recorded never excludes one.
            all_ones_check: Measure the frame's saturation: a blown frame is
                grabbed again once and, blown again, kept; the record carries
                the fraction either way. False leaves both None.
            timeout_s: How long, in seconds, each grab keeps taking frames
                for one that passes its checks, from 0 to 30 days.
            sum_count: How many frames are summed into the image.
            sum_delay_s: The wait, in seconds, between summed frames.
            sum_iteration_callback: Called with no arguments after each
                summed frame; None calls nothing.

        Returns:
            CapturedStill: The image and the instrument's record of its frame.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a
                layer; ``'not_a_number'`` or ``'out_of_range'``, a time is not
                a wait from 0 to 30 days, or the wait for the capture and
                every frame of its sum together is longer. Nothing is captured.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, this scope
                has no ``channel``; ``'not_connected'``, naming the camera,
                with none connected or one removed during the capture;
                ``'camera_not_streaming'``, its feed is stopped;
                ``'channel_not_shown'``, ``channel`` is not the shown layer,
                nothing captured; ``'other_led_lit'``, given ``channel``, an
                LED other than its own is lit before the grab (nothing
                captured) or lit the frame;
                ``'exclusive_activity_running'``, a run, a diagnostic or a
                home holds the scope and this call is not made under it.
            ConfigurationDefectError: ``'layers_unresolved'``, this scope's
                layers could not be resolved. Nothing is captured.
            CaptureFailedError: no frame passed, with the capture's own
                reason (``scope.imaging.capture_and_wait``).
            CameraSettingRejected: given ``channel``, the camera refused one
                of its settings; nothing was captured.
        """
        for name, value in (('timeout_s', timeout_s), ('sum_delay_s', sum_delay_s)):
            refuse_unless_wait_s(value, name)
        hold = None
        if channel is not None:
            self.scope.channel_identity.refuse_unless_on_scope(
                channel, 'capture_still', then='be captured'
            )
            if channel != self._shown_layer:
                raise HardwareCommandRefusedError('channel_not_shown', 'capture_still')
            hold = LayerHold(layer=channel, request=self._layer_camera_request(channel))
        still = self.scope.imaging.capture_still(
            hold,
            force_to_8bit=force_to_8bit,
            accept_dark=accept_dark,
            exclude_sources=exclude_sources,
            all_ones_check=all_ones_check,
            timeout_s=timeout_s,
            sum_count=sum_count,
            sum_delay_s=sum_delay_s,
            sum_iteration_callback=sum_iteration_callback,
        )
        if channel is not None and lit_other_than(
            self.scope.illumination, channel, still.record.illumination_ma
        ):
            raise HardwareCommandRefusedError('other_led_lit', 'capture_still')
        return still

    def _layer_camera_request(self, layer: str) -> LayerCameraRequest:
        """``layer``'s stored gain, exposure and auto-gain, with the auto-gain
        bounds its channel class and the installation's override give it."""
        import modules.config_helpers as config_helpers

        with self.settings_lock:
            stored = self.settings[layer]
            gain_db = stored['gain_db']
            exposure_ms = stored['exposure_ms']
            auto_gain = stored['auto_gain']
            auto_gain_settings = config_helpers.get_auto_gain_settings(self.settings)
            ceilings = copy.deepcopy(self.settings['ag_ae_max_exposure_ms'])
        auto_gain_settings['max_exposure_ms'] = config_helpers.get_ag_ae_max_exposure_ms(
            layer, ceilings
        )
        # The floor rides beside the ceiling so an auto-gain lock can say
        # whether exposure bottomed out of the usable range (AT_MINIMUM).
        auto_gain_settings['min_exposure_ms'] = config_helpers.get_ag_ae_min_exposure_ms(layer)
        return LayerCameraRequest(
            gain_db=gain_db,
            exposure_ms=exposure_ms,
            auto_gain=auto_gain,
            auto_gain_settings=auto_gain_settings,
        )

    @api
    def new_protocol(
        self,
        *,
        tiling: str = '1x1',
        use_zstacking: bool = False,
        period_s: float | None = None,
        duration_s: float | None = None,
    ) -> 'Protocol':
        """Build a protocol from this session's settings, as the GUI's New does.

        One step per acquiring layer at every well of the session's labware,
        at the current objective, tiled and z-stacked as asked. The
        protocols API refuses the build when no layer acquires, where an
        empty protocol would otherwise come back for a click that meant
        steps; a labware with no wells still gives an empty protocol, which
        ``add_step`` fills at the current position.

        Args:
            tiling: The tile grid at each well, a grid the installation's
                tiling catalogue offers, such as ``'3x3'``; ``'1x1'`` (the
                default) is one tile.
            use_zstacking: Stack each step through the stored z-stack
                (``get_zstack_params``); False takes one plane.
            period_s: The protocol's period, in seconds; None is the stored
                default's.
            duration_s: The protocol's duration, in seconds; 0 is one scan,
                and None is the stored default's.

        Raises:
            ProtocolRunRefusedError: ``'no_acquiring_layer'``, no layer
                acquires; ``'tiling_unknown'``, a grid this
                installation does not offer; ``'overlap_out_of_range'``, the
                stored tile overlap outside 0 to 50; ``'tiles_outside_travel'``,
                a tile past the X or Y travel; ``'zstack_not_configured'``, a
                z-stack with no extent; ``'positions_unreachable'``, a grid on
                a scope with no X/Y motor or a z-stack on one with no Z
                motor; ``'zslices_outside_travel'``, a slice past the Z travel.
            ObjectiveUnknownError: the objective in the light path is
                unknown, so the config cannot be assembled:
                ``'slot_unknown'``, ``'slot_unassigned'`` or
                ``'not_in_catalogue'`` on a turret scope; ``'none_selected'``
                with no turret.
            ProtocolScheduleRefusedError: ``period_s`` or ``duration_s`` is
                one no protocol can run.
            AxisStateUnknownError: an acquiring layer has no saved focus, so
                its steps would save the current Z, and Z does not know its
                position. Reported once.
            AxisMovingError: ``'axis_moving'``, Z is moving, so the current
                Z is a place it is passing through. Nothing is built.
        """
        from modules.protocol import Protocol, schedule_from_units

        config = self.get_sequenced_capture_config(tiling=tiling, use_zstacking=use_zstacking)
        # Converted here, judged where the protocol is built.
        for key, seconds in (('period', period_s), ('duration', duration_s)):
            if seconds is not None:
                config[key] = schedule_from_units(key, seconds, judged=False, unit='seconds')
        layer_configs = config['layer_configs']
        self.scope.protocols.refuse_no_acquiring_layer(layer_configs)
        if any(
            Protocol.layer_acquires(cfg) and cfg['focus'] is None for cfg in layer_configs.values()
        ):
            self.scope.motion.refuse_unknown_positions(
                ('Z',), recording=True, then='make a new protocol'
            )
        return self.scope.protocols.create_protocol(input_config=config)

    @api
    def create_empty_protocol(self) -> 'Protocol':
        """A protocol with no steps, on this session's labware and timing.

        Needs no objective: it has no step to stamp one into, so it can be
        created while the objective in the light path is unknown -- at
        startup, before the turret is in a known slot. Steps added later
        carry the objective they were taken with.

        Raises:
            ProtocolScheduleRefusedError: the stored period or duration is one
                no protocol can run.
        """
        import modules.config_helpers as config_helpers

        return self.scope.protocols.create_protocol(
            empty_config=config_helpers.get_empty_protocol_config_from_settings(
                self.get_settings_snapshot(), self.scope.wellplate_loader
            )
        )

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def add_step(
        self,
        protocol: 'Protocol',
        *,
        channels: 'Iterable[str] | None' = None,
        at: PlateLocation | None = None,
        before_step_index: int | None = None,
        after_step_index: int | None = None,
        use_zstacking: bool = False,
    ) -> list[str]:
        """Add a step to ``protocol`` from this session's settings and live position.

        The entry point a caller with no GUI uses to do what Add Step
        does: one step per layer whose ``acquire`` is set, at the live
        stage position on the protocol's plate, with the current objective,
        in the settings' channel order. The protocols API performs the add
        and refuses when nothing would be added; this composes its inputs
        from the session the same way the GUI's handler does.

        Returns the inserted step names, in protocol order: each stack's
        slices when stacked.

        Args:
            protocol: The protocol the steps are added to.
            channels: The channels to add a step for, in that order, instead
                of the acquiring layers: each takes its stored settings, and
                one whose ``acquire`` is None is added as an image step, so a
                caller adds without changing the settings. None (the
                default) adds one step per acquiring layer.
            at: Where on the protocol's plate the steps are, instead of the
                live position: X and Y in mm, Z in um. None (the default)
                takes the live position.
            before_step_index: The step the new steps are placed before.
            after_step_index: The step the new steps are placed after. With
                neither given, they follow the last step; both given is
                refused.
            use_zstacking: Make each added step a z-stack of the stored
                z-stack settings, as ``new_protocol`` does for its steps; the
                protocol's other steps are left as they are.

        Raises:
            AxisMovingError: ``'axis_moving'``, with no ``at``, an axis is
                moving, so the live position is a place it is passing
                through. Nothing is added.
            ProtocolRunRefusedError: ``'step_position_unknown'``, an axis does
                not know its position; ``'no_acquiring_layer'``, no layer
                acquires. With ``use_zstacking``, also
                ``'zstack_not_configured'``, ``'positions_unreachable'`` or
                ``'zslices_outside_travel'``, as ``apply_zstacking`` refuses
                the stored stack. Nothing is added.
            ObjectiveUnknownError: the objective in the light path is
                unknown, as ``runtime_state.resolve_current_objective`` says
                why (``'slot_unassigned'``, the turret's current slot has no
                objective). Nothing is added.
            ConfigurationDefectError: ``'layers_unresolved'``, ``channels`` is given
                and this scope's layers could not be resolved. Nothing is added.
            ArgumentRefusedError: ``'channel_unknown'``, a name in ``channels``
                is not a channel.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, this scope has
                no layer named in ``channels``; ``'not_connected'``, with no
                ``at``, the model's motor controller is not connected, so
                there is no live position. Nothing is added.
            StepNotFoundError: ``'no_such_step'``, an impossible
                ``before_step_index`` / ``after_step_index``.
            StepEditRefusedError: ``'step_value_refused'``, both given.
        """
        if channels is None:
            layer_configs = self.get_layer_configs()
            channel_order = self.settings['step_channel_order']
        else:
            channels = tuple(channels)
            for channel in channels:
                self.scope.channel_identity.refuse_unless_on_scope(
                    channel, 'add_step', then='take a step'
                )
            layer_configs = self.get_layer_configs(list(channels))
            # A layer named is acquired whatever its stored acquire, as a
            # still unless it is stored as a video.
            for layer_config in layer_configs.values():
                if layer_config['acquire'] is None:
                    layer_config['acquire'] = AcquireMode.IMAGE
            channel_order = list(channels)
        return self.scope.protocols.add_step(
            protocol,
            layer_configs=layer_configs,
            stim_configs=self.get_stim_configs(),
            plate_position=at if at is not None else self.plate_position_on(protocol.labware_id()),
            channel_order=channel_order,
            before_step_index=before_step_index,
            after_step_index=after_step_index,
            zstack_params=self.get_zstack_params() if use_zstacking else None,
        )

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def update_step(
        self,
        protocol: 'Protocol',
        step_index: int,
        *,
        channel: str,
        label: str | None = None,
    ) -> str:
        """Rewrite a step of ``protocol`` from this session's settings and live position.

        The entry point a caller with no GUI uses to do what Update Step
        does: step ``step_index`` takes ``channel``'s settings, the live stage
        position on the protocol's plate and the current objective. The
        protocols API performs the update, reads the objective, and refuses
        it when the position or the objective is unknown; this composes its
        other inputs from the session the same way ``add_step`` does.

        Returns the step's name after the update.

        Args:
            protocol: The protocol whose step is rewritten.
            step_index: The step rewritten, counted from 0.
            channel: The layer whose settings the step takes. When that
                layer's stimulation is enabled the edit is a stimulation
                edit, and the step keeps the channel it already acquires.
            label: The step's new label; None (the default) keeps its label.

        Raises:
            AxisMovingError: ``'axis_moving'``, an axis is moving, so the live
                position is a place it is passing through. The step is
                unchanged.
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a channel.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, this scope has
                no ``channel``. Nothing is changed on either.
            ObjectiveUnknownError: the objective in the light path is
                unknown, as ``add_step`` raises it (``'slot_unassigned'``, the
                turret's current slot has no objective). The step is
                unchanged.
            ProtocolRunRefusedError: ``'step_position_unknown'``, an axis
                does not know its position; ``'no_acquiring_layer'``,
                ``channel`` is not set to acquire; ``'channel_already_in_add'``,
                another step of the same add already takes that channel. The
                step is unchanged.
            StepNotFoundError: ``'no_such_step'``, ``step_index`` is not a
                step of ``protocol``, judged before anything else.
        """
        self.scope.channel_identity.refuse_unless_on_scope(
            channel, 'update_step', then='take a step'
        )
        return self.scope.protocols.update_step(
            protocol,
            step_index,
            layer=channel,
            layer_configs=self.get_layer_configs(),
            stim_configs=self.get_stim_configs(),
            plate_position=self.plate_position_on(protocol.labware_id()),
            label=label,
        )

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def save_focus(
        self, protocol: 'Protocol', channel: str, *, step_index: int | None = None
    ) -> SavedFocus:
        """Save the live Z as ``channel``'s focus, and as step ``step_index``'s Z.

        The layer's focus is what every new step of the layer is born at.
        The step takes the Z only when it is a step of ``channel``; a step of
        another channel is left alone. No other step is written: every step
        of a layer is born at the same focus, so a step matching the old
        focus says nothing about whether its user wants the new one.
        ``apply_focus_to_channel_steps`` writes them all.

        Args:
            protocol: The protocol whose step may take the Z.
            channel: The layer whose focus is saved.
            step_index: The step, counted from 0, that takes the Z when it
                is a step of ``channel``; None (the default) writes no step.

        Raises:
            ProtocolRunRefusedError: ``'positions_unreachable'`` -- this scope
                has no Z axis. Nothing is written.
            AxisStateUnknownError: Z lost its reference. Nothing is written.
            AxisMovingError: ``'axis_moving'``, Z is moving, so its number is a
                place it is passing through. Nothing is written.
            StepNotFoundError: ``'no_such_step'``, ``step_index`` is not a
                step of ``protocol``. Nothing is written.
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a channel.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, this scope has
                no ``channel``.
            ConfigurationDefectError: ``'layers_unresolved'``, this scope's layers
                could not be resolved. Nothing is written.
        """
        self.scope.channel_identity.refuse_unless_on_scope(
            channel, 'save_focus', then='take a focus'
        )
        step = None if step_index is None else protocol.step_row(step_index=step_index)
        z = self.scope.protocols.focus_z_um(then='save the focus')
        self._store_layer_focus(channel, z)
        if step is None or step['Color'] != channel:
            logger.info(f'[Session  ] Focus saved: {channel} Z={z}, no step written')
            return SavedFocus(z_um=z, step_index=None)
        self.scope.protocols.set_step_z(protocol, step_index, z)
        logger.info(
            f'[Session  ] Focus saved: {channel} Z={z}, and as the Z of step {step_index + 1}'
        )
        return SavedFocus(z_um=z, step_index=step_index)

    @api(sets='saved_focus_z_um', admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def save_channel_focus(self, channel: str, z_um: float) -> None:
        """Store ``z_um`` as ``channel``'s focus, the Z every new step of the layer is born at.

        The door for a caller that holds a number rather than a stage: the
        Autofocus button saving the Z its run chose, a script saving one it
        measured. It says only that the number is a Z this scope's focus can
        reach; it guarantees nothing about where the number came from.
        ``save_focus``, which saves the live Z, writes through the same path.

        Args:
            channel: The layer whose focus is stored.
            z_um: The focus, in micrometres; it must lie within Z's travel.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a channel.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, this scope has
                no ``channel``.
            ConfigurationDefectError: ``'layers_unresolved'``, this scope's layers
                could not be resolved. Nothing is written.
            ProtocolRunRefusedError: ``'positions_unreachable'`` -- this scope
                has no Z axis. Nothing is written.
            ArgumentRefusedError: ``'not_a_number'``, ``z_um`` is NaN or
                infinite. Nothing is written.
            PositionOutOfRangeError: ``z_um`` lies outside Z's travel.
                Nothing is written.
        """
        self.scope.channel_identity.refuse_unless_on_scope(
            channel, 'save_channel_focus', then='take a focus'
        )
        z = self._store_layer_focus(channel, z_um)
        logger.info(f'[Session  ] Focus saved: {channel} Z={z}')

    def _store_layer_focus(self, layer: str, z_um: float) -> float:
        """The one write of a layer's focus, refused unless Z can reach it; returns the Z stored.

        ``save_all_bookmarks`` alone writes a focus beside it: it stores the
        bookmark and every layer's focus under one hold of the lock, and its
        Z is the live one, which ``focus_z_um`` has already checked.
        """
        z = self.scope.protocols.check_focus_z(z_um, then='save the focus')
        with self.settings_lock:
            self._store_setting(f'{layer}.focus', z)
        return z

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def save_bookmark(self, axes: 'Iterable[StageAxis]') -> PlateLocation:
        """Save where the stage is on ``axes`` as the bookmark, as the bookmark buttons do.

        X and Y are stored in plate millimetres on the selected plate, Z in
        micrometres: the frame the go-to-bookmark moves read them in. The one
        writer of ``bookmark``.

        Args:
            axes: The axes saved, each ``'X'``, ``'Y'`` or ``'Z'``; an axis
                not named keeps its stored bookmark.

        Returns:
            The values stored, as ``bookmark`` reads them; an axis not in
            ``axes`` is None.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, a name in ``axes`` that
                is not X, Y or Z. Nothing is written.
            AxisStateUnknownError: ``'position_unknown'``, an axis it reads
                does not know its position, so the number it reports is not
                one to save: the axes in ``axes``, X and Y together. Nothing
                is written.
            AxisMovingError: ``'axis_moving'``, an axis it reads is moving, so
                its number is a place it is passing through: the axes in
                ``axes``, X and Y together. Nothing is written.
            HardwareCommandRefusedError: ``'not_connected'`` -- the motor
                controller is not connected; ``'axis_absent'`` -- this scope
                has no motor for an axis in ``axes``, so there is no place on
                it to save. Nothing is written.
        """
        import modules.config_helpers as config_helpers

        axes = tuple(axes)
        for axis in axes:
            self.scope.motion.refuse_absent('save_bookmark', axis)
        # Only the axes kept are read, so a Z bookmark is saved while X moves.
        position = config_helpers.kept_plate_position(
            self.scope, self.settings['protocol']['labware'], axes, then='save the bookmark'
        )
        saved = PlateLocation(
            x_mm=position.x_mm if 'X' in axes else None,
            y_mm=position.y_mm if 'Y' in axes else None,
            z_um=position.z_um if 'Z' in axes else None,
        )
        stored = {'X': saved.x_mm, 'Y': saved.y_mm, 'Z': saved.z_um}
        with self.settings_lock:
            for axis in axes:
                self._store_setting(f'bookmark.{axis.lower()}', stored[axis])
        logger.info(f'[Session  ] Bookmark saved: {saved}')
        return saved

    @api
    @property
    def bookmark(self) -> PlateLocation:
        """The bookmark: X and Y in plate millimetres on the selected plate, Z in micrometres.

        What ``save_bookmark`` and ``save_all_bookmarks`` stored, in the
        frame the go-to-bookmark moves read it in.

        Raises:
            Nothing.
        """
        with self.settings_lock:
            stored = self.settings['bookmark']
            return PlateLocation(x_mm=stored['x'], y_mm=stored['y'], z_um=stored['z'])

    @api(stop='Operation.stop', admitted_by=(Admission.LANE_CALL,))
    def start_go_to_bookmark(self, axes: 'Iterable[StageAxis]') -> Operation:
        """Start ``axes`` moving to the bookmark, as the go-to-bookmark buttons do; return the started moves.

        Reads ``bookmark`` for the axes given, as ``save_bookmark`` wrote
        them: X and Y in mm on the selected plate (``start_move_to_plate``,
        the axes given on one lane task), Z in um (``start_move_z``).

        Args:
            axes: The axes moved, each ``'X'``, ``'Y'`` or ``'Z'``; an axis
                not named is left where it is.

        Returns:
            One ``Operation`` for the moves started, done once every one is
            (``Operation.all_of``).

        Raises:
            ArgumentRefusedError: ``'not_offered'``, a name in ``axes`` that
                is not X, Y or Z. Nothing was driven.
            HardwareCommandRefusedError: ``'not_connected'``,
                ``'axis_absent'``, ``'lid_open_before_motion'`` or ``'monitor_ended'``, as
                ``MotionAPI.start_move_to_plate``. Nothing was driven.
            PositionOutOfRangeError: The bookmark is outside what this
                stage can reach. Nothing was driven.
            AxisStateUnknownError: An axis given does not know its
                position. Nothing was driven.
            MoveFaultError: ``'driver_failed'``, the board did not take a
                command.
        """
        mark = self.bookmark
        motion = self.scope.motion
        moves = []
        if 'X' in axes or 'Y' in axes:
            moves.append(
                motion.start_move_to_plate(
                    PlateLocation(
                        x_mm=mark.x_mm if 'X' in axes else None,
                        y_mm=mark.y_mm if 'Y' in axes else None,
                        z_um=None,
                    )
                )
            )
        if 'Z' in axes:
            moves.append(motion.start_move_z(mark.z_um))
        return Operation.all_of(tuple(moves))

    @api(admitted_by=(Admission.LANE_CALL,))
    def go_to_bookmark(self, axes: 'Iterable[StageAxis]') -> None:
        """Move ``axes`` to the bookmark, and return once they have arrived.

        ``start_go_to_bookmark`` followed by its operation's ``result()``.

        Args:
            axes: The axes moved, each ``'X'``, ``'Y'`` or ``'Z'``; an axis
                not named is left where it is.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, as
                ``start_go_to_bookmark``.
            HardwareCommandRefusedError: ``'not_connected'``,
                ``'axis_absent'``, ``'lid_open_before_motion'`` or ``'monitor_ended'``, as
                ``MotionAPI.start_move_to_plate``. Nothing was driven.
            PositionOutOfRangeError: The bookmark is outside what this
                stage can reach. Nothing was driven.
            AxisStateUnknownError: An axis given does not know its
                position. Nothing was driven.
            MoveFaultError: ``'driver_failed'``, the board did not take a
                command; or an axis did not arrive and is UNKNOWN, the motion
                monitor's ``'stalled'``, ``'board_lost'``,
                ``'position_unread_after_move'``, ``'status_unread'`` or
                ``'monitor_ended_mid_move'``, or ``'faulted'``; see ``MotionAPI.start_move_z``.
            MoveStoppedError: ``'stopped'``, a stop ended a move;
                ``'superseded'``, another move or a home took an axis first.
            BlockingCallRefusedError: ``'wait_on_ui_thread'``, called on
                the UI thread, whose form is ``start_go_to_bookmark``.
                Nothing was driven.
        """
        refuse_waiting_on_ui_thread('go_to_bookmark')
        self.start_go_to_bookmark(axes).result()

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def save_all_bookmarks(self) -> PlateLocation:
        """Save the live Z as the Z bookmark and as the focus of every layer this scope has.

        What Set All Bookmarks does. Returns the Z bookmark stored, as
        ``bookmark`` reads it; X and Y, which it does not save, are None.

        Raises:
            ProtocolRunRefusedError: ``'positions_unreachable'`` -- this scope
                has no Z axis. Nothing is written.
            AxisStateUnknownError: Z lost its reference. Nothing is written.
            AxisMovingError: ``'axis_moving'``, Z is moving, so its number is a
                place it is passing through. Nothing is written.
        """
        z = self.scope.protocols.focus_z_um(then='save the bookmarks')
        layers = sorted(self._layers_on_scope())
        with self.settings_lock:
            self._store_setting('bookmark.z', z)
            for layer in layers:
                self._store_setting(f'{layer}.focus', z)
        logger.info(f'[Session  ] Bookmarks saved: Z={z}, and as the focus of {layers}')
        return PlateLocation(x_mm=None, y_mm=None, z_um=z)

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def apply_focus_to_channel_steps(self, protocol: 'Protocol', channel: str) -> int:
        """Save the live Z as ``channel``'s focus and as the Z of its every step.

        Returns how many steps took it.

        Args:
            protocol: The protocol whose steps of ``channel`` take the Z.
            channel: The layer whose focus is saved and whose steps take it.

        Raises:
            ProtocolRunRefusedError: ``'positions_unreachable'`` -- this scope
                has no Z axis. Nothing is written.
            AxisStateUnknownError: Z lost its reference. Nothing is written.
            AxisMovingError: ``'axis_moving'``, Z is moving, so its number is a
                place it is passing through. Nothing is written.
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a channel.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, this scope has
                no ``channel``.
            ConfigurationDefectError: ``'layers_unresolved'``, this scope's layers
                could not be resolved. Nothing is written.
        """
        self.scope.channel_identity.refuse_unless_on_scope(
            channel, 'apply_focus_to_channel_steps', then='take a focus'
        )
        z = self.scope.protocols.focus_z_um(then='apply the focus')
        self._store_layer_focus(channel, z)
        updated = self.scope.protocols.apply_focus_to_channel_steps(protocol, channel, z)
        logger.info(f'[Session  ] Focus applied: {channel} Z={z} to {updated} step(s)')
        return updated

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def delete_step(self, protocol: 'Protocol', step_index: int) -> None:
        """Remove step ``step_index`` from ``protocol``, as the Delete button does.

        Args:
            protocol: The protocol the step is removed from.
            step_index: The step removed, counted from 0.

        Raises:
            StepNotFoundError: ``'no_such_step'``, ``step_index`` is not a
                step of ``protocol``. Nothing is removed.
        """
        self.scope.protocols.delete_step(protocol, step_index)

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def rename_step(self, protocol: 'Protocol', step_index: int, name: str) -> str:
        """Give step ``step_index`` of ``protocol`` the label ``name``; returns its new name.

        Args:
            protocol: The protocol whose step is renamed.
            step_index: The step renamed, counted from 0.
            name: The step's new label; it holds at least one letter, digit,
                dash or underscore.

        Raises:
            StepNotFoundError: ``'no_such_step'``, ``step_index`` is not a
                step of ``protocol``. The step keeps its name.
            StepEditRefusedError: ``'step_value_refused'``, ``name`` has no
                letter, digit, dash or underscore. The step keeps its name.
        """
        return self.scope.protocols.rename_step(protocol, step_index, name)

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def set_protocol_labware(self, protocol: 'Protocol', labware_id: str) -> str:
        """Put ``protocol`` on the plate ``labware_id``; returns the key it took.

        The protocol's plate only: the scope's is ``select_labware``'s. On a
        scope with no XY stage the protocol takes Center Plate.

        Args:
            protocol: The protocol put on the plate.
            labware_id: The plate's name in the labware catalogue.

        Raises:
            CatalogueNameRefusedError: ``'labware_unknown'``, ``labware_id`` is not a
                plate the catalogue has. The protocol keeps its plate.
        """
        return self.scope.protocols.set_labware(protocol, labware_id)

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def set_protocol_schedule(
        self,
        protocol: 'Protocol',
        *,
        period_s: float,
        duration_s: float,
    ) -> None:
        """Give ``protocol`` a new period and duration in seconds, as its two fields do.

        One scan is 0 for both.

        Args:
            protocol: The protocol rescheduled.
            period_s: The time from one scan's start to the next, in
                seconds.
            duration_s: How long the protocol runs, in seconds.

        Raises:
            ProtocolScheduleRefusedError: either cannot be run; the protocol
                keeps the schedule it had.
        """
        from modules.protocol import schedule_from_units

        protocol.modify_time_params(
            period=schedule_from_units('period', period_s, judged=False, unit='seconds'),
            duration=schedule_from_units('duration', duration_s, judged=False, unit='seconds'),
        )

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def set_protocol_capture_root(self, protocol: 'Protocol', root: str) -> None:
        """Give ``protocol`` the capture root that prefixes its runs' file names.

        Args:
            protocol: The protocol given the root.
            root: The capture root, kept as given; the prefix a run's names
                take from it is ``protocol.capture_prefix()``.

        Raises:
            Nothing.
        """
        protocol.modify_capture_root(capture_root=root)

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def set_step_autofocus(
        self, protocol: 'Protocol', *, enabled: bool, step_index: int | None = None
    ) -> None:
        """Turn autofocus on or off for step ``step_index`` of ``protocol``, or for every step.

        Args:
            protocol: The protocol whose steps are set.
            enabled: True turns autofocus on, False off.
            step_index: The step set, counted from 0; None (the default)
                sets every step.

        Raises:
            StepNotFoundError: ``step_index`` is not a step of ``protocol``.
                Nothing changes.
        """
        if step_index is None:
            protocol.modify_autofocus_all_steps(enabled=enabled)
        else:
            protocol.modify_autofocus(step_index=step_index, enabled=enabled)

    @api(reads=True)
    def turret_slot_questions(self, protocol: 'Protocol') -> list['TurretSlotQuestion']:
        """What to ask about ``protocol``'s turret slots: one question per objective, or none.

        A step that cannot say which slot it means is one whose objective is
        in two or more slots and that recorded none, or whose recorded slot
        now holds another objective. Each is refused by a run and by going
        to it until ``point_steps_at_turret_slot`` answers its question.
        LumaViewPro asks when a protocol is opened. Empty on a scope with no
        turret, and for steps whose objective no slot holds, which a load
        and a run refuse as ``turret_objectives_unassigned``.

        Args:
            protocol: The protocol whose steps are read.

        Raises:
            Nothing.
        """
        return self.scope.protocols.turret_slot_questions(protocol)

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def point_steps_at_turret_slot(
        self, protocol: 'Protocol', objective_id: str, slot: int
    ) -> tuple[int, ...]:
        """Answer ``objective_id``'s turret slot question: its steps now record ``slot``.

        Points exactly the steps ``turret_slot_questions`` names for
        ``objective_id``; every other step keeps its slot. Returns
        the indexes of the steps pointed, which are empty when no question names it.

        Args:
            protocol: The protocol whose steps are pointed.
            objective_id: The objective the question is about, as
                ``turret_slot_questions`` names it.
            slot: The turret slot its steps record; it holds
                ``objective_id`` now.

        Raises:
            ArgumentRefusedError: ``'slot_not_holding_objective'``, ``slot``
                does not hold ``objective_id`` now (``offered``: the slots
                that do). Nothing is changed.
        """
        return self.scope.protocols.point_steps_at_turret_slot(protocol, objective_id, slot)

    @api(reads=True)
    def tiling_config(self) -> 'TilingConfig':
        """The tiling grids this installation offers: the scope's copy, read at bring-up.

        What a caller needs to choose a grid: ``available`` lists the
        labels ``new_protocol`` and ``apply_tiling`` accept, and
        ``default`` is the one to preselect; ``Protocol.tiling()`` names
        the grid a protocol's steps already carry. Read once with the
        installation's other catalogues, so a tiling.json edited while the
        program runs is seen at the next start, and every protocol is built
        against the same grids. The record is frozen: a caller cannot change
        the grids a session offers.

        Raises:
            Nothing.
        """
        return self.scope.protocols.tiling_config()

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def apply_tiling(self, protocol: 'Protocol', tiling: str) -> None:
        """Expand every step of ``protocol`` into the tile grid ``tiling``, as Apply does.

        The tiles are spaced for the configured frame, binning and tile
        overlap, laid out on the protocol's own plate.

        Args:
            protocol: The protocol whose steps are tiled.
            tiling: The grid, one of ``tiling_config().available``, such as
                ``'3x3'``.

        Raises:
            ProtocolRunRefusedError: ``tiling`` is not a grid this
                installation offers, the protocol is already tiled, a step's
                objective is not in the catalogue, the scope has no X/Y
                motor, or a tile falls outside the stage's travel. Nothing
                changes.
            CatalogueNameRefusedError: ``'labware_unknown'``, the protocol's plate is
                not in the catalogue. Nothing changes.
        """
        import modules.config_helpers as config_helpers

        self.scope.protocols.apply_tiling(
            protocol,
            tiling,
            frame_dimensions=config_helpers.get_frame_dimensions_from_settings(self.settings),
            binning_size=config_helpers.get_binning_from_settings(self.settings),
            overlap_percent=self.settings['tiling_overlap_percent'],
        )

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def apply_zstacking(
        self,
        protocol: 'Protocol',
        *,
        range_um: float,
        step_size_um: float,
        z_reference: ZStackReference,
    ) -> None:
        """Expand every step of ``protocol`` not already in a stack into a z-stack.

        Args:
            protocol: The protocol whose steps are stacked.
            range_um: The stack's extent in Z, in micrometres; greater than
                zero.
            step_size_um: The distance between slices, in micrometres;
                greater than zero.
            z_reference: Where each step's Z sits in its stack.

        Raises:
            ProtocolRunRefusedError: ``range_um`` or ``step_size_um`` is not
                greater than zero, the scope has no Z motor, or a slice falls
                outside the Z travel. Nothing changes.
            ArgumentRefusedError: ``'not_a_number'``, ``range_um`` or
                ``step_size_um`` is not a finite number. Nothing changes.
            ArgumentRefusedError: ``'not_offered'``, ``z_reference`` is not a
                ``ZStackReference``. Nothing changes.
        """
        self.scope.protocols.apply_zstacking(
            protocol, range_um=range_um, step_size_um=step_size_um, z_reference=z_reference
        )

    @api(admitted_by=(Admission.LANE_CALL, Admission.GUARDED_WRITE))
    @_guarded_write
    def go_to_step(self, protocol: 'Protocol', step_index: int) -> None:
        """Go to step ``step_index`` of ``protocol``; return once the camera holds the step's layer and the stage has arrived.

        What ``start_go_to_step`` does, then the started operation's
        ``result()``: the step's layer goes on the camera while the stage
        travels; see ``start_go_to_step`` for what going to a step does and
        refuses.
        Both waits run in this caller's thread, so the IO lane takes other
        work meanwhile.

        It raises what ``start_go_to_step`` raises, and the following.

        Args:
            protocol: The protocol whose step is gone to.
            step_index: The step gone to, counted from 0.

        Raises:
            MoveFaultError: ``'driver_failed'``, the board did not take a
                command; or an axis did not arrive and is UNKNOWN, the motion
                monitor's ``'stalled'``, ``'board_lost'``,
                ``'position_unread_after_move'``, ``'status_unread'`` or
                ``'monitor_ended_mid_move'``, or ``'faulted'``; see ``MotionAPI.start_move_z``.
            MoveStoppedError: ``'stopped'``, a stop ended a move;
                ``'superseded'``, another move or a home took an axis first.
                The layer and the preview are the step's.
            CameraSettingRejected: the camera refused a setting of the
                step's layer, raised once the stage has arrived; see
                ``apply_layer_camera``.
            BlockingCallRefusedError: ``'wait_on_ui_thread'``, called on
                the UI thread, whose form is ``start_go_to_step``. Nothing
                was started.
        """
        refuse_waiting_on_ui_thread('go_to_step')
        # Read before the start, as the start reads it: the step this call
        # was made for, whatever the list holds once the lane has run.
        layer = protocol.step_row(step_index=step_index)['Color']
        with self._a_layer_change(layer, 'go_to_step'):
            going = self._start_moving_to_step(protocol, step_index)
            try:
                self.apply_layer_camera(layer)
            finally:
                # A refused apply still waits out the travel it started, so
                # the caller is never handed a raise with the stage still moving.
                going.result()

    @api(stop='Operation.stop', admitted_by=(Admission.LANE_CALL, Admission.GUARDED_WRITE))
    @_guarded_write
    def start_go_to_step(self, protocol: 'Protocol', step_index: int) -> Operation:
        """Start going to step ``step_index`` of ``protocol``, as a click on a step does.

        One task on the scope's IO lane: the axes are asked once whether
        they know their position; the turret turns to the step's objective
        and X, Y and Z are started towards the step together, on the
        protocol's own plate (``ProtocolsAPI.step_targets``, the targets a
        run computes); the step's values go into its layer's live settings,
        the layer acquiring as the step does and its focus at the step's Z,
        so the layer and the step agree; and the step's LED preview is
        applied -- its channel at its current when ``protocol_led_on`` is
        set, every channel dark when not. It returns once that task has run,
        before the stage arrives, with one ``Operation`` for the moves it
        started, X and Y as one request (``MotionAPI.start_move_xy``) and
        then Z: done once every one has ended, its ``result()`` returns None
        when all arrived and raises the first that did not, in X, Y, Z
        order, as ``MotionAPI.start_move_z`` says; its
        ``stop()`` stops the moves still under way. ``go_to_step`` waits on
        it. A person's click is a gesture and does not wait; a fault on the
        way is the motion monitor's to report. Only the axes this scope has
        are moved: a manual scope moves nothing, does the rest and returns
        an operation already done.

        A repeat of the step this session last went to (a re-click, a
        re-typed number) does everything but the preview: a channel the
        person lit or put out in between stays as they left it, and when
        another layer was shown meanwhile the step's layer is shown again,
        lit or dark as the shown channel was.

        Once that task has run, the step's layer goes on the camera
        (``apply_layer_camera``) before this returns, so a still asked after
        it is taken at the step's settings. It waits on the camera lane, so
        it is refused on the UI thread; a click calls it from a worker.

        Args:
            protocol: The protocol whose step is gone to.
            step_index: The step gone to, counted from 0.

        Raises:
            StepNotFoundError: ``step_index`` is not a step of ``protocol``.
                Nothing changes.
            ProtocolRunRefusedError: this scope cannot put the step's
                objective in the light path. Nothing changes.
            ArgumentRefusedError: ``'channel_unknown'``, the step's colour is
                not a layer. Nothing changes.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, this scope has
                no layer of the step's colour. Nothing changes.
            ArgumentRefusedError: ``'channel_unknown'``, the step's
                stimulation names a layer this release has not. Nothing
                changes.
            ProtocolRunRefusedError: ``'values_out_of_range'``, a value
                the step would store in its layers is outside its layer
                setting's range, each named by column and bound. Nothing
                changes.
            ConfigurationDefectError: ``'layers_unresolved'``, this scope's
                layers could not be resolved. Nothing changes.
            AxisStateUnknownError: an axis the step moves does not know
                its position. Nothing changes.
            HardwareCommandRefusedError: ``'not_connected'``, this scope's
                motor controller or camera is not connected, or its LED
                controller is not and the step's preview would light; or a
                run, a diagnostic or a home holds the
                scope, or a manual recording runs and the step's layer is not
                the shown one or its preview would light. Nothing changes.
            PositionOutOfRangeError: the step lies outside an axis's travel;
                the axes before it have moved, nothing else changes.
            MoveFaultError: ``'driver_failed'``, the board did not take an
                axis's command; or the turret's move to the step's slot did
                not arrive; see ``MotionAPI.start_move_z``.
            MoveStoppedError: ``'stopped'`` or ``'superseded'``, the turret's
                move to the step's slot was ended.
            CameraSettingRejected: the camera refused a setting of the
                step's layer; the moves are under way, and the layer and the
                preview are the step's.
            BlockingCallRefusedError: ``'wait_on_ui_thread'``, called on
                the UI thread. Nothing was started.
        """
        refuse_waiting_on_ui_thread('start_go_to_step')
        layer = protocol.step_row(step_index=step_index)['Color']
        with self._a_layer_change(layer, 'start_go_to_step'):
            going = self._start_moving_to_step(protocol, step_index)
            self.apply_layer_camera(layer)
        return going

    def _start_moving_to_step(self, protocol: 'Protocol', step_index: int) -> Operation:
        """The IO-lane task of going to a step, and the operation of its moves."""
        from modules.protocol import refuse_values_out_of_range, step_layer_values_refused

        step = protocol.step_row(step_index=step_index)
        # Converted here, from the step read above, so the lane moves to the
        # step this call was made for whatever the list holds by then; the
        # conversion asks first whether the scope can put the step's
        # objective in the light path.
        targets = self.scope.protocols.step_targets(protocol, step_index)
        self.scope.channel_identity.refuse_unless_on_scope(
            step['Color'], 'go_to_step', then='be gone to'
        )
        stim_configs = step.get('Stim_Config')
        if isinstance(stim_configs, dict):
            unknown = sorted(set(stim_configs) - set(common_utils.get_layers()))
            if unknown:
                raise ArgumentRefusedError(
                    'channel_unknown',
                    argument="the step's stimulation layer",
                    value=unknown[0],
                    offered=tuple(common_utils.get_layers()),
                )
        # Before the lane starts a move: the lane stores the step's values
        # into its layers once the moves are under way.
        refuse_values_out_of_range(step_layer_values_refused(step), f'Step {step_index + 1}')
        # Every member inside bounds its own wait, so the task has no bound
        # of its own to add.
        moves = self.io_executor.call(
            IOTask(action=self._go_to_step_on_lane, args=(protocol, step_index, step, targets)),
            'go_to_step',
            timeout_s=None,
        )
        return Operation.all_of(moves)

    def _go_to_step_on_lane(
        self, protocol: 'Protocol', step_index: int, step, targets: 'StepTargets'
    ) -> tuple[Operation, ...]:
        """The lane half of ``go_to_step``: ask once, start the moves, load the layer, preview.

        Returns the started moves: one XY request for the X and Y this scope
        has (done at once with neither), and Z's move when it has Z.
        """
        motion = self.scope.motion
        # A motorized scope whose controller is out of reach cannot go to the
        # step: one that never came up has no axes, so the moves below would
        # be none and the step a silent no-op.
        motion.refuse_controller_not_connected('go_to_step')
        # The step's layer goes onto the camera after the moves; with no
        # camera the step would be moved to and stored with nothing to see.
        self.scope.camera.refuse_camera_not_connected('go_to_step')
        last = self._last_step_gone_to
        shows = last is None or last[0] is not protocol or last[1] != step_index
        # A first visit is the step's preview. A repeat leaves the LEDs as the
        # person left them while the step's layer is shown, and moves the shown
        # channel, lit or dark as its LED reads, back to it when another was shown.
        was_lit = self._layer_lit(self._shown_layer)
        moves_light = shows or self._shown_layer != step['Color']
        lit = self.settings['protocol_led_on'] if shows else was_lit
        if moves_light and lit and self._led_address(step['Color']) is not None:
            # A preview that would light needs the LED controller, asked
            # before anything moves; a dark one needs no board. A recording
            # refuses the light, so it is asked here too, before the stage
            # is sent anywhere.
            self.scope.illumination.refuse_controller_not_connected('go_to_step')
            refusing = self.activity_claim.refusing_holder(
                current_taking(), falsifies_recording=True
            )
            if refusing is not None:
                raise command_refused(refusing, 'go_to_step')
        # The turret included: a failed turret home leaves T unknown
        # while the stage axes still know theirs. One command, so a capture
        # asked while the turret turns waits for the step's moves rather
        # than taking the new objective at the previous step's position.
        with motion.one_command(self.scope.capabilities.axes, then='go to the step'):
            if targets.turret_slot is not None:
                # The step's own Z move follows, so the turret need not put Z back.
                motion.move_turret(targets.turret_slot, restore_z=False)
            moves = (motion.start_move_xy(targets.x_um, targets.y_um),)
            if targets.z_um is not None:
                moves += (motion.start_move_z(targets.z_um),)
        self._load_step_into_layer(step)
        self._last_step_gone_to = (protocol, step_index)
        if not moves_light:
            return moves
        # After the step's moves are started, in the same task: a toggle the
        # person makes while the stage travels lands after the step's preview.
        # The step's values are now its layer's, so the layer lights at the
        # step's current.
        self._show_leds(step['Color'], lit)
        self._show_layer(step['Color'], was_lit=was_lit)
        return moves

    def _load_step_into_layer(self, step) -> None:
        """The step's values into its layer's live settings, under ``settings_lock``.

        A run never writes here: it reads its steps from the protocol, and
        the person's live-view configuration is theirs to keep across it.
        The step's config dicts are copied; the protocol owns them. The
        layer's saved focus is left as it is: it is where New starts the
        next protocol, so visiting a step must not move it. A step's
        stimulation spans every layer it names, and each takes its own.
        """
        color = step['Color']
        layer_values = {
            'autofocus': step['Auto_Focus'],
            'false_color': step['False_Color'],
            'illumination_ma': step['Illumination'],
            'gain_db': step['Gain'],
            'auto_gain': step['Auto_Gain'],
            'exposure_ms': step['Exposure'],
            'sum': step['Sum'],
        }
        video_config = step.get('Video Config')
        if isinstance(video_config, dict):
            layer_values['video_config'] = copy.deepcopy(video_config)
        stim_configs = step.get('Stim_Config')
        with self.settings_lock:
            for key, value in layer_values.items():
                self._store_setting(f'{color}.{key}', value)
            self._write_layer_acquire(color, step['Acquire'])
            if isinstance(stim_configs, dict):
                for stim_layer, stim_config in stim_configs.items():
                    self._store_setting(f'{stim_layer}.stim_config', copy.deepcopy(stim_config))

    def _led_address(self, layer: str) -> int | None:
        """The LED address ``layer`` lights on this unit, else None.

        None for a layer this unit's identity gives no LED; when the layer
        is one that drives an LED elsewhere, that is logged, since its
        preview would otherwise just not light with nothing naming why.
        """
        address = self.scope.illumination.color2ch(layer)
        if (
            address is None
            and self.scope.led_connected
            and layer in common_utils.get_layers_with_led()
        ):
            logger.warning(
                f"[Session  ] This scope has no '{layer}' LED channel; preview will not light."
            )
        return address

    def _show_leds(self, layer: str, lit: bool) -> None:
        """``layer``'s LED lit at its stored current when ``lit``, every other LED dark.

        The one light-one-darken-rest outside a run, through the lease-free
        transition: it diffs against the LED store, so a channel already
        right does not blink, and refuses before anything is turned off.
        """
        with self.settings_lock:
            illumination_ma = self.settings[layer]['illumination_ma']
        self.scope.illumination.apply_transition(
            LedTransition.MANUAL_STEP,
            LedTransitionCtx(
                channel=self._led_address(layer),
                illumination_ma=illumination_ma,
                preview_on=lit,
            ),
        )

    def _layer_lit(self, layer: str) -> bool:
        """Whether ``layer``'s LED reads lit in the LED store."""
        address = self.scope.illumination.color2ch(layer)
        if address is None or not self.scope.led_connected:
            return False
        readings = self.scope.illumination.get_led_states()
        return any(r.address == address and r.state == LedState.LIT for r in readings)

    def _show_layer(self, layer: str, *, was_lit: bool | None = None) -> None:
        """Make ``layer`` the shown layer, and tell the listeners when it moved
        or when the act that called this changed its light.

        ``was_lit`` is the shown layer's light as the act read it before it
        wrote; None for a caller that cannot read one (a run's step and end
        hear of the LED only once it has changed), whose light the LED
        event tells.
        """
        lit = self._layer_lit(layer)
        if layer == self._shown_layer and (was_lit is None or was_lit == lit):
            return
        self._shown_layer = layer
        from modules.notification_center import notifications

        notifications.tell_listeners(
            list(self._shown_channel_listeners), layer, lit, category='Illumination'
        )

    def _step_lit_by_run(self, layer: str) -> None:
        """A run's step has shown ``layer``: it is the shown layer."""
        if self._shown_before_run is None:
            self._shown_before_run = self._shown_layer
        self._show_layer(layer)

    def _show_the_layer_shown_before_the_run(self) -> None:
        """The run's cleanup has restored the camera: show the layer shown
        before the run again, at its stored settings.

        A write made to it during the run was stored only, and goes on the
        camera here; with none, the camera already holds it and nothing is
        written. Its LED is what the run's end policy left. Called on the
        protocol thread, under the run's taking, so the camera half is
        admitted.
        """
        before, self._shown_before_run = self._shown_before_run, None
        if before is None:
            return
        self._show_layer(before)
        if self.scope.camera_connected:
            self.apply_layer_camera(before)

    @api(reads=True)
    @property
    def shown_channel(self) -> ShownChannel:
        """The channel the scope shows: its layer, and whether its LED is lit.

        Whether it is lit is the LED store's answer at this read, whoever lit
        or darkened it.

        Raises:
            Nothing.
        """
        layer = self._shown_layer
        return ShownChannel(layer, self._layer_lit(layer))

    @api(admitted_by=(Admission.LANE_CALL,))
    def show_channel(self, channel: str) -> ShownChannel:
        """Show ``channel``: every other LED dark, ``channel`` lit at its stored
        current if the shown channel was lit, then its camera settings.

        The one way the shown channel changes outside a run; a run's steps
        and ``go_to_step`` move it too. Each refusal is decided before any
        LED is turned off, so a refused call leaves the LEDs as they were.
        A layer this unit gives no LED is shown with every LED dark.

        Args:
            channel: The layer shown.

        Returns:
            The shown channel now.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a
                channel; ``'illumination_out_of_range'``, its stored current
                is outside the attached board's range.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, this scope has
                no ``channel``; ``'not_connected'``, naming the LED
                controller the light needs or the camera; or
                ``'exclusive_activity_running'``, a run, a diagnostic or a
                home holds the scope, or a manual recording runs and
                ``channel`` is not the shown one or the shown channel is lit
                (a recording keeps one layer and the light it started under;
                nothing is changed).
            ConfigurationDefectError: ``'layers_unresolved'``, this scope's
                layers could not be resolved.
            CameraSettingRejected: the camera refused a setting of the
                layer; its LEDs are already shown.
        """
        self.scope.channel_identity.refuse_unless_on_scope(channel, 'show_channel', then='be shown')
        with self._a_layer_change(channel, 'show_channel'):
            # The light is decided on the lane, in the task that writes it: read
            # off it, another show's switch could be caught between its old
            # LED going dark and its new one lighting.
            self.io_executor.call(
                IOTask(action=self._show_channel_light_on_lane, args=(channel,)),
                'show_channel',
                timeout_s=None,
            )
            self.apply_layer_camera(channel)
        return self.shown_channel

    def _show_channel_light_on_lane(self, channel: str) -> None:
        """The lane half of ``show_channel``: ``channel`` lit if the shown layer is, every other LED dark."""
        was_lit = self._layer_lit(self._shown_layer)
        self._show_leds(channel, was_lit)
        self._show_layer(channel, was_lit=was_lit)

    @contextlib.contextmanager
    def _a_layer_change(self, layer: str, member: str):
        """Change the shown layer to ``layer`` with no recording running.

        A manual recording's frames are named for one layer, so a change of
        it would falsify them: it is refused while one runs, before anything
        is changed, and a recording cannot start while it is under way (the
        claim's falsifying-change count). Showing the layer already shown is
        no change. A control session held by someone else refuses it here
        too, before anything is changed. Any other holder of the scope
        refuses the change itself, on the lane it reaches.

        Raises:
            HardwareCommandRefusedError: ``'exclusive_activity_running'``,
                naming the recording.
            ControlSessionRefusedError: ``'control_session_held'``, naming
                the session.
        """
        if layer == self._shown_layer:
            yield
            return
        claim = self.activity_claim
        refusing = claim.enter_falsifying_change(current_taking())
        if isinstance(refusing, ControlSession) or (
            refusing is not None and refusing.kind == WorkKind.RECORDING
        ):
            raise command_refused(refusing, member)
        try:
            yield
        finally:
            if refusing is None:
                claim.leave_falsifying_change()

    @api(admitted_by=(Admission.LANE_CALL,))
    def light_shown_channel(self, lit: bool) -> ShownChannel:
        """Light the shown channel's LED at its stored current, or darken it,
        then put its stored camera settings on the camera.

        Every other LED is dark either way. A layer this unit gives no LED
        has nothing to light, and stays dark. The camera is left on the
        layer's stored settings (``apply_layer_camera``; nothing is written
        when it holds them already), unless no camera is connected.

        Args:
            lit: True lights the shown channel's LED at its stored current;
                False darkens it.

        Returns:
            The shown channel now.

        Raises:
            ArgumentRefusedError: ``'illumination_out_of_range'``, the
                layer's stored current is outside the attached board's
                range.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, this scope has
                no shown layer; ``'not_connected'``, naming the LED
                controller the light needs; or
                ``'exclusive_activity_running'``, a run, a diagnostic or a
                home holds the scope, or ``lit`` is True while a manual
                recording runs (a recording keeps the light it started
                under; nothing is changed).
            ConfigurationDefectError: ``'layers_unresolved'``, this scope's
                layers could not be resolved.
            CameraSettingRejected: the camera refused a setting of the
                layer; its LED is already lit or dark.
        """
        layer = self._shown_layer
        self.scope.channel_identity.refuse_unless_on_scope(
            layer, 'light_shown_channel', then='be lit'
        )
        was_lit = self._layer_lit(layer)
        self._show_leds(layer, lit)
        self._show_layer(layer, was_lit=was_lit)
        if self.scope.camera_connected:
            self.apply_layer_camera(layer)
        return self.shown_channel

    @api(in_process=True, event=SettingsChanged)
    def add_settings_listener(self, listener: Callable[[tuple[str, ...]], None]) -> None:
        """Tell ``listener`` ``(paths)`` each time a write stores settings, once it has let go of the store.

        Whoever wrote them -- the GUI, a client, a script, the camera
        storing what it applied.

        Args:
            listener: Called as ``listener(paths)``, ``paths`` the dotted
                paths one hold stored, together: a member that writes
                several settings tells them once. Called on the writer's
                thread, outside the settings lock, so a GUI listener
                schedules its drawing.

        Raises:
            Nothing.
        """
        self._settings_listeners.append(listener)

    @api(in_process=True)
    def remove_settings_listener(self, listener: Callable[[tuple[str, ...]], None]) -> None:
        """Stop telling ``listener``; a listener never added is a no-op.

        Args:
            listener: The callable ``add_settings_listener`` registered.

        Raises:
            Nothing.
        """
        with contextlib.suppress(ValueError):
            self._settings_listeners.remove(listener)

    def _tell_settings_changed(self, paths: tuple[str, ...]) -> None:
        from modules.notification_center import notifications

        notifications.tell_listeners(list(self._settings_listeners), paths, category='Settings')

    @api(in_process=True, event=ShownChannel)
    def add_shown_channel_listener(self, listener: Callable[[str, bool], None]) -> None:
        """Tell ``listener`` ``(channel, lit)`` each time the shown channel moves, or the session lights or darkens it.

        The session's own acts are told: ``show_channel``,
        ``light_shown_channel`` and ``go_to_step``, and a run's step when it
        moves the shown channel. Every LED change, whoever made it -- a raw
        LED write, a run's light, its end policy -- is an ``led`` event
        (``add_led_listener``).

        Args:
            listener: Called as ``listener(channel, lit)``, the layer now
                shown and whether its LED is lit, on the thread that changed
                it, so a GUI listener schedules its drawing.

        Raises:
            Nothing.
        """
        self._shown_channel_listeners.append(listener)

    @api(in_process=True)
    def remove_shown_channel_listener(self, listener: Callable[[str, bool], None]) -> None:
        """Stop telling ``listener``; a listener never added is a no-op.

        Args:
            listener: The callable ``add_shown_channel_listener`` registered.

        Raises:
            Nothing.
        """
        with contextlib.suppress(ValueError):
            self._shown_channel_listeners.remove(listener)

    @api(reads=True)
    def protocol_size_advisory(self, protocol: 'Protocol') -> 'ProtocolSizeAdvisory | None':
        """Ask a protocol whether it is large enough to warn the user about.

        The composer a client uses, rather than ``Protocol.size_advisory``
        itself: the two settings the estimate needs (whether video is saved
        as frames, and the global FPS cap) are the session's, resolved the
        way the run path resolves them, so the advisory and the run it is
        advising about cannot be sized differently.

        Args:
            protocol: The protocol sized, as a full run of it would be.

        Raises:
            Nothing.
        """
        import modules.config_helpers as config_helpers
        from modules.protocol_state_machine import SequencedCaptureRunMode

        run_settings = config_helpers.get_sequenced_run_settings(
            self.settings, run_mode=SequencedCaptureRunMode.FULL_PROTOCOL
        )
        return protocol.size_advisory(
            video_as_frames=run_settings['video_as_frames'],
            global_max_fps=run_settings['video_max_fps'],
        )

    @api(in_process=True, reads=True)
    def get_settings_snapshot(self) -> dict:
        """A deep copy of the settings dict, taken under the lock.

        A worker thread takes one of these at task entry and reads from it
        for the rest of the task, rather than reading a dict another
        thread may be part-way through rewriting. In-process: a wire client
        reads a setting by its path, through ``get_setting``.

        ``objective_id`` is the active objective, as ``get_setting`` answers
        it: on a turret scope, the slot's in the light path, or None while
        that is unknown.

        Raises:
            Nothing.
        """
        objective_id = self.scope.runtime_state.get_current_objective_id()
        snapshot = self._copy_of_the_store()
        snapshot['objective_id'] = objective_id
        return snapshot

    def _copy_of_the_store(self) -> dict:
        # The store as it is, for the file: on a turret scope its objective_id
        # is the pick a turretless scope proposes, never the slot's answer.
        with self.settings_lock:
            return copy.deepcopy(self.settings)

    @api(reads=True)
    def get_setting(self, path: str) -> 'SettingValue':
        """A copy of one setting, named by its dotted path: ``'stage_offset'``, ``'video.max_fps'``.

        Taken under the lock, so it is never a value part-way through a
        write; a copy, so changing it changes no setting (``update_settings``
        and the Session's members do).

        A setting with a read of its own is read there, so one setting has
        one read: ``get_setting('binning')`` is refused for
        ``scope.camera.binning``, ``get_setting('objective_id')`` for
        ``objective_id``. The live folder and the protocol file, which only
        the host moves, are read here.

        Args:
            path: The setting's dotted path, each segment a key one level
                down the settings.

        Raises:
            ArgumentRefusedError: ``'not_a_setting'``, the settings have no
                value at ``path``; ``'has_member'``, the setting at or above
                ``path`` has its own read, named in ``member``.
        """
        member = settings_paths.member_for(path)
        read = sets_of(getattr(_owner_of(member), member.name)) if member else None
        if read is not None:
            raise ArgumentRefusedError(
                'has_member', argument='path', value=path, member=member.path_of(read)
            )
        return self._read_stored_setting(path)

    def _read_stored_setting(self, path: str) -> 'SettingValue':
        # The store's own value: the scope's reader, from which the active
        # objective is derived, so it cannot be the derived answer.
        with self.settings_lock:
            value = self.settings
            for segment in path.split('.'):
                if not isinstance(value, dict) or segment not in value:
                    raise ArgumentRefusedError('not_a_setting', argument='path', value=path)
                value = value[segment]
            return copy.deepcopy(value)

    @api(admitted_by=(Admission.LANE_CALL, Admission.GUARDED_WRITE))
    def update_settings(self, path: str, value: 'SettingValue') -> None:
        """Write one setting, named by its dotted path: ``'video.max_fps'``, ``'BF.sum'``.

        The one write to the live settings for any caller, on any thread.
        Reads take ``get_settings_snapshot``; a write that skips this can
        tear a snapshot being taken concurrently, and takes none of the
        checks below.

        A setting that has its own member -- the objective, the plate, the
        image mode, a layer's acquire mode or focus, the live folder, ... --
        is changed only through that member, which checks it against the
        scope or changes another setting with it; the refusal names the
        member. A setting that decides how the next start reaches the scope
        or the machine -- the REST server and its key, the start mode, the
        profilers and debugging switches -- is read only from the
        installation's settings file, so no caller can reconfigure the scope
        it drives. A block (``'video'``) is not written whole: each of its
        settings has a path.

        Args:
            path: The setting's dotted path, each segment a key one level
                down the settings.
            value: The value stored, of the setting's kind and within its
                range.

        Raises:
            SettingRefusedError: ``'has_member'``, ``path`` is owned by a
                member (named by its path from the session:
                ``scope.camera.set_frame_size`` for the camera's); ``'host_only'``, its member is
                in-process, so only the application hosting the scope changes
                it (``live_folder``, ``protocol.filepath``);
                ``'installation_only'``, it is set
                only by the installation; ``'derived'``, it is derived from
                the release (``step_channel_order``); ``'not_a_setting'``, it is not a
                setting; ``'block'``, it names a block; ``'wrong_kind'``,
                ``value`` is not the setting's kind; ``'out_of_range'``, it
                is outside its range (a ``protocol.period`` or
                ``protocol.duration`` no protocol can run, and a layer's
                illumination above the attached LED board's maximum, among
                them). Nothing is written.
            ConfigurationDefectError: ``'layers_unresolved'``, the write is to the
                shown layer's camera settings and this scope's layers could not be
                resolved; the value is stored, and the camera is not changed.

        A numpy scalar in ``value``, at any depth, is stored as the Python
        value it holds.

        A layer's gain or exposure is stored as written, the person's
        intent; a camera that cannot reach it is driven at its own limit
        when the layer is applied (``CameraAPI.applied_gain_db_for``).
        A write to the shown layer's gain or exposure (or, while it is on
        auto-gain, to the auto-gain bounds) is on the camera, and to its
        illumination on its lit LED, when this returns; a write to a hidden
        layer, or one made while a run, a diagnostic or a home holds the
        scope, is stored only, and so is an illumination written while a
        manual recording runs. Such a write waits on the camera or LED lane,
        so a caller on the UI thread hands it to a worker. Each auto-gain
        bound is stored alone; a minimum left above the maximum is refused
        when auto-gain is armed with them (``ArgumentRefusedError``,
        ``'gain_range_inverted'``), so this write raises it after the store
        while the shown layer is on auto-gain, and the camera keeps its arm.
        The log level and the autofocus score log (``'logging.*'``) take
        effect as they are written.
        """
        value = python_values(value)
        member = settings_paths.member_for(path)
        if member is not None and mark_of(getattr(_owner_of(member), member.name)) == IN_PROCESS:
            # The live folder confines every path a wire caller sends, so only
            # the host moves it; advice naming the member alone sends a REST
            # client to a route that does not exist.
            raise SettingRefusedError(
                'host_only',
                path,
                f'it is changed by {member.words}, which only the application '
                'hosting the scope can call; a REST client reads it and cannot move it',
                member=member.path,
            )
        settings_paths.check_write(
            self.scope.settings_template,
            path,
            value,
            led_max_ma=self.scope.capabilities.led_max_ma,
        )
        preference = path in settings_paths.PREFERENCES
        with contextlib.nullcontext() if preference else a_guarded_write(self.activity_claim):
            with self.settings_lock:
                self._store_setting(path, value)
                if path.startswith('logging.'):
                    _apply_logging(self.settings)
            self._put_written_setting_on_the_hardware(path)

    def _an_activity_holds_the_scope(self) -> bool:
        """Whether a run, a diagnostic or a home holds the scope: a layer
        write then is stored only (Eric's D3)."""
        holder = self.activity_claim.holder
        return holder is not None and holder.kind in SCOPE_HOLDING_KINDS

    def _put_written_setting_on_the_hardware(self, path: str) -> None:
        """Put a write to the shown layer's hardware settings on the hardware.

        The shown layer's gain and exposure, and the auto-gain bounds while
        it is on auto-gain, go on the camera; its illumination goes on its
        LED while it is lit. Any other write, a hidden layer's included, is
        stored only. While a run, a diagnostic or a home holds the scope the
        write is stored only, and the run's end puts the shown layer back on
        the camera from the store. While a manual recording runs its
        illumination is stored only: the recording refuses every light.
        """
        layer = self._shown_layer
        with self.settings_lock:
            auto_gain = self.settings[layer]['auto_gain']
        camera = path in (f'{layer}.gain_db', f'{layer}.exposure_ms') or (
            auto_gain and path.startswith(('protocol.autogain.', 'ag_ae_max_exposure_ms'))
        )
        holder = self.activity_claim.holder
        led = (
            path == f'{layer}.illumination_ma'
            and self._layer_lit(layer)
            and (holder is None or holder.kind != WorkKind.RECORDING)
        )
        if not (camera or led):
            return
        if self._an_activity_holds_the_scope():
            return
        if led:
            self._show_leds(layer, True)
        if camera and self.scope.camera_connected:
            self.apply_layer_camera(layer)

    @api(in_process=True)
    @_guarded_write
    def set_live_folder(self, folder: str) -> None:
        """Make ``folder`` the live folder, where captures and runs are saved.

        The one writer of ``live_folder``, stored as it is at load.

        Args:
            folder: The folder's path; one given relative to the installation
                is made absolute, and the folder is created.

        Raises:
            SettingRefusedError: ``'out_of_range'``, ``folder`` is a string
                no file system can name (a NUL byte), or a folder that cannot
                be created (a file in the way, no permission, a drive that is
                not there). Nothing is written.
        """
        try:
            stored, unreachable = settings_init.bring_up_live_folder(folder, self.scope.source_path)
        except ValueError as e:
            # pathlib's answer to a string no file system can name (a NUL byte).
            raise SettingRefusedError(
                'out_of_range', 'live_folder', f'{folder!r} is not a path'
            ) from e
        if unreachable is not None:
            # Told now, not at the first capture into it.
            raise SettingRefusedError(
                'out_of_range', 'live_folder', f'the folder cannot be created ({unreachable})'
            ) from unreachable
        with self.settings_lock:
            self._store_setting('live_folder', stored)

    @api(in_process=True, reads=True)
    def live_folder_path(self, name: str) -> pathlib.Path:
        """The absolute path that ``name``, a name under the live folder, names.

        The one door from a wire caller's path argument to the file system:
        a REST bridge passes every path a caller gives through this, so the
        caller reaches the live folder and nothing beside it. A Python or GUI
        caller passes any path straight to the member it calls. The live
        folder is not created here: one that is missing is an unplugged drive
        or a stale setting.

        Args:
            name: A path relative to the live folder, such as
                ``ProtocolData/run1``.

        Raises:
            LiveFolderPathRefusedError: ``'outside_live_folder'``, ``name`` is
                empty, absolute, carries a drive (``C:x``, ``C:\\x``) or a
                network share, is no file system's name (a NUL byte), or
                resolves outside the live folder through
                ``..`` or a link; ``'capture_location_unusable'``, the live
                folder is missing or is not a folder.
        """
        try:
            root = path_utils.require_capture_location(self.get_setting('live_folder'))
        except path_utils.CaptureLocationError as e:
            raise LiveFolderPathRefusedError('capture_location_unusable', name, str(e)) from e
        root = root.resolve()
        # Read as Windows reads it on every host, so a drive, a share or a
        # rooted name is refused wherever the server runs; its root also
        # catches a POSIX absolute name.
        as_windows = pathlib.PureWindowsPath(name)
        if (
            not name
            or '\x00' in name
            or as_windows.drive
            or as_windows.root
            or not path_utils.resolves_inside(root, root / name)
        ):
            raise LiveFolderPathRefusedError(
                'outside_live_folder',
                name,
                f'{name!r} does not name a place inside the live folder {root}. Give a '
                'name relative to the live folder, such as ProtocolData/run1.',
            )
        return (root / name).resolve()

    @api(in_process=True)
    @_guarded_write
    def put_live_folder_file(self, name: str, data: bytes) -> pathlib.Path:
        """Write ``data`` as a new file under the live folder: a wire client's upload.

        A file already there is refused, so an upload never replaces a file
        a capture wrote. On the wire it is ``PUT /api/v1/files/<name>``, the
        body the file's bytes; a Python caller writes its own files.

        Args:
            name: The new file's name under the live folder, as
                ``live_folder_path`` takes it; the folders on its way are
                made.
            data: The file's bytes.

        Returns:
            The file written.

        Raises:
            LiveFolderPathRefusedError: ``'outside_live_folder'`` and
                ``'capture_location_unusable'``, as ``live_folder_path``
                raises them; ``'file_exists'``, a file is already at the
                name; ``'not_a_folder'``, a file is where one of its folders
                would be.
            LiveFolderFileNotWrittenError: The disk refused the write;
                nothing is left at the name.
        """
        path = self.live_folder_path(name)
        try:
            path_utils.write_new_file(path, data)
        except OSError as e:
            raise LiveFolderFileNotWrittenError(name, e) from e
        return path

    @api(reads=True)
    def live_folder_listing(self, name: str = '.') -> tuple[LiveFolderEntry, ...]:
        """What the folder ``name`` under the live folder holds, one level, sorted by name.

        Each entry is named by its full name under the live folder,
        ``/``-separated on every host, so it is taken as it is by any path
        parameter, a download and a further listing. An entry that is gone
        by the time it is read -- a file a run renames into place, a link to
        nothing -- is not listed. A
        link that leads outside the live folder is listed as what it leads
        to, and refused where its name is used, as ``live_folder_path``
        refuses it.

        Args:
            name: The folder listed, a name under the live folder as
                ``live_folder_path`` takes it; ``'.'`` (the default) lists the
                live folder itself.

        Raises:
            LiveFolderPathRefusedError: as ``live_folder_path`` raises it;
                ``'path_not_found'``, ``name`` names nothing; ``'not_a_folder'``,
                ``name`` is a file.
        """
        root = self.live_folder_path('.')
        folder = path_utils.existing_folder(self.live_folder_path(name), given=name)
        entries = []
        # By the name's text: a Windows path compares without case, so
        # sorting the paths would order the same folder differently by host.
        for child in sorted(folder.iterdir(), key=lambda path: path.name):
            try:
                found = child.stat()
            except FileNotFoundError:
                continue
            entries.append(
                LiveFolderEntry(
                    name=child.relative_to(root).as_posix(),
                    kind=EntryKind.FOLDER if stat.S_ISDIR(found.st_mode) else EntryKind.FILE,
                    size_bytes=found.st_size,
                    modified=datetime.datetime.fromtimestamp(found.st_mtime).astimezone(),
                )
            )
        return tuple(entries)

    @api(in_process=True)
    @_guarded_write
    def set_protocol_filepath(self, file_path: str) -> None:
        """Remember ``file_path`` as the protocol to open at the next start; ``''`` forgets it.

        The one writer of ``protocol.filepath``, which the next start opens.

        Args:
            file_path: The protocol file to open at the next start; ``''``
                opens none.

        Raises:
            Nothing.
        """
        with self.settings_lock:
            self._store_setting('protocol.filepath', file_path)

    @_guarded_write
    def open_protocol(self, file_path: FilePath) -> 'Protocol':
        """Load the protocol at ``file_path`` with its Layer Settings, and remember it.

        The GUI's Load: ``load_protocol``, then ``apply_channel_settings``, then
        ``set_protocol_filepath``, so the next start opens it. The path is
        written last, once the scope is on the protocol's plate and the layer
        controls hold its settings: a refused file leaves the remembered path
        where it was.

        It is not part of the L2 API surface, like the remembered path it
        writes.

        A file that is refused or fails to load leaves no path remembered.

        Raises:
            LiveFolderPathRefusedError: ``'path_not_found'``, nothing is at
                ``file_path``; ``'not_a_file'``, a folder is.
            ProtocolNotLoadedError: As ``load_protocol``.
            ProtocolFormatError: As ``load_protocol``.
            ProtocolRunRefusedError: ``'objective_not_mounted'``,
                ``'objectives_require_turret'``,
                ``'turret_objectives_unassigned'`` or
                ``'layer_not_on_scope'``: the file holds a step this scope
                cannot perform, a step naming a channel no release has
                (``'channel_unknown'``), or Layer Settings no layer setting
                takes (``'values_out_of_range'``), as ``load_protocol``.
            ConfigError: As ``load_protocol`` and ``apply_channel_settings``.
            HardwareCommandRefusedError: ``'exclusive_activity_running'``,
                as ``load_protocol`` and ``apply_channel_settings``.
        """
        protocol = self.load_protocol(file_path)
        self.apply_channel_settings(protocol)
        self.set_protocol_filepath(os.fspath(file_path))
        return protocol

    @_guarded_write
    def open_remembered_protocol(self) -> 'Protocol | None':
        """Load the protocol the last start left behind, with its Layer Settings.

        The start-up half of ``set_protocol_filepath``: ``open_protocol`` on the
        remembered path. A path is forgotten only when there is no
        file left to remember, or the file cannot be read. A refusal keeps it:
        the file is real and was chosen, and what is wrong (the turret's
        glass, the plate, a layer, the file's contents) can be put right and
        the protocol loaded again.

        The GUI's start-up adoption; not part of the L2 API surface, like the
        remembered path it reads.

        Returns:
            The protocol, or None when no path is remembered or no file is
            at it, whether nothing or a folder is there (the path is then
            forgotten, logged at INFO: a protocol moved or deleted since is
            not a fault).

        Raises:
            ProtocolNotLoadedError: The file is there and cannot be read; the
                path is forgotten.
            ProtocolFormatError: As ``load_protocol``; the path is kept.
            ProtocolRunRefusedError: As ``load_protocol``; the path is kept.
            ConfigError: As ``load_protocol``; the path is kept.
            HardwareCommandRefusedError: As ``load_protocol``; the path is
                kept.
        """
        with self.settings_lock:
            file_path = self.settings['protocol']['filepath']
        # Asked before any path test: Path('') is the working folder, which
        # exists, so an empty path would be loaded and fail as a fault.
        if not file_path:
            return None
        try:
            return self.open_protocol(file_path)
        except LiveFolderPathRefusedError as gone:
            logger.info(f'[Session   ] The remembered protocol is gone ({gone}); forgotten.')
            self.set_protocol_filepath('')
            return None
        except ProtocolNotLoadedError:
            self.set_protocol_filepath('')
            raise

    def _store_settings(self, updates: dict[str, object]) -> None:
        """Put each of ``updates`` (dotted path to value) in the live settings, under one ``settings_lock`` hold.

        The scope's writer (``Lumascope.bind_settings``): the camera saves
        the settings it changes through it, past ``update_settings``'
        refusal of the settings it owns.
        """
        with self.settings_lock:
            for path, value in updates.items():
                self._store_setting(path, value)

    def _store_setting(self, path: str, value: object) -> None:
        """Under ``settings_lock``: put ``value`` at ``path`` in the live settings.

        The Session's own members write their settings through this,
        past ``update_settings``' refusal of the settings they own. The
        value is stored as plain data (``python_values``): a closed set's
        member as its string, a numpy scalar as its number. Every write
        wakes the saver, which saves a window later, outside the lock.

        While a control session is held, a write that is not its holder's is
        refused here too, preferences aside: every member that writes is a
        guarded write and was refused at its entry, so this refuses only a
        member that skipped it, loudly, instead of storing.

        Raises:
            ControlSessionRefusedError: ``'control_session_held'``. Nothing
                was stored.
        """
        if path not in settings_paths.PREFERENCES:
            session = self.activity_claim.refusing_control(current_taking())
            if session is not None:
                raise control_refused(session)
        *blocks, leaf = path.split('.')
        holder = self.settings
        for block in blocks:
            holder = holder[block]
        holder[leaf] = python_values(value)
        self.settings_lock.stored(path)
        if self._settings_saver is not None:
            self._settings_saver.wake()

    def _start_saving_to(self, file: FilePath) -> None:
        """Save every accepted write to ``file`` from now on, and the settings at the close."""
        self._settings_file = os.fspath(file)
        self._settings_saver = SettingsSaver(self._save_settings_file)
        self._settings_saver.start()

    def _save_settings_file(self) -> None:
        """Write the live settings to the session's file, whole, or say why not.

        On the saver's thread, and once on the close's after the saver has
        stopped, so two saves never run at once. Never under
        ``settings_lock``: the snapshot takes it, and a save started inside
        a write would wait on itself.

        A save withheld -- the settings provisional, or no hardware this
        session -- leaves every write in effect and in memory, and is told
        once per reason. A write the drive refuses is reported where it
        fails, and the file there is unchanged.
        """
        file = self._settings_file
        if settings_init.settings_are_provisional():
            withheld = 'settings_provisional'
        elif self.scope.no_hardware:
            withheld = 'settings_unmeasured'
        else:
            withheld = None
        from modules.notification_center import notifications

        if withheld is not None:
            if withheld not in self._unsaved_told:
                self._unsaved_told.add(withheld)
                notifications.report_outcome(
                    SettingsNotSavedNotice(withheld, file), solicited=False, category='Settings'
                )
            return
        snapshot = self._copy_of_the_store()
        t0 = time.monotonic()
        try:
            write_whole(file, snapshot)
        except OSError as e:
            fault = SettingsFileNotWrittenError(file, e)
            fault.__cause__ = e
            notifications.report_outcome(fault, solicited=False, category='Settings')
            return
        dt = time.monotonic() - t0
        if dt > 0.1:
            logger.warning(f'[Session  ] saving the settings to {file} took {dt * 1000:.0f}ms')

    @api(sets='model', admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def select_model(self, model: str) -> None:
        """Save the operator's scope model for the next start.

        The running scope is not changed: its capabilities are fixed at
        construction, so re-resolving its layers for the new model would
        leave one scope carrying two models -- the new one in its layers,
        the old one in its capabilities and its saved files. The selection
        takes effect when the scope is next brought up, and on a scope whose
        motor board reports its own model the board's report still wins
        there.

        A model that is not the one running is reported as a
        ``ScopeModelDeferredNotice``, so every client hears that it waits
        for the next start.

        Args:
            model: A model the release's catalogue lists.

        Raises:
            ScopeModelUnknownError: the scope's catalogue does not list
                ``model``; nothing is saved.
        """
        scope_models = self.scope.scope_models
        if model not in scope_models:
            raise ScopeModelUnknownError(model, scope_models)
        with self.settings_lock:
            self._store_setting('microscope', model)
        logger.info(f'[Session  ] scope model {model!r} saved; it applies at the next start')
        # Reported once the lock is released: listeners run on this thread.
        if model != self.scope.channel_identity.model:
            from modules.notification_center import notifications

            notifications.report_outcome(
                ScopeModelDeferredNotice(model), solicited=True, category='Microscope'
            )

    @api
    @property
    def model(self) -> str:
        """The saved scope model: the one ``select_model`` saved, brought up at the next start.

        The running scope's model is ``scope.channel_identity.model``; the two
        differ until the next bring-up, which ``model_at_next_start`` says.

        Raises:
            Nothing.
        """
        return self.settings['microscope']

    @api
    @property
    def model_at_next_start(self) -> str | None:
        """The saved model when it is not the one running, else None.

        What ``select_model`` saved waits for the next bring-up, so until
        then the scope runs as one model and the settings name another; a
        caller shows this rather than comparing the two itself.

        Raises:
            Nothing.
        """
        saved = self.settings['microscope']
        return None if saved == self.scope.channel_identity.model else saved

    def _put_a_stageless_scope_on_center_plate(self) -> None:
        """Replace a stored plate a scope with no XY stage cannot be on with Center Plate.

        Such a scope has one field and no wells to move between, so every
        protocol made from the settings is born on Center Plate and no host
        has to move it there afterwards. The stored plate is a value the
        hardware cannot take, replaced by what it can, and logged.
        """
        from modules.labware_loader import CENTER_PLATE

        if self.scope.capabilities.has_xy_stage:
            return
        stored = self.settings['protocol']['labware']
        if not self.select_labware(CENTER_PLATE):
            return
        logger.info(
            f'[Session  ] stored plate {stored!r} replaced by {CENTER_PLATE!r}: '
            'this scope has no XY stage'
        )

    def _replace_an_unresolvable_stored_plate(self) -> list[tuple[str, object, object]]:
        """Replace a stored plate the catalogue cannot resolve with the shipped one.

        An empty name or a plate the catalogue no longer has; a null or a
        non-string never reaches here, since the load replaces it and a
        session's door refuses it, by the writer's rule. The catalogue is
        first available here, at bring-up, so the name is judged here
        rather than with the load's other stored values. Refusing it
        instead brought the GUI up on the shipped template, dropping every
        other stored setting for one plate name.

        Returns:
            ``[('protocol.labware', stored, shipped)]`` when replaced, else ``[]``.
        """
        from modules.labware_loader import CENTER_PLATE

        stored = self.settings['protocol']['labware']
        if self.scope.wellplate_loader.is_known_plate(stored):
            return []
        # A scope with no XY stage has one plate, which no stored value
        # changes; the shipped plate is a stage scope's.
        shipped = (
            self.scope.settings_template['protocol']['labware']
            if self.scope.capabilities.has_xy_stage
            else CENTER_PLATE
        )
        self.select_labware(shipped)
        logger.info(
            f'[Session  ] stored plate {stored!r} replaced by {shipped!r}: '
            'the catalogue has no such plate'
        )
        return [('protocol.labware', stored, shipped)]

    def _replace_an_unknown_stored_objective(self) -> list[tuple[str, object, object]]:
        """Replace a stored objective id the catalogue does not hold with the shipped one.

        On a scope with no turret the stored id is the objective, read at
        bring-up and stamped as the scale of every capture, so an id that
        names no catalogue objective is replaced here, where the catalogue
        is first available, as the plate is. Refusing it instead brought the
        GUI up on the shipped template, dropping every other stored setting
        for one id. A turreted scope does not read the stored id: its
        objective is the slot's assignment, and an unknown assignment is the
        objective question's.

        Returns:
            ``[('objective_id', stored, shipped)]`` when replaced, else ``[]``.
        """
        if self.scope.capabilities.has_turret:
            return []
        stored = self.settings.get('objective_id')
        try:
            self.scope.objective_helper.get_objective_info(objective_id=stored)
        except ConfigError:
            pass
        else:
            return []
        shipped = self.scope.settings_template['objective_id']
        # Stored, not selected: the scope is not initialized yet, so there is
        # no runtime state to write beside the store; ``initialize`` reads
        # the stored id, as it does the model stored above.
        with self.settings_lock:
            self._store_setting('objective_id', shipped)
        logger.info(
            f'[Session  ] stored objective {stored!r} replaced by {shipped!r}: '
            'the catalogue has no such objective'
        )
        return [('objective_id', stored, shipped)]

    @_guarded_write
    def configure_scope(self) -> None:
        """Configure the scope from this session's settings -- the bring-up.

        Once, after construction, on a real scope: adopt the model the
        hardware reports when the catalogue knows it (a WRITE into this
        session's ``settings['microscope']`` -- hardware truth outranks
        the stored selection; a model outside the catalogue, or no motor
        board to ask, leaves the stored one), normalize the turret slot
        keys a caller-supplied dict may still carry as JSON strings,
        resolve the model's catalogue entry, replace a stored plate the
        catalogue cannot resolve with the shipped one (told once, as a
        ``StoredSettingReplacedNotice``), build the init config and run
        ``Lumascope.initialize`` -- which refuses a stored objective the
        catalogue does not have on a scope with no turret; on a turreted
        scope the objective stays unknown until the turret is in a known
        slot. The scope reads the plate and objective from these settings
        whenever it acts on them. Last, the camera takes BF's stored
        settings (``apply_layer_camera``), the one step that waits on the
        camera lane; a setting it does not take, a write that outlasts the
        lane's bound and a camera that leaves are kept on the bring-up
        record and told once, and the bring-up goes on. The
        factories run this for the scope they build; a host that constructs
        the session directly, or hands ``create`` its own scope, calls it
        once itself. Every other step runs on the calling thread.

        Raises:
            ConfigError: a settings key ``initialize`` cannot do without is
                missing (``frame``; ``objective_id`` on a scope with no
                turret); or that ``objective_id`` names no shipped objective.
            HardwareCommandRefusedError: a run, a diagnostic or a recording
                holds the scope. The configuration rewrites the LEDs, the
                camera geometry and acceleration under whatever holds it,
                and its writes run inline, where no lane refuses them.
        """
        replaced: list[tuple[str, object, object]] = []
        try:
            self._configure_scope(replaced)
        finally:
            _tell_stored_replacements(replaced)

    def _configure_scope(self, replaced: list[tuple[str, object, object]]) -> None:
        """``configure_scope``'s steps, adding what they replace to ``replaced``."""
        from modules.scope_init_config import ScopeInitConfig

        holder = self.activity_claim.holder
        if holder is not None:
            raise HardwareCommandRefusedError(
                'exclusive_activity_running', 'configure_scope', holder.kind
            )
        scope_models = self.scope.scope_models
        # The hardware's own model outranks the stored selection, and it
        # has to land before the two reads of the selection below, or a
        # unit whose file says the wrong model configures for the wrong
        # axes. The motor driver caches its identity at connect, so the
        # read is synchronous; no board (or a board with no model) reports
        # None and the stored selection stands. The board's own report, not
        # the scope's resolved model, which falls back to the selection.
        detected = self.scope.diagnostics.get_motor_info().model
        stored = self.settings.get('microscope')
        if detected is not None and detected in scope_models and detected != stored:
            with self.settings_lock:
                self._store_setting('microscope', detected)
            logger.info(
                f'[Session  ] scope reports model {detected}; settings said {stored!r} '
                '-- the hardware wins'
            )
        elif detected is not None and detected not in scope_models:
            logger.info(
                f'[Session  ] scope reports model {detected}, not in the catalogue; '
                f'the stored model {stored!r} stands'
            )
        self._put_a_stageless_scope_on_center_plate()
        scope_config = scope_models.get(self.settings.get('microscope'))
        # Before anything is commanded: every well position is computed on it.
        replaced.extend(self._replace_an_unresolvable_stored_plate())
        replaced.extend(self._replace_an_unknown_stored_objective())
        config = ScopeInitConfig.from_settings(
            self.settings,
            scope_config=scope_config,
            channel_identity=self.scope.channel_identity,
        )
        self.scope.initialize(config)
        self._date_an_undated_objective_confirmation()
        # The camera streams from here on, in every host that configures a
        # scope, so the imaging API starts watching the stream here: a stall
        # nobody is reading is reported to every client, not only to a GUI.
        self.scope.imaging.start_stream_check(self._scheduler)
        # Likewise a finished run's file writer: nobody waits on its files
        # once the run has ended, so a stuck one is reported to every client.
        self.sequenced_capture_runner.start_file_writer_check(self._scheduler)
        # Read once so a session that changes nothing still records the
        # scale it starts with (the read records the optics). On a turreted
        # scope the slot is not known until the turret is homed, so the
        # record says that instead.
        if self.scope.runtime_state.get_current_objective() is None:
            logger.info(
                '[Session  ] objective at bring-up: unknown until the turret is in a known slot'
            )
        self._tell_capped_layers()
        self._apply_bring_up_layer()

    def _tell_capped_layers(self) -> None:
        """Tell, as one notice, every saved layer gain or exposure the attached camera cannot reach.

        A saved layer value above the camera's range is kept as saved and
        applied at the camera's limit (``CameraAPI.applied_gain_db_for``
        and kin); each apply logs it, and this is where the person is told,
        once per bring-up for every capped layer at once. With no camera
        the cache declares no limit, so nothing is capped and nothing is
        told.
        """
        camera = self.scope.camera
        capped: list[tuple[str, str, float, float]] = []
        with self.settings_lock:
            for record in self.scope.channel_identity.channels:
                layer = record.key_name
                gain = camera.applied_gain_db_for(self.settings[layer]['gain_db'])
                if gain.capped:
                    capped.append((layer, 'gain_db', gain.stored_db, gain.applied_db))
                exposure = camera.applied_exposure_ms_for(self.settings[layer]['exposure_ms'])
                if exposure.capped:
                    capped.append((layer, 'exposure_ms', exposure.stored_ms, exposure.applied_ms))
        if not capped:
            return
        from modules.exceptions import LayerSettingCappedNotice
        from modules.notification_center import notifications

        notifications.report_outcome(
            LayerSettingCappedNotice(capped), solicited=False, category='Camera'
        )

    def _apply_bring_up_layer(self) -> None:
        """Show BF dark and put its stored camera settings on the camera, as the GUI opens on BF.

        Bring-up has darkened every LED, so BF is shown dark whatever was
        shown before. The camera half is skipped, and said so once, when
        there is no camera to set (bring-up has already reported it missing;
        a second report would only repeat it) or this scope has no BF layer
        (a newer unit whose layers this release cannot resolve still comes
        up). The values are capped to the camera's range first, so a
        refusal here is the camera failing a write. It is one of bring-up's
        camera writes (``scope.write_camera_at_bring_up``): a setting the
        camera refuses or does not answer, a write that outlasts the camera
        lane's bound and a camera that leaves are kept on the camera's part
        of the bring-up record and told once, and the scope comes up.
        """
        layer = common_utils.DEFAULT_LAYER
        self._show_layer(layer)
        if not self.scope.camera_connected:
            logger.info(f'[Session  ] bring-up: no camera, so {layer} is not applied to it')
            return
        if layer not in self._layers_on_scope():
            logger.info(
                f'[Session  ] bring-up: this scope has no {layer} layer, '
                'so no layer is applied to the camera'
            )
            return
        self.scope.write_camera_at_bring_up(lambda: self.apply_layer_camera(layer))

    @api(reads=True)
    def bring_up_record(self) -> 'BringUpRecord':
        """What bring-up found, substituted and set aside, for a client that asks later.

        Which parts came up and, for each that did not, why; what bring-up
        used in place of a saved setting the camera could not take, the
        saved value beside it; and the settings file set aside with its
        reason, while the app runs on the shipped template. A client given
        to ``create(outcome_listener=...)`` heard these as outcomes; this is
        the same facts held, read-through from the scope and the settings
        store, for one that connects afterwards.

        Raises:
            Nothing.
        """
        from modules.lumascope_api.bring_up import SettingsSetAside

        rejected = settings_init.rejected_current_json
        set_aside = (
            None if rejected is None else SettingsSetAside(pathlib.Path(rejected[0]), rejected[1])
        )
        return dataclasses.replace(self.scope.bring_up_record(), settings_set_aside=set_aside)

    @api
    @slow_task_budget(SUPPORT_REPORT_SLOW_TASK_S)
    def make_support_report(
        self,
        *,
        output_dir: FilePath,
        include_bandwidth_test: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> 'SupportReportSaved':
        """Make the full Tech Support Report: the boards, the motors, the camera and the files.

        Blocks for minutes. The hardware steps hold the scope for a
        diagnostic, so a run cannot start under a homing or a fan sweep; a
        report started while a run holds the scope skips those steps and
        says so in the ZIP. A step that fails is written into the ZIP; the
        report does not stop for it.

        Args:
            include_bandwidth_test: Also time the camera's frame delivery
                (adds minutes).
            output_dir: The folder the ZIP is written into, made if it is
                missing. A host that writes where a person looks passes
                ``path_utils.desktop_folder()``; the session chooses none.
            on_progress: Called with a percentage and what the report is
                doing, from the thread the report runs on.

        Returns:
            Where the ZIP is, and the words that say where to send it.

        Raises:
            LiveFolderPathRefusedError: ``'not_a_folder'``, a file is at
                ``output_dir`` or above it; refused before anything is made.
            SupportReportNotSavedError: no ZIP was saved; chained from the
                failure, whose words it carries.
        """
        from modules.tech_support_report import (
            SupportReportKind,
            SupportReportSaved,
            TechSupportReport,
        )

        report = TechSupportReport(session=self)
        with self._count_in_flight(live_work.WorkKind.SUPPORT_REPORT):
            return SupportReportSaved(
                report.generate(
                    callback=on_progress,
                    include_bandwidth_test=include_bandwidth_test,
                    output_dir=output_dir,
                ),
                SupportReportKind.SUPPORT_REPORT,
            )

    @api
    def make_logs_zip(
        self,
        *,
        output_dir: FilePath,
        on_progress: ProgressCallback | None = None,
    ) -> 'SupportReportSaved':
        """Zip the logs, the data folder, the recent protocols and the video receipts.

        Touches no hardware and does not hold the scope: it is for sending
        the record of what already happened.

        Args:
            output_dir: The folder the ZIP is written into, made if it is
                missing. A host that writes where a person looks passes
                ``path_utils.desktop_folder()``; the session chooses none.
            on_progress: Called with a percentage and what the zip is doing,
                from the thread it runs on.

        Returns:
            Where the ZIP is, and the words that say where to send it.

        Raises:
            LiveFolderPathRefusedError: ``'not_a_folder'``, a file is at
                ``output_dir`` or above it; refused before anything is made.
            SupportReportNotSavedError: no ZIP was saved; chained from the
                failure, whose words it carries.
        """
        from modules.tech_support_report import (
            SupportReportKind,
            SupportReportSaved,
            TechSupportReport,
        )

        report = TechSupportReport(session=self)
        with self._count_in_flight(live_work.WorkKind.LOGS_ZIP):
            return SupportReportSaved(
                report.generate_logs_only(callback=on_progress, output_dir=output_dir),
                SupportReportKind.LOGS_ZIP,
            )

    @api
    @property
    def app_version(self) -> str | None:
        """The application's version, ``4.0.0-beta36``; None where the build has none.

        ``version.txt``'s first line, through its one reader. A build with no
        readable ``version.txt`` answers None rather than a name it does not
        have.

        Raises:
            Nothing.
        """
        version, _build_timestamp = path_utils.read_version()
        return version or None

    @api
    @property
    def app_runtime(self) -> path_utils.AppRuntime:
        """How the application was launched: ``source``, ``bundle`` or ``installed``.

        ``source``: Python running a checkout or pip install. ``bundle``: a
        packaged build the installer did not install. ``installed``: a
        packaged build in the folder the installer wrote, whose data lives in
        the user's Documents folder.

        Raises:
            Nothing.
        """
        return path_utils.app_runtime()

    @api(reads=True)
    def plugin_health(self) -> 'PluginHealth | None':
        """The loaded plugins, the ones that did not load, and their runtime errors.

        None on a session whose host never asked for plugins (``load_plugins``).

        Raises:
            Nothing.
        """
        return None if self.plugins is None else self.plugins.health()

    @api(in_process=True)
    def load_plugins(self) -> None:
        """Load the installed plugins and the built-ins into this session.

        The host's one call to have plugins; a session it is never made on
        has none. Each plugin's ``register`` is handed this session as its
        ctx. A plugin that does not load is reported and the rest load;
        nothing here raises for a plugin. Under ``no_engineering`` the
        plugin named ``etaluma_engineering`` is left out, unreported, as if
        not installed.
        Call it once, after ``create``.

        Raises:
            AlreadyStartedError: ``'plugins_loaded'``, this session already
                loaded its plugins. Still a ``RuntimeError``.
        """
        if self.plugins is not None:
            raise AlreadyStartedError(
                'plugins_loaded', 'This session already loaded its plugins; it loads them once.'
            )
        version, _build_timestamp = path_utils.read_version()
        self.plugins = PluginRegistry(self.scope)
        self.plugins.load(self, version)

    @api(in_process=True)
    def unload_plugins(self) -> None:
        """Call each loaded plugin's ``unregister``, last loaded first.

        A host whose close has to stop plugin work before anything else --
        the GUI's, before it stops runs and saves settings -- calls this
        first; ``shutdown`` calls it as its first step for every other host.
        A second call, and a call on a session with no plugins, does
        nothing. A plugin's own failure is logged and the rest unload.

        Raises:
            Nothing.
        """
        if self.plugins is not None:
            self.plugins.unload(self)

    def _run_protocol_complete_processors(
        self,
        run_dir: pathlib.Path,
        write_batch: 'RunWriteBatch',
        trigger_source: Caller,
        protocol_name: str,
    ) -> None:
        """Hand a finished Full Protocol's folder to the opted-in processors.

        The manifest's ``trigger_source`` is who started the run: ``'gui'``,
        ``'rest'`` or ``'api'`` (``modules.caller``).

        The run engine calls this once per Full Protocol at the run's end,
        once its own builds are on the post-processing lane. The processors
        are put on the lane behind them, isolated from the run, without
        waiting: there they wait for the run's images to be written, then run
        with the files' outcome. Their task's outcome is answered here as it
        settles, so it is never left with nobody to read it.
        """
        if self.plugins is None:
            return
        from modules.notification_center import notifications

        run_dir_str = str(run_dir)
        manifest = {
            'protocol_name': protocol_name,
            'run_dir': run_dir_str,
            'trigger_source': trigger_source,
        }

        def processors() -> None:
            write_batch.wait_complete(None)
            self.plugins.run_protocol_complete_processors(
                run_dir_str, manifest, run_dir_str, write_batch.outcome
            )

        def answer(settled: concurrent.futures.Future) -> None:
            # The processors report their own failures inside the task; what
            # reaches here is the task's own.
            failure = settled.exception()
            if failure is not None:
                notifications.report_outcome(failure, solicited=False, category='Plugins')

        waiter = concurrent.futures.Future()
        waiter.set_running_or_notify_cancel()
        try:
            self.post_processing.lane.submit(
                IOTask(action=processors), 'plugins.run_protocol_complete_processors', waiter=waiter
            )
        except Exception as refused:
            # The lane refused the task (closed at shutdown).
            notifications.report_outcome(refused, solicited=False, category='Plugins')
            return
        waiter.add_done_callback(answer)

    @api
    @property
    def plugin_api_level(self) -> int:
        """What this LumaViewPro does that a plugin may rely on.

        ``modules.plugins.PLUGIN_API_LEVEL``; each level's additions are
        listed in plugin-tutorial.md, "The plugin API level".

        Raises:
            Nothing.
        """
        return PLUGIN_API_LEVEL

    @api(in_process=True)
    def host_steady_ns(self) -> int:
        """The host's steady time now, in nanoseconds.

        The clock every steady stamp the scope takes is on, for an
        in-process caller setting its own "now" against them.

        Raises:
            Nothing.
        """
        return self.scope.host_steady_ns()

    @api(reads=True)
    def settings_are_provisional(self) -> bool:
        """Is the app running on defaults nobody has agreed to keep?

        True while the user's current.json could not be used and no one
        has decided its fate. While it holds, accepted writes are kept in
        memory and not saved (``SettingsNotSavedNotice``) -- resolve with
        retire_rejected_settings() after the user has chosen to start
        over.

        Raises:
            Nothing.
        """
        return settings_init.settings_are_provisional()

    @api
    def retire_rejected_settings(self) -> pathlib.Path | None:
        """Resolve the provisional-settings state: retire the rejected file.

        Moves the unusable current.json aside (renamed, never deleted --
        it is the user's only copy) so a fresh one can take its place,
        and clears the provisional state, so the session saves again: at
        its next accepted write, and at its close. Call only
        after a human has chosen to start over. Returns the retired
        path, or None when nothing was provisional.

        Raises:
            SettingsFileNotReplacedError: the rejected file could not be moved
                aside. The settings stay provisional, so the question can be
                answered again.
        """
        return settings_init.retire_rejected_current_json()

    @api(in_process=True, reads=True)
    def capture_settings_snapshot(self) -> dict:
        """A settings snapshot for composing a capture or a run.

        ``get_settings_snapshot``, refused while the active objective
        (``runtime_state.resolve_current_objective``) is unknown: a capture
        stamps its scale, so it is never composed without one.

        Raises:
            ObjectiveUnknownError: The active objective is unknown:
                ``'slot_unknown'``, ``'slot_unassigned'`` or
                ``'not_in_catalogue'`` on a turret scope; ``'none_selected'``
                with no turret.
        """
        objective_id, _ = self.scope.runtime_state.resolve_current_objective()
        snapshot = self.get_settings_snapshot()
        snapshot['objective_id'] = objective_id
        self._read_autofocus_as_off_without_z(
            snapshot[layer] for layer in common_utils.get_layers()
        )
        return snapshot

    # ------------------------------------------------------------------
    # The objective: the question, the answer and the plain writers
    # ------------------------------------------------------------------

    @api(reads=True)
    def objective_question(self) -> 'ObjectiveQuestion | None':
        """Does the objective need confirming? The question, or None.

        A read, for any host: the GUI renders the answer as a popup, a
        REST caller reads it as state. Two ways the session cannot know
        what is in the light path: no person has confirmed the objective
        for this scope's light path (never on this install -- the settings
        template ships a default that would otherwise set image scale
        silently forever -- or only on a scope that differed in having a
        turret), or, on a turret scope (``capabilities.has_turret``, the
        declared model's answer when the board did not come up, so a stale
        objective does not pass unasked then), the slot in the light path
        has no assignment. The slot is the live one
        (``motion.turret_status``), so an answer names the glass that is
        actually in the light path.

        Four conditions withhold an owed question, each leaving one log
        line per call so a bundle can say why nothing was asked: with no
        hardware there is nothing in the light path and no capture to
        stamp; while settings are provisional nothing is saved, so an
        answer given now would be lost -- the host re-asks when they
        resolve; while an activity holds the scope, or a control session
        is held by someone other than the caller, the answer is refused
        too (``confirm_objective``), so asking would only ask again after
        each refusal -- the host re-asks when the hold or the session
        ends. No line is
        logged for a returned question: the renderer logs its own show,
        and a polled read must not log per poll.

        Raises:
            ObjectiveUnknownError: ``'slot_unknown'``, the objective is not
                confirmed for this light path and the turret model's slot is
                unknown -- there is no slot to answer for until the turret is
                homed or moved.
        """
        has_turret = self.scope.capabilities.has_turret
        confirmed = self.settings['objective_confirmed']
        first_run = confirmed != self._objective_topology()
        slots = self.settings['turret_objectives']
        slot = self.scope.motion.turret_status().slot if has_turret else None
        # A known slot with no assignment owes the question. An unknown slot
        # alone does not: it is unknown during every turret move, and asking
        # then would put the question up while the turret is still turning.
        # The objective is unknown meanwhile, and captures refuse with why.
        slot_unassigned = has_turret and slot is not None and slots.get(slot) is None
        if not (first_run or slot_unassigned):
            return None
        if self.scope.no_hardware:
            logger.info('[Session  ] objective question withheld -- no hardware this session')
            return None
        if self.settings_are_provisional():
            logger.info(
                '[Session  ] objective question deferred -- settings are provisional and '
                'the answer could not be kept'
            )
            return None
        holder = self.activity_claim.owner
        if holder is not None:
            logger.info(
                f'[Session  ] objective question deferred -- a {holder} holds the scope and '
                'would refuse the answer'
            )
            return None
        control = self.activity_claim.refusing_control(current_taking())
        if control is not None:
            logger.info(
                f'[Session  ] objective question deferred -- "{control.name}" has control '
                'and would refuse the answer'
            )
            return None
        if has_turret and slot is None:
            raise ObjectiveUnknownError('slot_unknown')
        return self._objective_question_at(slot)

    @api(reads=True)
    def current_slot_objective_question(self) -> ObjectiveQuestion:
        """The objective question for the turret slot in the light path, owed or not.

        A person asking to say which objective is installed in the current
        slot: to change an assignment, or to name a slot nothing is assigned
        to, at any time rather than only when ``objective_question`` owes it.
        The question is that read's, proposing the slot's assignment, so an
        unchanged answer leaves the slot as it is. The answer goes back
        through ``confirm_objective`` with the slot the question names, which
        is the slot written even if the turret has moved since. On a scope
        without a turret the question names no slot and proposes the
        selected objective once one has been confirmed, the catalogue
        default before.

        A hold or provisional settings refuse nothing here: the answer's own
        write refuses a change under a hold, and a write while settings are
        provisional is kept in memory like any other setting's, and not
        saved (``settings_are_provisional``).

        Raises:
            ObjectiveUnknownError: ``'slot_unknown'``, the slot in the light
                path is unknown -- the turret is turning or has not been homed.
        """
        if not self.scope.capabilities.has_turret:
            return self._objective_question_at(None)
        slot = self.scope.motion.turret_status().slot
        if slot is None:
            raise ObjectiveUnknownError('slot_unknown')
        return self._objective_question_at(slot)

    def _objective_question_at(self, slot: 'int | None') -> ObjectiveQuestion:
        # The one proposal rule for both reads; slot is None only on a
        # turretless scope.
        slots = self.settings['turret_objectives']
        confirmed = self.settings['objective_confirmed']
        choices = tuple(self.scope.objective_helper.get_objectives_list())
        # The slot's own assignment names glass someone confirmed. On a
        # turretless scope the stored objective_id does once a confirmation
        # exists: owed then, the confirmation was a turret's, and the stored id
        # is the pick the scope last had before it. Never confirmed, it is the
        # shipped template's value. Everything else gets the catalogue's default.
        if slot is not None:
            proposed = slots.get(slot)
        else:
            proposed = self.settings['objective_id'] if confirmed is not None else None
        if proposed not in choices:
            from modules.objectives_loader import DEFAULT_PROPOSED_OBJECTIVE_ID

            proposed = DEFAULT_PROPOSED_OBJECTIVE_ID
        return ObjectiveQuestion(turret_slot=slot, proposed=proposed, choices=choices)

    @api(admitted_by=(Admission.LANE_CALL, Admission.GUARDED_WRITE))
    @_guarded_write
    def confirm_objective(self, objective_id: str, slot: 'int | None' = None) -> bool:
        """Answer the objective question: this objective is in the light path.

        On a turret scope, assigns the objective to ``slot``: the answer
        records the glass in a slot and never turns the turret. With no
        turret, selects it (``select_objective``). Records that a person has
        confirmed the objective for this light path, turret or turretless,
        so a scope brought up with the other asks again. Returns whether the
        active objective changed.

        Args:
            objective_id: The objective's catalogue key.
            slot: The turret slot, 1-4, the objective is in; required on a
                turret scope, and refused on a scope with none. None is the
                default.

        Raises:
            ArgumentRefusedError: ``'slot_required'``, no
                ``slot`` on a turret scope. Nothing is written.
            HardwareCommandRefusedError: ``'axis_absent'``, a
                ``slot`` on a scope with no turret. Nothing is
                written.
            CatalogueNameRefusedError: ``'objective_not_in_catalogue'``,
                ``objective_id`` is not exactly a catalogue key. Nothing is
                written.
            PositionOutOfRangeError: ``slot`` is not a slot
                number 1-4.
        """
        if self.scope.capabilities.has_turret:
            if slot is None:
                raise ArgumentRefusedError('slot_required', argument='slot', value=None)
            before = self.scope.runtime_state.get_current_objective_id()
            self.assign_turret_objective(slot, objective_id)
            changed = self.scope.runtime_state.get_current_objective_id() != before
        else:
            if slot is not None:
                self._refuse_without_a_turret('confirm_objective')
            changed = self.select_objective(objective_id)
        with self.settings_lock:
            self._store_setting('objective_confirmed', self._objective_topology())
        logger.info(
            f'[Session  ] objective confirmed: {objective_id!r}'
            + (f' at turret slot {slot}' if slot is not None else '')
        )
        return changed

    @api
    @property
    def objective_id(self) -> str | None:
        """The active objective's catalogue key, or None while it is not known.

        On a turret scope it is the assignment of the slot in the light path,
        None while that slot is unknown or unassigned; on a turretless scope,
        the one ``select_objective`` selected
        (``scope.runtime_state.resolve_current_objective`` says why it is
        unknown).

        Raises:
            Nothing.
        """
        return self.scope.runtime_state.get_current_objective_id()

    @api(sets='objective_id', admitted_by=(Admission.LANE_CALL, Admission.GUARDED_WRITE))
    @_guarded_write
    def select_objective(self, objective_id: str) -> bool:
        """Make ``objective_id`` the active objective. Returns whether it changed.

        The one writer of the active objective for every host. With no
        turret, the selected objective is the ``objective_id`` setting, which
        the scope reads. On a turret scope the active objective is the
        assignment of the slot in the light path, so picking one turns the
        turret to the one slot holding it (``move_turret``, Z parked and
        restored) and waits for it; which glass is in which slot is
        ``assign_turret_objective``'s, and a pick writes no slot. The resolved
        optics are recorded by the scope's runtime state the next time the
        objective is read, before any capture stamps it. Picking the
        objective already active is a no-op.

        Args:
            objective_id: The objective's catalogue key.

        Raises:
            CatalogueNameRefusedError: ``'objective_not_in_catalogue'``,
                ``objective_id`` is not exactly a catalogue key. The refusal
                lands before any write or move.
            ArgumentRefusedError: On a turret scope,
                ``'objective_in_several_slots'``, two or more slots hold
                ``objective_id`` (``offered``: the slots), or
                ``'objective_not_on_turret'``, none does (``offered``: what
                the slots hold). Nothing is moved.
            HardwareCommandRefusedError: A run, a diagnostic or a recording
                holds the scope (``'exclusive_activity_running'``), nothing
                written or moved; or the turret move's own refusal,
                ``'not_connected'`` or ``'axis_absent'``, nothing moved.
            AxisStateUnknownError: The turret has not been homed.
            MoveFaultError: The Z park, the turret move or the Z restore
                did not arrive (see ``MotionAPI.start_move_z``); the slot is
                unknown afterwards.
            MoveStoppedError: ``'stopped'`` or ``'superseded'``, the Z park,
                the turret move or the Z restore was ended; the slot is
                unknown afterwards.
        """
        # Refuses an id that is not a catalogue key, before any write and
        # before the no-op compare: an unknown active objective is None, so
        # a None asked for would otherwise match it and pass unrefused.
        self.scope.objective_helper.get_objective_info(objective_id=objective_id)
        if objective_id == self.scope.runtime_state.get_current_objective_id():
            return False
        self._refuse_configuration_change_while_held('select_objective')
        if self.scope.capabilities.has_turret:
            self.scope.motion.move_turret(self._the_slot_holding(objective_id), restore_z=True)
        else:
            with self.settings_lock:
                self._store_setting('objective_id', objective_id)
        return True

    def _the_slot_holding(self, objective_id: str) -> int:
        slots = self.scope.runtime_state.get_turret_config()
        holding = slots_holding(objective_id, slots)
        if len(holding) > 1:
            raise ArgumentRefusedError(
                'objective_in_several_slots',
                argument='objective_id',
                value=objective_id,
                offered=holding,
            )
        if not holding:
            raise ArgumentRefusedError(
                'objective_not_on_turret',
                argument='objective_id',
                value=objective_id,
                offered=tuple(held for _, held in sorted(slots.items()) if held is not None),
            )
        return holding[0]

    def _objective_topology(self) -> str:
        # Which light path a confirmation given now is for.
        turret, turretless = settings_paths.OBJECTIVE_TOPOLOGIES
        return turret if self.scope.capabilities.has_turret else turretless

    def _date_an_undated_objective_confirmation(self) -> None:
        # A confirmation stored before it recorded its light path is taken as
        # this scope's: the upgrade asks nothing, and a later change between
        # turret and turretless asks. Only bring-up knows which this scope is.
        if self.settings['objective_confirmed'] is not True:
            return
        topology = self._objective_topology()
        with self.settings_lock:
            self._store_setting('objective_confirmed', topology)
        logger.info(f'[Session  ] objective confirmation recorded as given on a {topology} scope')

    def _refuse_without_a_turret(self, member: str) -> None:
        # The scope fact is said first, before the slot or the catalogue is
        # judged: a slot write on a turretless scope lay in the settings until
        # a change of model made it a turret's assignment.
        if not self.scope.capabilities.has_turret:
            turret = MissingPart.axis('T')
            raise HardwareCommandRefusedError(turret.reason, member, missing=turret)

    # ------------------------------------------------------------------
    # The labware
    # ------------------------------------------------------------------

    @api
    @property
    def labware_id(self) -> str:
        """The current plate's catalogue key, the one ``select_labware`` selected.

        Its record is ``scope.labware[labware_id]``.

        Raises:
            Nothing.
        """
        return self.settings['protocol']['labware']

    @api(sets='labware_id', admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def select_labware(self, labware_id: str) -> bool:
        """Make ``labware_id`` the current plate. Returns whether it changed.

        The one writer of the active labware for every host: the plate the
        settings name, which the scope reads.

        The name is stored in the catalogue's spelling: a plate renamed
        since a protocol or settings file named it is accepted under the old
        name and written under the key, so the settings store never carries
        a spelling the catalogue lacks, and whether the plate changed is
        decided on the key rather than on how it was spelled.

        On a scope with no XY stage the one plate is Center Plate: another
        plate's wells are places it cannot move between, and a protocol made
        on one would image one field under every well's name.

        Args:
            labware_id: The plate's name in the labware catalogue, or a name
                the plate was known by before a rename.

        Raises:
            CatalogueNameRefusedError: ``'labware_unknown'``, the labware
                catalogue has no plate by that name. Nothing is written.
            HardwareCommandRefusedError: ``'axis_absent'``, this scope has no
                XY stage and ``labware_id`` is not Center Plate
                (``missing`` is ``MissingPart.XY_STAGE``). A run, a diagnostic
                or a recording holds the scope and ``labware_id`` is not the
                plate in place (``'exclusive_activity_running'``): each states
                its positions against the plate it started with. Nothing is
                written.
        """
        from modules.labware_loader import CENTER_PLATE

        labware_id = self.scope.wellplate_loader.resolve_plate_key(labware_id)
        if not self.scope.capabilities.has_xy_stage and labware_id != CENTER_PLATE:
            raise HardwareCommandRefusedError(
                'axis_absent', 'select_labware', missing=MissingPart.XY_STAGE
            )
        if labware_id == self.settings['protocol']['labware']:
            # A holder is refused only a different plate: the GUI re-selects
            # the current one whenever its panels redraw, under any hold.
            return False
        self._refuse_configuration_change_while_held('select_labware')
        with self.settings_lock:
            self._store_setting('protocol.labware', labware_id)
        logger.info(f'[Session  ] Labware set to {labware_id!r}')
        return True

    @api(sets='turret_objective', admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def assign_turret_objective(self, slot: int, objective_id: str) -> None:
        """Bind ``objective_id`` to the turret's ``slot``.

        Binding the objective the slot already holds is a no-op.

        Args:
            slot: The turret slot, 1-4.
            objective_id: The objective's catalogue key.

        Raises:
            HardwareCommandRefusedError: ``'axis_absent'``, the scope has no
                turret, said before ``slot`` or ``objective_id`` is
                judged; or a run, a diagnostic or a recording holds the
                scope (``'exclusive_activity_running'``). Nothing is written.
            PositionOutOfRangeError: ``slot`` is not a slot number 1-4.
            CatalogueNameRefusedError: ``'objective_not_in_catalogue'``,
                ``objective_id`` is not exactly a catalogue key.
        """
        self._refuse_without_a_turret('assign_turret_objective')
        refuse_unless_turret_slot(slot)
        self.scope.objective_helper.get_objective_info(objective_id=objective_id)
        if self.settings['turret_objectives'].get(slot) == objective_id:
            return
        self._refuse_configuration_change_while_held('assign_turret_objective')
        with self.settings_lock:
            # The slots are keyed by integer, which a dotted path cannot
            # name, so the slot map is stored whole.
            slots = {**self.settings['turret_objectives'], slot: objective_id}
            self._store_setting('turret_objectives', slots)

    @api(reads=True)
    def turret_objective(self, slot: int) -> 'str | None':
        """The objective bound to the turret's ``slot``, or None while it holds none.

        What ``assign_turret_objective`` bound; the active objective is the
        slot's in the light path (``objective_id``).

        Args:
            slot: The turret slot read, 1-4.

        Raises:
            HardwareCommandRefusedError: ``'axis_absent'``, the scope has no
                turret, said before ``slot`` is judged.
            PositionOutOfRangeError: ``slot`` is not a slot number 1-4.
        """
        self._refuse_without_a_turret('turret_objective')
        refuse_unless_turret_slot(slot)
        return self.settings['turret_objectives'][slot]

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def clear_turret_objective(self, slot: int) -> None:
        """Leave the turret's ``slot`` unassigned.

        Logged for every host: with that slot in the light path the active
        objective is now unknown, and a support bundle can only explain a
        capture refused for that afterwards if the clear is in the record.

        Args:
            slot: The turret slot cleared, 1-4.

        Raises:
            HardwareCommandRefusedError: ``'axis_absent'``, the scope has no
                turret, said before ``slot`` is judged; or a run, a
                diagnostic or a recording holds the scope and the slot has an
                assignment (``'exclusive_activity_running'``). Nothing is
                written.
            PositionOutOfRangeError: ``slot`` is not a slot number 1-4.
        """
        self._refuse_without_a_turret('clear_turret_objective')
        refuse_unless_turret_slot(slot)
        if self.settings['turret_objectives'].get(slot) is not None:
            self._refuse_configuration_change_while_held('clear_turret_objective')
        with self.settings_lock:
            slots = {**self.settings['turret_objectives'], slot: None}
            self._store_setting('turret_objectives', slots)
        logger.info(
            f'[Session  ] Turret slot {slot} cleared; the active objective is now '
            f'{self.scope.runtime_state.get_current_objective_id()!r}'
        )

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def clear_current_turret_objective(self) -> None:
        """Leave the turret slot in the light path unassigned.

        The counterpart of ``assign_turret_objective`` for the slot in the
        light path: that says which objective is installed in a slot, this
        says none is known to be in the current one. The slot is the API's,
        never the caller's guess of it.

        Raises:
            HardwareCommandRefusedError: ``'axis_absent'``, the scope has no
                turret; or a run, a diagnostic or a recording holds the
                scope and the slot has an assignment
                (``'exclusive_activity_running'``). Nothing is written.
            ObjectiveUnknownError: ``'slot_unknown'``, the slot in the light
                path is unknown, so there is no slot to clear.
        """
        self._refuse_without_a_turret('clear_current_turret_objective')
        slot = self.scope.motion.turret_status().slot
        if slot is None:
            raise ObjectiveUnknownError('slot_unknown')
        self.clear_turret_objective(slot)

    def _refuse_configuration_change_while_held(self, member: str) -> None:
        # A run reads the active objective at every capture, so a change
        # mid-run would stamp a different scale into the rest of the run's
        # files than the objective its steps were built for. A diagnostic
        # holds the scope the same way: its measurements are taken against
        # the objective it started under. A recording states every frame at
        # the pixel size and in the plate frame it started with, so a new
        # objective or plate would leave the scope disagreeing with its file.
        holder = self.activity_claim.owner
        if holder is not None:
            raise HardwareCommandRefusedError('exclusive_activity_running', member, holder)

    @api(reads=True)
    def get_current_plate_position(self) -> PlateLocation:
        """The stage's position in plate coordinates, on the session's labware.

        Returns:
            Where the scope is on the plate: X and Y in mm, Z in um.

        Raises:
            AxisStateUnknownError: ``'position_unknown'``, an axis the scope
                has does not know its position (before a home, after a lost
                reference, while it homes), so there is no position to
                convert. Raised, not reported.
            AxisMovingError: ``'axis_moving'``, an axis is moving, so its
                number is a place it is passing through.
            HardwareCommandRefusedError: ``'not_connected'``, the model's
                motor controller is not connected.
        """
        import modules.config_helpers as config_helpers

        return config_helpers.kept_plate_position(
            self.scope, self.settings['protocol']['labware'], tuple(StageAxis), then='try again'
        )

    def plate_position_on(self, labware_id: str) -> PlateLocation:
        """The stage's position in plate coordinates, on the plate ``labware_id`` names.

        For a caller that states the position on a plate other than the
        session's live selection: a protocol's own plate, or the plate in a
        run's settings snapshot. It refuses a moving axis, whose number is a
        place it is passing through; the step or run the position goes into
        is refused there when an axis's position is unknown.
        It is not part of the L2 API surface: a caller states a position
        through the step and run members that read it.

        Returns:
            Where the scope is on that plate: X and Y in mm, Z in um.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, the model's
                motor controller is not connected.
            AxisMovingError: ``'axis_moving'``, an axis it reads is moving,
                so its number is a place it is passing through.
            ConfigError: ``labware_id`` is not a plate the catalogue has.
        """
        import modules.config_helpers as config_helpers

        return config_helpers.get_current_plate_position(self.scope, labware_id)

    # ------------------------------------------------------------------
    # The camera's capture settings: applied, then stored
    # ------------------------------------------------------------------

    @api(sets='scale_bar_enabled', admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def set_scale_bar(self, enabled: bool) -> None:
        """Draw the scale bar on captured images, or stop, and store it.

        The one writer of ``scale_bar.enabled``, which the imaging API reads
        at each capture.

        Args:
            enabled: True draws the scale bar on captured images; False does
                not.

        Raises:
            Nothing.
        """
        with self.settings_lock:
            self._store_setting('scale_bar.enabled', enabled)

    @api
    @property
    def scale_bar_enabled(self) -> bool:
        """Whether captured images are drawn with the scale bar, as ``set_scale_bar`` set it.

        Raises:
            Nothing.
        """
        return self.settings['scale_bar']['enabled']

    @api
    @property
    def acceleration_limit_pct(self) -> int:
        """The motors' acceleration limit, as a percent of the firmware's maximum.

        The one ``set_acceleration_limit`` stored, which bring-up gives the
        motor controller.

        Raises:
            Nothing.
        """
        return self.settings['motion']['acceleration_max_pct']

    @api(sets='acceleration_limit_pct', admitted_by=(Admission.LANE_CALL, Admission.GUARDED_WRITE))
    @_guarded_write
    def set_acceleration_limit(self, limit_pct: int) -> None:
        """Set the motors' acceleration limit, as a percent of the firmware's maximum, and store it.

        The one writer of ``motion.acceleration_max_pct``, stored once the
        motor controller took the value.

        Args:
            limit_pct: The limit, a percent of the firmware's maximum
                acceleration, within the range the motion API accepts.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller to take it. Nothing
                is stored.
            AccelerationLimitRefusedError: ``limit_pct`` is outside the range
                the motion API accepts (a ValueError). Nothing is stored.
        """
        self.scope.motion.set_acceleration_limit(limit_pct=limit_pct)
        with self.settings_lock:
            self._store_setting('motion.acceleration_max_pct', limit_pct)

    # --- Hardware commands: NOT forwarded ---
    # The Session surface deliberately carries no hardware-command
    # forwarders. L2 callers reach hardware through the composition
    # root the Session exposes -- session.scope.illumination.*,
    # session.scope.motion.*, session.scope.imaging.*,
    # session.scope.runtime_state.* -- so every command has exactly one
    # public spelling and the Session owns only what is session-scoped:
    # lifecycle (create / shutdown /
    # start_ / run_application_session / enable_metrics / disable_metrics), the
    # protocol runner, run-state queries, the settings-composition
    # getters above, and the members that apply a setting and store it
    # (a setting has one store, and it is the Session's).

    # ------------------------------------------------------------------
    # Protocol runner
    # ------------------------------------------------------------------

    @api
    def create_protocol_runner(self) -> 'ProtocolRunner':
        """The session's one ProtocolRunner (memoized).

        Wraps the session-composed sequenced-capture engine; repeated
        calls return the same instance. The accessor takes no
        arguments: every dependency (engine, executors, autofocus
        pair) is session composition, so there is nothing per-call to
        configure -- and nothing for a second caller's differing
        configuration to silently lose.

        Raises:
            Nothing.
        """
        from modules.protocol_runner import ProtocolRunner

        if self._protocol_runner is None:
            self._protocol_runner = ProtocolRunner(session=self)
        return self._protocol_runner

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @api(in_process=True)
    def enable_metrics(self, display: DisplayRates | None = None) -> None:
        """Start the session's periodic metrics logging.

        Logs the host's environment fingerprint (``[ENV METRICS]``) first,
        then schedules the periodic ticks. Uses the session's scheduler and the
        ``settings.profiling.metrics_interval_s`` cadence override when
        present. Metrics stay opt-in by the call itself: a host that runs
        the scope for others calls it -- the GUI does, and so does a
        server host -- and a script that only drives the scope need not.

        Args:
            display: The host's live display (the GUI's), whose drawn
                rate and frame intervals the system tick logs beside the
                camera's. A host with no display passes none.

        Raises:
            AlreadyStartedError: ``'metrics_running'``, metrics are already
                running; ``disable_metrics`` lets them start again. A second
                ``MetricsLogger.start`` would silently overwrite its
                schedule handles and orphan the first set as untracked,
                forever-ticking events. Still a ``RuntimeError``.
        """
        if self._metrics_started:
            raise AlreadyStartedError(
                'metrics_running',
                'Metrics are already running; stop them before starting them again.',
            )
        # Cadence resolution, in precedence order: an explicit setting wins
        # everywhere (a bench operator asking for a specific interval must be
        # honoured even on an engineering machine); otherwise engineering mode
        # takes the sub-minute bench cadence; otherwise nothing is passed and
        # the logger's own hourly default applies.
        #
        # The engineering flag is the session's, not settings['mode']: the
        # engineering plugin can turn it on when plugins load, so the settings
        # file and the flag disagree on exactly the machines that care. A host
        # loads its plugins before it starts metrics.
        start_kwargs = {}
        interval_s = self.settings['profiling']['metrics_interval_s']
        if interval_s is not None:
            start_kwargs['system_metrics_interval_s'] = float(interval_s)
        elif self.engineering_mode:
            start_kwargs['system_metrics_interval_s'] = ENGINEERING_METRICS_INTERVAL_S
        # The host's fixed facts (boot time, OS, Pylon SDK, Defender state)
        # head the series they explain, so a post-mortem can read any tick
        # against the exact host state without the ticks repeating them.
        log_environment_once()
        self.metrics_logger.start(self._scheduler, display=display, **start_kwargs)
        self._metrics_started = True

    @api(in_process=True)
    def disable_metrics(self) -> None:
        """Stop the session's periodic metrics logging. Idempotent.

        Never shuts the scheduler itself down -- a shut scheduler
        refuses all future schedules, and the host may start metrics
        again with the same instance.

        Raises:
            Nothing.
        """
        if not self._metrics_started:
            return
        self._metrics_started = False
        self.metrics_logger.stop()

    @api(in_process=True)
    def shutdown(self) -> None:
        """Finish what the session is doing, then tear down everything it constructed.

        First the close: from its first step nothing new takes the scope (a
        run, a recording, a home or a diagnostic asked for meanwhile, or
        still taking the scope once it has finished, raises
        ``SessionClosingError``); plugins unload, so a plugin's own work
        stops through its ``unregister``, and metrics stop; the live run is
        stopped (its ending ``aborted`` / ``shutdown``) and a live recording
        is stopped. Then it waits, without a time limit, until
        ``live_work`` lists nothing: the run's writes and post-run builds,
        the recording's file, post-processing builds running or queued, a
        support report, a still. ``discard_close_drain`` is the escape from
        waiting on video frames, and any client may call it meanwhile. A
        finished run's file writer judged stuck (``protocol_files_stalled``)
        is recovered by the close itself, its unwritten images given up on,
        counted and reported. While it waits the close logs what it waits for.

        Then the teardown. The bundle the session holds always stops: its long-lived consumer
        threads first, then the FILE lane and the worker pool. When
        ``owns_scope`` (a factory BUILT the scope), the hardware half
        follows: the LEDs drained through the io lane while its worker is
        alive, then the scope disconnected, which shuts its IO and CAMERA
        lanes before it turns the LEDs off and stops motion inline -- so a
        headless host gets the same teardown the GUI gets. The consumers
        stop BEFORE the lanes they consume (a consumer mid-iteration that
        finds its lane already shut can hang on a dispatch that never
        fires; scope_display_thread consumes camera_executor,
        protocol_thread drives io + camera + file). Running metrics stop
        first, or their ticks would outlive the executors they snapshot.

        A scope handed to the constructor is left connected with its lanes
        running: it is the caller's. Callable from any thread, a run's ``run_ended``
        handler included. One close runs: a call on another thread while it
        runs waits for it and returns; a call after it is a logged no-op; a
        call from inside the close returns at once; a call that raised
        part-way can be called again.

        Raises:
            ScopeDisconnectError: a part of an owned scope did not shut down
                cleanly; every teardown step has still run, and a second
                call completes.
        """
        with self._close_lock:
            if self._shut_down or self._close_running:
                logger.info('[Session  ] shutdown() called again -- nothing to do')
                return
            self._close_running = True
            try:
                self._finish_live_work()
                self._save_at_close()
                self._tear_down()
            finally:
                self._close_running = False
                self.activity_claim.end_closing(closed=self._shut_down)

    _CLOSE_WAIT_LOG_S = 5.0

    def _finish_live_work(self) -> None:
        """Refuse what is new, stop the live run and recording, then wait for the rest."""
        from modules.exceptions import FileWriterNotStuckError

        self.activity_claim.begin_closing()
        # A held control session first, or it would refuse the close's own
        # writes -- a stuck writer's recovery, a discard. Its run and
        # recording are stopped below with the rest.
        self.activity_claim.end_live_control()
        # Plugins first, while everything they hold still runs: an
        # unregister stops its own work and may wait for it.
        self.unload_plugins()
        self.disable_metrics()
        runner = self.sequenced_capture_runner
        if runner is not None:
            runner.force_reset(reason='session close')
        if self.manual_recording.is_recording:
            self.manual_recording.stop()
        started = time.monotonic()
        last_logged = None
        while True:
            work = self.live_work.work
            if not work:
                break
            if self.activity_claim.owner not in SCOPE_HOLDING_KINDS and self.protocol_files_stalled:
                try:
                    given_up = self.recover_file_writer()
                except FileWriterNotStuckError:
                    # It moved again between the two reads: still writing,
                    # so the close keeps waiting for it.
                    given_up = None
                if given_up is not None:
                    logger.warning(
                        f'[Session  ] shutdown: the file writer was stuck; gave up on '
                        f"{given_up} of the run's image write(s)"
                    )
            now = time.monotonic()
            if last_logged is None or now - last_logged >= self._CLOSE_WAIT_LOG_S:
                waiting_for = '; '.join(
                    item.name if item.left is None else f'{item.name} ({item.left} left)'
                    for item in work
                )
                logger.info(
                    f'[Session  ] shutdown: waiting {now - started:.1f} s for {waiting_for}'
                )
                last_logged = now
            time.sleep(0.05)
        if last_logged is not None:
            logger.info(
                f'[Session  ] shutdown: the session finished its work after '
                f'{time.monotonic() - started:.1f} s'
            )

    def _save_at_close(self) -> None:
        """Stop the saver, then save the settings the session ended with, once.

        After the session's work, so the file holds what a run's last step
        wrote; before the teardown, which a failed save does not stop.
        """
        if self._settings_saver is None:
            return
        self._settings_saver.stop()
        self._save_settings_file()

    def _tear_down(self) -> None:
        """Release everything the session constructed, the hardware last."""
        # The session owns its scheduler: a session over a caller's scope
        # still ends its own timers (a live health check outliving the
        # session would fire into torn-down state).
        self._scheduler.shutdown()
        if self.autofocus_thread is not None:
            self.autofocus_thread.stop(timeout=2.0)
        if self._owns_scope and self.io_executor.worker_alive and self.io_executor.accepts_work():
            # Drain the LEDs through the io lane BEFORE the lanes go down,
            # on the same serial-bus lane as every other LED write, so it
            # cannot race an in-flight LED task the way a bare thread did.
            # Only while the lane's worker is alive: a submission to a lane
            # with no worker is never serviced, and the wait would run out
            # its bound for nothing -- disconnect() below turns the LEDs
            # off inline either way. And only while the lane takes work: a
            # disconnect before this shut the lane without waiting for its
            # worker, which can still be alive while the lane refuses. The
            # scope's own off asks presence first, so with no LED
            # controller it writes nothing. The 2 s
            # bound keeps the calling thread from blocking on slow serial.
            logger.info('[Session  ] shutdown: leds_off through the io lane')
            try:
                from modules.sequential_io_executor import IOTask

                fut = self.io_executor.put(
                    IOTask(action=self.scope.illumination._leds_off_if_present),
                    return_future=True,
                    override=self._io_override_key,
                )
                if fut is None:
                    logger.warning('[Session  ] io lane refused the shutdown leds_off')
                else:
                    try:
                        fut.result(timeout=2.0)
                    except TimeoutError:
                        # The lane can still be draining protocol-abort
                        # cleanup, which turns the LEDs off itself -- this
                        # expiry does not mean LEDs were left on. The cached
                        # channel state answers that question directly.
                        lit = [
                            r.name or f'address {r.address}'
                            for r in self.scope.illumination.get_led_states() or ()
                            if r.state == LedState.LIT
                        ]
                        state_text = (
                            'channels still ON: ' + ', '.join(lit) if lit else 'all channels OFF'
                        )
                        logger.warning(
                            f'[Session  ] shutdown leds_off still queued on the io lane '
                            f'after 2.0s; LED state cache reports {state_text}'
                        )
                    except Exception as e:
                        logger.warning(f'[Session  ] shutdown leds_off failed: {e}')
            except Exception as e:
                logger.warning(f'[Session  ] leds_off submission failed during shutdown: {e}')
        self.executor_bundle.shutdown()
        if self._owns_scope:
            # disconnect() shuts the scope's lanes first, so nothing still
            # queued on them runs after the off, then turns the LEDs off
            # inline and bounded, stops motion, ends the motion monitor and
            # unregisters the atexit hook; it is repeatable, so a host's own
            # later disconnect is harmless.
            self.scope.disconnect()
        # After the disconnect, which can settle the moves it stops, so their
        # done-callbacks run before the session reports itself closed.
        from modules.operation import join_callback_thread

        join_callback_thread()
        # Last, so a listener hears what the teardown itself reported.
        for listener in list(self._outcome_listeners):
            self.remove_outcome_listener(listener)
        self._shut_down = True

    @api(in_process=True, stop='Operation.stop')
    def start_application_session(
        self,
        *,
        disable_homing: bool = False,
        home_fn: Callable[[HomeTarget], object] | None = None,
        turret_fn: Callable[[int], object] | None = None,
    ) -> Operation:
        """Start the standard startup motion without waiting for it; returns its ``Operation``.

        The one implementation of the startup motion for every host:
        the App's launch and its reconnect handler once open-coded the
        same ALL-axis home + turret-positioning pair, and the two
        copies drifted.

        The motion, in order:

        1. home ALL axes via ``move_home``. Firmware homes Z, T, X, Y in
           one routine; on Z-only boards it homes what it has and
           reports the missing axes.

        2. (when ``self.scope.capabilities.has_turret`` is True) move T
           to position 1; the active objective is then slot 1's
           assignment.

        3. on the simulator, place the stage at its sample plane.

        The scope is the startup's from before this returns until the motion
        ends, as it is a home's: every other request, from any client, is
        refused naming the home, and the controls lock. So a turret pick made
        meanwhile is refused rather than undone by step 2. Asked under a live
        taking, the motion is that activity's work and takes nothing. The
        claim is released before the operation is done, so a caller told
        the motion has ended finds the scope free. Its ``stop()`` is
        ``stop_motion`` while the motion drives, refused ``'home_not_stoppable'``
        while the home drives on a board that cannot stop one -- the home
        runs to its end and nothing after it moves -- and nothing once the
        motion has ended.

        A failed home is reported (``report_outcome``) and ends the motion
        without the turret move -- an absolute move against a reference the
        home did not establish; the operation settles with None. A refusal
        that is not for the hardware's state is a defect, and settles the
        operation with it.

        ``disable_homing=True`` skips every step: no startup motion on any
        axis. The turret is left where it is, like the stage axes, in no
        known slot -- positioning it without a home would be an absolute
        move against a reference the caller asked us not to establish. The
        skip is the requested behaviour, so it is logged, not signalled.

        A scope whose model has no motor board (``scope.motion_expected``
        False) has nothing to home, so it issues no startup motion either;
        nor does one whose motor board bring-up recorded as missing, which
        bring-up's own report has already named. A skip takes nothing and
        returns an operation already done.

        Args:
            disable_homing: If True, issue no startup motion at all.
            home_fn: Callable taking a ``HomeTarget`` that homes it and
                raises as ``MotionAPI.home`` does; the startup motion hands it
                ``HomeTarget.ALL``. Defaults to the motion API.
            turret_fn: Callable taking a turret slot. Defaults to
                the motion API.

        Raises:
            HardwareCommandRefusedError: ``'home_in_flight'`` or
                ``'exclusive_activity_running'``: something already holds
                the scope. Nothing was taken.
        """
        skipped = Operation()
        concurrent.futures.Future.set_result(skipped, None)
        if disable_homing:
            logger.info('startup motion skipped: homing disabled; the turret is left where it is')
            return skipped
        if not self.scope.motion_expected:
            logger.info('startup motion skipped: this scope model has no motor board')
            return skipped
        from modules.lumascope_api.bring_up import Part

        # Bring-up has already reported the missing board once, in its own
        # report; homing it would only have the home refused as "not
        # connected" and show the same absence a second time.
        if any(status.part == Part.MOTOR for status in self.scope.bring_up_record().missing):
            logger.info('startup motion skipped: the motor board did not come up at bring-up')
            return skipped

        if home_fn is None:
            home_fn = self.scope.motion.home
        if turret_fn is None:
            turret_fn = lambda slot: self.scope.motion.move_turret(slot)

        @slow_task_budget(STARTUP_MOTION_SLOW_TASK_S)
        def startup_motion() -> None:
            self._startup_motion(home_fn, turret_fn)

        # The whole motion is one home's taking, so no request lands between
        # its moves; the diagnostics worker carries it, because the moves
        # themselves wait on the io lane.
        body, release_if_unrun, stop, taking = self.scope.motion.claim_home(startup_motion)
        waiter: concurrent.futures.Future[None] = concurrent.futures.Future()
        # Marked running before the lane holds it, so only the lane settles
        # it and nothing cancels it beside the claim.
        waiter.set_running_or_notify_cancel()
        try:
            with acting(taking):
                self.executor_bundle.diagnostics_executor.submit(
                    IOTask(action=body), 'start_application_session', waiter=waiter
                )
        except BaseException:
            release_if_unrun()
            raise
        return Operation.following(waiter, stop=stop, first=release_if_unrun)

    @api(in_process=True)
    def run_application_session(
        self,
        *,
        disable_homing: bool = False,
        home_fn: Callable[[HomeTarget], object] | None = None,
        turret_fn: Callable[[int], object] | None = None,
    ) -> None:
        """Run the standard startup motion, and wait for it.

        ``start_application_session``, waited for: returns when the startup
        motion has ended and the scope is free, and raises what its
        operation would. Headless / REST callers use this exact call to apply the
        standard startup orchestration without copy-pasting from the App.

        Args:
            disable_homing: If True, issue no startup motion at all.
            home_fn: Callable taking a ``HomeTarget`` that homes it and
                raises as ``MotionAPI.home`` does; the startup motion hands it
                ``HomeTarget.ALL``. Defaults to the motion API.
            turret_fn: Callable taking a turret slot. Defaults to
                the motion API.

        Raises:
            BlockingCallRefusedError: ``'blocking_on_worker'``, called from
                an executor's worker. The motion runs on the diagnostics
                worker and waits on the io lane, so a worker waiting for it
                can be waiting on itself; ``start_application_session`` is the
                form a worker uses. Still a ``RuntimeError``.
                ``'wait_on_ui_thread'``, called on the UI thread, whose form
                is ``start_application_session`` too. Nothing was started.
            HardwareCommandRefusedError: as ``start_application_session``.
        """
        refuse_blocking_on_a_worker('run_application_session')
        refuse_waiting_on_ui_thread('run_application_session')
        self.start_application_session(
            disable_homing=disable_homing, home_fn=home_fn, turret_fn=turret_fn
        ).result()

    def _startup_motion(
        self, home_fn: Callable[[HomeTarget], object], turret_fn: Callable[[int], object]
    ) -> None:
        """The startup motion's steps, run under its taking on the diagnostics worker."""
        # Wait for the home's result and honor it. Turret positioning is
        # an absolute move against the reference frame the home was
        # supposed to establish; running it after a failed home is the
        # secondary cascade users report -- a second error on top of the
        # home's own, for motion that could never have been correct.
        try:
            home_fn(HomeTarget.ALL)
        except (HomingFailedError, HardwareCommandRefusedError) as e:
            # A refusal for the hardware's state (no board, the lid open, the
            # stage unpowered) is the scope's to report; any other is a defect.
            if (
                isinstance(e, HardwareCommandRefusedError)
                and e.reason not in HARDWARE_STATE_REASONS
            ):
                raise
            from modules.notification_center import notifications

            notifications.report_outcome(e, solicited=False, category='Motion')
            # The failure itself was logged once, at its own level, by the
            # report above; this line only records what was skipped.
            logger.info('startup turret positioning skipped: the stage reference is unknown')
            return

        try:
            if self.scope.capabilities.has_turret:
                # Every session starts at position 1, the slot the firmware's
                # home leaves the turret on. After a real home this move is a
                # physical no-op.
                START_POSITION = 1
                turret_fn(START_POSITION)

            # After the home, not instead of it: the simulator homes to the
            # floor exactly as the instrument does, and only then is placed
            # where its simulated sample is. A real scope is left alone here --
            # its operator does the focusing, and startup moving their stage
            # for them is not a convenience.
            self.scope.move_to_simulated_sample_plane()
        except MoveStoppedError as e:
            if e.reason != 'stopped':
                raise
            # STOP was pressed while the motion ran, so its later legs drive
            # nothing: the person asked for the stage to stop, and it has.
            logger.info(f'startup motion ended by STOP ({e.axis} {e.reason})')
