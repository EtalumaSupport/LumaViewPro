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
import json
import os
import pathlib
import threading
import time
import typing
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

import modules.settings_init as settings_init
from lvp_logger import logger
from modules import (
    binning,
    common_utils,
    image_mode,
    kivy_utils,
    live_work,
    path_utils,
    settings_paths,
)
from modules.activity_claim import (
    SCOPE_HOLDING_KINDS,
    ActivityClaim,
    HeldClaim,
    acting,
    the_holder_named,
)
from modules.common_utils import CustomJSONizer
from modules.exceptions import (
    HARDWARE_STATE_REASONS,
    CameraSettingUnsupportedError,
    ConfigError,
    DiagnosticRefusedError,
    FileWriterNotStuckError,
    HardwareCommandRefusedError,
    HomingFailedError,
    LiveFolderPathRefusedError,
    ObjectiveUnknownError,
    ProtocolNotLoadedError,
    Remedy,
    RemedyUnknownError,
    ScopeDisconnectError,
    ScopeModelUnknownError,
    SettingRefusedError,
    SettingsSaveRefusedError,
)
from modules.kivy_utils import UiDispatcher
from modules.live_work import LiveWork, WorkItem
from modules.lumascope_api.illumination import LedLease, LedTransition, LedTransitionCtx
from modules.manual_capture import ManualCaptureController
from modules.manual_recording import ManualRecordingController
from modules.metrics_logger import ENGINEERING_METRICS_INTERVAL_S, MetricsLogger
from modules.plugins import PLUGIN_API_LEVEL, PluginRegistry
from modules.run_outcome import RunEnding
from modules.scheduler import Scheduler, ThreadingTimerScheduler
from modules.sequential_io_executor import IOTask, refuse_blocking_on_a_worker, slow_task_budget
from modules.api_surface import FilePath, api, api_fields

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

# How long shutdown lets a finished run's images finish writing before it
# gives up on them and takes the file lane down. Budget row:
# shutdown_run_files_wait_s in PERFORMANCE_BUDGETS.md.
_SHUTDOWN_RUN_FILES_WAIT_S = 10.0

# The reports a session counts while they run, by their live-work kind.
_REPORT_NAMES = {
    live_work.SUPPORT_REPORT: 'A support report',
    live_work.LOGS_ZIP: 'A logs zip',
}

# A protocol's Layer Settings column beside the layer-settings key it is
# saved from and restored into; Acquire and Stim_Enabled are handled apart.
_LAYER_SETTINGS_KEYS = (
    ('Illumination', 'illumination_ma'),
    ('Gain', 'gain_db'),
    ('Auto_Gain', 'auto_gain'),
    ('Exposure', 'exposure_ms'),
    ('False_Color', 'false_color'),
    ('Sum', 'sum'),
)

# ProtocolRunner is referenced only in a return annotation; it is
# imported function-locally to avoid a circular import. Declare it here
# for the annotation without a runtime import.
if TYPE_CHECKING:
    import datetime

    from drivers.simulated_camera import SimulatedStall
    from modules.labware_loader import WellPlateLoader
    from modules.lumascope_api import Lumascope
    from modules.lumascope_api.bring_up import BringUpRecord
    from modules.lumascope_api.imaging import AutoGainLock
    from modules.lumascope_api.motion import MoveInFlight
    from modules.notification_center import Notification
    from modules.objectives_loader import ObjectiveLoader
    from modules.protocol import Protocol, ProtocolSizeAdvisory
    from modules.lumascope_api.protocols import StepTargets
    from modules.sequenced_capture_runner import RunHandle
    from modules.protocol_runner import ProtocolRunner
    from modules.sequential_io_executor import SequentialIOExecutor
    from modules.plugins import PluginHealth
    from modules.post_processing_api import PostProcessingAPI
    from modules.settings_init import SettingValue
    from modules.tech_support_report import SupportReportSaved


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


@api_fields('choices', 'proposed', 'turret_position')
@dataclasses.dataclass(frozen=True)
class ObjectiveQuestion:
    """The objective is unknowable; this is what to ask the user.

    Returned by ``ScopeSession.objective_question`` when no one has ever
    confirmed the objective on this install, or the turret sits on a slot
    with no assignment. The pixel size derived from the objective is
    stamped into the scale bar and every saved image's metadata, and a
    wrong scale cannot be told from a measured one afterwards -- so the
    session exposes the question instead of assuming silently, and a
    host renders it however it likes. The answer goes back through
    ``ScopeSession.confirm_objective``.

    Attributes:
        turret_position: The slot the answer binds to, or None on a
            non-turret model.
        proposed: The catalogue id to offer as the default.
        choices: The catalogue, in its shipped order.
    """

    turret_position: int | None
    proposed: str
    choices: tuple[str, ...]


@api_fields('z', 'step_idx')
@dataclasses.dataclass(frozen=True)
class SavedFocus:
    """What ``ScopeSession.save_focus`` wrote.

    Attributes:
        z: The Z saved as the layer's focus, in um.
        step_idx: The step that took it as its Z, or None when no step did.
    """

    z: float
    step_idx: int | None


@api_fields('engineering_mode', 'manual_capture', 'manual_recording', 'post_processing', 'scope')
class ScopeSession:
    """Owns the shared, GUI-independent state for one microscope session."""

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
        self.settings_lock = threading.Lock()
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
        # start_metrics; the scheduler existing does not start them.
        if scheduler is None:
            scheduler = ThreadingTimerScheduler(
                name_prefix='LVP-SessionTimer',
                on_callback_error=_scheduler_callback_error,
            )
        self._scheduler = scheduler
        # The one store for "metrics are running"; start_metrics /
        # stop_metrics are its only writers. Host-serialized (main
        # thread in the GUI): a threaded host must serialize
        # start_metrics / stop_metrics itself.
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
        # Support reports and logs zips in flight, by kind: each runs on its
        # caller's thread -- the GUI's diagnostics lane, a script's own -- so
        # the members count them, and the live-work read sees a script's
        # report as it sees the GUI's.
        self._reports_in_flight = dict.fromkeys(_REPORT_NAMES, 0)
        self._reports_lock = threading.Lock()
        from modules import coord_transformations

        # Stateless: several instances are not several stores, so the
        # session keeps its own for the plate<->stage conversions it serves.
        self.coordinate_transformer = coord_transformations.CoordinateTransformer()
        # The one store for engineering mode. Set from what the host was
        # built in; the engineering plugin turns it on when it loads, unless
        # no_engineering, the host's word, which that plugin honours, that it
        # must not. Every reader --
        # a run, a still, the metrics cadence, the GUI -- reads it here.
        self.engineering_mode = engineering_mode
        self.no_engineering = no_engineering
        # The session stops the bundle it holds at shutdown(), whoever built
        # it: a caller that hands one to the constructor hands it over. The
        # IO and CAMERA lanes the bundle holds are the scope's, and stop
        # when the scope disconnects.
        self.executor_bundle = executor_bundle
        # True only when a factory BUILT the scope, False for a scope a host
        # passed in. Decides whether shutdown() runs the hardware half (LEDs
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
            tiling_configs_path=scope.protocols.tiling_configs_path,
            has_turret=lambda: scope.capabilities.has_turret,
            settings_snapshot=self.get_settings_snapshot,
        )
        # Run-state listeners: zero-argument callables notified on every
        # run-state transition edge (claim grant/release, a run's return
        # to IDLE after its cleanup, file-drain exit). They fire on the TRANSITIONING thread,
        # possibly under engine locks, so a listener must only schedule
        # or re-read the level-derivation properties below -- never
        # acquire engine locks or trust edge context.
        self._run_state_listeners: list = []
        # The step the last go_to_step previewed, as (protocol, index): a
        # repeat of it is a re-click, whose preview would put out a channel
        # the person lit in between. Forgotten at every run transition, so
        # the first click after a run previews whatever the run left.
        self._last_step_gone_to: tuple | None = None
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
        scope.bind_settings(self.get_setting)
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
            activity_claim=self.activity_claim,
            scheduler=self._scheduler,
        )
        # Manual stills: one at a time, on the camera lane, which already
        # refuses them while a run holds the camera. No activity claim -- a
        # still must not refuse Record, a run or a reconnect.
        self.manual_capture = ManualCaptureController(
            scope=scope,
            settings_snapshot=self.get_settings_snapshot,
            engineering_mode=lambda: self.engineering_mode,
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
            on_run_idle=self._run_went_idle,
            on_protocol_files_written=self._run_protocol_complete_processors,
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
    def source_path(self) -> str:
        """The data folder this session runs on: its scope's, never a copy."""
        return self.scope.source_path

    @property
    def wellplate_loader(self) -> 'WellPlateLoader':
        """The labware catalogue: the scope's one copy, read from its data folder."""
        return self.scope.wellplate_loader

    @property
    def objective_helper(self) -> 'ObjectiveLoader':
        """The objective catalogue: the scope's one copy, read from its data folder."""
        return self.scope.objective_helper

    def _register_scope_services(self, scope) -> None:
        """Register the session's services on a scope (the one bring-up).

        The camera override key and the activity claim its homes take live
        on the scope but belong to the session's composition. Construction
        comes through here so no scope the session drives can be left
        un-serviced -- the bring-up steps are spelled out exactly once.
        """
        scope.set_camera_override_key(self._camera_override_key)
        scope.set_activity_claim(self.activity_claim)

    @api
    @contextlib.contextmanager
    def diagnostic_claim(self) -> Iterator[HeldClaim]:
        """Hold the scope for a diagnostic for the length of a ``with`` block.

        A diagnostic -- a characterization, the support report's hardware
        steps -- drives every axis, the LEDs and the camera directly. While
        it holds the claim, a run or a recording start is refused, the
        controls lock and the objective cannot change, exactly as during a
        run. The claim is released when the block ends, including on a
        raise, so no caller owns the release path.

        A run the diagnostic lent its claim to (``run_autofocus(claim=...)``)
        may still be live when the block ends -- a caller that raised
        between starting it and waiting on it. The release waits for that
        run to go idle first: its LED lease lives under this claim, and
        releasing underneath a live run hands the scope to the next taker
        mid-sweep. If the run is still live after the wait, this raises and
        the claim is kept with the run, released when the run ends.

        Yields:
            The held claim.

        Raises:
            DiagnosticRefusedError: A run, a recording, a home or
                another diagnostic holds the scope. Nothing was taken.
            RuntimeError: At the block's end, a run under this claim was
                still live after the wait; the claim is released when the
                run ends.
        """
        held = self.activity_claim.try_claim('diagnostic')
        if held is None:
            holder = self.activity_claim.holder
            kind = holder.kind if holder is not None else None
            # The holder can release between the failed take and this read;
            # the refusal still stands, it just cannot name who refused it.
            named = the_holder_named(holder)
            raise DiagnosticRefusedError(
                reason='exclusive_activity_running',
                title='Another Activity Running',
                message=f'{named} is using the microscope. Let it finish, then start the diagnostic.',
                holder=kind,
                holder_trigger=(holder.run_trigger_source if holder is not None else None),
            )
        try:
            with acting(held):
                yield held
        finally:
            if not self.sequenced_capture_runner.wait_for_run_idle(DIAGNOSTIC_EXIT_RUN_IDLE_WAIT_S):
                with self._release_at_run_idle_lock:
                    self._release_at_run_idle.append(held)
                # The run can have ended between the wait and the hand-over,
                # its idle already told: then nobody else will release it.
                if not self.sequenced_capture_runner.run_in_progress():
                    self._run_went_idle()
                raise RuntimeError(
                    'diagnostic_claim: a run under this claim is still live after '
                    f'{DIAGNOSTIC_EXIT_RUN_IDLE_WAIT_S:.0f} s; the claim is released when it ends'
                )
            held.release()

    @api
    @property
    def is_protocol_running(self) -> bool:
        """True while a protocol-class run holds the scope.

        Scans, full protocols, zstacks, and autofocus runs all hold the
        'protocol' claim, so all read True here -- and so does a run acting
        under a diagnostic's lent claim, whose holder stays the diagnostic.
        The claim releases at run-cleanup end; the post-run file drain is
        visible on run_lockout / protocol_files_draining, not here.
        """
        return self.activity_claim.run_holder is not None

    @api
    @property
    def run_in_progress(self) -> bool:
        """True while the engine's run is in any phase, its teardown included.

        Wider than is_protocol_running, which the claim answers: a run is in
        progress from its start until its cleanup has finished, the moment
        after its claim is handed back included.
        """
        return self.sequenced_capture_runner.run_in_progress()

    @api
    def held_by_other(self, run: 'RunHandle | None') -> bool:
        """Whether the scope is held by anything but *run*, a run start() returned.

        What a run control greys on while leaving its own run's Stop live;
        None asks whether anything holds the scope at all.
        """
        return self.sequenced_capture_runner.held_by_other(run)

    # ------------------------------------------------------------------
    # Run-state facts and derivations
    #
    # Each FACT has exactly one owner (the claim, the recording engine,
    # the file writer, the scope config); everything a consumer needs is
    # a synchronous DERIVATION over them. All reads are lock-free
    # attribute/queue reads, so these properties are safe from any
    # thread, including inside a transition listener.
    # ------------------------------------------------------------------

    @api
    @property
    def exclusive_activity(self) -> 'str | None':
        """The current exclusive-activity owner: None, 'protocol',
        'recording', 'diagnostic' or 'home'."""
        return self.activity_claim.owner

    @api
    @property
    def live_work(self) -> LiveWork:
        """Everything the session is still doing, each piece from its owner.

        What holds the scope (a run, a recording, a home, a diagnostic --
        and a run starting or unwinding with no holder yet or any more);
        then what finishes after it lets go: a recording's file, a run's
        video steps, a finished run's images and its post-run builds;
        post-processing builds running and queued; a support report or logs
        zip; a still being saved. Each item says how much it has left, and
        a build how far it has got. Read lock-free, owner by owner, from any
        thread; ``closing`` once the close has begun, ``closed`` once the
        session has shut down.
        """
        work: list[WorkItem] = []
        runner = self.sequenced_capture_runner
        holder = self.activity_claim.holder
        if holder is not None:
            work.append(WorkItem(holder.kind, the_holder_named(holder)))
        elif runner.run_live:
            work.append(WorkItem(live_work.PROTOCOL, 'A run, starting or ending'))
        recording = self.manual_recording
        if recording.is_busy and not recording.is_recording:
            work.append(
                WorkItem(
                    live_work.RECORDING_FINISH,
                    "A recording's file",
                    left=recording.pending_writes,
                )
            )
        if runner.video_drain_busy:
            work.append(
                WorkItem(
                    live_work.RUN_VIDEO_FINISH,
                    "A run's video",
                    left=runner.video_pending_writes,
                )
            )
        if self.protocol_files_draining:
            work.append(
                WorkItem(live_work.RUN_FILES, "A run's images", left=self.protocol_files_pending)
            )
        work.extend(
            WorkItem(live_work.POST_RUN_STEP, f"A run's {step}")
            for step in runner.post_run_steps_running
        )
        work.extend(self.post_processing.work())
        with self._reports_lock:
            reports = dict(self._reports_in_flight)
        for kind, count in reports.items():
            if count:
                work.append(WorkItem(kind, _REPORT_NAMES[kind], left=count))
        if self.manual_capture.in_flight:
            work.append(WorkItem(live_work.STILL, 'A still being saved'))
        return LiveWork(
            work=tuple(work), closing=self.activity_claim.closing, closed=self._shut_down
        )

    @contextlib.contextmanager
    def _report_in_flight(self, kind: str):
        """Count a support report or logs zip from its call to its return."""
        with self._reports_lock:
            self._reports_in_flight[kind] += 1
        try:
            yield
        finally:
            with self._reports_lock:
                self._reports_in_flight[kind] -= 1

    @api
    @property
    def close_drain_pending(self) -> bool:
        """True while a close would cut video short: a recording live, draining or finishing, or a run's video step writing.

        What a close would interrupt on the video side, in one read: a
        manual recording's own drain, or a finished run's video-step
        tail. A closing host needs both, and asking it to OR them itself
        puts the derivation somewhere headless and REST cannot reach.

        True for a LIVE recording too, since its frames are also
        outstanding -- a caller that needs "still capturing" specifically
        wants ``manual_recording.is_recording``, which is the narrower fact.
        """
        return self.manual_recording.is_busy or self.sequenced_capture_runner.video_drain_busy

    @api
    @property
    def close_drain_frames(self) -> int:
        """How many video frames a close would wait for, across both drains.

        The count beside close_drain_pending, from the same two sources: a
        manual recording's queue and a run's video-step tail. Each has its
        own recording engine, so no frame is counted twice.
        """
        return (
            self.manual_recording.pending_writes
            + self.sequenced_capture_runner.video_pending_writes
        )

    @api
    def discard_close_drain(self) -> None:
        """Drop every video frame still queued in either drain, loudly.

        A close's one escape from waiting on the drains close_drain_pending
        reads; frames already on disk stay.
        """
        self.manual_recording.discard_pending()
        self.sequenced_capture_runner.discard_video_pending()

    @api
    @property
    def protocol_files_draining(self) -> bool:
        """True from the end of a run until its last file is written.

        False while the run is live -- the run's own state answers then --
        and once its files are all on disk or given up on.
        """
        batch = self.sequenced_capture_runner.write_batch()
        return batch is not None and batch.draining

    @api
    @property
    def protocol_files_pending(self) -> int:
        """How many of a finished run's file writes are still to finish, the
        one in flight included; 0 when nothing is draining."""
        batch = self.sequenced_capture_runner.write_batch()
        return batch.pending if batch is not None and batch.draining else 0

    @api
    @property
    def protocol_files_stalled(self) -> bool:
        """True while a run's files are draining and the write in flight has
        stopped making progress -- by the same stall threshold the run
        refusal uses, so what a display says of a stuck writer and what a
        new run is refused for cannot disagree."""
        from modules.protocol_image_writer import WRITE_STALL_FATAL_S

        return self.protocol_files_draining and self.sequenced_capture_runner.write_batch().stalled(
            WRITE_STALL_FATAL_S
        )

    @api
    @property
    def protocol_files_stuck_write(self) -> str:
        """The write in flight on the file lane, named for a stall report."""
        return self.file_io_executor.describe_running_task()

    @api
    @property
    def run_lockout(self) -> bool:
        """True while a run, a diagnostic, a home, or a run's post-run file
        drain owns the scope.

        The drain term encodes a deliberate asymmetry: a finished
        protocol frees its claim while its files drain, but the control
        surface stays locked until the queue empties.
        """
        return self.run_lockout_named is not None

    @api
    @property
    def run_lockout_named(self) -> str | None:
        """What locks the controls, as a sentence a person reads; None while nothing does.

        The run by its kind, a diagnostic as one (a characterization and the
        support report share its claim), a home, or a finished run's files
        still writing. The one rule ``run_lockout`` reads.
        """
        holder = self.activity_claim.holder
        if holder is not None and holder.kind in SCOPE_HOLDING_KINDS:
            return f'{the_holder_named(holder)} is in progress.'
        if self.protocol_files_draining:
            return "A protocol's files are still being written."
        return None

    @api
    @property
    def recording_active(self) -> bool:
        """True while a manual recording is LIVE; its drain reads False, with
        its claim still held and still refusing new runs.

        Live implies it holds the claim: the engine takes the claim before
        it goes live and releases it only after selection has closed.
        """
        return self.manual_recording.is_recording

    @api
    @property
    def controls_locked(self) -> bool:
        """True while the full control surface locks: any run lockout,
        or a live manual recording (a draining recording frees the
        controls while its claim still refuses new runs)."""
        return self.run_lockout or self.recording_active

    @api
    @property
    def motion_enabled(self) -> bool:
        """True when user stage motion is allowed: the scope actually has
        an XY stage and no run lockout holds. Evaluated at read -- there
        is no cached copy to mis-restore.

        The stage fact is read from the driver rather than from the
        configured scope model. The model is user-editable while the
        app runs, so a copy of it kept here goes stale the moment
        someone selects a different scope, and then reports stage
        motion available on a scope that has no stage."""
        return self.scope.capabilities.has_xy_stage and not self.run_lockout

    @api
    def add_run_state_listener(self, listener: Callable[[], None]) -> None:
        """Register a run-state transition listener and level-sync it.

        The immediate call is the level republish: transitions are
        edges, and a listener registered after a grant would otherwise
        never see it.
        """
        self._run_state_listeners.append(listener)
        listener()

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
        for listener in list(self._run_state_listeners):
            try:
                listener()
            except Exception as ex:
                notifications.report_outcome(ex, solicited=False, category='Run State')

    @api
    def add_outcome_listener(self, listener: Callable[['Notification'], None]) -> None:
        """Hear every outcome the scope reports from now on.

        ``listener(notification)`` is called once per delivery, on the thread
        that reported it, so it must return promptly and must not wait on the
        scope. Each ``Notification`` carries its ``kind``, words, ``reason``,
        ``remedy``, ``outcome_id`` and ``shown``: an outcome muted when it
        happened (an unattended run, the dedup window, shutdown) arrives with
        ``shown`` False, and arrives again with the same ``outcome_id`` and
        ``shown`` True if it is later shown.

        Bring-up has happened by the time a session exists; pass the listener
        to ``create(outcome_listener=...)`` to hear it. The outcomes heard are
        every session's in this process, as the notification centre is one
        per process. ``shutdown()`` removes the listener.
        """
        from modules.notification_center import Severity, notifications

        notifications.add_listener(listener, min_severity=Severity.DEBUG)
        self._outcome_listeners.append(listener)

    @api
    def remove_outcome_listener(self, listener: Callable[['Notification'], None]) -> None:
        """Stop ``listener`` hearing outcomes; a listener never added is a no-op."""
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
        (a script, a REST server) never sets it. None (the default) calls
        each callback directly on the thread that dispatches it, and a
        callback's raise is reported rather than raised.

        ``dispatcher.thread`` is the thread ``schedule`` delivers on: a
        run's ``wait()`` on that thread is refused, since what it waits for
        is delivered there.
        """
        kivy_utils._set_ui_dispatcher(dispatcher)

    @api(in_process=True)
    @classmethod
    def create(
        cls,
        settings: dict,
        source_path: str | None = None,
        scope: 'Lumascope | None' = None,
        *,
        simulate: bool = False,
        warn_pre_release: bool = True,
        engineering_mode: bool = False,
        no_engineering: bool = False,
        sim_camera_stall: 'SimulatedStall | None' = None,
        sim_file_stall: 'SimulatedStall | None' = None,
        outcome_listener: Callable[['Notification'], None] | None = None,
    ) -> 'ScopeSession':
        """Create a session, constructing defaults for any missing components.

        This is the one composition path: the GUI, REST and scripts all
        build their session here, passing what only a host knows as the
        keyword arguments below. Pass ``scope`` to reuse one you built;
        omit it and the factory builds it.

        ``source_path`` is the data folder the factory builds the scope on;
        None (default) is the installation's own folder. The session's data
        folder and catalogues are always its scope's, so a folder is refused
        beside ``scope``: the scope was given its folder when it was built.

        The executor bundle is always built, around the scope's own IO and
        CAMERA lanes, via executor_registry.create_default, so every caller
        gets the topology the GUI runs: FILE + WORKER_POOL executors plus
        protocol_thread (started) and scope_display_thread (constructed,
        not started).

        A scope the factory builds is brought up before this returns
        (``configure_scope``, then the camera start gate released) and is
        torn down by ``shutdown()``; a scope passed in is the caller's
        bring-up and the caller's teardown.

        Keyword arguments, each with the one consumer it feeds:
            simulate: build a simulated scope (the model from
                ``settings['microscope']``, the motor board's tier from
                ``settings['simulator_tier']``). Ignored when ``scope`` is
                passed.
            warn_pre_release: whether this construction fires the
                pre-release FutureWarning -- the factory's own call and the
                scope constructor's. A host that ships with the API passes
                False; a separately shipped caller leaves the default.
            engineering_mode: the session's engineering mode to begin with;
                a plugin may turn it on when the host loads plugins.
            no_engineering: the host's word, which the engineering plugin
                honours, that it must not turn engineering mode on.
            sim_camera_stall: a stall for the simulated camera's stream, so a
                simulated scope shows a stream that stops delivering; refused
                beside ``scope`` and by the scope itself unless it is
                simulated with the simulated camera.
            sim_file_stall: a stall for the session's file lane, so a
                simulated scope shows a save drive that stops answering:
                ``after_s`` seconds after bring-up the lane's worker is held
                for ``for_s`` seconds. Refused beside ``scope`` and unless
                ``simulate``.
            outcome_listener: heard from before the scope is built, so it
                hears what bring-up reports; the session's
                ``add_outcome_listener`` says what it receives. A factory
                that raises takes it back; otherwise the session's
                ``shutdown()`` does.
        """
        from modules.lumascope_api._lumascope import _fire_pre_release_warning
        from modules.path_utils import get_source_root

        if scope is not None and source_path is not None:
            raise ValueError(
                'ScopeSession.create: source_path is refused beside a scope -- the '
                'session reads its data folder and catalogues from the scope, so pass '
                'the folder to Lumascope(source_path=...) instead'
            )
        if scope is not None and sim_camera_stall is not None:
            raise ValueError(
                'ScopeSession.create: sim_camera_stall is refused beside a scope -- the '
                'stall is set on the camera when the scope is built, so pass it to '
                'Lumascope(sim_camera_stall=...) instead'
            )
        if sim_file_stall is not None and scope is not None:
            raise ValueError(
                'ScopeSession.create: sim_file_stall is refused beside a scope -- it '
                'simulates the drive a scope this factory builds saves to'
            )
        if sim_file_stall is not None and not simulate:
            raise ValueError(
                'ScopeSession.create: sim_file_stall needs a simulated scope -- a real '
                "scope's file lane writes to a real drive"
            )
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
            built_scope = False
            if scope is None:
                import modules.lumascope_api as lumascope_api

                scope = lumascope_api.Lumascope(
                    simulate=simulate,
                    warn_pre_release=warn_pre_release,
                    configured_model=settings.get('microscope'),
                    sim_tier=cls._simulator_tier(settings) if simulate else 'fast',
                    fx2_debug_wire=settings['fx2_debug_wire_enabled'],
                    source_path=get_source_root(source_path),
                    sim_camera_stall=sim_camera_stall,
                )
                # The bring-up -- configure from settings, then release the
                # camera start gate -- happens below, once the session exists,
                # for a scope THIS factory built. A scope passed in by a caller
                # is that caller's bring-up responsibility: they call
                # configure_scope() themselves, which releases the start gate.
                built_scope = True

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
            # already know whether the scope is its own.
            try:
                session = cls(
                    settings=settings,
                    scope=scope,
                    executor_bundle=executor_bundle,
                    autofocus_runner=autofocus_runner,
                    autofocus_thread=autofocus_thread,
                    owns_scope=built_scope,
                    engineering_mode=engineering_mode,
                    no_engineering=no_engineering,
                )
            except BaseException:
                # No session exists to tear down -- a scope another session holds
                # refuses a second claim -- so stop what this factory started.
                autofocus_thread.stop(timeout=2.0)
                executor_bundle.shutdown()
                if built_scope:
                    cls._report_teardown_failure(scope.disconnect)
                raise
            if outcome_listener is not None:
                session._outcome_listeners.append(outcome_listener)
            if built_scope:
                cls._bring_up(session, replaced)
            if sim_file_stall is not None:
                session._hold_the_file_lane(sim_file_stall)
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

        def _hold(_dt: float = 0) -> None:
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
        from modules.lumascope_api._constants import SIMULATOR_TIERS

        if 'simulator_tier' not in settings:
            raise ConfigError(
                "settings have no 'simulator_tier'; a prepared settings dict carries it "
                'from the shipped template'
            )
        tier = settings['simulator_tier']
        if tier not in SIMULATOR_TIERS:
            raise ConfigError(f'simulator_tier {tier!r} is not one of {SIMULATOR_TIERS}')
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
    def load_user_settings(source_path: str) -> dict:
        """The user's configuration, as the GUI would configure a scope from it.

        For a host with no GUI -- a script, a server -- to pass to
        ``create``: ``create(ScopeSession.load_user_settings(root),
        simulate=...)``. Reading is its own call rather than a default of
        ``create`` so a caller that meant to pass settings and forgot is
        refused for the missing argument instead of quietly configured
        from whatever is on disk.

        Raises:
            ConfigError: ``source_path`` holds neither settings file, or the
                user's ``current.json`` is unusable.
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
            raise ConfigError(
                f'no data/settings.json under {source_path!r}: not an LVP '
                'installation root; pass source_path or run from one'
            ) from e
        return settings

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
    def _bring_up(cls, session: 'ScopeSession', replaced: list) -> None:
        """Configure the scope a factory built; ``initialize`` releases the
        camera start gate last, after the capture pixel format. A raise
        anywhere in here leaves the caller with no session object to tear
        down, so this tears down what the factory started before it lets
        the raise out. What the bring-up replaces is added to ``replaced``,
        the load's list, for the factory to tell once."""
        try:
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

    @api
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

    @api
    def apply_remedy(self, remedy: Remedy) -> int:
        """Take the action a refusal named as its remedy, and return its answer.

        The one place a remedy's name becomes an action: the GUI's offer and
        a REST caller answer a refusal through this, never by resolving the
        name themselves. A name arrives as data -- from a refusal another
        layer built, or from a caller sending it back -- so only the members
        listed here are reachable through it. The member still decides
        whether the remedy applies now, and refuses if not.

        Returns:
            The remedy's own answer. The one remedy offered,
            ``recover_file_writer``, answers how many of the run's images
            it gave up on.

        Raises:
            RemedyUnknownError: the remedy names a member not offered as one.
        """
        remedies: dict[str, Callable[[], int]] = {
            'recover_file_writer': self.recover_file_writer,
        }
        action = remedies.get(remedy.member)
        if action is None:
            raise RemedyUnknownError(remedy.member, remedies)
        return action()

    @api
    def get_layer_configs(self, specific_layers: list | None = None) -> dict:
        import modules.config_helpers as config_helpers

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

    @api
    def saved_focus(self, layer: str) -> float:
        """The Z saved as ``layer``'s focus.

        Raises:
            FocusNotSavedError: no focus was ever saved for ``layer``; a step
                built for it takes the current Z instead.
        """
        from modules.exceptions import FocusNotSavedError

        focus = self.settings[layer]['focus']
        if focus is None:
            raise FocusNotSavedError(layer)
        return focus

    @api
    def get_stim_configs(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_stim_configs(self.settings)

    @api
    def get_enabled_stim_configs(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_enabled_stim_configs(self.settings)

    @api
    def get_auto_gain_settings(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_auto_gain_settings(self.settings)

    @api
    def get_sequenced_capture_config(
        self,
        *,
        tiling: str = '1x1',
        use_zstacking: bool = False,
    ) -> dict:
        """The sequenced capture config for this session's settings.

        The entry point a caller with no GUI uses to assemble the config a
        run takes. Tiling and z-stacking are arguments rather than stored
        settings: neither survives a restart, so there is nothing for a
        session to read them from and a caller states what it wants.

        The GUI builds the same config through the same builder, supplying
        these two from its widgets.

        Its ``current_z`` is Z's position, or None while Z has none, so a
        config never carries a number nobody read; ``new_protocol`` refuses
        that before a step would be saved at it. A scope with no Z motor
        takes the plate read's answer.
        """
        import modules.config_helpers as config_helpers

        z = self.scope.motion.axis_positions().get('Z')
        return config_helpers.get_sequenced_capture_config_from_settings(
            self.capture_settings_snapshot(),
            objective_helper=self.objective_helper,
            wellplate_loader=self.wellplate_loader,
            current_z=z.position if z is not None else self.get_current_plate_position()['z'],
            tiling=tiling,
            use_zstacking=use_zstacking,
        )

    @api
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

        Raises:
            ProtocolNotLoadedError, ProtocolFormatError,
            ProtocolRunRefusedError: As
                ``ProtocolsAPI.load_protocol`` raises them.
            ConfigError, HardwareCommandRefusedError: As ``select_labware``
                raises them; the scope stays on the plate it had.
        """
        protocol = self.scope.protocols.load_protocol(file_path=file_path)
        self.scope.protocols.set_labware(protocol, protocol.labware())
        self.select_labware(protocol.labware())
        return protocol

    def _layers_on_scope(self) -> set[str]:
        """The layers this scope has, by key name: none when unresolved."""
        return {record.key_name for record in self.scope.layer_identity.layers}

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
                    f'({self.scope.layer_identity.model}) has no {name} layer; '
                    'it is set to acquire nothing.'
                )
                self.settings[name]['acquire'] = None

    @api
    def apply_layer_settings(self, protocol: 'Protocol') -> None:
        """Put a protocol's Layer Settings into this session's layer controls.

        The other half of ``save_protocol``, and what the GUI's Load does
        once ``load_protocol`` has accepted the plate. Every layer stops
        acquiring and stimulating; each layer the protocol names then takes
        its acquire mode and every value its row holds. A blank value leaves
        that layer's control as it was. A layer this release does not know
        (not in ``common_utils.get_layers()``), or this scope does not have,
        is logged and dropped.

        Raises:
            ProtocolFormatError: A protocol built in memory carries a
                Layer Settings cell that is not of its column's type; no
                layer is changed. A loaded protocol was refused at load.
        """
        rows = protocol.layer_settings()
        layers = common_utils.get_layers()
        present = self._layers_on_scope()
        with self.settings_lock:
            for name in layers:
                self.settings[name]['acquire'] = None
                stim = self.settings[name].get('stim_config')
                if stim is not None:
                    stim['enabled'] = False
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
                        f'scope ({self.scope.layer_identity.model}) does not have; '
                        'that layer is dropped on load.'
                    )
                    continue
                layer = self.settings[name]
                layer['acquire'] = row['Acquire']
                for column, key in _LAYER_SETTINGS_KEYS:
                    if row[column] is not None:
                        layer[key] = row[column]
                stim = layer.get('stim_config')
                if row['Stim_Enabled'] is not None and isinstance(stim, dict):
                    stim['enabled'] = row['Stim_Enabled']

    @api
    def save_protocol(self, protocol: 'Protocol', file_path: FilePath) -> pathlib.Path:
        """Write a protocol to a file, with this session's Layer Settings.

        ``.tsv`` is added to a name that does not end in it. The block holds
        each layer set to acquire an image or a video, with the values its
        controls hold now; ``apply_layer_settings`` puts them back. What the
        GUI opens at its next start-up is not changed: a script's scratch
        save is not the person's protocol.

        Returns:
            The path written.

        Raises:
            ProtocolNotSavedError: The file could not be written; a file
                already at the path is unchanged.
        """
        path = pathlib.Path(file_path)
        if path.suffix.lower() != '.tsv':
            path = path.with_name(path.name + '.tsv')
        layer_settings = {}
        with self.settings_lock:
            for name in common_utils.get_layers():
                layer = self.settings[name]
                if layer.get('acquire') not in ('image', 'video'):
                    continue
                stim = layer.get('stim_config')
                layer_settings[name] = {
                    'Layer': name,
                    'Acquire': layer['acquire'],
                    **{column: layer.get(key, '') for column, key in _LAYER_SETTINGS_KEYS},
                    'Stim_Enabled': (
                        stim['enabled'] if isinstance(stim, dict) and 'enabled' in stim else ''
                    ),
                }
        protocol.to_file(file_path=path, layer_settings=layer_settings)
        return path

    def _refuse_layer_not_on_scope(self, layer: str, *, then: str) -> None:
        """Refuse ``layer`` unless this scope has it.

        Raises:
            ConfigError: this scope has no ``layer``, or its layers could
                not be resolved.
        """
        if layer in self._layers_on_scope():
            return
        identity = self.scope.layer_identity
        if identity.layers:
            raise ConfigError(
                f'this scope ({identity.model}) has no {layer} layer; '
                f'its layers are {sorted(self._layers_on_scope())}'
            )
        raise ConfigError(
            f"this scope's layers could not be resolved (model {identity.model}), "
            f'so {layer} cannot {then}'
        )

    @api
    def set_layer_acquire(self, layer: str, mode: 'str | None') -> None:
        """Set what a layer captures: ``'image'``, ``'video'``, or None (nothing).

        The layers set to acquire are the ones ``new_protocol`` and
        ``add_step`` build steps for and a composite merges. A layer set to
        acquire stops stimulating: one layer does not capture and
        stimulate at once.

        Raises:
            ConfigError: ``layer`` is not one of this release's layers,
                ``mode`` is not ``'image'``, ``'video'`` or None, or
                ``mode`` is not None and this scope does not have ``layer``;
                nothing is changed. Setting a layer to acquire nothing is
                always admitted.
        """
        if layer not in common_utils.get_layers():
            raise ConfigError(
                f'{layer!r} is not a layer; the layers are {common_utils.get_layers()}'
            )
        if mode not in ('image', 'video', None):
            raise ConfigError(f"acquire mode {mode!r} is not 'image', 'video' or None")
        if mode is not None:
            self._refuse_layer_not_on_scope(layer, then='be set to acquire')
        with self.settings_lock:
            self._write_layer_acquire(layer, mode)

    def _write_layer_acquire(self, layer: str, mode: 'str | None') -> None:
        """Under ``settings_lock``: ``layer`` acquires ``mode``, and stops stimulating if it does."""
        self.settings[layer]['acquire'] = mode
        stim = self.settings[layer].get('stim_config')
        if mode is not None and stim is not None:
            stim['enabled'] = False

    @api
    def set_layer_auto_gain(self, layer: str, enabled: bool) -> 'AutoGainLock | None':
        """Turn a layer's auto-gain on or off, as the GUI's Auto Gain/Exp box does.

        Turning it on stores the preference only: the camera arms when the
        layer is next applied (``apply_layer_camera``). Turning it off locks a standing arm
        (``scope.imaging.lock_auto_gain``) and stores what the camera reached
        as the layer's manual setting: the lock's ``gain_db`` and its
        ``stored_exposure_ms``, rounded to 0.1 dB and 0.01 ms, the resolution
        the stored settings carry. Each is stored only when the camera
        reported it; with no arm standing, or a lock that found nothing
        usable, they are left as they were. Turning it off waits on the
        camera lane.

        Returns:
            The lock when turning off (its ``state`` is None when no arm
            stood), so a caller can read the state the camera reached;
            None when turning on.

        Raises:
            ConfigError: ``layer`` is not one of this release's layers,
                ``enabled`` is not a bool, or ``enabled`` is True and this
                scope does not have ``layer``; nothing is changed. Turning
                auto-gain off is always admitted.
            The lock's refusal, when a run, a diagnostic or a home holds the scope;
                nothing is stored.
        """
        if layer not in common_utils.get_layers():
            raise ConfigError(
                f'{layer!r} is not a layer; the layers are {common_utils.get_layers()}'
            )
        if not isinstance(enabled, bool):
            raise ConfigError(f'auto-gain enabled must be True or False, got {enabled!r}')
        lock = None
        if enabled:
            self._refuse_layer_not_on_scope(layer, then='run auto-gain')
        else:
            lock = self.scope.imaging.lock_auto_gain()
        with self.settings_lock:
            stored = self.settings[layer]
            if lock is not None and lock.state is not None:
                if common_utils.is_valid_gain_db(lock.gain_db):
                    stored['gain_db'] = round(lock.gain_db, 1)
                if common_utils.is_valid_exposure_ms(lock.exposure_ms):
                    stored['exposure_ms'] = round(lock.stored_exposure_ms, 2)
            stored['auto_gain'] = enabled
        return lock

    @api
    def apply_layer_camera(self, layer: str) -> dict | None:
        """Put ``layer``'s stored exposure, gain and auto-gain on the camera, and wait.

        The one way a layer's settings reach the camera outside a run:
        bring-up applies BF, ``go_to_step`` applies the step's layer, and
        the GUI applies the layer whose control changed. A stored auto-gain
        arms the live auto loop, capped to the layer's channel class and the
        installation's override, as the GUI's Auto Gain/Exp box does; a
        camera without hardware auto-gain applies the layer manually
        (``scope.imaging.apply_layer_camera_settings``).

        Returns:
            What ``scope.imaging.apply_layer_camera_settings`` returns: the
            gain and exposure now in effect.

        Raises:
            ConfigError: this scope has no ``layer``; nothing is applied.
            HardwareCommandRefusedError: a run, a diagnostic or a home holds the
                scope (an autofocus is a run), or ``'not_connected'``, naming
                the camera, with none connected; nothing is applied.
            CameraSettingRejected: the camera refused a setting; the others
                were applied.
        """
        import modules.config_helpers as config_helpers

        self._refuse_layer_not_on_scope(layer, then='be applied to the camera')
        with self.settings_lock:
            stored = self.settings[layer]
            gain_db = stored['gain_db']
            exposure_ms = stored['exposure_ms']
            auto_gain = stored['auto_gain']
            auto_gain_settings = config_helpers.get_auto_gain_settings(self.settings)
            overrides = copy.deepcopy(self.settings.get('ag_ae_max_exposure_ms', {}))
        auto_gain_settings['max_exposure_ms'] = config_helpers.get_ag_ae_max_exposure_ms(
            layer, overrides
        )
        # The floor rides beside the ceiling so an auto-gain lock can say
        # whether exposure bottomed out of the usable range (AT_MINIMUM).
        auto_gain_settings['min_exposure_ms'] = config_helpers.get_ag_ae_min_exposure_ms(layer)
        return self.scope.imaging.apply_layer_camera_settings(
            layer=layer,
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
        period: 'datetime.timedelta | None' = None,
        duration: 'datetime.timedelta | None' = None,
    ) -> 'Protocol':
        """Build a protocol from this session's settings, as the GUI's New does.

        One step per acquiring layer at every well of the session's labware,
        at the current objective, tiled and z-stacked as asked. Its period and
        duration are the ones given; one left out (None) is the stored
        default's. One scan is ``timedelta(0)``. The
        protocols API refuses the build when no layer acquires, where an
        empty protocol would otherwise come back for a click that meant
        steps; a labware with no wells still gives an empty protocol, which
        ``add_step`` fills at the current position.

        Raises:
            ProtocolRunRefusedError: reason ``no_acquiring_layer``, logged
                and notified once.
            ConfigError: the config cannot be assembled (the objective in
                the light path is unknown; a z-stack with no extent).
            ProtocolScheduleRefusedError: ``period`` or ``duration`` is one no
                protocol can run.
            AxisStateUnknownError: an acquiring layer has no saved focus, so
                its steps would save the current Z, and Z does not know its
                position. Reported once.
        """
        from modules.protocol import Protocol

        config = self.get_sequenced_capture_config(tiling=tiling, use_zstacking=use_zstacking)
        if period is not None:
            config['period'] = period
        if duration is not None:
            config['duration'] = duration
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
        """
        import modules.config_helpers as config_helpers

        return self.scope.protocols.create_protocol(
            empty_config=config_helpers.get_empty_protocol_config_from_settings(
                self.get_settings_snapshot(), self.wellplate_loader
            )
        )

    @api
    def add_step(
        self,
        protocol: 'Protocol',
        *,
        before_step: int | None = None,
        after_step: int | None = None,
    ) -> list[str]:
        """Add a step to ``protocol`` from this session's settings and live position.

        The entry point a caller with no GUI uses to do what Add Step
        does: one step per layer whose ``acquire`` is set, at the live
        stage position on the protocol's plate, with the current objective,
        in the settings' channel order. The protocols API performs the add
        and refuses when nothing would be added; this composes its inputs
        from the session the same way the GUI's handler does.

        Returns the inserted step names, in protocol order.
        """
        # None when unknown: the protocols API refuses that by name, notified.
        objective_id = self.scope.runtime_state.get_current_objective_id()
        return self.scope.protocols.add_step(
            protocol,
            layer_configs=self.get_layer_configs(),
            stim_configs=self.get_stim_configs(),
            plate_position=self.plate_position_on(protocol.labware()),
            objective_id=objective_id,
            channel_order=self.settings.get('step_channel_order', None),
            before_step=before_step,
            after_step=after_step,
        )

    @api
    def update_step(
        self,
        protocol: 'Protocol',
        step_idx: int,
        *,
        layer: str,
        label: str | None = None,
    ) -> str:
        """Rewrite a step of ``protocol`` from this session's settings and live position.

        The entry point a caller with no GUI uses to do what Update Step
        does: step ``step_idx`` takes ``layer``'s settings, the live stage
        position on the protocol's plate and the current objective. The
        protocols API performs the update and refuses it when the position
        or the objective is unknown; this composes its inputs from the
        session the same way ``add_step`` does.

        Returns the step's name after the update.
        """
        # None when unknown: the protocols API refuses that by name, notified.
        objective_id = self.scope.runtime_state.get_current_objective_id()
        return self.scope.protocols.update_step(
            protocol,
            step_idx,
            layer=layer,
            layer_configs=self.get_layer_configs(),
            stim_configs=self.get_stim_configs(),
            plate_position=self.plate_position_on(protocol.labware()),
            objective_id=objective_id,
            label=label,
        )

    @api
    def save_focus(
        self, protocol: 'Protocol', layer: str, *, step_idx: int | None = None
    ) -> SavedFocus:
        """Save the live Z as ``layer``'s focus, and as step ``step_idx``'s Z.

        The layer's focus is what every new step of the layer is born at.
        The step takes the Z only when it is a step of ``layer``; a step of
        another channel is left alone. No other step is written: every step
        of a layer is born at the same focus, so a step matching the old
        focus says nothing about whether its user wants the new one.
        ``apply_focus_to_layer_steps`` writes them all.

        Raises:
            ProtocolRunRefusedError: ``positions_unreachable`` -- this scope
                has no Z axis. Nothing is written.
            AxisStateUnknownError: Z lost its reference. Nothing is written.
            ProtocolError: ``step_idx`` is not a step of ``protocol``.
                Nothing is written.
            ConfigError: this scope has no ``layer``. Nothing is written.
        """
        self._refuse_layer_not_on_scope(layer, then='take a focus')
        step = None if step_idx is None else protocol.step(idx=step_idx)
        z = self.scope.protocols.focus_z(then='save the focus')
        self._store_layer_focus(layer, z)
        if step is None or step['Color'] != layer:
            logger.info(f'[Session  ] Focus saved: {layer} Z={z}, no step written')
            return SavedFocus(z=z, step_idx=None)
        self.scope.protocols.set_step_z(protocol, step_idx, z)
        logger.info(f'[Session  ] Focus saved: {layer} Z={z}, and as the Z of step {step_idx}')
        return SavedFocus(z=z, step_idx=step_idx)

    @api
    def save_layer_focus(self, layer: str, z_um: float) -> None:
        """Store ``z_um`` as ``layer``'s focus, the Z every new step of the layer is born at.

        The door for a caller that holds a number rather than a stage: the
        Autofocus button saving the Z its run chose, a script saving one it
        measured. It says only that the number is a Z this scope's focus can
        reach; it guarantees nothing about where the number came from.
        ``save_focus``, which saves the live Z, writes through the same path.

        Raises:
            ConfigError: this scope has no ``layer``. Nothing is written.
            ProtocolRunRefusedError: ``positions_unreachable`` -- this scope
                has no Z axis. Nothing is written.
            PositionOutOfRangeError: ``z_um`` is NaN or infinite, or lies
                outside Z's travel. Nothing is written.
        """
        self._refuse_layer_not_on_scope(layer, then='take a focus')
        z = self._store_layer_focus(layer, z_um)
        logger.info(f'[Session  ] Focus saved: {layer} Z={z}')

    def _store_layer_focus(self, layer: str, z_um: float) -> float:
        """The one write of a layer's focus, refused unless Z can reach it; returns the Z stored.

        ``save_all_bookmarks`` alone writes a focus beside it: it stores the
        bookmark and every layer's focus under one hold of the lock, and its
        Z is the live one, which ``focus_z`` has already checked.
        """
        z = self.scope.protocols.check_focus_z(z_um, then='save the focus')
        with self.settings_lock:
            self._store_setting(f'{layer}.focus', z)
        return z

    @api
    def save_bookmark(self, axes: 'Iterable[str]') -> dict:
        """Save where the stage is on ``axes`` as the bookmark, as the bookmark buttons do.

        X and Y are stored in plate millimetres on the selected plate, Z in
        micrometres: the frame the go-to-bookmark moves read them in. The one
        writer of ``bookmark``.

        Returns:
            The values stored, by lower-case axis name.

        Raises:
            AxisStateUnknownError: an axis's position is not known, so the
                number it reports is not one to save. Nothing is written.
            HardwareCommandRefusedError: ``'not_connected'`` -- the motor
                controller is not connected. Nothing is written.
        """
        axes = tuple(axes)
        self.scope.motion.refuse_unknown_positions(axes, recording=True, then='save the bookmark')
        position = self.get_current_plate_position()
        saved = {axis.lower(): position[axis.lower()] for axis in axes}
        with self.settings_lock:
            for key, value in saved.items():
                self._store_setting(f'bookmark.{key}', value)
        logger.info(f'[Session  ] Bookmark saved: {saved}')
        return saved

    @api
    def save_all_bookmarks(self) -> float:
        """Save the live Z as the Z bookmark and as the focus of every layer this scope has.

        What Set All Bookmarks does. Returns the Z.

        Raises:
            ProtocolRunRefusedError: ``positions_unreachable`` -- this scope
                has no Z axis. Nothing is written.
            AxisStateUnknownError: Z lost its reference. Nothing is written.
        """
        z = self.scope.protocols.focus_z(then='save the bookmarks')
        layers = sorted(self._layers_on_scope())
        with self.settings_lock:
            self._store_setting('bookmark.z', z)
            for layer in layers:
                self._store_setting(f'{layer}.focus', z)
        logger.info(f'[Session  ] Bookmarks saved: Z={z}, and as the focus of {layers}')
        return z

    @api
    def apply_focus_to_layer_steps(self, protocol: 'Protocol', layer: str) -> int:
        """Save the live Z as ``layer``'s focus and as the Z of its every step.

        Returns how many steps took it.

        Raises:
            ProtocolRunRefusedError: ``positions_unreachable`` -- this scope
                has no Z axis. Nothing is written.
            AxisStateUnknownError: Z lost its reference. Nothing is written.
            ConfigError: this scope has no ``layer``. Nothing is written.
        """
        self._refuse_layer_not_on_scope(layer, then='take a focus')
        z = self.scope.protocols.focus_z(then='apply the focus')
        self._store_layer_focus(layer, z)
        updated = self.scope.protocols.apply_focus_to_layer_steps(protocol, layer, z)
        logger.info(f'[Session  ] Focus applied: {layer} Z={z} to {updated} step(s)')
        return updated

    @api
    def delete_step(self, protocol: 'Protocol', step_idx: int) -> None:
        """Remove step ``step_idx`` from ``protocol``, as the Delete button does.

        Raises:
            ProtocolError: ``step_idx`` is not a step of ``protocol``.
                Nothing is removed.
        """
        self.scope.protocols.delete_step(protocol, step_idx)

    @api
    def rename_step(self, protocol: 'Protocol', step_idx: int, name: str) -> str:
        """Give step ``step_idx`` of ``protocol`` the label ``name``; returns its new name.

        Raises:
            ProtocolError: ``step_idx`` is not a step of ``protocol``, or
                ``name`` has no letter, digit, dash or underscore. The step
                keeps its name.
        """
        return self.scope.protocols.rename_step(protocol, step_idx, name)

    @api
    def set_protocol_labware(self, protocol: 'Protocol', plate_key: str) -> str:
        """Put ``protocol`` on the plate ``plate_key``; returns the key it took.

        The protocol's plate only: the scope's is ``select_labware``'s. On a
        scope with no XY stage the protocol takes Center Plate.

        Raises:
            ConfigError: ``plate_key`` is not a plate the catalogue has. The
                protocol keeps its plate.
        """
        return self.scope.protocols.set_labware(protocol, plate_key)

    @api
    def apply_tiling(self, protocol: 'Protocol', tiling: str) -> None:
        """Expand every step of ``protocol`` into the tile grid ``tiling``, as Apply does.

        The tiles are spaced for the configured frame, binning and tile
        overlap, laid out on the protocol's own plate.

        Raises:
            ProtocolRunRefusedError: ``tiling`` is not a grid this
                installation offers, the protocol is already tiled, a step's
                objective is not in the catalogue, the scope has no X/Y
                motor, or a tile falls outside the stage's travel. Nothing
                changes.
            ConfigError: the protocol's plate is not in the catalogue.
                Nothing changes.
        """
        import modules.config_helpers as config_helpers

        self.scope.protocols.apply_tiling(
            protocol,
            tiling,
            frame_dimensions=config_helpers.get_frame_dimensions_from_settings(self.settings),
            binning_size=self.get_binning_size(),
            overlap_percent=self.settings['tiling_overlap_percent'],
        )

    @api
    def apply_zstacking(
        self,
        protocol: 'Protocol',
        *,
        range_um: float,
        step_size_um: float,
        z_reference: str,
    ) -> None:
        """Expand every step of ``protocol`` not already in a stack into a z-stack.

        ``z_reference`` says where each step's Z sits in its stack:
        ``'top'``, ``'center'`` or ``'bottom'``.

        Raises:
            ProtocolRunRefusedError: ``range_um`` or ``step_size_um`` is not
                greater than zero, the scope has no Z motor, or a slice falls
                outside the Z travel. Nothing changes.
            ConfigError: ``z_reference`` is not one of the three. Nothing
                changes.
        """
        self.scope.protocols.apply_zstacking(
            protocol, range_um=range_um, step_size_um=step_size_um, z_reference=z_reference
        )

    @api
    def go_to_step(self, protocol: 'Protocol', step_idx: int) -> None:
        """Go to step ``step_idx`` of ``protocol``; return once the camera holds the step's layer and the stage has arrived.

        ``start_go_to_step``, then ``apply_layer_camera`` for the step's
        layer while the stage travels, then each started move's ``wait()``;
        see ``start_go_to_step`` for what going to a step does and refuses.
        Both waits run in this caller's thread, so the IO lane takes other
        work meanwhile.

        Raises:
            Everything ``start_go_to_step`` raises, and
            MoveNotCompletedError: an axis did not arrive at the step; see
                ``MoveInFlight.wait``. The layer and the preview are the
                step's.
            CameraSettingRejected: the camera refused a setting of the
                step's layer, raised once the stage has arrived; see
                ``apply_layer_camera``.
        """
        # Read before the start, as the start reads it: the step this call
        # was made for, whatever the list holds once the lane has run.
        layer = protocol.step(idx=step_idx)['Color']
        moves = self.start_go_to_step(protocol, step_idx)
        try:
            self.apply_layer_camera(layer)
        finally:
            # A refused apply still waits out the travel it started, so the
            # caller is never handed a raise with the stage still moving.
            for move in moves:
                move.wait()

    @api
    def start_go_to_step(self, protocol: 'Protocol', step_idx: int) -> 'tuple[MoveInFlight, ...]':
        """Start going to step ``step_idx`` of ``protocol``, as a click on a step does.

        One task on the scope's IO lane: the axes are asked once whether
        they know their position; the turret turns to the step's objective
        and X, Y and Z are started towards the step together, on the
        protocol's own plate (``ProtocolsAPI.step_targets``, the targets a
        run computes); the step's values go into its layer's live settings,
        the layer acquiring as the step does and its focus at the step's Z,
        so the layer and the step agree; and the step's LED preview is
        applied -- its channel at its current when ``protocol_led_on`` is
        set, every channel dark when not. It returns the started X, Y and Z
        moves once that task has run, before the stage arrives: each move's
        ``wait()`` says whether it got there, and ``go_to_step`` waits on
        them. A person's click is a gesture and does not wait; a fault on
        the way is the motion monitor's to report. Only the axes this scope
        has are moved: a manual scope moves nothing, does the rest and
        returns no moves.

        A repeat of the step this session last went to (a re-click, a
        re-typed number) does everything but the preview: a channel the
        person lit or put out in between stays as they left it.

        The camera is not set to the step's layer here: a click's caller
        applies it (``apply_layer_camera``), and ``go_to_step`` does.

        Raises:
            StepNotFoundError: ``step_idx`` is not a step of ``protocol``.
                Nothing changes.
            ProtocolRunRefusedError: this scope cannot put the step's
                objective in the light path. Nothing changes.
            ConfigError: this scope has no layer of the step's colour, or
                the step's stimulation names a layer this release has not.
                Nothing changes.
            AxisStateUnknownError: an axis the step moves does not know
                its position. Nothing changes.
            HardwareCommandRefusedError: ``'not_connected'``, this scope's
                motor controller or camera is not connected, or its LED
                controller is not and the step's preview would light;
                ``'scope_disconnected'``
                after ``disconnect()``; or a run, a diagnostic or a home holds the
                scope. Nothing changes.
            PositionOutOfRangeError: the step lies outside an axis's travel;
                the axes before it have moved, nothing else changes.
            MoveNotCompletedError: the turret did not reach the step's slot,
                or the board did not take an axis's command.
        """
        step = protocol.step(idx=step_idx)
        self.scope.protocols.refuse_unaddressable_objectives([step['Objective']])
        self._refuse_layer_not_on_scope(step['Color'], then='be gone to')
        stim_configs = step.get('Stim_Config')
        if isinstance(stim_configs, dict):
            unknown = sorted(set(stim_configs) - set(common_utils.get_layers()))
            if unknown:
                raise ConfigError(
                    f'step {step_idx} stimulates {unknown}, not layers; '
                    f'the layers are {common_utils.get_layers()}'
                )
        # Converted here, from the step read above, so the lane moves to the
        # step this call was made for whatever the list holds by then.
        targets = self.scope.protocols.step_targets(protocol, step_idx)
        # Every member inside bounds its own wait, so the task has no bound
        # of its own to add.
        return self.io_executor.call(
            IOTask(action=self._go_to_step_on_lane, args=(protocol, step_idx, step, targets)),
            'go_to_step',
            timeout_s=None,
        )

    def _go_to_step_on_lane(
        self, protocol: 'Protocol', step_idx: int, step, targets: 'StepTargets'
    ) -> 'tuple[MoveInFlight, ...]':
        """The lane half of ``go_to_step``: ask once, start the moves, load the layer, preview.

        Returns the started moves, one per axis this scope has among X, Y
        and Z; none on a manual scope.
        """
        motion = self.scope.motion
        # A motorized scope whose controller is out of reach cannot go to the
        # step: one that never came up has no axes, so the moves below would
        # be none and the step a silent no-op.
        motion.refuse_controller_not_connected('go_to_step')
        # The step's layer goes onto the camera after the moves; with no
        # camera the step would be moved to and stored with nothing to see.
        self.scope.imaging.refuse_camera_not_connected('go_to_step')
        last = self._last_step_gone_to
        preview = None
        if last is None or last[0] is not protocol or last[1] != step_idx:
            preview = self._step_led_ctx(step)
            # A preview that would light needs the LED controller, asked
            # before anything moves; a dark one needs no board.
            if LedLease.target_leds(LedTransition.MANUAL_STEP, preview):
                self.scope.illumination.refuse_controller_not_connected('go_to_step')
        # The turret included: a failed turret home leaves T unknown
        # while the stage axes still know theirs.
        motion.refuse_unknown_positions(
            self.scope.capabilities.axes, recording=False, then='go to the step'
        )
        if targets.turret_slot is not None:
            # The step's own Z move follows, so the turret need not put Z back.
            motion.move_turret(targets.turret_slot, restore_z=False)
        moves = tuple(
            motion.start_move_absolute(axis, target)
            for axis, target in (('X', targets.x), ('Y', targets.y), ('Z', targets.z))
            if target is not None
        )
        self._load_step_into_layer(step)
        self._last_step_gone_to = (protocol, step_idx)
        if preview is None:
            return moves
        # After the step's moves are started, in the same task: a toggle the
        # person makes while the stage travels lands after the step's preview.
        self.scope.illumination.apply_transition(LedTransition.MANUAL_STEP, preview)
        return moves

    def _load_step_into_layer(self, step) -> None:
        """The step's values into its layer's live settings, under ``settings_lock``.

        A run never writes here: it reads its steps from the protocol, and
        the person's live-view configuration is theirs to keep across it.
        The step's config dicts are copied; the protocol owns them. The
        layer's focus takes the step's Z, so its Goto Focus lands where the
        step was set up. A step's stimulation spans every layer it names,
        and each takes its own.
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
            'focus': step['Z'],
        }
        video_config = step.get('Video Config')
        if isinstance(video_config, dict):
            layer_values['video_config'] = copy.deepcopy(video_config)
        stim_configs = step.get('Stim_Config')
        with self.settings_lock:
            self.settings[color].update(layer_values)
            self._write_layer_acquire(color, step['Acquire'])
            if isinstance(stim_configs, dict):
                for stim_layer, stim_config in stim_configs.items():
                    self.settings[stim_layer]['stim_config'] = copy.deepcopy(stim_config)

    def _step_led_ctx(self, step) -> LedTransitionCtx:
        """The step's LED preview: its channel when the preview is on, all dark when off.

        The MANUAL_STEP transition diffs this against the LEDs' cached
        state, so it clears a lit channel of another colour without
        blinking a same-colour one. Outside a run nothing holds the LED
        lease, so the transition goes through the lease-free
        ``apply_transition``.
        """
        color = step['Color']
        channel = self.scope.illumination.color2ch(color)
        if (
            channel is None
            and self.scope.led_connected
            and color in common_utils.get_layers_with_led()
        ):
            # A step of a layer this unit's identity lacks would otherwise
            # just not light, with nothing anywhere naming why.
            logger.warning(
                f"[Session  ] This scope has no '{color}' LED channel; preview will not light."
            )
        return LedTransitionCtx(
            channel=channel,
            illumination_ma=step['Illumination'],
            preview_on=self.settings['protocol_led_on'],
        )

    @api
    def protocol_size_advisory(self, protocol: 'Protocol') -> 'ProtocolSizeAdvisory | None':
        """Ask a protocol whether it is large enough to warn the user about.

        This is not part of the L2 API surface -- it exists because the two
        settings the estimate needs (whether video is saved as frames, and the
        global FPS cap) are resolved at the session tier rather than in the
        GUI, not to serve a REST caller; there is no REST or headless caller
        today, and this has exactly one caller.

        Resolved the way the run path resolves them, so the advisory and the
        run it is advising about cannot be sized differently.
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

    @api
    def get_settings_snapshot(self) -> dict:
        """A deep copy of the settings dict, taken under the lock.

        A worker thread takes one of these at task entry and reads from it
        for the rest of the task, rather than reading a dict another
        thread may be part-way through rewriting.
        """
        with self.settings_lock:
            return copy.deepcopy(self.settings)

    @api
    def get_setting(self, path: str) -> 'SettingValue':
        """A copy of one setting, named by its dotted path: ``'stage_offset'``, ``'scale_bar.enabled'``.

        Taken under the lock, so it is never a value part-way through a
        write; a copy, so changing it changes no setting (``update_settings``
        and the Session's members do).

        Raises:
            ConfigError: The settings have no value at ``path``.
        """
        with self.settings_lock:
            value = self.settings
            for segment in path.split('.'):
                if not isinstance(value, dict) or segment not in value:
                    raise ConfigError(f'the settings have no {path!r}')
                value = value[segment]
            return copy.deepcopy(value)

    @api
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

        Raises:
            SettingRefusedError: ``path`` is owned by a Session member
                (named), is set only by the installation, is not a setting,
                or names a block; or ``value`` is not the setting's kind or
                is outside its range. Nothing is written.
            ConfigError: these settings were never prepared from the
                template and lack the block the path is in. Nothing is
                written.
            ProtocolScheduleRefusedError: a ``protocol.period`` or
                ``protocol.duration`` no protocol can run. Nothing is
                written.
        """
        settings_paths.check_write(self.scope.settings_template, path, value)
        with self.settings_lock:
            self._store_setting(path, value)

    @api(in_process=True)
    def set_live_folder(self, folder: str) -> None:
        """Make ``folder`` the live folder, where captures and runs are saved.

        The one writer of ``live_folder``, stored as it is at load: a folder
        given relative to the installation is made absolute, and created. A
        folder that cannot be created is still stored; captures into it are
        refused, naming it, until it is reachable.

        Raises:
            SettingRefusedError: ``'out_of_range'``, ``folder`` is a string
                no file system can name (a NUL byte). Nothing is written.
        """
        try:
            stored = settings_init.bring_up_live_folder(logger, folder, self.scope.source_path)
        except ValueError as e:
            # pathlib's answer to a string no file system can name (a NUL byte).
            raise SettingRefusedError(
                'out_of_range', 'live_folder', f'{folder!r} is not a path'
            ) from e
        with self.settings_lock:
            self._store_setting('live_folder', stored)

    @api(in_process=True)
    def live_folder_path(self, name: str) -> pathlib.Path:
        """The absolute path that ``name``, a name under the live folder, names.

        The one door from a wire caller's path argument to the file system:
        a REST bridge passes every path a caller gives through this, so the
        caller reaches the live folder and nothing beside it. A Python or GUI
        caller passes any path straight to the member it calls. The live
        folder is not created here: one that is missing is an unplugged drive
        or a stale setting.

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
    def set_protocol_filepath(self, file_path: str) -> None:
        """Remember ``file_path`` as the protocol to open at the next start; ``''`` forgets it.

        The one writer of ``protocol.filepath``, which the next start opens.
        """
        with self.settings_lock:
            self._store_setting('protocol.filepath', file_path)

    @api(in_process=True)
    def open_protocol(self, file_path: FilePath) -> 'Protocol':
        """Load the protocol at ``file_path`` with its Layer Settings, and remember it.

        The GUI's Load: ``load_protocol``, then ``apply_layer_settings``, then
        ``set_protocol_filepath``, so the next start opens it. The path is
        written last, once the scope is on the protocol's plate and the layer
        controls hold its settings: a refused file leaves the remembered path
        where it was.

        It is not part of the L2 API surface, like the remembered path it
        writes.

        Raises:
            ProtocolNotLoadedError, ProtocolFormatError,
            ProtocolRunRefusedError, ConfigError,
            HardwareCommandRefusedError: As ``load_protocol`` and
                ``apply_layer_settings`` raise them; no path is remembered.
        """
        protocol = self.load_protocol(file_path)
        self.apply_layer_settings(protocol)
        self.set_protocol_filepath(os.fspath(file_path))
        return protocol

    @api(in_process=True)
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
            The protocol, or None when no path is remembered or its file is
            gone (the path is then forgotten, logged at INFO: a protocol moved
            or deleted since is not a fault).

        Raises:
            ProtocolNotLoadedError: The file is there and cannot be read; the
                path is forgotten.
            ProtocolFormatError, ProtocolRunRefusedError, ConfigError,
            HardwareCommandRefusedError: As ``load_protocol`` raises them;
                the path is kept.
        """
        with self.settings_lock:
            file_path = self.settings['protocol']['filepath']
        # Asked before any path test: Path('') is the working folder, which
        # exists, so an empty path would be loaded and fail as a fault.
        if not file_path:
            return None
        if not pathlib.Path(file_path).exists():
            logger.info(f'[Session   ] The remembered protocol {file_path} is gone; forgotten.')
            self.set_protocol_filepath('')
            return None
        try:
            return self.open_protocol(file_path)
        except ProtocolNotLoadedError:
            self.set_protocol_filepath('')
            raise

    def _store_setting(self, path: str, value: object) -> None:
        """Under ``settings_lock``: put ``value`` at ``path`` in the live settings.

        The Session's own members write their settings through this,
        past ``update_settings``' refusal of the settings they own.
        """
        *blocks, leaf = path.split('.')
        holder = self.settings
        for block in blocks:
            if not isinstance(holder.get(block), dict):
                raise ConfigError(
                    f'these settings have no {block!r} block for {path}: they were '
                    'not prepared from the template'
                )
            holder = holder[block]
        holder[leaf] = value

    @api
    def select_model(self, model: str) -> None:
        """Save the operator's scope model for the next start.

        The running scope is not changed: its capabilities are fixed at
        construction, so re-resolving its layers for the new model would
        leave one scope carrying two models -- the new one in its layers,
        the old one in its capabilities and its saved files. The selection
        takes effect when the scope is next brought up, and on a scope whose
        motor board reports its own model the board's report still wins
        there.

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

    @api
    @property
    def model_at_next_start(self) -> str | None:
        """The saved model when it is not the one running, else None.

        What ``select_model`` saved waits for the next bring-up, so until
        then the scope runs as one model and the settings name another; a
        caller shows this rather than comparing the two itself.
        """
        saved = self.settings['microscope']
        return None if saved == self.scope.layer_identity.model else saved

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
        # select_labware refuses a protocol block that is not a mapping.
        block = self.settings.get('protocol')
        stored = block.get('labware') if isinstance(block, dict) else None
        if not self.select_labware(CENTER_PLATE):
            return
        logger.info(
            f'[Session  ] stored plate {stored!r} replaced by {CENTER_PLATE!r}: '
            'this scope has no XY stage'
        )

    def _replace_an_unresolvable_stored_plate(self) -> list[tuple[str, object, object]]:
        """Replace a stored plate the catalogue cannot resolve with the shipped one.

        A null, a non-string, an empty name or a plate the catalogue no
        longer has. The catalogue is first available here, at bring-up, so
        the name is judged here rather than with the load's other stored
        values. Refusing it instead brought the GUI up on the shipped
        template, dropping every other stored setting for one plate name.

        Returns:
            ``[('protocol.labware', stored, shipped)]`` when replaced, else ``[]``.
        """
        block = self.settings.get('protocol')
        stored = block.get('labware') if isinstance(block, dict) else None
        if self.wellplate_loader.is_known_plate(stored):
            return []
        shipped = self.scope.settings_template['protocol']['labware']
        self.select_labware(shipped)
        logger.info(
            f'[Session  ] stored plate {stored!r} replaced by {shipped!r}: '
            'the catalogue has no such plate'
        )
        return [('protocol.labware', stored, shipped)]

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
        camera lane. The
        factories run this for the scope they build; a host that constructs
        the session directly, or hands ``create`` its own scope, calls it
        once itself. Every other step runs on the calling thread.

        Raises:
            ConfigError: a settings key ``initialize`` cannot do without is
                missing (``frame``; ``objective_id`` on a scope with no
                turret); or that ``objective_id`` names no shipped objective.
            CameraSettingRejected: the camera refused a write of BF's
                settings.
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
        detected = self.scope.diagnostics.get_motor_info()['model']
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
        # A caller-supplied dict never went through prepare_settings, whose
        # normalizer is the one boundary between the file's string slot keys
        # and the runtime's ints; without it every slot reads as unassigned.
        settings_init._normalize_turret_slot_keys(self.settings)
        self._put_a_stageless_scope_on_center_plate()
        scope_config = scope_models.get(self.settings.get('microscope'))
        # Before anything is commanded: every well position is computed on it.
        replaced.extend(self._replace_an_unresolvable_stored_plate())
        config = ScopeInitConfig.from_settings(
            self.settings,
            scope_config=scope_config,
            layer_identity=self.scope.layer_identity,
            turreted=self.scope_has_turret(),
        )
        self.scope.initialize(config)
        self._store_delivered_geometry()
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
        self._apply_bring_up_layer()

    def _apply_bring_up_layer(self) -> None:
        """Put BF's stored camera settings on the camera, as the GUI opens on BF.

        Skipped, and said so once, when there is no camera to set (bring-up
        has already reported it missing; a second report would only repeat
        it) or this scope has no BF layer (a newer unit whose layers this
        release cannot resolve still comes up). The values are capped to the
        camera's range first, so a refusal here is the camera failing a
        write, and it fails bring-up like any other part that does not come
        up.
        """
        layer = common_utils.DEFAULT_LAYER
        if not self.scope.camera_connected:
            logger.info(f'[Session  ] bring-up: no camera, so {layer} is not applied to it')
            return
        if layer not in self._layers_on_scope():
            logger.info(
                f'[Session  ] bring-up: this scope has no {layer} layer, '
                'so no layer is applied to the camera'
            )
            return
        self.apply_layer_camera(layer)

    @api
    def bring_up_record(self) -> 'BringUpRecord':
        """What bring-up found, substituted and set aside, for a client that asks later.

        Which parts came up and, for each that did not, why; what bring-up
        used in place of a saved setting the camera could not take, the
        saved value beside it; and the settings file set aside with its
        reason, while the app runs on the shipped template. A client given
        to ``create(outcome_listener=...)`` heard these as outcomes; this is
        the same facts held, read-through from the scope and the settings
        store, for one that connects afterwards.
        """
        from modules.lumascope_api.bring_up import SettingsSetAside

        rejected = settings_init.rejected_current_json
        set_aside = None if rejected is None else SettingsSetAside(*rejected)
        return dataclasses.replace(self.scope.bring_up_record(), settings_set_aside=set_aside)

    @api
    @slow_task_budget(SUPPORT_REPORT_SLOW_TASK_S)
    def make_support_report(
        self,
        *,
        include_bandwidth_test: bool = False,
        output_dir: FilePath | None = None,
        on_progress: Callable[[int, str], None] | None = None,
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
            output_dir: Where the ZIP is written; the Desktop by default,
                the home folder when there is none.
            on_progress: Called with a percentage and what the report is
                doing, from the thread the report runs on.

        Returns:
            Where the ZIP is, and the words that say where to send it.

        Raises:
            SupportReportNotSavedError: no ZIP was saved; chained from the
                failure, whose words it carries.
        """
        from modules.tech_support_report import SupportReportSaved, TechSupportReport

        report = TechSupportReport(session=self)
        with self._report_in_flight(live_work.SUPPORT_REPORT):
            return SupportReportSaved(
                report.generate(
                    callback=on_progress,
                    include_bandwidth_test=include_bandwidth_test,
                    output_dir=output_dir,
                ),
                'support report',
            )

    @api
    def make_logs_zip(
        self,
        *,
        output_dir: FilePath | None = None,
        on_progress: Callable[[int, str], None] | None = None,
    ) -> 'SupportReportSaved':
        """Zip the logs, the data folder, the recent protocols and the video receipts.

        Touches no hardware and does not hold the scope: it is for sending
        the record of what already happened.

        Args:
            output_dir: Where the ZIP is written; the Desktop by default,
                the home folder when there is none.
            on_progress: Called with a percentage and what the zip is doing,
                from the thread it runs on.

        Returns:
            Where the ZIP is, and the words that say where to send it.

        Raises:
            SupportReportNotSavedError: no ZIP was saved; chained from the
                failure, whose words it carries.
        """
        from modules.tech_support_report import SupportReportSaved, TechSupportReport

        report = TechSupportReport(session=self)
        with self._report_in_flight(live_work.LOGS_ZIP):
            return SupportReportSaved(
                report.generate_logs_only(callback=on_progress, output_dir=output_dir),
                'logs zip',
            )

    @api
    def plugin_health(self) -> 'PluginHealth | None':
        """The loaded plugins, the ones that did not load, and their runtime errors.

        None on a session whose host never asked for plugins (``load_plugins``).
        """
        return None if self.plugins is None else self.plugins.health()

    @api(in_process=True)
    def load_plugins(self) -> None:
        """Load the installed plugins and the built-ins into this session.

        The host's one call to have plugins; a session it is never made on
        has none. Each plugin's ``register`` is handed this session as its
        ctx. A plugin that does not load is reported and the rest load;
        nothing here raises for a plugin. Call it once, after ``create``.

        Raises:
            RuntimeError: this session already loaded its plugins.
        """
        if self.plugins is not None:
            raise RuntimeError('ScopeSession.load_plugins: this session already loaded its plugins')
        version, _build_timestamp = path_utils.read_version()
        self.plugins = PluginRegistry()
        self.plugins.load(self, version)

    @api(in_process=True)
    def unload_plugins(self) -> None:
        """Call each loaded plugin's ``unregister``, last loaded first.

        A host whose close has to stop plugin work before anything else --
        the GUI's, before it stops runs and saves settings -- calls this
        first; ``shutdown`` calls it as its first step for every other host.
        A second call, and a call on a session with no plugins, does
        nothing. A plugin's own failure is logged and the rest unload.
        """
        if self.plugins is not None:
            self.plugins.unload(self)

    def _run_protocol_complete_processors(
        self, run_dir: pathlib.Path, files: str, trigger_source: str, protocol_name: str
    ) -> None:
        """Hand a finished Full Protocol's folder to the opted-in processors.

        The run engine calls this once per Full Protocol, on the run's own
        post-run thread, after its images are written and its hyperstack
        build has ended. The processors run on the post-processing lane,
        isolated from the run, and this thread waits for them, so their
        lane task is never left with nobody to read its outcome.
        """
        if self.plugins is None:
            return
        run_dir_str = str(run_dir)
        manifest = {
            'protocol_name': protocol_name,
            'run_dir': run_dir_str,
            'trigger_source': trigger_source,
        }
        try:
            self.post_processing.lane.call(
                IOTask(
                    action=self.plugins.run_protocol_complete_processors,
                    args=(run_dir_str, manifest, run_dir_str, files),
                ),
                'plugins.run_protocol_complete_processors',
                None,
            )
        except Exception as refused:
            # The lane refused the task (closed at shutdown); the processors
            # report their own failures inside it.
            from modules.notification_center import notifications

            notifications.report_outcome(refused, solicited=False, category='Plugins')

    @api
    @property
    def plugin_api_level(self) -> int:
        """What this LumaViewPro does that a plugin may rely on.

        ``modules.plugins.PLUGIN_API_LEVEL``; each level's additions are
        listed in LumascopeSkills.md, "Plugin API level".
        """
        return PLUGIN_API_LEVEL

    @api
    def settings_are_provisional(self) -> bool:
        """Is the app running on defaults nobody has agreed to keep?

        True while the user's current.json could not be used and no one
        has decided its fate. While it holds, every save aimed at
        current.json raises SettingsSaveRefusedError -- resolve with
        retire_rejected_settings() after the user has chosen to start
        over.
        """
        return settings_init.settings_are_provisional()

    @api
    def retire_rejected_settings(self) -> 'str | None':
        """Resolve the provisional-settings state: retire the rejected file.

        Moves the unusable current.json aside (renamed, never deleted --
        it is the user's only copy) so a fresh one can take its place,
        and clears the provisional state so saves work again. Call only
        after a human has chosen to start over. Returns the retired
        path, or None when nothing was provisional.
        """
        return settings_init.retire_rejected_current_json()

    @api(in_process=True)
    def save_settings(self, file: str = './data/current.json', *, force: bool = False) -> None:
        """Write the settings dict to disk as JSON.

        Refused (raising) when writing would destroy real data: when no
        hardware was connected this session the sliders sit at their
        defaults (0.01 ms exposure and the like), and writing those over
        a user's real per-channel values silently loses them -- a caller
        that means the save regardless passes force=True, since an API
        write has no slider behind it to misread.

        Raises:
            SettingsSaveRefusedError: reason='settings_provisional' when
                the app is running on the shipped template because
                current.json could not be used AND the save targets
                current.json (force does not override; a save aimed at
                any other destination still writes). reason='no_hardware'
                when no hardware was connected this session and force is
                not set.
            ConfigError: the scope has not been configured
                (``configure_scope`` has not run), so whether it has a
                turret -- and so which slot to record -- is not known.
        """
        logger.info('[Session  ] save_settings()')

        # Outside the force gate on purpose: force means "save even though no
        # hardware was connected", not "save over a file we were told to leave
        # alone". The settings in memory right now are the shipped template,
        # loaded because the user's own file could not be used; writing them
        # to current.json would replace their entire configuration with
        # defaults. Resolved by retire_rejected_settings() once the user
        # has actually chosen to start over.
        if settings_init.settings_are_provisional() and settings_init.targets_current_json(file):
            logger.warning(
                '[Session  ] save_settings: refused -- running on default '
                'settings because current.json could not be used. Not '
                'overwriting it until the user decides.'
            )
            raise SettingsSaveRefusedError(reason='settings_provisional', file=file)

        if not force:
            scope = self.scope
            # Found at bring-up, not connected now: the sliders hold the
            # user's real values once hardware configured them, and a scope
            # unplugged since -- an FX2 scope has no board left to answer --
            # still has a session worth saving.
            had_hardware = bool(scope) and not scope.no_hardware
            if not had_hardware:
                logger.info(
                    '[Session  ] save_settings: refused -- no hardware was '
                    'connected this session (would overwrite real per-channel '
                    'values with slider defaults). Pass force=True to override.'
                )
                raise SettingsSaveRefusedError(reason='no_hardware', file=file)

        if isinstance(file, str) and (file[-5:].lower() != '.json'):
            file = file + '.json'

        t0 = time.monotonic()
        settings_snapshot = self.get_settings_snapshot()
        # The persisted turret position is the slot a person last turned to,
        # taken at save, which the next session's slot lookup prefers when two
        # slots carry one objective. On a turreted scope nothing writes
        # objective_id: the objective is the slot's assignment, so the file
        # keeps whatever it held -- never null, which a launch as a
        # turretless model would refuse.
        if self.scope.runtime_state.is_turreted():
            slot = self.scope.motion.get_preferred_turret_slot()
            if slot is not None:
                settings_snapshot['turret_position'] = slot
        # Resolve relative paths against source_path instead of relying on CWD
        if not os.path.isabs(file):
            file = os.path.join(self.source_path, file)
        with open(file, 'w') as write_file:
            json.dump(settings_snapshot, write_file, indent=4, cls=CustomJSONizer)
        dt = time.monotonic() - t0
        if dt > 0.1:
            logger.warning(f'[Session  ] save_settings took {dt * 1000:.0f}ms')

        if self.plugins is not None:
            self.plugins.settings_saved(self, settings_snapshot)

    @api
    def capture_settings_snapshot(self) -> dict:
        """A settings snapshot for composing a capture or a run.

        ``get_settings_snapshot`` with ``objective_id`` set to the active
        objective (``runtime_state.resolve_current_objective``), which on a turreted scope
        the stored settings do not carry. Not for saving: a turreted scope
        persists no objective_id of its own.

        Raises:
            ObjectiveUnknownError: The active objective is unknown.
        """
        objective_id, _ = self.scope.runtime_state.resolve_current_objective()
        snapshot = self.get_settings_snapshot()
        snapshot['objective_id'] = objective_id
        self._read_autofocus_as_off_without_z(
            snapshot[layer] for layer in common_utils.get_layers()
        )
        return snapshot

    def get_objective_info(self, objective_id: str) -> dict:
        """Objective metadata for an EXPLICIT id.

        Candidate lookups (turret assignment, settings load, FOV
        refresh) read objectives the current selection does not name,
        so a current-only getter cannot serve them.
        """
        return self.objective_helper.get_objective_info(objective_id=objective_id)

    # ------------------------------------------------------------------
    # The objective: the question, the answer and the plain writers
    # ------------------------------------------------------------------

    def scope_has_turret(self) -> bool:
        """Does this scope have a turret, as well as it can be known?

        The board when the board is talking; the declared model only when
        it is not.

        The declaration alone was wrong in the common direction. The
        shipped template declares LS850, whose catalogue entry has no
        turret, so a real LS850T running on shipped settings was never
        asked for the objective at its current slot -- and whatever the
        stored objective happened to be went on setting the image scale.

        The declaration is still the answer for a motorboard that is not
        connected, and that case is the reason it was chosen: a dead board
        reports no axes, so believing it would say "no turret" and let the
        scope answer with its stored objective instead of the one in the
        light path. Between a
        board that cannot speak and a file that can be wrong, the file is
        the better witness.

        A composition detail, not part of the L2 API surface: it exists so
        the two startup questions below ask one question once, rather than
        each reading the declaration and drifting apart.
        """
        if self.scope.motor_connected:
            return bool(self.scope.capabilities.has_turret)
        import modules.config_helpers as config_helpers

        return config_helpers.model_has_turret(
            self.scope.scope_models, self.scope.layer_identity.model
        )

    @api
    def objective_question(self) -> 'ObjectiveQuestion | None':
        """Does the objective need confirming? The question, or None.

        A read, for any host: the GUI renders the answer as a popup, a
        REST caller reads it as state. Two ways the session cannot know
        what is in the light path: no person has ever confirmed the
        objective on this install (the settings template ships a default
        that would otherwise set image scale silently forever), or, on a
        DECLARED turret model, the slot in the light path has no
        assignment. The declared model, not the live capability: a dead
        motorboard reports no axes, and that is exactly when a stale
        objective must not pass unasked. The slot is the live one
        (``motion.get_turret_slot``), so an answer names the glass that is
        actually in the light path.

        Three conditions withhold an owed question, each leaving one log
        line per call so a bundle can say why nothing was asked: with no
        hardware there is nothing in the light path and no capture to
        stamp; while settings are provisional every write is refused, so
        an answer given now would be lost -- the host re-asks when they
        resolve; while an activity holds the scope the answer is refused
        too (``confirm_objective``), so asking would only ask again after
        each refusal -- the host re-asks when the hold ends. No line is
        logged for a returned question: the renderer logs its own show,
        and a polled read must not log per poll.

        Raises:
            ConfigError: the catalogue is empty, or the model catalogue
                cannot be read.
            ObjectiveUnknownError: the objective has never been confirmed
                on this install and the turret model's slot is unknown --
                there is no slot to answer for until the turret is homed
                or moved.
        """
        has_turret = self.scope_has_turret()
        first_run = not self.settings.get('objective_confirmed', False)
        slots = self.settings.get('turret_objectives') or {}
        position = self.scope.motion.get_turret_slot() if has_turret else None
        # A known slot with no assignment owes the question. An unknown slot
        # alone does not: it is unknown during every turret move, and asking
        # then would put the question up while the turret is still turning.
        # The objective is unknown meanwhile, and captures refuse with why.
        slot_unassigned = has_turret and position is not None and slots.get(position) is None
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
        if has_turret and position is None:
            raise ObjectiveUnknownError('slot_unknown')
        choices = tuple(self.objective_helper.get_objectives_list())
        if not choices:
            raise ConfigError(
                'the objective catalogue is empty; cannot ask which objective is installed'
            )
        # Only the slot's own assignment names glass anyone has confirmed. The
        # stored objective_id does not: the question is owed on a turretless
        # scope only before anyone has confirmed one, when it is the shipped
        # template's value. Everything else gets the catalogue's default.
        proposed = slots.get(position) if has_turret else None
        if proposed not in choices:
            from modules.objectives_loader import DEFAULT_PROPOSED_OBJECTIVE_ID

            proposed = DEFAULT_PROPOSED_OBJECTIVE_ID
        return ObjectiveQuestion(turret_position=position, proposed=proposed, choices=choices)

    @api
    def confirm_objective(self, objective_id: str, turret_position: 'int | None' = None) -> bool:
        """Answer the objective question: this objective is in the light path.

        With ``turret_position`` given, assigns the objective to that slot;
        otherwise selects it (``select_objective``). Records that a person
        has confirmed the objective on this install. Returns whether the
        active objective changed.

        Raises:
            ConfigError: ``objective_id`` is not exactly a catalogue key.
                Nothing is written.
            ObjectiveUnknownError: No ``turret_position`` was given on a
                turreted scope whose slot is unknown.
            ValueError: ``turret_position`` is not a slot number 1-4.
        """
        if turret_position is None:
            changed = self.select_objective(objective_id)
        else:
            before = self.scope.runtime_state.get_current_objective_id()
            self.assign_turret_objective(turret_position, objective_id)
            if not self.scope.runtime_state.is_turreted():
                self.select_objective(objective_id)
            changed = self.scope.runtime_state.get_current_objective_id() != before
        with self.settings_lock:
            self.settings['objective_confirmed'] = True
        logger.info(
            f'[Session  ] objective confirmed: {objective_id!r}'
            + (f' at turret position {turret_position}' if turret_position is not None else '')
        )
        return changed

    @api
    def select_objective(self, objective_id: str) -> bool:
        """Make ``objective_id`` the active objective. Returns whether it changed.

        The one writer of the active objective for every host. With no
        turret, the selected objective is the ``objective_id`` setting, which
        the scope reads. On a turreted scope the active
        objective IS the slot's assignment, so picking one assigns it to the
        slot in the light path -- the person is saying what is installed
        there. The resolved optics are recorded by the scope's runtime state
        the next time the objective is read, before any capture stamps it.
        Picking the objective already active is a no-op.

        Raises:
            ConfigError: ``objective_id`` is not exactly a catalogue key.
                The refusal lands before any write.
            ObjectiveUnknownError: On a turreted scope, the slot in the
                light path is unknown, so there is no slot to assign.
            HardwareCommandRefusedError: A run, a diagnostic or a recording
                holds the scope
                (``exclusive_activity_running``). Nothing is written.
        """
        if objective_id == self.scope.runtime_state.get_current_objective_id():
            return False
        # Refuses an id that is not a catalogue key, before any write.
        self.objective_helper.get_objective_info(objective_id=objective_id)
        self._refuse_configuration_change_while_held('select_objective')
        if self.scope.runtime_state.is_turreted():
            slot = self.scope.motion.get_turret_slot()
            if slot is None:
                raise ObjectiveUnknownError('slot_unknown')
            self.assign_turret_objective(slot, objective_id)
        else:
            with self.settings_lock:
                self._store_setting('objective_id', objective_id)
        return True

    # ------------------------------------------------------------------
    # The labware
    # ------------------------------------------------------------------

    @api
    def select_labware(self, labware_name: str) -> bool:
        """Make ``labware_name`` the current plate. Returns whether it changed.

        The one writer of the active labware for every host: the plate the
        settings name, which the scope reads.

        The name is stored in the catalogue's spelling: a plate renamed
        since a protocol or settings file named it is accepted under the old
        name and written under the key, so the settings store never carries
        a spelling the catalogue lacks, and whether the plate changed is
        decided on the key rather than on how it was spelled.

        Raises:
            ConfigError: ``labware_name`` is not a string, the loader
                cannot resolve the name, or the settings have no protocol block to hold the
                selection. Nothing is written.
            HardwareCommandRefusedError: A run, a diagnostic or a recording
                holds the scope and ``labware_name`` is not the plate in place
                (``exclusive_activity_running``): each states its positions
                against the plate it started with. Nothing is written.
        """
        labware_name = self.wellplate_loader.resolve_plate_key(labware_name)
        protocol_settings = self.settings.get('protocol')
        if not isinstance(protocol_settings, dict):
            # Settings handed straight to a factory skip the template merge
            # that puts this block there, so it can be missing -- and a
            # hand-edited file can put something that is not a mapping in its
            # place. Named here, so the refusal says what is wrong rather
            # than failing inside the write.
            raise ConfigError(
                'settings have no usable protocol block; the labware selection '
                f'has nowhere to live (found {type(protocol_settings).__name__})'
            )
        if labware_name == protocol_settings.get('labware'):
            # A holder is refused only a different plate: the GUI re-selects
            # the current one whenever its panels redraw, under any hold.
            return False
        self._refuse_configuration_change_while_held('select_labware')
        with self.settings_lock:
            self._store_setting('protocol.labware', labware_name)
        logger.info(f'[Session  ] Labware set to {labware_name!r}')
        return True

    @api
    def assign_turret_objective(self, position: int, objective_id: str) -> None:
        """Bind ``objective_id`` to turret slot ``position``.

        Binding the objective the slot already holds is a no-op.

        Raises:
            ValueError: ``position`` is not a slot number 1-4.
            ConfigError: ``objective_id`` is not exactly a catalogue key.
            HardwareCommandRefusedError: A run, a diagnostic or a recording
                holds the scope
                (``exclusive_activity_running``). Nothing is written.
        """
        self._check_turret_slot(position)
        if objective_id not in self.objective_helper.get_objectives_list():
            raise ConfigError(f'unknown objective {objective_id!r}; the catalogue has no such key')
        if self.settings['turret_objectives'].get(position) == objective_id:
            return
        self._refuse_configuration_change_while_held('assign_turret_objective')
        with self.settings_lock:
            self.settings['turret_objectives'][position] = objective_id

    @api
    def clear_turret_objective(self, position: int) -> None:
        """Leave turret slot ``position`` unassigned.

        Logged for every host: with that slot in the light path the active
        objective is now unknown, and a support bundle can only explain a
        capture refused for that afterwards if the clear is in the record.

        Raises:
            ValueError: ``position`` is not a slot number 1-4.
            HardwareCommandRefusedError: A run, a diagnostic or a recording
                holds the scope and the slot
                has an assignment (``exclusive_activity_running``). Nothing
                is written.
        """
        self._check_turret_slot(position)
        if self.settings['turret_objectives'].get(position) is not None:
            self._refuse_configuration_change_while_held('clear_turret_objective')
        with self.settings_lock:
            self.settings['turret_objectives'][position] = None
        logger.info(
            f'[Session  ] Turret position {position} cleared; the active objective is now '
            f'{self.scope.runtime_state.get_current_objective_id()!r}'
        )

    @api
    def clear_current_turret_objective(self) -> None:
        """Leave the turret slot in the light path unassigned.

        The counterpart of ``select_objective`` on a turreted scope: that
        says which objective is installed in the current slot, this says
        none is known to be. The slot is the API's, never the caller's
        guess of it.

        Raises:
            ObjectiveUnknownError: The slot in the light path is unknown,
                so there is no slot to clear (``slot_unknown``).
            HardwareCommandRefusedError: A run, a diagnostic or a recording
                holds the scope and the slot
                has an assignment (``exclusive_activity_running``). Nothing
                is written.
        """
        slot = self.scope.motion.get_turret_slot()
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

    @staticmethod
    def _check_turret_slot(position) -> None:
        if not isinstance(position, int) or isinstance(position, bool) or not 1 <= position <= 4:
            raise ValueError(f'turret slot must be a whole number 1-4, got {position!r}')

    @api
    def get_current_plate_position(self) -> dict:
        """The stage's position in plate coordinates, on the session's labware.

        Returns:
            dict: ``'x'`` and ``'y'`` in mm, ``'z'`` in um.

        Raises:
            AxisStateUnknownError: an axis the scope has does not know its
                position (before a home, after a lost reference), so there
                is no position to convert. Reported once.
            HardwareCommandRefusedError: ``'not_connected'``, the model's
                motor controller is not connected.
        """
        # The cable before the reference: with the controller gone every
        # axis is unknown, and the remedy is the cable, not a home.
        self.scope.motion.refuse_controller_not_connected('get_current_plate_position')
        self.scope.motion.refuse_unknown_positions(
            ('X', 'Y', 'Z'), recording=True, then='try again'
        )
        return self.plate_position_on(self.settings.get('protocol', {}).get('labware'))

    def plate_position_on(self, labware_id: str) -> dict:
        """The stage's position in plate coordinates, on the plate ``labware_id`` names.

        For a caller that states the position on a plate other than the
        session's live selection: a protocol's own plate, or the plate in a
        run's settings snapshot. It asks nothing about the axes; the step or
        run the position goes into is refused there when one is unknown.
        It is not part of the L2 API surface: a caller states a position
        through the step and run members that read it.

        Returns:
            dict: ``'x'`` and ``'y'`` in mm, ``'z'`` in um.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, the model's
                motor controller is not connected.
            ConfigError: ``labware_id`` is not a plate the catalogue has.
        """
        import modules.config_helpers as config_helpers

        return config_helpers.get_current_plate_position(
            self.scope,
            self.settings,
            self.coordinate_transformer,
            self.wellplate_loader,
            labware_id,
        )

    # ------------------------------------------------------------------
    # The camera's capture settings: applied, then stored
    # ------------------------------------------------------------------

    @api
    def set_scale_bar(self, enabled: bool) -> None:
        """Draw the scale bar on captured images, or stop, and store it.

        The one writer of ``scale_bar.enabled``, which the imaging API reads
        at each capture.
        """
        with self.settings_lock:
            self._store_setting('scale_bar.enabled', enabled)

    @api
    def set_acceleration_limit(self, val_pct: int) -> None:
        """Set the motors' acceleration limit, as a percent of the firmware's maximum, and store it.

        The one writer of ``motion.acceleration_max_pct``, stored once the
        motor controller took the value.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller to take it. Nothing
                is stored.
            AccelerationLimitRefusedError: ``val_pct`` is outside the range
                the motion API accepts (a ValueError). Nothing is stored.
        """
        self.scope.motion.set_acceleration_limit(val_pct=val_pct)
        with self.settings_lock:
            self._store_setting('motion.acceleration_max_pct', val_pct)

    @api
    def set_high_conversion_gain(self, enabled: bool) -> bool:
        """Turn the camera's high conversion gain on or off, then store it.

        High conversion gain lowers the sensor's read-noise floor at the cost
        of dynamic range. The one writer of ``camera.high_conversion_gain``:
        the setting is stored only once the camera took it, so the store
        never names a mode the camera is not in.

        Returns:
            True when the camera took it and it is stored. False when the
            camera has no such mode or it refused (each reported by the
            imaging API); nothing is stored.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected. Nothing is stored.
        """
        if not self.scope.imaging.set_conversion_gain_mode('High' if enabled else 'Low'):
            return False
        with self.settings_lock:
            self._store_setting('camera.high_conversion_gain', enabled)
        return True

    @api
    def set_line_noise_reduction(self, enabled: bool) -> bool:
        """Turn the camera's line-noise filter on or off, then store it.

        The one writer of ``camera.line_noise_reduction``, stored only once
        the camera took it, as ``set_high_conversion_gain`` is.

        Returns:
            True when the camera took it and it is stored. False when the
            camera has no such filter or it refused (each reported by the
            imaging API); nothing is stored.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected. Nothing is stored.
        """
        if not self.scope.imaging.set_line_noise_reduction(enabled):
            return False
        with self.settings_lock:
            self._store_setting('camera.line_noise_reduction', enabled)
        return True

    @api
    def set_image_mode(self, mode: str) -> bool:
        """Capture in ``mode``: apply the camera format it needs, then store it.

        The one writer of ``settings['image_mode']`` for every host. The mode
        names a capture depth, and the format is chosen from the ones this
        camera reports, so a sensor is never asked for a format it lacks. A
        camera that reports none (none is connected) has nothing to apply;
        the mode is stored and bring-up applies it.

        Returns:
            True when the mode is stored. False when the camera went away
            between the format query and the apply; nothing is stored.

        Raises:
            ConfigError: ``mode`` is not an image mode. Nothing is stored.
            CameraSettingRejected: The camera refused the format. Nothing is
                stored, so captures are never tagged with a depth the camera
                is not delivering.
        """
        capture_depth = image_mode.resolve_image_mode(mode)['capture_depth']
        imaging = self.scope.imaging
        target = image_mode.select_capture_pixel_format(
            capture_depth, self.scope.capabilities.camera_pixel_formats
        )
        if target is not None and not imaging.set_pixel_format(target):
            return False
        with self.settings_lock:
            self.settings['image_mode'] = mode
        return True

    @api
    def set_binning_size(self, size: int) -> 'dict | None':
        """Bin the camera by ``size``, keep the framed region, and store both.

        The one writer of the binning for every host. The framed region is
        held unbinned (``frame['native_width'/'native_height']``), so a new
        binning divides that region by the new factor rather than rescaling
        the last displayed size, and cycling the binning round-trips. The
        binning goes first and the frame after it, because the camera's frame
        limits depend on the binning in force; one caller running both in
        order is what keeps a frame edit from being worked out against a
        binning the camera has not reached.

        Returns:
            The frame the camera delivers, ``{'width', 'height'}``.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected. Nothing is stored.
            CameraSettingUnsupportedError: This camera does not offer
                ``size``. Nothing reaches the camera.
            CameraSettingRejected: The camera refused the binning (nothing is
                stored) or the frame after it (the binning is stored, with the
                frame the camera reports holding at it).
        """
        imaging = self.scope.imaging
        # Asked before the offered sizes: a scope with no camera offers
        # none, and "does not support" would name the wrong cause.
        imaging.refuse_camera_not_connected('set_binning_size')
        offered = self.scope.capabilities.camera_binning_sizes
        label = binning.binning_size_int_to_str(size)
        if size not in offered:
            raise CameraSettingUnsupportedError(
                'binning',
                size,
                offered,
                title='Binning not supported',
                message=f'This camera does not support {label} binning.',
            )
        native = self._native_frame()
        if not imaging.set_binning_size(size):
            return None
        with self.settings_lock:
            self.settings['binning']['size'] = label
            held = imaging.frame_size_cached
            self.settings['frame']['width'] = int(held['width'])
            self.settings['frame']['height'] = int(held['height'])
        return self._apply_frame(native, self._target_frame(native, size))

    @api
    def set_frame_size(self, width: int, height: int) -> 'dict | None':
        """Frame the camera at ``width`` x ``height`` at the stored binning, and store it.

        The one writer of the frame for every host. The size is what the
        person sees and captures (post-binning); the unbinned region it
        implies is stored beside it. A size the camera already delivers is
        not written again.

        Returns:
            The frame the camera delivers: the request floored to even
            sides.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected; ``'scope_disconnected'`` after
                ``disconnect()``. Nothing is stored.
            CameraSettingOutOfRangeError: The size, floored to even sides, is
                below the camera's minimum frame or above the scope's maximum
                at the stored binning. Nothing is stored.
            CameraSettingRejected: The camera refused the frame. Nothing is
                stored.
        """
        factor = binning.binning_size_str_to_int(self.settings['binning']['size'])
        native = {'width': int(width) * factor, 'height': int(height) * factor}
        target = binning.native_to_displayed(
            native, factor, self.scope.imaging.get_pixel_alignment()
        )
        return self._apply_frame(native, target)

    @api
    def get_binning_size(self) -> int:
        """The binning factor in force: the one the camera took and the store holds.

        ``set_binning_size`` stores a factor only once the camera took it, so
        the store is the camera's binning; every pixel-size and tile-spacing
        reader asks here.
        """
        import modules.config_helpers as config_helpers

        return config_helpers.get_binning_from_settings(self.settings)

    @api
    def frame_at_binning(self, size: int) -> dict:
        """The frame ``set_binning_size(size)`` will ask the camera for; nothing is applied.

        The same arithmetic the apply uses, answered from the store and the
        camera's cached limits, so a display can show the new frame beside
        the new binning while the apply is still running. The camera's own
        answer (its grid, its minimum at the new binning) can differ; the
        stored frame after the apply is the truth.
        """
        return self._target_frame(self._native_frame(), size)

    def _native_frame(self) -> dict:
        """The stored unbinned region, or one rebuilt from the displayed frame.

        The stored pair is returned as it is, never re-capped: a small
        reading during a reconnect would otherwise shrink the stored region
        for good. Settings saved before the pair existed hold only the
        displayed size, so the region is rebuilt as displayed x stored
        binning, capped at the largest frame the scope delivers.
        """
        frame = self.settings['frame']
        if 'native_width' in frame and 'native_height' in frame:
            native = {'width': int(frame['native_width']), 'height': int(frame['native_height'])}
            source = 'stored'
        else:
            factor = binning.binning_size_str_to_int(self.settings['binning']['size'])
            displayed = {'width': int(frame['width']), 'height': int(frame['height'])}
            maximum = self.scope.capabilities.camera_max_frame_size
            cap = (
                {'width': maximum[0], 'height': maximum[1]}
                if maximum
                else {
                    'width': displayed['width'] * factor,
                    'height': displayed['height'] * factor,
                }
            )
            native = binning.displayed_to_native(displayed, factor, cap)
            source = f'rebuilt from {displayed["width"]}x{displayed["height"]} at {factor}x'
        # Whether the region came from the store or was rebuilt, and from
        # what: a rebuild against the wrong binning is how the region once
        # drifted, so the inputs stay in the log.
        logger.info(f'[Session  ] native frame: {source} -> {native["width"]}x{native["height"]}')
        return native

    def _target_frame(self, native: dict, factor: int) -> dict:
        """The displayed frame a stored ``native`` region gives at a new ``factor``.

        The region divided by the binning and floored to even sides, so it
        follows from the region alone. It is raised to the camera's
        minimum: a Pylon camera floors only to its maximum and refuses a
        smaller request outright, and no one asked for this frame -- a
        binning change derives it, so there is no request to refuse. A frame
        a person asks for is refused out of range instead (``set_frame_size``).
        """
        imaging = self.scope.imaging
        target = binning.native_to_displayed(native, factor, imaging.get_pixel_alignment())
        minimum = imaging.min_frame_size_cached
        if minimum is not None:
            target = {
                'width': max(target['width'], minimum['width']),
                'height': max(target['height'], minimum['height']),
            }
        return target

    def _store_delivered_geometry(self) -> None:
        """Store the binning and frame bring-up's camera took, as the Session's writers do after an apply.

        Bring-up applies the stored binning and frame to the camera directly.
        A binning the camera does not offer is replaced by the one it
        reports, and a stored frame above the scope's maximum is refitted to
        it, so what the camera delivers can differ from what was stored.
        Storing the delivered pair keeps the settings, the frame fields and
        every reader of them on the geometry the camera actually holds. The
        two are stored together because a frame beside a binning the camera
        is not at describes a region the sensor does not have, and the next
        frame or binning change is worked out from that pair. The native
        region is left as stored: it is the intent, and a small reading must
        not shrink it for good. With no camera connected nothing was
        delivered, so nothing is stored.
        """
        if not self.scope.camera_connected:
            return
        imaging = self.scope.imaging
        delivered = imaging.frame_size_cached
        label = binning.binning_size_int_to_str(imaging.get_binning_size())
        with self.settings_lock:
            stored_label = self.settings['binning']['size']
            frame = self.settings['frame']
            stored = {'width': frame['width'], 'height': frame['height']}
            if stored == delivered and stored_label == label:
                return
            self.settings['binning']['size'] = label
            frame['width'] = int(delivered['width'])
            frame['height'] = int(delivered['height'])
        logger.info(
            f'[Session  ] stored {stored_label} {stored["width"]}x{stored["height"]}; the camera '
            f'delivers {label} {delivered["width"]}x{delivered["height"]} -- the delivered '
            f'binning and frame are stored'
        )

    def _apply_frame(self, native: dict, target: dict) -> 'dict | None':
        """Apply the displayed ``target``, then store it and the ``native`` region it came from."""
        imaging = self.scope.imaging
        if target == imaging.frame_size_cached:
            delivered = target
        else:
            delivered = imaging.set_frame_size(target['width'], target['height'])
            if delivered is None:
                return None
        with self.settings_lock:
            frame = self.settings['frame']
            frame['native_width'] = int(native['width'])
            frame['native_height'] = int(native['height'])
            frame['width'] = int(delivered['width'])
            frame['height'] = int(delivered['height'])
        return dict(delivered)

    # --- Hardware commands: NOT forwarded ---
    # The Session surface deliberately carries no hardware-command
    # forwarders. L2 callers reach hardware through the composition
    # root the Session exposes -- session.scope.illumination.*,
    # session.scope.motion.*, session.scope.imaging.*,
    # session.scope.runtime_state.* -- so every command has exactly one
    # public spelling and the Session owns only what is session-scoped:
    # lifecycle (create / shutdown /
    # start_application_session / start_metrics / stop_metrics), the
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
        """
        from modules.protocol_runner import ProtocolRunner

        if self._protocol_runner is None:
            self._protocol_runner = ProtocolRunner(session=self)
        return self._protocol_runner

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @api(in_process=True)
    def start_metrics(self) -> None:
        """Start the session's periodic metrics logging.

        Uses the session's scheduler and the
        ``settings.profiling.metrics_interval_s`` cadence override when
        present. Metrics stay opt-in by the call itself: headless hosts
        simply never call this.

        Raises:
            RuntimeError: Metrics are already running. A second
                ``MetricsLogger.start`` silently overwrites its
                schedule handles and orphans the first set as
                untracked, forever-ticking events -- so a double start
                refuses loudly instead.
        """
        if self._metrics_started:
            raise RuntimeError(
                'ScopeSession.start_metrics: metrics are already running; '
                'a second start would orphan the existing schedule handles'
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
        interval_s = self.settings.get('profiling', {}).get('metrics_interval_s')
        if interval_s is not None:
            start_kwargs['system_metrics_interval_s'] = float(interval_s)
        elif self.engineering_mode:
            start_kwargs['system_metrics_interval_s'] = ENGINEERING_METRICS_INTERVAL_S
        self.metrics_logger.start(self._scheduler, **start_kwargs)
        self._metrics_started = True

    @api(in_process=True)
    def stop_metrics(self) -> None:
        """Stop the session's periodic metrics logging. Idempotent.

        Never shuts the scheduler itself down -- a shut scheduler
        refuses all future schedules, and the host may start metrics
        again with the same instance.
        """
        if not self._metrics_started:
            return
        self._metrics_started = False
        self.metrics_logger.stop()

    @api(in_process=True)
    def shutdown(self) -> None:
        """Tear down everything this session constructed.

        The bundle the session holds always stops: its long-lived consumer
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

        A scope passed in is left connected with its lanes running: it is
        the caller's. A second call is a logged no-op; a call that raised
        part-way can be called again.

        Raises:
            ScopeDisconnectError: a part of an owned scope did not shut down
                cleanly; every teardown step has still run, and a second
                call completes.
        """
        if self._shut_down:
            logger.info('[Session  ] shutdown() called again -- nothing to do')
            return
        # Plugins first, while everything they hold still runs: an
        # unregister may stop its own run and wait for it.
        self.unload_plugins()
        self.stop_metrics()
        # Settle any run's merge outcome FIRST. The executor teardown below
        # does not wait for the file lanes to drain, so a merge still
        # waiting on this run's writes can never finish -- and a caller
        # blocked on the result would wait out its whole bound for an
        # answer that is no longer coming.
        runner = self.sequenced_capture_runner
        if runner is not None:
            # The fallback is used only when the run never reached
            # cleanup and so recorded no ending of its own; a run that
            # already reported one keeps it, and 'shutdown' says only
            # that the merge is what the teardown cut short.
            runner.settle_unfinished_run(
                'shutdown',
                fallback=RunEnding(
                    'aborted',
                    'shutdown',
                    'Session Shutdown',
                    'The session shut down before the run reported.',
                ),
            )
            # A finished run's images still being written get a bounded
            # chance to land before the lanes go down below; whatever is
            # still outstanding then is given up on and counted, never
            # cleared silently with the lane's queue. A run still live has
            # not closed its writes, so nothing can complete them during a
            # wait: they are given up on at once.
            batch = runner.write_batch()
            if (
                batch is not None
                and batch.outcome is None
                and not (batch.draining and batch.wait_complete(_SHUTDOWN_RUN_FILES_WAIT_S))
            ):
                batch.abandon('Session shutdown')
        # The session owns its scheduler: a session over a caller's scope
        # still ends its own timers (a live health check outliving the
        # session would fire into torn-down state).
        self._scheduler.shutdown()
        if self.autofocus_thread is not None:
            self.autofocus_thread.stop(timeout=2.0)
        if self._owns_scope and self.io_executor.worker_alive:
            # Drain the LEDs through the io lane BEFORE the lanes go down,
            # on the same serial-bus lane as every other LED write, so it
            # cannot race an in-flight LED task the way a bare thread did.
            # Only while the lane's worker is alive: a submission to a lane
            # with no worker is never serviced, and the wait would run out
            # its bound for nothing -- disconnect() below turns the LEDs
            # off inline either way. The scope's own off asks presence
            # first, so with no LED controller it writes nothing. The 2 s
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
                        states = self.scope.illumination.get_led_states()
                        lit = sorted(c for c, s in states.items() if s.get('enabled'))
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
        # Last, so a listener hears what the teardown itself reported.
        for listener in list(self._outcome_listeners):
            self.remove_outcome_listener(listener)
        self._shut_down = True

    @api(in_process=True)
    def begin_application_session(
        self,
        *,
        disable_homing: bool = False,
        home_fn: Callable[[str], object] | None = None,
        turret_fn: Callable[[int], object] | None = None,
    ) -> concurrent.futures.Future[None]:
        """Start the standard startup motion without waiting for it; returns its Future.

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
        claim is released before the Future settles, so a caller told the
        motion has ended finds the scope free.

        A failed home is reported (``report_outcome``) and ends the motion
        without the turret move -- an absolute move against a reference the
        home did not establish; the Future settles with None. A refusal that
        is not for the hardware's state is a defect, and settles the Future
        with it.

        ``disable_homing=True`` skips every step: no startup motion on any
        axis. The turret is left where it is, like the stage axes, in no
        known slot -- positioning it without a home would be an absolute
        move against a reference the caller asked us not to establish. The
        skip is the requested behaviour, so it is logged, not signalled.

        A scope whose model has no motor board (``scope.motion_expected``
        False) has nothing to home, so it issues no startup motion either;
        nor does one whose motor board bring-up recorded as missing, which
        bring-up's own report has already named. A skip takes nothing and
        returns a settled Future.

        Args:
            disable_homing: If True, issue no startup motion at all.
            home_fn: Callable taking an axis name that homes it and raises
                as ``MotionAPI.home`` does. Defaults to the motion API.
            turret_fn: Callable taking a turret position. Defaults to
                the motion API.

        Raises:
            HardwareCommandRefusedError: ``'home_in_flight'`` or
                ``'exclusive_activity_running'``: something already holds
                the scope. Nothing was taken.
        """
        settled: concurrent.futures.Future[None] = concurrent.futures.Future()
        # Marked running before anyone else holds it, so only the lane
        # settles it and a caller's cancel() cannot release the claim.
        settled.set_running_or_notify_cancel()
        if disable_homing:
            logger.info('startup motion skipped: homing disabled; the turret is left where it is')
            settled.set_result(None)
            return settled
        if not self.scope.motion_expected:
            logger.info('startup motion skipped: this scope model has no motor board')
            settled.set_result(None)
            return settled
        from modules.lumascope_api.bring_up import MOTOR

        # Bring-up has already reported the missing board once, in its own
        # report; homing it would only have the home refused as "not
        # connected" and show the same absence a second time.
        if self.scope.bring_up_record().part(MOTOR).missing:
            logger.info('startup motion skipped: the motor board did not come up at bring-up')
            settled.set_result(None)
            return settled

        if home_fn is None:
            home_fn = self.scope.motion.home
        if turret_fn is None:
            turret_fn = lambda position: self.scope.motion.move_turret(position)

        @slow_task_budget(STARTUP_MOTION_SLOW_TASK_S)
        def startup_motion() -> None:
            self._startup_motion(home_fn, turret_fn)

        # The whole motion is one home's taking, so no request lands between
        # its moves; the diagnostics worker carries it, because the moves
        # themselves wait on the io lane.
        body, release_if_unrun, taking = self.scope.motion.claim_home(startup_motion)
        try:
            with acting(taking):
                self.executor_bundle.diagnostics_executor.submit(
                    IOTask(action=body), 'begin_application_session', waiter=settled
                )
        except BaseException:
            release_if_unrun()
            raise
        settled.add_done_callback(lambda _settled: release_if_unrun())
        return settled

    @api(in_process=True)
    def start_application_session(
        self,
        *,
        disable_homing: bool = False,
        home_fn: Callable[[str], object] | None = None,
        turret_fn: Callable[[int], object] | None = None,
    ) -> None:
        """Run the standard startup motion, and wait for it.

        ``begin_application_session``, waited for: returns when the startup
        motion has ended and the scope is free, and raises what its Future
        would. Headless / REST callers use this exact call to apply the
        standard startup orchestration without copy-pasting from the App.

        Raises:
            RuntimeError: called from an executor's worker. The motion runs
                on the diagnostics worker and waits on the io lane, so a
                worker waiting for it can be waiting on itself.
            HardwareCommandRefusedError: as ``begin_application_session``.
        """
        refuse_blocking_on_a_worker('start_application_session')
        self.begin_application_session(
            disable_homing=disable_homing, home_fn=home_fn, turret_fn=turret_fn
        ).result()

    def _startup_motion(
        self, home_fn: Callable[[str], object], turret_fn: Callable[[int], object]
    ) -> None:
        """The startup motion's steps, run under its taking on the diagnostics worker."""
        # Wait for the home's result and honor it. Turret positioning is
        # an absolute move against the reference frame the home was
        # supposed to establish; running it after a failed home is the
        # secondary cascade users report -- a second error on top of the
        # home's own, for motion that could never have been correct.
        try:
            home_fn('ALL')
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
