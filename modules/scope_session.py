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

import contextlib
import copy
import dataclasses
import json
import os
import threading
import time
import typing
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

import modules.app_context as _app_ctx
import modules.settings_init as settings_init
from modules import binning, common_utils, image_mode
from lvp_logger import logger
from modules.activity_claim import SCOPE_HOLDING_KINDS, ActivityClaim, HeldClaim, acting
from modules.common_utils import CustomJSONizer
from modules.exceptions import (
    CameraSettingUnsupportedError,
    ConfigError,
    DiagnosticRefusedError,
    FileWriterNotStuckError,
    HardwareCommandRefusedError,
    HomingFailedError,
    ObjectiveUnknownError,
    ScopeDisconnectError,
    ScopeModelUnknownError,
    SettingsSaveRefusedError,
)
from modules.manual_capture import ManualCaptureController
from modules.manual_recording import ManualRecordingController
from modules.metrics_logger import ENGINEERING_METRICS_INTERVAL_S, MetricsLogger
from modules.run_outcome import RunEnding
from modules.scheduler import Scheduler, ThreadingTimerScheduler

# How long a diagnostic's end waits for a run it lent its claim to. The
# window of one autofocus inside a characterization. Per
# PERFORMANCE_BUDGETS.md row diagnostic_exit_run_idle_wait_s.
DIAGNOSTIC_EXIT_RUN_IDLE_WAIT_S = 120.0

# How long shutdown lets a finished run's images finish writing before it
# gives up on them and takes the file lane down. Budget row:
# shutdown_run_files_wait_s in PERFORMANCE_BUDGETS.md.
_SHUTDOWN_RUN_FILES_WAIT_S = 10.0

# ProtocolRunner is referenced only in a return annotation; it is
# imported function-locally to avoid a circular import. Declare it here
# for the annotation without a runtime import.
if TYPE_CHECKING:
    from modules.labware_loader import WellPlateLoader
    from modules.objectives_loader import ObjectiveLoader
    from modules.protocol import Protocol, ProtocolSizeAdvisory
    from modules.protocol_runner import ProtocolRunner
    from modules.sequential_io_executor import SequentialIOExecutor


def _scheduler_callback_error(exc: BaseException) -> None:
    """A scheduled callback died on its timer thread; say so loudly.

    The scheduler's default is to swallow the exception, which is the
    wrong default here: the callbacks the session schedules include the
    recording health check, whose entire purpose is loud failure.
    """
    logger.error('[ScopeSession] scheduled callback raised', exc_info=exc)


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


class ScopeSession:
    """Owns the shared, GUI-independent state for one microscope session."""

    def __init__(
        self,
        settings: dict,
        scope,
        executor_bundle,
        protocol_thread=None,
        autofocus_runner=None,
        autofocus_thread=None,
        z_ui_update_func=None,
        owns_scope: bool = False,
        scheduler: Scheduler | None = None,
        settings_saved_hook=None,
        engineering_mode: bool = False,
    ):
        self.settings = settings
        # The lock lives with the dict it guards. Every host hands the same
        # dict to whatever else it composes, so a lock held anywhere else
        # can only guard one of the aliases -- which is no guard at all.
        # Readers on other threads take a snapshot; writers use
        # update_settings.
        self.settings_lock = threading.Lock()
        # Fired after a successful save, with the snapshot that was
        # written. The GUI passes its plugin notifier; a headless host
        # passes nothing, because there is no plugin registry to notify.
        self._settings_saved_hook = settings_saved_hook
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
        from modules import coord_transformations

        # Stateless: several instances are not several stores, so the
        # session keeps its own for the plate<->stage conversions it serves.
        self.coordinate_transformer = coord_transformations.CoordinateTransformer()
        # The mode this session was built in, never written afterwards. The
        # GUI's live flag lives on its own context and is flipped by a
        # plugin after the session exists, so a GUI run passes that flag
        # itself; this is the store a headless run reads, the only one such
        # a process has.
        self.engineering_mode = engineering_mode
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
        # Run-state listeners: zero-argument callables notified on every
        # run-state transition edge (claim grant/release, a run's return
        # to IDLE after its cleanup, file-drain exit). They fire on the TRANSITIONING thread,
        # possibly under engine locks, so a listener must only schedule
        # or re-read the level-derivation properties below -- never
        # acquire engine locks or trust edge context.
        self._run_state_listeners: list = []
        # The single arbitration point for exclusive activities: a
        # protocol run and a video recording each claim here before
        # committing, so the two can never run concurrently. Enforcement
        # lives with the claimants (the sequenced-capture runner's
        # refusal gate and the recording engine's start), which take
        # this handle by injection.
        self.activity_claim = ActivityClaim(on_transition=self.notify_run_state)
        # The device lanes ask the claim before running work, so while a run
        # or a diagnostic holds the scope only its own work reaches the
        # hardware, whoever submits. The IO key is kept for the one named
        # override on that lane, shutdown's LED drain; the camera key goes to
        # the scope for its temperature read, so the lanes ask before the
        # scope is serviced.
        self._io_override_key = self.io_executor.ask_claim(self.activity_claim)
        self._camera_override_key = self.camera_executor.ask_claim(self.activity_claim)
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
            engineering_mode=self.engineering_mode,
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
        self.z_ui_update_func = z_ui_update_func
        from modules.sequenced_capture_runner import SequencedCaptureRunner

        self.sequenced_capture_runner = SequencedCaptureRunner(
            scope=scope,
            stage_offset=settings.get('stage_offset', {}),
            protocol_thread=self.protocol_thread,
            file_io_executor=self.file_io_executor,
            autofocus_thread=autofocus_thread,
            autofocus_runner=autofocus_runner,
            z_ui_update_func=z_ui_update_func,
            activity_claim=self.activity_claim,
            on_run_idle=self.notify_run_state,
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

        The camera override key lives on the scope but belongs to the
        session's composition. Construction comes through here so no scope
        the session drives can be left un-serviced -- the bring-up steps are
        spelled out exactly once.
        """
        scope.set_camera_override_key(self._camera_override_key)

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
        mid-sweep. If the run is still live after the wait, the claim is
        kept with it, as a stuck run keeps its own claim, and this raises.

        Yields:
            The held claim.

        Raises:
            DiagnosticRefusedError: A run, a recording or another
                diagnostic holds the scope. Nothing was taken.
            RuntimeError: At the block's end, a run under this claim was
                still live after the wait; the claim stays held.
        """
        held = self.activity_claim.try_claim('diagnostic')
        if held is None:
            holder = self.activity_claim.holder
            kind = holder.kind if holder is not None else None
            # The holder can release between the failed take and this read;
            # the refusal still stands, it just cannot name who refused it.
            named = f'A {kind} activity' if kind else 'Another exclusive activity'
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
                raise RuntimeError(
                    'diagnostic_claim: a run under this claim is still live after '
                    f'{DIAGNOSTIC_EXIT_RUN_IDLE_WAIT_S:.0f} s; the claim stays held with it'
                )
            held.release()

    @property
    def is_protocol_running(self) -> bool:
        """True while a protocol-class run holds the exclusive claim.

        Scans, full protocols, zstacks, and autofocus runs all hold the
        'protocol' claim, so all read True here. The claim releases at
        run-cleanup end; the post-run file drain is visible on
        run_lockout / protocol_files_draining, not here.
        """
        return self.activity_claim.owner == 'protocol'

    # ------------------------------------------------------------------
    # Run-state facts and derivations
    #
    # Each FACT has exactly one owner (the claim, the recording engine,
    # the file writer, the scope config); everything a consumer needs is
    # a synchronous DERIVATION over them. All reads are lock-free
    # attribute/queue reads, so these properties are safe from any
    # thread, including inside a transition listener.
    # ------------------------------------------------------------------

    @property
    def exclusive_activity(self) -> 'str | None':
        """The current exclusive-activity owner: None, 'protocol',
        'recording', or 'diagnostic'."""
        return self.activity_claim.owner

    @property
    def close_drain_pending(self) -> bool:
        """True while either video drain still holds queued frames.

        What a close would interrupt on the video side, in one read: a
        manual recording's own drain, or a finished run's video-step
        tail. A closing host needs both, and asking it to OR them itself
        puts the derivation somewhere headless and REST cannot reach.

        True for a LIVE recording too, since its frames are also
        outstanding -- a caller that needs "still capturing" specifically
        wants ``manual_recording.is_recording``, which is the narrower fact.
        """
        return self.manual_recording.is_busy or self.sequenced_capture_runner.video_drain_busy

    @property
    def protocol_files_draining(self) -> bool:
        """True from the end of a run until its last file is written.

        False while the run is live -- the run's own state answers then --
        and once its files are all on disk or given up on.
        """
        batch = self.sequenced_capture_runner.write_batch()
        return batch is not None and batch.draining

    @property
    def protocol_files_pending(self) -> int:
        """How many of a finished run's file writes are still to finish, the
        one in flight included; 0 when nothing is draining."""
        batch = self.sequenced_capture_runner.write_batch()
        return batch.pending if batch is not None and batch.draining else 0

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

    @property
    def protocol_files_stuck_write(self) -> str:
        """The write in flight on the file lane, named for a stall report."""
        return self.file_io_executor.describe_running_task()

    @property
    def run_lockout(self) -> bool:
        """True while a run, a diagnostic, or a run's post-run file drain
        owns the scope.

        The drain term encodes a deliberate asymmetry: a finished
        protocol frees its claim while its files drain, but the control
        surface stays locked until the queue empties.
        """
        return self.activity_claim.owner in SCOPE_HOLDING_KINDS or self.protocol_files_draining

    @property
    def controls_locked(self) -> bool:
        """True while the full control surface locks: any run lockout,
        or a LIVE manual recording (a draining recording frees the
        controls while its claim still refuses new runs)."""
        return self.run_lockout or (
            self.activity_claim.owner == 'recording' and self.manual_recording.is_recording
        )

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

    def add_run_state_listener(self, listener) -> None:
        """Register a run-state transition listener and level-sync it.

        The immediate call is the level republish: transitions are
        edges, and a listener registered after a grant would otherwise
        never see it.
        """
        self._run_state_listeners.append(listener)
        listener()

    def notify_run_state(self) -> None:
        """Notify every run-state listener (level semantics: listeners
        re-read the derivations; an extra notification is harmless)."""
        for listener in list(self._run_state_listeners):
            try:
                listener()
            except Exception:
                logger.exception('[ScopeSession] run-state listener failed')

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        settings: dict,
        source_path: str | None = None,
        scope: object | None = None,
        *,
        simulate: bool = False,
        warn_pre_release: bool = True,
        ui_dispatcher: Callable[[Callable, float], Any] | None = None,
        af_ui_update_func: Callable[[float], None] | None = None,
        settings_saved_hook: Callable[[dict], None] | None = None,
        engineering_mode: bool = False,
        display_ctx_provider: Callable[[], Any] | None = None,
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
            ui_dispatcher: ``schedule_once(func, dt)``'s shape; the four
                lanes marshal their callbacks through it. None runs them
                inline on the worker. Refused beside ``scope``: a scope's
                lanes marshal through the dispatcher it was built with, so
                pass it to the ``Lumascope`` instead.
            af_ui_update_func: ``(pos) -> None``; the autofocus runner's
                ``ui_update_func`` and the capture engine's
                ``z_ui_update_func`` -- one callable, both consumers.
            settings_saved_hook: called with the snapshot after a
                successful ``save_settings``.
            engineering_mode: stored on the session as the mode it was
                built in.
            display_ctx_provider: the display thread's context provider
                (host-only: the GUI's app context; None for a host with
                no display).
        """
        from modules.lumascope_api._lumascope import _fire_pre_release_warning
        from modules.path_utils import get_source_root

        if scope is not None and source_path is not None:
            raise ValueError(
                'ScopeSession.create: source_path is refused beside a scope -- the '
                'session reads its data folder and catalogues from the scope, so pass '
                'the folder to Lumascope(source_path=...) instead'
            )
        if scope is not None and ui_dispatcher is not None:
            raise ValueError(
                'ScopeSession.create: ui_dispatcher is refused beside a scope -- the '
                "scope's lanes marshal through the dispatcher it was built with, so "
                'pass it to Lumascope(ui_dispatcher=...) instead'
            )
        if warn_pre_release:
            _fire_pre_release_warning()

        built_scope = False
        if scope is None:
            import modules.lumascope_api as lumascope_api

            scope = lumascope_api.Lumascope(
                simulate=simulate,
                warn_pre_release=warn_pre_release,
                configured_model=settings.get('microscope'),
                sim_tier=cls._simulator_tier(settings) if simulate else 'fast',
                ui_dispatcher=ui_dispatcher,
                fx2_debug_wire=settings['fx2_debug_wire_enabled'],
                source_path=get_source_root(source_path),
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
            ui_dispatcher=ui_dispatcher,
            ctx_provider=display_ctx_provider,
        )

        # Service registration (the camera override key) happens in
        # __init__ for every session-composed scope -- nothing here.

        autofocus_runner, autofocus_thread = cls._build_autofocus_pair(
            scope=scope,
            ui_update_func=af_ui_update_func,
        )

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
                z_ui_update_func=af_ui_update_func,
                owns_scope=built_scope,
                settings_saved_hook=settings_saved_hook,
                engineering_mode=engineering_mode,
            )
        except BaseException:
            # No session exists to tear down -- a scope another session holds
            # refuses a second claim -- so stop what this factory started.
            autofocus_thread.stop(timeout=2.0)
            executor_bundle.shutdown()
            if built_scope:
                cls._report_teardown_failure(scope.disconnect)
            raise
        if built_scope:
            cls._bring_up(session)
        return session

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
        missing = runtime_missing(DEFAULT_DIALECT) if tier == 'firmware' else None
        if missing is not None:
            logger.warning(
                '[Session  ] simulator_tier is firmware, but there is no MicroPython runtime '
                f'here ({missing}): running the fast tier'
            )
            return 'fast'
        return tier

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
            ConfigError: ``source_path`` holds no shipped template, or the
                user's ``current.json`` is unusable.
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
    def _bring_up(cls, session: 'ScopeSession') -> None:
        """Configure the scope a factory built; ``initialize`` releases the
        camera start gate last, after the capture pixel format. A raise
        anywhere in here leaves the caller with no session object to tear
        down, so this tears down what the factory started before it lets
        the raise out."""
        try:
            session.configure_scope()
        except BaseException:
            cls._report_teardown_failure(session.shutdown)
            raise
        # The one marker for "the camera is grabbing and the session is
        # up": a host measures its own consumer's start against it.
        logger.info('[Session  ] bring-up complete: scope configured, camera streaming')

    @staticmethod
    def _build_autofocus_pair(*, scope, ui_update_func=None):
        """Real AF runner + started AF thread for a factory-built session,
        so every host gets the same wiring; ``ui_update_func`` is the
        host's Z-position renderer, None for a host with no display."""
        from modules.autofocus_runner import AutofocusRunner
        from modules.autofocus_thread import AutofocusThread

        autofocus_runner = AutofocusRunner(scope=scope, ui_update_func=ui_update_func)
        autofocus_thread = AutofocusThread(afe=autofocus_runner)
        autofocus_thread.start()
        return autofocus_runner, autofocus_thread

    # ------------------------------------------------------------------
    # Convenience wrappers (delegate to config_helpers / scope_commands)
    # ------------------------------------------------------------------

    def recover_file_writer(self) -> int:
        """Give up on a finished run's unwritten images and unlock a stuck writer.

        L2 counterpart of the GUI's stalled-writer recovery: when a
        protocol run's file writer stops making progress, every
        subsequent run is refused with the ``files_writing_stalled``
        reason until the writer is recovered or the app restarts. This
        method is that recovery for headless / REST / SDK callers: the
        run's outstanding images are given up on and counted, and the
        worker stuck mid-write is abandoned and replaced. Nothing else
        queued on the lane is discarded.

        Returns:
            How many of the run's images were given up on.

        Raises:
            HardwareCommandRefusedError: a run or a diagnostic holds the
                scope. The outstanding writes are that run's own captures.
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

    def get_stim_configs(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_stim_configs(self.settings)

    def get_enabled_stim_configs(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_enabled_stim_configs(self.settings)

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

        The entry point a caller with no GUI uses to assemble the config a
        run takes. Tiling and z-stacking are arguments rather than stored
        settings: neither survives a restart, so there is nothing for a
        session to read them from and a caller states what it wants.

        The GUI builds the same config through the same builder, supplying
        these two from its widgets.
        """
        import modules.config_helpers as config_helpers

        return config_helpers.get_sequenced_capture_config_from_settings(
            self.capture_settings_snapshot(),
            objective_helper=self.objective_helper,
            wellplate_loader=self.wellplate_loader,
            tiling=tiling,
            use_zstacking=use_zstacking,
        )

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

    def add_step(
        self,
        protocol: 'Protocol',
        *,
        before_step: int | None = None,
        after_step: int | None = None,
    ) -> list[str]:
        """Add a step to ``protocol`` from this session's settings and live position.

        The entry point a caller with no GUI uses to do what Add Step
        does: one step per layer whose ``acquire`` is set, at the current
        plate position, with the current objective, in the settings'
        channel order. The protocols API performs the add and refuses when
        nothing would be added; this composes its inputs from the session
        the same way the GUI's handler does.

        Returns the inserted step names, in protocol order.
        """
        # None when unknown: the protocols API refuses that by name, notified.
        objective_id = self.scope.runtime_state.get_current_objective_id()
        return self.scope.protocols.add_step(
            protocol,
            layer_configs=self.get_layer_configs(),
            stim_configs=self.get_stim_configs(),
            plate_position=self.get_current_plate_position(),
            objective_id=objective_id,
            channel_order=self.settings.get('step_channel_order', None),
            before_step=before_step,
            after_step=after_step,
        )

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
        does: step ``step_idx`` takes ``layer``'s settings, the current
        plate position and the current objective. The protocols API
        performs the update and refuses it when the position or the
        objective is unknown; this composes its inputs from the session
        the same way ``add_step`` does.

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
            plate_position=self.get_current_plate_position(),
            objective_id=objective_id,
            label=label,
        )

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

    def get_settings_snapshot(self) -> dict:
        """A deep copy of the settings dict, taken under the lock.

        A worker thread takes one of these at task entry and reads from it
        for the rest of the task, rather than reading a dict another
        thread may be part-way through rewriting.
        """
        with self.settings_lock:
            return copy.deepcopy(self.settings)

    def update_settings(self, key: str, value: object) -> None:
        """Write one top-level settings key under the lock.

        The write path for any caller that is not on the host's own
        thread. Reads may go straight to `settings`; a write that skips
        this can tear a snapshot being taken concurrently.
        """
        with self.settings_lock:
            self.settings[key] = value

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
            ScopeModelUnknownError: the catalogue does not list ``model``;
                nothing is saved.
            ConfigError: the catalogue itself has no usable ``Models``
                section.
        """
        from modules import layer_record

        scope_models = layer_record.load_scope_models()
        if model not in scope_models:
            raise ScopeModelUnknownError(model, scope_models)
        self.update_settings('microscope', model)
        logger.info(f'[Session  ] scope model {model!r} saved; it applies at the next start')

    def configure_scope(self) -> None:
        """Configure the scope from this session's settings -- the bring-up.

        Once, after construction, on a real scope: adopt the model the
        hardware reports when the catalogue knows it (a WRITE into this
        session's ``settings['microscope']`` -- hardware truth outranks
        the stored selection; a model outside the catalogue, or no motor
        board to ask, leaves the stored one), normalize the turret slot
        keys a caller-supplied dict may still carry as JSON strings,
        resolve the model's catalogue entry, select the labware, build the
        init config and run ``Lumascope.initialize`` -- which selects the
        stored objective on a scope with no turret; on a turreted scope the
        objective stays unknown until the turret is in a known slot. The
        factories run this for the scope they build; a host that constructs
        the session directly, or hands ``create`` its own scope, calls it
        once itself. Every step runs on the calling thread; nothing here
        dispatches.

        Raises:
            ConfigError: a settings key ``initialize`` cannot do without is
                missing (``frame``; ``objective_id`` on a scope with no
                turret); that ``objective_id`` names no shipped objective;
                a data file a helper needs is absent or
                unreadable (``labware.json``, ``objectives.json``); or the
                model catalogue has no usable ``Models`` section.
            HardwareCommandRefusedError: a run, a diagnostic or a recording
                holds the scope. The configuration rewrites the LEDs, the
                camera geometry and acceleration under whatever holds it,
                and its writes run inline, where no lane refuses them.
        """
        import modules.config_helpers as config_helpers
        from modules import layer_record
        from modules.scope_init_config import ScopeInitConfig

        holder = self.activity_claim.holder
        if holder is not None:
            raise HardwareCommandRefusedError(
                'exclusive_activity_running', 'configure_scope', holder.kind
            )
        # The catalogue first: its refusal must land before anything below
        # mutates the caller's dict.
        scope_models = layer_record.load_scope_models()
        # The hardware's own model outranks the stored selection, and it
        # has to land before the two reads of the selection below, or a
        # unit whose file says the wrong model configures for the wrong
        # axes. The motor driver caches its identity at connect, so the
        # read is synchronous; no board (or a board with no model) reports
        # None and the stored selection stands.
        detected = self.scope.diagnostics.get_microscope_model()
        stored = self.settings.get('microscope')
        if detected is not None and detected in scope_models and detected != stored:
            self.update_settings('microscope', detected)
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
        scope_config = scope_models.get(self.settings.get('microscope'))
        _labware_id, labware = config_helpers.get_selected_labware_from_settings(
            self.settings, self.wellplate_loader
        )
        config = ScopeInitConfig.from_settings(
            self.settings,
            labware,
            scope_config=scope_config,
            layer_identity=self.scope.layer_identity,
            turreted=self.scope_has_turret(),
        )
        self.scope.initialize(config)
        # Read once so a session that changes nothing still records the
        # scale it starts with (the read records the optics). On a turreted
        # scope the slot is not known until the turret is homed, so the
        # record says that instead.
        if self.scope.runtime_state.get_current_objective() is None:
            logger.info(
                '[Session  ] objective at bring-up: unknown until the turret is in a known slot'
            )

    def settings_are_provisional(self) -> bool:
        """Is the app running on defaults nobody has agreed to keep?

        True while the user's current.json could not be used and no one
        has decided its fate. While it holds, every save aimed at
        current.json raises SettingsSaveRefusedError -- resolve with
        retire_rejected_settings() after the user has chosen to start
        over.
        """
        return settings_init.settings_are_provisional()

    def retire_rejected_settings(self) -> 'str | None':
        """Resolve the provisional-settings state: retire the rejected file.

        Moves the unusable current.json aside (renamed, never deleted --
        it is the user's only copy) so a fresh one can take its place,
        and clears the provisional state so saves work again. Call only
        after a human has chosen to start over. Returns the retired
        path, or None when nothing was provisional.
        """
        return settings_init.retire_rejected_current_json()

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
            had_hardware = bool(
                scope and (scope.camera_connected or scope.motor_connected or scope.led_connected)
            )
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

        if self._settings_saved_hook is not None:
            try:
                self._settings_saved_hook(settings_snapshot)
            except Exception:
                logger.exception('[Session  ] save_settings: saved-hook failed')

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
        from modules import layer_record

        return config_helpers.model_has_turret(
            layer_record.load_scope_models(), self.scope.layer_identity.model
        )

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

    def select_objective(self, objective_id: str) -> bool:
        """Make ``objective_id`` the active objective. Returns whether it changed.

        The one writer of the active objective for every host. With no
        turret, the selected objective is the live store and moves with the
        settings copy that is persisted. On a turreted scope the active
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
            self.scope.runtime_state.set_objective(objective_id=objective_id)
            with self.settings_lock:
                self.settings['objective_id'] = objective_id
        return True

    # ------------------------------------------------------------------
    # The labware
    # ------------------------------------------------------------------

    def select_labware(self, labware_name: str) -> bool:
        """Make ``labware_name`` the current plate. Returns whether it changed.

        The one writer of the active labware for every host: the settings
        store and the scope's runtime state move together, or neither
        moves. Bring-up sets the plate from settings and offers no way
        back, so without this a caller that is not the GUI can start with
        a plate but never switch one.

        The name is stored in the catalogue's spelling: a plate renamed
        since a protocol or settings file named it is accepted under the old
        name and written under the key, so the settings store never carries
        a spelling the catalogue lacks, and whether the plate changed is
        decided on the key rather than on how it was spelled.

        Raises:
            ConfigError: ``labware_name`` is not a string, the loader
                cannot resolve the name, or the settings have no protocol block to hold the
                selection. Refused before either store is written: a write
                that half-lands leaves the settings store and the runtime
                state describing different plates, and every well
                position computed from the wrong one is silently wrong.
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
            # place. Named here, before either store moves: reaching into it
            # at the write below would raise with the runtime state already
            # changed, and a store that cannot hold the plate is not one to
            # write half of.
            raise ConfigError(
                'settings have no usable protocol block; the labware selection '
                f'has nowhere to live (found {type(protocol_settings).__name__})'
            )
        changed = labware_name != protocol_settings.get('labware')
        # Both stores are written even when the settings key already reads
        # the new name, because that key is not evidence about the scope.
        # Anything that writes it before calling here -- and the protocol
        # load does exactly that, one line before the spinner event that
        # reaches this member -- would otherwise make the selection look
        # finished and leave the runtime state on the previous plate. The
        # writes are idempotent; only the report of a change is not.
        labware = self.wellplate_loader.get_plate(plate_key=labware_name)
        # A holder is refused only a different plate: the GUI re-selects the
        # current one whenever its panels redraw, under any hold, and that
        # moves neither store.
        installed = self.scope.runtime_state.get_labware()
        if changed or installed is None or installed.config != labware.config:
            self._refuse_configuration_change_while_held('select_labware')
        self.scope.runtime_state.set_labware(labware=labware)
        with self.settings_lock:
            protocol_settings['labware'] = labware_name
        if changed:
            logger.info(f'[Session  ] Labware set to {labware_name!r}')
        return changed

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
        self.scope.runtime_state.set_turret_config(self.settings['turret_objectives'])

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
        self.scope.runtime_state.set_turret_config(self.settings['turret_objectives'])
        logger.info(
            f'[Session  ] Turret position {position} cleared; the active objective is now '
            f'{self.scope.runtime_state.get_current_objective_id()!r}'
        )

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

    def get_current_plate_position(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_current_plate_position(
            self.scope,
            self.settings,
            self.coordinate_transformer,
            self.wellplate_loader,
        )

    # ------------------------------------------------------------------
    # The camera's capture settings: applied, then stored
    # ------------------------------------------------------------------

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
            capture_depth, imaging.get_supported_pixel_formats()
        )
        if target is not None and not imaging.set_pixel_format(target):
            return False
        with self.settings_lock:
            self.settings['image_mode'] = mode
        return True

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
            The frame the camera delivers, ``{'width', 'height'}``. None when
            no camera is connected; nothing is stored.

        Raises:
            CameraSettingUnsupportedError: This camera does not offer
                ``size``. Nothing reaches the camera.
            CameraSettingRejected: The camera refused the binning (nothing is
                stored) or the frame after it (the binning is stored, with the
                frame the camera reports holding at it).
        """
        imaging = self.scope.imaging
        offered = imaging.get_available_binning_sizes()
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

    def set_frame_size(self, width: int, height: int) -> 'dict | None':
        """Frame the camera at ``width`` x ``height`` at the stored binning, and store it.

        The one writer of the frame for every host. The size is what the
        person sees and captures (post-binning); the unbinned region it
        implies is stored beside it. A size the camera already delivers is
        not written again.

        Returns:
            The frame the camera delivers, which may differ from the request
            by the camera's grid. None when no camera is connected; nothing
            is stored.

        Raises:
            CameraSettingOutOfRangeError: The size, floored to the camera's
                grid, is below the camera's minimum frame or above its sensor
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

        The stored pair is returned as it is, never re-capped against the
        live sensor size: a small reading during a reconnect would otherwise
        shrink the stored region for good. Settings saved before the pair
        existed hold only the displayed size, so the region is rebuilt as
        displayed x stored binning, capped at the sensor.
        """
        frame = self.settings['frame']
        if 'native_width' in frame and 'native_height' in frame:
            native = {'width': int(frame['native_width']), 'height': int(frame['native_height'])}
            source = 'stored'
        else:
            factor = binning.binning_size_str_to_int(self.settings['binning']['size'])
            displayed = {'width': int(frame['width']), 'height': int(frame['height'])}
            cap = self.scope.imaging.get_native_resolution() or {
                'width': displayed['width'] * factor,
                'height': displayed['height'] * factor,
            }
            native = binning.displayed_to_native(displayed, factor, cap)
            source = f'rebuilt from {displayed["width"]}x{displayed["height"]} at {factor}x'
        # Whether the region came from the store or was rebuilt, and from
        # what: a rebuild against the wrong binning is how the region once
        # drifted, so the inputs stay in the log.
        logger.info(f'[Session  ] native frame: {source} -> {native["width"]}x{native["height"]}')
        return native

    def _target_frame(self, native: dict, factor: int) -> dict:
        """The displayed frame a stored ``native`` region gives at a new ``factor``.

        The region divided by the binning and floored to the camera's grid,
        so it follows from the region alone. It is raised to the camera's
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
        # The engineering flag is read off the app context, not
        # settings['mode']: the engineering plugin can flip it during plugin
        # load, so the settings file and the live flag disagree on exactly the
        # machines that care. Plugin load completes before metrics start, and
        # an unset context (headless, REST, tests) reads as False -- production
        # cadence, which is the safe direction.
        start_kwargs = {}
        interval_s = self.settings.get('profiling', {}).get('metrics_interval_s')
        if interval_s is not None:
            start_kwargs['system_metrics_interval_s'] = float(interval_s)
        elif getattr(_app_ctx.ctx, 'engineering_mode', False):
            start_kwargs['system_metrics_interval_s'] = ENGINEERING_METRICS_INTERVAL_S
        self.metrics_logger.start(self._scheduler, **start_kwargs)
        self._metrics_started = True

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
        self.stop_metrics()
        # Settle any run's merge outcome FIRST. The executor teardown below
        # does not wait for the file lanes to drain, so a merge still
        # waiting on this run's writes can never finish -- and a caller
        # blocked on the result would wait out its whole bound for an
        # answer that is no longer coming.
        runner = self.sequenced_capture_runner
        if runner is not None:
            outcome = runner.run_outcome()
            if outcome is not None:
                # The fallback is used only when the run never reached
                # cleanup and so recorded no ending of its own; a run that
                # already reported one keeps it, and 'shutdown' says only
                # that the merge is what the teardown cut short.
                outcome.settle_unfinished(
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
            # off inline either way. The 2 s bound keeps the calling
            # thread from blocking on slow serial.
            logger.info('[Session  ] shutdown: leds_off through the io lane')
            try:
                from modules.sequential_io_executor import IOTask

                fut = self.io_executor.put(
                    IOTask(action=self.scope.illumination._leds_off_impl),
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
        self._shut_down = True

    def start_application_session(
        self,
        *,
        disable_homing: bool = False,
        home_fn: typing.Callable | None = None,
        turret_fn: typing.Callable | None = None,
    ) -> None:
        """Queue the standard startup home + turret-positioning sequence.

        The one implementation of the startup motion for every host:
        the App's launch and its reconnect handler once open-coded the
        same ALL-axis home + turret-positioning pair, and the two
        copies drifted.

        After this method returns, the io_executor has been told to:

        1. home ALL axes via ``move_home``. Firmware homes Z, T, X, Y in
           one routine; on Z-only boards it homes what it has and
           reports the missing axes.

        2. (when ``self.scope.capabilities.has_turret`` is True) move T
           to position 1; the active objective is then slot 1's
           assignment.

        ``disable_homing=True`` skips BOTH steps: no startup motion on
        any axis. The turret is left where it is, like the stage axes,
        in no known slot -- positioning it without a
        home would be an absolute move against a reference the caller
        asked us not to establish. The skip is the requested behaviour,
        so it is logged, not signalled.

        A scope whose model has no motor board (``scope.motion_expected``
        False) has nothing to home, so it issues no startup motion either.

        Headless / REST callers can use this exact same call to apply
        the standard startup orchestration without copy-pasting from
        the App.

        Args:
            disable_homing: If True, issue no startup motion at all.
            home_fn: Callable taking an axis name that homes it and raises
                as ``MotionAPI.home`` does. Defaults to the motion API.
            turret_fn: Callable taking a turret position. Defaults to
                the motion API.

        The two motion callables are injected the same way the metrics
        scheduler is: the hosting environment supplies its own, and the
        API default is what everything else gets. The Kivy app passes
        its home wrapper, which sets the window title during the home,
        and takes the API's turret move. Defaulting to the API instead
        of importing the UI is what lets a headless caller run this at
        all: a widget path reaches ``ctx.motion_settings``, which is
        None until a widget tree exists.
        """
        if disable_homing:
            logger.info('startup motion skipped: homing disabled; the turret is left where it is')
            return
        if not self.scope.motion_expected:
            logger.info('startup motion skipped: this scope model has no motor board')
            return

        if home_fn is None:
            home_fn = self.scope.motion.home
        if turret_fn is None:
            turret_fn = lambda position: self.scope.motion.move_turret(position)

        # Wait for the home's result and honor it. Turret positioning is
        # an absolute move against the reference frame the home was
        # supposed to establish; running it after a failed home is the
        # secondary cascade users report -- a second error on top of the
        # home's own, for motion that could never have been correct.
        try:
            home_fn('ALL')
        except (HomingFailedError, HardwareCommandRefusedError) as e:
            if isinstance(e, HardwareCommandRefusedError) and e.reason != 'not_connected':
                raise
            from modules.notification_center import notifications

            # A scope with no hardware gets one consolidated popup that
            # already covers the missing motor, so its home failure is
            # logged, not shown a second time.
            notifications.report_outcome(
                e,
                solicited=False,
                category='Motion',
                log_only=self.scope.no_hardware,
            )
            logger.error(
                'Homing did not succeed -- skipping startup turret '
                'positioning; the stage reference is unknown'
            )
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
