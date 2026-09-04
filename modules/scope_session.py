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
    # or, for headless / test use:
    session = ScopeSession.create_headless(settings=settings)
"""

import copy
import dataclasses
import json
import os
import threading
import time
import typing
from typing import TYPE_CHECKING

import modules.app_context as _app_ctx
import modules.settings_init as settings_init
from lvp_logger import logger
from modules.activity_claim import ActivityClaim
from modules.common_utils import CustomJSONizer
from modules.exceptions import SettingsSaveRefusedError, ConfigError
from modules.manual_recording import ManualRecordingController
from modules.metrics_logger import ENGINEERING_METRICS_INTERVAL_S
from modules.scheduler import Scheduler, ThreadingTimerScheduler

# ProtocolRunner is referenced only in a return annotation; it is
# imported function-locally to avoid a circular import. Declare it here
# for the annotation without a runtime import.
if TYPE_CHECKING:
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
        io_executor,
        camera_executor,
        wellplate_loader=None,
        coordinate_transformer=None,
        objective_helper=None,
        source_path: str = '.',
        executor_bundle=None,
        file_io_executor=None,
        protocol_thread=None,
        autofocus_runner=None,
        autofocus_thread=None,
        z_ui_update_func=None,
        owns_executors: bool = False,
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
        # start_metrics / stop_metrics / set_scope itself.
        self._metrics_started = False
        self.io_executor = io_executor
        self.camera_executor = camera_executor
        self.wellplate_loader = wellplate_loader
        self.coordinate_transformer = coordinate_transformer
        self.objective_helper = objective_helper
        self.source_path = source_path
        # The mode this session was built in, never written afterwards. The
        # GUI's live flag lives on its own context and is flipped by a
        # plugin after the session exists, so a GUI run passes that flag
        # itself; this is the store a headless run reads, the only one such
        # a process has.
        self.engineering_mode = engineering_mode
        # Every host hands its bundle in (the session re-registers it on
        # the scope at every rebind), so bundle-presence says nothing
        # about who owns the executor topology's teardown -- that fact is
        # owns_executors, passed True only by the factories that BUILT
        # the topology. Deriving ownership from the bundle would let a
        # host-composed session tear down its host's executors.
        self.executor_bundle = executor_bundle
        self._owns_executors = owns_executors
        # The canonical file-IO executor lives on the bundle; expose it here
        # alongside io_executor / camera_executor so callers (e.g. ProtocolRunner)
        # source the one shared FILE executor instead of constructing a
        # duplicate. A bundle-less host passes its handle in -- the session
        # cannot read the file-drain fact without it.
        self.file_io_executor = file_io_executor or (
            executor_bundle.file_io_executor if executor_bundle else None
        )
        # Service the scope NOW, after the handle derivations above and
        # before any collaborator is composed: a session-composed scope
        # must never exist un-serviced, or its dispatch falls back to
        # inline execution on the calling thread (unserialized, and a
        # protocol fence cannot reach an inline task).
        self._register_scope_services(scope)

        self.focus_round = 0
        # Run-state listeners: zero-argument callables notified on every
        # run-state transition edge (claim grant/release, file-drain
        # exit, scope rebind). They fire on the TRANSITIONING thread,
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
        # Manual video recording, composed with the session claim so a
        # recording and a protocol run are mutually exclusive for every
        # caller tier (GUI, L2, REST).
        self.manual_recording = ManualRecordingController(
            scope=scope,
            settings=settings,
            activity_claim=self.activity_claim,
            scheduler=self._scheduler,
        )
        if self.file_io_executor is not None:
            self.file_io_executor.add_protocol_idle_listener(self.notify_run_state)

        # The run engine and its autofocus pair are SESSION-composed:
        # one SequencedCaptureRunner per session, shared by the GUI,
        # ProtocolRunner, and (later) REST -- a second engine instance
        # would duplicate run state beside the shared claim. Hosts
        # inject their own AF pair / protocol thread; bundle-building
        # factories construct real ones; a bare session composes an
        # engine with what it has (an AF-bearing run then refuses or
        # fails loudly at the producer site).
        self.protocol_thread = protocol_thread or (
            executor_bundle.protocol_thread if executor_bundle else None
        )
        self.autofocus_runner = autofocus_runner
        self.autofocus_thread = autofocus_thread
        self.z_ui_update_func = z_ui_update_func
        from modules.sequenced_capture_runner import SequencedCaptureRunner

        self.sequenced_capture_runner = SequencedCaptureRunner(
            scope=scope,
            stage_offset=settings.get('stage_offset', {}),
            io_executor=io_executor,
            protocol_thread=self.protocol_thread,
            file_io_executor=self.file_io_executor,
            camera_executor=camera_executor,
            autofocus_thread=autofocus_thread,
            autofocus_runner=autofocus_runner,
            z_ui_update_func=z_ui_update_func,
            activity_claim=self.activity_claim,
            coordinate_transformer=coordinate_transformer,
            wellplate_loader=wellplate_loader,
        )
        self._protocol_runner = None
        # Refusals say busy-with-what: a recording refused by a running
        # run names the run's trigger through this lookup.
        self.manual_recording.run_trigger_lookup = self.sequenced_capture_runner.run_trigger_source

    def _register_scope_services(self, scope) -> None:
        """Register the session's services on a scope (the one bring-up).

        Executors, the executor bundle, and the protocol source path all
        live on the scope but belong to the session's composition; a
        scope missing them dispatches inline (unserialized, unfenceable)
        and its protocol constructors cannot resolve their data files.
        Construction and set_scope both come through here so no scope
        the session drives can be left un-serviced -- the bring-up steps
        are spelled out exactly once.

        The bundle is registered only when held: register_executor_bundle
        overwrites the metrics logger's bundle unconditionally, so a
        None-bundle call would blank a pre-wired scope's metrics wiring.
        """
        scope.register_executors(
            camera_executor=self.camera_executor,
            io_executor=self.io_executor,
            file_io_executor=self.file_io_executor,
        )
        if self.executor_bundle is not None:
            scope.register_executor_bundle(self.executor_bundle, settings=self.settings)
        scope.protocols.register_source_path(self.source_path)

    def set_scope(self, scope) -> None:
        """Rewire this session onto a NEW scope after a reconnect.

        The session and its recording controller each hold the scope by
        reference; left unrewired after a reconnect they keep driving
        the discarded, disconnected scope (start_application_session
        homes it; a recording captures from it). The new scope is
        serviced FIRST (executors, bundle, source path), before any
        holder is rewired onto it, so nothing can dispatch against a
        rewired-but-unserviced scope.

        Raises:
            RuntimeError: An exclusive activity still owns the hardware.
                Both facts are checked -- the activity claim (a run mid
                flight would mix two hardware identities in one run) AND
                the recording controller's busy state, which outlives
                the claim: the recording engine releases its claim
                before the post-drain finish thread completes, and that
                finish still touches the scope. The claim alone would
                un-guard the drain window.
        """
        holder = self.activity_claim.owner
        if self.manual_recording.is_busy or holder is not None:
            busy_with = holder if holder is not None else 'a finishing recording'
            raise RuntimeError(
                f'ScopeSession.set_scope: refusing to swap the scope while '
                f'{busy_with} owns the hardware; stop it and let it finish '
                f'before reconnecting'
            )
        # Capture the OLD logger before the scope handle is reassigned:
        # after the swap, self.scope.metrics_logger is the NEW one, and
        # stopping that instead would leave the old ticks running
        # beside the restarted logger (double-ticking).
        old_metrics_logger = self.scope.metrics_logger if self._metrics_started else None
        self._register_scope_services(scope)
        self.scope = scope
        self.manual_recording.set_scope(scope)
        self.sequenced_capture_runner.set_scope(scope)
        if self.autofocus_runner is not None:
            self.autofocus_runner.set_scope(scope)
        if old_metrics_logger is not None:
            # Metrics were running: move them to the new scope with the
            # same scheduler and cadence. The old logger's system and
            # watchdog ticks survive a disconnect (they read
            # host-lifetime objects), so they must be stopped here.
            self._metrics_started = False
            old_metrics_logger.stop()
            self.start_metrics()
        # Level republish: listeners registered before the swap re-read
        # the derivations against the new scope's world.
        self.notify_run_state()

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
        """The current exclusive-activity owner: None, 'protocol', or
        'recording'."""
        return self.activity_claim.owner

    @property
    def recording_capturing(self) -> bool:
        """True while a manual recording is live (not its drain)."""
        return self.manual_recording.is_recording

    @property
    def protocol_files_draining(self) -> bool:
        """True while a run's file writer still holds pending work."""
        file_io_executor = self.file_io_executor
        return bool(file_io_executor is not None and file_io_executor.is_protocol_queue_active())

    @property
    def run_lockout(self) -> bool:
        """True while a run OR its post-run file drain owns the scope.

        The drain term encodes a deliberate asymmetry: a finished
        protocol frees its claim while its files drain, but the control
        surface stays locked until the queue empties.
        """
        return self.activity_claim.owner == 'protocol' or self.protocol_files_draining

    @property
    def controls_locked(self) -> bool:
        """True while the full control surface locks: any run lockout,
        or a LIVE manual recording (a draining recording frees the
        controls while its claim still refuses new runs)."""
        return self.run_lockout or (
            self.activity_claim.owner == 'recording' and self.recording_capturing
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
        source_path: str = '.',
        scope: object | None = None,
        io_executor: 'SequentialIOExecutor | None' = None,
        camera_executor: 'SequentialIOExecutor | None' = None,
    ) -> 'ScopeSession':
        """Create a session, constructing defaults for any missing components.

        This is the main entry point.  Pass in existing objects when the GUI
        has already created them, or omit them for headless / script use.

        When io_executor / camera_executor are omitted, the full production
        executor bundle is built via executor_registry.create_default so L2
        callers get the same topology lumaviewpro.py runs: IO + CAMERA +
        FILE + WORKER_POOL executors plus protocol_thread (started) and
        scope_display_thread (constructed, not started). When callers pass
        executor handles in, those are used and no bundle is created.
        """
        from modules.lumascope_api._lumascope import _fire_pre_release_warning

        _fire_pre_release_warning()

        built_scope = False
        if scope is None:
            import modules.lumascope_api as lumascope_api

            scope = lumascope_api.Lumascope(configured_model=settings.get('microscope'))
            # The bring-up -- configure from settings, then release the
            # camera start gate -- happens below, once the session exists,
            # for a scope THIS factory built. A scope passed in by a caller
            # is that caller's bring-up responsibility: they call
            # configure_scope() and start_streaming() themselves.
            built_scope = True

        executor_bundle = None
        if io_executor is None and camera_executor is None:
            from modules.executor_registry import create_default

            executor_bundle = create_default(ui_dispatcher=None)
            io_executor = executor_bundle.io_executor
            camera_executor = executor_bundle.camera_executor
        else:
            from modules.sequential_io_executor import SequentialIOExecutor

            if io_executor is None:
                io_executor = SequentialIOExecutor(name='IO')
            if camera_executor is None:
                camera_executor = SequentialIOExecutor(name='CAMERA')

        # Service registration (executors, bundle, source path) happens in
        # __init__ for every session-composed scope -- nothing here.

        wellplate_loader, coordinate_transformer, objective_helper = cls._build_helpers(source_path)

        autofocus_runner, autofocus_thread = cls._build_autofocus_pair(
            scope=scope,
            camera_executor=camera_executor,
            io_executor=io_executor,
            file_io_executor=executor_bundle.file_io_executor if executor_bundle else None,
        )

        session = cls(
            settings=settings,
            scope=scope,
            io_executor=io_executor,
            camera_executor=camera_executor,
            wellplate_loader=wellplate_loader,
            coordinate_transformer=coordinate_transformer,
            objective_helper=objective_helper,
            source_path=source_path,
            executor_bundle=executor_bundle,
            autofocus_runner=autofocus_runner,
            autofocus_thread=autofocus_thread,
            owns_executors=executor_bundle is not None,
        )
        if built_scope:
            cls._bring_up(session)
        return session

    @classmethod
    def create_headless(
        cls,
        settings: dict | None = None,
        source_path: str = '.',
        engineering_mode: bool = False,
    ) -> 'ScopeSession':
        """Create a headless session with simulated hardware.

        Convenience factory for REST API, CLI scripts, and tests.
        Uses simulated drivers so no physical hardware is needed.

        Builds the full production executor topology (IO + CAMERA + FILE +
        WORKER_POOL + protocol_thread + scope_display_thread) so headless
        callers get the same pipelining as lumaviewpro.py instead of a
        degraded 2-executor subset.
        """
        from modules.lumascope_api._lumascope import _fire_pre_release_warning
        from modules.executor_registry import create_default
        import modules.lumascope_api as lumascope_api

        _fire_pre_release_warning()

        if settings is None:
            from modules.settings_init import settings as default_settings

            if default_settings is not None:
                settings = default_settings.copy()
            else:
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

        scope = lumascope_api.Lumascope(simulate=True, configured_model=settings.get('microscope'))
        wellplate_loader, coordinate_transformer, objective_helper = cls._build_helpers(source_path)

        executor_bundle = create_default(ui_dispatcher=None)

        # Service registration (executors, bundle, source path) happens in
        # __init__ for every session-composed scope -- nothing here.

        autofocus_runner, autofocus_thread = cls._build_autofocus_pair(
            scope=scope,
            camera_executor=executor_bundle.camera_executor,
            io_executor=executor_bundle.io_executor,
            file_io_executor=executor_bundle.file_io_executor,
        )

        session = cls(
            settings=settings,
            scope=scope,
            io_executor=executor_bundle.io_executor,
            camera_executor=executor_bundle.camera_executor,
            wellplate_loader=wellplate_loader,
            coordinate_transformer=coordinate_transformer,
            objective_helper=objective_helper,
            source_path=source_path,
            executor_bundle=executor_bundle,
            autofocus_runner=autofocus_runner,
            autofocus_thread=autofocus_thread,
            owns_executors=True,
            engineering_mode=engineering_mode,
        )
        cls._bring_up(session)
        return session

    @staticmethod
    def _build_helpers(source_path: str) -> tuple:
        """The three data-file helpers, each guarded so one corrupt file
        disables one feature with a notification instead of the whole
        composition. A helper that failed is ``None`` here and refuses at
        ``configure_scope`` by name."""
        # Every silent helper-init failure has a downstream AttributeError
        # waiting for whichever UI action first reads the missing helper;
        # surface a warning at the failure site so the user knows which
        # subsystem is unavailable and why.
        from modules.notification_center import notifications

        wellplate_loader = None
        coordinate_transformer = None
        objective_helper = None

        # Loader failures disable a major feature (plate UI / coord
        # conversion / objective lookup). Log at error + exc_info so
        # the traceback lands in the main log; notification level
        # stays at warning to match the existing regression-test
        # contract. Broad Exception catch is legitimate at this
        # top-level boundary -- each loader raises a mix of ValueError /
        # RuntimeError / FileNotFoundError / json.JSONDecodeError plus
        # generic Exception() paths inside objectives_loader.

        try:
            from modules import labware_loader

            wellplate_loader = labware_loader.WellPlateLoader(source_path=source_path)
        except Exception as e:
            logger.error(f'[ScopeSession] Could not load wellplate loader: {e}', exc_info=True)
            notifications.warning(
                'Configuration',
                'Wellplate loader unavailable',
                'Labware configuration could not load. '
                'Plate-based UI (tile plans, well picker) will not work. '
                'Check that data/labware.json exists and is valid.',
            )

        try:
            from modules import coord_transformations

            coordinate_transformer = coord_transformations.CoordinateTransformer()
        except Exception as e:
            logger.error(
                f'[ScopeSession] Could not load coordinate transformer: {e}', exc_info=True
            )
            notifications.warning(
                'Configuration',
                'Coordinate transformer unavailable',
                'Coordinate transformer could not load. '
                'Stage coordinate conversion (plate <-> stage) will not work.',
            )

        try:
            from modules import objectives_loader

            objective_helper = objectives_loader.ObjectiveLoader(source_path=source_path)
        except Exception as e:
            logger.error(f'[ScopeSession] Could not load objective helper: {e}', exc_info=True)
            notifications.warning(
                'Configuration',
                'Objective helper unavailable',
                'Objective configuration could not load. '
                'Objective selection and lookup will not work. '
                'Check that data/objectives.json exists and is valid.',
            )

        return wellplate_loader, coordinate_transformer, objective_helper

    @classmethod
    def _bring_up(cls, session: 'ScopeSession') -> None:
        """Configure the scope a factory built, then release the camera start
        gate -- in that order, because ``initialize`` applies the capture
        pixel format synchronously and wants the gate still closed. A raise
        anywhere in here leaves the caller with no session object to tear
        down, so this tears down what the factory started before it lets
        the raise out; a caller's own executor lanes are never touched."""
        try:
            session.configure_scope()
            session.scope.imaging.start_streaming()
        except BaseException:
            session._abandon()
            session.scope.disconnect()
            raise

    def _abandon(self) -> None:
        """Stop what a factory started for a session it will not return.

        Owned executors, the display and protocol threads and the AF thread
        go through ``shutdown``. A session over caller-passed lanes stops
        only its own scheduler and AF thread: ``shutdown`` would stop the
        caller's lanes too, and a lane cannot be restarted.
        """
        if self._owns_executors:
            self.shutdown()
            return
        self._scheduler.shutdown()
        if self.autofocus_thread is not None:
            self.autofocus_thread.stop(timeout=2.0)

    @staticmethod
    def _build_autofocus_pair(*, scope, camera_executor, io_executor, file_io_executor):
        """Real AF runner + started AF thread for a factory-built session,
        so headless AF-bearing runs get the same wiring the GUI composes."""
        from modules.autofocus_runner import AutofocusRunner
        from modules.autofocus_thread import AutofocusThread

        autofocus_runner = AutofocusRunner(
            scope=scope,
            camera_executor=camera_executor,
            io_executor=io_executor,
            file_io_executor=file_io_executor,
        )
        autofocus_thread = AutofocusThread(afe=autofocus_runner)
        autofocus_thread.start()
        return autofocus_runner, autofocus_thread

    # ------------------------------------------------------------------
    # Convenience wrappers (delegate to config_helpers / scope_commands)
    # ------------------------------------------------------------------

    def recover_file_writer(self) -> bool:
        """Discard pending protocol file writes and unlock a wedged writer.

        L2 counterpart of the GUI's stalled-writer recovery: when a
        protocol run's file writer stops making progress, every
        subsequent run is refused with the ``files_writing_stalled``
        reason until the writer is recovered or the app restarts. This
        method is that recovery for headless / REST / SDK callers:
        pending (unsaved) writes are discarded, protocol mode ends, and
        a worker stuck mid-write is abandoned and replaced.

        Returns:
            True when a recovery was dispatched; False when this session
            holds no file-IO executor (the hosting GUI owns the bundle,
            and its own recovery surface applies).
        """
        if self.file_io_executor is None:
            return False
        self.file_io_executor.recover_wedged_protocol_queue()
        return True

    def get_layer_configs(self, specific_layers=None) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_layer_configs(self.settings, specific_layers)

    def get_stim_configs(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_stim_configs(self.settings)

    def get_enabled_stim_configs(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_enabled_stim_configs(self.settings)

    def get_auto_gain_settings(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_auto_gain_settings(self.settings)

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

    def configure_scope(self) -> None:
        """Configure the scope from this session's settings -- the bring-up.

        Once, after construction, on a real scope: normalize the turret slot
        keys a caller-supplied dict may still carry as JSON strings, resolve
        the declared model's catalogue entry, adopt the slot-1 objective,
        select the labware, build the init config and run
        ``Lumascope.initialize``. The factories run this for the scope they
        build; a host that constructs the session directly, or hands
        ``create`` its own scope, calls it once itself. Every step runs on
        the calling thread; nothing here dispatches.

        Raises:
            ConfigError: a settings key ``initialize`` cannot do without is
                missing (``frame``, ``objective_id``); ``objective_id`` names
                no shipped objective; a data file a helper needs is absent or
                unreadable (``labware.json``, ``objectives.json``); or the
                model catalogue has no usable ``Models`` section.
        """
        import modules.config_helpers as config_helpers
        from modules import layer_record
        from modules.scope_init_config import ScopeInitConfig

        # A caller-supplied dict never went through prepare_settings, whose
        # normalizer is the one boundary between the file's string slot keys
        # and the runtime's ints; without it the adoption below reads slot 1
        # as unassigned and silently keeps the stored objective.
        settings_init._normalize_turret_slot_keys(self.settings)
        scope_models = layer_record.load_scope_models()
        scope_config = scope_models.get(self.settings.get('microscope'))
        self.adopt_turret_slot1_objective(
            model_has_turret=config_helpers.model_has_turret(scope_models, self.settings)
        )
        for helper, data_file in (
            (self.wellplate_loader, 'labware.json'),
            (self.objective_helper, 'objectives.json'),
            (self.coordinate_transformer, 'the coordinate transformer'),
        ):
            if helper is None:
                raise ConfigError(
                    f'cannot configure the scope: {data_file} did not load under '
                    f'{self.source_path!r} (see the earlier error); a data root '
                    'without the shipped files is not an installation'
                )
        _labware_id, labware = config_helpers.get_selected_labware_from_settings(
            self.settings, self.wellplate_loader
        )
        config = ScopeInitConfig.from_settings(
            self.settings,
            labware,
            scope_config=scope_config,
            layer_identity=self.scope.layer_identity,
        )
        self.scope.initialize(config)
        # The objective in place at bring-up never passes through the
        # selection member, so without this a session that changed
        # nothing would have no record of the scale it was using.
        # `initialize` has just validated the id, so the lookup is a dict.
        objective_id = self.settings['objective_id']
        info = self.objective_helper.get_objective_info(objective_id=objective_id)
        self._log_resolved_optics(objective_id, info['focal_length'])

    def adopt_turret_slot1_objective(self, model_has_turret: bool) -> None:
        """Make position 1's assignment the session's starting objective.

        This method is not part of the L2 API surface: ``configure_scope``
        calls it once per bring-up, before settings are consumed, and the
        reconnect handler calls it the same way; an L2 caller changes
        objectives through the selection surface.

        Startup leaves the turret at position 1 (homing puts it there),
        so the stored objective_id is a leftover from the previous
        session, not a fact about what sits in the light path: a session
        that ended on another slot, or an assignment reset, leaves it
        naming glass the turret does not hold -- and the pixel size
        derived from it is stamped into the scale bar and saved-image
        metadata. Whatever position 1 holds IS the starting objective;
        call this before anything consumes settings.

        Args:
            model_has_turret: The DECLARED model's turret flag
                (scopes.json), not live capabilities -- a scope whose
                motorboard is dead reports no axes, and that
                broken-hardware case is exactly when the stale
                objective would otherwise survive. No-op when False:
                on non-turret models objective_id is the user's free
                choice.
        """
        if not model_has_turret:
            return
        turret_objectives = self.settings.get('turret_objectives') or {}
        slot1_objective = turret_objectives.get(1)
        if slot1_objective is None:
            # Nothing assigned at the starting position: keep the stored
            # objective rather than inventing one; the unassigned-slot
            # prompt owns resolving this with the user.
            return
        if self.settings.get('objective_id') == slot1_objective:
            return
        logger.info(
            f'[Session  ] Starting objective follows turret position 1: '
            f'{slot1_objective!r} (stored selection was '
            f'{self.settings.get("objective_id")!r})'
        )
        with self.settings_lock:
            self.settings['objective_id'] = slot1_objective

    def settings_are_provisional(self) -> bool:
        """Is the app running on defaults nobody has agreed to keep?

        True while the user's current.json could not be read and no one
        has decided its fate. While it holds, every save aimed at
        current.json raises SettingsSaveRefusedError -- resolve with
        retire_rejected_settings() after the user has chosen to start
        over.
        """
        return settings_init.settings_are_provisional()

    def retire_rejected_settings(self) -> 'str | None':
        """Resolve the provisional-settings state: retire the unreadable file.

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
                current.json could not be read AND the save targets
                current.json (force does not override; a save aimed at
                any other destination still writes). reason='no_hardware'
                when no hardware was connected this session and force is
                not set.
        """
        logger.info('[Session  ] save_settings()')

        # Outside the force gate on purpose: force means "save even though no
        # hardware was connected", not "save over a file we were told to leave
        # alone". The settings in memory right now are the shipped template,
        # loaded because the user's own file could not be read; writing them
        # to current.json would replace their entire configuration with
        # defaults. Resolved by retire_rejected_settings() once the user
        # has actually chosen to start over.
        if settings_init.settings_are_provisional() and settings_init.targets_current_json(file):
            logger.warning(
                '[Session  ] save_settings: refused -- running on default '
                'settings because current.json could not be read. Not '
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

    def get_current_objective_info(self) -> 'tuple[str, dict]':
        import modules.config_helpers as config_helpers

        return config_helpers.get_current_objective_info(self.settings, self.objective_helper)

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

    def _require_objective_catalogue(self) -> None:
        if self.objective_helper is None:
            raise ConfigError(
                'the objective catalogue is unavailable: objectives.json did not load '
                f'under {self.source_path!r}'
            )

    def objective_question(self) -> 'ObjectiveQuestion | None':
        """Does the objective need confirming? The question, or None.

        A read, for any host: the GUI renders the answer as a popup, a
        REST caller reads it as state. Two ways the session cannot know
        what is in the light path: no person has ever confirmed the
        objective on this install (the settings template ships a default
        that would otherwise set image scale silently forever), or the
        DECLARED turret model's current position has no assignment. The
        declared model, not the live capability: a dead motorboard
        reports no axes, and that is exactly when a stale objective must
        not pass unasked.

        Two conditions withhold an owed question, each leaving one log
        line per call so a bundle can say why nothing was asked: with no
        hardware there is nothing in the light path and no capture to
        stamp; while settings are provisional every write is refused, so
        an answer given now would be lost -- the host re-asks when they
        resolve. No line is logged for a returned question: the renderer
        logs its own show, and a polled read must not log per poll.

        Raises:
            ConfigError: the catalogue is unavailable or empty, the
                model catalogue cannot be read, or the stored
                ``turret_position`` is not a whole number.
        """
        import modules.config_helpers as config_helpers
        from modules import layer_record

        self._require_objective_catalogue()
        has_turret = config_helpers.model_has_turret(
            layer_record.load_scope_models(), self.settings
        )
        first_run = not self.settings.get('objective_confirmed', False)
        slots = self.settings.get('turret_objectives') or {}
        if has_turret:
            # A stored 0, '' or False reads as position 1, the slot homing
            # leaves the turret on. Coerced here because nothing types the
            # value where the file is read, so a hand-edited "2" arrives
            # as a string.
            raw = self.settings.get('turret_position') or 1
            try:
                position = int(raw)
            except (TypeError, ValueError):
                raise ConfigError(f'turret_position must be a whole number, got {raw!r}') from None
            slot_unassigned = slots.get(position) is None
        else:
            position = None
            slot_unassigned = False
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
        choices = tuple(self.objective_helper.get_objectives_list())
        if not choices:
            raise ConfigError(
                'the objective catalogue is empty; cannot ask which objective is installed'
            )
        proposed = (slots.get(position) if position is not None else None) or self.settings.get(
            'objective_id'
        )
        if proposed not in choices:
            proposed = choices[0]
        return ObjectiveQuestion(turret_position=position, proposed=proposed, choices=choices)

    def confirm_objective(self, objective_id: str, turret_position: 'int | None' = None) -> bool:
        """Answer the objective question: this objective is in the light path.

        Selects the objective, assigns it to ``turret_position`` when one
        is given, and records that a person has confirmed the objective
        on this install. Returns whether the objective changed.

        Raises:
            ConfigError: ``objective_id`` is not exactly a catalogue key,
                or the catalogue is unavailable. Nothing is written.
            ValueError: ``turret_position`` is not a slot number 1-4.
        """
        changed = self.select_objective(objective_id)
        if turret_position is not None:
            self.assign_turret_objective(turret_position, objective_id)
        with self.settings_lock:
            self.settings['objective_confirmed'] = True
        logger.info(
            f'[Session  ] objective confirmed: {objective_id!r}'
            + (f' at turret position {turret_position}' if turret_position is not None else '')
        )
        return changed

    def select_objective(self, objective_id: str) -> bool:
        """Make ``objective_id`` the active objective. Returns whether it changed.

        The one writer of the active objective for every host: the
        settings store and the scope's runtime state move together, and
        the resolved optics are recorded, because the pixel size derived
        here is stamped into every capture. Selecting the objective
        already held is a no-op -- a programmatic re-selection (a turret
        move, a settings load) is not a change.

        Raises:
            ConfigError: ``objective_id`` is not exactly a catalogue key.
                Refused before any write, for every id including the one
                held: the catalogue loader matches prefixes, so a partial
                id would otherwise bind silently to the first match, and
                '' to the first entry.
        """
        self._require_objective_catalogue()
        if objective_id not in self.objective_helper.get_objectives_list():
            raise ConfigError(f'unknown objective {objective_id!r}; the catalogue has no such key')
        if objective_id == self.settings.get('objective_id'):
            return False
        info = self.objective_helper.get_objective_info(objective_id=objective_id)
        # Selecting an objective the turret does not hold is a normal step
        # of assigning it: the user picks the objective, then binds it to
        # a position. Logged, not refused: the moments where an unassigned
        # objective actually blocks something (creating, modifying, adding
        # to and running a protocol) each refuse there.
        assigned = [
            objective
            for objective in (self.settings.get('turret_objectives') or {}).values()
            if objective is not None
        ]
        if assigned and objective_id not in assigned:
            logger.info(
                f'[Session  ] Objective {objective_id!r} selected with no turret '
                f'position assigned; assigned objectives are {assigned}'
            )
        self.scope.runtime_state.set_objective(objective_id=objective_id)
        with self.settings_lock:
            self.settings['objective_id'] = objective_id
        self._log_resolved_optics(objective_id, info['focal_length'])
        return True

    def assign_turret_objective(self, position: int, objective_id: str) -> None:
        """Bind ``objective_id`` to turret slot ``position``.

        Raises:
            ValueError: ``position`` is not a slot number 1-4.
            ConfigError: ``objective_id`` is not exactly a catalogue key.
        """
        self._require_objective_catalogue()
        self._check_turret_slot(position)
        if objective_id not in self.objective_helper.get_objectives_list():
            raise ConfigError(f'unknown objective {objective_id!r}; the catalogue has no such key')
        with self.settings_lock:
            self.settings['turret_objectives'][position] = objective_id
        self.scope.runtime_state.set_turret_config(self.settings['turret_objectives'])

    def clear_turret_objective(self, position: int) -> None:
        """Leave turret slot ``position`` unassigned.

        Raises:
            ValueError: ``position`` is not a slot number 1-4.
        """
        self._check_turret_slot(position)
        with self.settings_lock:
            self.settings['turret_objectives'][position] = None
        self.scope.runtime_state.set_turret_config(self.settings['turret_objectives'])

    @staticmethod
    def _check_turret_slot(position) -> None:
        if not isinstance(position, int) or isinstance(position, bool) or not 1 <= position <= 4:
            raise ValueError(f'turret slot must be a whole number 1-4, got {position!r}')

    def set_turret_position(self, position: int) -> None:
        """Record the slot the turret landed on.

        Records, never refuses: the position is a fact of the motion that
        already happened, and a slot key outside 1-4 is constructible from
        a hand-edited settings file, so refusing here would only turn a
        landed move into an error. Recording the position the turret is
        already on is a no-op. A change onto a slot with no assignment is
        logged as a warning on a scope with a live turret -- the previous
        objective would otherwise keep setting the image scale silently;
        ``objective_question`` then owes the question.

        Raises:
            TypeError: ``position`` is not an int.
        """
        if not isinstance(position, int) or isinstance(position, bool):
            raise TypeError(f'turret position must be an int, got {position!r}')
        if position == self.settings.get('turret_position'):
            return
        with self.settings_lock:
            self.settings['turret_position'] = position
        if (
            self.scope.capabilities.has_turret
            and (self.settings.get('turret_objectives') or {}).get(position) is None
        ):
            logger.warning(f'[Session  ] turret at position {position} with no objective assigned')

    def _log_resolved_optics(self, objective_id: str, focal_length: float) -> None:
        import modules.config_helpers as config_helpers

        config_helpers.log_resolved_optics(
            objective_id,
            focal_length,
            self.scope.imaging.get_binning_size(),
            capabilities=self.scope.capabilities,
        )

    def get_current_plate_position(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_current_plate_position(
            self.scope,
            self.settings,
            self.coordinate_transformer,
            self.wellplate_loader,
        )

    # --- Hardware commands: NOT forwarded ---
    # The Session surface deliberately carries no hardware-command
    # forwarders. L2 callers reach hardware through the composition
    # root the Session exposes -- session.scope.illumination.*,
    # session.scope.motion.*, session.scope.imaging.*,
    # session.scope.runtime_state.* -- so every command has exactly one
    # public spelling and the Session owns only what is session-scoped:
    # lifecycle (create / start_executors / shutdown /
    # start_application_session / start_metrics / stop_metrics), the
    # protocol runner, run-state queries, and the settings-composition
    # getters above.

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

    def start_executors(self) -> None:
        """Start the IO and camera executors."""
        self.io_executor.start()
        self.camera_executor.start()

    def start_metrics(self) -> None:
        """Start the scope's periodic metrics logging.

        Uses the session's scheduler and the
        ``settings.profiling.metrics_interval_s`` cadence override when
        present. Metrics stay opt-in by the call itself: headless hosts
        simply never call this. A scope whose MetricsLogger failed to
        construct (that failure is logged as a warning at construction)
        is tolerated as a no-op: metrics are observability, not a
        reason to fail the session lifecycle.

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
        metrics_logger = self.scope.metrics_logger
        if metrics_logger is None:
            logger.warning(
                '[ScopeSession] start_metrics: the scope has no MetricsLogger; '
                'periodic metrics stay off'
            )
            return
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
        metrics_logger.start(self._scheduler, **start_kwargs)
        self._metrics_started = True

    def stop_metrics(self) -> None:
        """Stop the scope's periodic metrics logging. Idempotent.

        Never shuts the scheduler itself down -- a shut scheduler
        refuses all future schedules, and set_scope must be able to
        restart metrics on the next scope with the same instance.
        """
        if not self._metrics_started:
            return
        self._metrics_started = False
        metrics_logger = self.scope.metrics_logger
        if metrics_logger is not None:
            metrics_logger.stop()

    def shutdown_executors(self) -> None:
        """Shut down the IO and camera executors."""
        self.io_executor.shutdown()
        self.camera_executor.shutdown()

    def shutdown(self) -> None:
        """Tear down everything this session constructed.

        Teardown scope follows the explicit ownership fact, never
        bundle-presence: every host hands its bundle in for scope
        servicing, so "holds a bundle" no longer means "built the
        topology". A session that OWNS its executors (the factories)
        stops the long-lived consumer threads BEFORE the executor lanes
        they consume -- a consumer mid-iteration that finds its lane
        already shut down can hang on a dispatch that never fires
        (scope_display_thread consumes camera_executor; protocol_thread
        drives io + camera + file lanes). A non-owning session still
        stops the handles the caller passed in (io, camera, and the AF
        thread) -- not-owner is not a no-op -- but never the host's
        bundle. Running metrics stop first, or their ticks would
        outlive the executors they snapshot.
        """
        self.stop_metrics()
        # Settle any run's merge outcome FIRST. The executor teardown below
        # does not wait for the file lanes to drain, so a merge still
        # waiting on this run's writes can never finish -- and a caller
        # blocked on the result would wait out its whole bound for an
        # answer that is no longer coming. Ahead of the non-owner early
        # return, because a borrowed-executor session tears down the same
        # way from the waiter's point of view.
        runner = self.sequenced_capture_runner
        if runner is not None:
            outcome = runner.merge_outcome()
            if outcome is not None:
                outcome.settle_unfinished('shutdown')
        # The session owns its scheduler: shut it down here, before the
        # non-owner early return below, so a session that borrowed its
        # executors still ends its own timers (a live health check
        # outliving the session would fire into torn-down state).
        self._scheduler.shutdown()
        if self.autofocus_thread is not None:
            self.autofocus_thread.stop(timeout=2.0)
        if not self._owns_executors:
            self.shutdown_executors()
            return
        bundle = self.executor_bundle
        bundle.scope_display_thread.stop()
        bundle.protocol_thread.stop(timeout=2.0)
        bundle.io_executor.shutdown(wait=False)
        bundle.camera_executor.shutdown(wait=False)
        bundle.file_io_executor.shutdown(wait=False)
        bundle.worker_pool.shutdown(wait=False)

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
           to position 1, the slot whose objective was adopted at
           configure, and record it in ``settings['turret_position']``
           so later code reads the actual position.

        ``disable_homing=True`` skips BOTH steps: no startup motion on
        any axis. The turret is left where it is, like the stage axes,
        and no turret position is recorded -- positioning it without a
        home would be an absolute move against a reference the caller
        asked us not to establish. The skip is the requested behaviour,
        so it is logged, not signalled.

        Headless / REST callers can use this exact same call to apply
        the standard startup orchestration without copy-pasting from
        the App.

        Args:
            disable_homing: If True, issue no startup motion at all.
            home_fn: Callable taking an axis name and returning whether
                the home succeeded. Defaults to the motion API.
            turret_fn: Callable taking a turret position. Defaults to
                the motion API.

        The two motion callables are injected the same way the metrics
        scheduler is: the hosting environment supplies its own, and the
        API default is what everything else gets. The Kivy app passes
        the ``ui_helpers`` wrappers, which drive the turret through the
        widget that also reconciles the objective, spinner and button
        state -- policy that lives in the UI and has no API equivalent
        yet. Defaulting to the API instead of importing the UI is what
        lets a headless caller run this at all: the widget path reaches
        ``ctx.motion_settings``, which is None until a widget tree
        exists, so before injection this method could not run outside
        the GUI despite the docstring above promising it could.
        """
        if disable_homing:
            logger.info('startup motion skipped: homing disabled; the turret is left where it is')
            return

        if home_fn is None:
            home_fn = lambda axis: self.scope.motion.move_home_and_wait(axis)
        if turret_fn is None:
            turret_fn = lambda position: self.scope.motion.move_turret(position)

        # Wait for the home's result and honor it. Turret positioning is
        # an absolute move against the reference frame the home was
        # supposed to establish; running it after a failed home is the
        # secondary cascade users report -- a second error on top of the
        # home's own, for motion that could never have been correct. The
        # home already notified, so this stays a log.
        if not home_fn('ALL'):
            logger.error(
                'Homing did not succeed -- skipping startup turret '
                'positioning; the stage reference is unknown'
            )
            return

        if self.scope.capabilities.has_turret:
            # Every session starts at position 1: the objective was
            # already adopted from that slot, so positioning anywhere
            # else would split the claimed optics from the physical
            # glass. After a real home this move is a physical no-op,
            # but it still must be issued -- it is the only startup
            # path that highlights the turret button.
            START_POSITION = 1
            self.set_turret_position(START_POSITION)
            turret_fn(START_POSITION)
