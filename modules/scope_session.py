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

from typing import TYPE_CHECKING

from lvp_logger import logger
from modules.activity_claim import ActivityClaim
from modules.manual_recording import ManualRecordingController

# ProtocolRunner is referenced only in a return annotation; it is
# imported function-locally to avoid a circular import. Declare it here
# for the annotation without a runtime import.
if TYPE_CHECKING:
    from modules.protocol_runner import ProtocolRunner


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
    ):
        self.settings = settings
        self.scope = scope
        self.io_executor = io_executor
        self.camera_executor = camera_executor
        self.wellplate_loader = wellplate_loader
        self.coordinate_transformer = coordinate_transformer
        self.objective_helper = objective_helper
        self.source_path = source_path
        # When ScopeSession.create* built the executors itself, the bundle
        # is held here so headless callers can shut down protocol_thread /
        # scope_display_thread cleanly. When lumaviewpro.py is the host,
        # this is None -- the host owns the bundle on ctx.executor_bundle.
        self.executor_bundle = executor_bundle
        # The canonical file-IO executor lives on the bundle; expose it here
        # alongside io_executor / camera_executor so callers (e.g. ProtocolRunner)
        # source the one shared FILE executor instead of constructing a
        # duplicate. A bundle-less host (the GUI) passes its handle in --
        # the session cannot read the file-drain fact without it.
        self.file_io_executor = file_io_executor or (
            executor_bundle.file_io_executor if executor_bundle else None
        )

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
        )
        # Whether the running configuration declares an XY stage. Written
        # by the scope-config apply path; motion_enabled derives from it.
        # Defaults True so bundle-less and headless hosts keep motion
        # until a config says otherwise.
        self.xystage_configured = True
        if self.file_io_executor is not None:
            self.file_io_executor.add_protocol_idle_listener(self.notify_run_state)

    def set_scope(self, scope) -> None:
        """Rewire this session onto a NEW scope after a reconnect.

        The session and its recording controller each hold the scope by
        reference; left unrewired after a reconnect they keep driving
        the discarded, disconnected scope (start_application_session
        homes it; a recording captures from it).

        Raises:
            RuntimeError: A recording is live, draining, or finishing.
                Its frame listener and writer belong to the old scope;
                swapping underneath it would capture from one camera
                and finish against another.
        """
        if self.manual_recording.is_busy:
            raise RuntimeError(
                'ScopeSession.set_scope: a recording is still active; '
                'stop it and let it finish before reconnecting the scope'
            )
        self.scope = scope
        self.manual_recording.set_scope(scope)
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
        """True when user stage motion is allowed: the configuration
        declares an XY stage and no run lockout holds. Evaluated at
        read -- there is no cached copy to mis-restore."""
        return self.xystage_configured and not self.run_lockout

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
        scope=None,
        io_executor=None,
        camera_executor=None,
    ):
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

        if scope is None:
            import modules.lumascope_api as lumascope_api

            scope = lumascope_api.Lumascope()
            # connect() leaves the camera configured but NOT grabbing (the
            # camera-lifecycle start gate). The GUI releases the gate in its
            # own bring-up; for a scope THIS session constructed there is no
            # other bring-up, so release it here or headless captures time
            # out forever. Scopes passed in by a caller are that caller's
            # bring-up responsibility (already released, or deliberately
            # stopped -- either way not ours to restart).
            scope.imaging.start_streaming()

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

        # LAYER-A': register executors on the scope so scope.X_async /
        # scope.X_sync can dispatch without callers passing executor handles.
        # When the bundle was built, register file_io_executor too so
        # protocol_image_writer + IOTask file-IO paths land on the dedicated
        # queue instead of falling back to inline execution.
        scope.register_executors(
            camera_executor=camera_executor,
            io_executor=io_executor,
            file_io_executor=executor_bundle.file_io_executor if executor_bundle else None,
        )
        if executor_bundle is not None:
            scope.register_executor_bundle(executor_bundle, settings=settings)
        # Register source_path for the protocol constructors -- falls back
        # to current working dir for the rare ScopeSession path that
        # doesn't pass source_path.
        scope.protocols.register_source_path(source_path)

        # Optional helpers -- import and construct if available.
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

        return cls(
            settings=settings,
            scope=scope,
            io_executor=io_executor,
            camera_executor=camera_executor,
            wellplate_loader=wellplate_loader,
            coordinate_transformer=coordinate_transformer,
            objective_helper=objective_helper,
            source_path=source_path,
            executor_bundle=executor_bundle,
        )

    @classmethod
    def create_headless(cls, settings: dict | None = None, source_path: str = '.'):
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
                import json

                from modules.settings_init import _resolve_settings_path

                try:
                    settings_path = _resolve_settings_path(source_path)
                    with open(settings_path) as f:
                        settings = json.load(f)
                except FileNotFoundError:
                    settings = {}

        scope = lumascope_api.Lumascope(simulate=True)
        # Release the camera start gate: connect() leaves the sim camera
        # configured but NOT grabbing, and this factory is the whole
        # bring-up for the simulated session -- without the release every
        # capture would time out.
        scope.imaging.start_streaming()

        executor_bundle = create_default(ui_dispatcher=None)

        # LAYER-A': register all three executor handles on the scope so
        # scope.X_async / scope.X_sync + protocol_image_writer file-IO
        # paths land on the proper queues.
        scope.register_executors(
            camera_executor=executor_bundle.camera_executor,
            io_executor=executor_bundle.io_executor,
            file_io_executor=executor_bundle.file_io_executor,
        )
        # LVP-A-13: wire the bundle so metrics_logger.snapshot() reports
        # all 4 executor queue depths instead of a degraded subset.
        scope.register_executor_bundle(executor_bundle, settings=settings)
        # Register source_path (defaults to "." in headless).
        scope.protocols.register_source_path(source_path)

        return cls(
            settings=settings,
            scope=scope,
            io_executor=executor_bundle.io_executor,
            camera_executor=executor_bundle.camera_executor,
            source_path=source_path,
            executor_bundle=executor_bundle,
        )

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

    def get_current_objective_info(self) -> dict:
        import modules.config_helpers as config_helpers

        return config_helpers.get_current_objective_info(self.settings, self.objective_helper)

    def get_current_plate_position(self) -> 'dict | None':
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
    # start_application_session), the protocol runner, run-state
    # queries, and the settings-composition getters above.

    # ------------------------------------------------------------------
    # Protocol runner
    # ------------------------------------------------------------------

    def create_protocol_runner(self, **kwargs) -> 'ProtocolRunner':
        """Create a ProtocolRunner bound to this session.

        Returns a ProtocolRunner that can run scans and protocols
        using this session's scope, settings, and executors.
        """
        from modules.protocol_runner import ProtocolRunner

        return ProtocolRunner(session=self, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_executors(self) -> None:
        """Start the IO and camera executors."""
        self.io_executor.start()
        self.camera_executor.start()

    def shutdown_executors(self) -> None:
        """Shut down the IO and camera executors."""
        self.io_executor.shutdown()
        self.camera_executor.shutdown()

    def shutdown(self) -> None:
        """Tear down everything this session constructed.

        For a session that built its own executor bundle (create /
        create_headless with no caller-supplied executors), stops the
        long-lived consumer threads BEFORE the executor lanes they
        consume -- a consumer mid-iteration that finds its lane already
        shut down can hang on a dispatch that never fires.
        scope_display_thread consumes camera_executor; protocol_thread
        drives io + camera + file lanes. For a session running on a
        caller's executors, only the handles the caller passed in are
        stopped; the caller owns the rest of its topology.
        """
        bundle = self.executor_bundle
        if bundle is None:
            self.shutdown_executors()
            return
        bundle.scope_display_thread.stop()
        bundle.protocol_thread.stop(timeout=2.0)
        bundle.io_executor.shutdown(wait=False)
        bundle.camera_executor.shutdown(wait=False)
        bundle.file_io_executor.shutdown(wait=False)
        bundle.worker_pool.shutdown(wait=False)

    def start_application_session(self, *, disable_homing: bool = False) -> None:
        """LVP-A-5: queue the standard startup home + turret-positioning sequence.

        Replaces the inline blocks in lumaviewpro.py:on_start AND
        ui/microscope_settings.py reconnect handler -- both previously
        open-coded the same ALL-axis home + turret-positioning sequence
        with a Rule-2 single-source-of-truth violation (drift risk if
        one branch ever updated without the other).

        After this method returns, the io_executor has been told to:

        1. (when ``disable_homing=False``) home ALL axes via ``move_home``.
           Firmware homes Z, T, X, Y in one routine; on Z-only boards
           it homes what it has and reports the missing axes.

        2. (when ``self.scope.capabilities.has_turret`` is True) move T-axis
           to the position that matches ``settings['objective_id']`` --
           falls back to position 1 if the objective isn't in the turret
           config. Updates ``settings['turret_position']`` so later code
           reads the actual position.

        Headless / REST callers can use this exact same call to apply
        the standard startup orchestration without copy-pasting from
        the App.

        Args:
            disable_homing: If True, skip the home step but still run
                turret-positioning. Matches the App's ``--no-home``
                CLI flag semantics.
        """
        # Local import to avoid circular import at module load -- ui_helpers
        # imports many UI modules but the functions used here (move_home,
        # move_absolute) operate on the scope and don't actually
        # need a GUI surface.
        from ui.ui_helpers import move_home, move_absolute

        if not disable_homing:
            move_home('ALL')

        if self.scope.capabilities.has_turret:
            objective_id = self.settings.get('objective_id')
            turret_position = self.scope.motion.get_turret_position_for_objective_id(
                objective_id=objective_id,
                persisted_position=self.settings.get('turret_position'),
            )
            if turret_position is None:
                DEFAULT_POSITION = 1
                logger.info(
                    f'Turret position for set objective {objective_id} not '
                    f'in turret objectives configuration. Setting to '
                    f'position {DEFAULT_POSITION}'
                )
                turret_position = DEFAULT_POSITION

            self.settings['turret_position'] = turret_position
            move_absolute(
                axis='T',
                position=turret_position,
                wait_until_complete=True,
            )
