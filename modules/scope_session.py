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

import os
import threading

from lvp_logger import logger


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
        source_path: str = ".",
    ):
        self.settings = settings
        self.scope = scope
        self.io_executor = io_executor
        self.camera_executor = camera_executor
        self.wellplate_loader = wellplate_loader
        self.coordinate_transformer = coordinate_transformer
        self.objective_helper = objective_helper
        self.source_path = source_path

        self.protocol_running = threading.Event()
        self.focus_round = 0

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        settings: dict,
        source_path: str = ".",
        scope=None,
        io_executor=None,
        camera_executor=None,
    ):
        """Create a session, constructing defaults for any missing components.

        This is the main entry point.  Pass in existing objects when the GUI
        has already created them, or omit them for headless / script use.
        """
        from modules.sequential_io_executor import SequentialIOExecutor

        if scope is None:
            import modules.lumascope_api as lumascope_api
            scope = lumascope_api.Lumascope()

        if io_executor is None:
            io_executor = SequentialIOExecutor(name="IO")

        if camera_executor is None:
            camera_executor = SequentialIOExecutor(name="CAMERA")

        # LAYER-A': register executors on the scope so scope.X_async /
        # scope.X_sync can dispatch without callers passing executor handles.
        scope.register_executors(
            camera_executor=camera_executor,
            io_executor=io_executor,
        )
        # LAYER-I: register source_path for scope.load_protocol /
        # create_protocol — falls back to current working dir for the
        # rare ScopeSession path that doesn't pass source_path.
        scope.register_source_path(source_path)

        # Optional helpers — import and construct if available.
        # Every silent helper-init failure has a downstream AttributeError
        # waiting for whichever UI action first reads the missing helper;
        # surface a warning at the failure site so the user knows which
        # subsystem is unavailable and why.
        from modules.notification_center import notifications

        wellplate_loader = None
        coordinate_transformer = None
        objective_helper = None

        try:
            from modules import labware_loader
            wellplate_loader = labware_loader.WellPlateLoader(source_path=source_path)
        except Exception as e:
            logger.warning(f"[ScopeSession] Could not load wellplate loader: {e}")
            notifications.warning("Configuration", "Wellplate loader unavailable",
                f"Labware configuration could not load: {type(e).__name__}: {e}. "
                f"Plate-based UI (tile plans, well picker) will not work. "
                f"Check that data/labware.json exists and is valid.")

        try:
            from modules import coord_transformations
            coordinate_transformer = coord_transformations.CoordinateTransformer()
        except Exception as e:
            logger.warning(f"[ScopeSession] Could not load coordinate transformer: {e}")
            notifications.warning("Configuration", "Coordinate transformer unavailable",
                f"Coordinate transformer could not load: {type(e).__name__}: {e}. "
                f"Stage coordinate conversion (plate <-> stage) will not work.")

        try:
            from modules import objectives_loader
            objective_helper = objectives_loader.ObjectiveLoader(source_path=source_path)
        except Exception as e:
            logger.warning(f"[ScopeSession] Could not load objective helper: {e}")
            notifications.warning("Configuration", "Objective helper unavailable",
                f"Objective configuration could not load: {type(e).__name__}: {e}. "
                f"Objective selection and lookup will not work. "
                f"Check that data/objectives.json exists and is valid.")

        return cls(
            settings=settings,
            scope=scope,
            io_executor=io_executor,
            camera_executor=camera_executor,
            wellplate_loader=wellplate_loader,
            coordinate_transformer=coordinate_transformer,
            objective_helper=objective_helper,
            source_path=source_path,
        )

    @classmethod
    def create_headless(cls, settings: dict | None = None, source_path: str = "."):
        """Create a headless session with simulated hardware.

        Convenience factory for REST API, CLI scripts, and tests.
        Uses simulated drivers so no physical hardware is needed.
        """
        from modules.sequential_io_executor import SequentialIOExecutor
        import modules.lumascope_api as lumascope_api

        if settings is None:
            from modules.settings_init import settings as default_settings
            if default_settings is not None:
                settings = default_settings.copy()
            else:
                # Settings not loaded yet (e.g. headless/test usage) — load from disk
                import json
                settings_path = os.path.join(source_path, "data", "settings.json")
                if os.path.exists(settings_path):
                    with open(settings_path) as f:
                        settings = json.load(f)
                else:
                    settings = {}

        scope = lumascope_api.Lumascope(simulate=True)

        io_executor = SequentialIOExecutor(name="IO")
        camera_executor = SequentialIOExecutor(name="CAMERA")

        # LAYER-A': register executors on the scope (headless session).
        scope.register_executors(
            camera_executor=camera_executor,
            io_executor=io_executor,
        )
        # LAYER-I: register source_path (defaults to "." in headless).
        scope.register_source_path(source_path)

        return cls(
            settings=settings,
            scope=scope,
            io_executor=io_executor,
            camera_executor=camera_executor,
            source_path=source_path,
        )

    # ------------------------------------------------------------------
    # Convenience wrappers (delegate to config_helpers / scope_commands)
    # ------------------------------------------------------------------

    def get_layer_configs(self, specific_layers=None):
        import modules.config_helpers as config_helpers
        return config_helpers.get_layer_configs(self.settings, specific_layers)

    def get_stim_configs(self):
        import modules.config_helpers as config_helpers
        return config_helpers.get_stim_configs(self.settings)

    def get_enabled_stim_configs(self):
        import modules.config_helpers as config_helpers
        return config_helpers.get_enabled_stim_configs(self.settings)

    def get_auto_gain_settings(self):
        import modules.config_helpers as config_helpers
        return config_helpers.get_auto_gain_settings(self.settings)

    def get_current_objective_info(self):
        import modules.config_helpers as config_helpers
        return config_helpers.get_current_objective_info(self.settings, self.objective_helper)

    def get_current_plate_position(self):
        import modules.config_helpers as config_helpers
        return config_helpers.get_current_plate_position(
            self.scope, self.settings, self.coordinate_transformer, self.wellplate_loader,
        )

    def log_system_metrics(self):
        import modules.config_helpers as config_helpers
        config_helpers.log_system_metrics(self.settings)

    # --- LED commands (thin shims around Lumascope's executor-backed API) ---

    def leds_off(self, callback=None):
        self.scope.leds_off_async(callback=callback)

    def led_on(self, channel, illumination, callback=None, cb_kwargs=None):
        self.scope.led_on_async(
            channel, illumination, callback=callback, cb_kwargs=cb_kwargs,
        )

    def led_off(self, channel, callback=None, cb_kwargs=None):
        self.scope.led_off_async(
            channel, callback=callback, cb_kwargs=cb_kwargs,
        )

    def led_on_sync(self, channel, illumination, timeout=5):
        self.scope.led_on_sync(channel, illumination, timeout=timeout)

    # --- Motion commands ---

    def move_absolute(self, axis, pos, wait_until_complete=False,
                      overshoot_enabled=True, callback=None, cb_kwargs=None):
        self.scope.motion.move_absolute_async(
            axis, pos,
            wait_until_complete=wait_until_complete,
            overshoot_enabled=overshoot_enabled,
            callback=callback, cb_kwargs=cb_kwargs,
        )

    def move_relative(self, axis, um, wait_until_complete=False,
                      overshoot_enabled=True, callback=None, cb_kwargs=None):
        self.scope.motion.move_relative_async(
            axis, um,
            wait_until_complete=wait_until_complete,
            overshoot_enabled=overshoot_enabled,
            callback=callback, cb_kwargs=cb_kwargs,
        )

    def move_home(self, axis, callback=None, cb_args=None):
        self.scope.motion.move_home_async(
            axis, callback=callback, cb_args=cb_args,
        )

    # ------------------------------------------------------------------
    # Protocol runner
    # ------------------------------------------------------------------

    def create_protocol_runner(self, **kwargs):
        """Create a ProtocolRunner bound to this session.

        Returns a ProtocolRunner that can run scans and protocols
        using this session's scope, settings, and executors.
        """
        from modules.protocol_runner import ProtocolRunner
        return ProtocolRunner(session=self, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_executors(self):
        """Start the IO and camera executors."""
        self.io_executor.start()
        self.camera_executor.start()

    def shutdown_executors(self):
        """Shut down the IO and camera executors."""
        self.io_executor.shutdown()
        self.camera_executor.shutdown()

    def start_application_session(self, *, disable_homing: bool = False):
        """LVP-A-5: queue the standard startup home + turret-positioning sequence.

        Replaces the inline blocks in lumaviewpro.py:on_start AND
        ui/microscope_settings.py reconnect handler -- both previously
        open-coded the same ALL-axis home + turret-positioning IOTasks
        with a Rule-2 single-source-of-truth violation (drift risk if
        one branch ever updated without the other).

        After this method returns, two IOTasks have been put on the
        io executor:

        1. (when ``disable_homing=False``) ALL-axis ``move_home``.
           Firmware homes Z, T, X, Y in one routine; on Z-only boards
           it homes what it has and reports the missing axes.

        2. (when ``self.scope.motion.has_turret()`` is True) Absolute T-axis
           move to the position that matches ``settings['objective_id']``
           -- falls back to position 1 if the objective isn't in the
           turret config. Updates ``settings['turret_position']`` so
           later code reads the actual position.

        Headless / REST callers can use this exact same call to apply
        the standard startup orchestration without copy-pasting from
        the App.

        Args:
            disable_homing: If True, skip the home step but still run
                turret-positioning. Matches the App's ``--no-home``
                CLI flag semantics.
        """
        # Local import to avoid circular import at module load — ui_helpers
        # imports many UI modules but the functions used here (move_home,
        # move_absolute_position) operate on the scope and don't actually
        # need a GUI surface.
        from ui.ui_helpers import move_home, move_absolute_position
        from modules.sequential_io_executor import IOTask

        if not disable_homing:
            self.io_executor.put(IOTask(move_home, args=('ALL',)))

        if self.scope.motion.has_turret():
            objective_id = self.settings.get('objective_id')
            turret_position = self.scope.motion.get_turret_position_for_objective_id(
                objective_id=objective_id,
                persisted_position=self.settings.get('turret_position'),
            )
            if turret_position is None:
                DEFAULT_POSITION = 1
                logger.info(
                    f"Turret position for set objective {objective_id} not "
                    f"in turret objectives configuration. Setting to "
                    f"position {DEFAULT_POSITION}")
                turret_position = DEFAULT_POSITION

            self.settings['turret_position'] = turret_position
            self.io_executor.put(IOTask(
                move_absolute_position,
                kwargs={
                    "axis": 'T',
                    "pos": turret_position,
                    "wait_until_complete": True,
                },
            ))
