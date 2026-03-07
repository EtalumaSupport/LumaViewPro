"""
ScopeSession — GUI-independent state container for a microscope session.

Consolidates the shared state that was previously scattered across module-level
globals in lumaviewpro.py.  LumaViewPro, the REST API, and standalone scripts
can each create (or share) a ScopeSession instance and pass it to the
functions in config_helpers and scope_commands.

Usage
-----
    from modules.scope_session import ScopeSession

    session = ScopeSession.create(settings=settings, source_path=source_path)
    # or, for headless / test use:
    session = ScopeSession.create_headless(settings=settings)
"""

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
            import lumascope_api
            scope = lumascope_api.Lumascope()

        if io_executor is None:
            io_executor = SequentialIOExecutor(name="IO")

        if camera_executor is None:
            camera_executor = SequentialIOExecutor(name="CAMERA")

        # Optional helpers — import and construct if available
        wellplate_loader = None
        coordinate_transformer = None
        objective_helper = None

        try:
            import labware_loader
            wellplate_loader = labware_loader.WellPlateLoader()
        except Exception as e:
            logger.warning(f"[ScopeSession] Could not load wellplate loader: {e}")

        try:
            import coord_transformations
            coordinate_transformer = coord_transformations.CoordinateTransformer()
        except Exception as e:
            logger.warning(f"[ScopeSession] Could not load coordinate transformer: {e}")

        try:
            import objectives_loader
            objective_helper = objectives_loader.ObjectiveLoader()
        except Exception as e:
            logger.warning(f"[ScopeSession] Could not load objective helper: {e}")

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

    # --- LED commands ---

    def leds_off(self, callback=None):
        import modules.scope_commands as scope_commands
        scope_commands.leds_off(self.scope, self.io_executor, callback=callback)

    def led_on(self, channel, illumination, callback=None, cb_kwargs=None):
        import modules.scope_commands as scope_commands
        scope_commands.led_on(
            self.scope, self.io_executor, channel, illumination,
            callback=callback, cb_kwargs=cb_kwargs,
        )

    def led_off(self, channel, callback=None, cb_kwargs=None):
        import modules.scope_commands as scope_commands
        scope_commands.led_off(
            self.scope, self.io_executor, channel,
            callback=callback, cb_kwargs=cb_kwargs,
        )

    def led_on_sync(self, channel, illumination, timeout=5):
        import modules.scope_commands as scope_commands
        scope_commands.led_on_sync(
            self.scope, self.io_executor, channel, illumination, timeout=timeout,
        )

    # --- Motion commands ---

    def move_absolute(self, axis, pos, wait_until_complete=False,
                      overshoot_enabled=True, callback=None, cb_kwargs=None):
        import modules.scope_commands as scope_commands
        scope_commands.move_absolute(
            self.scope, self.io_executor, axis, pos,
            wait_until_complete=wait_until_complete,
            overshoot_enabled=overshoot_enabled,
            callback=callback, cb_kwargs=cb_kwargs,
        )

    def move_relative(self, axis, um, wait_until_complete=False,
                      overshoot_enabled=True, callback=None, cb_kwargs=None):
        import modules.scope_commands as scope_commands
        scope_commands.move_relative(
            self.scope, self.io_executor, axis, um,
            wait_until_complete=wait_until_complete,
            overshoot_enabled=overshoot_enabled,
            callback=callback, cb_kwargs=cb_kwargs,
        )

    def move_home(self, axis, callback=None, cb_args=None):
        import modules.scope_commands as scope_commands
        scope_commands.move_home(
            self.scope, self.io_executor, axis,
            callback=callback, cb_args=cb_args,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_executors(self):
        """Start the IO and camera executors."""
        self.io_executor.start()
        self.camera_executor.start()

    def stop_executors(self):
        """Stop the IO and camera executors."""
        self.io_executor.stop()
        self.camera_executor.stop()
