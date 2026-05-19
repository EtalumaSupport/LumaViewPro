#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import contextlib
import datetime
import os
import pathlib
import threading
import time
import warnings

import numpy as np

# Import Lumascope Hardware files
from drivers.motorboard import MotorBoard
from drivers.ledboard import LEDBoard
from drivers.pyloncamera import PylonCamera
from modules.exceptions import CaptureError, ConfigError
from modules.lumascope_api import _constants as _api_constants
try:
    from drivers.idscamera import IDSCamera
except ImportError:
    IDSCamera = None
# FX2 (Lumaview Classic LS560/LS620/LS720) — the import side-effect is
# the entire point: it fires the @camera_registry.register('fx2') and
# @led_registry.register('fx2') decorators inside the module. Nothing
# in this file references fx2driver names directly; the registry
# instantiates FX2Camera + FX2LEDController via 'auto' fallthrough when
# Pylon/IDS aren't found. Wrapped in try/except so dev machines without
# pyusb / libusb1 don't crash LVP at startup (matches IDS pattern above).
try:
    import drivers.fx2driver  # noqa: F401
except ImportError:
    pass
from drivers.camera import Camera
from drivers.simulated_camera import SimulatedCamera
from drivers.simulated_motorboard import SimulatedMotorBoard
from drivers.simulated_ledboard import SimulatedLEDBoard
from drivers.null_motorboard import NullMotionBoard
from drivers.null_ledboard import NullLEDBoard
from drivers.protocols import MotorBoardProtocol, LEDBoardProtocol
from drivers.registry import motor_registry, led_registry, camera_registry
from modules.scope_capabilities import ScopeCapabilities

# Import additional libraries
from lvp_logger import logger, version
import logging as _logging

_api_log = _logging.getLogger('LVP.api')

import modules.autofocus_functions as autofocus_functions
import modules.common_utils as common_utils
import modules.coord_transformations as coord_transformations
from lib import profile_trace
import modules.objectives_loader as objectives_loader
import modules.image_utils as image_utils
from modules.sequential_io_executor import SequentialIOExecutor, IOTask
from modules.frame_validity import FrameValidity
from modules.notification_center import notifications


class AxisState:
    """Possible states for a motion axis."""
    UNKNOWN = 'unknown'    # Not homed / state not known
    IDLE = 'idle'          # At known position, not moving
    MOVING = 'moving'      # Move commanded, not yet arrived
    HOMING = 'homing'      # Homing sequence in progress


# ---------------------------------------------------------------------------
# Rule 14 helpers (notify on failure)
#
# #632/#539 introduced `_try_connect_board` to replace the silent
# `try/except: NullBoard()` pattern that hid LED-side failures. The audit
# (docs/AUDIT_RULE_14_NOTIFY_2026-04-24.md) flagged two places that still
# needed this plus the camera path, and recommended hoisting the helpers
# to module scope so they can be reused by `__init__`, `create_diagnostic`,
# and any future connect path without duplicating the error-class routing.
# Keep the module-scope helpers as the single source of truth; call sites
# should be one-liners.
# ---------------------------------------------------------------------------

def _notify_board_failure(label, short, message):
    """Surface a board-connect failure to the user via notification_center.

    Safe to call from any thread. Falls back to a debug log if the
    notification_center import fails (e.g. during very-early startup).
    """
    try:
        from modules.notification_center import notifications
        notifications.warning(label, f"{label} {short}", message)
    except Exception as nx:
        logger.debug(f'{label}: notification center unavailable: {nx}')


def _try_connect_board(label, ctor, null_ctor):
    """Construct a board, classify any failure, notify the user, fall back
    to `null_ctor()` so callers don't crash on missing hardware.

    The board constructor (LEDBoard / MotorBoard / ...) calls
    SerialBoard.connect() internally, which catches its OWN exceptions and
    logs without re-raising. That means a PermissionError on open leaves
    `board.found=True` (port was discovered) but `board.driver=None` (open
    failed) -- we detect that here and surface it as a clear failure instead
    of silently substituting Null*.

    Rule 14: every case logs visibly and notifies the user with an
    actionable, error-class-specific message.
    """
    try:
        board = ctor()
        if not getattr(board, 'found', False):
            logger.error(f'{label}: not detected on USB')
            _notify_board_failure(label, "not detected",
                f"{label} not found on USB. Check USB cable and 24V power.")
            return null_ctor()
        if getattr(board, 'driver', None) is None:
            logger.error(f'{label}: detected on {board.port} but driver failed to open '
                         f'(port may be held by another program -- Thonny, etc.)')
            _notify_board_failure(label, "port in use or unreachable",
                f"{label} detected on {board.port} but the port could not be opened. "
                f"Close other programs holding the port (Thonny, serial monitors), "
                f"then restart LVP.")
            return null_ctor()
        return board
    except PermissionError as e:
        logger.error(f'{label}: PermissionError opening port: {e}')
        _notify_board_failure(label, "port in use",
            f"{label} port is in use by another program (e.g. Thonny). "
            f"Close the other program and restart LVP to reconnect.")
        return null_ctor()
    except FileNotFoundError as e:
        logger.error(f'{label}: FileNotFoundError on port: {e}')
        _notify_board_failure(label, "port not found",
            f"{label} port disappeared during connect. Check USB cable.")
        return null_ctor()
    except Exception as e:
        logger.error(f'{label}: connect failed: {type(e).__name__}: {e}', exc_info=True)
        _notify_board_failure(label, "connect failed",
            f"Could not connect to {label}: {type(e).__name__}: {e}")
        return null_ctor()


def _notify_camera_failure(exc):
    """Rule 14 audit A1 -- surface camera-init failure to the user.

    The camera registry raises a variety of exception types depending on
    which backend (pypylon, ids_peak, FX2, simulated). pypylon's
    RuntimeException for "camera already open in another application"
    is the high-frequency case that Pylon Viewer / a second LVP instance
    produces and deserves a dedicated message.
    """
    exc_type = type(exc).__name__
    # Don't import pypylon at module load (adds cold-start time on
    # non-Pylon rigs). Match by type name string instead.
    if exc_type in ('RuntimeException', 'GenericException', 'LogicalErrorException'):
        title = "Camera in use"
        body = (f"Camera appears to be open in another application "
                f"(Pylon Viewer, another LVP instance, etc.). "
                f"Close it and restart LVP. ({exc_type}: {exc})")
    elif isinstance(exc, PermissionError):
        title = "Camera port in use"
        body = (f"Camera port is in use by another program. "
                f"Close the other program and restart LVP. ({exc})")
    elif isinstance(exc, FileNotFoundError):
        title = "Camera not detected"
        body = (f"Camera not found. Check USB cable and power. ({exc})")
    else:
        title = "Camera not initialized"
        body = (f"Could not connect to camera: {exc_type}: {exc}. "
                f"Check USB cable, power, and close other programs that "
                f"may hold the camera.")
    _notify_board_failure("Camera", title, body)


class Lumascope():

    # --- Input validation constants ---
    LED_MAX_MA = 1000       # Maximum LED current in milliamps (matches firmware CH_MAX)
    # LED channel set comes from self._led_driver.available_channels() — varies by
    # Canonical home for these is `_constants.py`; alias on the class so
    # existing callers (`scope._VALID_AXIS_NAMES`, `Lumascope.MOTOR_POSITION_LIMIT`)
    # keep working. Sub-API modules import from `_constants.py` directly
    # to avoid a circular dep with this file.
    _VALID_AXIS_NAMES = _api_constants._VALID_AXIS_NAMES
    MOTOR_POSITION_LIMIT = _api_constants.MOTOR_POSITION_LIMIT

    def __init__(self, simulate: bool = False, camera_type: str = 'auto',
                 register_atexit: bool = True,
                 register_metrics: bool = True):
        """Initialize Microscope.

        Args:
            simulate: If True, use simulated hardware (no USB devices needed).
            camera_type: Camera registry kind. 'auto' (default) tries the
                registered real cameras in descending priority order
                (Pylon → IDS today). Accepted explicit values: 'pylon',
                'ids', 'sim', or any other key registered in
                `drivers/registry.py::camera_registry`. Post-B2 this is
                the only parameter the caller needs to steer driver
                selection -- motion and LED drivers always use 'auto'.
            register_atexit: If True (default), register a Python atexit
                hook that turns off all LEDs and disconnects on
                interpreter shutdown. Tests that construct Lumascope
                outside the Kivy app should leave this enabled -- the LED
                stays on if a test crashes mid-LED-on otherwise. Set to
                False only when the caller has its own equivalent
                shutdown path that supersedes the atexit hook.
            register_metrics: If True (default), construct a
                MetricsLogger on this Lumascope. Doesn't START it --
                callers must call ``self.metrics_logger.start(scheduler)``
                with an environment-appropriate Scheduler (Kivy app
                uses KivyClockScheduler, REST/headless use
                ThreadingTimerScheduler). Tests that don't need
                periodic logging set False.
        """
        self._simulated = simulate
        self._coordinate_transformer = coord_transformations.CoordinateTransformer()
        self._objectives_loader = objectives_loader.ObjectiveLoader()

        # LED state slots (_led_listeners, _led_state, _led_owners,
        # _led_owner_lock, _led_listeners_lock, _led_lock) live on
        # IlluminationAPI per Wave 7 Phase 3d.

        # Camera state slots (_camera_listeners + lock, _frame_buffer,
        # _capturing_event, _focusing_event, _capture_return,
        # _autofocus_return, _suppress_value_warnings, _scale_bar,
        # _camera_cache + lock, _binning_size, _camera_temp_event,
        # _camera_temp_unschedule_fn, frame_validity) live on ImagingAPI
        # per Wave 7 Phase 4d.

        # ----- Motion Control Board -----
        # Constructed BEFORE MotionAPI so MotionAPI._driver resolves on
        # the first call. Driver selection goes through the motor registry
        # (audit B2) -- 'auto' tries real drivers in descending priority
        # order and falls back to NullMotionBoard if all fail, so no
        # manual try/except needed.
        motor_kwargs: dict = {}
        if simulate:
            from modules.settings_init import settings
            motor_kwargs['model'] = (settings.get('microscope', 'LS850')
                                     if settings else 'LS850')
        self._motion_driver: MotorBoardProtocol = motor_registry.create(
            'auto', simulate=simulate, **motor_kwargs
        )
        if simulate:
            logger.info(
                f'[SCOPE API ] Using SIMULATED Motor Board '
                f'(model={motor_kwargs.get("model")})'
            )

        # ----- MotionAPI (Wave 7 Phase 2c) -----
        # Constructed AFTER the motion driver so _driver resolves correctly.
        # init_axes() sizes per-axis dicts to detect_present_axes(); then
        # start_monitor() spawns the background poll thread. NullMotionBoard
        # returns [] from detect_present_axes(), so a system with no motor
        # hardware ends up with empty dicts throughout.
        from modules.lumascope_api.motion import MotionAPI  # local-import: avoid cycle
        self.motion = MotionAPI(self, self._motion_driver)
        present_axes = self._motion_driver.detect_present_axes()
        self.motion.init_axes(present_axes)
        self.motion.start_monitor()

        # ----- LED Control Board -----
        # Same registry-based selection as motion (audit B2).
        self._led_driver: LEDBoardProtocol = led_registry.create('auto', simulate=simulate)
        if simulate:
            logger.info('[SCOPE API ] Using SIMULATED LED Board')

        # ----- Camera -----
        # Driver selection via camera_registry (audit B2). `camera_type`
        # accepts: 'auto' (tries pylon → ids by priority), 'pylon', 'ids',
        # 'sim', or any other registered camera kind. Default 'auto' is
        # the right choice for most callers; the pre-B2 default was
        # "pylon" which skipped auto-detect — callers that rely on that
        # continue to pass camera_type='pylon' explicitly.
        # _frame_buffer slot moved to ImagingAPI in Wave 7 Phase 4d.
        self._camera_driver = None
        camera_kwargs: dict = {}
        if simulate:
            camera_kwargs['z_position_func'] = lambda: self._motion_driver.current_pos('Z')
        try:
            self._camera_driver: Camera = camera_registry.create(
                camera_type, simulate=simulate, **camera_kwargs
            )
            if simulate:
                self._camera_driver.load_cycle_images()
                logger.info('[SCOPE API ] Using SIMULATED Camera')
        except Exception as _cam_exc:
            logger.exception('[SCOPE API ] Camera Board Not Initialized')
            # Rule 14 A1: pre-fix code logged only; the user saw no popup and
            # every camera-dependent UI action silently returned None/False.
            # Same pattern #632/#539 fixed for the LED + motor boards.
            _notify_camera_failure(_cam_exc)

        # ----- ScopeCapabilities (audit B7) -----
        # Single source of truth for "what does this scope have" — built
        # once from the three drivers, frozen thereafter. Callers should
        # prefer `scope.capabilities.*` over the wrapper methods below.
        # Runtime connection state (`motor_connected`, `led_connected`)
        # stays as live properties on Lumascope — those must reflect
        # disconnects and can't be snapshotted.
        self.capabilities = ScopeCapabilities.from_drivers(
            motion=self._motion_driver,
            led=self._led_driver,
            camera=self._camera_driver,
            led_max_ma=self.LED_MAX_MA,
        )

        # ----- Sub-API wiring (Wave 7 Phase 1+) -----
        # Six sub-APIs: motion, illumination, imaging, diagnostics,
        # capabilities, io. motion was already constructed above (Phase 2c
        # requires earlier construction so init_axes / start_monitor can run
        # before the LED/camera drivers are set up). Remaining sub-APIs:
        from modules.lumascope_api.illumination import IlluminationAPI
        from modules.lumascope_api.imaging import ImagingAPI
        from modules.lumascope_api.diagnostics import DiagnosticsAPI
        from modules.lumascope_api.io import IOAPI
        self.illumination = IlluminationAPI(self, self._led_driver)
        self.imaging = ImagingAPI(self, self._camera_driver)
        self.diagnostics = DiagnosticsAPI(self)
        self.io = IOAPI(self)

        # Partial-hardware notification deferred to initialize(config) —
        # we need scope-config knowledge to distinguish "LS620 correctly
        # has no motor" from "LS820 motor failed to connect."

        # Track whether any real hardware was found.
        # Camera check reads the (private) driver handle directly because
        # the public `self.camera` attribute is the new ImagingAPI in
        # Wave 7 Phase 1 -- not the driver.
        self._no_hardware = (
            not simulate
            and isinstance(self._led_driver, NullLEDBoard)
            and isinstance(self._motion_driver, NullMotionBoard)
            and self._camera_driver is None
        )
        if self._no_hardware:
            logger.warning('[SCOPE API ] No hardware detected (LED, motor, and camera all failed to initialize)')

        # --- Thread synchronization (CR-2 / CR-6) ---
        # _state_lock protects individual shared-state reads/writes
        self._state_lock = threading.Lock()
        # Per-device locks — each device communicates over a different port
        # and can operate independently. Split from the old global _hw_lock
        # to allow LED stim pulses during camera grabs and motor moves.
        # Threading audit §10.2 — wrapped with TimedLock for contention tracing.
        # _led_lock relocated to IlluminationAPI per Wave 7 Phase 3d.
        self._cam_lock = profile_trace.TimedLock(threading.RLock(), name="lumascope._cam_lock")
        # Global lock for multi-device atomic operations (e.g., LED on + capture + LED off).
        # Only used by acquire_exclusive() — individual methods use per-device locks.
        self._hw_lock = threading.RLock()

        # _capturing_event, _focusing_event, _capture_return,
        # _autofocus_return moved to ImagingAPI in Wave 7 Phase 4d.
        # _homing_event and _turreting_event live on MotionAPI (Phase 2c).
        self.last_focus_score = None

        # self.is_stepping = False         # Is the microscope currently attempting to capture a step
        # self.step_capture_return = False # Will be image at step settings if ready to pull, else False

        self._labware = None              # The labware currently installed
        self._objective = None            # The objective currently selected/installed
        self._turret_config = {}          # The objectives loaded into the turret (if present)
        self._stage_offset = None         # The stage offset for the microscope
        self._last_turret_position = None # Stores the last known turret position
        self.engineering_mode = False      # Set by UI to enable engineering features
        # _suppress_value_warnings moved to ImagingAPI in Wave 7 Phase 4d.

        # LAYER-A' executor handles. Registered post-construction via
        # register_executors() so that tests using `Lumascope(simulate=True)`
        # can construct without needing real executors. Methods that
        # submit IOTasks (led_on_async, move_absolute_async, etc.) will
        # raise RuntimeError if executors aren't registered.
        self._camera_executor = None
        self._io_executor = None
        self._file_io_executor = None

        # LAYER-I source-path handle. Registered via register_source_path()
        # at startup. load_protocol() / create_protocol() use it to find
        # data/tiling.json without UI callers having to know the layout.
        self._source_path = None

        # Frame validity, camera_cache, scale_bar, _binning_size, +
        # _camera_listeners/_frame_buffer/_capturing_event/_focusing_event/
        # _capture_return/_autofocus_return/_suppress_value_warnings/
        # _camera_temp_event init relocated to ImagingAPI.__init__ in
        # Wave 7 Phase 4d. _load_camera_timing + _populate_camera_cache
        # are now ImagingAPI methods and run automatically during
        # ImagingAPI.__init__. Lumascope wires up the motion-settle check
        # against the relocated frame_validity instance below.
        def _motion_settle_check(source: str) -> bool:
            # For absent axes (e.g., LS820 has no X/Y), treat UNKNOWN as settled.
            # Axes that were never homed or moved stay UNKNOWN -- they shouldn't
            # block frame validity for sources that don't apply.
            idle_or_absent = (AxisState.IDLE, AxisState.UNKNOWN)
            if source == 'z_move':
                return self.motion.get_axis_state('Z') in idle_or_absent
            elif source == 'xy_move':
                return (self.motion.get_axis_state('X') in idle_or_absent and
                        self.motion.get_axis_state('Y') in idle_or_absent)
            elif source == 'turret':
                return self.motion.get_axis_state('T') in idle_or_absent
            return True
        self.imaging.frame_validity.set_settle_check(_motion_settle_check)
        # _load_camera_timing relocated to ImagingAPI; called during
        # ImagingAPI.__init__ via the settle-check setup completion.
        self.imaging._load_camera_timing()

        # Populate position cache from firmware so get_current_position()
        # returns correct values immediately (not 0.0 from empty cache).
        # Critical for standalone scripts that read position right after
        # creating Lumascope (e.g., backlash characterization).
        if self.motor_connected:
            try:
                self.motion.refresh_position_cache()
            except Exception:
                pass  # OK — cache stays at 0.0 if firmware unresponsive

        # LVP-A-13: pre-construct MetricsLogger so every Lumascope user
        # (Kivy app, REST API, headless tests, CLI tools) shares the
        # same metrics surface — engineering plugin / status endpoints
        # can call self.metrics_logger.snapshot_executors() etc. without
        # waiting for the host to register one. Lifecycle is two-phase:
        # __init__ constructs (this block); the host calls
        # self.metrics_logger.start(scheduler) once it knows which
        # scheduler is appropriate for its environment. Doesn't start
        # any timers / Clock events here, so test fixtures don't pay
        # for periodic work they don't want.
        #
        # executor_bundle is None at construction; the host calls
        # register_executor_bundle() after the bundle exists, before
        # calling metrics_logger.start.
        self.metrics_logger = None
        self._executor_bundle = None
        if register_metrics:
            try:
                from modules.metrics_logger import MetricsLogger
                self.metrics_logger = MetricsLogger(
                    scope=self,
                    executor_bundle=None,  # set later via register_executor_bundle
                    settings={},           # ditto
                )
            except Exception as _e:
                logger.warning(
                    f'[SCOPE API ] MetricsLogger construction failed: {_e}')

        # LVP-A-7: register the emergency-shutdown atexit hook so EVERY
        # Lumascope user (Kivy app, REST server, headless tests, CLI
        # tools) gets the LED-off-and-disconnect safety net automatically.
        # Was previously inline in lumaviewpro.py:541-549, leaving every
        # non-GUI entry point silently unprotected — exactly the failure
        # mode the comment cited (LED stays on, sample overheats).
        if register_atexit:
            try:
                import atexit
                atexit.register(self._emergency_shutdown)
            except Exception as _e:
                logger.warning(
                    f'[SCOPE API ] atexit registration failed: {_e}')


    def initialize(self, config) -> None:
        """Configure scope from connected to ready-to-use.

        Call once after construction.  Sets all scope-level hardware
        configuration.  Does NOT set per-layer camera settings (gain,
        exposure, auto-gain) -- those are the caller's responsibility
        for the active layer.

        Args:
            config: ScopeInitConfig instance with all scope-level settings.
        """
        self._notify_partial_hardware(config)
        self.illumination.leds_off()
        self.set_labware(config.labware)
        if config.turret_config:
            self.set_turret_config(config.turret_config)
        self.set_objective(config.objective_id)
        self.imaging.set_binning_size(config.binning_size)
        self.imaging.set_frame_size(config.frame_width, config.frame_height)
        self.set_stage_offset(config.stage_offset)
        self.imaging.set_scale_bar(enabled=config.scale_bar_enabled)
        self.motion.set_acceleration_limit(val_pct=config.acceleration_pct)
        logger.info('[SCOPE API ] Scope initialized')

    def _notify_partial_hardware(self, config) -> None:
        """Warn user about missing hardware, filtered by scope expectations.

        An LS620 with no motor is not a failure -- its scopes.json says
        Focus/XYStage/Turret are all false. Only warn for hardware the
        scope was supposed to have. Simulators never warn.
        """
        if self._simulated:
            return
        missing = []
        if config.expects_led and isinstance(self._led_driver, NullLEDBoard):
            missing.append("LED Board")
        if config.expects_motion and isinstance(self._motion_driver, NullMotionBoard):
            missing.append("Motor Controller")
        if not getattr(self._camera_driver, 'active', None):
            missing.append("Camera")
        if missing:
            notifications.warning(
                "Hardware", "Partial Hardware Detected",
                f"Not connected: {', '.join(missing)}. Some features will be unavailable.",
            )



    # --- Camera state cache accessors (zero SDK calls) ---



    @property
    def camera_active(self) -> bool:
        return self.imaging.camera_active

    @property
    def camera_gain(self) -> float:
        return self.imaging.camera_gain

    @property
    def camera_exposure_ms(self) -> float:
        return self.imaging.camera_exposure_ms

    @property
    def camera_frame_size(self) -> dict:
        return self.imaging.camera_frame_size

    @property
    def camera_max_frame_size(self) -> dict:
        return self.imaging.camera_max_frame_size

    @property
    def camera_min_frame_size(self) -> dict:
        return self.imaging.camera_min_frame_size

    @property
    def camera_max_exposure(self) -> float | None:
        return self.imaging.camera_max_exposure

    @property
    def camera_max_gain(self) -> float | None:
        return self.imaging.camera_max_gain

    @property
    def camera_pixel_format(self) -> str:
        return self.imaging.camera_pixel_format

    @property
    def is_capturing(self) -> bool:
        return self.imaging.is_capturing

    @is_capturing.setter
    def is_capturing(self, value: bool) -> None:
        self.imaging.is_capturing = value

    @property
    def is_focusing(self) -> bool:
        return self.imaging.is_focusing

    @is_focusing.setter
    def is_focusing(self, value: bool) -> None:
        self.imaging.is_focusing = value

    @property
    def capture_return(self):
        return self.imaging.capture_return

    @capture_return.setter
    def capture_return(self, value) -> None:
        self.imaging.capture_return = value

    @property
    def autofocus_return(self):
        return self.imaging.autofocus_return

    @autofocus_return.setter
    def autofocus_return(self, value) -> None:
        self.imaging.autofocus_return = value

    @property
    def scale_bar_config(self) -> dict:
        return self.imaging.scale_bar_config

    @property
    def scale_bar_enabled(self) -> bool:
        return self.imaging.scale_bar_enabled

    # --- Frame validity accessors (per LAYER-F / Rule 1) ---
    # External callers must use these instead of reaching through
    # `self.frame_validity.X` directly. The frame_validity attribute
    # remains accessible for tests that need to introspect pending state.

    @property
    def frame_is_valid(self) -> bool:
        return self.imaging.frame_is_valid

    def frames_until_valid(self, exclude_sources: tuple=()) -> int:
        return self.imaging.frames_until_valid(exclude_sources)

    def count_frame(self) -> None:
        return self.imaging.count_frame()

    # --- Executor-backed command API (LAYER-A' / Rule 2) ---
    #
    # Single canonical path for hardware operations that need executor
    # dispatch: caller invokes scope.X_async(...) or scope.X_sync(...);
    # Lumascope picks the right executor internally. Replaces the older
    # modules/scope_commands.py helper functions where the caller had
    # to pass an executor on every call (parallel-paths anti-pattern).

    def register_executors(self, *, camera_executor=None, io_executor=None,
                           file_io_executor=None) -> None:
        """Register the executor handles used by the X_async / X_sync command methods.

        Call once at startup after the executors are constructed. Tests
        that don't drive the executor-backed API can skip this -- those
        methods raise RuntimeError if invoked without executors registered.

        Args:
            camera_executor: Executor for camera-bound IOTasks.
            io_executor: Executor for general IO/motion IOTasks.
            file_io_executor: Executor for file-IO IOTasks.
        """
        self._camera_executor = camera_executor
        self._io_executor = io_executor
        self._file_io_executor = file_io_executor

    def register_executor_bundle(self, executor_bundle, settings=None) -> None:
        """LVP-A-13: register the ExecutorBundle + settings dict for MetricsLogger.

        Lumascope construction (__init__) creates a MetricsLogger but
        cannot fill in the bundle yet -- the bundle is created later by
        ExecutorRegistry.create_default in the host's startup path.
        Call this once after the bundle exists, BEFORE calling
        ``self.metrics_logger.start(scheduler)``. Settings dict is
        optional; defaults to ``{}`` if MetricsLogger was created with
        a placeholder.

        Args:
            executor_bundle: ExecutorBundle instance to attach.
            settings: Optional settings dict for MetricsLogger.
        """
        self._executor_bundle = executor_bundle
        if self.metrics_logger is not None:
            self.metrics_logger._bundle = executor_bundle
            if settings is not None:
                self.metrics_logger._settings = settings

    def register_source_path(self, source_path) -> None:
        """Register the LVP source/data path used by protocol API methods.

        Used by ``load_protocol()`` and ``create_protocol()`` to find
        ``data/tiling.json``. Called once at startup. Tests that don't
        drive the protocol API can skip this.

        Args:
            source_path: Path-like to the LVP source/data root.
        """
        self._source_path = source_path

    def _tiling_configs_path(self):
        """Resolve data/tiling.json from the registered source path."""
        import pathlib
        if self._source_path is None:
            raise RuntimeError(
                "Lumascope.load_protocol/create_protocol require "
                "register_source_path() to have been called."
            )
        return pathlib.Path(self._source_path) / "data" / "tiling.json"

    # --- Protocol API (LAYER-I / LV-16) ---

    def load_protocol(self, file_path) -> 'Protocol':
        """Load a Protocol from disk.

        Wraps ``Protocol.from_file(...)`` and resolves
        ``data/tiling.json`` from the registered source_path.

        Args:
            file_path: Path to the protocol file.

        Returns:
            Protocol: The loaded Protocol instance.

        Raises:
            ProtocolFormatError: On format issues (same surface as
                Protocol.from_file).
        """
        from modules.protocol import Protocol
        return Protocol.from_file(
            file_path=file_path,
            tiling_configs_file_loc=self._tiling_configs_path(),
        )

    def create_protocol(self, *, config=None, input_config=None,
                        empty_config=None) -> 'Protocol':
        """Construct a Protocol in-memory.

        Three modes (pass exactly one):
          - config={...}: full config dict passed to Protocol() directly.
          - input_config={...}: partial config (positions, layer_configs,
            etc.); routed through Protocol.from_config which fills defaults.
          - empty_config={...}: labware/objective config for an empty-steps
            protocol; routed through Protocol.create_empty.
        tiling_configs_file_loc is resolved internally from the registered
        source_path.

        Args:
            config: Full config dict, or None.
            input_config: Partial config dict, or None.
            empty_config: Empty-steps config dict, or None.

        Returns:
            Protocol: Newly constructed Protocol instance.

        Raises:
            ValueError: If exactly one of config/input_config/empty_config
                was not provided.
        """
        from modules.protocol import Protocol
        provided = sum(
            1 for x in (config, input_config, empty_config) if x is not None
        )
        if provided != 1:
            raise ValueError(
                "create_protocol(): pass exactly one of config=, "
                "input_config=, or empty_config="
            )
        tcfg = self._tiling_configs_path()
        if input_config is not None:
            return Protocol.from_config(
                input_config=input_config,
                tiling_configs_file_loc=tcfg,
            )
        if empty_config is not None:
            return Protocol.create_empty(
                config=empty_config,
                tiling_configs_file_loc=tcfg,
            )
        return Protocol(
            tiling_configs_file_loc=tcfg,
            config=config,
        )

    @staticmethod
    def sanitize_step_name(input: str) -> str:
        """Sanitize a step name string.

        Thin pass-through to ``Protocol.sanitize_step_name`` so UI /
        module callers don't need to import the Protocol data class for
        this utility.

        Args:
            input: Raw step name to sanitize.

        Returns:
            str: Sanitized step name.
        """
        from modules.protocol import Protocol
        return Protocol.sanitize_step_name(input=input)

    def _require_executor(self, executor, name):
        if executor is None:
            raise RuntimeError(
                f"Lumascope.{name} requires register_executors() to have "
                f"been called with the relevant executor handle."
            )
        return executor

    # --- LED command API ---
    # All LED methods relocated to IlluminationAPI in Wave 7 Phase
    # 3c/3d; forwarders retired in 3f. Callers use scope.illumination.

    # --- Camera command API ---

    def set_gain_sync(self, gain, *, timeout=5) -> None:
        return self.imaging.set_gain_sync(gain, timeout=timeout)

    def set_exposure_sync(self, exposure, *, timeout=5) -> None:
        return self.imaging.set_exposure_sync(exposure, timeout=timeout)

    def capture_and_wait_sync(self, *, timeout: float=30, **kwargs) -> 'np.ndarray | bool | None':
        return self.imaging.capture_and_wait_sync(timeout=timeout, **kwargs)

    # LED change listeners + LED ownership: relocated to IlluminationAPI
    # in Wave 7 Phase 3d; forwarders retired in 3f.

    def save_camera_state(self, tag: str) -> dict:
        return self.imaging.save_camera_state(tag)

    def restore_camera_state(self, snapshot: dict) -> None:
        return self.imaging.restore_camera_state(snapshot)

    # ------------------------------------------------------------------
    # Camera change listeners
    # ------------------------------------------------------------------

    def add_camera_listener(self, listener) -> None:
        return self.imaging.add_camera_listener(listener)

    def remove_camera_listener(self, listener) -> None:
        return self.imaging.remove_camera_listener(listener)


    def axes_present(self) -> list[str]:
        """Get list of axes physically present on this scope.

        Thin wrapper over `self.capabilities.axes`. New code should
        prefer reading from `scope.capabilities.axes` directly.

        Returns:
            list[str]: e.g. ['Z'], ['X', 'Y', 'Z'], or ['X', 'Y', 'Z', 'T']
        """
        return list(self.capabilities.axes)

    def has_axis(self, axis: str) -> bool:
        """Check if an axis is physically present on this scope.

        Thin wrapper over ``self.capabilities.axes``.

        Args:
            axis: Axis name to check ("X", "Y", "Z", "T").

        Returns:
            bool: True if the axis is present.
        """
        return axis in self.capabilities.axes

    def travel_limit_um(self, axis: str) -> float:
        """Get the travel limit for an axis in um.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            float: Travel limit in um, or MOTOR_POSITION_LIMIT if unknown.
        """
        try:
            return float(self._motion_driver.motorconfig.travel_limit_um(axis))
        except Exception:
            return float(self.MOTOR_POSITION_LIMIT)

    @property
    def motor_connected(self) -> bool:
        """Whether the motor controller is connected.

        Returns:
            bool: True if a real (non-Null) motor board is connected.
        """
        return not isinstance(self._motion_driver, NullMotionBoard) and self._motion_driver.is_connected()

    @property
    def led_connected(self) -> bool:
        """Whether the LED controller is connected.

        Returns:
            bool: True if a real (non-Null) LED board is connected.
        """
        return not isinstance(self._led_driver, NullLEDBoard) and self._led_driver.is_connected()

    def lens_focal_length(self) -> float:
        """Get tube lens focal length from motorconfig.

        Returns:
            float: Focal length in mm (default 47.8).
        """
        return self._motion_driver.motorconfig.lens_focal_length()

    def pixel_size(self) -> float:
        """Get camera pixel size from motorconfig.

        Returns:
            float: Pixel size in um/pixel (default 2.0).
        """
        return self._motion_driver.motorconfig.pixel_size()

    # --- CR-6: Exclusive lock for multi-step hardware operations ---

    @contextlib.contextmanager
    def acquire_exclusive(self):
        """Context manager for multi-step hardware operations.

        Prevents interleaving of compound operations (e.g., set gain + capture).
        Uses RLock so a thread that already holds the lock can re-enter.

        Usage::

            with scope.acquire_exclusive():
                scope.set_led_ma('Blue', 10)
                image = scope.imaging.capture_and_wait()
        """
        self._hw_lock.acquire()
        try:
            yield
        finally:
            self._hw_lock.release()

    def disconnect(self) -> bool:
        """Disconnect from all hardware (LED, motion, camera).

        Best-effort teardown: every sub-system is attempted even if a
        prior one raises. State is always reset to the Null variants
        and `_invalidate_camera_cache` always runs, so a partial failure
        cannot leave the API holding a stale connected driver.

        Returns:
            bool: True if all three sub-disconnects succeeded. False if
                any sub-system raised or the camera driver returned
                False. Each failure is logged and surfaced via
                notification_center; programmatic callers can branch
                on the bool for diagnostic / shutdown-sequencing
                decisions.
        """
        logger.info('[SCOPE API ] Disconnecting from microscope...')

        # LVP-A-1: stop motors before tearing down the serial port so we
        # don't leave a stage/turret moving against an end-stop after
        # the host stops responding to status polls. Defense in depth --
        # every disconnect path benefits without relying on the caller
        # to remember.
        self.motion.stop_motion()

        # Stop the motion monitor and reset axis states -- MotionAPI.disconnect()
        # handles both: signals the monitor thread, waits for it, then resets
        # all axes to UNKNOWN and sets arrival events so waiters unblock.
        self.motion.disconnect()

        # Each sub-system: only attempt disconnect on a driver that
        # has one. Skips both the canonical no-op states (NullLEDBoard,
        # NullMotionBoard, self._camera_driver is None) and edge-case test
        # fixtures that bend the type system (e.g. `scope.led = object()`
        # for partial-hardware-warning tests). A skipped sub-system
        # counts as ok=True -- "nothing to tear down" is success, not
        # failure. Real drivers that raise inside disconnect() still
        # flip *_ok to False and fire a Rule-14 notification.
        led_ok = True
        if (not isinstance(self._led_driver, NullLEDBoard)
                and hasattr(self._led_driver, 'disconnect')):
            try:
                self._led_driver.disconnect()
            except Exception as ex:
                led_ok = False
                logger.exception(f"[SCOPE API ] LED disconnect failed: {ex}")
                notifications.error(
                    "Hardware",
                    "LED disconnect failed",
                    f"LED board teardown raised {type(ex).__name__}: {ex}. "
                    f"The serial port may be left open; reconnecting "
                    f"may require a process restart.")
        self._led_driver = NullLEDBoard()

        motion_ok = True
        if (not isinstance(self._motion_driver, NullMotionBoard)
                and hasattr(self._motion_driver, 'disconnect')):
            try:
                self._motion_driver.disconnect()
            except Exception as ex:
                motion_ok = False
                logger.exception(f"[SCOPE API ] Motion disconnect failed: {ex}")
                notifications.error(
                    "Hardware",
                    "Motor disconnect failed",
                    f"Motor board teardown raised {type(ex).__name__}: {ex}. "
                    f"The serial port may be left open; reconnecting "
                    f"may require a process restart.")
        self._motion_driver = NullMotionBoard()

        camera_ok = True
        if self._camera_driver is not None and hasattr(self._camera_driver, 'disconnect'):
            try:
                camera_ok = bool(self._camera_driver.disconnect())
            except Exception as ex:
                camera_ok = False
                logger.exception(f"[SCOPE API ] Camera disconnect failed: {ex}")
                notifications.error(
                    "Hardware",
                    "Camera disconnect failed",
                    f"Camera teardown raised {type(ex).__name__}: {ex}. "
                    f"USB resources may not be fully released until the "
                    f"app restarts.")
            self._camera_driver = None
        elif self._camera_driver is not None:
            # Camera lacked a `disconnect` method (test-fixture artifact);
            # clear the slot but don't claim success on a real teardown.
            self._camera_driver = None
        self.imaging._invalidate_camera_cache()

        all_ok = led_ok and motion_ok and camera_ok
        if all_ok:
            logger.info('[SCOPE API ] Microscope disconnected')
        else:
            logger.warning(
                f'[SCOPE API ] Microscope disconnected with errors '
                f'(led_ok={led_ok}, motion_ok={motion_ok}, '
                f'camera_ok={camera_ok})')
        return all_ok

    def _emergency_shutdown(self):
        """LVP-A-7: best-effort safety shutdown for atexit / abnormal exit.

        Guards LEDs and motor against the interpreter terminating mid-
        operation: turns off all LEDs, then disconnects (which now also
        stops motion via the LVP-A-1 chain). Swallows every exception so
        atexit completes cleanly even when the logging stack or hardware
        access is already torn down.
        """
        try:
            self.illumination.leds_off()
        except Exception:
            pass
        try:
            self.disconnect()
        except Exception:
            pass
        try:
            logger.info(
                '[SCOPE API ] _emergency_shutdown complete '
                '(LEDs off, disconnected)')
        except Exception:
            pass

    @property
    def no_hardware(self) -> bool:
        """True if no real hardware was detected (LED, motor, and camera all missing).

        Returns:
            bool: True when all three subsystems are absent or stubbed.
        """
        return self._no_hardware

    def are_all_connected(self) -> bool:
        """Check if LED, motion, and camera boards are all connected.

        Returns:
            bool: True if all three components are connected.
        """
        logger.info('[SCOPE API ] Performing connection check...')
        led = not isinstance(self._led_driver, NullLEDBoard) and self._led_driver.is_connected()
        motion = self.motor_connected
        camera = self._camera_driver is not None and self._camera_driver.is_connected()

        if not led:
            logger.info('[SCOPE API ] Connection Check: LED Board not connected')
        if not motion:
            logger.info('[SCOPE API ] Connection Check: Motion Board not connected')
        if not camera:
            logger.info('[SCOPE API ] Connection Check: Camera not connected')

        if led and motion and camera:
            logger.info('[SCOPE API ] Connection Check: All components connected')

        return led and motion and camera

    ########################################################################
    # SCOPE CONFIGURATION FUNCTIONS
    ########################################################################
    def set_labware(self, labware) -> None:
        """Set the current labware (well plate) for the microscope.

        Args:
            labware: Labware object describing the well plate geometry.
        """
        self._labware = labware

    def get_labware(self):
        """Get the currently installed labware.

        Returns:
            The current labware object, or None if not set.
        """
        return self._labware

    def set_objective(self, objective_id: str) -> None:
        """Set the active objective by ID.

        Args:
            objective_id: Objective identifier (e.g. "4x", "10x", "20x").
        """
        self._objective_id = objective_id
        self._objective = self._objectives_loader.get_objective_info(objective_id=objective_id)

    def get_current_objective_id(self) -> str | None:
        """Get the ID of the currently active objective.

        Returns:
            str | None: e.g. '20x Oly', or None if not set.
        """
        return getattr(self, '_objective_id', None)

    def get_objective_info(self, objective_id: str) -> dict:
        """Get objective metadata by ID.

        Args:
            objective_id: Objective identifier (e.g. "4x", "10x", "20x").

        Returns:
            dict: Objective info including focal_length, magnification, etc.
        """
        return self._objectives_loader.get_objective_info(objective_id=objective_id)

    def get_available_objectives(self) -> list[str]:
        """Get list of all available objective IDs.

        Returns:
            list[str]: Objective identifiers (e.g. ["4x", "10x Oly", "20x Oly"]).
        """
        return self._objectives_loader.get_objectives_list()

    def get_current_objective(self) -> dict | None:
        """Get the currently active objective info.

        Returns:
            dict | None: Active objective metadata, or None if not set.
        """
        return self._objective

    def compute_focus_score(self, image) -> float:
        """Compute focus score (Vollath F4) on an image.

        Args:
            image: numpy array (grayscale).

        Returns:
            float: Focus score. Higher = sharper.
        """
        return autofocus_functions.focus_function(
            image=image, skip_score_logging=True
        )

    def set_turret_config(self, turret_config: dict[int,str]) -> None:
        """Set the turret objective configuration.

        Args:
            turret_config: Mapping of turret position (1-4) to objective ID.
        """
        self._turret_config = turret_config

    def get_turret_config(self) -> dict:
        """Get the current turret objective configuration.

        Returns:
            dict: Mapping of turret position to objective ID.
        """
        return self._turret_config

    def set_scale_bar(self, enabled: bool, color: str=None) -> None:
        return self.imaging.set_scale_bar(enabled, color)

    def set_stage_offset(self, stage_offset) -> None:
        """Set the stage offset for coordinate transformations.

        Args:
            stage_offset: Stage offset dict with axis offsets.
        """
        self._stage_offset = stage_offset

    def get_available_binning_sizes(self) -> list:
        return self.imaging.get_available_binning_sizes()

    def set_binning_size(self, size: int) -> bool:
        return self.imaging.set_binning_size(size)

    def get_binning_size(self) -> int:
        return self.imaging.get_binning_size()

    def get_pixel_format(self) -> str | None:
        return self.imaging.get_pixel_format()

    def set_pixel_format(self, pixel_format: str) -> bool:
        return self.imaging.set_pixel_format(pixel_format)

    def get_supported_pixel_formats(self) -> tuple:
        return self.imaging.get_supported_pixel_formats()

    def set_device_link_throughput_limit(self, mode: str, value_bps: int | None=None) -> bool:
        return self.imaging.set_device_link_throughput_limit(mode, value_bps)

    def set_acquisition_stop_mode(self, mode: str) -> bool:
        return self.imaging.set_acquisition_stop_mode(mode)

    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float=1.0) -> None:
        return self.imaging.set_max_acquisition_frame_rate(enabled, fps)

    def register_frame_callback(self, cb) -> None:
        return self.imaging.register_frame_callback(cb)

    def unregister_frame_callback(self, cb) -> None:
        return self.imaging.unregister_frame_callback(cb)

    def set_bandwidth_reserve_mode(self, mode: str) -> bool:
        return self.imaging.set_bandwidth_reserve_mode(mode)

    def set_gev_packet_size(self, size_bytes: int) -> bool:
        return self.imaging.set_gev_packet_size(size_bytes)

    def set_gev_inter_packet_delay(self, delay_ticks: int) -> bool:
        return self.imaging.set_gev_inter_packet_delay(delay_ticks)

    def set_max_transfer_size(self, value_bytes: int) -> bool:
        return self.imaging.set_max_transfer_size(value_bytes)

    def set_num_max_queued_urbs(self, value: int) -> bool:
        return self.imaging.set_num_max_queued_urbs(value)


    ########################################################################
    # LED BOARD FUNCTIONS
    # Methods relocated to IlluminationAPI in Wave 7 Phase 3c / 3d;
    # forwarders retired in 3f. Callers use scope.illumination.<method>.
    ########################################################################

    ########################################################################
    # CAMERA FUNCTIONS
    ########################################################################

    def get_image(self, force_to_8bit: bool=True, earliest_image_ts: datetime.datetime | None=None, timeout: datetime.timedelta=datetime.timedelta(seconds=5), all_ones_check: bool=False, sum_count: int=1, sum_delay_s: float=0, sum_iteration_callback=None, force_new_capture: bool=False, new_capture_timeout: int=1000) -> 'np.ndarray | bool':
        return self.imaging.get_image(force_to_8bit, earliest_image_ts, timeout, all_ones_check, sum_count, sum_delay_s, sum_iteration_callback, force_new_capture, new_capture_timeout)

    def get_image_with_chunks_from_buffer(self, force_to_8bit: bool=True) -> tuple:
        return self.imaging.get_image_with_chunks_from_buffer(force_to_8bit)

    def get_image_from_buffer(self, force_to_8bit: bool=True) -> tuple:
        return self.imaging.get_image_from_buffer(force_to_8bit)

    def get_next_save_path(self, path) -> str:
        """Get the next save path given an existing save path.

        Increments the trailing numeric ID component on the filename and
        returns the new path string.

        Args:
            path: Path of the format
                ``./{save_folder}/{well_label}_{color}_{file_id}.tiff``.

        Returns:
            str: Next save path with ``file_id`` incremented.
        """

        NUM_SEQ_DIGITS = 6
        # Handle both .tiff and .ome.tiff by detecting multiple extensions if present
        # pathlib doesn't seem to handle multiple extensions natively
        path2 = pathlib.Path(path)
        extension = ''.join(path2.suffixes)
        stem = path2.name[:len(path2.name)-len(extension)]
        seq_separator_idx = stem.rfind('_')
        stem_base = stem[:seq_separator_idx]
        seq_num_str = stem[seq_separator_idx+1:]
        seq_num = int(seq_num_str)

        next_seq_num = seq_num + 1
        next_seq_num_str = f"{next_seq_num:0>{NUM_SEQ_DIGITS}}"

        new_path = path2.parent / f"{stem_base}_{next_seq_num_str}{extension}"
        return str(new_path)


    def generate_image_save_path(self, save_folder, file_root, append,
                                 tail_id_mode, output_format) -> 'pathlib.Path':
        """Generate a unique save path for an image given the naming inputs.

        Resolves collisions per ``tail_id_mode`` ("increment" auto-numbers
        until free, "if_collision" only adds a suffix on actual collision,
        ``None`` returns the bare path).

        Args:
            save_folder: Directory to save into (str or Path).
            file_root: Filename prefix.
            append: String appended to filename (e.g. color label).
            tail_id_mode: One of ``"increment"``, ``"if_collision"``, or
                ``None``.
            output_format: ``"TIFF"`` or ``"OME-TIFF"``.

        Returns:
            pathlib.Path: Full save path with appropriate extension and
                disambiguation suffix.

        Raises:
            ConfigError: If ``tail_id_mode`` is not implemented.
        """
        if isinstance(save_folder, str):
            save_folder = pathlib.Path(save_folder)

        if file_root is None:
            file_root = ""

        # Append turret position in engineering mode
        if self.engineering_mode and self._last_turret_position is not None:
            append = f"{append}_T{self._last_turret_position}"

        if output_format == 'OME-TIFF':
            file_extension = ".ome.tiff"
        else:
            file_extension = ".tiff"

        # generate filename and save path string
        if tail_id_mode == "increment":
            initial_id = '_000001'
            filename =  f"{file_root}{append}{initial_id}{file_extension}"
            path = save_folder / filename

            # Obtain next save path if current directory already exists
            while os.path.exists(path):
                path = self.get_next_save_path(path)

        elif tail_id_mode == "if_collision":
            # Write-time defense for duplicate step Names (#636). Use the
            # plain filename when no file exists; only add a numeric
            # suffix on actual collision. Keeps happy-path filenames
            # unchanged for well-formed protocols.
            base_path = save_folder / f"{file_root}{append}{file_extension}"
            if not os.path.exists(base_path):
                path = base_path
            else:
                n = 1
                while True:
                    path = save_folder / f"{file_root}{append}_{n:06d}{file_extension}"
                    if not os.path.exists(path):
                        break
                    n += 1

        elif tail_id_mode is None:
            filename =  f"{file_root}{append}{file_extension}"
            path = save_folder / filename

        else:
            raise ConfigError(f"tail_id_mode: {tail_id_mode} not implemented")

        return path

    def get_well_label(self) -> str:
        """Get the well label for the current stage XY position.

        Maps the current target X/Y stage position to a plate-frame
        coordinate using the registered labware and stage offset, then
        looks up the matching well label.

        Returns:
            str: Well label (e.g. ``"A1"``).

        Raises:
            Exception: Re-raises any error encountered reading target
                position; logged before re-raise.
        """
        labware = self._labware

        # Get target position
        try:
            x_target = self.motion.get_target_position('X')
            y_target = self.motion.get_target_position('Y')
        except Exception:
            logger.exception('[LVP API  ] Error getting target position.')
            raise

        x_target, y_target = self._coordinate_transformer.stage_to_plate(
            labware=labware,
            stage_offset=self._stage_offset,
            sx=x_target,
            sy=y_target
        )

        return labware.get_well_label(x=x_target, y=y_target)

    def generate_image_metadata(self, color, x, y, z) -> dict:
        """Build TIFF metadata dict for the current capture settings and position.

        Args:
            color (str): Channel color name (e.g. "Blue", "BF").
            x (float): Stage X position in um (or None).
            y (float): Stage Y position in um (or None).
            z (float): Stage Z position in um (or None).

        Returns:
            dict: Metadata including channel, positions, exposure, gain, pixel size.

        Raises:
            ConfigError: If objective, labware, or stage offset are not set.
        """
        def _validate():
            if self._objective is None:
                raise ConfigError(f"[SCOPE API ] Objective not set")

            if 'focal_length' not in self._objective:
                raise ConfigError(f"[SCOPE API ] Objective focal length not provided")

            if self._labware is None:
                raise ConfigError(f"[SCOPE API ] Labware not set")

            if self._stage_offset is None:
                raise ConfigError(f"[SCOPE API ] Stage offset not set")

        _validate()

        if x is None:
            x = 0
        if y is None:
            y = 0
        if z is None:
            z = 0

        px, py = self._coordinate_transformer.stage_to_plate(
            labware=self._labware,
            stage_offset=self._stage_offset,
            sx=x,
            sy=y
        )
        well_label = self.get_well_label()

        px = round(px, common_utils.max_decimal_precision('x'))
        py = round(py, common_utils.max_decimal_precision('y'))
        z  = round(z,  common_utils.max_decimal_precision('z'))

        pixel_size_um = round(
            common_utils.get_pixel_size(
                focal_length=self._objective['focal_length'],
                binning_size=self.imaging._binning_size,
            ),
            common_utils.max_decimal_precision('pixel_size'),
        )

        now_host = datetime.datetime.now()
        metadata = {
            'camera_make': 'Etaluma',
            'microscope': self.get_microscope_model(),
            'software': f'LumaViewPro {version}',
            'channel': color,
            'datetime': now_host.strftime("%Y:%m:%d %H:%M:%S"),      # Format for metadata
            'sub_sec_time': f"{now_host.microsecond // 1000:03d}",
            'objective': self._objective,
            'focal_length': self._objective['focal_length'],
            'plate_pos_mm': {'x': px, 'y': py},
            'x_pos': px,
            'y_pos': py,
            'z_pos_um': z,
            'exposure_time_ms': round(self.imaging.get_exposure_time(), common_utils.max_decimal_precision('exposure')),
            'gain_db': round(self.imaging.get_gain(), common_utils.max_decimal_precision('gain')),
            'illumination_ma': round(self.illumination.get_led_ma(color=color), common_utils.max_decimal_precision('illumination')),
            'binning_size': self.imaging._binning_size,
            'pixel_size_um': pixel_size_um,
            'well_label': well_label,
            'timestamp_iso': now_host.isoformat(timespec='microseconds'),
        }

        # Camera-side timestamp + frame-id provenance, when the camera
        # supports chunk data (Pylon ace 2 / dart M / dart R always; IDS
        # has ExposureTime/Gain but no ChunkTimestamp yet -- Stage 2 work).
        # Read the most recent chunks; they're captured at-grab-time and
        # are the right values for the most recent frame on this thread.
        try:
            handler = getattr(self._camera_driver, 'cam_image_handler', None)
            chunks = handler.get_last_chunks() if handler is not None else None
        except Exception:
            chunks = None
        if chunks is not None:
            ts_ticks = chunks.get('Timestamp')
            if ts_ticks is not None:
                metadata['timestamp_camera_ticks'] = int(ts_ticks)
            tick_hz = getattr(self._camera_driver, 'timestamp_tick_frequency_hz', None)
            if tick_hz is not None:
                metadata['timestamp_camera_tick_hz'] = int(tick_hz)
            frame_id = chunks.get('FrameID')
            if frame_id is not None:
                metadata['frame_id'] = int(frame_id)

        return metadata

    def prepare_image_for_saving(
        self,
        array: np.ndarray,
        save_folder: str,
        file_root: str,
        append: str,
        color: str,
        tail_id_mode: str,
        output_format: str,
        true_color: str,
        x,
        y,
        z,
        out_12to16: np.ndarray | None = None,
    ) -> dict:
        """Prepare an image array and metadata for saving to disk.

        Flips the image vertically, converts bit depth if needed, generates
        the save path and metadata.

        Args:
            array: Raw image array from drivers.
            save_folder: Directory to save into.
            file_root: Filename prefix.
            append: String appended to filename (e.g. color label).
            color: Color label for the filename.
            tail_id_mode: "increment" for auto-numbered files, or None.
            output_format: "TIFF" or "OME-TIFF".
            true_color: Actual channel color for metadata.
            x: Stage X position in um.
            y: Stage Y position in um.
            z: Stage Z position in um.

        Returns:
            dict: Contains 'image' (ndarray) and 'metadata' (dict with 'file_loc').
        """
        metadata = self.generate_image_metadata(color=true_color, x=x, y=y, z=z)

        if array.dtype == np.uint16:
            array = image_utils.convert_12bit_to_16bit(array, out=out_12to16)

        array = np.flip(array, 0)

        path = self.generate_image_save_path(
            save_folder=save_folder,
            file_root=file_root,
            append=append,
            tail_id_mode=tail_id_mode,
            output_format=output_format
        )

        metadata['file_loc'] = path

        return {
            'image': array,
            'metadata': metadata,
        }


    def save_image(
        self,
        array,
        save_folder = './capture',
        file_root = 'img_',
        append = 'ms',
        color = 'BF',
        tail_id_mode = "increment",
        output_format: str = "TIFF",
        true_color: str = 'BF',
        x=None,
        y=None,
        z=None,
        use_false_color_16bit: bool | None = None,
        out_12to16: np.ndarray | None = None,
        false_color_buf: np.ndarray | None = None,
        rgb_buf: np.ndarray | None = None,
    ) -> str:
        """Save an image array to a TIFF file with metadata.

        Args:
            array: Image array to save.
            save_folder: Directory to save into.
            file_root: Filename prefix.
            append: String appended to filename.
            color: Color label for the filename.
            tail_id_mode: "increment" for auto-numbered files, or None.
            output_format: "TIFF" or "OME-TIFF".
            true_color: Actual channel color for metadata.
            x: Stage X position in um.
            y: Stage Y position in um.
            z: Stage Z position in um.

        Returns:
            str: Path to the saved file.
        """

        # PIW-2: removed redundant `check_disk_space("/")` warn — checked the wrong
        # path (root, not save_folder), only logged, and protocol_image_writer.py
        # already aborts on save-folder space exhaustion. Actual write failures
        # surface through the try/except below.

        # Camera silent-stuck or grab-timeout produces None; raise typed
        # exception so the IOTask popup carries a user-friendly message
        # instead of a raw AttributeError. The deeper recovery work
        # (camera reset / USB reset on persistent stuck) lives elsewhere.
        if array is None:
            raise CaptureError(
                "Camera did not return an image. The capture was skipped; "
                "the protocol will retry on the next step."
            )

        image_data = self.prepare_image_for_saving(
            array=array,
            save_folder=save_folder,
            file_root=file_root,
            append=append,
            color=color,
            tail_id_mode=tail_id_mode,
            output_format=output_format,
            true_color=true_color,
            x=x,
            y=y,
            z=z,
            out_12to16=out_12to16,
        )

        image = image_data['image']
        metadata = image_data['metadata']
        file_loc = metadata['file_loc']

        if output_format == 'OME-TIFF':
            ome=True
        else:
            ome=False

        try:
            image_utils.write_tiff(
                data=image,
                file_loc=file_loc,
                metadata=metadata,
                ome=ome,
                color=color,
                use_false_color_16bit=use_false_color_16bit,
                false_color_buf=false_color_buf,
                rgb_buf=rgb_buf,
            )

            logger.info(f'[SCOPE API ] Saving Image to {file_loc}')
        except Exception:
            logger.exception("[SCOPE API ] Error: Unable to save. Perhaps save folder does not exist?")
            notifications.error("FileIO", "Image Save Failed",
                f"Failed to save image to {file_loc}. Check disk space and permissions.")
            raise

        # Env-gated handle-leak tracking; zero overhead when disabled.
        # Enable with LVP_HANDLE_TRACE=1.
        from lib.handle_trace import tick as _h_tick
        _h_tick('save_image')

        return file_loc


    def save_live_image(
            self,
            save_folder = './capture',
            file_root = 'img_',
            append = 'ms',
            color = 'BF',
            tail_id_mode = "increment",
            force_to_8bit: bool = True,
            output_format: str = "TIFF",
            true_color: str = 'BF',
            earliest_image_ts: datetime.datetime | None = None,
            timeout: datetime.timedelta = datetime.timedelta(seconds=5),
            all_ones_check: bool = False,
            sum_count: int = 1,
            sum_delay_s: float = 0,
            sum_iteration_callback = None,
            turn_off_all_leds_after: bool = False,
            use_executor: bool = False,
        ) -> str | None:

        """Grab the current live image from the camera and save to a TIFF file.

        Combines get_image() and save_image() in one call. Optionally turns off
        all LEDs after capture.

        Args:
            save_folder: Directory to save into.
            file_root: Filename prefix.
            append: String appended to filename.
            color: Color label for the filename.
            tail_id_mode: "increment" for auto-numbered files, or None.
            force_to_8bit: Convert 12-bit images to 8-bit.
            output_format: "TIFF" or "OME-TIFF".
            true_color: Actual channel color for metadata.
            earliest_image_ts: Reject frames before this timestamp.
            timeout: Max time to wait for a valid frame.
            all_ones_check: Reject saturated frames.
            sum_count: Number of frames to sum.
            sum_delay_s: Delay between summed frames.
            sum_iteration_callback: Called after each summed frame.
            turn_off_all_leds_after: Turn off all LEDs after capture.
            use_executor: Reserved for future use.

        Returns:
            str | None: Path to saved file, or None on failure.
        """

        # PIW-2: removed redundant `check_disk_space("/")` warn — see save_image() above.

        array = self.imaging.capture_and_wait(
            force_to_8bit=force_to_8bit,
            earliest_image_ts=earliest_image_ts,
            timeout=timeout,
            all_ones_check=all_ones_check,
            sum_count=sum_count,
            sum_delay_s=sum_delay_s,
            sum_iteration_callback=sum_iteration_callback,
        )

        if turn_off_all_leds_after:
            self.illumination.leds_off()

        if array is False:
            return

        return self.save_image(array, save_folder, file_root, append, color, tail_id_mode, output_format=output_format, true_color=true_color)


    def get_max_width(self) -> int:
        return self.imaging.get_max_width()

    def get_max_height(self) -> int:
        return self.imaging.get_max_height()

    def get_width(self) -> int:
        return self.imaging.get_width()

    def get_height(self) -> int:
        return self.imaging.get_height()

    def set_frame_size(self, w: int, h: int) -> None:
        return self.imaging.set_frame_size(w, h)

    def get_frame_size(self) -> dict | None:
        return self.imaging.get_frame_size()


    def get_gain(self) -> float:
        return self.imaging.get_gain()

    def set_gain(self, gain: float) -> None:
        return self.imaging.set_gain(gain)

    def set_auto_gain(self, state: bool, settings: dict) -> None:
        return self.imaging.set_auto_gain(state, settings)

    @contextlib.contextmanager
    def suppress_value_warnings(self):
        return self.imaging.suppress_value_warnings()

    def set_exposure_time(self, t: float) -> None:
        return self.imaging.set_exposure_time(t)

    def get_exposure_time(self) -> float:
        return self.imaging.get_exposure_time()

    def set_auto_exposure_time(self, state: bool=True) -> None:
        return self.imaging.set_auto_exposure_time(state)

    def apply_layer_camera_settings(self, gain: float, exposure_ms: float, auto_gain: bool=False, auto_gain_settings: dict | None=None) -> None:
        return self.imaging.apply_layer_camera_settings(gain, exposure_ms, auto_gain, auto_gain_settings)

    def update_auto_gain_target_brightness(self, target_brightness: float) -> None:
        return self.imaging.update_auto_gain_target_brightness(target_brightness)

    def auto_gain_once(self, state: bool, target_brightness: float, min_gain: float, max_gain: float) -> None:
        return self.imaging.auto_gain_once(state, target_brightness, min_gain, max_gain)

    def update_camera_config(self):
        return self.imaging.update_camera_config()

    def camera_is_connected(self) -> bool:
        return self.imaging.camera_is_connected()

        #return True

    def get_camera_temps(self) -> dict:
        return self.imaging.get_camera_temps()

    def log_camera_temps(self) -> None:
        return self.imaging.log_camera_temps()

    def start_camera_temp_logging(self, schedule_interval_fn, unschedule_fn, *, interval_s: float=14400.0) -> None:
        return self.imaging.start_camera_temp_logging(schedule_interval_fn, unschedule_fn, interval_s=interval_s)

    def stop_camera_temp_logging(self, unschedule_fn=None) -> None:
        return self.imaging.stop_camera_temp_logging(unschedule_fn)

    ########################################################################
    # MOTION CONTROL FUNCTIONS
    ########################################################################
    @contextlib.contextmanager
    def reference_position_logger(self):
        """Context manager that logs limit-switch status before and after homing.

        Use as ``with scope.reference_position_logger(): ... home ...``.
        Emits forced-INFO log lines so the limit-switch state pre/post
        homing is preserved for diagnostics.
        """
        before = self.motion.get_limit_switch_status_all_axes()
        logger.info(f"Limit switch status before homing: {before}", extra={'force_error': True})
        yield
        after = self.motion.get_limit_switch_status_all_axes()
        logger.info(f"Limit switch status after homing: {after}", extra={'force_error': True})

    def get_microscope_model(self) -> str | None:
        """Get the microscope model identifier from the motion board.

        Returns:
            str | None: Model string, or None if motion board inactive.
        """
        return self._motion_driver.get_microscope_model()

    def get_motor_info(self) -> dict:
        """Get motor controller information.

        Returns:
            dict: Keys 'model', 'serial_number', 'firmware_version'.
                  Values are None/unknown if board inactive.
        """
        info = self._motion_driver.fullinfo()
        return {
            'model': info.get('model', 'unknown'),
            'serial_number': info.get('serial_number', 'unknown'),
            'firmware_version': getattr(self._motion_driver, 'firmware_version', None),
        }

    def get_led_info(self) -> dict:
        """Get LED controller information.

        Returns:
            dict: Keys 'firmware_version', 'connected'.
        """
        if not self._led_driver or not self._led_driver.is_connected():
            return {'firmware_version': None, 'connected': False}

        return {
            'firmware_version': getattr(self._led_driver, 'firmware_version', None),
            'connected': True,
        }

    def get_camera_info(self) -> dict:
        """Get camera information.

        Returns:
            dict: Keys 'model', 'pixel_format', 'connected'.
        """
        if not self._camera_driver or not self._camera_driver.active:
            return {'model': None, 'pixel_format': None, 'connected': False}

        return {
            'model': self._camera_driver.get_model_name(),
            'pixel_format': self._camera_driver.get_pixel_format(),
            'connected': True,
        }

    def get_camera_temperatures(self) -> dict:
        """Get all camera temperature sensor readings.

        Returns:
            dict: Mapping of sensor name to temperature in °C.
            Empty dict if camera is inactive or has no temperature sensors.
        """
        if not self._camera_driver or not self._camera_driver.active:
            return {}
        try:
            return self._camera_driver.get_all_temperatures()
        except Exception as e:
            logger.debug(f'[SCOPE API ] get_camera_temperatures failed: {e}')
            return {}

    # ------------------------------------------------------------------
    # Diagnostic API (LAYER-D / LV-23, LV-24, LV-32, LV-40)
    # Tech-support / bring-up / bench tools route diagnostics through
    # these methods so the API layer owns Rule-13 logging and Rule-14
    # error visibility. Modules MUST NOT call `self._camera_driver.get_image()`,
    # `scope.led.exchange_command()`, etc. directly — see audit doc
    # `docs/AUDIT_LAYER_VIOLATIONS_2026-05-01.md` Cluster D.
    # ------------------------------------------------------------------

    def get_camera_diagnostic_info(self) -> dict:
        """Read-only snapshot of camera state for diagnostics.

        Returns the values that ``modules/tech_support_report.py`` and
        bench tools used to read directly off the driver. Each field is
        independently guarded so partial driver support yields a partial
        dict rather than an exception.

        Returns:
            dict: Camera diagnostic snapshot. Keys may include
                'model', 'resolution', 'pixel_format', 'gain', 'exposure_ms',
                'max_gain', 'max_exposure_ms', 'temperatures', plus per-key
                error strings for fields the driver couldn't supply.
                Returns ``{'connected': False}`` if the camera is inactive.
        """
        if not self._camera_driver or not self._camera_driver.active:
            return {'connected': False}

        info: dict = {'connected': True}

        def _try(key, fn):
            try:
                info[key] = fn()
            except Exception as e:
                info[key] = f'Error: {e}'

        _try('model', lambda: self._camera_driver.get_model_name())
        _try('pixel_format', lambda: self._camera_driver.get_pixel_format())

        try:
            fs = self._camera_driver.get_frame_size()
            info['resolution'] = f"{fs.get('width', '?')}x{fs.get('height', '?')}"
            info['frame_size'] = fs
        except Exception as e:
            info['resolution'] = f'Error: {e}'

        _try('gain', lambda: self.imaging.get_gain())
        _try('exposure_ms', lambda: self.imaging.get_exposure_time())
        _try('max_gain', lambda: self._camera_driver.get_max_gain())
        _try('max_exposure_ms', lambda: self._camera_driver.get_max_exposure())

        info['temperatures'] = self.get_camera_temperatures()
        return info

    def run_camera_bandwidth_test(
        self,
        num_frames: int,
        *,
        timeout_s: float = 60.0,
        progress_cb=None,
    ) -> dict:
        """Run an N-frame camera throughput test through the production capture path.

        Routes every frame grab through ``Lumascope.get_image()`` so the
        bandwidth numbers reflect what protocol/preview capture actually
        sees. Bypassing this method (calling ``self._camera_driver.get_image()``
        directly) is a Rule-1 layer violation and the resulting numbers
        are not comparable to production capture.

        Args:
            num_frames: Total frames to grab.
            timeout_s: Hard wall-clock cutoff in seconds; the test stops
                early and marks ``passed=False`` if exceeded.
            progress_cb: Optional ``callback(percent_int, message_str)``
                called every 250 frames.

        Returns:
            dict: Same shape as the legacy ``CameraBandwidthTest.run()`` --
                num_frames_requested, num_frames_received, num_frames_none,
                num_frames_error, total_bytes, elapsed_seconds,
                mb_per_second, fps_actual, frame_sizes, errors, passed.
        """
        results = {
            'num_frames_requested': int(num_frames),
            'num_frames_received': 0,
            'num_frames_none': 0,
            'num_frames_error': 0,
            'total_bytes': 0,
            'elapsed_seconds': 0,
            'mb_per_second': 0.0,
            'fps_actual': 0.0,
            'frame_sizes': [],
            'errors': [],
            'passed': True,
        }

        # Annotate with current camera state — same fields the legacy
        # tech-support test attached to its result dict.
        cam_info = self.get_camera_diagnostic_info()
        if cam_info.get('connected'):
            for key in ('resolution', 'pixel_format'):
                if key in cam_info:
                    results[key] = cam_info[key]

        if not self._camera_driver or not self._camera_driver.active:
            results['passed'] = False
            results['errors'].append('Camera not active')
            return results

        frame_size_set = set()
        start = time.monotonic()
        for i in range(int(num_frames)):
            if progress_cb and i % 250 == 0:
                try:
                    progress_cb(int(100 * i / max(num_frames, 1)),
                                f"Frame {i}/{num_frames}")
                except Exception:
                    pass
            try:
                # force_to_8bit=False keeps native depth so frame size
                # reflects the actual bytes the SDK delivered.
                frame = self.imaging.get_image(force_to_8bit=False, force_new_capture=True)
                if frame is None or frame is False:
                    results['num_frames_none'] += 1
                else:
                    results['num_frames_received'] += 1
                    nbytes = getattr(frame, 'nbytes', None) or len(frame)
                    results['total_bytes'] += nbytes
                    frame_size_set.add(int(nbytes))
            except Exception as e:
                results['num_frames_error'] += 1
                if len(results['errors']) < 20:
                    results['errors'].append(
                        f"Frame {i}: {type(e).__name__}: {e}")

            if time.monotonic() - start > timeout_s:
                results['errors'].append(
                    f"Timeout at frame {i} after {timeout_s}s")
                results['passed'] = False
                break

        elapsed = time.monotonic() - start
        results['elapsed_seconds'] = round(elapsed, 2)
        if elapsed > 0:
            results['mb_per_second'] = round(
                results['total_bytes'] / (1024 * 1024) / elapsed, 2)
            results['fps_actual'] = round(
                results['num_frames_received'] / elapsed, 1)
        results['frame_sizes'] = sorted(frame_size_set)

        if results['num_frames_none'] > 0:
            results['passed'] = False
            results['errors'].append(
                f"{results['num_frames_none']} frames returned None -- "
                f"possible USB disconnect or bandwidth issue")
        if results['num_frames_error'] > 0:
            results['passed'] = False
        if len(frame_size_set) > 1:
            results['passed'] = False
            results['errors'].append(
                f"Inconsistent frame sizes: {sorted(frame_size_set)} -- "
                f"possible data corruption or config change during test")

        logger.info(
            f"[SCOPE API ] run_camera_bandwidth_test: {results['num_frames_received']}/{num_frames} "
            f"frames in {results['elapsed_seconds']}s "
            f"({results['mb_per_second']} MB/s, {results['fps_actual']} fps), "
            f"passed={results['passed']}"
        )
        return results

    def run_grab_lifecycle_benchmark(
        self,
        num_cycles: int = 100,
        inter_cycle_delay_ms: float = 0.0,
        vary_settings: bool = False,
        *,
        slow_threshold_s: float = 3.0,
        progress_cb=None,
    ) -> dict:
        """Characterize stop_grabbing/start_grabbing latency under back-to-back cycling.

        CAM-1 step (0a) -- empirical floor for the SDK's "minimum safe
        interval between StopGrabbing and the next StartGrabbing" instead
        of relying on Basler-published numbers. Typical case is 130-150 ms;
        the pathological ~11 s case has been observed when StopGrabbing
        fires within ~275 ms of a prior StartGrabbing before the camera
        produces a frame. Sweeping ``inter_cycle_delay_ms`` through
        0/50/100/200/500/1000 ms across runs reveals the smallest delay
        that yields ZERO slow cycles.

        Stays inside the API: drops to ``self._camera_driver.stop_grabbing`` /
        ``start_grabbing`` directly, which is a Rule-1 downward call from
        the API into its driver -- same pattern as ``set_frame_size`` etc.

        Args:
            num_cycles: Stop/start cycles to perform.
            inter_cycle_delay_ms: Sleep between StopGrabbing and the next
                StartGrabbing (and any settings churn).
            vary_settings: When True, alternate gain (1.0 ↔ 4.0) and
                exposure (10 ms ↔ 50 ms) between cycles to reproduce the
                per-step protocol pattern that caused STALL-1.
            slow_threshold_s: Cycle wall-time considered "slow" -- counted
                separately so the operator sees how often the pathological
                case fires under the chosen delay.
            progress_cb: Optional ``callback(percent_int, message_str)``
                called every 10 cycles.

        Returns:
            dict with: num_cycles, inter_cycle_delay_ms, vary_settings,
                slow_threshold_s, slow_cycle_count, slow_cycles (list of
                {idx, cycle_s, stop_s, start_s}), cycle_p50/p95/p99,
                stop_p50/p95/p99, start_p50/p95/p99, total_elapsed_s,
                camera_model, pylon_version, errors, written_to.
        """
        results = {
            'num_cycles': int(num_cycles),
            'inter_cycle_delay_ms': float(inter_cycle_delay_ms),
            'vary_settings': bool(vary_settings),
            'slow_threshold_s': float(slow_threshold_s),
            'slow_cycle_count': 0,
            'slow_cycles': [],
            'cycle_p50_s': 0.0, 'cycle_p95_s': 0.0, 'cycle_p99_s': 0.0,
            'stop_p50_s': 0.0,  'stop_p95_s': 0.0,  'stop_p99_s': 0.0,
            'start_p50_s': 0.0, 'start_p95_s': 0.0, 'start_p99_s': 0.0,
            'total_elapsed_s': 0.0,
            'camera_model': None,
            'pylon_version': None,
            'errors': [],
            'written_to': None,
        }

        if not self._camera_driver or not self._camera_driver.active:
            results['errors'].append('Camera not active')
            return results

        cam_info = self.get_camera_diagnostic_info()
        results['camera_model'] = cam_info.get('model')
        results['pylon_version'] = cam_info.get('sdk_version') or cam_info.get('pylon_version')

        cycle_times, stop_times, start_times = [], [], []
        delay_s = max(0.0, float(inter_cycle_delay_ms) / 1000.0)

        # Snapshot current settings so we can restore even when vary_settings
        # is on — the benchmark must not leave the camera in an arbitrary state.
        original_gain = getattr(self._camera_driver, 'gain', None)
        original_exposure = getattr(self._camera_driver, 'exposure_time', None)

        t_overall_start = time.monotonic()
        for i in range(int(num_cycles)):
            if progress_cb and i % 10 == 0:
                try:
                    progress_cb(int(100 * i / max(num_cycles, 1)),
                                f"Cycle {i}/{num_cycles}")
                except Exception:
                    pass

            cycle_start = time.monotonic()
            try:
                t0 = time.monotonic()
                self._camera_driver.stop_grabbing()
                stop_s = time.monotonic() - t0

                if delay_s > 0:
                    time.sleep(delay_s)

                if vary_settings:
                    # Alternate between two presets — small enough churn
                    # not to dominate the cycle, large enough that GenICam
                    # node-map writes are real.
                    if i % 2 == 0:
                        self.imaging.set_gain(1.0)
                        self.imaging.set_exposure_time(10.0)
                    else:
                        self.imaging.set_gain(4.0)
                        self.imaging.set_exposure_time(50.0)

                t1 = time.monotonic()
                self._camera_driver.start_grabbing()
                start_s = time.monotonic() - t1
            except Exception as e:
                results['errors'].append(
                    f"Cycle {i}: {type(e).__name__}: {e}")
                # Try to leave the camera grabbing for the next iteration;
                # if it fails, the next stop_grabbing will surface it too.
                continue

            cycle_s = time.monotonic() - cycle_start
            cycle_times.append(cycle_s)
            stop_times.append(stop_s)
            start_times.append(start_s)

            if cycle_s >= slow_threshold_s:
                results['slow_cycle_count'] += 1
                # Cap the per-cycle log to keep the JSON small even on
                # pathological runs (every cycle slow).
                if len(results['slow_cycles']) < 50:
                    results['slow_cycles'].append({
                        'idx': i,
                        'cycle_s': round(cycle_s, 4),
                        'stop_s': round(stop_s, 4),
                        'start_s': round(start_s, 4),
                    })

        results['total_elapsed_s'] = round(time.monotonic() - t_overall_start, 3)

        # Restore caller's gain/exposure so vary_settings doesn't leak state.
        try:
            if vary_settings and original_gain is not None:
                self.imaging.set_gain(float(original_gain))
            if vary_settings and original_exposure is not None:
                self.imaging.set_exposure_time(float(original_exposure))
        except Exception as e:
            results['errors'].append(
                f"Restore settings failed: {type(e).__name__}: {e}")

        def _pct(samples, q):
            if not samples:
                return 0.0
            return round(float(np.percentile(samples, q)), 4)

        results['cycle_p50_s'] = _pct(cycle_times, 50)
        results['cycle_p95_s'] = _pct(cycle_times, 95)
        results['cycle_p99_s'] = _pct(cycle_times, 99)
        results['stop_p50_s']  = _pct(stop_times,  50)
        results['stop_p95_s']  = _pct(stop_times,  95)
        results['stop_p99_s']  = _pct(stop_times,  99)
        results['start_p50_s'] = _pct(start_times, 50)
        results['start_p95_s'] = _pct(start_times, 95)
        results['start_p99_s'] = _pct(start_times, 99)

        # Persist to data/camera_timing/ keyed by model + sdk version + delay
        # so a sweep across delays produces one file per data point.
        try:
            import json
            model = results['camera_model'] or 'unknown_camera'
            sdk = results['pylon_version'] or 'unknown_sdk'
            safe_model = str(model).replace(' ', '_').replace('/', '_')
            safe_sdk = str(sdk).replace(' ', '_').replace('/', '_')
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            timing_dir = pathlib.Path(os.path.dirname(__file__)).parent / 'data' / 'camera_timing'
            timing_dir.mkdir(parents=True, exist_ok=True)
            out_path = timing_dir / (
                f'grab_lifecycle_benchmark_{safe_model}_sdk{safe_sdk}_'
                f'delay{int(inter_cycle_delay_ms)}ms_{ts}.json'
            )
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            results['written_to'] = str(out_path)
        except Exception as e:
            results['errors'].append(
                f"Persist failed: {type(e).__name__}: {e}")

        logger.info(
            f"[SCOPE API ] run_grab_lifecycle_benchmark: {num_cycles} cycles, "
            f"delay={inter_cycle_delay_ms}ms, vary={vary_settings} -> "
            f"cycle p50={results['cycle_p50_s']}s p95={results['cycle_p95_s']}s "
            f"p99={results['cycle_p99_s']}s, slow={results['slow_cycle_count']} "
            f"(>={slow_threshold_s}s), total={results['total_elapsed_s']}s"
        )
        return results

    @staticmethod
    def _human_os_version() -> str:
        """Render OS version in a form humans can recognise.

        ``platform.release()`` on macOS returns the Darwin kernel
        version (e.g. ``24.6.0``) which nobody can map to "macOS 14.x"
        by inspection. ``platform.mac_ver()[0]`` returns the actual
        macOS version (e.g. ``14.5``); equivalent on Windows is
        ``platform.win32_ver()[0]``. Falls back to system + release
        on Linux / unknown.
        """
        import platform as _pl
        sys_name = _pl.system()
        try:
            if sys_name == 'Darwin':
                mac = _pl.mac_ver()[0]
                if mac:
                    return f'macOS {mac}'
            elif sys_name == 'Windows':
                win = _pl.win32_ver()[0]
                if win:
                    return f'Windows {win}'
        except Exception:
            pass
        return f'{sys_name} {_pl.release()}'

    @staticmethod
    def _safe_pylon_versions() -> dict:
        """Best-effort capture of pypylon + pylon SDK runtime versions.

        Both reads are wrapped: pypylon may not be installed (FX2-only
        installs), or the runtime version helper may have been renamed
        between SDK versions.
        """
        out = {'pypylon_version': None, 'pylon_sdk_version': None}
        try:
            import pypylon as _pyp
            out['pypylon_version'] = getattr(_pyp, '__version__', None)
        except Exception:
            pass
        try:
            from pypylon import pylon as _pylon
            for fn_name in ('GetPylonVersion', 'GetVersionString'):
                fn = getattr(_pylon, fn_name, None)
                if callable(fn):
                    try:
                        out['pylon_sdk_version'] = str(fn())
                        break
                    except Exception:
                        continue
        except Exception:
            pass
        return out

    def run_pylon_diagnostic_probe(
        self,
        duration_s: float = 3.0,
        *,
        drain_camera_side_errors: bool = True,
        progress_cb=None,
    ) -> dict:
        """One-shot Pylon-camera diagnostic probe with JSON output.

        Captures camera identity, current configuration, and stream-
        grabber statistics counter deltas over a sampling window of
        ``duration_s`` seconds. Adds host metadata (OS, hostname,
        pypylon + pylon SDK versions) and writes a single JSON file
        to ``data/pylon_probe/`` keyed on
        ``<model>__sn<serial>__fw<firmware>__<host>__dltl<config>__<datetime>.json``.

        Designed for cross-host / cross-camera / cross-firmware
        comparison: the filename pattern keeps a sweep's outputs
        sortable, and ``firmware_version`` + ``dltl_config`` are also
        promoted to top-level JSON keys for filter-by-load.

        Does NOT change grab state. If the camera is not currently
        grabbing, the deltas will be near-zero (stats counters do not
        advance without an active grab loop). Caller is expected to
        be in live preview when calling this method.

        Args:
            duration_s: Sampling window in seconds. Default 3.0
                matches the bench probe shape used to characterize
                dart vs ace 2 on Mac (Firmware DAILY_LOG.md).
            drain_camera_side_errors: When True, drain the camera's
                ``BslErrorPresent`` queue and capture the list of
                opaque error codes (per Basler "evaluated by support",
                no public translation table for ace 2 / dart R).
            progress_cb: Optional ``callback(percent_int, message_str)``
                called at probe start, mid-sample, and end.

        Returns:
            dict: Snapshot from driver plus host / timestamps /
                output_path metadata. Returns
                ``{'connected': False, 'errors': [...]}`` if no
                camera is active. Returns the driver's
                ``{'supported': False, 'reason': ...}`` shape for
                IDS or other non-Pylon drivers.
        """
        if progress_cb is not None:
            try:
                progress_cb(0, 'starting Pylon diagnostic probe')
            except Exception:
                pass

        if not self._camera_driver or not self._camera_driver.active:
            return {'connected': False, 'errors': ['Camera not active']}

        if not hasattr(self._camera_driver, 'read_diagnostic_snapshot'):
            return {
                'connected': False,
                'supported': False,
                'errors': [
                    f'{type(self._camera_driver).__name__} does not implement '
                    f'read_diagnostic_snapshot'
                ],
            }

        # Driver-level snapshot
        snapshot = self._camera_driver.read_diagnostic_snapshot(
            duration_s=duration_s,
            drain_camera_side_errors=drain_camera_side_errors,
        )

        # Non-Pylon stub returns supported=False; pass through unchanged
        if snapshot.get('supported') is False:
            if progress_cb is not None:
                try:
                    progress_cb(100, 'driver does not support diagnostic probe')
                except Exception:
                    pass
            return snapshot

        if progress_cb is not None:
            try:
                progress_cb(70, 'snapshot captured; collecting host metadata')
            except Exception:
                pass

        # Host metadata
        import socket
        import platform as _platform
        host_versions = self._safe_pylon_versions()
        snapshot['host'] = {
            'os': self._human_os_version(),
            'hostname': socket.gethostname(),
            'machine': _platform.machine(),
            'pypylon_version': host_versions['pypylon_version'],
            'pylon_sdk_version': host_versions['pylon_sdk_version'],
        }

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        end_iso = now_utc.isoformat()
        start_iso = (now_utc - datetime.timedelta(
            seconds=snapshot.get('duration_s_actual', duration_s)
        )).isoformat()
        snapshot['timestamps'] = {'start_iso': start_iso, 'end_iso': end_iso}

        # Filter-by-load top-level keys (per v4 author request: easier
        # to grep across many files than parsing camera.firmware_version
        # nested)
        snapshot['firmware_version'] = (
            snapshot.get('camera', {}).get('firmware_version')
        )

        dltl_token = self._dltl_filename_token(snapshot.get('config', {}))
        snapshot['dltl_config'] = dltl_token

        # JSON file write
        try:
            import json
            out_dir = (
                pathlib.Path(os.path.dirname(__file__)).parent
                / 'data' / 'pylon_probe'
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            def _safe_token(v: str | None, fallback: str) -> str:
                s = str(v) if v is not None else fallback
                # Filenames: replace separators that would break the
                # __ split-pattern, and any path separators.
                for bad in (' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|'):
                    s = s.replace(bad, '_')
                return s

            model_t = _safe_token(
                snapshot.get('camera', {}).get('model_name'), 'unknown_model')
            serial_t = _safe_token(
                snapshot.get('camera', {}).get('serial'), 'unknown_serial')
            fw_t = _safe_token(snapshot.get('firmware_version'), 'unknown_fw')
            host_t = _safe_token(
                snapshot['host']['hostname'], 'unknown_host'
            ).replace('.', '_')
            ts_t = now_utc.strftime('%Y%m%dT%H%M%SZ')

            fname = f'{model_t}__sn{serial_t}__fw{fw_t}__{host_t}__{dltl_token}__{ts_t}.json'
            out_path = out_dir / fname
            with open(out_path, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            snapshot['output_path'] = str(out_path)
        except Exception as e:
            snapshot.setdefault('errors', []).append(
                f'JSON write failed: {type(e).__name__}: {e}'
            )

        if progress_cb is not None:
            try:
                progress_cb(100, 'complete')
            except Exception:
                pass

        return snapshot

    @staticmethod
    def _dltl_filename_token(config: dict) -> str:
        """Encode the DLTL config as a short filename-safe token.

        Examples:
            DLTL Off                  -> 'dltloff'
            DLTL On at 160 MB/s       -> 'dltl160M'
            DLTL On at 197.43 MB/s    -> 'dltl197M' (rounded)
            anything else / missing   -> 'dltlunknown'

        ``int(round(...))`` handles non-round sweep values cleanly --
        v4 author flagged the case where a sweep set DLTL to an
        intermediate value with sub-MB/s precision.
        """
        mode = config.get('dltl_mode')
        if isinstance(mode, str) and mode.lower() == 'off':
            return 'dltloff'
        value = config.get('dltl_value_bps')
        if isinstance(value, (int, float)) and value > 0:
            return f'dltl{int(round(value / 1_000_000))}M'
        return 'dltlunknown'

    def _diagnostic_target_board(self, target: str):
        """Resolve a diagnostic-target string ('led' | 'motor') to a driver board.

        Internal helper for ``send_diagnostic_command*``. Raises
        ``ValueError`` for an unknown target so a typo in tech-support
        code fails loudly rather than silently picking the wrong board.
        """
        target = target.lower() if isinstance(target, str) else target
        if target == 'led':
            return self._led_driver
        if target in ('motor', 'motion'):
            return self._motion_driver
        raise ValueError(
            f"send_diagnostic_command: unknown target {target!r} "
            f"(expected 'led' or 'motor')")

    def send_diagnostic_command(
        self,
        target: str,
        command: str,
        *,
        response_numlines: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """Send a single firmware diagnostic command and return the response.

        Wraps the driver's ``exchange_command`` with API-layer logging
        (Rule 13). Diagnostic clients (tech-support report, bench tools)
        MUST go through this method instead of reaching the driver directly
        (LV-24 / LV-32 / LV-40).

        Args:
            target: 'led' or 'motor'.
            command: Firmware command string (e.g. ``'INFO'``, ``'FACTORY'``).
            response_numlines: Forwarded to driver; how many response lines
                to read before returning (driver-specific default if None).
            timeout: Per-call serial timeout in seconds, or None for the
                driver's default.

        Returns:
            str: Response from the board, ``'Board not connected'`` if the
                target board is None/inactive, or ``'Error: <msg>'`` if the
                exchange raised.
        """
        try:
            board = self._diagnostic_target_board(target)
        except ValueError as e:
            logger.warning(f'[SCOPE API ] send_diagnostic_command: {e}')
            return f'Error: {e}'

        if board is None or not getattr(board, 'found', False):
            return 'Board not connected'

        logger.debug(
            f'[SCOPE API ] send_diagnostic_command(target={target}, command={command!r}, '
            f'response_numlines={response_numlines}, timeout={timeout})'
        )
        try:
            kwargs = {}
            if response_numlines is not None:
                kwargs['response_numlines'] = response_numlines
            if timeout is not None:
                kwargs['timeout'] = timeout
            resp = board.exchange_command(command, **kwargs)
            return resp if resp is not None else 'None'
        except Exception as e:
            logger.warning(
                f'[SCOPE API ] send_diagnostic_command({target}, {command!r}) failed: {e}'
            )
            return f'Error: {e}'

    def send_diagnostic_command_multiline(
        self,
        target: str,
        command: str,
        *,
        timeout: float = 60,
        end_markers: list[str] | None = None,
    ) -> 'str | list[str]':
        """Send a firmware diagnostic command expected to return multiple lines.

        For SELFTEST, INFO with multi-line output, etc. Wraps the driver's
        ``exchange_multiline`` with API-layer logging.

        Args:
            target: 'led' or 'motor'.
            command: Firmware command string.
            timeout: Total timeout in seconds.
            end_markers: Substrings marking end-of-response. Default
                ``['PASS', 'FAIL', 'COMPLETE', 'DONE', 'ERROR']``.

        Returns:
            Response (driver-defined; typically str or list[str]),
            ``'Board not connected'``, or ``'Error: <msg>'``.
        """
        try:
            board = self._diagnostic_target_board(target)
        except ValueError as e:
            logger.warning(f'[SCOPE API ] send_diagnostic_command_multiline: {e}')
            return f'Error: {e}'

        if board is None or not getattr(board, 'found', False):
            return 'Board not connected'

        if end_markers is None:
            end_markers = ['PASS', 'FAIL', 'COMPLETE', 'DONE', 'ERROR']

        logger.debug(
            f'[SCOPE API ] send_diagnostic_command_multiline(target={target}, '
            f'command={command!r}, timeout={timeout}, end_markers={end_markers})'
        )
        try:
            result = board.exchange_multiline(
                command, timeout=timeout, end_markers=end_markers)
            return result if result else 'No response'
        except Exception as e:
            logger.warning(
                f'[SCOPE API ] send_diagnostic_command_multiline({target}, {command!r}) failed: {e}'
            )
            return f'Error: {e}'

    @classmethod
    def create_diagnostic(cls) -> 'Lumascope':
        """Create a minimal Lumascope for diagnostics (no camera init).

        Connects to LED and motor boards only. For use by tools like
        the tech support report that need board access without the full
        application stack.

        Returns:
            Lumascope: Instance with led/motion connected, camera=None.
        """
        instance = cls.__new__(cls)
        # Minimal init -- just enough for board communication
        instance._simulated = False
        instance._objectives_loader = objectives_loader.ObjectiveLoader()
        instance._coordinate_transformer = coord_transformations.CoordinateTransformer()

        # Camera cache
        instance._camera_cache_lock = threading.Lock()
        instance._camera_cache = {
            'active': False, 'gain': 0.0, 'exposure_ms': 20.0,
            'frame_size': {'width': 0, 'height': 0},
            'max_frame_size': {'width': 0, 'height': 0},
            'min_frame_size': {'width': 0, 'height': 0},
            'max_exposure': None,
            'max_gain': None,
        }

        # State locks
        instance._state_lock = threading.Lock()
        instance._objective = None
        instance._objective_id = None

        # Connect boards -- motion driver first so MotionAPI._driver resolves
        # correctly at construction time. The helpers are at module scope so
        # __init__, create_diagnostic, and future callers share one code path.
        from drivers.null_ledboard import NullLEDBoard
        from drivers.null_motorboard import NullMotionBoard
        instance._led_driver = _try_connect_board('LED board', LEDBoard, NullLEDBoard)
        instance._motion_driver = _try_connect_board('Motor board', MotorBoard, NullMotionBoard)

        # Construct MotionAPI and populate per-axis state (mirrors __init__ sequence).
        from modules.lumascope_api.motion import MotionAPI  # local-import: avoid cycle
        instance.motion = MotionAPI(instance, instance._motion_driver)
        present_axes = instance._motion_driver.detect_present_axes()
        instance.motion.init_axes(present_axes)
        instance.motion.start_monitor()

        instance.camera = None
        instance._frame_buffer = None

        # Build capabilities (audit B7) -- diagnostic instances still need
        # this so any code that reads scope.capabilities.* works.
        instance.capabilities = ScopeCapabilities.from_drivers(
            motion=instance._motion_driver,
            led=instance._led_driver,
            camera=None,
            led_max_ma=cls.LED_MAX_MA,
        )

        logger.info('[SCOPE API ] Diagnostic scope created '
                    f'(LED={instance.led_connected}, '
                    f'Motor={instance.motor_connected})')
        return instance

    def get_camera_profile_info(self) -> dict | None:
        """Get detailed camera profile information for display.

        Returns:
            dict with model, sensor, pixel_size_um, shutter, resolution,
            gain_range, max_exposure, binning_sizes. None if no camera.
        """
        if not self._camera_driver or not self._camera_driver.active:
            return None
        try:
            profile = self._camera_driver.profile
            exposure_min_us = getattr(profile, 'exposure_min_us', None)
            exposure_min_ms = (exposure_min_us / 1000.0
                                 if exposure_min_us is not None else None)
            return {
                'model': profile.model_name,
                'sensor': profile.sensor,
                'pixel_size_um': profile.pixel_size_um,
                'shutter': profile.shutter,
                'resolution': profile.native_resolution,
                'gain_min_db': profile.gain.total_min_db,
                'gain_max_db': profile.gain.total_max_db,
                'exposure_min_us': exposure_min_us,
                'exposure_min_ms': exposure_min_ms,
                'max_exposure_ms': self.imaging.camera_max_exposure,
                'binning_sizes': profile.binning_sizes,
            }
        except Exception as e:
            logger.debug(f'[SCOPE API ] get_camera_info failed: {e}')
            return None

    def get_system_info(self) -> dict:
        """Get consolidated system information for all hardware.

        Returns:
            dict: Keys 'motor', 'led', 'camera', 'simulated', 'lvp_version'.
        """
        return {
            'motor': self.get_motor_info(),
            'led': self.get_led_info(),
            'camera': self.get_camera_info(),
            'simulated': self._simulated,
            'lvp_version': version,
        }

    ########################################################################
    # INTEGRATED SCOPE FUNCTIONS
    ########################################################################

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ILLUMINATE AND CAPTURE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def capture(self) -> None:
        return self.imaging.capture()

    def capture_complete(self) -> None:
        return self.imaging.capture_complete()


    def capture_blocking(self) -> 'np.ndarray | bool | None':
        return self.imaging.capture_blocking()

    def _get_latest_chunks(self) -> dict | None:
        """Return per-frame chunk metadata for the most recent successful
        grab, or None if chunks aren't available.

        Camera handlers expose chunks differently:
          - PylonCamera.ImageHandler: composition -- chunks at handler._base
          - IDSCamera.ImageHandler: inheritance -- chunks at handler directly
          - FX2 / simulators: no chunks at all -> None

        Always returns None on any access path failure -- frame_validity
        falls back to skip-frames calibration when chunks aren't available.
        """
        if self._camera_driver is None:
            return None
        handler = getattr(self._camera_driver, 'cam_image_handler', None)
        if handler is None:
            return None
        # Composition (Pylon) first, then inheritance (IDS / direct base).
        base = getattr(handler, '_base', handler)
        if not hasattr(base, 'get_last_chunks'):
            return None
        try:
            return base.get_last_chunks()
        except Exception:
            return None

    def capture_and_wait(self, force_to_8bit: bool=True, *, exclude_sources: tuple=(), all_ones_check: bool=False, earliest_image_ts: datetime.datetime | None=None, timeout: datetime.timedelta=datetime.timedelta(seconds=0), sum_count: int=1, sum_delay_s: float=0, sum_iteration_callback=None) -> 'np.ndarray | bool':
        return self.imaging.capture_and_wait(force_to_8bit, exclude_sources=exclude_sources, all_ones_check=all_ones_check, earliest_image_ts=earliest_image_ts, timeout=timeout, sum_count=sum_count, sum_delay_s=sum_delay_s, sum_iteration_callback=sum_iteration_callback)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # AUTOFOCUS Functionality
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Legacy autofocus methods (autofocus, autofocus_iterate, focus_best) removed
    # 2026-03-31 — superseded by AutofocusRunner. No callers remained.

# Static methods for save_image functionality
    @staticmethod
    def get_next_save_path_static(path) -> str:
        """Get the next save path given an existing save path (static version).

        Static counterpart to ``get_next_save_path``; usable without a
        Lumascope instance.

        Args:
            path: Path of the format
                ``./{save_folder}/{well_label}_{color}_{file_id}.tiff``.

        Returns:
            str: Next save path with ``file_id`` incremented.
        """
        NUM_SEQ_DIGITS = 6
        # Handle both .tiff and .ome.tiff by detecting multiple extensions if present
        # pathlib doesn't seem to handle multiple extensions natively
        path2 = pathlib.Path(path)
        extension = ''.join(path2.suffixes)
        stem = path2.name[:len(path2.name)-len(extension)]
        seq_separator_idx = stem.rfind('_')
        stem_base = stem[:seq_separator_idx]
        seq_num_str = stem[seq_separator_idx+1:]
        seq_num = int(seq_num_str)

        next_seq_num = seq_num + 1
        next_seq_num_str = f"{next_seq_num:0>{NUM_SEQ_DIGITS}}"

        new_path = path2.parent / f"{stem_base}_{next_seq_num_str}{extension}"
        return str(new_path)

    @staticmethod
    def generate_image_save_path_static(save_folder, file_root, append,
                                        tail_id_mode, output_format) -> 'pathlib.Path':
        """Generate a unique save path for an image (static version).

        Static counterpart to ``generate_image_save_path``; usable without
        a Lumascope instance. Resolves collisions per ``tail_id_mode``.

        Args:
            save_folder: Directory to save into (str or Path).
            file_root: Filename prefix.
            append: String appended to filename.
            tail_id_mode: ``"increment"`` or ``None``.
            output_format: ``"TIFF"`` or ``"OME-TIFF"``.

        Returns:
            pathlib.Path: Full save path.

        Raises:
            ConfigError: If ``tail_id_mode`` is not implemented.
        """
        if isinstance(save_folder, str):
            save_folder = pathlib.Path(save_folder)

        if file_root is None:
            file_root = ""

        if output_format == 'OME-TIFF':
            file_extension = ".ome.tiff"
        else:
            file_extension = ".tiff"

        # generate filename and save path string
        if tail_id_mode == "increment":
            initial_id = '_000001'
            filename =  f"{file_root}{append}{initial_id}{file_extension}"
            path = save_folder / filename

            # Obtain next save path if current directory already exists
            while os.path.exists(path):
                path = Lumascope.get_next_save_path_static(path)

        elif tail_id_mode is None:
            filename =  f"{file_root}{append}{file_extension}"
            path = save_folder / filename

        else:
            raise ConfigError(f"tail_id_mode: {tail_id_mode} not implemented")

        return path

    @staticmethod
    def generate_image_metadata_static(
        color, x, y, z, objective, labware, stage_offset, coordinate_transformer,
        binning_size, exposure_time_ms, gain_db, illumination_ma
    ) -> dict:
        """Build TIFF metadata dict (static version).

        Static counterpart to ``generate_image_metadata``; usable without
        a Lumascope instance. Validates that objective, labware, and
        stage_offset are provided.

        Args:
            color: Channel color name.
            x: Stage X position in um (or None).
            y: Stage Y position in um (or None).
            z: Stage Z position in um (or None).
            objective: Objective dict containing ``focal_length``.
            labware: Labware configuration object.
            stage_offset: Stage offset configuration.
            coordinate_transformer: CoordinateTransformer instance.
            binning_size: Camera binning factor.
            exposure_time_ms: Exposure time in ms.
            gain_db: Gain in dB.
            illumination_ma: Illumination current in mA.

        Returns:
            dict: TIFF metadata dict with channel, positions, exposure,
                gain, pixel size, and well label.

        Raises:
            ConfigError: If objective, labware, or stage_offset are missing.
        """
        def _validate():
            if objective is None:
                raise ConfigError(f"[SCOPE API ] Objective not set")

            if 'focal_length' not in objective:
                raise ConfigError(f"[SCOPE API ] Objective focal length not provided")

            if labware is None:
                raise ConfigError(f"[SCOPE API ] Labware not set")

            if stage_offset is None:
                raise ConfigError(f"[SCOPE API ] Stage offset not set")

        _validate()

        if x is None:
            x = 0
        if y is None:
            y = 0
        if z is None:
            z = 0

        px, py = coordinate_transformer.stage_to_plate(
            labware=labware,
            stage_offset=stage_offset,
            sx=x,
            sy=y
        )

        px = round(px, common_utils.max_decimal_precision('x'))
        py = round(py, common_utils.max_decimal_precision('y'))
        z  = round(z,  common_utils.max_decimal_precision('z'))

        pixel_size_um = round(
            common_utils.get_pixel_size(
                focal_length=objective['focal_length'],
                binning_size=binning_size,
            ),
            common_utils.max_decimal_precision('pixel_size'),
        )

        metadata = {
            'camera_make': 'Etaluma',
            'software': f'LumaViewPro {version}',
            'channel': color,
            'datetime': datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),      # Format for metadata
            'sub_sec_time': f"{datetime.datetime.now().microsecond // 1000:03d}",
            'objective': objective,
            'focal_length': objective['focal_length'],
            'plate_pos_mm': {'x': px, 'y': py},
            'x_pos': px,
            'y_pos': py,
            'z_pos_um': z,
            'exposure_time_ms': round(exposure_time_ms, common_utils.max_decimal_precision('exposure')),
            'gain_db': round(gain_db, common_utils.max_decimal_precision('gain')),
            'illumination_ma': round(illumination_ma, common_utils.max_decimal_precision('illumination')),
            'binning_size': binning_size,
            'pixel_size_um': pixel_size_um,
        }

        return metadata

    @staticmethod
    def prepare_image_for_saving_static(
        array: np.ndarray,
        save_folder: str,
        file_root: str,
        append: str,
        color: str,
        tail_id_mode: str,
        output_format: str,
        true_color: str,
        x, y, z,
        objective, labware, stage_offset, coordinate_transformer,
        binning_size, exposure_time_ms, gain_db, illumination_ma
    ) -> dict:
        """Prepare an image array and metadata for saving (static version).

        Static counterpart to ``prepare_image_for_saving``; usable without
        a Lumascope instance. Generates the save path, applies false
        color, and flips the image vertically.

        Args:
            array: Raw image array from the driver.
            save_folder: Directory to save into.
            file_root: Filename prefix.
            append: String appended to filename.
            color: Color label for the filename.
            tail_id_mode: ``"increment"`` or ``None``.
            output_format: ``"TIFF"`` or ``"OME-TIFF"``.
            true_color: Actual channel color for metadata.
            x: Stage X position in um.
            y: Stage Y position in um.
            z: Stage Z position in um.
            objective: Objective dict.
            labware: Labware configuration.
            stage_offset: Stage offset configuration.
            coordinate_transformer: CoordinateTransformer instance.
            binning_size: Camera binning factor.
            exposure_time_ms: Exposure time in ms.
            gain_db: Gain in dB.
            illumination_ma: Illumination current in mA.

        Returns:
            dict: Contains ``'image'`` (ndarray) and ``'metadata'``
                (dict including ``'file_loc'``).
        """
        metadata = Lumascope.generate_image_metadata_static(
            color=true_color, x=x, y=y, z=z,
            objective=objective, labware=labware, stage_offset=stage_offset,
            coordinate_transformer=coordinate_transformer, binning_size=binning_size,
            exposure_time_ms=exposure_time_ms, gain_db=gain_db, illumination_ma=illumination_ma
        )

        if array.dtype == np.uint16:
            array = image_utils.convert_12bit_to_16bit(array)

        img = image_utils.add_false_color(array=array, color=color)
        img = np.flip(img, 0)

        path = Lumascope.generate_image_save_path_static(
            save_folder=save_folder,
            file_root=file_root,
            append=append,
            tail_id_mode=tail_id_mode,
            output_format=output_format
        )

        metadata['file_loc'] = path

        return {
            'image': img,
            'metadata': metadata,
        }

    @staticmethod
    def save_image_static(
        array,
        save_folder='./capture',
        file_root='img_',
        append='ms',
        color='BF',
        tail_id_mode="increment",
        output_format: str = "TIFF",
        true_color: str = 'BF',
        x=None, y=None, z=None,
        objective=None, labware=None, stage_offset=None, coordinate_transformer=None,
        binning_size=None, exposure_time_ms=None, gain_db=None, illumination_ma=None,
        use_false_color_16bit: bool | None = None,
    ) -> str:
        """Save an image array to a TIFF file with metadata (static version).

        Static counterpart to ``save_image``; doesn't require a Lumascope
        instance.

        Args:
            array: Image array to save.
            save_folder: Directory to save in.
            file_root: Filename prefix.
            append: String appended to filename.
            color: Color channel identifier for the filename.
            tail_id_mode: How to handle filename incrementing.
            output_format: ``"TIFF"`` or ``"OME-TIFF"``.
            true_color: True color for metadata.
            x: Stage X position in um.
            y: Stage Y position in um.
            z: Stage Z position in um.
            objective: Objective dict containing ``focal_length``.
            labware: Labware configuration.
            stage_offset: Stage offset configuration.
            coordinate_transformer: CoordinateTransformer instance.
            binning_size: Camera binning factor.
            exposure_time_ms: Exposure time in milliseconds.
            gain_db: Camera gain in dB.
            illumination_ma: LED illumination in mA.
            use_false_color_16bit: Optional override for false-color
                rendering at 16-bit depth.

        Returns:
            str: Path to the saved file.

        Raises:
            CaptureError: If the image cannot be written.
        """

        # Same None-gate as save_image; this static dupe retires alongside
        # the instance method when image-save helpers move out of the API.
        if array is None:
            raise CaptureError(
                "Camera did not return an image. The capture was skipped; "
                "the protocol will retry on the next step."
            )

        image_data = Lumascope.prepare_image_for_saving_static(
            array=array,
            save_folder=save_folder,
            file_root=file_root,
            append=append,
            color=color,
            tail_id_mode=tail_id_mode,
            output_format=output_format,
            true_color=true_color,
            x=x, y=y, z=z,
            objective=objective, labware=labware, stage_offset=stage_offset,
            coordinate_transformer=coordinate_transformer, binning_size=binning_size,
            exposure_time_ms=exposure_time_ms, gain_db=gain_db, illumination_ma=illumination_ma
        )

        image = image_data['image']
        metadata = image_data['metadata']
        file_loc = metadata['file_loc']

        if output_format == 'OME-TIFF':
            ome=True
        else:
            ome=False

        try:
            image_utils.write_tiff(
                data=image,
                file_loc=file_loc,
                metadata=metadata,
                ome=ome,
                color=color,
                use_false_color_16bit=use_false_color_16bit,
            )

            logger.info(f'[SCOPE API ] Saving Image to {file_loc}')
        except Exception:
            logger.error(f"[SCOPE API ] Error: Unable to save. Perhaps save folder does not exist? {file_loc}")
            raise CaptureError(f"Unable to save image to {file_loc}")

        return file_loc
