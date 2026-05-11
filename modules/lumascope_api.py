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
    # LED channel set comes from self.led.available_channels() — varies by
    # hardware (RP2040 = 6, FX2/Lumaview Classic = 4).
    # Per-axis state dicts (_pos_cache, _axis_state, _arrival_events,
    # _move_profile) are built from self.motion.detect_present_axes() at
    # init — they reflect actual hardware. This `_VALID_AXIS_NAMES` tuple
    # is the structural axis-name vocabulary used only for input sanity
    # checks ("did the caller pass a real axis letter?"), not for
    # capability queries. Use `axes_present()` for "what does this scope
    # have?".
    _VALID_AXIS_NAMES = ('X', 'Y', 'Z', 'T')
    # Absolute position bounds (um) — generous outer limits; per-axis travel
    # limits are enforced by the motor board itself.
    MOTOR_POSITION_LIMIT = 1_000_000  # 1 meter in um

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

        # Locks for the per-axis state dicts. The dicts themselves are
        # built below, AFTER the motion driver is constructed, so they
        # only contain the axes the hardware actually has.
        self._pos_cache_lock = threading.Lock()
        # Threading audit §10.2 — TimedLock on the hot axis-state lock records
        # contention to lock_trace.csv when LVP_PROFILE_TRACE=1. §4.5 of the
        # threading audit flagged the invariant "never hold _axis_state_lock
        # across a serial call" — trace reveals whether that invariant holds
        # under real workloads.
        self._axis_state_lock = profile_trace.TimedLock(threading.Lock(), name="lumascope._axis_state_lock")

        # Motion monitor wakeup — set when any axis starts MOVING, cleared when
        # all axes are back to IDLE. The monitor thread sleeps on this.
        self._motion_wake = threading.Event()

        # Position change listeners — push-based UI update mechanism.
        # Each listener is called with (axis: str, target: float, state: str)
        # whenever a position cache update or axis state transition occurs.
        # Listeners are called from the thread that caused the change (typically
        # the IO executor), so they MUST schedule UI work via Clock.schedule_once.
        self._position_listeners_lock = threading.Lock()
        self._position_listeners = []

        # LED change listeners — push-based UI update mechanism.
        # Each listener is called with (color: str, enabled: bool, mA: float,
        # owner: str) whenever any LED channel changes state.  Fires from the
        # thread that caused the change, so listeners MUST schedule UI work
        # via Clock.schedule_once.
        self._led_listeners_lock = threading.Lock()
        self._led_listeners = []

        # LED state — API-level source of truth (Rule 2: "the API owns
        # hardware state"). Completes the 2026-04-02 audit's design
        # intent: the API was always supposed to own LED state, but the
        # implementation only got as far as ownership + observers +
        # save/restore. State queries (get_led_ma, led_enabled, etc.)
        # still delegated to the driver — which worked for LEDBoard
        # (has an internal led_ma dict) but broke for FX2LEDController
        # (thin translator, returns sentinels). This dict is the
        # primary store, analogous to _pos_cache for motor position.
        # Updated inside led_on / led_off / leds_off; read by all
        # state-query methods. See docs/AUDIT_LED_STATE_FX2.md.
        self._led_state: dict[str, dict] = {}
        # Each entry: color -> {'enabled': True, 'illumination': float, 'owner': str}

        # LED ownership tracking — prevents subsystems from turning off LEDs
        # they did not turn on.  Each led_on with an owner records who claimed
        # the channel.  led_off with a non-matching owner is a no-op.
        # leds_off() without owner is the "nuclear" option (shutdown only).
        self._led_owner_lock = threading.Lock()
        self._led_owners = {}  # color -> owner tag

        # Camera change listeners — push-based UI update mechanism.
        # Each listener is called with (param: str, value: float) whenever
        # camera gain or exposure changes.  param is 'gain' or 'exposure'.
        # Fires from the thread that caused the change, so listeners MUST
        # schedule UI work via Clock.schedule_once.
        self._camera_listeners_lock = threading.Lock()
        self._camera_listeners = []

        # Lock for motion profile dict (built below, after motion driver init).
        self._move_profile_lock = threading.Lock()

        # ----- Motion Control Board -----
        # Constructed BEFORE the per-axis state dicts so we can size them
        # to the axes the hardware actually has (audit B4). Constructed
        # BEFORE the motion monitor thread so the thread always sees a
        # valid `self.motion`. Driver selection goes through the motor
        # registry (audit B2) — 'auto' tries real drivers in descending
        # priority order and falls back to NullMotionBoard if all fail,
        # so no manual try/except needed.
        motor_kwargs: dict = {}
        if simulate:
            from modules.settings_init import settings
            motor_kwargs['model'] = (settings.get('microscope', 'LS850')
                                     if settings else 'LS850')
        self.motion: MotorBoardProtocol = motor_registry.create(
            'auto', simulate=simulate, **motor_kwargs
        )
        if simulate:
            logger.info(
                f'[SCOPE API ] Using SIMULATED Motor Board '
                f'(model={motor_kwargs.get("model")})'
            )

        # ----- Per-axis state dicts (sized to actual hardware) -----
        # NullMotionBoard.detect_present_axes() returns [], so a system
        # with no motor hardware ends up with empty dicts. _set_axis_state
        # and the move_*_position methods handle that case as a Rule 8
        # silent no-op for absent axes.
        present_axes = self.motion.detect_present_axes()
        self._pos_cache = {ax: 0.0 for ax in present_axes}
        self._axis_state = {ax: AxisState.UNKNOWN for ax in present_axes}
        self._arrival_events = {ax: threading.Event() for ax in present_axes}
        for ev in self._arrival_events.values():
            ev.set()  # Start as "arrived" (not moving)
        self._move_profile = {ax: None for ax in present_axes}

        # ----- Motion monitor thread -----
        # Started AFTER motion + per-axis dicts are populated so the
        # thread never sees an inconsistent partial init state.
        self._motion_monitor_stop = threading.Event()
        self._motion_monitor_thread = threading.Thread(
            target=self._motion_monitor_loop,
            name='motion-monitor',
            daemon=True,
        )
        self._motion_monitor_thread.start()

        # ----- LED Control Board -----
        # Same registry-based selection as motion (audit B2).
        self.led: LEDBoardProtocol = led_registry.create('auto', simulate=simulate)
        if simulate:
            logger.info('[SCOPE API ] Using SIMULATED LED Board')

        # ----- Camera -----
        # Driver selection via camera_registry (audit B2). `camera_type`
        # accepts: 'auto' (tries pylon → ids by priority), 'pylon', 'ids',
        # 'sim', or any other registered camera kind. Default 'auto' is
        # the right choice for most callers; the pre-B2 default was
        # "pylon" which skipped auto-detect — callers that rely on that
        # continue to pass camera_type='pylon' explicitly.
        # PF-5: _image_buffer retired — get_image() chains via a local variable.
        self._frame_buffer = None
        self.camera = None
        camera_kwargs: dict = {}
        if simulate:
            camera_kwargs['z_position_func'] = lambda: self.motion.current_pos('Z')
        try:
            self.camera: Camera = camera_registry.create(
                camera_type, simulate=simulate, **camera_kwargs
            )
            if simulate:
                self.camera.load_cycle_images()
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
            motion=self.motion,
            led=self.led,
            camera=self.camera,
            led_max_ma=self.LED_MAX_MA,
        )

        # Partial-hardware notification deferred to initialize(config) —
        # we need scope-config knowledge to distinguish "LS620 correctly
        # has no motor" from "LS820 motor failed to connect."

        # Track whether any real hardware was found
        self._no_hardware = (
            not simulate
            and isinstance(self.led, NullLEDBoard)
            and isinstance(self.motion, NullMotionBoard)
            and not hasattr(self, 'camera')
        )
        if self._no_hardware:
            logger.warning('[SCOPE API ] No hardware detected (LED, motor, and camera all failed to initialize)')

        # --- Thread synchronization (CR-2 / CR-6) ---
        # _state_lock protects individual shared-state reads/writes
        self._state_lock = threading.Lock()
        # Per-device locks — each device communicates over a different port
        # and can operate independently. Split from the old global _hw_lock
        # to allow LED stim pulses during camera grabs and motor moves.
        # Threading audit §10.2 — both wrapped with TimedLock for contention tracing.
        self._led_lock = profile_trace.TimedLock(threading.RLock(), name="lumascope._led_lock")
        self._cam_lock = profile_trace.TimedLock(threading.RLock(), name="lumascope._cam_lock")
        # Global lock for multi-device atomic operations (e.g., LED on + capture + LED off).
        # Only used by acquire_exclusive() — individual methods use per-device locks.
        self._hw_lock = threading.RLock()

        # Boolean operation flags use threading.Event for wait/signal
        self._homing_event = threading.Event()       # set => homing in progress
        self._capturing_event = threading.Event()    # set => capture in progress
        self._focusing_event = threading.Event()     # set => autofocus in progress
        self._turreting_event = threading.Event()    # set => turret move in progress

        # Initialize scope status
        self._capture_return = False     # Will be image if capture is ready to pull, else False
        self._autofocus_return = False   # Will be z-position if focus is ready to pull, else False
        self.last_focus_score = None

        # self.is_stepping = False         # Is the microscope currently attempting to capture a step
        # self.step_capture_return = False # Will be image at step settings if ready to pull, else False

        self._labware = None              # The labware currently installed
        self._objective = None            # The objective currently selected/installed
        self._turret_config = {}          # The objectives loaded into the turret (if present)
        self._stage_offset = None         # The stage offset for the microscope
        self._last_turret_position = None # Stores the last known turret position
        self.engineering_mode = False      # Set by UI to enable engineering features
        # When True, programmatic value-range warnings (sub-0.1ms exposure,
        # future similar setters) are silenced. Internal callers that sweep
        # full ranges (camera characterization, dynamic-range tests) enter
        # this via `suppress_value_warnings()`; the warnings exist for L1
        # researchers who type microsecond values thinking ms.
        self._suppress_value_warnings = False

        # LAYER-A' executor handles. Registered post-construction via
        # register_executors() so that tests using `Lumascope(simulate=True)`
        # can construct without needing real executors. Methods that
        # submit IOTasks (led_on_async, move_absolute_async, etc.) will
        # raise RuntimeError if executors aren't registered.
        self._camera_executor = None
        self._io_executor = None
        self._file_io_executor = None
        self._autofocus_io_executor = None

        # LAYER-I source-path handle. Registered via register_source_path()
        # at startup. load_protocol() / create_protocol() use it to find
        # data/tiling.json without UI callers having to know the layout.
        self._source_path = None

        self.frame_validity = FrameValidity()
        # Register motion settle check — frame validity won't clear motion
        # sources until the axis has physically stopped moving.
        def _motion_settle_check(source: str) -> bool:
            # For absent axes (e.g., LS820 has no X/Y), treat UNKNOWN as settled.
            # Axes that were never homed or moved stay UNKNOWN — they shouldn't
            # block frame validity for sources that don't apply.
            idle_or_absent = (AxisState.IDLE, AxisState.UNKNOWN)
            if source == 'z_move':
                return self.get_axis_state('Z') in idle_or_absent
            elif source == 'xy_move':
                return (self.get_axis_state('X') in idle_or_absent and
                        self.get_axis_state('Y') in idle_or_absent)
            elif source == 'turret':
                return self.get_axis_state('T') in idle_or_absent
            return True
        self.frame_validity.set_settle_check(_motion_settle_check)
        self._load_camera_timing()
        if self.camera:
            self._binning_size = self.camera.get_binning_size()
        else:
            self._binning_size = 1

        self._scale_bar = {
            'enabled': False,
            'color': None,
        }

        # Camera state cache — push-based, not polled.
        # Updated when camera connects and after every set_gain/set_exposure/etc.
        # UI reads from cache with zero SDK calls.
        self._camera_cache_lock = threading.Lock()
        self._camera_cache = {
            'active': False,
            'gain': 0.0,
            'exposure_ms': 0.0,
            'frame_size': {'width': 0, 'height': 0},
            'max_frame_size': {'width': 0, 'height': 0},
            'min_frame_size': {'width': 0, 'height': 0},
            'max_exposure': 0.0,
            'max_gain': 0.0,
            'pixel_format': None,
            'binning': 1,
        }
        self._populate_camera_cache()

        # Populate position cache from firmware so get_current_position()
        # returns correct values immediately (not 0.0 from empty cache).
        # Critical for standalone scripts that read position right after
        # creating Lumascope (e.g., backlash characterization).
        if self.motor_connected:
            try:
                self.refresh_position_cache()
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
        self.leds_off()
        self.set_labware(config.labware)
        if config.turret_config:
            self.set_turret_config(config.turret_config)
        self.set_objective(config.objective_id)
        self.set_binning_size(config.binning_size)
        self.set_frame_size(config.frame_width, config.frame_height)
        self.set_stage_offset(config.stage_offset)
        self.set_scale_bar(enabled=config.scale_bar_enabled)
        self.set_acceleration_limit(val_pct=config.acceleration_pct)
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
        if config.expects_led and isinstance(self.led, NullLEDBoard):
            missing.append("LED Board")
        if config.expects_motion and isinstance(self.motion, NullMotionBoard):
            missing.append("Motor Controller")
        if not hasattr(self, 'camera') or not getattr(self.camera, 'active', None):
            missing.append("Camera")
        if missing:
            notifications.warning(
                "Hardware", "Partial Hardware Detected",
                f"Not connected: {', '.join(missing)}. Some features will be unavailable.",
            )


    # --- Motion monitor (Phase 1A) ---

    _MOTION_POLL_INTERVAL = 0.02  # 50 Hz

    def _motion_monitor_loop(self):
        """Background thread: polls firmware for axis arrival at 50 Hz.

        Sleeps on ``_motion_wake`` when all axes are IDLE. Wakes when any
        axis transitions to MOVING. Polls ``get_target_status()`` per
        MOVING axis and transitions them to IDLE on arrival. This is the
        single place where firmware target-status queries happen during
        normal operation -- all other code reads the in-memory axis state.
        """
        while not self._motion_monitor_stop.is_set():
            # Sleep until something starts moving (or shutdown)
            self._motion_wake.wait()
            if self._motion_monitor_stop.is_set():
                break

            # Poll moving axes until all arrive
            while not self._motion_monitor_stop.is_set():
                moving_axes = []
                with self._axis_state_lock:
                    moving_axes = [
                        ax for ax, st in self._axis_state.items()
                        if st == AxisState.MOVING
                    ]

                if not moving_axes:
                    # Also check overshoot — if overshoot is active,
                    # the monitor should keep running
                    if hasattr(self.motion, 'overshoot') and self.motion.overshoot:
                        time.sleep(self._MOTION_POLL_INTERVAL)
                        continue
                    # All axes arrived — go back to sleep
                    self._motion_wake.clear()
                    break

                # Query firmware for each MOVING axis
                with profile_trace.timer(
                    "motion_trace.csv",
                    "ts_ms,duration_ms,event,axis,detail",
                    lambda: ["poll", ",".join(moving_axes), ""],
                ):
                    for ax in moving_axes:
                        if self._motion_monitor_stop.is_set():
                            break
                        try:
                            if self.motion.is_connected() and self.get_target_status(ax):
                                # Axis has arrived — transition to IDLE
                                self._set_axis_state(ax, AxisState.IDLE)
                            else:
                                # Still moving — fire position listener so UI
                                # updates crosshair during motion (fixes #601)
                                self._fire_position_listeners(ax)
                        except Exception as e:
                            logger.warning(f'[SCOPE API ] Motion monitor: target_status({ax}) failed: {e}')

                time.sleep(self._MOTION_POLL_INTERVAL)

    def _stop_motion_monitor(self):
        """Stop the motion monitor thread (called during disconnect)."""
        self._motion_monitor_stop.set()
        self._motion_wake.set()  # unblock if sleeping
        if self._motion_monitor_thread.is_alive():
            self._motion_monitor_thread.join(timeout=1.0)

    def _load_camera_timing(self):
        """Load per-camera timing config if available.

        Looks for data/camera_timing/<model>.json and overrides
        FrameValidity.SKIP_FRAMES with measured values.
        """
        if not self.camera or not self.camera.active:
            return
        try:
            import json
            model = getattr(self.camera, 'model_name', None)
            if not model:
                return
            # Normalize model name for filename
            safe_name = model.replace(' ', '_')
            timing_dir = pathlib.Path(os.path.dirname(__file__)).parent / 'data' / 'camera_timing'
            timing_path = timing_dir / f'{safe_name}.json'
            if not timing_path.exists():
                return
            with open(timing_path) as f:
                config = json.load(f)
            self.frame_validity.load_camera_timing(config)
            logger.info(f'[SCOPE API ] Loaded camera timing config from {timing_path}')
        except Exception as e:
            logger.warning(f'[SCOPE API ] Failed to load camera timing config: {e}')

    # --- Camera state cache accessors (zero SDK calls) ---

    def _populate_camera_cache(self):
        """Populate camera cache from hardware. Called at init and on reconnect."""
        if not self.camera or not self.camera.active:
            with self._camera_cache_lock:
                self._camera_cache['active'] = False
            return

        try:
            cache = {
                'active': True,
                'gain': self.camera.get_gain() or 0.0,
                'exposure_ms': self.camera.get_exposure_t() or 0.0,
                'frame_size': self.camera.get_frame_size() or {'width': 0, 'height': 0},
                'max_frame_size': self.camera.get_max_frame_size() or {'width': 0, 'height': 0},
                'min_frame_size': self.camera.get_min_frame_size() or {'width': 0, 'height': 0},
                'max_exposure': self.camera.get_max_exposure() or None,
                'max_gain': self.camera.get_max_gain() if hasattr(self.camera, 'get_max_gain') else None,
                'pixel_format': self.camera.get_pixel_format() if hasattr(self.camera, 'get_pixel_format') else None,
                'binning': self.camera.get_binning_size() if hasattr(self.camera, 'get_binning_size') else 1,
            }
            with self._camera_cache_lock:
                self._camera_cache.update(cache)
            logger.info('[SCOPE API ] Camera cache populated')
        except Exception as e:
            logger.warning(f'[SCOPE API ] Failed to populate camera cache: {e}')
            with self._camera_cache_lock:
                self._camera_cache['active'] = bool(self.camera and self.camera.active)

    def _invalidate_camera_cache(self):
        """Mark camera cache as inactive (e.g. on disconnect)."""
        with self._camera_cache_lock:
            self._camera_cache['active'] = False

    @property
    def camera_active(self) -> bool:
        """Whether the camera is connected and active (reads cache).

        Returns:
            bool: True if the camera is currently active.
        """
        with self._camera_cache_lock:
            return self._camera_cache['active']

    @property
    def camera_gain(self) -> float:
        """Current camera gain in dB (reads cache).

        Returns:
            float: Cached gain value in dB.
        """
        with self._camera_cache_lock:
            return self._camera_cache['gain']

    @property
    def camera_exposure_ms(self) -> float:
        """Current camera exposure time in ms (reads cache).

        Returns:
            float: Cached exposure time in milliseconds.
        """
        with self._camera_cache_lock:
            return self._camera_cache['exposure_ms']

    @property
    def camera_frame_size(self) -> dict:
        """Current camera frame size as {'width': int, 'height': int} (reads cache).

        Returns:
            dict: Copy of the cached frame size dict.
        """
        with self._camera_cache_lock:
            return dict(self._camera_cache['frame_size'])

    @property
    def camera_max_frame_size(self) -> dict:
        """Maximum camera frame size (reads cache).

        Returns:
            dict: Copy of the cached max frame size dict.
        """
        with self._camera_cache_lock:
            return dict(self._camera_cache['max_frame_size'])

    @property
    def camera_min_frame_size(self) -> dict:
        """Minimum camera frame size (reads cache).

        Returns:
            dict: Copy of the cached min frame size dict.
        """
        with self._camera_cache_lock:
            return dict(self._camera_cache['min_frame_size'])

    @property
    def camera_max_exposure(self) -> float | None:
        """Maximum camera exposure time in ms, or None if no camera is connected.

        Returns None (not a sentinel 0.0) so callers can distinguish
        "camera missing" from a real driver value. See #616.

        Returns:
            float | None: Max exposure time in ms, or None if unavailable.
        """
        with self._camera_cache_lock:
            value = self._camera_cache.get('max_exposure')
        if not value or value <= 0:
            return None
        return float(value)

    @property
    def camera_max_gain(self) -> float | None:
        """Maximum camera gain in dB, or None if no camera is connected.

        Parallel to camera_max_exposure -- lets the UI size the gain
        slider to the connected camera's profile-declared cap instead
        of a universal hardcoded 48 dB that can drive the image past
        the sensor's usable range (observed on LS620 2026-04-16).

        Returns:
            float | None: Max gain in dB, or None if unavailable.
        """
        with self._camera_cache_lock:
            value = self._camera_cache.get('max_gain')
        if value is None or value <= 0:
            return None
        return float(value)

    @property
    def camera_pixel_format(self) -> str:
        """Current camera pixel format (e.g. 'Mono8', 'Mono12') (reads cache).

        Returns:
            str: Cached pixel format string.
        """
        with self._camera_cache_lock:
            return self._camera_cache.get('pixel_format', 'Mono8')

    # --- CR-2: Thread-safe properties for shared state ---

    @property
    def is_homing(self) -> bool:
        """True while the microscope is homing.

        Returns:
            bool: True if a homing operation is in progress.
        """
        return self._homing_event.is_set()

    @is_homing.setter
    def is_homing(self, value: bool) -> None:
        """Set the homing-in-progress flag."""
        if value:
            self._homing_event.set()
        else:
            self._homing_event.clear()

    @property
    def is_turreting(self) -> bool:
        """True while the turret is moving.

        Returns:
            bool: True if a turret motion is in progress.
        """
        return self._turreting_event.is_set()

    @is_turreting.setter
    def is_turreting(self, value: bool) -> None:
        """Set the turret-motion-in-progress flag."""
        if value:
            self._turreting_event.set()
        else:
            self._turreting_event.clear()

    @property
    def is_capturing(self) -> bool:
        """True while the microscope is capturing an image.

        Returns:
            bool: True if a capture is in progress.
        """
        return self._capturing_event.is_set()

    @is_capturing.setter
    def is_capturing(self, value: bool) -> None:
        """Set the capture-in-progress flag."""
        if value:
            self._capturing_event.set()
        else:
            self._capturing_event.clear()

    @property
    def is_focusing(self) -> bool:
        """True while the microscope is running autofocus.

        Returns:
            bool: True if an autofocus run is in progress.
        """
        return self._focusing_event.is_set()

    @is_focusing.setter
    def is_focusing(self, value: bool) -> None:
        """Set the autofocus-in-progress flag."""
        if value:
            self._focusing_event.set()
        else:
            self._focusing_event.clear()

    @property
    def capture_return(self):
        """Latest capture result (image array or False/None).

        Returns:
            Image array on success, or False/None when no capture has
            completed yet.
        """
        with self._state_lock:
            return self._capture_return

    @capture_return.setter
    def capture_return(self, value) -> None:
        """Store the latest capture result."""
        with self._state_lock:
            self._capture_return = value

    @property
    def autofocus_return(self):
        """Latest autofocus result.

        Returns:
            The most recent autofocus return value (driver-defined), or
            None if autofocus has not run.
        """
        with self._state_lock:
            return self._autofocus_return

    @autofocus_return.setter
    def autofocus_return(self, value) -> None:
        """Store the latest autofocus result."""
        with self._state_lock:
            self._autofocus_return = value

    @property
    def scale_bar_config(self) -> dict:
        """Return a snapshot of scale bar settings.

        Returns:
            dict: Copy of the scale bar config (e.g. enabled, color).
        """
        with self._state_lock:
            return dict(self._scale_bar)

    @property
    def scale_bar_enabled(self) -> bool:
        """Whether the scale bar overlay is enabled.

        Returns:
            bool: True if the scale bar is enabled.
        """
        with self._state_lock:
            return bool(self._scale_bar.get('enabled', False))

    # --- Frame validity accessors (per LAYER-F / Rule 1) ---
    # External callers must use these instead of reaching through
    # `self.frame_validity.X` directly. The frame_validity attribute
    # remains accessible for tests that need to introspect pending state.

    @property
    def frame_is_valid(self) -> bool:
        """True if all pending hardware state changes have settled.

        ``frame_validity`` is the SSOT (see modules/frame_validity.py).

        Returns:
            bool: True when no pending state changes are outstanding.
        """
        return self.frame_validity.is_valid

    def frames_until_valid(self, exclude_sources: tuple = ()) -> int:
        """Number of frames that must be grabbed before the next valid frame.

        Delegates to frame_validity.

        Args:
            exclude_sources: Sources to exclude from the validity check.

        Returns:
            int: Number of additional frames to drain before validity. 0 if
                already valid.
        """
        return self.frame_validity.frames_until_valid(
            exclude_sources=exclude_sources,
        )

    def count_frame(self) -> None:
        """Record that a frame was grabbed from the camera.

        Delegates to frame_validity (no driver call).
        """
        self.frame_validity.count_frame()

    # --- Executor-backed command API (LAYER-A' / Rule 2) ---
    #
    # Single canonical path for hardware operations that need executor
    # dispatch: caller invokes scope.X_async(...) or scope.X_sync(...);
    # Lumascope picks the right executor internally. Replaces the older
    # modules/scope_commands.py helper functions where the caller had
    # to pass an executor on every call (parallel-paths anti-pattern).

    def register_executors(self, *, camera_executor=None, io_executor=None,
                           file_io_executor=None, autofocus_io_executor=None) -> None:
        """Register the executor handles used by the X_async / X_sync command methods.

        Call once at startup after the executors are constructed. Tests
        that don't drive the executor-backed API can skip this -- those
        methods raise RuntimeError if invoked without executors registered.

        Args:
            camera_executor: Executor for camera-bound IOTasks.
            io_executor: Executor for general IO/motion IOTasks.
            file_io_executor: Executor for file-IO IOTasks.
            autofocus_io_executor: Executor for autofocus IOTasks.
        """
        self._camera_executor = camera_executor
        self._io_executor = io_executor
        self._file_io_executor = file_io_executor
        self._autofocus_io_executor = autofocus_io_executor

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

    def leds_off_async(self, *, callback=None) -> None:
        """Submit ``leds_off`` to the io_executor.

        No-op if LED disconnected.

        Args:
            callback: Optional completion callback.
        """
        if not self.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return
        ex = self._require_executor(self._io_executor, 'leds_off_async')
        ex.put(IOTask(action=self.leds_off, callback=callback))
        logger.info('[SCOPE API ] leds_off_async()')

    def led_on_async(self, channel, illumination, *, callback=None,
                     cb_kwargs=None, owner: str = '') -> None:
        """Submit ``led_on(channel, illumination)`` to the io_executor.

        Args:
            channel: Channel number or color name.
            illumination: Illumination current in mA.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
            owner: Optional ownership tag for the LED state.
        """
        if not self.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return
        kwargs = {'owner': owner} if owner else {}
        ex = self._require_executor(self._io_executor, 'led_on_async')
        ex.put(IOTask(
            action=self.led_on,
            args=(channel, illumination),
            kwargs=kwargs,
            callback=callback,
            cb_kwargs=cb_kwargs,
        ))

    def led_off_async(self, channel, *, callback=None, cb_kwargs=None,
                      owner: str = '') -> None:
        """Submit ``led_off(channel)`` to the io_executor.

        Args:
            channel: Channel number or color name.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
            owner: Optional ownership tag; only matching owner can turn
                off the channel.
        """
        if not self.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return
        kwargs = {'channel': channel}
        if owner:
            kwargs['owner'] = owner
        ex = self._require_executor(self._io_executor, 'led_off_async')
        ex.put(IOTask(
            action=self.led_off,
            kwargs=kwargs,
            callback=callback,
            cb_kwargs=cb_kwargs,
        ))

    def led_on_sync(self, channel, illumination, *, timeout=5,
                    owner: str = '') -> None:
        """Run ``led_on`` through the io_executor and block until done.

        Args:
            channel: Channel number or color name.
            illumination: Illumination current in mA.
            timeout: Max seconds to wait for completion.
            owner: Optional ownership tag for the LED state.
        """
        if not self.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return
        kwargs = {'owner': owner} if owner else {}
        ex = self._require_executor(self._io_executor, 'led_on_sync')
        task = IOTask(action=self.led_on, args=(channel, illumination),
                      kwargs=kwargs)
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout)

    def leds_off_sync(self, *, timeout=5) -> None:
        """Run ``leds_off`` through the io_executor and block until done.

        Args:
            timeout: Max seconds to wait for completion.
        """
        if not self.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return
        ex = self._require_executor(self._io_executor, 'leds_off_sync')
        task = IOTask(action=self.leds_off)
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout)

    # --- Camera command API ---

    def set_gain_sync(self, gain, *, timeout=5) -> None:
        """Run ``set_gain`` through the camera_executor and block until done.

        Args:
            gain: Gain value in dB.
            timeout: Max seconds to wait for completion.
        """
        ex = self._require_executor(self._camera_executor, 'set_gain_sync')
        task = IOTask(action=self.set_gain, args=(gain,))
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout)

    def set_exposure_sync(self, exposure, *, timeout=5) -> None:
        """Run ``set_exposure_time`` through the camera_executor and block.

        Args:
            exposure: Exposure time in milliseconds.
            timeout: Max seconds to wait for completion.
        """
        ex = self._require_executor(self._camera_executor, 'set_exposure_sync')
        task = IOTask(action=self.set_exposure_time, args=(exposure,))
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout)

    def capture_and_wait_sync(self, *, timeout: float = 30, **kwargs) -> 'np.ndarray | bool | None':
        """Run ``capture_and_wait`` through the camera_executor and block.

        Args:
            timeout: Max seconds to wait for completion.
            **kwargs: Forwarded to ``capture_and_wait``.

        Returns:
            The captured image array, or None on failure.
        """
        ex = self._require_executor(self._camera_executor, 'capture_and_wait_sync')
        task = IOTask(action=self.capture_and_wait, kwargs=kwargs)
        fut = ex.put(task, return_future=True)
        if fut:
            return fut.result(timeout=timeout)
        return None

    # --- Motion command API ---

    def move_absolute_async(self, axis, pos, *, wait_until_complete=False,
                            overshoot_enabled=True, callback=None,
                            cb_kwargs=None) -> None:
        """Submit ``move_absolute_position`` to the io_executor.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            pos: Target position in um.
            wait_until_complete: If True, block until move finishes.
            overshoot_enabled: Allow Z overshoot for backlash compensation.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
        """
        ex = self._require_executor(self._io_executor, 'move_absolute_async')
        ex.put(IOTask(
            action=self.move_absolute_position,
            kwargs={
                'axis': axis,
                'pos': pos,
                'wait_until_complete': wait_until_complete,
                'overshoot_enabled': overshoot_enabled,
            },
            callback=callback,
            cb_kwargs=cb_kwargs,
        ))

    def move_absolute_sync(self, axis, pos, *, wait_until_complete=True,
                           overshoot_enabled=True, timeout=30) -> None:
        """Run ``move_absolute_position`` through the io_executor and block.

        Blocks until both the IOTask completes and (when
        ``wait_until_complete``) the stage has physically arrived.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            pos: Target position in um.
            wait_until_complete: If True, block until move finishes.
            overshoot_enabled: Allow Z overshoot for backlash compensation.
            timeout: Max seconds to wait for completion.
        """
        ex = self._require_executor(self._io_executor, 'move_absolute_sync')
        task = IOTask(
            action=self.move_absolute_position,
            kwargs={
                'axis': axis,
                'pos': pos,
                'wait_until_complete': wait_until_complete,
                'overshoot_enabled': overshoot_enabled,
            },
        )
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout)

    def move_relative_async(self, axis, um, *, wait_until_complete=False,
                            overshoot_enabled=True, callback=None,
                            cb_kwargs=None) -> None:
        """Submit ``move_relative_position`` to the io_executor.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            um: Distance to move in um.
            wait_until_complete: If True, block until move finishes.
            overshoot_enabled: Allow Z overshoot for backlash compensation.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
        """
        ex = self._require_executor(self._io_executor, 'move_relative_async')
        ex.put(IOTask(
            action=self.move_relative_position,
            kwargs={
                'axis': axis,
                'um': um,
                'wait_until_complete': wait_until_complete,
                'overshoot_enabled': overshoot_enabled,
            },
            callback=callback,
            cb_kwargs=cb_kwargs,
        ))

    def move_home_async(self, axis, *, callback=None, cb_args=None) -> None:
        """Home an axis (or the whole scope) via the io_executor.

        Args:
            axis: 'Z' or 'T' homes that single axis. 'ALL' (or legacy 'XY')
                homes everything the board has via self.home() -- firmware
                homes Z and T first as part of the same routine.
            callback: Optional completion callback.
            cb_args: Optional positional args passed to the callback.
        """
        ex = self._require_executor(self._io_executor, 'move_home_async')
        a = axis.upper()
        # Homing legitimately takes 10-60+ seconds depending on travel
        # distance and starting position — well above the 5 sec default
        # slow-task threshold. Bump to 120s; only a true stall warrants
        # a warning here.
        HOME_THRESHOLD = 120.0
        if a == 'Z':
            ex.put(IOTask(action=self.zhome, callback=callback, cb_args=cb_args,
                          slow_task_threshold_sec=HOME_THRESHOLD))
        elif a in ('ALL', 'XY'):
            ex.put(IOTask(action=self.home, callback=callback, cb_args=cb_args,
                          slow_task_threshold_sec=HOME_THRESHOLD))
        elif a == 'T':
            ex.put(IOTask(action=self.thome, callback=callback, cb_args=cb_args,
                          slow_task_threshold_sec=HOME_THRESHOLD))
        else:
            logger.warning(f'[SCOPE API ] Unknown home axis: {axis}')

    # --- Axis state accessors (zero serial I/O) ---

    def get_axis_state(self, axis: str) -> str:
        """Get the current state of an axis.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            str: One of AxisState.UNKNOWN, IDLE, MOVING, HOMING.
        """
        with self._axis_state_lock:
            return self._axis_state.get(axis, AxisState.UNKNOWN)

    def add_position_listener(self, listener) -> None:
        """Register a callback for position/state changes on any axis.

        The listener is called with ``(axis, target_pos, state)`` whenever
        the position cache or axis state changes.  It fires from the thread
        that caused the change (IO executor, motion monitor, etc.), so
        listeners **must** schedule any UI work via ``Clock.schedule_once``.

        Args:
            listener: ``callable(axis: str, target: float, state: str)``
        """
        with self._position_listeners_lock:
            self._position_listeners.append(listener)

    def remove_position_listener(self, listener) -> None:
        """Unregister a position listener.

        Args:
            listener: A callable previously passed to
                ``add_position_listener``. Silently ignores listeners that
                are not currently registered.
        """
        with self._position_listeners_lock:
            try:
                self._position_listeners.remove(listener)
            except ValueError:
                pass

    def _fire_position_listeners(self, axis: str):
        """Notify all position listeners of a change on *axis*."""
        with self._pos_cache_lock:
            target = self._pos_cache.get(axis, 0.0)
        with self._axis_state_lock:
            state = self._axis_state.get(axis, AxisState.UNKNOWN)
        with self._position_listeners_lock:
            listeners = list(self._position_listeners)
        for fn in listeners:
            try:
                fn(axis, target, state)
            except Exception as ex:
                _api_log.debug(f'position listener error: {ex}')

    # ------------------------------------------------------------------
    # LED change listeners
    # ------------------------------------------------------------------

    def add_led_listener(self, listener) -> None:
        """Register a callback for LED state changes.

        The listener is called with ``(color, enabled, mA, owner)`` whenever
        any LED channel changes state.  It fires from the thread that caused
        the change, so listeners **must** schedule UI work via
        ``Clock.schedule_once``.

        Args:
            listener: ``callable(color: str, enabled: bool, mA: float, owner: str)``
        """
        with self._led_listeners_lock:
            self._led_listeners.append(listener)

    def remove_led_listener(self, listener) -> None:
        """Unregister an LED listener.

        Args:
            listener: A callable previously passed to ``add_led_listener``.
                Silently ignores listeners that are not currently registered.
        """
        with self._led_listeners_lock:
            try:
                self._led_listeners.remove(listener)
            except ValueError:
                pass

    def _fire_led_listeners(self, color: str, enabled: bool, mA: float,
                            owner: str = ''):
        """Notify all LED listeners of a state change on *color*."""
        with self._led_listeners_lock:
            listeners = list(self._led_listeners)
        for fn in listeners:
            try:
                fn(color, enabled, mA, owner)
            except Exception as ex:
                _api_log.debug(f'led listener error: {ex}')

    # ------------------------------------------------------------------
    # LED ownership
    # ------------------------------------------------------------------

    def save_led_state(self, tag: str) -> dict:
        """Snapshot the current LED state for later restoration.

        Args:
            tag: Descriptive name for the snapshot (for logging).

        Returns:
            dict: Snapshot suitable for passing to ``restore_led_state``.
        """
        states = self.get_led_states()
        with self._led_owner_lock:
            owners = dict(self._led_owners)
        snapshot = {'tag': tag, 'states': states, 'owners': owners}
        _api_log.info(f'save_led_state tag={tag}: '
                      f'{[c for c, s in states.items() if s.get("enabled")]}')
        return snapshot

    def restore_led_state(self, snapshot: dict, owner: str = '') -> None:
        """Restore LEDs to a previously saved state.

        Turns off channels owned by *owner* (or all if owner is empty),
        then re-enables channels that were on in the snapshot.

        Args:
            snapshot: Return value from ``save_led_state``.
            owner: If set, only turn off channels currently owned by
                this owner before restoring.
        """
        if not snapshot:
            return
        tag = snapshot.get('tag', '?')
        saved_states = snapshot.get('states', {})
        _api_log.info(f'restore_led_state tag={tag}')

        # Turn off what the owner turned on
        if owner:
            self.leds_off_owned(owner)
        else:
            self.leds_off()

        # Restore channels that were on in the snapshot
        for color, state in saved_states.items():
            if state.get('enabled', False):
                mA = state.get('illumination', 0)
                if mA and mA > 0:
                    ch = self.color2ch(color)
                    if ch is not None:
                        saved_owner = snapshot.get('owners', {}).get(color, '')
                        self.led_on(channel=ch, mA=mA, owner=saved_owner)

    def save_camera_state(self, tag: str) -> dict:
        """Snapshot the current camera gain and exposure for later restoration.

        Args:
            tag: Descriptive name for the snapshot (for logging).

        Returns:
            dict: Snapshot suitable for passing to ``restore_camera_state``.
        """
        gain = self.get_gain()
        exposure = self.get_exposure_time()
        snapshot = {'tag': tag, 'gain': gain, 'exposure': exposure}
        _api_log.info(f'save_camera_state tag={tag}: gain={gain} exp={exposure}')
        return snapshot

    def restore_camera_state(self, snapshot: dict) -> None:
        """Restore camera gain and exposure from a previously saved state.

        Args:
            snapshot: Return value from ``save_camera_state``.
        """
        if not snapshot:
            return
        tag = snapshot.get('tag', '?')
        _api_log.info(f'restore_camera_state tag={tag}')
        gain = snapshot.get('gain', -1)
        exposure = snapshot.get('exposure', 0)
        if gain >= 0:
            self.set_gain(gain)
        if exposure > 0:
            self.set_exposure_time(exposure)

    def leds_off_owned(self, owner: str) -> None:
        """Turn off only the LED channels owned by *owner*.

        Channels owned by other subsystems are left alone.

        Args:
            owner: The owner tag whose channels should be turned off.
        """
        if not self.led or not owner:
            return
        with self._led_owner_lock:
            channels_to_off = [color for color, own in self._led_owners.items()
                               if own == owner]
            for color in channels_to_off:
                self._led_owners.pop(color, None)
                self._led_state.pop(color, None)
        for color in channels_to_off:
            ch = self.color2ch(color)
            if ch is not None:
                with self._led_lock:
                    self.led.led_off(ch)
                self.frame_validity.invalidate('led')
                _api_log.info(f'led_off ch={ch} (owned release by {owner})')
                self._fire_led_listeners(color, False, 0.0, owner=owner)

    # ------------------------------------------------------------------
    # Camera change listeners
    # ------------------------------------------------------------------

    def add_camera_listener(self, listener) -> None:
        """Register a callback for camera setting changes.

        The listener is called with ``(param, value)`` whenever camera
        gain or exposure changes.  *param* is ``'gain'`` or ``'exposure'``.
        It fires from the thread that caused the change, so listeners
        **must** schedule UI work via ``Clock.schedule_once``.

        Note: this fires on set_gain/set_exposure_time (user actions),
        NOT on every camera frame grab -- zero overhead on display framerate.

        Args:
            listener: ``callable(param: str, value: float)``
        """
        with self._camera_listeners_lock:
            self._camera_listeners.append(listener)

    def remove_camera_listener(self, listener) -> None:
        """Unregister a camera listener.

        Args:
            listener: A callable previously passed to
                ``add_camera_listener``. Silently ignores listeners that
                are not currently registered.
        """
        with self._camera_listeners_lock:
            try:
                self._camera_listeners.remove(listener)
            except ValueError:
                pass

    def _fire_camera_listeners(self, param: str, value: float):
        """Notify all camera listeners of a setting change."""
        with self._camera_listeners_lock:
            listeners = list(self._camera_listeners)
        for fn in listeners:
            try:
                fn(param, value)
            except Exception as ex:
                _api_log.debug(f'camera listener error: {ex}')

    def _set_axis_state(self, axis: str, state: str):
        """Set the state of an axis (internal use only).

        When transitioning to MOVING/HOMING, clears the axis arrival event
        and wakes the motion monitor. When transitioning to IDLE, sets the
        arrival event so waiters unblock.  Fires position listeners on every
        transition.

        Silently no-ops for axes that are not present on this hardware
        (Rule 8). Per-axis dicts are sized to `motion.detect_present_axes()`
        at init, so hardcoded callers like `xycenter()` (X/Y) and `thome()`
        (T) automatically degrade to no-ops on scopes that lack those axes.
        """
        if axis not in self._arrival_events:
            return
        with self._axis_state_lock:
            old_state = self._axis_state.get(axis, AxisState.UNKNOWN)
            self._axis_state[axis] = state
        if profile_trace.ENABLE_PROFILE_TRACE and old_state != state:
            profile_trace.trace(
                "motion_trace.csv",
                "ts_ms,duration_ms,event,axis,detail",
                [int(time.time() * 1000), 0, "transition", axis, f"{old_state}->{state}"],
            )

        if state in (AxisState.MOVING, AxisState.HOMING):
            # Clear arrival event — axis is now in motion
            self._arrival_events[axis].clear()
            # Wake the motion monitor to start polling
            self._motion_wake.set()
        elif state == AxisState.IDLE:
            # Signal arrival — unblocks any wait_for_axis() callers
            self._arrival_events[axis].set()
            # Clear motion profile — predictor falls back to cache
            with self._move_profile_lock:
                self._move_profile[axis] = None

        self._fire_position_listeners(axis)

    def is_any_axis_moving(self) -> bool:
        """Check if any axis is currently MOVING or HOMING.

        Reads from the in-memory state dict -- zero serial I/O.

        Returns:
            bool: True if any axis is in MOVING or HOMING state.
        """
        with self._axis_state_lock:
            return any(
                s in (AxisState.MOVING, AxisState.HOMING)
                for s in self._axis_state.values()
            )

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
            return float(self.motion.motorconfig.travel_limit_um(axis))
        except Exception:
            return float(self.MOTOR_POSITION_LIMIT)

    @property
    def motor_connected(self) -> bool:
        """Whether the motor controller is connected.

        Returns:
            bool: True if a real (non-Null) motor board is connected.
        """
        return not isinstance(self.motion, NullMotionBoard) and self.motion.is_connected()

    @property
    def led_connected(self) -> bool:
        """Whether the LED controller is connected.

        Returns:
            bool: True if a real (non-Null) LED board is connected.
        """
        return not isinstance(self.led, NullLEDBoard) and self.led.is_connected()

    def lens_focal_length(self) -> float:
        """Get tube lens focal length from motorconfig.

        Returns:
            float: Focal length in mm (default 47.8).
        """
        return self.motion.motorconfig.lens_focal_length()

    def pixel_size(self) -> float:
        """Get camera pixel size from motorconfig.

        Returns:
            float: Pixel size in um/pixel (default 2.0).
        """
        return self.motion.motorconfig.pixel_size()

    # --- CR-6: Exclusive lock for multi-step hardware operations ---

    @contextlib.contextmanager
    def acquire_exclusive(self):
        """Context manager for multi-step hardware operations.

        Prevents interleaving of compound operations (e.g., set gain + capture).
        Uses RLock so a thread that already holds the lock can re-enter.

        Usage::

            with scope.acquire_exclusive():
                scope.set_led_ma('Blue', 10)
                image = scope.capture_and_wait()
        """
        self._hw_lock.acquire()
        try:
            yield
        finally:
            self._hw_lock.release()

    def stop_motion(self) -> None:
        """Stop all in-flight motor moves (LVP-A-1).

        Idempotent + safe-when-disconnected per Rule 4 + Rule 8 -- no-ops
        when the motor board isn't connected. Uses the firmware-side
        ``STOP`` command which the motor controller implements as
        ``motorstop`` (target=actual on all axes); same wire command the
        UI emergency-stop already uses, just routed through the API
        instead of an inline ``motion.exchange_command('STOP')``.

        Called as the first step of ``disconnect()`` so every disconnect
        path (App on_stop, REST shutdown, test teardown, future CLI
        tools) stops motors before tearing down the serial port.
        """
        if not self.motor_connected:
            return
        try:
            # LVP-A-1 followup: route through MotorBoard.motor_stop so
            # field firmware (2024-09-10 EL-0940-02) silently no-ops
            # instead of producing two FIRMWARE ERROR warnings per
            # shutdown. motor_stop returns True if STOP was accepted,
            # False if firmware doesn't implement it (cached).
            stopped = self.motion.motor_stop()
            if stopped:
                logger.info('[SCOPE API ] stop_motion: motors stopped')
            else:
                logger.debug(
                    '[SCOPE API ] stop_motion: firmware does not '
                    'implement STOP; motors will latch on disconnect')
        except Exception as e:
            # Rule 14 — log + notify, but don't re-raise: stop_motion
            # is called from shutdown paths where the caller can't
            # meaningfully recover and a raised exception would leave
            # disconnect() half-done.
            logger.warning(
                f'[SCOPE API ] stop_motion failed: {type(e).__name__}: {e}')
            try:
                from modules.notification_center import notifications
                notifications.warning(
                    'Motion', 'Motor stop failed',
                    f'STOP command failed during shutdown: '
                    f'{type(e).__name__}: {e}')
            except Exception:
                pass

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
        self.stop_motion()

        # Stop the motion monitor before disconnecting the motor board
        self._stop_motion_monitor()

        # Set all axes to UNKNOWN before disconnecting
        with self._axis_state_lock:
            for ax in self._axis_state:
                self._axis_state[ax] = AxisState.UNKNOWN
        # Set all arrival events so any blocked waiters unblock
        for ev in self._arrival_events.values():
            ev.set()

        # Each sub-system: only attempt disconnect on a driver that
        # has one. Skips both the canonical no-op states (NullLEDBoard,
        # NullMotionBoard, self.camera is None) and edge-case test
        # fixtures that bend the type system (e.g. `scope.led = object()`
        # for partial-hardware-warning tests). A skipped sub-system
        # counts as ok=True -- "nothing to tear down" is success, not
        # failure. Real drivers that raise inside disconnect() still
        # flip *_ok to False and fire a Rule-14 notification.
        led_ok = True
        if (not isinstance(self.led, NullLEDBoard)
                and hasattr(self.led, 'disconnect')):
            try:
                self.led.disconnect()
            except Exception as ex:
                led_ok = False
                logger.exception(f"[SCOPE API ] LED disconnect failed: {ex}")
                notifications.error(
                    "Hardware",
                    "LED disconnect failed",
                    f"LED board teardown raised {type(ex).__name__}: {ex}. "
                    f"The serial port may be left open; reconnecting "
                    f"may require a process restart.")
        self.led = NullLEDBoard()

        motion_ok = True
        if (not isinstance(self.motion, NullMotionBoard)
                and hasattr(self.motion, 'disconnect')):
            try:
                self.motion.disconnect()
            except Exception as ex:
                motion_ok = False
                logger.exception(f"[SCOPE API ] Motion disconnect failed: {ex}")
                notifications.error(
                    "Hardware",
                    "Motor disconnect failed",
                    f"Motor board teardown raised {type(ex).__name__}: {ex}. "
                    f"The serial port may be left open; reconnecting "
                    f"may require a process restart.")
        self.motion = NullMotionBoard()

        camera_ok = True
        if self.camera is not None and hasattr(self.camera, 'disconnect'):
            try:
                camera_ok = bool(self.camera.disconnect())
            except Exception as ex:
                camera_ok = False
                logger.exception(f"[SCOPE API ] Camera disconnect failed: {ex}")
                notifications.error(
                    "Hardware",
                    "Camera disconnect failed",
                    f"Camera teardown raised {type(ex).__name__}: {ex}. "
                    f"USB resources may not be fully released until the "
                    f"app restarts.")
            self.camera = None
        elif self.camera is not None:
            # Camera lacked a `disconnect` method (test-fixture artifact);
            # clear the slot but don't claim success on a real teardown.
            self.camera = None
        self._invalidate_camera_cache()

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
            self.leds_off()
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
        led = not isinstance(self.led, NullLEDBoard) and self.led.is_connected()
        motion = self.motor_connected
        camera = self.camera is not None and self.camera.is_connected()

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

    def get_turret_position_for_objective_id(
        self,
        objective_id: str,
        prefer_current: bool = True,
        persisted_position: int | None = None,
    ) -> int | None:
        """Find the turret position holding a given objective.

        Lookup ranking when multiple positions hold the same objective (#488):
            1. Persisted position from settings, if it matches objective_id
               and is provided by the caller. Honors the user's most
               recent explicit choice -- survives restarts and post-home
               situations where the current physical position is an
               artifact of the home routine (T zeros to 1), not user
               intent.
            2. Current physical T position, if it matches objective_id.
               Catches the case where the user has already rotated to a
               matching slot in this session and no persisted hint exists.
            3. First-match dict iteration (lowest position with the
               objective). Used when neither hint is available -- preserves
               today's fallback behavior.

        Args:
            objective_id: Objective identifier to search for.
            prefer_current: If True (default), check the current physical
                turret position when persisted_position is unavailable
                or doesn't match.
            persisted_position: Caller-supplied hint, typically
                ``settings.get('turret_position')``. None disables this
                tier of the lookup.

        Returns:
            int | None: Turret position (1-4), or None if not found.
        """
        if persisted_position is not None:
            if self._turret_config.get(persisted_position) == objective_id:
                return persisted_position

        if prefer_current:
            try:
                current_pos = self.get_current_position(axis='T')
                if self._turret_config.get(current_pos) == objective_id:
                    return current_pos
            except Exception:
                pass

        for turret_position, turret_objective_id in self._turret_config.items():
            if objective_id == turret_objective_id:
                return turret_position

        return None

    def is_current_turret_position_objective_set(self) -> bool:
        """Check whether the objective slot at the current turret position is set.

        Returns:
            bool: True if the current turret position has a configured
                objective ID; False if the slot is unconfigured.
        """
        position = self.get_current_position(axis='T')
        if self._turret_config[position] is None:
            return False

        return True

    def set_scale_bar(self, enabled: bool, color: str = None) -> None:
        """Configure the scale bar overlay on captured images.

        Args:
            enabled: Whether to draw the scale bar.
            color: Scale bar color (e.g. "white"). Uses default if None.
        """
        self._scale_bar['enabled'] = enabled
        if color is not None:
            self._scale_bar['color'] = color

    def set_stage_offset(self, stage_offset) -> None:
        """Set the stage offset for coordinate transformations.

        Args:
            stage_offset: Stage offset dict with axis offsets.
        """
        self._stage_offset = stage_offset

    def get_available_binning_sizes(self) -> list:
        """Return list of binning sizes supported by connected camera.

        Returns:
            list: Supported binning factors (e.g. ``[1, 2, 4]``). Defaults
                to ``[1]`` if no camera is active.
        """
        if not self.camera or not self.camera.active:
            return [1]
        try:
            return self.camera.profile.binning_sizes
        except (AttributeError, TypeError):
            return [1]

    def set_binning_size(self, size: int) -> bool:
        """Set camera pixel binning size.

        Args:
            size: Binning factor (1 = no binning, 2 = 2x2, etc.).

        Returns:
            bool: True if the driver applied the binning. False if the
                camera is absent, the driver returned False (size out of
                range, camera inactive), or the driver raised an
                exception. Caller can use the result to decide whether to
                proceed with operations that depend on the new binning.
        """
        try:
            self._binning_size = size

            if self.camera:
                ok = self.camera.set_binning_size(size=size)
            else:
                ok = False
            _api_log.info(f'set_binning {size}x{size} -> {ok}')
            return ok
        except Exception as ex:
            logger.exception(f"[SCOPE API ] Error setting binning size: {ex}")
            from modules.notification_center import notifications
            notifications.error("Camera", "Binning change failed",
                f"Could not set binning to {size}x{size}: {type(ex).__name__}: {ex}. "
                f"Camera may still be at previous binning -- verify actual frame size.")
            return False

    def get_binning_size(self) -> int:
        """Get the current camera binning size.

        Returns:
            int: Current binning factor (1 if camera inactive).
        """
        if not self.camera or not self.camera.active:
            return 1

        return self.camera.get_binning_size()

    def get_pixel_format(self) -> str | None:
        """Get the current camera pixel format.

        Returns:
            str | None: Pixel format string (e.g. 'Mono8'), or None if inactive.
        """
        if not self.camera or not self.camera.active:
            return None
        return self.camera.get_pixel_format()

    def set_pixel_format(self, pixel_format: str) -> bool:
        """Set the camera pixel format.

        Args:
            pixel_format: Format string (e.g. 'Mono8', 'Mono12').

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the driver returned False (unsupported format),
                or the driver raised. Never raises -- caller may safely
                check `if not scope.set_pixel_format(...)` for fallback.
        """
        if not self.camera or not self.camera.active:
            return False
        try:
            result = self.camera.set_pixel_format(pixel_format)
        except Exception as ex:
            logger.exception(f"[SCOPE API ] Error setting pixel format: {ex}")
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "Pixel format change failed",
                f"Could not set pixel format to {pixel_format}: "
                f"{type(ex).__name__}: {ex}. Camera may still be at the "
                f"previous format.")
            return False
        if result:
            with self._camera_cache_lock:
                self._camera_cache['pixel_format'] = pixel_format
        return result

    def get_supported_pixel_formats(self) -> tuple:
        """Get the list of supported camera pixel formats.

        Returns:
            tuple: Supported format strings, or empty tuple if inactive.
        """
        if not self.camera or not self.camera.active:
            return ()
        return self.camera.get_supported_pixel_formats()

    def set_device_link_throughput_limit(
        self,
        mode: str,
        value_bps: int | None = None,
    ) -> bool:
        """Set the camera's DeviceLinkThroughputLimit mode and value.

        Both nodes are live-writable per the SDK lock-state table -- no
        StopGrabbing/StartGrabbing wrap. Per-camera defaults bench-
        witnessed (USB3): ace 2 a2A3536-31umBAS at 360 MB/s -> 28.8 fps;
        dart daA3840-45um at 160 MB/s -> 18.7 fps. Setting ``mode='Off'``
        lets the camera run at sensor-readout maximum (~31.2 fps ace 2;
        ~44.9 fps dart on USB3).

        Used by the diagnostic-probe sweep in ``tools/`` to characterize
        failure rate vs throughput across camera + firmware + host
        cells. Per Basler docs: "Corrupt or dropped frames may occur if
        the DeviceLinkThroughputLimit parameter is too high" -- bench-
        test failure rate alongside fps before settling on a per-camera
        production default.

        **Transport caveat (GigE):** on GigE cameras (e.g. dmA3536-9gm)
        DLTL is bounded above by the GigE wire limit (~110 MB/s usable
        on 1 Gbps Ethernet). Setting above wire limit is a no-op; below
        caps fps proportionally. For GigE bandwidth control use
        ``set_gev_inter_packet_delay`` / ``set_bandwidth_reserve_mode``
        instead -- those are the GigE-side tools.

        Args:
            mode: ``'On'`` or ``'Off'`` (case-sensitive; matches Pylon
                enum entry symbolic names).
            value_bps: Throughput cap in bytes per second when
                ``mode='On'``. Ignored when ``mode='Off'``. If None
                while ``mode='On'``, only the mode is changed and the
                existing limit value is preserved.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the mode argument is invalid, or the driver
                returned False (unsupported by this driver).

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self.camera or not self.camera.active:
            return False
        if not hasattr(self.camera, 'set_device_link_throughput_limit'):
            logger.warning(
                f'[SCOPE API ] set_device_link_throughput_limit: '
                f'{type(self.camera).__name__} does not implement this method'
            )
            return False
        try:
            return bool(self.camera.set_device_link_throughput_limit(
                mode=mode, value_bps=value_bps,
            ))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting DLTL: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "DeviceLinkThroughputLimit change failed",
                f"Could not set DLTL to mode={mode}, value_bps={value_bps}: "
                f"{type(ex).__name__}: {ex}. Camera may still be at the "
                f"previous DLTL setting."
            )
            raise

    def set_acquisition_stop_mode(self, mode: str) -> bool:
        """Set BslAcquisitionStopMode (Pylon-only; no-op on IDS).

        Controls camera behavior when StopGrabbing fires during an
        in-flight exposure:

          - ``'Complete'`` (Pylon default): waits for the current
            exposure to finish before stopping.
          - ``'CancelExposure'``: stops cleanly; partial frame
            discarded.
          - ``'AbortExposure'``: aborts immediately; partial frame
            discarded.

        Default ``'Complete'`` waits up to the full exposure on long
        fluorescence captures (5-10 s) -- presents identically to a
        multi-second app-side stall when the user toggles modes.
        ``'AbortExposure'`` resolves the symptom but is bench-
        unvalidated on Etaluma's cameras. Setter is provided for
        bench characterization; default is unchanged.

        Args:
            mode: One of ``'Complete'``, ``'CancelExposure'``,
                ``'AbortExposure'``.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, mode is invalid, the driver doesn't
                implement the setter (IDS), or
                BslAcquisitionStopMode is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self.camera or not self.camera.active:
            return False
        if not hasattr(self.camera, 'set_acquisition_stop_mode'):
            logger.warning(
                f'[SCOPE API ] set_acquisition_stop_mode: '
                f'{type(self.camera).__name__} does not implement this method'
            )
            return False
        try:
            return bool(self.camera.set_acquisition_stop_mode(mode=mode))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting acquisition_stop_mode: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "BslAcquisitionStopMode change failed",
                f"Could not set acquisition_stop_mode to {mode!r}: "
                f"{type(ex).__name__}: {ex}. Camera may still be at "
                f"the previous stop-mode setting."
            )
            raise

    def set_bandwidth_reserve_mode(self, mode: str) -> bool:
        """Set BandwidthReserveMode (GigE-only Pylon node).

        ``'Default'`` reserves a portion of GigE bandwidth for
        retransmits; ``'Performance'`` dedicates all bandwidth to
        image transmit. Per dmA3536-9gm spec, ``'Performance'``
        unlocks 9.5 fps vs the default 9.3 fps.

        USB3 cameras do not expose the node; returns False so the
        bench-probe sweep can call this method unconditionally per
        cell.

        Args:
            mode: ``'Default'`` or ``'Performance'``.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the driver doesn't implement the setter,
                or the node is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self.camera or not self.camera.active:
            return False
        if not hasattr(self.camera, 'set_bandwidth_reserve_mode'):
            return False
        try:
            return bool(self.camera.set_bandwidth_reserve_mode(mode=mode))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting BandwidthReserveMode: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "BandwidthReserveMode change failed",
                f"Could not set BandwidthReserveMode to {mode!r}: "
                f"{type(ex).__name__}: {ex}."
            )
            raise

    def set_gev_packet_size(self, size_bytes: int) -> bool:
        """Set GevSCPSPacketSize (GigE-only Pylon node).

        Packet size in bytes. 1500 = standard Ethernet MTU; 9000 =
        typical jumbo-frame size. Larger packets reduce per-camera
        CPU + packet rate but require OS-level jumbo-frame config.

        USB3 cameras do not expose the node; returns False so the
        bench-probe sweep can call this method unconditionally per
        cell.

        Args:
            size_bytes: Packet size in bytes (positive int).

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, size_bytes is non-positive, the driver
                doesn't implement the setter, or the node is not
                exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self.camera or not self.camera.active:
            return False
        if not hasattr(self.camera, 'set_gev_packet_size'):
            return False
        try:
            return bool(self.camera.set_gev_packet_size(size_bytes=size_bytes))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting GevSCPSPacketSize: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "GevSCPSPacketSize change failed",
                f"Could not set GevSCPSPacketSize to {size_bytes}: "
                f"{type(ex).__name__}: {ex}."
            )
            raise

    def set_gev_inter_packet_delay(self, delay_ticks: int) -> bool:
        """Set GevSCPD (GigE inter-packet delay, in clock ticks).

        Inserts a wait between successive packets to throttle the
        camera. Used when multiple cameras share a single GigE link
        or when the host CPU can't keep up. 0 = no delay.

        USB3 cameras do not expose the node; returns False so the
        bench-probe sweep can call this method unconditionally per
        cell.

        Args:
            delay_ticks: Non-negative int; camera-specific tick rate.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, delay_ticks is negative, the driver doesn't
                implement the setter, or the node is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self.camera or not self.camera.active:
            return False
        if not hasattr(self.camera, 'set_gev_inter_packet_delay'):
            return False
        try:
            return bool(self.camera.set_gev_inter_packet_delay(
                delay_ticks=delay_ticks
            ))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting GevSCPD: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "GevSCPD change failed",
                f"Could not set GevSCPD to {delay_ticks}: "
                f"{type(ex).__name__}: {ex}."
            )
            raise

    def set_max_transfer_size(self, value_bytes: int) -> bool:
        """Set Pylon StreamGrabber MaxTransferSize (USB3 only).

        Bytes-per-USB-transfer the SDK requests from the kernel. Per
        Basler `stream-grabber-parameters.html` this is the named lever
        for the symptom "fails to receive image stream" -- decreasing
        the value works around kernel / driver USB-transfer-size
        constraints on some Windows hosts.

        USB3-only. The node is absent on GigE cameras and on the IDS
        SDK; returns False so the bench-probe sweep can call this
        method unconditionally per cell.

        Args:
            value_bytes: New MaxTransferSize in bytes.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the driver doesn't implement the setter, or
                the node is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self.camera or not self.camera.active:
            return False
        if not hasattr(self.camera, 'set_max_transfer_size'):
            return False
        try:
            return bool(self.camera.set_max_transfer_size(
                value_bytes=value_bytes
            ))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting MaxTransferSize: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "MaxTransferSize change failed",
                f"Could not set MaxTransferSize to {value_bytes}: "
                f"{type(ex).__name__}: {ex}."
            )
            raise

    def set_num_max_queued_urbs(self, value: int) -> bool:
        """Set Pylon StreamGrabber NumMaxQueuedUrbs (USB3 only).

        Number of USB Request Blocks the SDK keeps in flight to the
        kernel. Per Basler `stream-grabber-parameters.html` this is
        the named lever for "insufficient system memory"
        (0xe2010130 / 0xe2100001) -- decreasing the value reduces
        kernel URB allocation pressure on memory-constrained hosts.

        USB3-only. The node is absent on GigE cameras and on the IDS
        SDK; returns False so the bench-probe sweep can call this
        method unconditionally per cell.

        Args:
            value: New NumMaxQueuedUrbs (count).

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the driver doesn't implement the setter, or
                the node is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self.camera or not self.camera.active:
            return False
        if not hasattr(self.camera, 'set_num_max_queued_urbs'):
            return False
        try:
            return bool(self.camera.set_num_max_queued_urbs(value=value))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting NumMaxQueuedUrbs: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "NumMaxQueuedUrbs change failed",
                f"Could not set NumMaxQueuedUrbs to {value}: "
                f"{type(ex).__name__}: {ex}."
            )
            raise


    ########################################################################
    # LED BOARD FUNCTIONS
    ########################################################################

    def leds_enable(self) -> None:
        """Enable all LED channels (allows them to be turned on)."""
        if not self.led: return
        self.led.leds_enable()

    def leds_disable(self) -> None:
        """Disable all LED channels (prevents them from turning on)."""
        if not self.led: return
        self.led.leds_disable()

    def get_led_ma(self, color: str) -> float:
        """Get the current illumination level for an LED channel.

        Reads from the API-level _led_state cache (Rule 2). Does NOT
        delegate to the driver -- see AUDIT_LED_STATE_FX2.md Bug 4.

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            float: Illumination in milliamps, or -1 if channel is off or
                LED board unavailable.
        """
        if not self.led: return -1
        with self._led_owner_lock:
            entry = self._led_state.get(color)
            return entry['illumination'] if entry else -1.0

    def led_enabled(self, color: str) -> bool:
        """Whether a specific LED channel is currently on.

        Reads from the API-level _led_state cache (Rule 2). Pre-fix,
        this delegated to the driver's get_led_state, which for
        FX2LEDController always returned False -- making led_off a
        complete no-op (AUDIT_LED_STATE_FX2.md Bug 2).

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            bool: True if the channel is currently on.
        """
        if not self.led:
            return False
        with self._led_owner_lock:
            return self._led_state.get(color) is not None

    def led_illumination(self, color: str) -> float:
        """Current mA for an LED channel, or -1 if off.

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            float: Illumination in milliamps, or -1 if off / unavailable.
        """
        return self.get_led_ma(color)

    @property
    def led_states(self) -> dict:
        """Snapshot of all LED states {color: {enabled, illumination}}.

        Returns:
            dict: Mapping of color -> {'enabled': bool, 'illumination': float}.
                Empty if no LED board is connected.
        """
        if not self.led:
            return {}
        with self._led_owner_lock:
            return {
                color: {'enabled': True, 'illumination': entry['illumination']}
                for color, entry in self._led_state.items()
            }

    def get_led_state(self, color: str) -> dict:
        """Get the on/off state and illumination for an LED channel.

        Reads from the API-level _led_state cache (Rule 2).

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            dict: {'enabled': bool, 'illumination': float}.
        """
        if not self.led:
            return {'enabled': False, 'illumination': -1}
        with self._led_owner_lock:
            entry = self._led_state.get(color)
            if entry is None:
                return {'enabled': False, 'illumination': -1}
            return {'enabled': True, 'illumination': entry['illumination']}

    def get_led_states(self) -> dict:
        """Get state and illumination for all LED channels.

        Returns states for ALL channels the driver supports (not just
        currently-on channels).

        Returns:
            dict: Mapping of color -> {'enabled': bool, 'illumination': float}
                for every channel the driver supports. Empty if no LED
                board is connected.
        """
        if not self.led:
            return {}
        all_colors = self.led.available_colors()
        with self._led_owner_lock:
            return {
                color: (
                    {'enabled': True, 'illumination': self._led_state[color]['illumination']}
                    if color in self._led_state
                    else {'enabled': False, 'illumination': -1}
                )
                for color in all_colors
            }


    def led_on(self, channel, mA, block: bool = False, owner: str = '') -> None:
        """Turn on an LED channel at the specified current.

        Args:
            channel: Channel number (0-5) or color name string.
            mA: Illumination current in milliamps.
            block: If True, wait for confirmation from the LED board.
            owner: Optional ownership tag (e.g. 'autofocus', 'protocol').
                If set, only ``led_off`` / ``leds_off_owned`` with the same
                owner can turn this channel off.  Empty string (default) means
                no ownership tracking.

        Raises:
            ValueError: If channel or mA is out of range.
        """
        if not self.led: return

        if isinstance(channel, str):
            channel = self.color2ch(color=channel)

        valid_channels = self.led.available_channels()
        if channel not in valid_channels:
            raise ValueError(f"LED channel must be one of {valid_channels}, got {channel}")
        if not isinstance(mA, (int, float)) or mA < 0 or mA > self.LED_MAX_MA:
            raise ValueError(f"LED current must be 0-{self.LED_MAX_MA} mA, got {mA}")

        # Skip redundant command if channel is already on at the same current
        color_name = self.ch2color(channel)
        if color_name:
            current_ma = self.get_led_ma(color_name)
            # Rule 12 workaround: _led_state cache-equality trace for the
            # slider > ~150 mA silent-fail bench investigation. Gated by
            # LVP_FX2_DEBUG_WIRE env var to match drivers/fx2driver.py.
            # Remove together with fx2driver._FX2_DEBUG_WIRE block after
            # the 2026-04-21 bench session.
            if os.environ.get("LVP_FX2_DEBUG_WIRE") == "1":
                cached_entry = self._led_state.get(color_name)
                is_enabled = self.led_enabled(color_name)
                try:
                    delta = (None if current_ma is None
                             else abs(float(mA) - float(current_ma)))
                except Exception:
                    delta = 'ERR'
                _api_log.info(
                    '[FX2 LED diag] led_on cache-check color=%s '
                    'new_mA=%r (type=%s) cached_mA=%r (type=%s) '
                    'delta=%r enabled=%s cache_entry=%r',
                    color_name, mA, type(mA).__name__,
                    current_ma, type(current_ma).__name__,
                    delta, is_enabled, cached_entry,
                )
            if current_ma is not None and abs(float(mA) - float(current_ma)) < 0.01:
                if self.led_enabled(color_name):
                    return

        with self._led_lock:
            self.led.led_on(channel, mA, block=block)
        self.frame_validity.invalidate('led')
        _api_log.info(f'led_on ch={channel} mA={mA} owner={owner!r}')

        # Update API-level state cache + ownership (Rule 2). Unconditional
        # — empty owner ('') is recorded too, fixing AUDIT_LED_STATE_FX2.md
        # Bug 3 where UI clicks were never tracked because of an `if owner:`
        # gate that excluded empty strings.
        color_name = self.ch2color(channel)
        if color_name:
            with self._led_owner_lock:
                self._led_state[color_name] = {
                    'enabled': True,
                    'illumination': float(mA),
                    'owner': owner,
                }
                self._led_owners[color_name] = owner
            self._fire_led_listeners(color_name, True, float(mA), owner)

    def led_off(self, channel, owner: str = '') -> None:
        """Turn off an LED channel.

        Args:
            channel: Channel number (0-5) or color name string.
            owner: If set, only turn off if this owner currently owns
                the channel.  A non-matching owner is a no-op (logged).
                Empty string (default) turns off unconditionally.

        Raises:
            ValueError: If channel is out of range.
        """
        if not self.led: return

        if isinstance(channel, str):
            channel = self.color2ch(color=channel)

        valid_channels = self.led.available_channels()
        if channel not in valid_channels:
            raise ValueError(f"LED channel must be one of {valid_channels}, got {channel}")

        # Skip if channel is already off. Now reads from the API-level
        # _led_state cache, which is correct for both LEDBoard and FX2.
        # Pre-fix this delegated to the driver's get_led_state, which for
        # FX2 always returned False — making led_off a complete no-op
        # (AUDIT_LED_STATE_FX2.md Bug 2).
        color_name = self.ch2color(channel)
        if color_name and not self.led_enabled(color_name):
            return

        # Check ownership — if caller specifies an owner, only allow if it matches
        if owner and color_name:
            with self._led_owner_lock:
                entry = self._led_state.get(color_name, {})
                current_owner = entry.get('owner', '')
                if current_owner and current_owner != owner:
                    _api_log.debug(f'led_off blocked: ch={channel} owner={owner!r} '
                                   f'but owned by {current_owner!r}')
                    return

        with self._led_lock:
            self.led.led_off(channel)
        self.frame_validity.invalidate('led')
        _api_log.info(f'led_off ch={channel} owner={owner!r}')

        # Clear from API-level state cache + ownership
        if color_name:
            with self._led_owner_lock:
                self._led_state.pop(color_name, None)
                self._led_owners.pop(color_name, None)
            self._fire_led_listeners(color_name, False, 0.0, owner)

    def led_on_fast(self, channel, mA) -> None:
        """Turn on an LED with write-only (no read-back) for time-critical pulses.

        Args:
            channel: Channel number (0-5) or color name string.
            mA: Illumination current in milliamps.

        Raises:
            ValueError: If channel or mA is out of range.
        """
        if not self.led: return
        if isinstance(channel, str):
            channel = self.color2ch(color=channel)
        valid_channels = self.led.available_channels()
        if channel not in valid_channels:
            raise ValueError(f"LED channel must be one of {valid_channels}, got {channel}")
        if not isinstance(mA, (int, float)) or mA < 0 or mA > self.LED_MAX_MA:
            raise ValueError(f"LED current must be 0-{self.LED_MAX_MA} mA, got {mA}")
        with self._led_lock:
            self.led.led_on_fast(channel, mA)
        self.frame_validity.invalidate('led')
        color_name = self.ch2color(channel)
        if color_name:
            self._fire_led_listeners(color_name, True, float(mA), '')

    def led_off_fast(self, channel) -> None:
        """Turn off an LED with write-only (no read-back) for time-critical pulses.

        Args:
            channel: Channel number (0-5) or color name string.

        Raises:
            ValueError: If channel is out of range.
        """
        if not self.led: return
        if isinstance(channel, str):
            channel = self.color2ch(color=channel)
        valid_channels = self.led.available_channels()
        if channel not in valid_channels:
            raise ValueError(f"LED channel must be one of {valid_channels}, got {channel}")
        with self._led_lock:
            self.led.led_off_fast(channel)
        self.frame_validity.invalidate('led')
        color_name = self.ch2color(channel)
        if color_name:
            self._fire_led_listeners(color_name, False, 0.0, '')

    def leds_off_fast(self) -> None:
        """Turn off all LEDs with write-only (no read-back) for time-critical pulses."""
        if not self.led: return
        with self._led_lock:
            self.led.leds_off_fast()
        self.frame_validity.invalidate('led')
        with self._led_owner_lock:
            self._led_state.clear()
        for color in self.led.available_colors():
            self._fire_led_listeners(color, False, 0.0, '')

    def leds_off(self) -> None:
        """Turn off all LEDs (nuclear -- ignores ownership, clears all owners)."""
        if not self.led: return
        with self._led_lock:
            self.led.leds_off()
        with self._led_owner_lock:
            self._led_owners.clear()
            self._led_state.clear()
        self.frame_validity.invalidate('led')
        _api_log.info('leds_off')
        for color in self.led.available_colors():
            self._fire_led_listeners(color, False, 0.0, '')

    def get_led_status(self):
        """Get the LED board status register.

        Returns:
            Driver-defined status object (typically int bitfield), or
            None if no LED board is connected.
        """
        if not self.led: return
        return self.led.get_status()

    def wait_until_led_on(self) -> None:
        """Block until the LED board confirms an LED is on."""
        if not self.led: return
        self.led.wait_until_on()

    def ch2color(self, channel: int) -> str | None:
        """Convert a channel number to its color name string.

        Args:
            channel (int): Channel number (0=Blue, 1=Green, 2=Red, 3=BF, 4=PC, 5=DF).

        Returns:
            str: Color name (e.g. "Blue", "BF"), or None if LED board unavailable.
        """
        if not self.led: return
        return self.led.ch2color(channel)

    def color2ch(self, color: str) -> int | None:
        """Convert a color name string to its channel number.

        Args:
            color (str): Color name ("Blue", "Green", "Red", "BF", "PC", "DF").

        Returns:
            int: Channel number (0-5), or None if LED board unavailable.
        """
        if not self.led: return
        return self.led.color2ch(color)

    ########################################################################
    # CAMERA FUNCTIONS
    ########################################################################

    def get_image(
        self,
        force_to_8bit: bool = True,
        earliest_image_ts: datetime.datetime | None = None,
        timeout: datetime.timedelta = datetime.timedelta(seconds=5),
        all_ones_check: bool = False,
        sum_count: int = 1,
        sum_delay_s: float = 0,
        sum_iteration_callback = None,
        force_new_capture: bool = False,
        new_capture_timeout: int = 1000,
    ) -> 'np.ndarray | bool':
        """Grab and return an image from the camera.

        By default returns the last buffered frame. Set force_new_capture=True
        to trigger a fresh capture. Multiple frames can be summed for noise
        reduction via sum_count.

        Args:
            force_to_8bit: Convert 12-bit images to 8-bit output.
            earliest_image_ts: Reject frames captured before this timestamp.
            timeout: Max time to wait for a valid frame.
            all_ones_check: Reject saturated (all-max-value) frames.
            sum_count: Number of frames to sum for noise reduction.
            sum_delay_s: Delay in seconds between summed frames.
            sum_iteration_callback: Called after each summed frame.
            force_new_capture: If True, wait for a new camera capture.
            new_capture_timeout: Timeout (ms) for new capture grab.

        Returns:
            numpy.ndarray | False: Captured image array, or False on failure.
        """

        if not self.camera or not self.camera.active:
            return False

        tmp_buffer = []
        for idx in range(sum_count):
            start_time = datetime.datetime.now()
            stop_time = start_time + timeout

            while True:
                # Acquire cam_lock for camera grab — prevents concurrent
                # set_gain/set_exposure from another thread mid-frame.
                with self._cam_lock:
                    if force_new_capture:
                        grab_status, grab_image_ts = self.camera.grab_new_capture(new_capture_timeout)
                    else:
                        grab_status, grab_image_ts = self.camera.grab()

                    if grab_status:
                        self.frame_validity.count_frame()
                        tmp = self.camera.get_array()  # thread-safe copy

                if not grab_status:
                    # Check if camera disconnected — don't retry for 5 seconds
                    # if the camera is gone (H20).
                    if not self.camera.active:
                        logger.error("[SCOPE API ] get_image: camera disconnected")
                        from modules.notification_center import notifications
                        notifications.error("Camera", "Camera Disconnected",
                            "Camera is no longer available. Check USB connection.")
                        return False
                    if datetime.datetime.now() > stop_time:
                        logger.error(f"[SCOPE API ] get_image timeout ({stop_time}) exceeded")
                        return False
                    logger.debug("[SCOPE API ] get_image grab failed, retrying")
                    time.sleep(0.05)
                    continue

                if all_ones_check and not np.any(tmp != np.iinfo(tmp.dtype).max):
                    # Saturated frame — retry once to confirm, then accept.
                    # Saturated images are valid data (exposure/illumination
                    # too high), not a camera error. Don't loop until timeout.
                    retry_frame = None
                    with self._cam_lock:
                        retry_status, _ = self.camera.grab_new_capture(new_capture_timeout) if force_new_capture else self.camera.grab()
                        if retry_status:
                            self.frame_validity.count_frame()
                            retry_frame = self.camera.get_array()
                    # Saturation walk is outside cam_lock — no camera state needed,
                    # and the walk would otherwise block concurrent set_gain/set_exposure.
                    if retry_frame is not None:
                        if np.any(retry_frame != np.iinfo(retry_frame.dtype).max):
                            tmp = retry_frame  # retry was OK, use it
                        else:
                            logger.debug("[SCOPE API ] get_image: saturated frame confirmed on retry")

                # Accept the frame
                if earliest_image_ts is None:
                    tmp_buffer.append(tmp)
                    break

                if grab_image_ts > earliest_image_ts:
                    tmp_buffer.append(tmp)
                    break

                logger.warning(f"[SCOPE API ] get_image earliest_image_time {earliest_image_ts} not met -> Image TS: {grab_image_ts}")

                # Timestamp not met — check timeout then retry
                if datetime.datetime.now() > stop_time:
                    logger.error(f"[SCOPE API ] get_image timeout ({stop_time}) exceeded")
                    return False
                time.sleep(0.05)

            if sum_count > 1:
                earliest_image_ts = grab_image_ts + datetime.timedelta(milliseconds=1)
                if sum_iteration_callback is not None:
                    sum_iteration_callback()

                time.sleep(sum_delay_s)

        # PF-5: chain via a local variable instead of self.image_buffer. The
        # old field was a permanent shadow copy of the latest get_image result,
        # only ever read by get_image itself — Rule 2 violation that pinned a
        # frame indefinitely between calls. The _state_lock around per-write
        # didn't actually serialize concurrent get_image calls anyway (chained
        # writes from different threads could still interleave).
        if sum_count == 1:
            image = tmp if len(tmp_buffer) < 1 else tmp_buffer[0]
        else:
            orig_dtype = tmp_buffer[0].dtype
            max_value = np.iinfo(orig_dtype).max

            combined = np.zeros_like(tmp_buffer[0], dtype=np.uint32)
            for img in tmp_buffer:
                combined += img

            image = np.clip(combined, None, max_value).astype(orig_dtype)

        use_scale_bar = self._scale_bar['enabled']
        if self._objective is None:
            use_scale_bar = False

        if use_scale_bar:
            image = image_utils.add_scale_bar(
                image=image,
                objective=self._objective,
                binning_size=self._binning_size,
                color=self._scale_bar.get('color'),
            )

        if force_to_8bit and image.dtype != np.uint8:
            image = image_utils.convert_12bit_to_8bit(image)

        return image

    def get_image_with_chunks_from_buffer(
        self,
        force_to_8bit: bool = True,
    ) -> tuple:
        """Like ``get_image_from_buffer`` but also returns the per-frame chunks dict.

        Atomic snapshot: image + timestamp + chunks all came from the same
        grab. Used by the manual-record path so per-frame TIFF metadata
        reflects the camera-side chunk values for that exact frame (not
        a later frame's chunks paired with this frame's image).

        Returns:
            tuple: ``(image, timestamp, chunks)`` or ``(False, None, None)``
                if no frame is available. Chunks may be None for cameras
                without chunk support.
        """
        if not self.camera or not self.camera.active:
            return False, None, None

        grab_status, tmp, grab_image_ts, chunks = self.camera.grab_latest_with_chunks()
        if not grab_status or tmp is None:
            return False, None, None
        self.frame_validity.count_frame(chunk_data=chunks)

        with self._state_lock:
            self._frame_buffer = tmp

        use_scale_bar = self._scale_bar['enabled']
        if self._objective is None:
            use_scale_bar = False

        if use_scale_bar:
            tmp = image_utils.add_scale_bar(
                image=tmp,
                objective=self._objective,
                binning_size=self._binning_size,
                color=self._scale_bar.get('color'),
            )

        if force_to_8bit and tmp.dtype != np.uint8:
            tmp = image_utils.convert_12bit_to_8bit(tmp)

        return tmp, grab_image_ts, chunks

    def get_image_from_buffer(
        self,
        force_to_8bit: bool = True
        ) -> tuple:
        """Grab the latest buffered frame from the camera without forcing a new capture.

        Copy budget (per frame):
          - grab_latest(): 0 copies (returns reference from ImageHandler)
          - add_scale_bar(): 0 copies (modifies array in-place)
          - convert_12bit_to_8bit(): 1 copy (LUT indexing creates new array)
          - Total: 0 copies (8-bit) or 1 copy (12-bit with force_to_8bit)
          The caller adds 1 more copy via tobytes() for GPU blit.

        Args:
            force_to_8bit: Convert 12-bit images to 8-bit output.

        Returns:
            tuple: (image, timestamp) where image is numpy.ndarray and timestamp
                   is from the camera SDK, or (False, None) if unavailable.
        """
        if not self.camera or not self.camera.active:
            return False, None

        # Single-copy grab: grab_latest() returns the image directly,
        # avoiding the extra copy that grab() + get_array() would make.
        # This saves ~2.3MB copy + 1 lock acquisition per frame.
        grab_status, tmp, grab_image_ts = self.camera.grab_latest()
        if not grab_status or tmp is None:
            return False, None
        self.frame_validity.count_frame()

        with self._state_lock:
            self._frame_buffer = tmp

        use_scale_bar = self._scale_bar['enabled']
        if self._objective is None:
            use_scale_bar = False

        if use_scale_bar:
            tmp = image_utils.add_scale_bar(
                image=tmp,
                objective=self._objective,
                binning_size=self._binning_size,
                color=self._scale_bar.get('color'),
            )

        if force_to_8bit and tmp.dtype != np.uint8:
            tmp = image_utils.convert_12bit_to_8bit(tmp)

        return tmp, grab_image_ts

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
            x_target = self.get_target_position('X')
            y_target = self.get_target_position('Y')
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
                binning_size=self._binning_size,
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
            'exposure_time_ms': round(self.get_exposure_time(), common_utils.max_decimal_precision('exposure')),
            'gain_db': round(self.get_gain(), common_utils.max_decimal_precision('gain')),
            'illumination_ma': round(self.get_led_ma(color=color), common_utils.max_decimal_precision('illumination')),
            'binning_size': self._binning_size,
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
            handler = getattr(self.camera, 'cam_image_handler', None)
            chunks = handler.get_last_chunks() if handler is not None else None
        except Exception:
            chunks = None
        if chunks is not None:
            ts_ticks = chunks.get('Timestamp')
            if ts_ticks is not None:
                metadata['timestamp_camera_ticks'] = int(ts_ticks)
            tick_hz = getattr(self.camera, 'timestamp_tick_frequency_hz', None)
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

        # Camera silent-stuck or grab-timeout produces None; raise typed exception
        # so the IOTask popup carries an L1-friendly message instead of a raw
        # AttributeError traceback. Pairs with Rule 40 recovery contract work.
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

        # Bug-E diagnostic: env-gated handle-leak tracking.
        # Enable with LVP_HANDLE_TRACE=1; zero overhead when disabled.
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

        array = self.capture_and_wait(
            force_to_8bit=force_to_8bit,
            earliest_image_ts=earliest_image_ts,
            timeout=timeout,
            all_ones_check=all_ones_check,
            sum_count=sum_count,
            sum_delay_s=sum_delay_s,
            sum_iteration_callback=sum_iteration_callback,
        )

        if turn_off_all_leds_after:
            self.leds_off()

        if array is False:
            return

        return self.save_image(array, save_folder, file_root, append, color, tail_id_mode, output_format=output_format, true_color=true_color)


    def get_max_width(self) -> int:
        """Get the maximum pixel width of the camera sensor.

        Returns:
            int: Max width in pixels, or 0 if camera inactive.
        """
        if (not self.camera) or (not self.camera.active): return 0
        return self.camera.get_max_frame_size()['width']

    def get_max_height(self) -> int:
        """Get the maximum pixel height of the camera sensor.

        Returns:
            int: Max height in pixels, or 0 if camera inactive.
        """
        if (not self.camera) or (not self.camera.active): return 0
        return self.camera.get_max_frame_size()['height']

    def get_width(self) -> int:
        """Get the current frame width setting.

        Returns:
            int: Current width in pixels, or 0 if camera unavailable.
        """
        if not self.camera: return 0
        return self.camera.get_frame_size()['width']

    def get_height(self) -> int:
        """Get the current frame height setting.

        Returns:
            int: Current height in pixels, or 0 if camera unavailable.
        """
        if not self.camera: return 0
        return self.camera.get_frame_size()['height']

    def set_frame_size(self, w: int, h: int) -> None:
        """Set the camera frame size in pixels.

        Args:
            w: Frame width in pixels.
            h: Frame height in pixels.
        """

        if not self.camera or not self.camera.active: return
        self.camera.set_frame_size(w, h)
        with self._camera_cache_lock:
            self._camera_cache['frame_size'] = {'width': int(w), 'height': int(h)}

    def get_frame_size(self) -> dict | None:
        """Get the current camera frame size.

        Returns:
            dict | None: Contains 'width' and 'height' in pixels, or
                None if inactive.
        """

        if not self.camera or not self.camera.active: return
        return self.camera.get_frame_size()


    def get_gain(self) -> float:
        """Get the current camera gain.

        Returns:
            float: Gain in dB, or -1 if camera inactive.
        """

        if not self.camera or not self.camera.active: return -1
        return self.camera.get_gain()

    def set_gain(self, gain: float) -> None:
        """Set the camera gain.

        Args:
            gain: Gain value in dB.
        """
        if not self.camera or not self.camera.active: return
        # Skip redundant SDK call if gain hasn't changed
        if abs(float(gain) - self.camera_gain) < 0.001:
            return
        with self._cam_lock:
            self.camera.gain(gain)
        self.frame_validity.invalidate('gain')
        # Record requested gain so capture_and_wait's chunk-match can clear
        # the pending source once a frame's ChunkGain matches.
        self.frame_validity.set_target('gain', float(gain))
        with self._camera_cache_lock:
            self._camera_cache['gain'] = float(gain)
        _api_log.info(f'set_gain {gain}dB')
        self._fire_camera_listeners('gain', float(gain))

    def set_auto_gain(self, state: bool, settings: dict) -> None:
        """Enable or disable automatic gain adjustment.

        Args:
            state: True to enable auto gain, False to disable.
            settings: Dict with 'target_brightness', 'min_gain', 'max_gain'.
        """

        if not self.camera or not self.camera.active: return
        self.camera.auto_gain(
            state,
            target_brightness=settings['target_brightness'],
            min_gain=settings['min_gain'],
            max_gain=settings['max_gain'],
        )
        self.frame_validity.invalidate('gain')
        # Auto-gain dynamically adjusts the value; clear the manual target
        # so chunk-match falls back to skip-frames calibration.
        self.frame_validity.set_target('gain', None)

    @contextlib.contextmanager
    def suppress_value_warnings(self):
        """Suppress programmatic value-range warnings (sub-0.1ms exposure
        and similar) for the duration of the `with` block.

        Used by sweep-style internal callers (camera characterization
        dynamic_range / linearity stages) that walk the full setting
        range deliberately. The warnings exist for L1 researchers who
        type microsecond values thinking ms; they're noise when the
        char tool is exercising the API as designed.

        Restores the prior flag value (not unconditionally False) so
        nested `with` blocks behave correctly. Restoration runs on
        exception too -- exiting an exception-aborted char run leaves
        the API in a clean state for the next user action.
        """
        prior = self._suppress_value_warnings
        self._suppress_value_warnings = True
        try:
            yield
        finally:
            self._suppress_value_warnings = prior

    def set_exposure_time(self, t: float) -> None:
        """Set the camera exposure time.

        Args:
            t: Exposure time in milliseconds.
        """
        if not self.camera or not self.camera.active: return
        # Skip redundant SDK call if exposure hasn't changed
        if abs(float(t) - self.camera_exposure_ms) < 0.001:
            return
        if t < 0.1 and not self._suppress_value_warnings:
            import traceback
            _caller = ''.join(traceback.format_stack(limit=6)[-4:-1]).strip()
            logger.warning(f'[SCOPE API ] set_exposure_time({t}ms) is very low -- '
                           f'image will be nearly black. Value should be in milliseconds.\n'
                           f'Call stack:\n{_caller}')
        with self._cam_lock:
            self.camera.exposure_t(t)
        self.frame_validity.invalidate('exposure')
        # Record requested exposure for chunk-match. ChunkExposureTime is
        # microseconds; the API takes milliseconds. Convert at the seam so
        # the chunk value and frame_validity's tolerance share units.
        self.frame_validity.set_target('exposure', float(t) * 1000.0)
        with self._camera_cache_lock:
            self._camera_cache['exposure_ms'] = float(t)
        _api_log.info(f'set_exposure {t}ms')
        self._fire_camera_listeners('exposure', float(t))

    def get_exposure_time(self) -> float:
        """Get the current camera exposure time.

        Returns:
            float: Exposure time in milliseconds, or 0 if camera inactive.
        """

        if not self.camera or not self.camera.active: return 0
        exposure = self.camera.get_exposure_t()
        return exposure

    def set_auto_exposure_time(self, state: bool = True) -> None:
        """Enable or disable automatic exposure adjustment.

        Args:
            state: True to enable auto exposure, False to disable.
        """

        if not self.camera or not self.camera.active: return
        self.camera.auto_exposure_t(state)
        self.frame_validity.invalidate('exposure')
        # Auto-exposure dynamically adjusts the value; clear the manual
        # target so chunk-match falls back to skip-frames calibration.
        self.frame_validity.set_target('exposure', None)

    def apply_layer_camera_settings(self, gain: float, exposure_ms: float,
                                     auto_gain: bool = False,
                                     auto_gain_settings: dict | None = None) -> None:
        """Apply per-layer camera settings in a single batched call.

        Sets gain, exposure, and auto-gain state. Replaces 3 separate
        IOTask queues with a single call for atomicity.

        Args:
            gain: Camera gain in dB.
            exposure_ms: Exposure time in milliseconds.
            auto_gain: Whether auto-gain is enabled for this layer.
            auto_gain_settings: Dict with target_brightness, min_gain, max_gain
                               (required if auto_gain is True).
        """
        if not self.camera or not self.camera.active:
            return
        self.set_gain(gain)
        self.set_exposure_time(exposure_ms)
        if auto_gain_settings is not None:
            self.set_auto_gain(auto_gain, settings=auto_gain_settings)
        _api_log.info(f'apply_layer_camera_settings gain={gain}dB exp={exposure_ms}ms auto_gain={auto_gain}')

    def update_auto_gain_target_brightness(self, target_brightness: float) -> None:
        """Set the auto-gain target brightness on the camera.

        Args:
            target_brightness: Target brightness value (0.0 to 1.0).
        """
        if not self.camera or not self.camera.active:
            return
        self.camera.update_auto_gain_target_brightness(target_brightness)

    def auto_gain_once(self, state: bool, target_brightness: float,
                       min_gain: float, max_gain: float) -> None:
        """Run auto-gain for a single frame on the camera.

        Args:
            state: True to enable one-shot auto-gain.
            target_brightness: Target brightness (0.0 to 1.0).
            min_gain: Minimum gain in dB.
            max_gain: Maximum gain in dB.
        """
        if not self.camera or not self.camera.active:
            return
        self.camera.auto_gain_once(
            state=state,
            target_brightness=target_brightness,
            min_gain=min_gain,
            max_gain=max_gain,
        )

    def update_camera_config(self):
        """Context manager for batched camera config updates.

        Usage::

            with scope.update_camera_config():
                scope.set_gain(5.0)
                scope.set_exposure_time(100)

        Returns:
            A context manager. Falls back to ``contextlib.nullcontext()``
            when no camera is active.
        """
        if not self.camera or not self.camera.active:
            return contextlib.nullcontext()
        return self.camera.update_camera_config()

    def camera_is_connected(self) -> bool:
        """Check if the camera is active and connected.

        Returns:
            bool: True if camera is connected and active.
        """
        if not self.camera or not self.camera.active:
            return False

        return self.camera.is_connected()

        #return True

    def get_camera_temps(self) -> dict:
        """Get camera temperature readings.

        Returns:
            dict: Mapping of sensor name to temperature in Celsius. Empty if inactive.
        """

        if not self.camera or not self.camera.active:
            return {}

        return self.camera.get_all_temperatures()

    def log_camera_temps(self) -> None:
        """Emit one INFO line per camera temperature sensor.

        No-op when no camera is connected. Called once on startup and
        periodically by ``start_camera_temp_logging``.
        """
        if not self.camera_is_connected():
            return
        for source, temp in self.get_camera_temps().items():
            logger.info(
                f'[CAM Class ] Camera {source} Temperature : {temp:.2f} degC')

    def start_camera_temp_logging(
        self, schedule_interval_fn, unschedule_fn, *,
        interval_s: float = 14400.0) -> None:
        """LVP-A-2: own the periodic camera-temp logging schedule.

        Was previously a Clock.schedule_interval registered by the App
        and stored as a fresh attribute on the MainDisplay widget -- if
        MainDisplay was ever recreated (LS850/LS620 scope swap), the
        Clock event became orphaned and continued logging temps from a
        now-disconnected camera.

        Args:
            schedule_interval_fn: Callable matching ``Clock.schedule_interval(func, interval)``.
                Passed in so this module stays GUI-agnostic per Rule 15.
            unschedule_fn: Callable matching ``Clock.unschedule(event)``,
                used by ``stop_camera_temp_logging`` and on
                disconnect-while-logging.
            interval_s: Seconds between log emissions; default 4 hours.
        """
        # Defensive: if a previous logger is already running, stop it
        # before starting a new one (idempotent — safe to call repeatedly).
        if getattr(self, '_camera_temp_event', None) is not None:
            self.stop_camera_temp_logging(unschedule_fn)

        self._camera_temp_unschedule_fn = unschedule_fn
        self.log_camera_temps()  # one immediate sample

        def _tick(_dt=0):
            # Self-unschedule when the camera disconnects so a stale
            # event doesn't survive scope switches.
            if not self.camera_is_connected():
                self.stop_camera_temp_logging(unschedule_fn)
                return
            self.log_camera_temps()

        self._camera_temp_event = schedule_interval_fn(_tick, interval_s)
        logger.info(
            f'[SCOPE API ] start_camera_temp_logging: interval={interval_s}s')

    def stop_camera_temp_logging(self, unschedule_fn=None) -> None:
        """Cancel the periodic camera-temp logger if active.

        Idempotent -- safe to call when no logger is running. The
        unschedule_fn arg is optional; falls back to the function passed
        at start_camera_temp_logging time.
        """
        ev = getattr(self, '_camera_temp_event', None)
        if ev is None:
            return
        try:
            (unschedule_fn or self._camera_temp_unschedule_fn)(ev)
        except Exception as e:
            logger.warning(
                f'[SCOPE API ] stop_camera_temp_logging unschedule failed: {e}')
        self._camera_temp_event = None

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
        before = self.get_limit_switch_status_all_axes()
        logger.info(f"Limit switch status before homing: {before}", extra={'force_error': True})
        yield
        after = self.get_limit_switch_status_all_axes()
        logger.info(f"Limit switch status after homing: {after}", extra={'force_error': True})

    def get_axes_config(self) -> dict:
        """Get the axis configuration from the motion board.

        Returns:
            dict: Axis configuration (axes present, limits, etc.).
        """
        return self.motion.get_axes_config()

    def get_axis_limits(self, axis: str) -> dict:
        """Get the travel limits for an axis.

        Args:
            axis: Axis name ("X", "Y", "Z", or "T").

        Returns:
            dict: Contains 'min' and 'max' positions in um.
        """

        return self.motion.get_axis_limits(axis=axis)


    def zhome(self) -> bool:
        """Home the Z axis (focus).

        Returns:
            bool: True on successful Z homing. False if the driver
                returned False or raised (e.g. HardwareError on
                no-response / firmware-error). The user is notified on
                failure; programmatic callers can branch on the bool.
        """
        #if not self.motion: return
        _api_log.info('zhome START')
        self._set_axis_state('Z', AxisState.HOMING)
        self.frame_validity.invalidate('z_move')
        try:
            with self.reference_position_logger():
                result = self.motion.zhome()
            if result is False:
                logger.error('[SCOPE API ] Z homing failed')
                notifications.error("Motion", "Homing Failed",
                    "Z axis homing failed. Position is unknown.")
                self._set_axis_state('Z', AxisState.UNKNOWN)
                return False
            self._set_axis_state('Z', AxisState.IDLE)
            self.refresh_position_cache()
            _api_log.info('zhome DONE')
            return True
        except Exception:
            logger.exception('[SCOPE API ] Z homing exception')
            self._set_axis_state('Z', AxisState.UNKNOWN)
            notifications.error("Motion", "Homing Error",
                "Z axis homing encountered an error. Position is unknown.")
            _api_log.info('zhome DONE')
            return False

    def home(self) -> bool:
        """Home every axis the motor board has.

        This is the unified "home everything" entry point used by
        startup and the GUI Home button. The firmware's home routine
        homes Z, then T, then X/Y -- on a Z-only board (LS820) it homes
        Z and reports the missing X/Y; on a full XYZ scope it homes
        all three. The driver returns True for both cases (full and
        partial), raises HardwareError on real failure.

        Returns:
            bool: True on full or partial success. False if the motor
                is not connected, the driver returned False, or the
                driver raised (HardwareError or other). The user is
                notified on failure; programmatic callers can branch on
                the bool.
        """
        # Short-circuit on disconnected motor — without this, home()
        # dispatches into the driver where exchange_command tries to
        # auto-reconnect and burns its full timeout (~10 s). That was
        # the user-perceived "spinning beachball" in #632. Fire ONE
        # clean Rule 14 notification with the right cause, instead of
        # the misleading "Homing Failed. Position is unknown" that
        # implies a homing-mechanics problem.
        if not self.motor_connected:
            logger.warning('[SCOPE API ] home() called with motor not connected')
            notifications.error(
                "Motion",
                "Motor Not Connected",
                "Cannot home -- motor controller is not connected. "
                "Check the USB cable and that no other program "
                "(Thonny, mpremote, etc.) is holding the port.",
            )
            return False
        present_axes = self.axes_present()
        _api_log.info('home START')
        for ax in present_axes:
            self._set_axis_state(ax, AxisState.HOMING)
        if 'Z' in present_axes:
            self.frame_validity.invalidate('z_move')
        if 'X' in present_axes or 'Y' in present_axes:
            self.frame_validity.invalidate('xy_move')
        if 'T' in present_axes:
            self.frame_validity.invalidate('turret')
        self.is_homing = True
        try:
            with self.reference_position_logger():
                result = self.motion.home()
            if result is False:
                logger.error('[SCOPE API ] Homing failed')
                notifications.error("Motion", "Homing Failed",
                    "Homing failed. Position is unknown.")
                for ax in present_axes:
                    self._set_axis_state(ax, AxisState.UNKNOWN)
                return False
            for ax in present_axes:
                self._set_axis_state(ax, AxisState.IDLE)
            self.refresh_position_cache()
            return True
        except Exception:
            logger.exception('[SCOPE API ] Homing exception')
            for ax in present_axes:
                self._set_axis_state(ax, AxisState.UNKNOWN)
            notifications.error("Motion", "Homing Error",
                "Homing encountered an error. Position is unknown.")
            return False
        finally:
            self.is_homing = False
            _api_log.info('home DONE')

    def has_homed(self) -> bool:
        """Check if the scope has been homed since startup.

        Returns:
            bool: True if home() has succeeded at least once.
        """
        return self.motion.has_homed()

    def xycenter(self) -> None:
        """Move the XY stage to center position."""

        #if not self.motion: return
        self._set_axis_state('X', AxisState.MOVING)
        self._set_axis_state('Y', AxisState.MOVING)
        self.motion.xycenter()
        self._set_axis_state('X', AxisState.IDLE)
        self._set_axis_state('Y', AxisState.IDLE)
        self.refresh_position_cache()


    @contextlib.contextmanager
    def safe_turret_mover(self):
        """Context manager that lowers Z to 0 before turret motion and restores after.

        Use as ``with scope.safe_turret_mover(): ... move turret ...``.
        Sets ``is_turreting`` for the duration and restores the original
        Z position even if the body raises.
        """
        # Save off current Z position before moving Z to 0
        logger.info('[SCOPE API ] Moving Z to 0', extra={'force_error': True})
        initial_z = self.get_current_position(axis='Z')
        self.move_absolute_position('Z', pos=0, wait_until_complete=True)
        self.is_turreting = True
        try:
            yield
        finally:
            # Always clear the flag and restore Z, even if the body raised
            # (e.g. driver HardwareError from thome). Without this, a failed
            # turret home would leave is_turreting=True and the stage stuck
            # at Z=0.
            self.is_turreting = False
            logger.info(f'[SCOPE API ] Restoring Z to {initial_z}', extra={'force_error': True})
            self.move_absolute_position('Z', pos=initial_z, wait_until_complete=True)


    def thome(self) -> bool:
        """Home the turret axis. Moves Z to 0 during turret motion for safety.

        Returns:
            bool: True on successful turret homing (or when the board
                reports the turret is not present). False if the motor
                is not connected, the driver returned False, or the
                driver raised (HardwareError or other). The user is
                notified on failure; programmatic callers can branch on
                the bool.
        """
        # Short-circuit on disconnected motor — same rationale as
        # home() above. Without this, thome dispatches into the driver
        # where exchange_command burns its 15s timeout doing failed
        # auto-reconnect attempts. Fire one clean Rule 14 notification.
        if not self.motor_connected:
            logger.warning('[SCOPE API ] thome() called with motor not connected')
            notifications.error(
                "Motion",
                "Motor Not Connected",
                "Cannot home turret -- motor controller is not connected. "
                "Check the USB cable and that no other program is "
                "holding the port.",
            )
            return False

        # Move turret — set HOMING after Z is safe, not before.
        # Setting T to HOMING clears its arrival event, which would block
        # wait_until_finished_moving() inside safe_turret_mover's Z move.
        _api_log.info('thome START')
        try:
            with self.reference_position_logger():
                with self.safe_turret_mover():
                    self._set_axis_state('T', AxisState.HOMING)
                    self.frame_validity.invalidate('turret')
                    result = self.motion.thome()
            if result is False:
                logger.error('[SCOPE API ] Turret homing failed')
                notifications.error("Motion", "Homing Failed",
                    "Turret homing failed. Position is unknown.")
                self._set_axis_state('T', AxisState.UNKNOWN)
                return False
            self._set_axis_state('T', AxisState.IDLE)
            self.refresh_position_cache()
            _api_log.info('thome DONE')
            return True
        except Exception:
            logger.exception('[SCOPE API ] Turret homing exception')
            self._set_axis_state('T', AxisState.UNKNOWN)
            notifications.error("Motion", "Homing Error",
                "Turret homing encountered an error. Position is unknown.")
            _api_log.info('thome DONE')
            return False

    def has_thomed(self) -> bool:
        """Check if the turret has been homed since startup.

        Returns:
            bool: True if turret homing has been performed.
        """
        return self.motion.has_thomed()

    def tmove(self, position: int) -> None:
        """Move the turret to a specific position. Skips if already there.

        Args:
            position: Target turret position (1-4).
        """
        # Commanding a move of the T axis is slow, even if the move is to the current position.
        # Use caching to determine if T is requested to move to it's current position, and bypass the
        # move altogether if it is.
        if self._last_turret_position == position:
            return

        with self.safe_turret_mover():
            logger.info(f'[SCOPE API ] Moving T to position {position}')
            self.move_absolute_position('T', position, wait_until_complete=True)
            self._last_turret_position = position


    def has_turret(self) -> bool:
        """Check if the microscope has a turret axis.

        Thin wrapper over ``self.capabilities.has_turret``.

        Returns:
            bool: True if the scope reports a turret axis.
        """
        return self.capabilities.has_turret


    def refresh_position_cache(self) -> None:
        """Fetch all axis positions from hardware and update the cache.

        Called after homing completes to sync the cache with actual hardware
        positions.  During normal operation the cache is updated directly
        by move commands -- no polling needed.
        """
        positions = {}
        for ax in self.axes_present():
            try:
                pos = self.motion.target_pos(axis=ax)
                positions[ax] = pos if pos is not None else 0.0
            except Exception:
                positions[ax] = 0.0

        with self._pos_cache_lock:
            self._pos_cache.update(positions)
        for ax in positions:
            self._fire_position_listeners(ax)

    def get_target_position(self, axis: str | None = None) -> 'float | dict | None':
        """Get the target position for an axis (where it is commanded to go).

        Reads from the push-based position cache -- zero serial I/O.

        Args:
            axis: Axis name ("X", "Y", "Z", "T"), or None for all axes.

        Returns:
            float | dict: Position in um for a single axis, or dict of all
                axis positions. Returns 0 if motion board inactive, None if
                axis T requested but no turret present.
        """
        if (not self.motion.has_turret()) and (axis == 'T'):
            return None

        with self._pos_cache_lock:
            if axis is None:
                return dict(self._pos_cache)
            return self._pos_cache.get(axis, 0.0)

    def get_current_position(self, axis: str | None = None) -> 'float | dict':
        """Get the current position for an axis.

        During MOVING: returns predicted position based on trapezoidal
        ramp profile and elapsed time (smooth UI updates, zero serial I/O).
        During IDLE: returns cached target position (confirmed by firmware).

        Args:
            axis: Axis name ("X", "Y", "Z", "T"), or None for all axes.

        Returns:
            float | dict: Position in um for a single axis, or dict of all
                axis positions. Returns 0 if motion board inactive.
        """
        if axis is None:
            result = {}
            for ax in self.axes_present():
                result[ax] = self.get_current_position(ax)
            return result

        # If axis is moving and we have a motion profile, return predicted position.
        # The predictor gives smooth interpolation between 50Hz firmware polls.
        # If prediction fails or isn't available, fall through to cached target.
        with self._axis_state_lock:
            state = self._axis_state.get(axis, AxisState.UNKNOWN)
        if state == AxisState.MOVING:
            predicted = self._predicted_position(axis)
            if predicted is not None:
                return predicted

        # IDLE or no profile: cached target position (confirmed by firmware)
        with self._pos_cache_lock:
            return self._pos_cache.get(axis, 0.0)

    def _predicted_position(self, axis: str) -> float | None:
        """Predict position during a move using the trapezoidal ramp profile.

        Returns None if no motion profile is available (falls back to cache).
        Supports simple trapezoidal (a1/v1/d1=0) and 6-point ramps.
        """
        with self._move_profile_lock:
            profile = self._move_profile.get(axis)
            if profile is None:
                return None
            start_time = profile['start_time']
            start_pos = profile['start_pos']
            target_pos = profile['target_pos']
            ramp = profile['ramp']

        elapsed = time.monotonic() - start_time
        distance = abs(target_pos - start_pos)
        if distance < 0.01:  # trivially short move
            return target_pos
        direction = 1.0 if target_pos > start_pos else -1.0

        vmax = ramp['vmax']
        amax = ramp['amax']
        dmax = ramp['dmax']
        if amax <= 0 or dmax <= 0 or vmax <= 0:
            return None  # invalid ramp params

        # Simple trapezoidal profile (a1/v1/d1 are zero)
        t_accel = vmax / amax
        t_decel = vmax / dmax
        s_accel = 0.5 * amax * t_accel * t_accel
        s_decel = 0.5 * dmax * t_decel * t_decel

        if distance <= (s_accel + s_decel):
            # Triangular profile — never reaches VMAX
            import math
            t_peak = math.sqrt(2.0 * distance / (amax + amax * amax / dmax))
            v_peak = amax * t_peak
            s_accel_tri = 0.5 * amax * t_peak * t_peak
            t_decel_tri = v_peak / dmax
            total_time = t_peak + t_decel_tri

            if elapsed >= total_time:
                return target_pos
            elif elapsed <= t_peak:
                s = 0.5 * amax * elapsed * elapsed
            else:
                dt = elapsed - t_peak
                s = s_accel_tri + v_peak * dt - 0.5 * dmax * dt * dt
        else:
            # Full trapezoidal profile
            s_cruise = distance - s_accel - s_decel
            t_cruise = s_cruise / vmax
            total_time = t_accel + t_cruise + t_decel

            if elapsed >= total_time:
                return target_pos
            elif elapsed <= t_accel:
                s = 0.5 * amax * elapsed * elapsed
            elif elapsed <= (t_accel + t_cruise):
                dt = elapsed - t_accel
                s = s_accel + vmax * dt
            else:
                dt = elapsed - t_accel - t_cruise
                s = s_accel + s_cruise + vmax * dt - 0.5 * dmax * dt * dt

        # Clamp to [start, target] — never overshoot in prediction
        s = max(0.0, min(s, distance))
        return start_pos + direction * s


    def get_actual_position(self, axis: str) -> float:
        """Query the actual hardware position via serial (not cached).

        Unlike get_current_position() which returns the last commanded
        target, this queries the motor controller for where it actually is
        right now. Use during continuous motion sweeps where the stage is
        moving and the cache doesn't reflect the true position.

        Costs one serial round-trip (~5ms).

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            float: Current position in um. 0 if motor not connected.
        """
        if not self.motor_connected:
            return 0.0
        pos = self.motion.current_pos(axis)
        return pos if pos is not None else 0.0

    def set_motor_precision_mode(self, axis: str, enabled: bool) -> None:
        """Set motor precision mode for an axis.

        Precision mode uses accurate but slightly slower motor stopping.
        Use before autofocus fine passes or any measurement requiring
        precise Z positioning. Disable for coarse moves where speed
        matters more than final position accuracy.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            enabled: True for precise positioning, False for speed.
        """
        if not self.motor_connected:
            return
        self.motion.set_precision_mode(axis, enabled)


    def move_absolute_position(self, axis: str, pos: float,
                               wait_until_complete: bool = False,
                               overshoot_enabled: bool = True,
                               ignore_limits: bool = False) -> None:
        """Move an axis to an absolute position.

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").
            pos (float): Target position in um.
            wait_until_complete: If True, block until move finishes.
            overshoot_enabled: Allow Z overshoot for backlash compensation.
            ignore_limits: If True, skip software limit checks.

        Raises:
            ValueError: If axis is invalid or pos is not numeric / out of bounds.
        """
        if axis not in self._VALID_AXIS_NAMES:
            raise ValueError(f"Axis must be one of {self._VALID_AXIS_NAMES}, got {axis!r}")
        if not isinstance(pos, (int, float)):
            raise ValueError(f"Position must be numeric, got {type(pos).__name__}")
        if abs(pos) > self.MOTOR_POSITION_LIMIT:
            raise ValueError(f"Position {pos} um exceeds safety limit of +/-{self.MOTOR_POSITION_LIMIT} um")

        # Rule 8: silently no-op for axes that aren't present on this
        # hardware. _arrival_events is sized to detect_present_axes() at
        # init, so this is the canonical "is this axis trackable" check.
        if axis not in self._arrival_events:
            _api_log.debug(f'move_abs ignored: {axis} not present on this scope')
            return

        # Store motion profile for position prediction before moving
        with self._pos_cache_lock:
            start_pos = self._pos_cache.get(axis, 0.0)
        try:
            ramp = self.motion.motorconfig.ramp_params(axis)
        except Exception:
            ramp = None
        if ramp:
            with self._move_profile_lock:
                self._move_profile[axis] = {
                    'start_time': time.monotonic(),
                    'start_pos': start_pos,
                    'target_pos': float(pos),
                    'ramp': ramp,
                }

        # Write the hardware target BEFORE transitioning the axis to MOVING.
        # Previously the order was reversed: _set_axis_state(MOVING) cleared
        # the arrival event and woke the motion monitor, then motion.move_abs_pos
        # spent ~50ms on serial I/O (current_pos read + TARGET_W write) before
        # the hardware actually received the new target. During that window
        # the motion monitor could poll STATUS_R, observe the PRIOR move's
        # still-valid position_reached bit, and falsely set the arrival
        # event — causing wait_until_finished_moving to return before the
        # new move even began. See issue #618. With this order, by the
        # time the axis is marked MOVING the hardware XTARGET is already
        # the new value, so position_reached is reliably False and the
        # motion monitor polls until real arrival.
        try:
            self.motion.move_abs_pos(axis, pos, overshoot_enabled=overshoot_enabled, ignore_limits=ignore_limits)
        except Exception as e:
            with self._move_profile_lock:
                self._move_profile[axis] = None
            _api_log.error(f'move_abs {axis}={pos:.1f}um FAILED: {e}')
            raise
        self._set_axis_state(axis, AxisState.MOVING)
        with self._pos_cache_lock:
            self._pos_cache[axis] = float(pos)
        self._fire_position_listeners(axis)
        self.frame_validity.invalidate('z_move' if axis == 'Z' else 'xy_move')
        _api_log.info(f'move_abs {axis}={pos:.1f}um'
                      f'{" wait" if wait_until_complete else ""}')

        if wait_until_complete is True:
            self.wait_until_finished_moving()
            self._set_axis_state(axis, AxisState.IDLE)


    def move_relative_position(self, axis: str, um: float,
                               wait_until_complete: bool = False,
                               overshoot_enabled: bool = False) -> None:
        """Move an axis by a relative distance.

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").
            um (float): Distance to move in um.
            wait_until_complete: If True, block until move finishes.
            overshoot_enabled: Allow Z overshoot for backlash compensation.

        Raises:
            ValueError: If axis is invalid or um is not numeric / out of bounds.
        """
        if axis not in self._VALID_AXIS_NAMES:
            raise ValueError(f"Axis must be one of {self._VALID_AXIS_NAMES}, got {axis!r}")
        if not isinstance(um, (int, float)):
            raise ValueError(f"Distance must be numeric, got {type(um).__name__}")
        if abs(um) > self.MOTOR_POSITION_LIMIT:
            raise ValueError(f"Distance {um} um exceeds safety limit of +/-{self.MOTOR_POSITION_LIMIT} um")

        # Rule 8: silently no-op for axes that aren't present on this
        # hardware. See move_absolute_position for the rationale.
        if axis not in self._arrival_events:
            _api_log.debug(f'move_rel ignored: {axis} not present on this scope')
            return

        # Write hardware target BEFORE transitioning axis to MOVING —
        # same race fix as move_absolute_position (#618).
        try:
            self.motion.move_rel_pos(axis, um, overshoot_enabled=overshoot_enabled)
        except Exception as e:
            _api_log.error(f'move_rel {axis}={um:+.1f}um FAILED: {e}')
            raise
        self._set_axis_state(axis, AxisState.MOVING)
        with self._pos_cache_lock:
            self._pos_cache[axis] = self._pos_cache.get(axis, 0.0) + float(um)
        self._fire_position_listeners(axis)
        self.frame_validity.invalidate('z_move' if axis == 'Z' else 'xy_move')
        _api_log.info(f'move_rel {axis}={um:+.1f}um'
                      f'{" wait" if wait_until_complete else ""}')

        if wait_until_complete is True:
            self.wait_until_finished_moving()
            self._set_axis_state(axis, AxisState.IDLE)


    def get_home_status(self, axis: str) -> bool:
        """Check if an axis is at its home position.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            bool: True if the axis is homed, False otherwise or on error.
        """

        #if not self.motion: return True
        try:
            status = self.motion.home_status(axis)
            return status
        except Exception as e:
            logger.exception(f"[SCOPE API ] get_home_status({axis}) failed; treating as not home: {e}")
            return False

    def get_target_status(self, axis: str) -> bool:
        """Check if an axis has reached its target position.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            bool: True if at target (always True for T if no turret present).
        """

        #if not self.motion: return True

        # Handle case where we want to know if turret has reached its target, but there is no turret
        if (axis == 'T') and (not self.motion.has_turret()):
            return True

        try:
            status = self.motion.target_status(axis)
            return status
        except Exception as e:
            logger.exception(f"[SCOPE API ] get_target_status({axis}) failed; treating as not at target: {e}")
            return False

    def get_target_pos(self, axis: str) -> float:
        """Get the target position for an axis (error-safe version).

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            float: Target position in um, or -1 on error/no turret.
        """
        if (axis == 'T') and (not self.motion.has_turret()):
            return -1

        try:
            pos = self.motion.target_pos(axis)
            return pos if pos is not None else -1
        except Exception as e:
            logger.exception(f"[SCOPE API ] get_target_pos({axis}) failed; returning -1: {e}")
            return -1

    def get_reference_status(self, axis: str) -> str:
        """Get reference status register bits for an axis.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            str: 32-character binary string of register bits (MSB first).
        """

        #if not self.motion: return
        return self.motion.reference_status(axis=axis)


    def get_limit_switch_status(self, axis: str):
        """Get the limit switch status for an axis.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            Limit switch state for the specified axis (driver-defined).
        """
        return self.motion.limit_switch_status(axis=axis)


    def get_limit_switch_status_all_axes(self) -> dict:
        """Get limit switch status for all axes.

        Returns:
            dict: Mapping of axis name to limit switch state.
        """
        resp = {}
        for axis in self.axes_present():
            resp[axis] = self.get_limit_switch_status(axis=axis)
        return resp


    def get_overshoot(self) -> bool:
        """Check if the Z axis is currently in overshoot (backlash compensation) mode.

        Returns:
            bool: True if overshoot is in progress.
        """

        #if not self.motion: return False
        return self.motion.overshoot

    def is_moving(self) -> bool:
        """Check if any axis is currently moving.

        Reads from in-memory axis state -- zero serial I/O. The motion
        monitor thread handles firmware queries and state transitions.

        Returns:
            bool: True if any axis is MOVING/HOMING or overshoot is active.
        """
        if self.is_any_axis_moving():
            return True
        if self.get_overshoot():
            return True
        return False

    def wait_until_finished_moving(self, timeout: float = 120.0) -> bool:
        """Block until all axes have reached their target positions.

        Waits on per-axis arrival events set by the motion monitor thread.
        Zero serial I/O from the calling thread -- all firmware queries
        happen on the monitor thread at 50 Hz.

        Args:
            timeout: Maximum seconds to wait (default 120s).

        Returns:
            bool: True if all axes arrived, False if timed out.
        """
        deadline = time.monotonic() + timeout
        # Iterate arrival events directly (not axes_present) so a transient
        # motion.detect_present_axes() failure at call time can never cause
        # this to return True without actually waiting for the in-flight
        # move. _arrival_events was sized to detect_present_axes() at init
        # and never changes shape thereafter, so iterating its keys is the
        # canonical "every axis this scope can track" set. Events for
        # non-moving axes are .set() by construction.
        for ax in self._arrival_events:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(f'[SCOPE API ] wait_until_finished_moving timed out on axis {ax}')
                return False
            if not self._arrival_events[ax].wait(timeout=remaining):
                logger.warning(f'[SCOPE API ] wait_until_finished_moving timed out on axis {ax}')
                return False

        return True


    def set_acceleration_limit(self, val_pct: int) -> None:
        """Set the motor controller acceleration limit (percent of max).

        Silently ignores firmware that doesn't implement the command --
        legacy boards lack the acceleration-limits feature.

        Args:
            val_pct: Acceleration limit as a percent of the firmware max.
        """
        try:
            self.motion.set_acceleration_limits(val_pct=val_pct)
        except Exception:
            pass  # Legacy firmware doesn't support acceleration limits


    def get_microscope_model(self) -> str | None:
        """Get the microscope model identifier from the motion board.

        Returns:
            str | None: Model string, or None if motion board inactive.
        """
        # LV-6 PROBE: log every call so we can determine whether the
        # MainThread invocation from ui/microscope_settings.py:328 in
        # load_settings() goes to the wire or hits a driver-side cache.
        # If wire-bound, this is part of the MainThread-blocking startup
        # cluster (CAM-2 / Cluster A). If cached, decide-not-to-fix.
        # Remove this probe after the disposition is recorded.
        try:
            tname = threading.current_thread().name
            logger.info(f"[LV-6 DIAG] get_microscope_model() called from thread={tname}")
        except Exception:
            pass
        return self.motion.get_microscope_model()

    def get_motor_info(self) -> dict:
        """Get motor controller information.

        Returns:
            dict: Keys 'model', 'serial_number', 'firmware_version'.
                  Values are None/unknown if board inactive.
        """
        info = self.motion.fullinfo()
        return {
            'model': info.get('model', 'unknown'),
            'serial_number': info.get('serial_number', 'unknown'),
            'firmware_version': getattr(self.motion, 'firmware_version', None),
        }

    def get_led_info(self) -> dict:
        """Get LED controller information.

        Returns:
            dict: Keys 'firmware_version', 'connected'.
        """
        if not self.led or not self.led.is_connected():
            return {'firmware_version': None, 'connected': False}

        return {
            'firmware_version': getattr(self.led, 'firmware_version', None),
            'connected': True,
        }

    def get_camera_info(self) -> dict:
        """Get camera information.

        Returns:
            dict: Keys 'model', 'pixel_format', 'connected'.
        """
        if not self.camera or not self.camera.active:
            return {'model': None, 'pixel_format': None, 'connected': False}

        return {
            'model': self.camera.get_model_name(),
            'pixel_format': self.camera.get_pixel_format(),
            'connected': True,
        }

    def get_camera_temperatures(self) -> dict:
        """Get all camera temperature sensor readings.

        Returns:
            dict: Mapping of sensor name to temperature in °C.
            Empty dict if camera is inactive or has no temperature sensors.
        """
        if not self.camera or not self.camera.active:
            return {}
        try:
            return self.camera.get_all_temperatures()
        except Exception as e:
            logger.debug(f'[SCOPE API ] get_camera_temperatures failed: {e}')
            return {}

    # ------------------------------------------------------------------
    # Diagnostic API (LAYER-D / LV-23, LV-24, LV-32, LV-40)
    # Tech-support / bring-up / bench tools route diagnostics through
    # these methods so the API layer owns Rule-13 logging and Rule-14
    # error visibility. Modules MUST NOT call `self.camera.get_image()`,
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
        if not self.camera or not self.camera.active:
            return {'connected': False}

        info: dict = {'connected': True}

        def _try(key, fn):
            try:
                info[key] = fn()
            except Exception as e:
                info[key] = f'Error: {e}'

        _try('model', lambda: self.camera.get_model_name())
        _try('pixel_format', lambda: self.camera.get_pixel_format())

        try:
            fs = self.camera.get_frame_size()
            info['resolution'] = f"{fs.get('width', '?')}x{fs.get('height', '?')}"
            info['frame_size'] = fs
        except Exception as e:
            info['resolution'] = f'Error: {e}'

        _try('gain', lambda: self.get_gain())
        _try('exposure_ms', lambda: self.get_exposure_time())
        _try('max_gain', lambda: self.camera.get_max_gain())
        _try('max_exposure_ms', lambda: self.camera.get_max_exposure())

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
        sees. Bypassing this method (calling ``self.camera.get_image()``
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

        if not self.camera or not self.camera.active:
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
                frame = self.get_image(force_to_8bit=False, force_new_capture=True)
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

        Stays inside the API: drops to ``self.camera.stop_grabbing`` /
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

        if not self.camera or not self.camera.active:
            results['errors'].append('Camera not active')
            return results

        cam_info = self.get_camera_diagnostic_info()
        results['camera_model'] = cam_info.get('model')
        results['pylon_version'] = cam_info.get('sdk_version') or cam_info.get('pylon_version')

        cycle_times, stop_times, start_times = [], [], []
        delay_s = max(0.0, float(inter_cycle_delay_ms) / 1000.0)

        # Snapshot current settings so we can restore even when vary_settings
        # is on — the benchmark must not leave the camera in an arbitrary state.
        original_gain = getattr(self.camera, 'gain', None)
        original_exposure = getattr(self.camera, 'exposure_time', None)

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
                self.camera.stop_grabbing()
                stop_s = time.monotonic() - t0

                if delay_s > 0:
                    time.sleep(delay_s)

                if vary_settings:
                    # Alternate between two presets — small enough churn
                    # not to dominate the cycle, large enough that GenICam
                    # node-map writes are real.
                    if i % 2 == 0:
                        self.set_gain(1.0)
                        self.set_exposure_time(10.0)
                    else:
                        self.set_gain(4.0)
                        self.set_exposure_time(50.0)

                t1 = time.monotonic()
                self.camera.start_grabbing()
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
                self.set_gain(float(original_gain))
            if vary_settings and original_exposure is not None:
                self.set_exposure_time(float(original_exposure))
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

        if not self.camera or not self.camera.active:
            return {'connected': False, 'errors': ['Camera not active']}

        if not hasattr(self.camera, 'read_diagnostic_snapshot'):
            return {
                'connected': False,
                'supported': False,
                'errors': [
                    f'{type(self.camera).__name__} does not implement '
                    f'read_diagnostic_snapshot'
                ],
            }

        # Driver-level snapshot
        snapshot = self.camera.read_diagnostic_snapshot(
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
            return self.led
        if target in ('motor', 'motion'):
            return self.motion
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
        # Minimal init — just enough for board communication
        instance._simulated = False
        instance._objectives_loader = objectives_loader.ObjectiveLoader()
        instance._coordinate_transformer = coord_transformations.CoordinateTransformer()

        # Threading infrastructure (locks first; per-axis dicts after motion init)
        instance._pos_cache_lock = threading.Lock()
        # Threading audit §10.2 — matches the __init__ path wrapping.
        instance._axis_state_lock = profile_trace.TimedLock(threading.Lock(), name="lumascope._axis_state_lock.diag")
        instance._move_profile_lock = threading.Lock()
        instance._motion_wake = threading.Event()
        instance._motion_monitor_stop = threading.Event()

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
        instance._homing_event = threading.Event()
        instance._turreting_event = threading.Event()
        instance._objective = None
        instance._objective_id = None

        # Connect boards — motion before per-axis dicts so we can size them
        # to the axes the hardware actually has (audit B4).
        #
        # #632/#539 surfaced the original silent-swallow bug. The helpers
        # are now at module scope (see top of file) so __init__,
        # create_diagnostic, and future callers share one code path.
        from drivers.null_ledboard import NullLEDBoard
        from drivers.null_motorboard import NullMotionBoard
        instance.led = _try_connect_board('LED board', LEDBoard, NullLEDBoard)
        instance.motion = _try_connect_board('Motor board', MotorBoard, NullMotionBoard)

        # Per-axis state dicts sized to detect_present_axes() (audit B4).
        present_axes = instance.motion.detect_present_axes()
        instance._pos_cache = {ax: 0.0 for ax in present_axes}
        instance._axis_state = {ax: AxisState.UNKNOWN for ax in present_axes}
        instance._arrival_events = {ax: threading.Event() for ax in present_axes}
        for ev in instance._arrival_events.values():
            ev.set()
        instance._move_profile = {ax: None for ax in present_axes}

        instance.camera = None
        instance._frame_buffer = None

        # Build capabilities (audit B7) — diagnostic instances still need
        # this so any code that reads `scope.capabilities.*` works.
        instance.capabilities = ScopeCapabilities.from_drivers(
            motion=instance.motion,
            led=instance.led,
            camera=None,
            led_max_ma=cls.LED_MAX_MA,
        )

        instance._motion_monitor_thread = threading.Thread(
            target=instance._motion_monitor_loop,
            name='motion-monitor', daemon=True,
        )
        instance._motion_monitor_thread.start()

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
        if not self.camera or not self.camera.active:
            return None
        try:
            profile = self.camera.profile
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
                'max_exposure_ms': self.camera_max_exposure,
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
        """Capture an image with illumination, asynchronously. DEPRECATED.

        Schedules a deferred grab; the captured image lands in
        ``self.capture_return`` and ``is_capturing`` is True until the
        deferred completion fires.

        Deprecated: use ``capture_and_wait`` for synchronous capture, or
        run ``capture_and_wait`` in a worker thread for async semantics.
        Will be removed in a future release.
        """
        warnings.warn(
            "Lumascope.capture is deprecated. Use capture_and_wait() instead "
            "(or run it in a worker thread for async semantics).",
            DeprecationWarning, stacklevel=2,
        )

        if not self.led: return
        if not self.camera or not self.camera.active: return

        self.is_capturing = True
        self.capture_return = False

        # Async grab via timer thread; capture_and_wait inside the timer
        # handles the drain. delay=0 because validity drains adaptively
        # rather than waiting a fixed exposure-derived interval.
        capture_timer = threading.Timer(0, self.capture_complete)
        capture_timer.start()

    def capture_complete(self) -> None:
        """Deferred completion handler for ``capture``. DEPRECATED.

        Grabs the image into ``self.capture_return`` and clears
        ``is_capturing``. Called from a background timer thread; not
        intended to be called directly.
        """
        self.capture_return = self.capture_and_wait()
        self.is_capturing = False


    def capture_blocking(self) -> 'np.ndarray | bool | None':
        """Capture an image with illumination, blocking until the frame is ready. DEPRECATED.

        Deprecated: use ``capture_and_wait`` directly. Will be removed in
        a future release.

        Returns:
            numpy.ndarray | False | None: Captured image array, False on
                grab failure, or None if LED/camera are unavailable.
        """
        warnings.warn(
            "Lumascope.capture_blocking is deprecated. Use capture_and_wait() instead.",
            DeprecationWarning, stacklevel=2,
        )
        if not self.led: return
        if not self.camera or not self.camera.active: return

        return self.capture_and_wait()

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
        if self.camera is None:
            return None
        handler = getattr(self.camera, 'cam_image_handler', None)
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

    def capture_and_wait(self, force_to_8bit: bool = True, *,
                         exclude_sources: tuple = (),
                         all_ones_check: bool = False,
                         earliest_image_ts: datetime.datetime | None = None,
                         timeout: datetime.timedelta = datetime.timedelta(seconds=0),
                         sum_count: int = 1, sum_delay_s: float = 0,
                         sum_iteration_callback=None) -> 'np.ndarray | bool':
        """Capture a frame guaranteed to reflect the current hardware state.

        Uses frame-based settling: drains stale frames from the camera pipeline
        until frame_validity confirms all pending state changes (LED, gain,
        exposure, motion) have settled. Then grabs a fresh valid frame.

        Frame-based settling automatically adapts to the camera's frame rate --
        fast exposures drain quickly, slow exposures drain slowly, matching
        the actual camera pipeline depth.

        Args:
            force_to_8bit: Convert to 8-bit output.
            exclude_sources: Sources to ignore for validity (e.g. ('z_move',)
                for autofocus where Z motion doesn't need to fully settle).
            all_ones_check: Reject all-max-value frames (camera hardware issue).
            earliest_image_ts: Reject frames captured before this timestamp.
                Forwarded to the final get_image call; complements the
                frame-validity drain for callers that also want a wall-clock
                lower bound on the returned frame.
            timeout: Timeout for the final get_image call.
            sum_count: Number of frames to sum for noise reduction.
            sum_delay_s: Delay between summed frames.
            sum_iteration_callback: Called after each summed frame.

        Returns:
            numpy.ndarray | False: Captured image array on success, False
                on camera-inactive or frame-drain failure.
        """
        if not self.camera or not self.camera.active:
            return False

        exposure_s = self.get_exposure_time() / 1000
        grab_timeout = max(exposure_s * 3, 1.0)

        # Drain stale frames until all pending state changes have settled.
        # Per-frame chunk metadata flows into count_frame so chunks short-
        # circuit skip-frames for chunk-validatable sources (gain, exposure).
        # Cameras without chunks return None and fall back to the existing
        # skip-frames + settle-check path.
        while self.frame_validity.frames_until_valid(exclude_sources=exclude_sources) > 0:
            status, _ = self.camera.grab_new_capture(timeout=grab_timeout)
            if status:
                self.frame_validity.count_frame(chunk_data=self._get_latest_chunks())
            else:
                logger.warning('[SCOPE API ] capture_and_wait: frame drain failed')
                return False

        return self.get_image(
            force_to_8bit=force_to_8bit,
            earliest_image_ts=earliest_image_ts,
            all_ones_check=all_ones_check,
            timeout=timeout,
            sum_count=sum_count,
            sum_delay_s=sum_delay_s,
            sum_iteration_callback=sum_iteration_callback,
            force_new_capture=True,
            new_capture_timeout=grab_timeout,
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # AUTOFOCUS Functionality
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Legacy autofocus methods (autofocus, autofocus_iterate, focus_best) removed
    # 2026-03-31 — superseded by AutofocusExecutor. No callers remained.

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

        # Same None-gate as save_image; retires with this static path in
        # Wave 7 Phase 5 (modules/image_save.py extraction).
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
