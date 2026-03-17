#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import contextlib
import datetime
import os
import pathlib
import threading
import time

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
from drivers.camera import Camera
from drivers.simulated_camera import SimulatedCamera
from drivers.simulated_motorboard import SimulatedMotorBoard
from drivers.simulated_ledboard import SimulatedLEDBoard

# Import additional libraries
from lvp_logger import logger, version
import modules.autofocus_functions as autofocus_functions
import modules.common_utils as common_utils
import modules.coord_transformations as coord_transformations
import modules.objectives_loader as objectives_loader
import modules.image_utils as image_utils
from modules.sequential_io_executor import SequentialIOExecutor, IOTask
from modules.frame_validity import FrameValidity


class AxisState:
    """Possible states for a motion axis."""
    UNKNOWN = 'unknown'    # Not homed / state not known
    IDLE = 'idle'          # At known position, not moving
    MOVING = 'moving'      # Move commanded, not yet arrived
    HOMING = 'homing'      # Homing sequence in progress


class Lumascope():

    # --- Input validation constants ---
    LED_MAX_MA = 1000       # Maximum LED current in milliamps (matches firmware CH_MAX)
    LED_VALID_CHANNELS = range(6)  # Channels 0-5 (Blue, Green, Red, BF, PC, DF)
    VALID_AXES = ('X', 'Y', 'Z', 'T')
    # Absolute position bounds (um) — generous outer limits; per-axis travel
    # limits are enforced by the motor board itself.
    MOTOR_POSITION_LIMIT = 1_000_000  # 1 meter in um

    def __init__(self, simulate: bool = False, camera_type: str = "pylon"):
        """Initialize Microscope.

        Args:
            simulate: If True, use simulated hardware (no USB devices needed).
            camera_type: Camera backend to use ("pylon" or "ids").
        """
        self._simulated = simulate
        self._coordinate_transformer = coord_transformations.CoordinateTransformer()
        self._objectives_loader = objectives_loader.ObjectiveLoader()

        # Position cache — push-based, not polled.
        # Updated after every move command and after homing.
        # The 10 Hz UI polling loops read from cache with zero serial I/O.
        self._pos_cache_lock = threading.Lock()
        self._pos_cache = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'T': 0.0}

        # Axis state — push-based, tracks UNKNOWN/IDLE/MOVING/HOMING per axis.
        # Updated by move/home methods. Consumers read state instead of polling
        # firmware via serial, eliminating 4+ serial round-trips per is_moving() call.
        self._axis_state_lock = threading.Lock()
        self._axis_state = {'X': AxisState.UNKNOWN, 'Y': AxisState.UNKNOWN,
                            'Z': AxisState.UNKNOWN, 'T': AxisState.UNKNOWN}

        # Per-axis arrival events — set when axis transitions from MOVING to IDLE.
        # Waiters call event.wait() instead of polling serial.
        self._arrival_events = {ax: threading.Event() for ax in self.VALID_AXES}
        for ev in self._arrival_events.values():
            ev.set()  # Start as "arrived" (not moving)

        # Motion monitor wakeup — set when any axis starts MOVING, cleared when
        # all axes are back to IDLE. The monitor thread sleeps on this.
        self._motion_wake = threading.Event()

        # Start the motion monitor thread (detects axis arrival via firmware query)
        self._motion_monitor_stop = threading.Event()
        self._motion_monitor_thread = threading.Thread(
            target=self._motion_monitor_loop,
            name='motion-monitor',
            daemon=True,
        )
        self._motion_monitor_thread.start()

        # LED Control Board
        try:
            if simulate:
                self.led = SimulatedLEDBoard()
                logger.info('[SCOPE API ] Using SIMULATED LED Board')
            else:
                self.led = LEDBoard()
        except Exception:
            self.led = None
            logger.exception('[SCOPE API ] LED Board Not Initialized')

        # Motion Control Board
        try:
            if simulate:
                from modules.settings_init import settings
                sim_model = settings.get('microscope', 'LS850') if settings else 'LS850'
                self.motion = SimulatedMotorBoard(model=sim_model)
                logger.info(f'[SCOPE API ] Using SIMULATED Motor Board (model={sim_model})')
            else:
                self.motion = MotorBoard()
        except Exception:
            self.motion = None
            logger.exception('[SCOPE API ] Motion Board Not Initialized')

        # Camera
        self._image_buffer = None  # backing field for image_buffer property
        self._frame_buffer = None  # Pre-allocated buffer for get_image_from_buffer
        try:
            if simulate:
                self.camera: Camera = SimulatedCamera(
                    z_position_func=lambda: self.motion.current_pos('Z') if self.motion else 5000.0,
                )
                self.camera.load_cycle_images()
                logger.info('[SCOPE API ] Using SIMULATED Camera')
            elif camera_type == "ids":
                self.camera: Camera = IDSCamera()
            else:
                self.camera: Camera = PylonCamera()
        except Exception:
            logger.exception('[SCOPE API ] Camera Board Not Initialized')

        # Track whether any real hardware was found
        self._no_hardware = (
            not simulate
            and self.led is None
            and self.motion is None
            and not hasattr(self, 'camera')
        )
        if self._no_hardware:
            logger.warning('[SCOPE API ] No hardware detected (LED, motor, and camera all failed to initialize)')

        # --- Thread synchronization (CR-2 / CR-6) ---
        # _state_lock protects individual shared-state reads/writes
        self._state_lock = threading.Lock()
        # _hw_lock is an RLock for multi-step hardware operations
        self._hw_lock = threading.RLock()

        # Boolean operation flags use threading.Event for wait/signal
        self._homing_event = threading.Event()       # set => homing in progress
        self._capturing_event = threading.Event()    # set => capture in progress
        self._focusing_event = threading.Event()     # set => autofocus in progress

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
        self.frame_validity = FrameValidity()
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
            'pixel_format': None,
            'binning': 1,
        }
        self._populate_camera_cache()


    # --- Motion monitor (Phase 1A) ---

    _MOTION_POLL_INTERVAL = 0.02  # 50 Hz

    def _motion_monitor_loop(self):
        """Background thread: polls firmware for axis arrival at 50 Hz.

        Sleeps on ``_motion_wake`` when all axes are IDLE. Wakes when any
        axis transitions to MOVING. Polls ``get_target_status()`` per
        MOVING axis and transitions them to IDLE on arrival. This is the
        single place where firmware target-status queries happen during
        normal operation — all other code reads the in-memory axis state.
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
                    if self.motion and hasattr(self.motion, 'overshoot') and self.motion.overshoot:
                        time.sleep(self._MOTION_POLL_INTERVAL)
                        continue
                    # All axes arrived — go back to sleep
                    self._motion_wake.clear()
                    break

                # Query firmware for each MOVING axis
                for ax in moving_axes:
                    if self._motion_monitor_stop.is_set():
                        break
                    try:
                        if self.motion and self.motion.driver and self.get_target_status(ax):
                            # Axis has arrived — transition to IDLE
                            self._set_axis_state(ax, AxisState.IDLE)
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
                'max_exposure': self.camera.get_max_exposure() or 0.0,
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
        """Whether the camera is connected and active (reads cache)."""
        with self._camera_cache_lock:
            return self._camera_cache['active']

    @property
    def camera_gain(self) -> float:
        """Current camera gain in dB (reads cache)."""
        with self._camera_cache_lock:
            return self._camera_cache['gain']

    @property
    def camera_exposure_ms(self) -> float:
        """Current camera exposure time in ms (reads cache)."""
        with self._camera_cache_lock:
            return self._camera_cache['exposure_ms']

    @property
    def camera_frame_size(self) -> dict:
        """Current camera frame size as {'width': int, 'height': int} (reads cache)."""
        with self._camera_cache_lock:
            return dict(self._camera_cache['frame_size'])

    @property
    def camera_max_frame_size(self) -> dict:
        """Maximum camera frame size (reads cache)."""
        with self._camera_cache_lock:
            return dict(self._camera_cache['max_frame_size'])

    @property
    def camera_min_frame_size(self) -> dict:
        """Minimum camera frame size (reads cache)."""
        with self._camera_cache_lock:
            return dict(self._camera_cache['min_frame_size'])

    @property
    def camera_max_exposure(self) -> float:
        """Maximum camera exposure time in ms (reads cache)."""
        with self._camera_cache_lock:
            return self._camera_cache['max_exposure']

    # --- CR-2: Thread-safe properties for shared state ---

    @property
    def is_homing(self) -> bool:
        """True while the microscope is homing."""
        return self._homing_event.is_set()

    @is_homing.setter
    def is_homing(self, value: bool):
        if value:
            self._homing_event.set()
        else:
            self._homing_event.clear()

    @property
    def is_capturing(self) -> bool:
        """True while the microscope is capturing an image."""
        return self._capturing_event.is_set()

    @is_capturing.setter
    def is_capturing(self, value: bool):
        if value:
            self._capturing_event.set()
        else:
            self._capturing_event.clear()

    @property
    def is_focusing(self) -> bool:
        """True while the microscope is running autofocus."""
        return self._focusing_event.is_set()

    @is_focusing.setter
    def is_focusing(self, value: bool):
        if value:
            self._focusing_event.set()
        else:
            self._focusing_event.clear()

    @property
    def image_buffer(self):
        with self._state_lock:
            return self._image_buffer

    @image_buffer.setter
    def image_buffer(self, value):
        with self._state_lock:
            self._image_buffer = value

    @property
    def capture_return(self):
        with self._state_lock:
            return self._capture_return

    @capture_return.setter
    def capture_return(self, value):
        with self._state_lock:
            self._capture_return = value

    @property
    def autofocus_return(self):
        with self._state_lock:
            return self._autofocus_return

    @autofocus_return.setter
    def autofocus_return(self, value):
        with self._state_lock:
            self._autofocus_return = value

    @property
    def scale_bar_config(self):
        """Return a snapshot of scale bar settings."""
        with self._state_lock:
            return dict(self._scale_bar)

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

    def _set_axis_state(self, axis: str, state: str):
        """Set the state of an axis (internal use only).

        When transitioning to MOVING/HOMING, clears the axis arrival event
        and wakes the motion monitor. When transitioning to IDLE, sets the
        arrival event so waiters unblock.
        """
        with self._axis_state_lock:
            self._axis_state[axis] = state

        if state in (AxisState.MOVING, AxisState.HOMING):
            # Clear arrival event — axis is now in motion
            self._arrival_events[axis].clear()
            # Wake the motion monitor to start polling
            self._motion_wake.set()
        elif state == AxisState.IDLE:
            # Signal arrival — unblocks any wait_for_axis() callers
            self._arrival_events[axis].set()

    def is_any_axis_moving(self) -> bool:
        """Check if any axis is currently MOVING or HOMING.

        Reads from the in-memory state dict — zero serial I/O.

        Returns:
            bool: True if any axis is in MOVING or HOMING state.
        """
        with self._axis_state_lock:
            return any(
                s in (AxisState.MOVING, AxisState.HOMING)
                for s in self._axis_state.values()
            )

    def axes_present(self) -> list[str]:
        """Get list of axis names known to this scope.

        Built dynamically from the axis state dict (which is populated
        from VALID_AXES at init, and could be extended for external axes).

        Returns:
            list[str]: e.g. ['X', 'Y', 'Z', 'T']
        """
        with self._axis_state_lock:
            return list(self._axis_state.keys())

    def has_axis(self, axis: str) -> bool:
        """Check if a given axis is present.

        Args:
            axis: Axis name.

        Returns:
            bool: True if axis exists in the state model.
        """
        with self._axis_state_lock:
            return axis in self._axis_state

    def travel_limit_um(self, axis: str) -> float:
        """Get the travel limit for an axis in um.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            float: Travel limit in um, or MOTOR_POSITION_LIMIT if unknown.
        """
        if not self.motion or not self.motion.driver:
            return float(self.MOTOR_POSITION_LIMIT)
        try:
            return float(self.motion.motorconfig.travel_limit_um(axis))
        except Exception:
            return float(self.MOTOR_POSITION_LIMIT)

    @property
    def motor_connected(self) -> bool:
        """Whether the motor controller is connected (replaces scope.motion.driver checks)."""
        return bool(self.motion and self.motion.driver)

    def lens_focal_length(self) -> float:
        """Get tube lens focal length from motorconfig.

        Returns:
            float: Focal length in mm (default 47.8).
        """
        if not self.motion or not self.motion.driver:
            return 47.8
        return self.motion.motorconfig.lens_focal_length()

    def pixel_size(self) -> float:
        """Get camera pixel size from motorconfig.

        Returns:
            float: Pixel size in um/pixel (default 2.0).
        """
        if not self.motion or not self.motion.driver:
            return 2.0
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
                image = scope.capture_blocking()
        """
        self._hw_lock.acquire()
        try:
            yield
        finally:
            self._hw_lock.release()

    def disconnect(self):
        """Disconnect from all hardware (LED, motion, camera)."""
        logger.info('[SCOPE API ] Disconnecting from microscope...')

        # Stop the motion monitor before disconnecting the motor board
        self._stop_motion_monitor()

        # Set all axes to UNKNOWN before disconnecting
        with self._axis_state_lock:
            for ax in self._axis_state:
                self._axis_state[ax] = AxisState.UNKNOWN
        # Set all arrival events so any blocked waiters unblock
        for ev in self._arrival_events.values():
            ev.set()

        if self.led is not None:
            self.led.disconnect()
            self.led = None

        if self.motion is not None:
            self.motion.disconnect()
            self.motion = None

        if self.camera is not None:
            self.camera.disconnect()
            self.camera = None
        self._invalidate_camera_cache()

        logger.info('[SCOPE API ] Microscope disconnected')

    @property
    def no_hardware(self):
        """True if no real hardware was detected (LED, motor, and camera all missing)."""
        return self._no_hardware

    # def reconnect(self):
    #     logger.info('[SCOPE API ] Attempting to reconnect to microscope...')
    #     self.disconnect()
    #     self.__init__()
    #     logger.info('[SCOPE API ] Microscope reconnected')

    def are_all_connected(self) -> bool:
        """Check if LED, motion, and camera boards are all connected.

        Returns:
            bool: True if all three components are connected.
        """
        logger.info('[SCOPE API ] Performing connection check...')
        led = self.led is not None and self.led.is_connected()
        motion = self.motion is not None and self.motion.is_connected()
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

    # def reconnect_if_disconnected(self):
    #     if not self.are_all_connected():
    #         if not self.led.is_connected():
    #             self.led.connect()
    #         if not self.motion.is_connected():
    #             self.motion.connect()
    #         if not self.camera.is_connected():
    #             self.camera.connect()
    #         self.reconnect()


    ########################################################################
    # SCOPE CONFIGURATION FUNCTIONS
    ########################################################################
    def set_labware(self, labware):
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

    def set_objective(self, objective_id: str):
        """Set the active objective by ID.

        Args:
            objective_id: Objective identifier (e.g. "4x", "10x", "20x").
        """
        self._objective = self._objectives_loader.get_objective_info(objective_id=objective_id)

    def get_objective_info(self, objective_id: str):
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

    def set_turret_config(self, turret_config: dict[int,str]) -> None:
        """Set the turret objective configuration.

        Args:
            turret_config: Mapping of turret position (1-4) to objective ID.
        """
        self._turret_config = turret_config

    def get_turret_config(self):
        """Get the current turret objective configuration.

        Returns:
            dict: Mapping of turret position to objective ID.
        """
        return self._turret_config

    def get_turret_position_for_objective_id(self, objective_id: str, prefer_current: bool = True) -> int | None:
        """Find the turret position holding a given objective.

        When multiple positions hold the same objective, prefers the current
        turret position to avoid unnecessary moves (#488).

        Args:
            objective_id: Objective identifier to search for.
            prefer_current: If True (default), return the current turret
                position when it already holds the requested objective.

        Returns:
            int | None: Turret position (1-4), or None if not found.
        """
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
        position = self.get_current_position(axis='T')
        if self._turret_config[position] is None:
            return False

        return True

    def set_scale_bar(self, enabled: bool, color: str = None):
        """Configure the scale bar overlay on captured images.

        Args:
            enabled: Whether to draw the scale bar.
            color: Scale bar color (e.g. "white"). Uses default if None.
        """
        self._scale_bar['enabled'] = enabled
        if color is not None:
            self._scale_bar['color'] = color

    def set_stage_offset(self, stage_offset):
        """Set the stage offset for coordinate transformations.

        Args:
            stage_offset: Stage offset dict with axis offsets.
        """
        self._stage_offset = stage_offset

    def get_available_binning_sizes(self):
        """Return list of binning sizes supported by connected camera."""
        if not self.camera or not self.camera.active:
            return [1]
        try:
            return self.camera.profile.binning_sizes
        except (AttributeError, TypeError):
            return [1]

    def set_binning_size(self, size):
        """Set camera pixel binning size.

        Args:
            size (int): Binning factor (1 = no binning, 2 = 2x2, etc.).
        """
        try:
            self._binning_size = size

            if self.camera:
                self.camera.set_binning_size(size=size)
        except Exception as ex:
            logger.exception(f"[SCOPE API ] Error setting binning size: {ex}")

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
            bool: True if successful, False otherwise.
        """
        if not self.camera or not self.camera.active:
            return False
        result = self.camera.set_pixel_format(pixel_format)
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


    ########################################################################
    # LED BOARD FUNCTIONS
    ########################################################################

    def leds_enable(self):
        """Enable all LED channels (allows them to be turned on)."""
        if not self.led: return
        self.led.leds_enable()

    def leds_disable(self):
        """Disable all LED channels (prevents them from turning on)."""
        if not self.led: return
        self.led.leds_disable()

    def get_led_ma(self, color: str):
        """Get the current illumination level for an LED channel.

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            float: Illumination in milliamps, or -1 if LED board unavailable.
        """
        if not self.led: return -1

        return self.led.get_led_ma(color=color)


    def get_led_state(self, color: str):
        """Get the on/off state and illumination for an LED channel.

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            dict: LED state and illumination (mA), or -1 if unavailable.
        """
        if not self.led: return -1

        return self.led.get_led_state(color=color)


    def get_led_states(self):
        """Get state and illumination for all LED channels.

        Returns:
            dict: Mapping of color to state/illumination dict, or -1 if unavailable.
        """
        if not self.led: return -1

        return self.led.get_led_states()


    def led_on(self, channel, mA, block=False):
        """Turn on an LED channel at the specified current.

        Args:
            channel: Channel number (0-5) or color name string.
            mA: Illumination current in milliamps.
            block: If True, wait for confirmation from the LED board.

        Raises:
            ValueError: If channel or mA is out of range.
        """
        if not self.led: return

        if isinstance(channel, str):
            channel = self.color2ch(color=channel)

        if channel not in self.LED_VALID_CHANNELS:
            raise ValueError(f"LED channel must be 0-5, got {channel}")
        if not isinstance(mA, (int, float)) or mA < 0 or mA > self.LED_MAX_MA:
            raise ValueError(f"LED current must be 0-{self.LED_MAX_MA} mA, got {mA}")

        self.led.led_on(channel, mA, block=block)
        self.frame_validity.invalidate('led')

    def led_off(self, channel):
        """Turn off an LED channel.

        Args:
            channel: Channel number (0-5) or color name string.

        Raises:
            ValueError: If channel is out of range.
        """
        if not self.led: return

        if isinstance(channel, str):
            channel = self.color2ch(color=channel)

        if channel not in self.LED_VALID_CHANNELS:
            raise ValueError(f"LED channel must be 0-5, got {channel}")

        self.led.led_off(channel)
        self.frame_validity.invalidate('led')

    def led_on_fast(self, channel, mA):
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
        if channel not in self.LED_VALID_CHANNELS:
            raise ValueError(f"LED channel must be 0-5, got {channel}")
        if not isinstance(mA, (int, float)) or mA < 0 or mA > self.LED_MAX_MA:
            raise ValueError(f"LED current must be 0-{self.LED_MAX_MA} mA, got {mA}")
        self.led.led_on_fast(channel, mA)
        self.frame_validity.invalidate('led')

    def led_off_fast(self, channel):
        """Turn off an LED with write-only (no read-back) for time-critical pulses.

        Args:
            channel: Channel number (0-5) or color name string.

        Raises:
            ValueError: If channel is out of range.
        """
        if not self.led: return
        if isinstance(channel, str):
            channel = self.color2ch(color=channel)
        if channel not in self.LED_VALID_CHANNELS:
            raise ValueError(f"LED channel must be 0-5, got {channel}")
        self.led.led_off_fast(channel)
        self.frame_validity.invalidate('led')

    def leds_off_fast(self):
        """Turn off all LEDs with write-only (no read-back) for time-critical pulses."""
        if not self.led: return
        self.led.leds_off_fast()
        self.frame_validity.invalidate('led')

    def leds_off(self):
        """Turn off all LEDs."""
        if not self.led: return
        self.led.leds_off()
        self.frame_validity.invalidate('led')

    def get_led_status(self):
        """Get the LED board status register."""
        if not self.led: return
        return self.led.get_status()

    def wait_until_led_on(self):
        """Block until the LED board confirms an LED is on."""
        if not self.led: return
        self.led.wait_until_on()

    def ch2color(self, channel):
        """Convert a channel number to its color name string.

        Args:
            channel (int): Channel number (0=Blue, 1=Green, 2=Red, 3=BF, 4=PC, 5=DF).

        Returns:
            str: Color name (e.g. "Blue", "BF"), or None if LED board unavailable.
        """
        if not self.led: return
        return self.led.ch2color(channel)

    def color2ch(self, color):
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
        timeout: datetime.timedelta = datetime.timedelta(seconds=0),
        all_ones_check: bool = False,
        sum_count: int = 1,
        sum_delay_s: float = 0,
        sum_iteration_callback = None,
        force_new_capture = False,
        new_capture_timeout = 1000,
    ):
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
                all_ones_failed = False
                if force_new_capture:
                    grab_status, grab_image_ts = self.camera.grab_new_capture(new_capture_timeout)
                else:
                    grab_status, grab_image_ts = self.camera.grab()

                if grab_status:
                    self.frame_validity.count_frame()
                    tmp = self.camera.get_array()  # thread-safe copy

                    if all_ones_check:

                        if np.all(tmp == np.iinfo(tmp.dtype).max):
                            all_ones_failed = True
                            logger.warning(f"[SCOPE API ] get_image all_ones_check failed")

                    if not all_ones_failed:
                        if earliest_image_ts is None:
                            tmp_buffer.append(tmp)
                            break

                        if grab_image_ts > earliest_image_ts:
                            tmp_buffer.append(tmp)
                            break

                        logger.warning(f"[SCOPE API ] get_image earliest_image_time {earliest_image_ts} not met -> Image TS: {grab_image_ts}")


                # In case of timeout, if we hit the timeout because of the all ones check, then just let it continue and return the all ones image
                if datetime.datetime.now() > stop_time:
                    if not all_ones_failed:
                        logger.error(f"[SCOPE API ] get_image timeout stop_time ({stop_time}) exceeded")
                        return False
                    else:
                        logger.warning(f"[SCOPE API ] get_image timeout stop_time ({stop_time}) exceeded with all_ones_failed")
                        break

                if not grab_status:
                    logger.error(f"[SCOPE API ] get_image grab failed, retrying")

                time.sleep(0.05)

            if sum_count > 1:
                earliest_image_ts = grab_image_ts + datetime.timedelta(milliseconds=1)
                if sum_iteration_callback is not None:
                    sum_iteration_callback()

                time.sleep(sum_delay_s)

        # Add the images together
        if sum_count == 1:
            if len(tmp_buffer) < 1:
                self.image_buffer = tmp
            else:
                self.image_buffer = tmp_buffer[0]
        else:
            orig_dtype = tmp_buffer[0].dtype
            max_value = np.iinfo(orig_dtype).max

            combined = np.zeros_like(tmp_buffer[0], dtype=np.uint32)
            for img in tmp_buffer:
                combined += img

            self.image_buffer = np.clip(combined, None, max_value).astype(orig_dtype)

        use_scale_bar = self._scale_bar['enabled']
        if self._objective is None:
            use_scale_bar = False

        if use_scale_bar:
            self.image_buffer = image_utils.add_scale_bar(
                image=self.image_buffer,
                objective=self._objective,
                binning_size=self._binning_size,
                color=self._scale_bar.get('color'),
            )

        if force_to_8bit and self.image_buffer.dtype != np.uint8:
            self.image_buffer = image_utils.convert_12bit_to_8bit(self.image_buffer)

        return self.image_buffer

    def get_image_from_buffer(
        self,
        force_to_8bit: bool = True
        ):
        """Grab the latest buffered frame from the camera without forcing a new capture.

        Args:
            force_to_8bit: Convert 12-bit images to 8-bit output.

        Returns:
            numpy.ndarray | False: Image array, or False if camera inactive or grab failed.
        """
        if not self.camera or not self.camera.active:
            return False

        # Single-copy grab: grab_latest() returns the image directly,
        # avoiding the extra copy that grab() + get_array() would make.
        # This saves ~2.3MB copy + 1 lock acquisition per frame.
        grab_status, tmp, grab_image_ts = self.camera.grab_latest()
        if not grab_status or tmp is None:
            return False

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

        return tmp

    def get_next_save_path(self, path):
        """ GETS THE NEXT SAVE PATH GIVEN AN EXISTING SAVE PATH

            :param path of the format './{save_folder}/{well_label}_{color}_{file_id}.tiff'
            :returns the next save path './{save_folder}/{well_label}_{color}_{file_id + 1}.tiff'

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


    def generate_image_save_path(self, save_folder, file_root, append, tail_id_mode, output_format):
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

        elif tail_id_mode is None:
            filename =  f"{file_root}{append}{file_extension}"
            path = save_folder / filename

        else:
            raise ConfigError(f"tail_id_mode: {tail_id_mode} not implemented")

        return path

    def get_well_label(self):
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

        metadata = {
            'camera_make': 'Etaluma',
            'microscope': self.get_microscope_model(),
            'software': f'LumaViewPro {version}',
            'channel': color,
            'datetime': datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),      # Format for metadata
            'sub_sec_time': f"{datetime.datetime.now().microsecond // 1000:03d}",
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
        }

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
        z
    ):
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
            array = image_utils.convert_12bit_to_16bit(array)

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
        z=None
    ):
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

        if (common_utils.check_disk_space() < 1024):  # Check for at least 1 GB of free space
            logger.error(f"[SCOPE API ] Disk space < 1 GB. Image unlikely to save correctly.")

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
            z=z
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
            )

            logger.info(f'[SCOPE API ] Saving Image to {file_loc}')
        except Exception:
            logger.exception("[SCOPE API ] Error: Unable to save. Perhaps save folder does not exist?")

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
            timeout: datetime.timedelta = datetime.timedelta(seconds=0),
            all_ones_check: bool = False,
            sum_count: int = 1,
            sum_delay_s: float = 0,
            sum_iteration_callback = None,
            turn_off_all_leds_after: bool = False,
            use_executor = False
        ):

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

        if (common_utils.check_disk_space() < 1024):  # Check for at least 1 GB of free space
            logger.error(f"[SCOPE API ] Disk space < 1 GB. Image unlikely to save correctly.")
            
        array = self.get_image(
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


    def get_max_width(self):
        """Get the maximum pixel width of the camera sensor.

        Returns:
            int: Max width in pixels, or 0 if camera inactive.
        """
        if (not self.camera) or (not self.camera.active): return 0
        return self.camera.get_max_frame_size()['width']

    def get_max_height(self):
        """Get the maximum pixel height of the camera sensor.

        Returns:
            int: Max height in pixels, or 0 if camera inactive.
        """
        if (not self.camera) or (not self.camera.active): return 0
        return self.camera.get_max_frame_size()['height']

    def get_width(self):
        """Get the current frame width setting.

        Returns:
            int: Current width in pixels, or 0 if camera unavailable.
        """
        if not self.camera: return 0
        return self.camera.get_frame_size()['width']

    def get_height(self):
        """Get the current frame height setting.

        Returns:
            int: Current height in pixels, or 0 if camera unavailable.
        """
        if not self.camera: return 0
        return self.camera.get_frame_size()['height']

    def set_frame_size(self, w, h):
        """Set the camera frame size in pixels.

        Args:
            w (int): Frame width in pixels.
            h (int): Frame height in pixels.
        """

        if not self.camera or not self.camera.active: return
        self.camera.set_frame_size(w, h)
        with self._camera_cache_lock:
            self._camera_cache['frame_size'] = {'width': int(w), 'height': int(h)}

    def get_frame_size(self):
        """Get the current camera frame size.

        Returns:
            dict: Contains 'width' and 'height' in pixels, or None if inactive.
        """

        if not self.camera or not self.camera.active: return
        return self.camera.get_frame_size()


    def get_gain(self):
        """Get the current camera gain.

        Returns:
            float: Gain in dB, or -1 if camera inactive.
        """

        if not self.camera or not self.camera.active: return -1
        return self.camera.get_gain()

    def set_gain(self, gain):
        """Set the camera gain.

        Args:
            gain (float): Gain value in dB.
        """

        if not self.camera or not self.camera.active: return
        self.camera.gain(gain)
        self.frame_validity.invalidate('gain')
        with self._camera_cache_lock:
            self._camera_cache['gain'] = float(gain)

    def set_auto_gain(self, state: bool, settings: dict):
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

    def set_exposure_time(self, t):
        """Set the camera exposure time.

        Args:
            t (float): Exposure time in milliseconds.
        """

        if not self.camera or not self.camera.active: return
        self.camera.exposure_t(t)
        self.frame_validity.invalidate('exposure')
        with self._camera_cache_lock:
            self._camera_cache['exposure_ms'] = float(t)

    def get_exposure_time(self):
        """Get the current camera exposure time.

        Returns:
            float: Exposure time in milliseconds, or 0 if camera inactive.
        """

        if not self.camera or not self.camera.active: return 0
        exposure = self.camera.get_exposure_t()
        return exposure

    def set_auto_exposure_time(self, state = True):
        """Enable or disable automatic exposure adjustment.

        Args:
            state: True to enable auto exposure, False to disable.
        """

        if not self.camera or not self.camera.active: return
        self.camera.auto_exposure_t(state)
        self.frame_validity.invalidate('exposure')

    def update_auto_gain_target_brightness(self, target_brightness: float):
        """Set the auto-gain target brightness on the camera.

        Args:
            target_brightness: Target brightness value (0.0–1.0).
        """
        if not self.camera or not self.camera.active:
            return
        self.camera.update_auto_gain_target_brightness(target_brightness)

    def auto_gain_once(self, state: bool, target_brightness: float,
                       min_gain: float, max_gain: float):
        """Run auto-gain for a single frame on the camera.

        Args:
            state: True to enable one-shot auto-gain.
            target_brightness: Target brightness (0.0–1.0).
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

    ########################################################################
    # MOTION CONTROL FUNCTIONS
    ########################################################################
    @contextlib.contextmanager
    def reference_position_logger(self):
        before = self.get_limit_switch_status_all_axes()
        logger.info(f"Limit switch status before homing: {before}", extra={'force_error': True})
        yield
        after = self.get_limit_switch_status_all_axes()
        logger.info(f"Limit switch status after homing: {after}", extra={'force_error': True})

    def get_axes_config(self):
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


    def zhome(self):
        """Home the Z axis (focus)."""
        #if not self.motion: return
        self._set_axis_state('Z', AxisState.HOMING)
        with self.reference_position_logger():
            self.motion.zhome()
        self._set_axis_state('Z', AxisState.IDLE)
        self.refresh_position_cache()

    def xyhome(self):
        """Home the XY axes (stage). Z axis and turret always home first."""
        #if not self.motion: return
        for ax in ('X', 'Y', 'Z'):
            self._set_axis_state(ax, AxisState.HOMING)
        with self.reference_position_logger():
            self.is_homing = True
            self.motion.xyhome()
        for ax in ('X', 'Y', 'Z'):
            self._set_axis_state(ax, AxisState.IDLE)
        self.refresh_position_cache()

        return

        #while self.is_moving():
        #    time.sleep(0.01)
        #self.is_homing = False

    def has_xyhomed(self):
        """Check if the XY axes have been homed since startup.

        Returns:
            bool: True if XY homing has been performed.
        """
        return self.motion.has_xyhomed()

    def xyhome_iterate(self):
        if not self.is_moving():
            self.is_homing = False
            self.xyhome_timer.cancel()

    def xycenter(self):
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
        # Save off current Z position before moving Z to 0
        logger.info('[SCOPE API ] Moving Z to 0', extra={'force_error': True})
        initial_z = self.get_current_position(axis='Z')
        self.move_absolute_position('Z', pos=0, wait_until_complete=True)
        self.is_turreting = True
        yield
        self.is_turreting = False
        # Restore Z position
        logger.info(f'[SCOPE API ] Restoring Z to {initial_z}', extra={'force_error': True})
        self.move_absolute_position('Z', pos=initial_z, wait_until_complete=True)


    def thome(self):
        """Home the turret axis. Moves Z to 0 during turret motion for safety."""

        #if not self.motion:
        #    return

        # Move turret
        self._set_axis_state('T', AxisState.HOMING)
        with self.reference_position_logger():
            with self.safe_turret_mover():
                self.motion.thome()
        self._set_axis_state('T', AxisState.IDLE)
        self.refresh_position_cache()

    def has_thomed(self):
        """Check if the turret has been homed since startup.

        Returns:
            bool: True if turret homing has been performed.
        """
        return self.motion.has_thomed()

    def tmove(self, position):
        """Move the turret to a specific position. Skips if already there.

        Args:
            position (int): Target turret position (1-4).
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

        Returns:
            bool: True if a turret is present.
        """
        return self.motion.has_turret()


    def refresh_position_cache(self):
        """Fetch all axis positions from hardware and update the cache.

        Called after homing completes to sync the cache with actual hardware
        positions.  During normal operation the cache is updated directly
        by move commands — no polling needed.
        """
        if not self.motion or not self.motion.driver:
            return

        positions = {}
        for ax in ('X', 'Y', 'Z', 'T'):
            try:
                positions[ax] = self.motion.target_pos(axis=ax)
            except Exception:
                positions[ax] = 0.0

        with self._pos_cache_lock:
            self._pos_cache.update(positions)

    def get_target_position(self, axis=None):
        """Get the target position for an axis (where it is commanded to go).

        Reads from the push-based position cache — zero serial I/O.

        Args:
            axis: Axis name ("X", "Y", "Z", "T"), or None for all axes.

        Returns:
            float | dict: Position in um for a single axis, or dict of all
                axis positions. Returns 0 if motion board inactive, None if
                axis T requested but no turret present.
        """
        if not self.motion or not self.motion.driver:
            return 0

        if (not self.motion.has_turret()) and (axis == 'T'):
            return None

        with self._pos_cache_lock:
            if axis is None:
                return dict(self._pos_cache)
            return self._pos_cache.get(axis, 0.0)

    def get_current_position(self, axis=None):
        """Get the current position for an axis.

        Reads from the push-based position cache — zero serial I/O.
        For UI display purposes, this returns the last commanded position.
        For precise position (e.g. during autofocus), callers that need
        the actual hardware position should use motion.current_pos() directly.

        Args:
            axis: Axis name ("X", "Y", "Z", "T"), or None for all axes.

        Returns:
            float | dict: Position in um for a single axis, or dict of all
                axis positions. Returns 0 if motion board inactive.
        """
        if not self.motion or not self.motion.driver:
            return 0

        with self._pos_cache_lock:
            if axis is None:
                return dict(self._pos_cache)
            return self._pos_cache.get(axis, 0.0)


    def move_absolute_position(self, axis, pos, wait_until_complete=False, overshoot_enabled: bool = True, ignore_limits: bool = False):
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
        if axis not in self.VALID_AXES:
            raise ValueError(f"Axis must be one of {self.VALID_AXES}, got {axis!r}")
        if not isinstance(pos, (int, float)):
            raise ValueError(f"Position must be numeric, got {type(pos).__name__}")
        if abs(pos) > self.MOTOR_POSITION_LIMIT:
            raise ValueError(f"Position {pos} um exceeds safety limit of +/-{self.MOTOR_POSITION_LIMIT} um")

        #if not self.motion: return
        self._set_axis_state(axis, AxisState.MOVING)
        self.motion.move_abs_pos(axis, pos, overshoot_enabled=overshoot_enabled, ignore_limits=ignore_limits)
        with self._pos_cache_lock:
            self._pos_cache[axis] = float(pos)
        self.frame_validity.invalidate('z_move' if axis == 'Z' else 'xy_move')

        if wait_until_complete is True:
            self.wait_until_finished_moving()
            self._set_axis_state(axis, AxisState.IDLE)


    def move_relative_position(self, axis, um, wait_until_complete=False, overshoot_enabled: bool = False):
        """Move an axis by a relative distance.

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").
            um (float): Distance to move in um.
            wait_until_complete: If True, block until move finishes.
            overshoot_enabled: Allow Z overshoot for backlash compensation.

        Raises:
            ValueError: If axis is invalid or um is not numeric / out of bounds.
        """
        if axis not in self.VALID_AXES:
            raise ValueError(f"Axis must be one of {self.VALID_AXES}, got {axis!r}")
        if not isinstance(um, (int, float)):
            raise ValueError(f"Distance must be numeric, got {type(um).__name__}")
        if abs(um) > self.MOTOR_POSITION_LIMIT:
            raise ValueError(f"Distance {um} um exceeds safety limit of +/-{self.MOTOR_POSITION_LIMIT} um")

        #if not self.motion: return
        self._set_axis_state(axis, AxisState.MOVING)
        self.motion.move_rel_pos(axis, um, overshoot_enabled=overshoot_enabled)
        with self._pos_cache_lock:
            self._pos_cache[axis] = self._pos_cache.get(axis, 0.0) + float(um)
        self.frame_validity.invalidate('z_move' if axis == 'Z' else 'xy_move')

        if wait_until_complete is True:
            self.wait_until_finished_moving()
            self._set_axis_state(axis, AxisState.IDLE)


    def get_home_status(self, axis):
        """Check if an axis is at its home position.

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").

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

    def get_target_status(self, axis):
        """Check if an axis has reached its target position.

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").

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

    def get_target_pos(self, axis):
        """Get the target position for an axis (error-safe version).

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").

        Returns:
            float: Target position in um, or -1 on error/no turret.
        """
        if (axis == 'T') and (not self.motion.has_turret()):
            return -1

        try:
            return self.motion.target_pos(axis)
        except Exception as e:
            logger.exception(f"[SCOPE API ] get_target_pos({axis}) failed; returning -1: {e}")
            return -1

    def get_reference_status(self, axis):
        """Get reference status register bits for an axis.

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").

        Returns:
            str: 32-character binary string of register bits (MSB first).
        """

        #if not self.motion: return
        return self.motion.reference_status(axis=axis)


    def get_limit_switch_status(self, axis):
        """Get the limit switch status for an axis.

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").

        Returns:
            Limit switch state for the specified axis.
        """
        return self.motion.limit_switch_status(axis=axis)


    def get_limit_switch_status_all_axes(self):
        """Get limit switch status for all axes.

        Returns:
            dict: Mapping of axis name to limit switch state.
        """
        resp = {}
        for axis in ('X','Y','Z','T'):
            resp[axis] = self.get_limit_switch_status(axis=axis)
        return resp


    def get_overshoot(self):
        """Check if the Z axis is currently in overshoot (backlash compensation) mode.

        Returns:
            bool: True if overshoot is in progress.
        """

        #if not self.motion: return False
        return self.motion.overshoot

    def is_moving(self):
        """Check if any axis is currently moving.

        Reads from in-memory axis state — zero serial I/O. The motion
        monitor thread handles firmware queries and state transitions.

        Returns:
            bool: True if any axis is MOVING/HOMING or overshoot is active.
        """
        if not self.motion or not self.motion.driver:
            return False
        if self.is_any_axis_moving():
            return True
        if self.get_overshoot():
            return True
        return False

    def wait_until_finished_moving(self, timeout: float = 120.0):
        """Block until all axes have reached their target positions.

        Waits on per-axis arrival events set by the motion monitor thread.
        Zero serial I/O from the calling thread — all firmware queries
        happen on the monitor thread at 50 Hz.

        Args:
            timeout: Maximum seconds to wait (default 120s).

        Returns:
            bool: True if all axes arrived, False if timed out.
        """
        if not self.motion or not self.motion.driver:
            return True

        deadline = time.monotonic() + timeout
        for ax in self.VALID_AXES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(f'[SCOPE API ] wait_until_finished_moving timed out on axis {ax}')
                return False
            if not self._arrival_events[ax].wait(timeout=remaining):
                logger.warning(f'[SCOPE API ] wait_until_finished_moving timed out on axis {ax}')
                return False

        return True


    def set_acceleration_limit(self, val_pct: int):
        self.motion.set_acceleration_limits(val_pct=val_pct)


    def get_microscope_model(self):
        """Get the microscope model identifier from the motion board.

        Returns:
            str | None: Model string, or None if motion board inactive.
        """
        if not self.motion.driver:
            return None

        return self.motion.get_microscope_model()

    def get_motor_info(self) -> dict:
        """Get motor controller information.

        Returns:
            dict: Keys 'model', 'serial_number', 'firmware_version'.
                  Values are None/unknown if board inactive.
        """
        if not self.motion or not self.motion.driver:
            return {'model': None, 'serial_number': None, 'firmware_version': None}

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
        if not self.led or not self.led.driver:
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
    def capture(self):
        """INTEGRATED SCOPE FUNCTIONS
        Capture image with illumination"""

        if not self.led: return
        if not self.camera or not self.camera.active: return

        # Set capture states
        self.is_capturing = True
        self.capture_return = False

        # Wait time for exposure and rolling shutter
        wait_time = 2*self.get_exposure_time()/1000+0.2
        #print("Wait time = ", wait_time)

        # Start thread to wait until capture is complete
        capture_timer = threading.Timer(wait_time, self.capture_complete)
        capture_timer.start()

    def capture_complete(self):
        self.capture_return = self.get_image() # Grab image
        self.is_capturing = False


    def capture_blocking(self):
        if not self.led: return
        if not self.camera or not self.camera.active: return

        wait_time = 2*self.get_exposure_time()/1000+0.2
        time.sleep(wait_time)
        return self.get_image()

    def capture_and_wait(self, force_to_8bit=True, *, exclude_sources=(),
                         all_ones_check=False,
                         timeout=datetime.timedelta(seconds=0),
                         sum_count=1, sum_delay_s=0,
                         sum_iteration_callback=None):
        """Capture a frame guaranteed to reflect the current hardware state.

        Uses frame-based settling: drains stale frames from the camera pipeline
        until frame_validity confirms all pending state changes (LED, gain,
        exposure, motion) have settled. Then grabs a fresh valid frame.

        Frame-based settling automatically adapts to the camera's frame rate —
        fast exposures drain quickly, slow exposures drain slowly, matching
        the actual camera pipeline depth.

        Args:
            force_to_8bit: Convert to 8-bit output.
            exclude_sources: Sources to ignore for validity (e.g. ('z_move',)
                for autofocus where Z motion doesn't need to fully settle).
            all_ones_check: Reject all-max-value frames (camera hardware issue).
            timeout: Timeout for the final get_image call.
            sum_count: Number of frames to sum for noise reduction.
            sum_delay_s: Delay between summed frames.
            sum_iteration_callback: Called after each summed frame.
        """
        if not self.camera or not self.camera.active:
            return False

        exposure_s = self.get_exposure_time() / 1000
        grab_timeout = max(exposure_s * 3, 1.0)

        # Drain stale frames until all pending state changes have settled
        while self.frame_validity.frames_until_valid(exclude_sources=exclude_sources) > 0:
            status, _ = self.camera.grab_new_capture(timeout=grab_timeout)
            if status:
                self.frame_validity.count_frame()
            else:
                logger.warning('[SCOPE API ] capture_and_wait: frame drain failed')
                return False

        return self.get_image(
            force_to_8bit=force_to_8bit,
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

    # Functional, but not integrated with LVP, just for scripting at the moment.

    def autofocus(self, AF_min, AF_max, AF_range):
        """INTEGRATED SCOPE FUNCTIONS
        begin autofocus functionality"""

        # Check all hardware required
        if not self.led: return
        if not self.motion: return
        if not self.camera: return

        # Check if hardware is actively responding
        if self.led.driver is False: return
        if self.motion.driver is False: return
        if not self.camera.active: return

        # Set autofocus states
        self.is_focusing = True          # Is the microscope currently attempting autofocus
        self.autofocus_return = False    # Will be z-position if focus is ready to pull, else False

        # Determine center of AF
        center = self.get_current_position('Z')

        self.z_min = max(0, center-AF_range)      # starting minimum z-height for autofocus
        self.z_max = center+AF_range              # starting maximum z-height for autofocus
        self.resolution = AF_max                  # starting step size for autofocus

        self.AF_positions = []       # List of positions to step through
        self.focus_measures = []     # Measure focus score at each position
        self.last_focus_score = None    # Last / Previous focus score
        self.last_focus_pass = False # Are we on the last scan for autofocus?

        # Start the autofocus process at z-minimum
        self.move_absolute_position('Z', self.z_min)

        while not self.autofocus_iterate(AF_min):
            time.sleep(0.01)

    def autofocus_iterate(self, AF_min):
        """INTEGRATED SCOPE FUNCTIONS
        iterate autofocus functionality"""
        done=False

        # Ignore steps until conditions are met
        if self.is_moving(): return done  # needs to be in position
        if self.is_capturing: return done # needs to have completed capture with illumination

        # Is there a previous capture result to pull?
        if self.capture_return is False:
            # No -> start a capture event
            self.capture()
            return done

        else:
            # Yes -> pull the capture result and clear
            image = self.capture_return
            self.capture_return = False

        if image is False:
            # Stop thread image can't be acquired
            done = True
            return done

        # observe the image
        rows, cols = image.shape

        # Use center quarter of image for focusing
        image = image[int(rows/4):int(3*rows/4),int(cols/4):int(3*cols/4)]

        # calculate the position and focus measure
        try:
            current = self.get_current_position('Z')
            focus = autofocus_functions.focus_function(image=image)
            next_target = self.get_target_position('Z') + self.resolution
        except Exception:
            logger.exception('[SCOPE API ] Error talking to motion controller.')

        # append to positions and focus measures
        self.AF_positions.append(current)
        self.focus_measures.append(focus)

        if next_target <= self.z_max:
            self.move_relative_position('Z', self.resolution)
            return done

        # Adjust future steps if next_target went out of bounds
        # Calculate new step size for resolution
        prev_resolution = self.resolution
        self.resolution = prev_resolution / 3 # SELECT DESIRED RESOLUTION FRACTION

        if self.resolution < AF_min:
            self.resolution = AF_min
            self.last_focus_pass = True

        # compute best focus
        focus = self.focus_best(self.AF_positions, self.focus_measures)

        if not self.last_focus_pass:
            # assign new z_min, z_max, resolution, and sweep
            self.z_min = focus-prev_resolution
            self.z_max = focus+prev_resolution

            # reset positions and focus measures
            self.AF_positions = []
            self.focus_measures = []

            # go to new z_min
            self.move_absolute_position('Z', self.z_min)

        else:
            # go to best focus
            self.move_absolute_position('Z', focus) # move to absolute target

            # end autofocus sequence
            self.autofocus_return = focus
            self.is_focusing = False
            self.last_focus_score = focus

            # Stop thread image when autofocus is complete
            done=True
        return done

    def focus_best(self, positions, values, algorithm='direct'):
        """INTEGRATED SCOPE FUNCTIONS
        select best focus position for autofocus function"""

        logger.info('[SCOPE API ] Lumascope.focus_best()')
        if algorithm == 'direct':
            max_value = max(values)
            max_index = values.index(max_value)
            return positions[max_index]

        elif algorithm == 'mov_avg':
            avg_values = np.convolve(values, [.5, 1, 0.5], 'same')
            max_index = avg_values.argmax()
            return positions[max_index]

        else:
            return positions[0]


# Static methods for save_image functionality
    @staticmethod
    def get_next_save_path_static(path):
        """ GETS THE NEXT SAVE PATH GIVEN AN EXISTING SAVE PATH

            :param path of the format './{save_folder}/{well_label}_{color}_{file_id}.tiff'
            :returns the next save path './{save_folder}/{well_label}_{color}_{file_id + 1}.tiff'

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
    def generate_image_save_path_static(save_folder, file_root, append, tail_id_mode, output_format):
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
    ):
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
    ):
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
        binning_size=None, exposure_time_ms=None, gain_db=None, illumination_ma=None
    ):
        """CAMERA FUNCTIONS
        save image (as array) to file - static version that doesn't require Lumascope instance

        :param array: image array to save
        :param save_folder: folder to save image in
        :param file_root: root filename
        :param append: string to append to filename
        :param color: color channel identifier
        :param tail_id_mode: how to handle filename incrementing
        :param output_format: output format (TIFF or OME-TIFF)
        :param true_color: true color for metadata
        :param x: x position
        :param y: y position
        :param z: z position
        :param objective: objective dictionary with focal_length
        :param labware: labware configuration
        :param stage_offset: stage offset configuration
        :param coordinate_transformer: coordinate transformer instance
        :param binning_size: camera binning size
        :param exposure_time_ms: exposure time in milliseconds
        :param gain_db: camera gain in dB
        :param illumination_ma: LED illumination in mA
        """

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
                color=color
            )

            print(f'[SCOPE API ] Saving Image to {file_loc}')
        except Exception:
            print(f"[SCOPE API ] Error: Unable to save. Perhaps save folder does not exist? {file_loc}")
            raise CaptureError(f"Unable to save image to {file_loc}")

        return file_loc
