# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Simulated Camera — drop-in replacement for PylonCamera / IDSCamera.

No camera hardware required. Generates synthetic images, tracks all
camera state (exposure, gain, binning, frame size, pixel format), and
supports the full Camera ABC interface.
"""

import datetime
import threading
import time
from typing import Callable

import numpy as np
from scipy.ndimage import uniform_filter

from lvp_logger import logger
from drivers.camera import Camera


class SimulatedCamera(Camera):

    MODEL_NAME = 'SimulatedCamera-1920x1200'
    SERIAL_NUMBER = 'SIM-CAM-001'

    # Supported pixel formats
    PIXEL_FORMATS = ('Mono8', 'Mono10', 'Mono12')

    TIMING_FAST = {'grab_delay': 0.0}
    TIMING_REALISTIC = {'grab_delay': 0.005}  # ~5ms USB transfer overhead

    def __init__(
        self,
        width: int = 1920,
        height: int = 1200,
        grab_delay: float = 0.0,
        z_position_func: Callable[[], float] | None = None,
        timing: str = 'fast',
    ):
        self._width = width
        self._height = height
        self._grab_delay = grab_delay

        self._exposure_us = 10_000.0  # 10 ms in microseconds
        self._gain = 1.0
        self._pixel_format = 'Mono8'
        self._binning = 1
        self._grabbing = False
        self._auto_gain_enabled = False
        self._auto_gain_target_brightness = 0.5
        self._auto_gain_min = 0.0
        self._auto_gain_max = 20.0
        self._auto_exposure_enabled = False
        self._frame_rate_limit_enabled = False
        self._frame_rate_target = 30.0

        self._lock = threading.RLock()
        self._last_grab_ts = None

        # Synthetic image state — can be set externally for test scenarios
        self._test_pattern = 'gradient'  # 'gradient', 'black', 'white', 'noise', 'focus_target'

        # Z-dependent focus simulation
        self._z_position = 5000.0       # Current Z position (um)
        self._focal_z = 5000.0          # Z position of perfect focus (um)
        self._blur_per_um = 0.01        # Blur sigma increase per um of defocus
        self._z_position_func = z_position_func  # Optional: auto-query Z from motor

        # Pre-generated focus target (lazily created)
        self._focus_target_cache = None
        self._focus_target_cache_key = None

        # Apply timing preset (overrides grab_delay if preset given)
        self.set_timing_mode(timing)

        # Let the base class call connect()
        super().__init__()

    def set_timing_mode(self, mode: str):
        """Switch timing mode: 'fast' or 'realistic'."""
        if mode == 'realistic':
            preset = self.TIMING_REALISTIC
        elif mode == 'fast':
            preset = self.TIMING_FAST
        else:
            raise ValueError(f"Unknown timing mode: {mode!r}. Use 'fast' or 'realistic'.")
        self._grab_delay = preset['grab_delay']
        self._timing_mode = mode

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        with self._lock:
            self.active = True
            self.model_name = self.MODEL_NAME
            self._device_serial = self.SERIAL_NUMBER
            self._device_removed = False

            self._load_profile()
            self.init_camera_config()
            self._grabbing = True

            logger.info(f'[CAM Sim   ] Connected: {self.model_name} ({self._device_serial})')
            return True

    def disconnect(self) -> bool:
        with self._lock:
            if self.active:
                self._grabbing = False
                self.active = None
                logger.info('[CAM Sim   ] Disconnected')
                return True
            return False

    def is_connected(self) -> bool:
        if self.active in (False, None):
            return False
        if self._device_removed:
            return False
        return True

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _get_max_exposure_models(self) -> dict:
        return {
            'SimulatedCamera-1920x1200': 10_000,  # 10 seconds max
        }

    def init_camera_config(self):
        if not self.active:
            return
        self._pixel_format = 'Mono8'
        self._exposure_us = 10_000.0  # 10 ms
        self._gain = 1.0
        self._binning = 1

    # ------------------------------------------------------------------
    # Grabbing
    # ------------------------------------------------------------------
    def is_grabbing(self):
        return self._grabbing

    def start_grabbing(self):
        with self._lock:
            self._grabbing = True
            logger.info('[CAM Sim   ] start_grabbing')

    def stop_grabbing(self):
        with self._lock:
            self._grabbing = False
            logger.info('[CAM Sim   ] stop_grabbing')

    # ------------------------------------------------------------------
    # Frame size
    # ------------------------------------------------------------------
    def set_frame_size(self, w, h):
        with self._lock:
            self._width = max(48, min(4096, int(w / 48) * 48))
            self._height = max(4, min(4096, int(h / 4) * 4))

    def get_min_frame_size(self) -> dict:
        return {'width': 48, 'height': 4}

    def get_max_frame_size(self) -> dict:
        return {'width': 4096, 'height': 4096}

    def get_frame_size(self):
        return {'width': self._width, 'height': self._height}

    # ------------------------------------------------------------------
    # Pixel format
    # ------------------------------------------------------------------
    def set_pixel_format(self, pixel_format: str) -> bool:
        if pixel_format not in self.PIXEL_FORMATS:
            logger.error(f'[CAM Sim   ] Unsupported pixel format: {pixel_format}')
            return False
        with self._lock:
            self._pixel_format = pixel_format
        return True

    def get_pixel_format(self) -> str:
        return self._pixel_format

    def get_supported_pixel_formats(self) -> tuple:
        return self.PIXEL_FORMATS

    # ------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------
    def exposure_t(self, t):
        """Set exposure time in milliseconds."""
        if not self.active:
            return
        if t > self.max_exposure:
            logger.warning(f'[CAM Sim   ] Exposure {t}ms exceeds max ({self.max_exposure}ms)')
            return
        with self._lock:
            self._exposure_us = float(t) * 1000.0
            logger.info(f'[CAM Sim   ] Exposure set to {t}ms')

    def get_exposure_t(self):
        """Return exposure time in milliseconds."""
        if not self.active:
            return -1
        return self._exposure_us / 1000.0

    def auto_exposure_t(self, state=True):
        self._auto_exposure_enabled = state
        return True

    # ------------------------------------------------------------------
    # Model name
    # ------------------------------------------------------------------
    def find_model_name(self):
        self.model_name = self.MODEL_NAME

    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------
    def get_all_temperatures(self):
        return {'sensor': 35.0, 'board': 40.0}

    # ------------------------------------------------------------------
    # Frame rate
    # ------------------------------------------------------------------
    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float = 1.0):
        with self._lock:
            self._frame_rate_limit_enabled = enabled
            if enabled:
                self._frame_rate_target = fps

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------
    def set_binning_size(self, size: int) -> bool:
        if size < 1 or size > 4:
            logger.error(f'[CAM Sim   ] Unsupported bin size: {size}')
            return False
        with self._lock:
            self._binning = size
        return True

    def get_binning_size(self) -> int:
        return self._binning

    # ------------------------------------------------------------------
    # Z-dependent focus simulation
    # ------------------------------------------------------------------
    def set_z_position(self, z: float):
        """Set current Z position (um) for focus simulation."""
        self._z_position = float(z)

    def get_z_position(self) -> float:
        return self._z_position

    def set_focal_z(self, z: float):
        """Set the Z position (um) where focus is perfect."""
        self._focal_z = float(z)

    def get_focal_z(self) -> float:
        return self._focal_z

    def set_blur_per_um(self, value: float):
        """Set blur rate: uniform filter size increases by this per um of defocus."""
        self._blur_per_um = float(value)

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    def _make_focus_target(self, h: int, w: int, max_val: int) -> np.ndarray:
        """Generate a sharp focus target with multi-scale features.

        Creates a pattern with edges at multiple spatial frequencies so that
        Vollath F4 (and other focus metrics) produce a smooth, peaked response
        curve when the image is progressively blurred.
        """
        cache_key = (h, w, max_val)
        if self._focus_target_cache_key == cache_key and self._focus_target_cache is not None:
            return self._focus_target_cache

        img = np.zeros((h, w), dtype=np.float32)

        # Grid of fine lines (high frequency — most sensitive to defocus)
        grid_spacing = 8
        img[::grid_spacing, :] = max_val * 0.4
        img[:, ::grid_spacing] = max_val * 0.4

        # Scattered bright spots (simulates point-like features)
        rng = np.random.RandomState(42)  # deterministic
        n_spots = max(20, (h * w) // 5000)
        ys = rng.randint(0, h, n_spots)
        xs = rng.randint(0, w, n_spots)
        for y, x in zip(ys, xs):
            y0 = max(0, y - 2)
            y1 = min(h, y + 3)
            x0 = max(0, x - 2)
            x1 = min(w, x + 3)
            img[y0:y1, x0:x1] = max_val * 0.8

        # Medium-frequency checkerboard (16px blocks)
        block = 16
        yy = np.arange(h) // block
        xx = np.arange(w) // block
        checker = (yy[:, None] + xx[None, :]) % 2
        img += checker * max_val * 0.2

        self._focus_target_cache = img
        self._focus_target_cache_key = cache_key
        return img

    def _apply_defocus_blur(self, img: np.ndarray, max_val: int) -> np.ndarray:
        """Apply blur based on distance from focal Z position."""
        # Query Z position from motor if callback is wired
        if self._z_position_func is not None:
            try:
                self._z_position = self._z_position_func()
            except Exception:
                pass

        defocus = abs(self._z_position - self._focal_z)
        if defocus < 1.0:
            return img

        # uniform_filter size must be odd integer >= 1
        filter_size = int(defocus * self._blur_per_um * 2) * 2 + 1
        filter_size = min(filter_size, min(img.shape) // 2)
        if filter_size < 3:
            return img

        blurred = uniform_filter(img.astype(np.float32), size=filter_size)
        return np.clip(blurred, 0, max_val)

    def _generate_image(self) -> np.ndarray:
        """Generate a synthetic image based on current settings."""
        h = self._height // self._binning
        w = self._width // self._binning

        if self._pixel_format in ('Mono10', 'Mono12'):
            dtype = np.uint16
            max_val = 4095 if self._pixel_format == 'Mono12' else 1023
        else:
            dtype = np.uint8
            max_val = 255

        # Scale brightness by exposure and gain
        brightness = min(1.0, (self._exposure_us / 1_000_000.0) * self._gain * 10.0)

        if self._test_pattern == 'black':
            img = np.zeros((h, w), dtype=dtype)
        elif self._test_pattern == 'white':
            img = np.full((h, w), max_val, dtype=dtype)
        elif self._test_pattern == 'noise':
            img = np.random.randint(0, int(max_val * brightness) + 1, (h, w), dtype=dtype)
        elif self._test_pattern == 'focus_target':
            base = self._make_focus_target(h, w, max_val)
            img = self._apply_defocus_blur(base * brightness, max_val)
            img = img.astype(dtype)
        else:
            # Default gradient — also apply defocus blur if Z tracking is active
            row = np.linspace(0, max_val * brightness, w, dtype=np.float32)
            img = np.tile(row, (h, 1)).astype(dtype)

        return img

    def grab(self):
        """Return the last generated image (non-blocking)."""
        if not self._grabbing:
            return False, None

        if self._grab_delay > 0:
            time.sleep(self._grab_delay)

        with self._lock:
            self.array = self._generate_image()
            self._last_grab_ts = datetime.datetime.now()

        return True, self._last_grab_ts

    def grab_new_capture(self, timeout):
        """Generate a fresh image (blocking with timeout)."""
        if not self._grabbing:
            return False, None

        # Simulate exposure delay (capped to avoid slow tests)
        delay = min(self._exposure_us / 1_000_000.0, 0.1)
        if delay > 0:
            time.sleep(delay)

        with self._lock:
            self.array = self._generate_image()
            self._last_grab_ts = datetime.datetime.now()

        return True, self._last_grab_ts

    # ------------------------------------------------------------------
    # Gain
    # ------------------------------------------------------------------
    def get_gain(self):
        if not self.active:
            return -1
        return self._gain

    def gain(self, gain):
        if not self.active:
            return
        with self._lock:
            self._gain = float(gain)
            logger.info(f'[CAM Sim   ] Gain set to {gain}')

    def init_auto_gain_focus(
        self,
        auto_target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None,
    ):
        """Initialize auto-gain ROI and parameters (no-op in simulation)."""
        with self._lock:
            self._auto_gain_target_brightness = auto_target_brightness
            if min_gain is not None:
                self._auto_gain_min = min_gain
            if max_gain is not None:
                self._auto_gain_max = max_gain
        return True

    def auto_gain(
        self,
        state=True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None,
    ):
        with self._lock:
            self._auto_gain_enabled = state
            if state:
                self._auto_gain_target_brightness = target_brightness
                if min_gain is not None:
                    self._auto_gain_min = min_gain
                if max_gain is not None:
                    self._auto_gain_max = max_gain
                # Simulate convergence: set gain to mid-range
                self._gain = (self._auto_gain_min + self._auto_gain_max) / 2.0
        return True

    def auto_gain_once(
        self,
        state=True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None,
    ):
        if state:
            with self._lock:
                self._auto_gain_target_brightness = target_brightness
                if min_gain is not None:
                    self._auto_gain_min = min_gain
                if max_gain is not None:
                    self._auto_gain_max = max_gain
                # One-shot: converge gain toward target
                self._gain = (self._auto_gain_min + self._auto_gain_max) / 2.0
        return True

    def update_auto_gain_target_brightness(self, auto_target_brightness: float):
        with self._lock:
            self._auto_gain_target_brightness = auto_target_brightness
        return True

    def update_auto_gain_min_max(self, min_gain: float | None, max_gain: float | None):
        with self._lock:
            if min_gain is not None:
                self._auto_gain_min = min_gain
            if max_gain is not None:
                self._auto_gain_max = max_gain
        return True

    # ------------------------------------------------------------------
    # Test pattern
    # ------------------------------------------------------------------
    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black'):
        if enabled:
            self._test_pattern = pattern.lower()
        else:
            self._test_pattern = 'gradient'
