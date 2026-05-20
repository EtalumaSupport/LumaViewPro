# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Simulated Camera — drop-in replacement for PylonCamera / IDSCamera.

No camera hardware required. Generates synthetic images, tracks all
camera state (exposure, gain, binning, frame size, pixel format), and
supports the full Camera ABC interface.
"""

import contextlib
import datetime
import pathlib
import threading
import time
from typing import Callable

import numpy as np
from scipy.ndimage import uniform_filter

from lvp_logger import logger
from drivers.camera import Camera
from drivers.registry import camera_registry

# camera.log hookup: simulator records the same per-driver SDK-call
# trace that real drivers (pyloncamera/idscamera/fx2driver) write, so
# sim-mode runs produce a populated logs/camera.log for verification.
try:
    from lvp_logger import camera_logger as _cam_log
except ImportError:
    _cam_log = None


@camera_registry.register('sim', priority=100, is_simulator=True)
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

        # Per-frame callback delivery (mirrors the Pylon/IDS ImageHandler
        # callback surface). SimulatedCamera has no SDK callback thread,
        # so a host-side pump thread fires callbacks at the exposure rate
        # whenever any are registered AND grabbing is active. Tests that
        # exercise the production callback path use this; the display
        # pull-pipeline (grab/grab_latest) keeps working as before.
        self._frame_callbacks: list = []
        self._frame_callback_lock = threading.Lock()
        self._pump_thread: threading.Thread | None = None
        self._pump_stop = threading.Event()

        # Synthetic image state — can be set externally for test scenarios
        # 'gradient', 'black', 'white', 'noise', 'focus_target', 'image_cycle'
        self._test_pattern = 'gradient'

        # Image cycling: load real images from data/sim_images/ and cycle through
        self._cycle_images = []       # List of numpy arrays (grayscale)
        self._cycle_index = 0

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

    def set_timing_mode(self, mode: str) -> None:
        """Switch timing mode: 'fast' or 'realistic'.

        Args:
            mode: One of ``'fast'``, ``'realistic'``.

        Raises:
            ValueError: ``mode`` is not a known preset.
        """
        if mode == 'realistic':
            preset = self.TIMING_REALISTIC
        elif mode == 'fast':
            preset = self.TIMING_FAST
        else:
            raise ValueError(f"Unknown timing mode: {mode!r}. Use 'fast' or 'realistic'.")
        self._grab_delay = preset['grab_delay']
        self._timing_mode = mode

    def load_cycle_images(self, image_dir=None) -> None:
        """Load images from a directory for cycling through in simulate mode.

        Images are resized to match the camera resolution and converted
        to grayscale. If no directory is provided, checks data/sim_images/.
        If no images are found, generates 4 synthetic patterns instead.

        Args:
            image_dir: Path to directory containing image files (png, jpg, tiff).
                       If None, uses data/sim_images/ relative to the app root.
        """
        images = []

        if image_dir is None:
            # Try default location
            for candidate in [
                pathlib.Path(__file__).resolve().parent.parent / 'data' / 'sim_images',
                pathlib.Path('.') / 'data' / 'sim_images',
            ]:
                if candidate.is_dir():
                    image_dir = candidate
                    break

        if image_dir is not None:
            image_dir = pathlib.Path(image_dir)
            if image_dir.is_dir():
                try:
                    from PIL import Image as PILImage
                    for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
                        for fp in sorted(image_dir.glob(ext)):
                            try:
                                pil_img = PILImage.open(fp).convert('L')
                                h = self._height // self._binning
                                w = self._width // self._binning
                                pil_img = pil_img.resize((w, h), PILImage.LANCZOS)
                                images.append(np.array(pil_img, dtype=np.uint8))
                                logger.info(f'[SimCamera ] Loaded cycle image: {fp.name}')
                            except Exception as e:
                                logger.warning(f'[SimCamera ] Could not load {fp}: {e}')
                except ImportError:
                    logger.warning('[SimCamera ] Pillow not available -- cannot load cycle images')

        if not images:
            # Generate 4 synthetic patterns as fallback
            h = self._height // self._binning
            w = self._width // self._binning
            # 1: Horizontal gradient
            images.append(np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1)))
            # 2: Vertical gradient
            images.append(np.tile(np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1), (1, w)))
            # 3: Radial gradient (bullseye-like)
            y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
            r = np.sqrt(x.astype(float)**2 + y.astype(float)**2)
            images.append(((r / r.max()) * 255).astype(np.uint8))
            # 4: Checkerboard
            block = 40
            checker = np.indices((h, w)).sum(axis=0) // block % 2
            images.append((checker * 200 + 30).astype(np.uint8))
            logger.info(f'[SimCamera ] Generated {len(images)} synthetic cycle images')

        self._cycle_images = images
        self._cycle_index = 0
        self._test_pattern = 'image_cycle'
        logger.info(f'[SimCamera ] Image cycling enabled with {len(images)} images')

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """Mark the simulated camera as active and load its profile.

        Returns:
            bool: Always True.
        """
        with self._lock:
            self.active = True
            self.model_name = self.MODEL_NAME
            self._device_serial = self.SERIAL_NUMBER
            self._device_removed = False

            self._load_profile()
            self.init_camera_config()
            self._grabbing = True

            if _cam_log is not None: _cam_log.info(f'sim Connected: {self.model_name} ({self._device_serial})')
            logger.info(f'[CAM Sim   ] Connected: {self.model_name} ({self._device_serial})')
            return True

    def disconnect(self) -> bool:
        """Mark the simulated camera as disconnected.

        Returns:
            bool: True when the camera was active before this call,
                False when it was already disconnected.
        """
        with self._lock:
            if self.active:
                self._grabbing = False
                self.active = None
                if _cam_log is not None: _cam_log.info('sim Disconnected')
                logger.info('[CAM Sim   ] Disconnected')
                self._stop_callback_pump()
                return True
            return False

    def is_connected(self) -> bool:
        """Whether the simulated camera is currently connected.

        Returns:
            bool: True when active and the device-removed flag is clear.
        """
        if self.active in (False, None):
            return False
        if self._device_removed:
            return False
        return True

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def init_camera_config(self) -> None:
        """Reset simulated camera to default config (Mono8, 10 ms, gain=1, bin=1)."""
        if not self.active:
            return
        self._pixel_format = 'Mono8'
        self._exposure_us = 10_000.0  # 10 ms
        self._gain = 1.0
        self._binning = 1

    # ------------------------------------------------------------------
    # Grabbing
    # ------------------------------------------------------------------
    def is_grabbing(self) -> bool:
        """Return whether the simulated camera is currently acquiring.

        Returns:
            bool: True when ``start_grabbing()`` has been called and
                ``stop_grabbing()`` has not.
        """
        return self._grabbing

    def start_grabbing(self) -> None:
        """Begin acquiring frames in the simulator."""
        with self._lock:
            self._grabbing = True
            if _cam_log is not None: _cam_log.info('sim start_grabbing')
            logger.info('[CAM Sim   ] start_grabbing')
        # Re-spawn the pump if callbacks were registered while not grabbing.
        with self._frame_callback_lock:
            need_pump = bool(self._frame_callbacks)
        if need_pump:
            self._start_callback_pump()

    def stop_grabbing(self) -> None:
        """Stop acquiring frames in the simulator."""
        with self._lock:
            self._grabbing = False
            if _cam_log is not None: _cam_log.info('sim stop_grabbing')
            logger.info('[CAM Sim   ] stop_grabbing')
        self._stop_callback_pump()

    # ------------------------------------------------------------------
    # Per-frame callbacks (parity with Pylon/IDS ImageHandler surface)
    # ------------------------------------------------------------------
    def register_frame_callback(self, cb) -> None:
        """Register a callback fired on every simulated grab.

        Starts a small host-side pump thread on the first registration
        while ``_grabbing`` is True, so callers (manual record) see the
        same push-driven semantics they get from real cameras.
        """
        with self._frame_callback_lock:
            if cb not in self._frame_callbacks:
                self._frame_callbacks.append(cb)
            need_pump = bool(self._frame_callbacks) and self._grabbing
        if need_pump:
            self._start_callback_pump()

    def unregister_frame_callback(self, cb) -> None:
        """Remove a registered callback; stops the pump when none remain."""
        with self._frame_callback_lock:
            with contextlib.suppress(ValueError):
                self._frame_callbacks.remove(cb)
            still_active = bool(self._frame_callbacks)
        if not still_active:
            self._stop_callback_pump()

    def _start_callback_pump(self) -> None:
        """Spawn the callback pump if not already running."""
        if self._pump_thread is not None and self._pump_thread.is_alive():
            return
        self._pump_stop.clear()
        self._pump_thread = threading.Thread(
            target=self._callback_pump_loop,
            name='SimCameraPump',
            daemon=True,
        )
        self._pump_thread.start()

    def _stop_callback_pump(self) -> None:
        """Signal the pump to exit and join with a short timeout."""
        self._pump_stop.set()
        t = self._pump_thread
        if t is not None:
            t.join(timeout=2.0)
        self._pump_thread = None

    def _callback_pump_loop(self) -> None:
        """Fire registered callbacks at ``1 / exposure_s`` while grabbing.

        Generates a fresh image per tick so the callback gets a unique
        ``(image, ts, chunks=None)`` triple. SimulatedCamera has no
        chunk surface, so chunks is always None — recording callers
        already treat None as "skip chunk-derived metadata."
        """
        while not self._pump_stop.is_set():
            if not self._grabbing:
                # Pump only delivers while grabbing; cheap idle loop.
                if self._pump_stop.wait(0.05):
                    return
                continue
            with self._frame_callback_lock:
                cbs = list(self._frame_callbacks)
            if not cbs:
                return
            with self._lock:
                self.array = self._generate_image()
                ts = datetime.datetime.now()
                self._last_grab_ts = ts
                image = self.array.copy()
            for cb in cbs:
                try:
                    cb(image, ts, None)
                except Exception as e:
                    logger.exception(
                        f'[CAM Sim   ] frame callback raised: {e}'
                    )
            # Honor the configured exposure as the inter-frame interval.
            interval_s = max(self._exposure_us / 1_000_000.0, 0.001)
            if self._pump_stop.wait(interval_s):
                return

    # ------------------------------------------------------------------
    # Frame size
    # ------------------------------------------------------------------
    def set_frame_size(self, w: int, h: int) -> None:
        """Set the simulated camera frame size, clamped to valid ranges.

        Args:
            w: Target width in pixels (snapped to multiple of 48,
                clamped to [48, 4096]).
            h: Target height in pixels (snapped to multiple of 4,
                clamped to [4, 4096]).
        """
        with self._lock:
            self._width = max(48, min(4096, int(w / 48) * 48))
            self._height = max(4, min(4096, int(h / 4) * 4))
            if _cam_log is not None: _cam_log.info(f'sim set_frame_size({self._width}x{self._height})')

    def get_min_frame_size(self) -> dict:
        """Return the simulator's minimum supported frame size.

        Returns:
            dict: ``{'width': 48, 'height': 4}``.
        """
        return {'width': 48, 'height': 4}

    def get_max_frame_size(self) -> dict:
        """Return the simulator's maximum supported frame size.

        Returns:
            dict: ``{'width': 4096, 'height': 4096}``.
        """
        return {'width': 4096, 'height': 4096}

    def get_frame_size(self) -> dict:
        """Return the simulated camera's current frame size.

        Returns:
            dict: ``{'width': int, 'height': int}``.
        """
        return {'width': self._width, 'height': self._height}

    # ------------------------------------------------------------------
    # Pixel format
    # ------------------------------------------------------------------
    def set_pixel_format(self, pixel_format: str) -> bool:
        """Set the simulated camera pixel format.

        Args:
            pixel_format: Format identifier (must be in ``PIXEL_FORMATS``).

        Returns:
            bool: True on success, False when the format is not supported.
        """
        if pixel_format not in self.PIXEL_FORMATS:
            if _cam_log is not None: _cam_log.error(f'sim set_pixel_format({pixel_format}) UNSUPPORTED')
            logger.error(f'[CAM Sim   ] Unsupported pixel format: {pixel_format}')
            return False
        with self._lock:
            self._pixel_format = pixel_format
            if _cam_log is not None: _cam_log.info(f'sim set_pixel_format({pixel_format})')
        return True

    def get_pixel_format(self) -> str:
        """Return the simulated camera's current pixel format.

        Returns:
            str: One of ``PIXEL_FORMATS``.
        """
        return self._pixel_format

    def get_supported_pixel_formats(self) -> tuple:
        """Return the supported pixel formats.

        Returns:
            tuple: ``('Mono8', 'Mono10', 'Mono12')``.
        """
        return self.PIXEL_FORMATS

    # ------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------
    def exposure_t(self, exposure_ms: float) -> None:
        """Set exposure time in milliseconds.

        Silently clamps when ``exposure_ms`` exceeds ``max_exposure``
        (logs a warning); silently no-ops when the simulator is not
        active.

        Args:
            exposure_ms: Exposure time in milliseconds.
        """
        if not self.active:
            return
        if exposure_ms > self.max_exposure:
            if _cam_log is not None: _cam_log.warning(f'sim ExposureTime.SetValue({exposure_ms}ms) CLAMPED max={self.max_exposure}ms')
            logger.warning(f'[CAM Sim   ] Exposure {exposure_ms}ms exceeds max ({self.max_exposure}ms)')
            return
        with self._lock:
            self._exposure_us = float(exposure_ms) * 1000.0
            if _cam_log is not None: _cam_log.info(f'sim ExposureTime.SetValue({float(exposure_ms) * 1000.0:.0f}us) (={exposure_ms}ms)')
            logger.info(f'[CAM Sim   ] Exposure set to {exposure_ms}ms')

    def get_exposure_t(self) -> float:
        """Return exposure time in milliseconds.

        Returns:
            float: Exposure in ms, or -1 when the simulator is not active.
        """
        if not self.active:
            return -1
        return self._exposure_us / 1000.0

    def auto_exposure_t(self, state: bool = True) -> bool:
        """Enable or disable simulated auto-exposure (state stored only).

        Args:
            state: True to enable, False to disable.

        Returns:
            bool: Always True.
        """
        self._auto_exposure_enabled = state
        return True

    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------
    def get_all_temperatures(self) -> dict:
        """Return synthetic temperature telemetry.

        Returns:
            dict: ``{'sensor': 35.0, 'board': 40.0}``.
        """
        return {'sensor': 35.0, 'board': 40.0}

    # ------------------------------------------------------------------
    # Frame rate
    # ------------------------------------------------------------------
    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float = 1.0) -> None:
        """Enable or disable the simulated frame-rate cap.

        Args:
            enabled: True to enforce ``fps`` as the upper bound.
            fps: Cap value in frames per second.
        """
        with self._lock:
            self._frame_rate_limit_enabled = enabled
            if enabled:
                self._frame_rate_target = fps
            if _cam_log is not None: _cam_log.info(f'sim set_max_acquisition_frame_rate(enabled={enabled}, fps={fps})')

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------
    def set_binning_size(self, size: int) -> bool:
        """Set hardware binning factor for the simulator.

        Args:
            size: Binning factor (1-4 inclusive).

        Returns:
            bool: True on success, False when ``size`` is unsupported.
        """
        if size < 1 or size > 4:
            if _cam_log is not None: _cam_log.error(f'sim set_binning_size({size}) UNSUPPORTED')
            logger.error(f'[CAM Sim   ] Unsupported bin size: {size}')
            return False
        with self._lock:
            self._binning = size
            if _cam_log is not None: _cam_log.info(f'sim set_binning_size({size})')
        return True

    def get_binning_size(self) -> int:
        """Return the simulator's current binning factor.

        Returns:
            int: Binning factor (1 = no binning).
        """
        return self._binning

    # ------------------------------------------------------------------
    # Z-dependent focus simulation
    # ------------------------------------------------------------------
    def set_z_position(self, z: float) -> None:
        """Set current Z position (um) for focus simulation.

        Args:
            z: Current Z stage position in micrometers.
        """
        self._z_position = float(z)

    def get_z_position(self) -> float:
        """Return the current Z position used for focus simulation.

        Returns:
            float: Z position in micrometers.
        """
        return self._z_position

    def set_focal_z(self, z: float) -> None:
        """Set the Z position (um) where focus is perfect.

        Args:
            z: Focal Z position in micrometers.
        """
        self._focal_z = float(z)

    def get_focal_z(self) -> float:
        """Return the focal Z position.

        Returns:
            float: Focal Z position in micrometers.
        """
        return self._focal_z

    def set_blur_per_um(self, value: float) -> None:
        """Set blur rate: uniform filter size increases by this per um of defocus.

        Args:
            value: Blur sigma increase per um of defocus.
        """
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
        raw = (self._exposure_us / 1_000_000.0) * max(1.0, self._gain) * 10.0
        brightness = min(1.0, raw)
        # For image cycling, apply a floor so patterns are visible even at
        # short default exposures (2ms → raw=0.02, floor lifts to 0.5)
        if self._test_pattern == 'image_cycle':
            brightness = max(0.5, brightness)

        if self._test_pattern == 'image_cycle' and self._cycle_images:
            # Cycle through loaded/generated images
            src = self._cycle_images[self._cycle_index % len(self._cycle_images)]
            self._cycle_index += 1
            # Resize if binning changed since load
            if src.shape != (h, w):
                # Simple nearest-neighbor resize via slicing
                src_h, src_w = src.shape
                y_idx = np.linspace(0, src_h - 1, h, dtype=int)
                x_idx = np.linspace(0, src_w - 1, w, dtype=int)
                src = src[np.ix_(y_idx, x_idx)]
            # Scale to target dtype and apply brightness
            if dtype == np.uint16:
                img = (src.astype(np.float32) / 255.0 * max_val * brightness).astype(dtype)
            else:
                img = (src.astype(np.float32) * brightness).clip(0, max_val).astype(dtype)
        elif self._test_pattern == 'black':
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

    def grab(self) -> tuple:
        """Return the last generated image (non-blocking).

        When image cycling is active, simulates realistic camera behavior:
        a new frame isn't available until the exposure time has elapsed.
        This matches real cameras where grab() returns the latest buffered
        frame and the frame rate is limited by exposure time.

        Returns:
            tuple: ``(success: bool, timestamp: datetime | None)``.
        """
        if not self._grabbing:
            return False, None

        if self._grab_delay > 0:
            time.sleep(self._grab_delay)

        # Gate frame delivery on exposure time (realistic simulation)
        if self._test_pattern == 'image_cycle':
            exposure_s = self._exposure_us / 1_000_000.0
            now = time.monotonic()
            last = getattr(self, '_last_frame_time', 0.0)
            if now - last < exposure_s:
                # Not enough time has passed — return the previous frame
                return True, self._last_grab_ts
            self._last_frame_time = now

        with self._lock:
            self.array = self._generate_image()
            self._last_grab_ts = datetime.datetime.now()

        return True, self._last_grab_ts

    def grab_latest(self) -> tuple:
        """Single-copy grab for display pipeline (overrides Camera.grab_latest).

        SimulatedCamera doesn't use ImageHandlerBase, so we override
        to generate and return the image directly.

        Returns:
            tuple: ``(success: bool, image: np.ndarray | None,
                timestamp: datetime | None)``.
        """
        if not self._grabbing:
            return False, None, None

        if self._grab_delay > 0:
            time.sleep(self._grab_delay)

        if self._test_pattern == 'image_cycle':
            exposure_s = self._exposure_us / 1_000_000.0
            now = time.monotonic()
            last = getattr(self, '_last_frame_time', 0.0)
            if now - last < exposure_s:
                with self._lock:
                    img = self.array.copy() if self.array.size > 0 else None
                return True, img, self._last_grab_ts
            self._last_frame_time = now

        with self._lock:
            self.array = self._generate_image()
            self._last_grab_ts = datetime.datetime.now()
            img = self.array.copy()

        return True, img, self._last_grab_ts

    def grab_new_capture(self, timeout: float) -> tuple:
        """Generate a fresh image (blocking with timeout).

        Args:
            timeout: Accepted for API parity; a small per-call delay
                proportional to exposure is applied (capped at 0.1 s).

        Returns:
            tuple: ``(success: bool, timestamp: datetime | None)``.
        """
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
    def get_gain(self) -> float:
        """Return the simulated camera gain.

        Returns:
            float: Gain in dB, or -1 when the camera is not active.
        """
        if not self.active:
            return -1
        return self._gain

    def gain(self, gain: float) -> None:
        """Set the simulated camera gain.

        Args:
            gain: Gain in dB.
        """
        if not self.active:
            return
        with self._lock:
            self._gain = float(gain)
            if _cam_log is not None: _cam_log.info(f'sim Gain.SetValue({float(gain):.3f})')
            logger.info(f'[CAM Sim   ] Gain set to {gain}')

    def init_auto_gain_focus(
        self,
        auto_target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None,
    ) -> bool:
        """Initialize auto-gain ROI and parameters (no-op in simulation).

        Args:
            auto_target_brightness: Normalized brightness target (0.0-1.0).
            min_gain: Optional lower bound in dB.
            max_gain: Optional upper bound in dB.

        Returns:
            bool: Always True.
        """
        with self._lock:
            self._auto_gain_target_brightness = auto_target_brightness
            if min_gain is not None:
                self._auto_gain_min = min_gain
            if max_gain is not None:
                self._auto_gain_max = max_gain
        return True

    def auto_gain(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
    ) -> bool:
        """Enable or disable simulated continuous auto-gain.

        On enable, the simulator converges immediately by setting gain
        to the midpoint of [min_gain_db, max_gain_db].

        Args:
            state: True to enable, False to disable.
            target_brightness: Normalized brightness target (0.0-1.0).
            min_gain_db: Optional lower bound in dB.
            max_gain_db: Optional upper bound in dB.

        Returns:
            bool: Always True.
        """
        with self._lock:
            self._auto_gain_enabled = state
            if state:
                self._auto_gain_target_brightness = target_brightness
                if min_gain_db is not None:
                    self._auto_gain_min = min_gain_db
                if max_gain_db is not None:
                    self._auto_gain_max = max_gain_db
                # Simulate convergence: set gain to mid-range
                self._gain = (self._auto_gain_min + self._auto_gain_max) / 2.0
            if _cam_log is not None: _cam_log.info(f'sim auto_gain(state={state}, target={target_brightness}, min_db={min_gain_db}, max_db={max_gain_db})')
        return True

    def auto_gain_once(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
    ) -> bool:
        """Run a single simulated auto-gain iteration.

        Converges by setting gain to the midpoint of [min_gain_db, max_gain_db].

        Args:
            state: True to run, False to no-op.
            target_brightness: Normalized brightness target (0.0-1.0).
            min_gain_db: Optional lower bound in dB.
            max_gain_db: Optional upper bound in dB.

        Returns:
            bool: Always True.
        """
        if state:
            with self._lock:
                self._auto_gain_target_brightness = target_brightness
                if min_gain_db is not None:
                    self._auto_gain_min = min_gain_db
                if max_gain_db is not None:
                    self._auto_gain_max = max_gain_db
                # One-shot: converge gain toward target
                self._gain = (self._auto_gain_min + self._auto_gain_max) / 2.0
        return True

    def update_auto_gain_target_brightness(self, auto_target_brightness: float) -> bool:
        """Update the auto-gain target brightness.

        Args:
            auto_target_brightness: Normalized brightness target (0.0-1.0).

        Returns:
            bool: Always True.
        """
        with self._lock:
            self._auto_gain_target_brightness = auto_target_brightness
        return True

    def update_auto_gain_min_max(self, min_gain_db: float | None, max_gain_db: float | None) -> bool:
        """Update auto-gain bounds.

        Args:
            min_gain_db: Minimum gain in dB, or None to leave unchanged.
            max_gain_db: Maximum gain in dB, or None to leave unchanged.

        Returns:
            bool: Always True.
        """
        with self._lock:
            if min_gain_db is not None:
                self._auto_gain_min = min_gain_db
            if max_gain_db is not None:
                self._auto_gain_max = max_gain_db
        return True

    # ------------------------------------------------------------------
    # Test pattern
    # ------------------------------------------------------------------
    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black') -> None:
        """Enable or disable the simulator's test pattern generator.

        Args:
            enabled: True to enable the pattern, False to revert to ``'gradient'``.
            pattern: Pattern name (case-insensitive). Common values:
                ``'black'``, ``'white'``, ``'noise'``, ``'focus_target'``,
                ``'image_cycle'``.
        """
        if enabled:
            self._test_pattern = pattern.lower()
        else:
            self._test_pattern = 'gradient'
