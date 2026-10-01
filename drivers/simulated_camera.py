# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Simulated Camera -- drop-in replacement for PylonCamera / IDSCamera.

No camera hardware required. Generates synthetic images, tracks all
camera state (exposure, gain, binning, frame size, pixel format), and
supports the full Camera ABC interface.
"""

import datetime
import pathlib
import threading
import time
from collections.abc import Callable
from typing import ClassVar

import numpy as np
from scipy.ndimage import uniform_filter

from lvp_logger import logger
from drivers.camera import Camera, ImageHandlerBase
from drivers.registry import camera_registry
from drivers.simulated_specimen import specimen_frames

# camera.log hookup: simulator records the same per-driver SDK-call
# trace that real drivers (pyloncamera/idscamera/fx2driver) write, so
# sim-mode runs produce a populated logs/camera.log for verification.
try:
    from lvp_logger import camera_logger as _cam_log
except ImportError:
    _cam_log = None


class _SimImageHandler(ImageHandlerBase):
    """The real drivers' frame buffer, with a wait for the next stored frame.

    The simulator's acquisition thread stores into it exactly as a real
    driver's SDK thread does, so the frame count, the buffered frame and the
    per-frame callbacks all come from the one implementation every driver
    shares. The wait is what ``grab_new_capture`` needs: a frame stored after
    the call, without polling.
    """

    def __init__(self):
        super().__init__()
        self._stored = threading.Condition()

    def _store_frame(self, image, timestamp, chunks=None, *, significant_bits):
        super()._store_frame(image, timestamp, chunks, significant_bits=significant_bits)
        with self._stored:
            self._stored.notify_all()

    def wait_for_frame_after(self, since: int, timeout_s: float) -> bool:
        """True once a frame later than ordinal ``since`` is stored; False on timeout."""
        deadline = time.monotonic() + timeout_s
        with self._stored:
            while self.frames_delivered <= since:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._stored.wait(remaining)
            return True


@camera_registry.register('sim', priority=100, is_simulator=True)
class SimulatedCamera(Camera):
    MODEL_NAME = 'SimulatedCamera-1920x1200'
    SERIAL_NUMBER = 'SIM-CAM-001'

    # Supported pixel formats
    PIXEL_FORMATS = ('Mono8', 'Mono10', 'Mono12')

    TIMING_FAST: ClassVar[dict] = {'grab_delay': 0.0}
    TIMING_REALISTIC: ClassVar[dict] = {'grab_delay': 0.005}  # ~5ms USB transfer overhead

    # Free-run delivery ceiling in frames per second. A real camera's
    # sensor readout and link bandwidth bound its frame rate however
    # short the exposure; without a ceiling the simulator free-runs at
    # 1/exposure (1000 fps at 1 ms), which no camera delivers.
    _MAX_DELIVERY_FPS = 40.0

    def __init__(
        self,
        width: int = 1920,
        height: int = 1200,
        grab_delay: float = 0.0,
        z_position_func: Callable[[], float] | None = None,
        illumination_func: Callable[[], float] | None = None,
        timing: str = 'fast',
    ):
        # Native (unbinned) sensor size -- the fixed ceiling. _width/_height
        # below are the CURRENT frame size in post-binning (displayed) pixels,
        # matching the Pylon driver's convention: set_frame_size takes the
        # post-binning ROI, and the grabbed image is exactly that size (the
        # sensor delivers native/binning pixels). At binning 1 the frame
        # equals the native size.
        self._native_width = width
        self._native_height = height
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

        # A real camera free-runs: while it is grabbing, frames arrive on the
        # SDK's own thread whether or not anyone reads them, and a reader gets
        # the latest one. The simulator does the same on this acquisition
        # thread, storing each frame in the shared image handler. A simulator
        # that made a frame only when one was read showed a frame count that
        # stood still while nothing read it, so a stream that had died and a
        # stream nobody was watching looked the same.
        self._acquisition_thread: threading.Thread | None = None
        self._acquisition_stop = threading.Event()

        # Synthetic image state -- can be set externally for test scenarios
        # 'specimen', 'black', 'white', 'noise', 'focus_target', 'image_cycle'
        # 'specimen' is the no-pattern-requested state; anything unrecognized
        # renders it too, with a warning.
        self._test_pattern = 'specimen'

        # Image cycling: load real images from data/sim_images/ and cycle through
        self._cycle_images = []  # List of numpy arrays (grayscale)
        self._cycle_index = 0

        # Z-dependent focus simulation
        self._z_position = 5000.0  # Current Z position (um)
        self._focal_z = 5000.0  # Z position of perfect focus (um)
        self._blur_per_um = 0.01  # Blur sigma increase per um of defocus
        self._z_position_func = z_position_func  # Optional: auto-query Z from motor
        # Optional: auto-query how much light is on the sample. Unwired, the
        # camera renders as though the field were lit, which is what a camera
        # constructed on its own with no scope around it has to assume.
        self._illumination_func = illumination_func

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

    def load_cycle_images(self, image_dir: str | None = None) -> None:
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
                                h = self._height
                                w = self._width
                                pil_img = pil_img.resize((w, h), PILImage.LANCZOS)
                                images.append(np.array(pil_img, dtype=np.uint8))
                                logger.info(f'[SimCamera ] Loaded cycle image: {fp.name}')
                            except Exception as e:
                                logger.warning(f'[SimCamera ] Could not load {fp}: {e}')
                except ImportError:
                    logger.warning('[SimCamera ] Pillow not available -- cannot load cycle images')

        if not images:
            images = specimen_frames(self._height, self._width)
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
            # A fresh handler per connection, as the real drivers build one:
            # its frame count starts over, and the durable callback registry
            # is pushed onto it so a listener registered before a reconnect
            # keeps receiving frames.
            self.cam_image_handler = _SimImageHandler()
            self._reapply_frame_callbacks()
            # connect() returns CONFIGURED but NOT grabbing; the single
            # start fires later via open_and_start() (the start gate).

            if _cam_log is not None:
                _cam_log.info(f'sim Connected: {self.model_name} ({self._device_serial})')
            logger.info(f'[CAM Sim   ] Connected: {self.model_name} ({self._device_serial})')
            return True

    def disconnect(self) -> bool:
        """Mark the simulated camera as disconnected.

        Returns:
            bool: True when the camera was active before this call,
                False when it was already disconnected.
        """
        with self._lock:
            if not self.active:
                return False
            self._grabbing = False
            self.active = None
            if _cam_log is not None:
                _cam_log.info('sim Disconnected')
            logger.info('[CAM Sim   ] Disconnected')
        # Stop acquisition OUTSIDE self._lock (mirroring stop_grabbing): the
        # acquisition loop generates frames under self._lock, so joining it
        # while holding that lock stalls the full join timeout whenever it is
        # mid-frame.
        self._stop_acquisition()
        # Reset base lifecycle state OUTSIDE self._lock too:
        # _reset_lifecycle_state takes _lifecycle_lock, and holding sim's _lock
        # across it would create a _lock -> _lifecycle_lock acquisition order
        # (the base config path takes them the other way). Acquisition is
        # stopped above, so nothing writes the frame buffer from here on.
        self._reset_lifecycle_state()
        return True

    def is_connected(self) -> bool:
        """Whether the simulated camera is currently connected.

        Returns:
            bool: True when active and the device-removed flag is clear.
        """
        if self.active in (False, None):
            return False
        return not self._device_removed

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
            if _cam_log is not None:
                _cam_log.info('sim start_grabbing')
            logger.info('[CAM Sim   ] start_grabbing')
        self._start_acquisition()

    def stop_grabbing(self) -> None:
        """Stop acquiring frames in the simulator."""
        with self._lock:
            self._grabbing = False
            if _cam_log is not None:
                _cam_log.info('sim stop_grabbing')
            logger.info('[CAM Sim   ] stop_grabbing')
        self._stop_acquisition()

    def _start_acquisition(self) -> None:
        """Spawn the acquisition thread if not already running."""
        if self._acquisition_thread is not None and self._acquisition_thread.is_alive():
            return
        self._acquisition_stop.clear()
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_loop,
            name='SimCameraAcquisition',
            daemon=True,
        )
        self._acquisition_thread.start()

    def _stop_acquisition(self) -> None:
        """Signal the acquisition thread to exit and join with a short timeout.

        A listener that stops grabbing from inside its own frame callback runs
        on the acquisition thread, which cannot join itself; the stop flag
        alone ends the loop once the callback returns.
        """
        self._acquisition_stop.set()
        t = self._acquisition_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._acquisition_thread = None

    def _acquisition_loop(self) -> None:
        """Store one new frame per frame interval while grabbing.

        The interval is the exposure, never shorter than the delivery
        ceiling's period (``_MAX_DELIVERY_FPS``). Frames are due on a
        fixed schedule, so the host time spent generating and delivering
        a frame comes out of the interval rather than adding to it -- a
        real camera's frame period does not include host work. A loop
        that falls behind (generation slower than the interval) delivers
        the next frame at once and re-anchors, never bursting to catch up.

        Each frame goes through the image handler's ``_store_frame``, which
        counts it, buffers it with its depth and fires the per-frame
        callbacks. SimulatedCamera has no chunk surface, so chunks is always
        None -- recording callers already treat None as "skip chunk-derived
        metadata."
        """
        next_due = time.monotonic()
        while not self._acquisition_stop.is_set():
            handler = self.cam_image_handler
            if handler is None or not self._grabbing:
                return
            with self._lock:
                image = self._generate_image()
                bits = self.significant_bits
            # The transfer from a real camera to the host takes time after the
            # frame is exposed; the realistic preset charges it here, between
            # the frame being made and the host having it.
            if self._grab_delay > 0:
                time.sleep(self._grab_delay)
            handler._store_frame(image, datetime.datetime.now(), significant_bits=bits)
            # Honor the configured exposure as the inter-frame interval,
            # bounded below by the delivery ceiling.
            interval_s = max(self._exposure_us / 1_000_000.0, 1.0 / self._MAX_DELIVERY_FPS)
            next_due = max(next_due + interval_s, time.monotonic())
            if self._acquisition_stop.wait(next_due - time.monotonic()):
                return

    # ------------------------------------------------------------------
    # Frame size
    # ------------------------------------------------------------------
    def set_frame_size(self, w: int, h: int) -> dict:
        """Set the simulated camera frame size, clamped to valid ranges.

        ``w`` and ``h`` are post-binning (displayed) pixels, so the ceiling is
        the native sensor size divided by the current binning factor -- the
        same constraint Pylon enforces via ``Width.Max`` at the active binning.

        Args:
            w: Target width in pixels (snapped to a multiple of 48,
                clamped to [48, native_width / binning]).
            h: Target height in pixels (snapped to a multiple of 4,
                clamped to [4, native_height / binning]).

        Returns:
            dict: The delivered size ``{'width': int, 'height': int}`` after
                snapping and clamping, so the caller knows what was actually
                applied without a read-back.
        """
        # A geometry change stops the stream, applies, and restarts it, as it
        # does on a real body (pixel format and binning below do the same), so
        # no frame made under the old setting is stored after the change.
        with self.update_camera_config(), self._lock:
            max_w = self._native_width // self._binning
            max_h = self._native_height // self._binning
            self._width = max(48, min(max_w, int(w / 48) * 48))
            self._height = max(4, min(max_h, int(h / 4) * 4))
            if _cam_log is not None:
                _cam_log.info(f'sim set_frame_size({self._width}x{self._height})')
            return {'width': self._width, 'height': self._height}

    def get_min_frame_size(self) -> dict:
        """Return the simulator's minimum supported frame size.

        Returns:
            dict: ``{'width': 48, 'height': 4}``.
        """
        return {'width': 48, 'height': 4}

    def get_max_frame_size(self) -> dict:
        """Return the maximum frame size at the current binning.

        Post-binning ceiling = native sensor size / binning, matching Pylon's
        binning-dependent ``Width.Max`` / ``Height.Max``.

        Returns:
            dict: ``{'width': int, 'height': int}``.
        """
        return {
            'width': self._native_width // self._binning,
            'height': self._native_height // self._binning,
        }

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
            if _cam_log is not None:
                _cam_log.error(f'sim set_pixel_format({pixel_format}) UNSUPPORTED')
            logger.error(f'[CAM Sim   ] Unsupported pixel format: {pixel_format}')
            return False
        with self.update_camera_config(), self._lock:
            self._pixel_format = pixel_format
            if _cam_log is not None:
                _cam_log.info(f'sim set_pixel_format({pixel_format})')
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
    def exposure_t(self, exposure_ms: float) -> float | bool:
        """Set exposure time in milliseconds, returning the microseconds
        actually in effect.

        Refuses (``False``) rather than silently no-opping when the
        simulator is inactive or the request exceeds ``max_exposure``: the
        hardware value does not move on either path, so a caller that treated
        those as applied would record a chunk-match target the simulator never
        stamps.

        Args:
            exposure_ms: Exposure time in milliseconds.
        """
        if not self.active:
            return False
        if exposure_ms > self.max_exposure:
            if _cam_log is not None:
                _cam_log.warning(
                    f'sim ExposureTime.SetValue({exposure_ms}ms) CLAMPED max={self.max_exposure}ms'
                )
            logger.warning(
                f'[CAM Sim   ] Exposure {exposure_ms}ms exceeds max ({self.max_exposure}ms)'
            )
            return False
        with self._lock:
            self._exposure_us = float(exposure_ms) * 1000.0
            if _cam_log is not None:
                _cam_log.info(
                    f'sim ExposureTime.SetValue({float(exposure_ms) * 1000.0:.0f}us) (={exposure_ms}ms)'
                )
            logger.debug(f'[CAM Sim   ] Exposure set to {exposure_ms}ms')
            return self._exposure_us

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
            if _cam_log is not None:
                _cam_log.info(f'sim set_max_acquisition_frame_rate(enabled={enabled}, fps={fps})')

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
            if _cam_log is not None:
                _cam_log.error(f'sim set_binning_size({size}) UNSUPPORTED')
            logger.error(f'[CAM Sim   ] Unsupported bin size: {size}')
            return False
        with self.update_camera_config(), self._lock:
            self._binning = size
            # Frame is in post-binning pixels, so a larger binning shrinks the
            # post-binning ceiling (native / binning); clamp the current frame
            # down to it, the same way Pylon clamps Width to the reduced
            # Width.Max. The UI re-pushes set_frame_size after a binning change,
            # but isolated binning changes (no frame re-push) still stay valid.
            self._width = min(self._width, self._native_width // size)
            self._height = min(self._height, self._native_height // size)
            if _cam_log is not None:
                _cam_log.info(f'sim set_binning_size({size})')
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

        # Grid of fine lines (high frequency -- most sensitive to defocus)
        grid_spacing = 8
        img[::grid_spacing, :] = max_val * 0.4
        img[:, ::grid_spacing] = max_val * 0.4

        # Scattered bright spots (simulates point-like features)
        rng = np.random.RandomState(42)  # deterministic
        n_spots = max(20, (h * w) // 5000)
        ys = rng.randint(0, h, n_spots)
        xs = rng.randint(0, w, n_spots)
        for y, x in zip(ys, xs, strict=False):
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

    def _render_cycle_frame(self, h, w, dtype, max_val, brightness) -> np.ndarray:
        """Render the current specimen frame, resized and scaled to ``dtype``.

        Shared by the explicit ``image_cycle`` pattern and the
        no-pattern-requested fallback so the two cannot drift. A second copy
        of this is how the uint16 scaling below gets left out: without it a
        Mono10/Mono12 frame renders nearly black, because the source tops out
        at 255 against a max_val of 1023 or 4095.

        Frames are built on demand -- both callers are reachable before
        ``load_cycle_images()`` has run, and a camera-init failure skips that
        call entirely.
        """
        if not self._cycle_images:
            self._cycle_images = specimen_frames(h, w)
        src = self._cycle_images[self._cycle_index % len(self._cycle_images)]
        self._cycle_index += 1
        # Resize if binning changed since load -- nearest-neighbor via slicing
        if src.shape != (h, w):
            src_h, src_w = src.shape
            y_idx = np.linspace(0, src_h - 1, h, dtype=int)
            x_idx = np.linspace(0, src_w - 1, w, dtype=int)
            src = src[np.ix_(y_idx, x_idx)]
        # Floor so the field stays visible at the short default exposures.
        # Zero is exempt and must stay zero: it means no light reached the
        # sample, and a floor that lifts darkness back to a visible field
        # is what made an unlit capture indistinguishable from a lit one.
        if brightness > 0.0:
            brightness = max(0.5, brightness)
        if dtype == np.uint16:
            return (src.astype(np.float32) / 255.0 * max_val * brightness).astype(dtype)
        return (src.astype(np.float32) * brightness).clip(0, max_val).astype(dtype)

    def _illumination_scale(self) -> float:
        """1.0 when the sample is lit, 0.0 when nothing is.

        A camera sees what the illumination gives it, so an unlit field is
        black however long the exposure or however high the gain. Without
        this the simulator could not produce a dark frame at all, and the
        whole class of illumination failures -- an LED that did not come
        on, a channel left dark through a run -- was reproducible only on
        a bench.

        Unwired, the answer is 1.0: a camera built with no scope around it
        has no illumination to ask about, and rendering it black would
        make every standalone camera test describe a fault.

        Intensity is deliberately not modelled. The question this answers
        is whether there is light, which is what distinguishes a failure
        from a capture; how bright a lit field looks already follows
        exposure and gain.
        """
        if self._illumination_func is None:
            return 1.0
        return 1.0 if self._illumination_func() > 0.0 else 0.0

    def _generate_image(self) -> np.ndarray:
        """Generate a synthetic image based on current settings."""
        h = self._height
        w = self._width

        if self._pixel_format in ('Mono10', 'Mono12'):
            dtype = np.uint16
            max_val = 4095 if self._pixel_format == 'Mono12' else 1023
        else:
            dtype = np.uint8
            max_val = 255

        # Scale brightness by exposure and gain
        raw = (self._exposure_us / 1_000_000.0) * max(1.0, self._gain) * 10.0
        brightness = min(1.0, raw) * self._illumination_scale()
        if self._test_pattern == 'image_cycle':
            img = self._render_cycle_frame(h, w, dtype, max_val, brightness)
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
            # No specific pattern was asked for. Render the same specimen field
            # the cycle uses rather than a column ramp: a ramp is not what a
            # sample looks like, and reaching this branch used to undo the
            # cycle for the rest of the session. Built lazily because this
            # branch is reachable before load_cycle_images() has run -- a
            # camera-init failure skips that call entirely.
            if self._test_pattern != 'specimen':
                logger.warning(
                    f'[SimCamera ] Unknown test pattern '
                    f'{self._test_pattern!r} -- rendering the specimen field'
                )
            img = self._render_cycle_frame(h, w, dtype, max_val, brightness)

        return img

    def grab_new_capture(self, timeout_s: float) -> tuple:
        """Wait for a frame stored after this call, then return it.

        The acquisition thread is the only producer, as a real camera's SDK
        thread is, so a still capture waits for the stream rather than making
        a frame of its own: the frame it gets is one the stream delivered,
        counted once, under the settings in effect when it was made.

        Args:
            timeout_s: Wall-clock seconds to wait for the next frame.

        Returns:
            tuple: ``(success: bool, timestamp: datetime | None,
                seq: int | None)``. ``success=False`` when the camera is not
                grabbing or no frame arrived within ``timeout_s``.
        """
        handler = self.cam_image_handler
        if not self._grabbing or handler is None:
            return False, None, None
        if not handler.wait_for_frame_after(handler.frames_delivered, timeout_s):
            return False, None, None
        return self.grab()

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

    def gain(self, value: float) -> float | bool | None:
        """Set the simulated camera gain.

        Refuses (``False``) a value outside the range the profile declares,
        as a real body's gain node does, so a refused gain is reachable in
        the simulator and not only on a bench. The simulated gain does not
        move on a refusal.

        Args:
            value: Gain in dB.

        Returns:
            float | bool | None: See ``Camera.gain``.
        """
        if not self.active:
            return None
        low, high = self.min_gain, self.max_gain
        if (low is not None and float(value) < low) or float(value) > high:
            logger.warning(f'[CAM Sim   ] Gain {value} dB outside [{low}, {high}] dB')
            return False
        with self._lock:
            self._gain = float(value)
            if _cam_log is not None:
                _cam_log.info(f'sim Gain.SetValue({float(value):.3f})')
            logger.debug(f'[CAM Sim   ] Gain set to {value}')
        return float(value)

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

    def _converged_gain(self) -> float:
        """Return the gain the simulated auto-gain loop settles on.

        The midpoint of the auto-gain bounds, held inside the range the
        gain node declares: a real body's auto loop drives the same node a
        manual write does and cannot leave its range, so a caller whose
        bounds reach past the profile (a ``current.json`` carried over from
        a body with a higher ceiling) still gets a gain this camera accepts
        when it writes it back.
        """
        midpoint = (self._auto_gain_min + self._auto_gain_max) / 2.0
        low = self.min_gain if self.min_gain is not None else midpoint
        return min(max(midpoint, low), self.max_gain)

    def auto_gain(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
    ) -> bool:
        """Enable or disable simulated continuous auto-gain.

        On enable, the simulator converges immediately by setting gain
        to the midpoint of [min_gain_db, max_gain_db].

        Args:
            state: True to enable, False to disable.
            target_brightness: Normalized brightness target (0.0-1.0).
            min_gain_db: Optional lower bound in dB.
            max_gain_db: Optional upper bound in dB.
            ae_max_exposure_ms: Accepted for interface parity; the
                simulator has no auto-exposure bound to apply.

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
                self._gain = self._converged_gain()
            if _cam_log is not None:
                _cam_log.info(
                    f'sim auto_gain(state={state}, target={target_brightness}, min_db={min_gain_db}, max_db={max_gain_db})'
                )
        return True

    def auto_gain_once(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
    ) -> bool:
        """Run a single simulated auto-gain iteration.

        Converges on the midpoint of [min_gain_db, max_gain_db], held inside
        the gain node's range (see ``_converged_gain``).

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
                self._gain = self._converged_gain()
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

    def update_auto_gain_min_max(
        self, min_gain_db: float | None, max_gain_db: float | None
    ) -> bool:
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
            enabled: True to enable the pattern, False to revert to the
                specimen field the live view normally shows.
            pattern: Pattern name (case-insensitive). Common values:
                ``'black'``, ``'white'``, ``'noise'``, ``'focus_target'``,
                ``'image_cycle'``.
        """
        if enabled:
            self._test_pattern = pattern.lower()
        else:
            # Back to the specimen field, not a static ramp: turning a test
            # pattern OFF must restore what the view showed before it went on.
            self._test_pattern = 'specimen'
