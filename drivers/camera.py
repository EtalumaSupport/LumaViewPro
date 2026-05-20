# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
from abc import ABC, abstractmethod
import contextlib
import threading

import numpy as np

from lvp_logger import logger
try:
    from lvp_logger import camera_logger as _cam_log
except ImportError:
    _cam_log = None
from drivers.camera_profiles import CameraProfile, lookup_profile

default_max_exposure = 1_000 # in ms


class ImageHandlerBase:
    """Base class for camera image handlers (IDS and Pylon).

    Provides thread-safe frame buffer storage, copy-on-read, failure counting
    with auto-stop, and a consistent API for the Camera.grab() method.
    """

    MAX_CONSECUTIVE_FAILURES = 128

    def __init__(self):
        self._frame_lock = threading.Lock()
        self.last_result = False
        self.last_img = None
        self.last_img_ts = None
        self.last_chunks = None  # per-frame chunk metadata dict (None when unsupported)
        self._failed_grabs = 0
        # Per-frame consumers (manual record today; per-frame plugins later).
        # Snapshotted-then-released under _frame_lock at _store_frame time so
        # a slow callback never holds the SDK thread.
        self._frame_callbacks: list = []

    def get_last_image(self):
        """Return (success, image, timestamp). Thread-safe.

        No copy needed here — the stored frame is already a copy from the SDK
        callback (GetArray().copy() in Pylon, copy() in IDS). _store_frame()
        replaces the reference (not in-place), so the returned array remains
        valid even after the next frame arrives.
        """
        with self._frame_lock:
            if not self.last_result:
                return False, None, None
            return True, self.last_img, self.last_img_ts

    def get_last_image_with_chunks(self):
        """Return (success, image, timestamp, chunks). Atomic snapshot.

        Chunks are stored alongside the frame in `_store_frame` under the
        same lock, so reading both fields under one lock acquisition
        guarantees they came from the same frame. Without this, a caller
        that does `get_last_image()` followed by `get_last_chunks()` can
        observe image-N paired with chunks-N+1 if `_store_frame` runs in
        between (the camera thread grabs frames concurrently with the
        consumer thread). Used by the manual-record path so per-frame
        TIFF metadata is correct.
        """
        with self._frame_lock:
            if not self.last_result:
                return False, None, None, None
            return True, self.last_img, self.last_img_ts, self.last_chunks

    def get_last_chunks(self) -> dict | None:
        """Return per-frame chunk metadata for the most recent successful grab.

        Cameras that support GenICam chunk data populate this dict in their
        ImageHandler grab callback. Cameras without chunk support return
        None (default).

        Returned dict keys are GenICam attribute symbolic names
        ('ExposureTime', 'Gain', 'FrameID' on Basler USB3; 'ExposureTime' /
        'Gain' on IDS Peak which lacks ChunkFrameID). Values are floats /
        ints as reported by the camera. Returns None when no successful
        grab has occurred yet.
        """
        with self._frame_lock:
            if not self.last_result:
                return None
            return self.last_chunks

    def reset(self):
        """Clear frame buffer and failure counter."""
        with self._frame_lock:
            self.last_result = False
            self.last_img = None
            self.last_img_ts = None
            self.last_chunks = None
        self._failed_grabs = 0

    def register_frame_callback(self, cb) -> None:
        """Register a per-frame callback fired after every successful grab.

        Callback signature: ``cb(image, timestamp, chunks)``. Runs on the
        SDK callback thread (Pylon ``PylonImageGrab`` / IDS grab loop /
        simulated pump). Callbacks MUST NOT block -- they share the camera
        ingest thread with the next frame. Heavy work (file IO, image
        conversion) belongs on an executor; the callback's job is fast
        decision + enqueue.

        Registration is idempotent for the same callable.
        """
        with self._frame_lock:
            if cb not in self._frame_callbacks:
                self._frame_callbacks.append(cb)

    def unregister_frame_callback(self, cb) -> None:
        """Remove a callback registered via ``register_frame_callback``.

        No-op when ``cb`` is not currently registered.
        """
        with self._frame_lock, contextlib.suppress(ValueError):
            self._frame_callbacks.remove(cb)

    def _store_frame(self, image, timestamp, chunks: dict | None = None):
        """Called by subclass when a new frame is successfully grabbed.

        Args:
            image: numpy array (already copied from SDK buffer).
            timestamp: datetime when the frame arrived host-side.
            chunks: optional per-frame chunk metadata dict. None for cameras
                without chunk support; backward-compatible with callers that
                don't pass it.
        """
        with self._frame_lock:
            self.last_result = True
            self.last_img = image
            self.last_img_ts = timestamp
            self.last_chunks = chunks
            cbs = list(self._frame_callbacks)
        self._failed_grabs = 0
        # Snapshot under lock + invoke outside: a callback that takes >0
        # microseconds never extends the SDK thread's lock hold past the
        # storage write. One failing callback can't block its peers.
        for cb in cbs:
            try:
                cb(image, timestamp, chunks)
            except Exception as e:
                _cam_log.exception(f'[CAM Class ] frame callback raised: {e}')

    def _record_failure(self):
        """Called by subclass when a grab fails.

        Returns True if the failure count has reached MAX_CONSECUTIVE_FAILURES,
        indicating the caller should stop grabbing.
        """
        with self._frame_lock:
            self.last_result = False
        self._failed_grabs += 1
        if self._failed_grabs % 5 == 1:
            _cam_log.warning(f'[CAM Class ] Grab failed ({self._failed_grabs} consecutive)')
        return self._failed_grabs >= self.MAX_CONSECUTIVE_FAILURES


class Camera(ABC):
    def __init__(self):
        self._state_lock = threading.Lock()
        self._array_lock = threading.Lock()
        # CAM-3: serializes the entire stop/yield/start critical section
        # of update_camera_config() so two threads can't both be inside it
        # at once. update_camera_config() can yield arbitrarily long
        # configuration work (set_pixel_format, set_frame_size,
        # init_camera_config), so this is a separate lock from
        # _state_lock — _state_lock holds for ms, _lifecycle_lock can
        # hold for seconds.
        self._lifecycle_lock = threading.RLock()
        self._active = False
        self.array = np.array([])
        self.cam_image_handler: ImageHandlerBase | None = None
        self.model_name = None
        self._device_removed = False
        self._device_serial = None
        # Camera-side timestamp tick rate (Hz). Set by the driver at init
        # if the camera supports a Timestamp chunk; None for cameras
        # without chunk timestamps (downstream code skips per-frame
        # camera-tick metadata when None).
        self.timestamp_tick_frequency_hz: int | None = None
        self.profile: CameraProfile = CameraProfile()
        # Re-entrancy depth for ``update_camera_config()`` (CAM-4).
        # Protected by ``_state_lock``; only the outermost level
        # toggles the grab loop.
        self._update_config_depth = 0

        self.connect()
        # Registry contract: drivers signal "I couldn't find my hardware"
        # via `found=False`, and `drivers/registry.py::create('auto')` skips
        # such instances and tries the next candidate. PylonCamera and
        # IDSCamera both catch their connect-failure exception internally
        # and set `self.active = None` without raising — without this line,
        # the registry sees no exception and `getattr(instance, 'found', True)`
        # defaults to True, so the broken Pylon instance is returned and
        # FX2 (priority 80) never gets a turn. Discovered 2026-04-15 trying
        # to bring up an LS620 through LVP for the first time. The
        # `_active not in (False, None)` check matches `Camera.active`'s
        # three-state semantics (False=initial, <obj>=connected, None=disconnected).
        self.found = self._active not in (False, None)

    @property
    def active(self):
        """Thread-safe access to camera active state.

        Three-state semantics:
          False  -- not connected (initial state)
          <obj>  -- connected camera instance (truthy; e.g. pylon.InstantCamera)
          None   -- disconnected / device removed (set by _mark_disconnected)

        Returns:
            False, the connected camera instance, or None.
        """
        with self._state_lock:
            return self._active

    @active.setter
    def active(self, value) -> None:
        """Set the active-state value under the state lock."""
        with self._state_lock:
            self._active = value

    def __del__(self):
        try:
            with self._state_lock:
                is_active = bool(self._active)
            if is_active:
                self.disconnect()
        except Exception as e:
            _cam_log.warning(f'[CAM Class ] __del__ disconnect failed: {e}')

    def is_device_removed(self) -> bool:
        """Return whether the camera was marked as physically removed.

        Returns:
            bool: True after ``_mark_disconnected`` has been invoked.
        """
        with self._state_lock:
            return self._device_removed

    def _mark_disconnected(self):
        """Atomically mark camera as disconnected.

        Sets both flags together to avoid inconsistent state.
        Safe to call from any thread (including SDK callbacks).
        """
        was_connected = False
        with self._state_lock:
            was_connected = self._active is not None and not self._device_removed
            self._device_removed = True
            self._active = None
        if was_connected:
            _cam_log.error('[CAM Class ] Camera disconnected')

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the camera hardware.

        Subclasses must catch their own connect-failure exceptions and
        leave ``self._active`` as False or None on failure (the registry
        uses ``self.found`` to skip non-functional drivers).

        Returns:
            bool: True on success.
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the camera hardware and release resources.

        Returns:
            bool: True on success.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Return whether the camera is currently connected.

        Returns:
            bool: True if the SDK reports a live, attached camera.
        """
        pass

    @contextlib.contextmanager
    def update_camera_config(self):
        """Cross-thread-safe, re-entrant guard around the camera grab loop.

        Combines the CAM-3 (cross-thread serialization) and CAM-4
        (nested-from-same-thread) fixes into a single mechanism:

          - ``_lifecycle_lock`` (RLock) holds for the full
            stop/yield/start critical section so two threads can't both
            be mutating the grab loop simultaneously, and the same
            thread can re-enter without deadlock.
          - ``_update_config_depth`` counts the nesting level so only
            the OUTERMOST call toggles the grab loop. Inner re-entries
            are no-ops on the SDK side. Counter mutates only while we
            hold the RLock, so no separate lock is required.
          - ``camera.log`` :enter / :exit lines emit on every level so
            nested patterns (init_camera_config wrapping
            set_pixel_format) stay visible in diagnostic logs.

        Smoke 3 camera.log 2026-04-30 captured both failure modes that
        this method now covers.
        """
        try:
            from lvp_logger import camera_logger as _cam_log
        except Exception:
            _cam_log = None
        with self._lifecycle_lock:
            self._update_config_depth += 1
            depth = self._update_config_depth
            was_grabbing = False
            if _cam_log is not None:
                _cam_log.info(f'update_camera_config:enter depth={depth}')
            try:
                if depth == 1:
                    was_grabbing = self.is_grabbing()
                    if was_grabbing:
                        self.stop_grabbing()
                yield
            finally:
                self._update_config_depth -= 1
                end_depth = self._update_config_depth
                if end_depth == 0 and was_grabbing:
                    self.start_grabbing()
                if _cam_log is not None:
                    _cam_log.info(
                        f'update_camera_config:exit depth={depth} '
                        f'restarted={was_grabbing and end_depth == 0}'
                    )

    @abstractmethod
    def init_camera_config(self) -> None:
        """Apply the camera's startup configuration after connect().

        Subclasses set pixel format, default frame size, exposure, gain,
        binning, etc. Called inside ``update_camera_config()`` so the
        grab loop is paused while configuration is in progress.
        """
        pass

    @abstractmethod
    def start_grabbing(self) -> None:
        """Begin acquiring frames into the image handler."""
        pass

    @abstractmethod
    def stop_grabbing(self) -> None:
        """Stop acquiring frames and release any pending buffers."""
        pass

    @abstractmethod
    def is_grabbing(self) -> bool:
        """Return whether the camera is currently acquiring.

        Returns:
            bool: True when the SDK reports an active grab loop.
        """
        pass

    @abstractmethod
    def set_frame_size(self, w: int, h: int) -> bool:
        """Set the output frame size.

        Args:
            w: Frame width in pixels.
            h: Frame height in pixels.

        Returns:
            bool: True on success.
        """
        pass

    @abstractmethod
    def get_min_frame_size(self) -> dict:
        """Return the minimum supported frame size.

        Returns:
            dict: ``{'width': int, 'height': int}``.
        """
        pass

    @abstractmethod
    def get_max_frame_size(self) -> dict:
        """Return the maximum supported frame size.

        Returns:
            dict: ``{'width': int, 'height': int}``.
        """
        pass

    @abstractmethod
    def get_frame_size(self) -> dict:
        """Return the current frame size.

        Returns:
            dict: ``{'width': int, 'height': int}``.
        """
        pass

    @abstractmethod
    def set_pixel_format(self, pixel_format: str) -> bool:
        """Set the camera pixel format.

        Args:
            pixel_format: Format identifier (e.g. ``'Mono8'``,
                ``'Mono12'``).

        Returns:
            bool: True on success.
        """
        pass

    @abstractmethod
    def get_pixel_format(self) -> str:
        """Return the current camera pixel format.

        Returns:
            str: Format identifier (e.g. ``'Mono8'``).
        """
        pass

    @abstractmethod
    def get_supported_pixel_formats(self) -> tuple:
        """Return all pixel formats this camera can be set to.

        Returns:
            tuple: Format identifier strings supported by the SDK.
        """
        pass

    @abstractmethod
    def exposure_t(self, exposure_ms: float) -> None:
        """Set exposure time.

        Args:
            exposure_ms: Exposure time in milliseconds.
        """
        pass

    @abstractmethod
    def get_exposure_t(self) -> float:
        """Return the current exposure time.

        Returns:
            float: Exposure time in milliseconds.
        """
        pass

    @abstractmethod
    def auto_exposure_t(self, state: bool = True) -> None:
        """Enable or disable hardware auto-exposure.

        Args:
            state: True to enable, False to disable.
        """
        pass

    def get_model_name(self) -> str | None:
        """Return the cached camera model name.

        Returns:
            str | None: Cached model identifier, or None if not yet
                discovered.
        """
        return self.model_name

    @abstractmethod
    def get_all_temperatures(self) -> dict:
        """Read all available camera temperature sensors.

        Returns:
            dict: Sensor-name-keyed temperatures in degrees Celsius.
                Empty dict when the camera does not expose temperature
                telemetry.
        """
        pass

    def _load_profile(self):
        """Load the camera profile based on model_name.

        Called by subclass connect() after model_name is known. Subclasses
        should then call _query_dynamic_capabilities() to populate
        SDK-queried fields (gain min/max, exposure min/max). Per Rule 2,
        `profile.exposure_max_us` is the single source of truth for the
        max-exposure cap — `Camera.max_exposure` is a derived property
        that reads from it.
        """
        self.profile = lookup_profile(self.model_name)
        logger.info(f'[CAM Class ] Loaded profile: {self.profile.model_name} '
                     f'(sensor={self.profile.sensor}, driver={self.profile.driver})')

    def _query_dynamic_capabilities(self):
        """Query SDK for dynamic values and merge into profile.

        Subclasses should override to query gain min/max, exposure min/max,
        etc. from the camera SDK. The base implementation is a no-op.
        """
        pass

    @property
    def max_exposure(self) -> float:
        """Maximum exposure cap in milliseconds.

        Derived from `profile.exposure_max_us` — the single source of
        truth (Rule 2). The profile's value is the sensor-datasheet
        ceiling by default and may be overwritten by
        `_query_dynamic_capabilities()` at connect time with an SDK-
        queried or driver-narrowed cap (e.g. FX2's 178 ms safe-frame
        ceiling).
        """
        if self.profile and self.profile.exposure_max_us:
            return self.profile.exposure_max_us / 1000.0
        return float(default_max_exposure)

    def get_max_exposure(self) -> float:
        """Return the maximum exposure cap in milliseconds.

        Returns:
            float: Same value as the ``max_exposure`` property.
        """
        return self.max_exposure

    @property
    def max_gain(self) -> float:
        """Maximum gain cap in dB.

        Derived from `profile.gain.total_max_db` — the single source of
        truth (Rule 2). The profile's value is the sensor-datasheet
        ceiling by default and may be overwritten by
        `_query_dynamic_capabilities()` at connect time (Pylon / IDS
        live-query their SDK; FX2 hardcodes the MT9P031 value).
        """
        if self.profile and self.profile.gain and self.profile.gain.total_max_db is not None:
            return float(self.profile.gain.total_max_db)
        return 48.0  # legacy kv default — kept for cameras without a profile

    def get_max_gain(self) -> float:
        """Return the maximum gain cap in dB.

        Returns:
            float: Same value as the ``max_gain`` property.
        """
        return self.max_gain

    @abstractmethod
    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float = 1.0) -> None:
        """Enable or disable the SDK's frame-rate cap.

        Args:
            enabled: True to enforce ``fps`` as the upper bound.
            fps: Cap value in frames per second.
        """
        pass

    @abstractmethod
    def set_binning_size(self, size: int) -> bool:
        """Set hardware binning factor.

        Args:
            size: Binning factor (1, 2, 4, ...).

        Returns:
            bool: True on success.
        """
        pass

    @abstractmethod
    def get_binning_size(self) -> int:
        """Return the current hardware binning factor.

        Returns:
            int: Binning factor (1 = no binning).
        """
        pass

    def grab(self) -> tuple:
        """Grab the most recent frame from the image handler.

        On success, the image is also stored in ``self.array``.

        Returns:
            tuple: ``(success: bool, timestamp: datetime | None)``.
        """
        with self._state_lock:
            if self._active is None or self._device_removed:
                return False, None

        if not self.cam_image_handler:
            return False, None

        try:
            result, image, image_ts = self.cam_image_handler.get_last_image()
            if not result:
                return False, None

            with self._array_lock:
                self.array = image
            return True, image_ts
        except Exception as ex:
            _cam_log.exception(f"[CAM Class ] grab() - get_last_image() failed: {ex}")
            return False, None

    def get_array(self) -> np.ndarray:
        """Return a copy of the last grabbed image. Thread-safe.

        Returns:
            np.ndarray: Copy of the most recent frame, or an empty
                array when no frame has been grabbed yet.
        """
        with self._array_lock:
            return self.array.copy() if self.array.size > 0 else self.array

    def grab_latest(self) -> tuple:
        """Grab the latest frame and return it in one operation (single copy).

        Combines grab() + get_array() but avoids the second copy.
        The returned image is already a copy from the image handler,
        safe to use without further copying.

        Returns:
            tuple: ``(success: bool, image: np.ndarray | None,
                timestamp: datetime | None)``.
        """
        with self._state_lock:
            if self._active is None or self._device_removed:
                return False, None, None

        if not self.cam_image_handler:
            return False, None, None

        try:
            result, image, image_ts = self.cam_image_handler.get_last_image()
            if not result or image is None:
                return False, None, None

            # Store for other consumers (e.g. recording), but the returned
            # image IS the copy — callers don't need get_array().
            with self._array_lock:
                self.array = image
            return True, image, image_ts
        except Exception as ex:
            _cam_log.exception(f"[CAM Class ] grab_latest() failed: {ex}")
            return False, None, None

    def register_frame_callback(self, cb) -> None:
        """Register a per-frame callback on the driver's image handler.

        Default implementation delegates to ``cam_image_handler``;
        drivers without a handler (SimulatedCamera) override.
        """
        if not self.cam_image_handler:
            return
        self.cam_image_handler.register_frame_callback(cb)

    def unregister_frame_callback(self, cb) -> None:
        """Unregister a callback registered via ``register_frame_callback``.

        Default implementation delegates to ``cam_image_handler``;
        drivers without a handler (SimulatedCamera) override.
        """
        if not self.cam_image_handler:
            return
        self.cam_image_handler.unregister_frame_callback(cb)

    def grab_latest_with_chunks(self) -> tuple:
        """Like grab_latest, plus an atomic snapshot of the per-frame chunks dict.

        Used by the manual-record path so per-frame TIFF metadata reflects
        the chunks captured at the same grab as the image. Cameras without
        chunk support return chunks=None; downstream metadata writers
        gracefully skip the chunk-derived fields.

        Returns:
            tuple: ``(success: bool, image: np.ndarray | None,
                timestamp: datetime | None, chunks: dict | None)``.
        """
        with self._state_lock:
            if self._active is None or self._device_removed:
                return False, None, None, None

        if not self.cam_image_handler:
            return False, None, None, None

        try:
            result, image, image_ts, chunks = (
                self.cam_image_handler.get_last_image_with_chunks()
            )
            if not result or image is None:
                return False, None, None, None
            with self._array_lock:
                self.array = image
            return True, image, image_ts, chunks
        except Exception as ex:
            _cam_log.exception(f"[CAM Class ] grab_latest_with_chunks() failed: {ex}")
            return False, None, None, None

    @abstractmethod
    def grab_new_capture(self, timeout_s: float) -> tuple:
        """Grab a fresh capture-quality frame, waiting if necessary.

        Used by the still-capture path to guarantee the returned frame
        was acquired after the call (not a stale live-preview frame).

        Args:
            timeout_s: Maximum wait in seconds.

        Returns:
            tuple: ``(success: bool, image: np.ndarray | None,
                timestamp: datetime | None)``.
        """
        pass

    @abstractmethod
    def update_auto_gain_target_brightness(self, auto_target_brightness: float) -> None:
        """Update the target brightness for the auto-gain loop.

        Args:
            auto_target_brightness: Normalized target brightness (0.0
                to 1.0).
        """
        pass

    @abstractmethod
    def update_auto_gain_min_max(self, min_gain_db: float | None, max_gain_db: float | None) -> None:
        """Update the auto-gain bounds.

        Args:
            min_gain_db: Minimum gain in dB, or None to leave unchanged.
            max_gain_db: Maximum gain in dB, or None to leave unchanged.
        """
        pass

    @abstractmethod
    def get_gain(self) -> float:
        """Return the current camera gain.

        Returns:
            float: Gain in dB.
        """
        pass

    @abstractmethod
    def gain(self, gain: float) -> None:
        """Set the camera gain.

        Args:
            gain: Gain in dB.
        """
        pass

    @abstractmethod
    def auto_gain(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None
    ) -> None:
        """Enable or disable continuous auto-gain.

        Args:
            state: True to enable, False to disable.
            target_brightness: Normalized brightness target (0.0-1.0).
            min_gain_db: Optional lower bound in dB.
            max_gain_db: Optional upper bound in dB.
        """
        pass

    @abstractmethod
    def auto_gain_once(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None
    ) -> None:
        """Run a single auto-gain iteration.

        Args:
            state: True to run, False to no-op.
            target_brightness: Normalized brightness target (0.0-1.0).
            min_gain_db: Optional lower bound in dB.
            max_gain_db: Optional upper bound in dB.
        """
        pass

    @abstractmethod
    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black') -> None:
        """Enable or disable the SDK's test pattern generator.

        Args:
            enabled: True to enable the pattern, False to disable.
            pattern: Pattern name (SDK-specific; e.g. ``'Black'``).
        """
        pass
