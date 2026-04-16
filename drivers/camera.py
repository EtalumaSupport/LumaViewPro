# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
from abc import ABC, abstractmethod
import contextlib
import threading

import numpy as np

from lvp_logger import logger
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
        self._failed_grabs = 0

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

    def reset(self):
        """Clear frame buffer and failure counter."""
        with self._frame_lock:
            self.last_result = False
            self.last_img = None
            self.last_img_ts = None
        self._failed_grabs = 0

    def _store_frame(self, image, timestamp):
        """Called by subclass when a new frame is successfully grabbed."""
        with self._frame_lock:
            self.last_result = True
            self.last_img = image
            self.last_img_ts = timestamp
        self._failed_grabs = 0

    def _record_failure(self):
        """Called by subclass when a grab fails.

        Returns True if the failure count has reached MAX_CONSECUTIVE_FAILURES,
        indicating the caller should stop grabbing.
        """
        with self._frame_lock:
            self.last_result = False
        self._failed_grabs += 1
        if self._failed_grabs % 5 == 1:
            logger.warning(f'[CAM Class ] Grab failed ({self._failed_grabs} consecutive)')
        return self._failed_grabs >= self.MAX_CONSECUTIVE_FAILURES


class Camera(ABC):
    def __init__(self):
        self._state_lock = threading.Lock()
        self._array_lock = threading.Lock()
        self._active = False
        self.error_report_count = 0
        self.array = np.array([])
        self.cam_image_handler: ImageHandlerBase | None = None
        self.model_name = None
        self._device_removed = False
        self._device_serial = None
        self.profile: CameraProfile = CameraProfile()

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
        """
        with self._state_lock:
            return self._active

    @active.setter
    def active(self, value):
        with self._state_lock:
            self._active = value

    def __del__(self):
        try:
            with self._state_lock:
                is_active = bool(self._active)
            if is_active:
                self.disconnect()
        except Exception as e:
            logger.warning(f'[CAM Class ] __del__ disconnect failed: {e}')

    def is_device_removed(self) -> bool:
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
            logger.error('[CAM Class ] Camera disconnected')
            try:
                from modules.notification_center import notifications
                notifications.error("Camera", "Camera Disconnected",
                    "USB camera was removed. Reconnect and restart the app.")
            except Exception:
                pass  # Notification system may not be available during shutdown

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @contextlib.contextmanager
    def update_camera_config(self):
        was_grabbing = self.is_grabbing()

        if was_grabbing:
            self.stop_grabbing()

        try:
            yield
        finally:
            if was_grabbing:
                self.start_grabbing()

    @abstractmethod
    def init_camera_config(self):
        pass

    @abstractmethod
    def start_grabbing(self):
        pass

    @abstractmethod
    def stop_grabbing(self):
        pass

    @abstractmethod
    def is_grabbing(self):
        pass

    @abstractmethod
    def set_frame_size(self, w, h):
        pass

    @abstractmethod
    def get_min_frame_size(self) -> dict:
        pass

    @abstractmethod
    def get_max_frame_size(self) -> dict:
        pass

    @abstractmethod
    def get_frame_size(self):
        pass

    @abstractmethod
    def set_pixel_format(self, pixel_format: str) -> bool:
        pass

    @abstractmethod
    def get_pixel_format(self) -> str:
        pass

    @abstractmethod
    def get_supported_pixel_formats(self) -> tuple:
        pass

    @abstractmethod
    def exposure_t(self, t):
        pass

    @abstractmethod
    def get_exposure_t(self):
        pass

    @abstractmethod
    def auto_exposure_t(self, state = True):
        pass

    @abstractmethod
    def find_model_name(self):
        pass

    def get_model_name(self):
        return self.model_name

    @abstractmethod
    def get_all_temperatures(self):
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

    def get_max_exposure(self):
        return self.max_exposure

    @abstractmethod
    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float=1.0):
        pass

    @abstractmethod
    def set_binning_size(self, size: int) -> bool:
        pass

    @abstractmethod
    def get_binning_size(self) -> int:
        pass

    def grab(self):
        """Grab the most recent frame from the image handler.

        Returns (success: bool, timestamp: datetime | None).
        The image is available in self.array after a successful grab.
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
            logger.exception(f"[CAM Class ] grab() - get_last_image() failed: {ex}")
            return False, None

    def get_array(self):
        """Return a copy of the last grabbed image. Thread-safe."""
        with self._array_lock:
            return self.array.copy() if self.array.size > 0 else self.array

    def grab_latest(self):
        """Grab the latest frame and return it in one operation (single copy).

        Combines grab() + get_array() but avoids the second copy.
        The returned image is already a copy from the image handler,
        safe to use without further copying.

        Returns:
            (success: bool, image: np.ndarray | None, timestamp: datetime | None)
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
            logger.exception(f"[CAM Class ] grab_latest() failed: {ex}")
            return False, None, None

    @abstractmethod
    def grab_new_capture(self, timeout: float):
        pass

    @abstractmethod
    def update_auto_gain_target_brightness(self, auto_target_brightness: float):
        pass

    @abstractmethod
    def update_auto_gain_min_max(self, min_gain: float | None, max_gain: float | None):
        pass

    @abstractmethod
    def get_gain(self):
        pass

    @abstractmethod
    def gain(self, gain):
        pass

    @abstractmethod
    def auto_gain(
        self,
        state = True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None
    ):
        pass

    @abstractmethod
    def auto_gain_once(
        self,
        state = True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None
    ):
        pass

    @abstractmethod
    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black'):
        pass
