# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
from abc import ABC, abstractmethod
import contextlib
import threading

import numpy as np

from lvp_logger import logger

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
        """Return (success, image_copy, timestamp). Thread-safe, copy-on-read."""
        with self._frame_lock:
            if not self.last_result:
                return False, None, None
            img = self.last_img.copy() if self.last_img is not None else None
            return True, img, self.last_img_ts

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
        self.active = False
        self.error_report_count = 0
        self.array = np.array([])
        self.cam_image_handler: ImageHandlerBase | None = None
        self.model_name = None
        self.max_exposure = 100 # in ms
        self._device_removed = False
        self._device_serial = None

        self.max_exposure_dict = self._get_max_exposure_models()

        self.connect()

    def __del__(self):
        try:
            if self.active:
                self.disconnect()
        except Exception:
            pass

    def _mark_disconnected(self):
        """Atomically mark camera as disconnected.

        Sets both flags together to avoid inconsistent state.
        Safe to call from any thread (including SDK callbacks).
        """
        self._device_removed = True
        self.active = None

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

    def set_max_exposure_time(self):
        found_key = None
        for key in self.max_exposure_dict.keys():
            if self.model_name in key:
                found_key = key
                break

        if found_key is None:
            self.max_exposure = default_max_exposure
            return

        self.max_exposure = self.max_exposure_dict[found_key]
        logger.info(f"[CAM Class ] Max exposure set to {self.max_exposure} ms")

    def get_max_exposure(self):
        return self.max_exposure

    def _get_max_exposure_models(self) -> dict:
        """Return a dict mapping model name substrings to max exposure (ms).

        Subclasses should override to register their known models.
        """
        return {}

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
        if self.active is None or self._device_removed:
            return False, None

        if not self.cam_image_handler:
            return False, None

        try:
            result, image, image_ts = self.cam_image_handler.get_last_image()
            if not result:
                return False, None

            self.array = image
            return True, image_ts
        except Exception as ex:
            logger.exception(f"Failed to grab image: {ex}")
            return False, None

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
