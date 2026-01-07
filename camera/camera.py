from abc import ABC, abstractmethod
import numpy as np
import contextlib
from lvp_logger import logger

class Camera(ABC):
    def __init__(self):
        logger.info('[CAM Class ] Camera.__init__()')
        self.active = False
        self.error_report_count = 0
        self.array = np.array([])
        self.cam_image_handler = None
        self.model_name = None
        self.max_exposure = 100 # in ms
        self._device_removed = False
        self._device_serial = None

        self.max_exposure_dict = {
            "daA3840-45um": 1_000,
            "a2A3536-31umBAS": 10_000
        }

        self.connect()

    @abstractmethod
    def connect(self):
        self.init_camera_config()
        self.start_grabbing()

    @contextlib.contextmanager
    def update_camera_config(self):
        was_grabbing = self.is_grabbing()

        if was_grabbing:
            self.stop_grabbing()

        yield

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