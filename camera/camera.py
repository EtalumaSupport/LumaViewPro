from abc import ABC, abstractmethod
import numpy as np
import contextlib
from lvp_logger import logger

default_max_exposure = 1_000 # in ms

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
    def connect(self) -> bool:
        self.init_camera_config()
        self.start_grabbing()

    @abstractmethod
    def disconnect(self) -> bool:
        pass

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
    
    @abstractmethod
    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float=1.0):
        pass

    @abstractmethod
    def set_binning_size(self, size: int) -> bool:
        pass

    @abstractmethod
    def get_binning_size(self) -> int:
        pass