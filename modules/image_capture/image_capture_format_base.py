

from abc import ABC, abstractmethod
import logging

import modules.image_capture.image_capture_enums as image_capture_enums


class ImageCaptureFormatBase(ABC):

    def __init__(self, name: str):
        self._name = name
        self._logger = logging.getLogger(self._name)


    def supported_configs() -> tuple[image_capture_enums.ImageCaptureConfig]:
        return tuple()
    
    
    @abstractmethod
    def save(
        self,
        image_data,
    ):
        raise NotImplementedError
