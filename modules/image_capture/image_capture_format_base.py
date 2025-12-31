

import abc
import logging
import pathlib

import numpy as np
import tifffile as tf

import image_utils
import modules.image_capture.image_capture_enums as image_capture_enums


class ImageCaptureFormatBase(abc.ABC):

    def __init__(self, name: str):
        self._name = name
        self._logger = logging.getLogger(self._name)


    def supported_configs() -> tuple[image_capture_enums.ImageCaptureConfig]:
        return tuple()
    
    
    def _lookup_photometric(
        self,
        image_data,
        color_channel: str,
    ) -> tf.PHOTOMETRIC:
        if image_utils.is_color_image(image_data):
            photometric = tf.PHOTOMETRIC.RGB
        if color_channel in ('BF', 'PC', 'DF'):
            photometric = tf.PHOTOMETRIC.MINISBLACK
        elif color_channel in ('Red', 'Green', 'Blue'):
            photometric = tf.PHOTOMETRIC.PALETTE
        else:
            raise ValueError(f"Unexpected color value ({color_channel}) for tiff data generation")
        
        return photometric
    

    @abc.abstractmethod
    def save(
        self,
        image_data: np.ndarray,
        file_loc: pathlib.Path,
        metadata: dict,
        color_channel: str,
    ):
        raise NotImplementedError
