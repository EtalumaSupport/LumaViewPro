# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import datetime
import pathlib

import cv2
import numpy as np

import modules.image_utils as image_utils
from lvp_logger import logger


class VideoWriter:

    CODECS = [
        0,
        'mp4v',
        'mjpg',
        'ffv1',
    ]

    def __init__(
        self,
        output_file_loc: pathlib.Path,
        fps: float,
        include_timestamp_overlay: bool,
        codec: str = 'mp4v',
    ):
        self._output_file_loc = output_file_loc
        # Use provided codec if valid, otherwise default to mp4v
        self._codec = codec if codec in self.CODECS else self.CODECS[1]
        self._fourcc = self._get_fourcc_code(codec=self._codec)
        self._fps = fps
        self._include_timestamp_overlay = include_timestamp_overlay

        if not output_file_loc.parent.exists():
            output_file_loc.parent.mkdir(parents=True)

        self._video = None


    @staticmethod
    def _get_fourcc_code(codec: str | int):
        if isinstance(codec, str):
            fourcc = cv2.VideoWriter_fourcc(*codec)
        else:
            fourcc = codec

        return fourcc
    

    @staticmethod
    def _get_image_info(image: np.ndarray) -> tuple:
        is_color = True if image.ndim == 3 else False
        
        if is_color:
            frame_height, frame_width, _ = image.shape
        else:
            frame_height, frame_width = image.shape
        
        return (frame_height, frame_width), is_color
    

    @staticmethod
    def _get_timestamp_str(timestamp=None):
        if timestamp is not None:
            ts = timestamp
        else:
            ts = datetime.datetime.now()
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return ts_str


    def _init_video(self, image: np.ndarray):
        (height, width), is_color = self._get_image_info(image=image)

        self._shape = (height, width)

        self._video = cv2.VideoWriter(
            filename=str(self._output_file_loc),
            fourcc=self._fourcc,
            fps=self._fps,
            frameSize=(width, height),
            isColor=is_color,
        )

    def _is_correct_image_shape(self, image):
        (height, width), is_color = self._get_image_info(image=image)
        return (height, width) == self._shape

    def add_frame(self, image: np.ndarray, timestamp=None):

        # First frame initialization
        if self._video is None:
            self._init_video(image=image)
        
        if not self._is_correct_image_shape(image):
            logger.error("VideoWriter: Inconsistent Image Shape. Video will likely corrupt")
            logger.warning("VideoWriter: Currently continuing with writing (may want to change)")

        if self._include_timestamp_overlay:
            if timestamp is not None:
                ts = self._get_timestamp_str(timestamp)
            else:
                ts = self._get_timestamp_str()
            image = image_utils.add_timestamp(image=image, timestamp_str=ts)
        
        # mp4v codec only supports 8-bit — convert if needed (#424)
        if image.dtype != np.uint8:
            image = image_utils.convert_16bit_to_8bit(image) if image.dtype == np.uint16 else image.astype(np.uint8)
        success = self._video.write(image)
        if success is False:
            logger.error("VideoWriter: cv2.VideoWriter.write() returned failure — frame may be lost")


    def finish(self):
        if self._video is not None:
            try:
                self._video.release()
            except Exception as e:
                logger.error(f"VideoWriter: release() failed — video file may be corrupt: {e}")
        else:
            logger.warning("VideoWriter.finish() called without adding any frames. No video file was created.")

    def test_video(self, filename):
        logger.info(f"VideoWriter: Testing video {filename}")
        cap = cv2.VideoCapture(str(filename))
        if not cap.isOpened():
            logger.error("Video Writer: Output file is corrupt or unreadable")
            return False
        ok, test_frame = cap.read()
        if not ok:
            logger.error("Video Writer: No frames could be read back; file is probably corrupt")
            return False
        cap.release()
        return True
