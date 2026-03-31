# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import datetime
import pathlib

import cv2
import numpy as np

import modules.image_utils as image_utils
from lvp_logger import logger

# Try to import PyAV for H.264 encoding. Falls back to cv2 if unavailable.
try:
    import av
    _HAS_PYAV = True
except ImportError:
    _HAS_PYAV = False
    logger.info("VideoWriter: PyAV not available — falling back to OpenCV VideoWriter")


class VideoWriter:

    def __init__(
        self,
        output_file_loc: pathlib.Path,
        fps: float,
        include_timestamp_overlay: bool,
    ):
        self._output_file_loc = output_file_loc
        self._fps = fps
        self._include_timestamp_overlay = include_timestamp_overlay
        self._shape = None
        self._frame_count = 0

        if not output_file_loc.parent.exists():
            output_file_loc.parent.mkdir(parents=True)

        # Backend selection: PyAV (H.264) preferred, cv2 fallback
        self._use_pyav = _HAS_PYAV
        self._container = None  # PyAV container
        self._stream = None     # PyAV video stream
        self._cv2_video = None  # cv2.VideoWriter fallback

    @staticmethod
    def _get_image_info(image: np.ndarray) -> tuple:
        is_color = image.ndim == 3
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
        return ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _init_pyav(self, width, height, is_color):
        """Initialize PyAV H.264 encoder."""
        try:
            self._container = av.open(str(self._output_file_loc), mode='w')
            self._stream = self._container.add_stream('libx264', rate=int(self._fps))
            self._stream.width = width
            self._stream.height = height
            self._stream.pix_fmt = 'yuv420p'
            # Quality: CRF 23 is visually lossless for microscopy at reasonable file size
            # ultrafast: minimal CPU cost, slightly larger files. Microscopy
            # frames have low noise so quality difference is negligible.
            self._stream.options = {'crf': '23', 'preset': 'ultrafast'}
            self._is_color = is_color
            logger.info(f"VideoWriter: Opened H.264 encoder ({width}x{height} @ {int(self._fps)}fps)")
        except Exception as e:
            logger.warning(f"VideoWriter: PyAV init failed ({e}), falling back to cv2")
            self._use_pyav = False
            self._container = None
            self._stream = None
            self._init_cv2(width, height, is_color)

    def _init_cv2(self, width, height, is_color):
        """Initialize cv2 VideoWriter fallback (XVID/AVI)."""
        # Use XVID — bundled with OpenCV, works on all platforms
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fallback_path = self._output_file_loc.with_suffix('.avi')
        self._output_file_loc = fallback_path
        self._cv2_video = cv2.VideoWriter(
            filename=str(fallback_path),
            fourcc=fourcc,
            fps=self._fps,
            frameSize=(width, height),
            isColor=is_color,
        )
        if not self._cv2_video.isOpened():
            logger.error(f"VideoWriter: cv2 fallback ALSO failed to open {fallback_path}. "
                         f"No video will be written.")
        else:
            logger.info(f"VideoWriter: Using cv2 XVID fallback → {fallback_path}")

    def _init_video(self, image: np.ndarray):
        (height, width), is_color = self._get_image_info(image=image)
        self._shape = (height, width)

        if self._use_pyav:
            self._init_pyav(width, height, is_color)
        else:
            self._init_cv2(width, height, is_color)

    def _is_correct_image_shape(self, image):
        (height, width), _ = self._get_image_info(image=image)
        return (height, width) == self._shape

    def add_frame(self, image: np.ndarray, timestamp=None):
        # First frame initialization
        if self._container is None and self._cv2_video is None:
            self._init_video(image=image)

        if not self._is_correct_image_shape(image):
            logger.error("VideoWriter: Inconsistent Image Shape. Video will likely corrupt")

        if self._include_timestamp_overlay:
            ts = self._get_timestamp_str(timestamp)
            image = image_utils.add_timestamp(image=image, timestamp_str=ts)

        # Ensure 8-bit
        if image.dtype != np.uint8:
            image = image_utils.convert_16bit_to_8bit(image) if image.dtype == np.uint16 else image.astype(np.uint8)

        if self._use_pyav and self._stream is not None:
            try:
                # PyAV expects RGB for color, gray for mono
                if image.ndim == 3:
                    # OpenCV uses BGR — convert to RGB for PyAV
                    frame = av.VideoFrame.from_ndarray(image[:, :, ::-1], format='rgb24')
                else:
                    frame = av.VideoFrame.from_ndarray(image, format='gray')
                for packet in self._stream.encode(frame):
                    self._container.mux(packet)
                self._frame_count += 1
            except Exception as e:
                logger.error(f"VideoWriter: PyAV encode error: {e}")
        elif self._cv2_video is not None:
            success = self._cv2_video.write(image)
            if success is False:
                logger.error("VideoWriter: cv2.VideoWriter.write() returned failure — frame may be lost")
            self._frame_count += 1

    def finish(self):
        if self._use_pyav and self._container is not None:
            try:
                # Flush encoder
                for packet in self._stream.encode():
                    self._container.mux(packet)
                self._container.close()
                logger.info(f"VideoWriter: H.264 video closed ({self._frame_count} frames)")
            except Exception as e:
                logger.error(f"VideoWriter: PyAV close failed: {e}")
        elif self._cv2_video is not None:
            try:
                self._cv2_video.release()
            except Exception as e:
                logger.error(f"VideoWriter: cv2 release() failed: {e}")
        else:
            logger.warning("VideoWriter.finish() called without adding any frames.")

    def test_video(self, filename):
        logger.info(f"VideoWriter: Testing video {filename}")
        cap = cv2.VideoCapture(str(filename))
        if not cap.isOpened():
            logger.error("VideoWriter: Output file is corrupt or unreadable")
            return False
        ok, test_frame = cap.read()
        if not ok:
            logger.error("VideoWriter: No frames could be read back; file is probably corrupt")
            return False
        cap.release()
        return True
