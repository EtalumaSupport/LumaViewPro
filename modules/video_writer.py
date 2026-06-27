# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import datetime
import os
import pathlib
import threading
from fractions import Fraction

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
    logger.info('VideoWriter: PyAV not available -- falling back to OpenCV VideoWriter')


def fps_from_frames(captured_frames: int, duration_sec: float) -> float:
    """Playback frames-per-second for captured_frames recorded over duration_sec.

    Real division, never floor or int()-clamp. A slow recording (long-exposure
    or timelapse) can capture fewer frames than the seconds elapsed, so the true
    rate is below 1 fps: flooring it yields 0 (an empty, unplayable file that
    silently loses the recording) and clamping it up to 1 distorts the playback
    duration. The float rate is carried through to the encoder unchanged. The
    caller guarantees duration_sec > 0.
    """
    return captured_frames / duration_sec


class VideoWriter:
    def __init__(
        self,
        output_path: pathlib.Path,
        fps: float,
        width: int | None = None,
        height: int | None = None,
        *,
        color: str | None = None,
        include_timestamp_overlay: bool = False,
    ):
        """Encode a video file. Accepts mono frames + applies false-color
        internally when `color` is set; also accepts pre-colored RGB input.

        When both ``width`` and ``height`` are given the encoder eager-
        initializes in __init__. Otherwise it lazy-initializes from the
        first ``add_frame`` call (preserves the pre-1d caller pattern).

        Args:
            output_path: Destination video file. .mp4 routes to PyAV H.264;
                cv2 fallback rewrites the suffix to .avi.
            fps: Frames per second.
            width: Frame width in pixels. Optional; None defers to first frame.
            height: Frame height in pixels. Optional; None defers to first frame.
            color: Layer name ('Red', 'Green', 'Blue', 'Lumi', ...) for
                in-writer false-color. None encodes grayscale: PyAV gray
                pixfmt; cv2 isColor=False.
            include_timestamp_overlay: Overlay frame timestamps via image_utils.
        """
        self._output_path = pathlib.Path(output_path)
        self._fps = fps
        self._color = color
        self._include_timestamp_overlay = include_timestamp_overlay
        self._shape = (height, width) if (width is not None and height is not None) else None
        self._frame_count = 0
        # Frames the encoder accepted but failed to write -- counted so the
        # caller can warn the user that the output is short rather than
        # delivering a silently-truncated video.
        self._dropped_frames = 0
        # Protects _frame_count + encoder state for REST queries
        self._frame_lock = threading.Lock()

        if not self._output_path.parent.exists():
            self._output_path.parent.mkdir(parents=True)

        # Backend selection: PyAV (H.264) preferred, cv2 fallback
        self._use_pyav = _HAS_PYAV
        self._container = None  # PyAV container
        self._stream = None  # PyAV video stream
        self._cv2_video = None  # cv2.VideoWriter fallback
        self._finished = False

        # Eager-init only when caller provides dimensions AND color is set:
        # with color set the encoder always emits 3-channel RGB (false-color
        # from mono, or pre-colored input), so is_color is known without the
        # first frame. When color is None, whether the stream is color depends
        # on the first frame's ndim (a caller may feed pre-colored RGB into a
        # None-color writer), so defer encoder init to _lazy_init_from_frame
        # -- mirroring the no-dimensions path -- instead of locking in a gray
        # encoder that would corrupt RGB input.
        if self._shape is not None and color is not None:
            if self._use_pyav:
                self._init_pyav(width, height, True)
            else:
                self._init_cv2(width, height, True)

    @staticmethod
    def _get_timestamp_str(timestamp=None):
        if timestamp is not None:
            ts = timestamp
        else:
            ts = datetime.datetime.now()
        return ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def _encoder_rate(self) -> Fraction:
        """Canonical encoder frame rate, shared by every writer-init path.

        A slow recording (timelapse / long-exposure) captures fewer frames than
        the seconds elapsed, so its true rate is below 1 fps. Flooring such a
        rate to an integer turns a value like 0.3 into 0, which every encoder
        backend rejects -- producing an empty, unplayable file and losing the
        recording. Keeping the rate as a clean fraction preserves the real
        sub-1 rate so the encoders honor it and playback duration stays true;
        limit_denominator trims the binary-float artifacts of a value like 0.3
        to an exact ratio.
        """
        return Fraction(self._fps).limit_denominator()

    def _init_pyav(self, width, height, is_color):
        """Initialize PyAV H.264 encoder."""
        try:
            self._container = av.open(str(self._output_path), mode='w')
            # libx264 honors the fractional rate, so a true sub-1 fps recording
            # (timelapse / long-exposure) keeps its real duration -- see
            # _encoder_rate for why an int floor would lose it.
            self._stream = self._container.add_stream('libx264', rate=self._encoder_rate())
            # Multi-threaded libx264, capped to cores-2 so the encode scales
            # with the machine but always leaves headroom for the GUI/GL main
            # thread (uncapped it grabs every core and froze the GUI mid-encode
            # on an 8-core box). This was previously pinned to 1 thread to dodge
            # a lost-wakeup deadlock in libx264's thread-pool teardown
            # (x264_threadpool_delete on encoder close, which hung an 8-core box
            # AFTER the encode finished); the libx264 bundled with av 17.0.1
            # fixes that teardown, so the worker pool is safe again. Revert to
            # thread_count = 1 if the encoder-close deadlock ever returns.
            self._stream.thread_count = max(1, (os.cpu_count() or 4) - 2)
            self._stream.width = width
            self._stream.height = height
            self._stream.pix_fmt = 'yuv420p'
            # Quality: CRF 23 is visually lossless for microscopy at reasonable file size
            # ultrafast: minimal CPU cost, slightly larger files. Microscopy
            # frames have low noise so quality difference is negligible.
            self._stream.options = {'crf': '23', 'preset': 'ultrafast'}
            self._is_color = is_color
            logger.info(
                f'VideoWriter: Opened H.264 encoder ({width}x{height} @ {float(self._fps):g}fps)'
            )
        except Exception as e:
            logger.warning(f'VideoWriter: PyAV init failed ({e}), falling back to cv2')
            self._use_pyav = False
            self._container = None
            self._stream = None
            self._init_cv2(width, height, is_color)

    def _open_cv2_writer(self, fourcc, fallback_path, rate, width, height, is_color):
        """Construct one cv2.VideoWriter at the given rate."""
        return cv2.VideoWriter(
            filename=str(fallback_path),
            fourcc=fourcc,
            fps=rate,
            frameSize=(width, height),
            isColor=is_color,
        )

    def _init_cv2(self, width, height, is_color):
        """Initialize cv2 VideoWriter fallback (XVID/AVI)."""
        # Use XVID -- bundled with OpenCV, works on all platforms
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fallback_path = self._output_path.with_suffix('.avi')
        self._output_path = fallback_path
        # cv2.VideoWriter takes a double fps. The FFMPEG-backed AVI encoder
        # honors a true sub-1 rate (timelapse / long-exposure), so pass the real
        # rate rather than an int that would floor 0.3 to 0 and lose the file.
        rate = float(self._encoder_rate())
        self._cv2_video = self._open_cv2_writer(
            fourcc, fallback_path, rate, width, height, is_color
        )
        if not self._cv2_video.isOpened() and rate < 1.0:
            # OpenCV's built-in AVI/MJPEG encoder -- the fallback used when no
            # FFMPEG plugin is present -- refuses to open below 1 fps (it
            # asserts fps >= 1). Rather than ship an empty, unplayable file,
            # reopen at the 1 fps floor so the captured frames are preserved.
            # Playback then runs faster than the real capture rate; warn so that
            # speedup is not a silent surprise.
            logger.warning(
                f'VideoWriter: cv2/AVI backend rejected sub-1 fps ({rate:g}); the '
                f'built-in AVI encoder requires fps >= 1. Reopening at 1 fps -- '
                f'playback will run faster than the real capture rate.'
            )
            rate = 1.0
            self._cv2_video = self._open_cv2_writer(
                fourcc, fallback_path, rate, width, height, is_color
            )
        if not self._cv2_video.isOpened():
            logger.error(
                f'VideoWriter: cv2 fallback ALSO failed to open {fallback_path}. '
                f'No video will be written.'
            )
        else:
            logger.info(f'VideoWriter: Using cv2 XVID fallback -> {fallback_path}')

    def _is_correct_image_shape(self, image):
        if image.ndim == 3:
            h, w, _ = image.shape
        else:
            h, w = image.shape
        return (h, w) == self._shape

    def _lazy_init_from_frame(self, image: np.ndarray) -> None:
        """Initialize the encoder from the first frame when it was not opened
        eagerly at __init__ -- either no width/height was supplied, or color
        was None so is_color could not be fixed without the frame's ndim.
        Any RGB input OR a non-None ``color`` arg produces a 3-channel output
        stream.
        """
        if image.ndim == 3:
            h, w, _ = image.shape
        else:
            h, w = image.shape
        self._shape = (h, w)
        is_color_encode = (image.ndim == 3) or (self._color is not None)
        if self._use_pyav:
            self._init_pyav(w, h, is_color_encode)
        else:
            self._init_cv2(w, h, is_color_encode)

    def add_frame(self, image: np.ndarray, timestamp=None, significant_bits=None) -> None:
        """Add a frame to the video.

        Accepts mono 2D (H, W) input when `color` was set at __init__; the
        writer applies the layer false-color and cv2 BGR-swap (cv2 path
        only) before the encoder boundary. Also accepts pre-colored RGB
        input for back-compat callers that produce their own RGB.

        significant_bits scales a uint16 frame to 8-bit by its true payload
        depth (e.g. 12 for a right-aligned 12-bit frame). None falls back to
        treating uint16 as full 16-bit, which is correct for left-justified
        legacy frames.
        """
        with self._frame_lock:
            if self._finished:
                return

            # Init the encoder on the first frame when it was not opened
            # eagerly at __init__ -- no dimensions given, or color was None
            # so is_color had to wait for the frame ndim. Gate on the encoder
            # not yet existing (rather than a separate flag) so a caller that
            # injected its own _cv2_video / _stream is left intact.
            if self._stream is None and self._cv2_video is None:
                self._lazy_init_from_frame(image)

            if not self._is_correct_image_shape(image):
                logger.error('VideoWriter: Inconsistent Image Shape. Video will likely corrupt')

            # Ensure 8-bit
            if image.dtype != np.uint8:
                if significant_bits is not None:
                    image = image_utils.convert_to_8bit(image, significant_bits)
                elif image.dtype == np.uint16:
                    image = image_utils.convert_to_8bit(image, significant_bits=16)
                else:
                    image = image.astype(np.uint8)

            # Mono + color set -> apply false-color inside the writer.
            # Mono + color None -> pass through; gray encode.
            # RGB input -> pass through (pre-colored caller).
            if image.ndim == 2 and self._color is not None:
                image = image_utils.mono_to_rgb_falsecolor(image, layer=self._color)

            # Timestamp last -- after false-color -- so the text stays
            # neutral white instead of being tinted by the layer color map.
            if self._include_timestamp_overlay:
                ts = self._get_timestamp_str(timestamp)
                image = image_utils.add_timestamp(image=image, timestamp_str=ts)

            if self._use_pyav and self._stream is not None:
                try:
                    # PyAV expects RGB for color, gray for mono.
                    if image.ndim == 3:
                        frame = av.VideoFrame.from_ndarray(image, format='rgb24')
                    else:
                        frame = av.VideoFrame.from_ndarray(image, format='gray')
                    for packet in self._stream.encode(frame):
                        self._container.mux(packet)
                    self._frame_count += 1
                except Exception as e:
                    logger.error(f'VideoWriter: PyAV encode error: {e}')
                    self._dropped_frames += 1
            elif self._cv2_video is not None:
                # cv2.VideoWriter is the only BGR consumer in the save path;
                # swap at this boundary for 3-channel input. Mono passes
                # through unchanged (isColor=False at init).
                if image.ndim == 3:
                    image_for_cv2 = image[:, :, ::-1]
                else:
                    image_for_cv2 = image
                success = self._cv2_video.write(image_for_cv2)
                if success is False:
                    # cv2 returns None on success; only an explicit False is a
                    # reported failure. Count it as dropped, not written.
                    logger.error(
                        'VideoWriter: cv2.VideoWriter.write() returned failure -- frame lost'
                    )
                    self._dropped_frames += 1
                else:
                    self._frame_count += 1

    def close(self) -> None:
        """Flush encoder and close the container. Idempotent."""
        with self._frame_lock:
            if self._finished:
                return
            self._finished = True
        if self._use_pyav and self._container is not None:
            try:
                # Flush encoder
                for packet in self._stream.encode():
                    self._container.mux(packet)
                self._container.close()
                logger.info(f'VideoWriter: H.264 video closed ({self._frame_count} frames)')
            except Exception as e:
                logger.error(f'VideoWriter: PyAV close failed: {e}')
        elif self._cv2_video is not None:
            try:
                self._cv2_video.release()
            except Exception as e:
                logger.error(f'VideoWriter: cv2 release() failed: {e}')
        else:
            logger.warning('VideoWriter.close() called without adding any frames.')

    def get_progress(self) -> dict:
        """Thread-safe progress query for REST API consumers."""
        with self._frame_lock:
            return {
                'frame_count': self._frame_count,
                'dropped_frames': self._dropped_frames,
                'finished': self._finished,
                'output_file': str(self._output_path),
            }

    @property
    def dropped_frames(self) -> int:
        """Frames the encoder failed to write during this recording."""
        with self._frame_lock:
            return self._dropped_frames

    def test_video(self, filename):
        logger.info(f'VideoWriter: Testing video {filename}')
        cap = cv2.VideoCapture(str(filename))
        if not cap.isOpened():
            logger.error('VideoWriter: Output file is corrupt or unreadable')
            return False
        ok, _test_frame = cap.read()
        if not ok:
            logger.error('VideoWriter: No frames could be read back; file is probably corrupt')
            return False
        cap.release()
        return True
