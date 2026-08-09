# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import datetime
import os
import pathlib
import threading
from fractions import Fraction

# av is a hard dependency of every video encode: a broken install fails
# loudly here, at import, instead of silently degrading the recording to
# a different container and codec.
import av
import numpy as np

import modules.image_utils as image_utils
from lvp_logger import logger


# 90 kHz is the conventional MPEG timescale; sub-ms capture jitter
# survives pts quantization at this resolution.
VFR_TIME_BASE = Fraction(1, 90000)


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
        vfr: bool = False,
    ):
        """Encode a video file. Accepts mono frames + applies false-color
        internally when `color` is set; also accepts pre-colored RGB input.

        When both ``width`` and ``height`` are given the encoder eager-
        initializes in __init__. Otherwise it lazy-initializes from the
        first ``add_frame`` call (preserves the pre-1d caller pattern).

        Args:
            output_path: Destination video file (H.264 via PyAV).
            fps: Frames per second. In VFR mode this is the container's
                nominal rate only; real timing rides per-frame pts.
            width: Frame width in pixels. Optional; None defers to first frame.
            height: Frame height in pixels. Optional; None defers to first frame.
            color: Layer name ('Red', 'Green', 'Blue', 'Lumi', ...) for
                in-writer false-color. None encodes grayscale (gray
                pixfmt); a gray-rendering label (transmitted light --
                no false-color map) encodes grayscale the same way.
            include_timestamp_overlay: Overlay frame timestamps via image_utils.
            vfr: Variable-frame-rate timing: each frame's presentation
                time comes from its real capture timestamp, so a delivery
                stall plays at its true duration. Every ``add_frame`` call
                must then supply ``timestamp``.
        """
        self._output_path = pathlib.Path(output_path)
        self._fps = fps
        self._color = color
        self._include_timestamp_overlay = include_timestamp_overlay
        self._vfr = vfr
        # First-frame capture time; the pts origin so playback starts at 0.
        self._vfr_origin_s: float | None = None
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

        self._container = None  # PyAV container
        self._stream = None  # PyAV video stream
        self._finished = False

        # Eager-init only when caller provides dimensions AND the label
        # actually colors: a chromatic map always emits 3-channel RGB, so
        # is_color is known without the first frame. When color is None or
        # a gray-rendering label (per the false-color table -- transmitted
        # light has no map), whether the stream is color depends on the
        # first frame's ndim (a caller may feed pre-colored RGB into such
        # a writer), so defer encoder init to _lazy_init_from_frame --
        # mirroring the no-dimensions path -- instead of locking in a gray
        # encoder that would corrupt RGB input.
        if self._shape is not None and self._colors_output():
            self._init_pyav(width, height, True)

    def _colors_output(self) -> bool:
        """True when the color label applies a chromatic false-color map.

        The false-color table is the authority: a transmitted-light or
        unknown label yields gray pixels, so mono input stays mono and
        gray pixels get a gray encode instead of a 3-channel replica.
        """
        return self._color is not None and not image_utils.layer_renders_grayscale(self._color)

    @property
    def output_path(self) -> pathlib.Path:
        """The authoritative save location -- read this back when recording.

        May differ from the path the caller requested: a collision adds a
        numeric suffix. A record built from the requested path would
        attribute the wrong file to the capture whenever that applies.
        """
        return self._output_path

    def _resolve_collision_free_output(self):
        """Never silently overwrite an existing output.

        Runs at encoder init, immediately before the container opens, so
        the check and the open cannot disagree about the path. The plain
        name is kept when free; a numeric suffix is added only on actual
        collision, so happy-path filenames are unchanged.
        """
        if not self._output_path.exists():
            return
        requested_path = self._output_path
        n = 1
        while True:
            candidate = requested_path.with_name(
                f'{requested_path.stem}_{n:06d}{requested_path.suffix}'
            )
            if not candidate.exists():
                break
            n += 1
        logger.warning(
            f'Video filename collision: {requested_path.name} already '
            f'exists; saving as {candidate.name} instead.'
        )
        self._output_path = candidate

    @staticmethod
    def _get_timestamp_str(timestamp=None):
        if timestamp is None:
            ts = datetime.datetime.now()
        elif isinstance(timestamp, datetime.datetime):
            ts = timestamp
        else:
            ts = datetime.datetime.fromtimestamp(float(timestamp))
        return ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    @staticmethod
    def _timestamp_seconds(timestamp) -> float:
        """Normalize a caller timestamp (datetime or epoch seconds) to seconds."""
        if isinstance(timestamp, datetime.datetime):
            return timestamp.timestamp()
        return float(timestamp)

    def _encoder_rate(self) -> Fraction:
        """Canonical encoder frame rate for every writer-init path.

        A slow recording (timelapse / long-exposure) captures fewer frames than
        the seconds elapsed, so its true rate is below 1 fps. Flooring such a
        rate to an integer turns a value like 0.3 into 0, which the encoder
        rejects -- producing an empty, unplayable file and losing the
        recording. Keeping the rate as a clean fraction preserves the real
        sub-1 rate so the encoder honors it and playback duration stays true;
        limit_denominator trims the binary-float artifacts of a value like 0.3
        to an exact ratio.
        """
        return Fraction(self._fps).limit_denominator()

    def _init_pyav(self, width, height, is_color):
        """Initialize the H.264 encoder.

        A failed open propagates to the caller: there is no fallback
        backend, and swallowing the error here would silently produce no
        video at all.
        """
        self._resolve_collision_free_output()
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
        options = {'crf': '23', 'preset': 'ultrafast'}
        if self._vfr:
            # B-frames must be disabled for VFR: MP4 container duration
            # sums dts-based sample durations, and x264's B-frame
            # reordering compacts dts while the true times ride pts --
            # frames decode at the right times but the container reports
            # a fraction of the real duration. With bf=0, dts == pts and
            # the container duration is honest.
            options['bf'] = '0'
            self._stream.codec_context.time_base = VFR_TIME_BASE
        self._stream.options = options
        self._is_color = is_color
        logger.info(
            f'VideoWriter: Opened H.264 encoder ({width}x{height} @ {float(self._fps):g}fps)'
        )

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
        is_color_encode = (image.ndim == 3) or self._colors_output()
        self._init_pyav(w, h, is_color_encode)

    def add_frame(self, image: np.ndarray, timestamp=None, significant_bits=None) -> None:
        """Add a frame to the video.

        Accepts mono 2D (H, W) input when `color` was set at __init__; the
        writer applies the layer false-color before the encoder boundary.
        Also accepts pre-colored RGB input for back-compat callers that
        produce their own RGB.

        significant_bits scales a non-uint8 frame to 8-bit by its true
        payload depth (e.g. 12 for a right-aligned 12-bit frame). It is
        REQUIRED with non-uint8 pixels -- a missing depth raises rather
        than guessing, because either guess (full-width scale or a raw
        truncating cast) silently corrupts every pixel. None is legal
        only for uint8 input, which needs no scaling.

        In VFR mode ``timestamp`` (datetime or epoch seconds) is REQUIRED:
        it is the frame's presentation time, not decoration. A missing
        timestamp raises immediately rather than silently degrading the
        file's timing.

        Raises:
            ValueError: VFR mode and ``timestamp`` is None.
        """
        if self._vfr and timestamp is None:
            raise ValueError(
                'VideoWriter(vfr=True) requires a timestamp for every frame: '
                'per-frame pts is the timing authority'
            )
        with self._frame_lock:
            if self._finished:
                return

            # Init the encoder on the first frame when it was not opened
            # eagerly at __init__ -- no dimensions given, or color was None
            # so is_color had to wait for the frame ndim. Gate on the encoder
            # not yet existing (rather than a separate flag) so a caller that
            # injected its own _stream is left intact.
            if self._stream is None:
                self._lazy_init_from_frame(image)

            if not self._is_correct_image_shape(image):
                logger.error('VideoWriter: Inconsistent Image Shape. Video will likely corrupt')

            # Ensure 8-bit. A native-depth frame's payload depth is part
            # of what its pixels mean: a right-aligned 12-bit frame
            # scaled as 16-bit encodes full scale (4095) at 15/255 -- a
            # near-black video -- and a raw astype() truncates instead
            # of scaling. So depth is REQUIRED with non-uint8 pixels;
            # guessing either way produces a silently wrong video. Every
            # producer has the true depth (load_pixels returns it with
            # the pixels; the recording legs read it from their capture
            # config), so this raise is unreachable except through a new
            # caller that dropped the depth on the floor.
            if image.dtype != np.uint8:
                if significant_bits is None:
                    raise ValueError(
                        'VideoWriter.add_frame: significant_bits is required '
                        f'for {image.dtype} frames -- scaling without the true '
                        'payload depth corrupts every pixel'
                    )
                image = image_utils.convert_to_8bit(image, significant_bits)

            # Mono + chromatic label -> apply false-color inside the writer.
            # Mono + None or gray-rendering label -> pass through; gray encode.
            # RGB input -> pass through (pre-colored caller).
            if image.ndim == 2 and self._colors_output():
                image = image_utils.mono_to_rgb_falsecolor(image, layer=self._color)

            # Timestamp last -- after false-color -- so the text stays
            # neutral white instead of being tinted by the layer color map.
            if self._include_timestamp_overlay:
                ts = self._get_timestamp_str(timestamp)
                image = image_utils.add_timestamp(image=image, timestamp_str=ts)

            # Per-frame failure policy: an encode error costs exactly that
            # frame -- counted as dropped and surfaced by the caller --
            # never the whole recording.
            try:
                # PyAV expects RGB for color, gray for mono.
                if image.ndim == 3:
                    frame = av.VideoFrame.from_ndarray(image, format='rgb24')
                else:
                    frame = av.VideoFrame.from_ndarray(image, format='gray')
                if self._vfr:
                    ts_s = self._timestamp_seconds(timestamp)
                    if self._vfr_origin_s is None:
                        self._vfr_origin_s = ts_s
                    frame.pts = round((ts_s - self._vfr_origin_s) / VFR_TIME_BASE)
                    frame.time_base = VFR_TIME_BASE
                for packet in self._stream.encode(frame):
                    self._container.mux(packet)
                self._frame_count += 1
            except Exception as e:
                logger.error(f'VideoWriter: PyAV encode error: {e}')
                self._dropped_frames += 1

    def close(self) -> None:
        """Flush encoder and close the container. Idempotent."""
        with self._frame_lock:
            if self._finished:
                return
            self._finished = True
        if self._container is not None:
            try:
                # Flush encoder
                for packet in self._stream.encode():
                    self._container.mux(packet)
                self._container.close()
                logger.info(f'VideoWriter: H.264 video closed ({self._frame_count} frames)')
            except Exception as e:
                logger.error(f'VideoWriter: PyAV close failed: {e}')
        else:
            logger.warning('VideoWriter.close() called without adding any frames.')

    @property
    def dropped_frames(self) -> int:
        """Frames the encoder failed to write during this recording."""
        with self._frame_lock:
            return self._dropped_frames
