# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Shared frame-edge helpers for video recording write lanes.

Every recording controller (manual today, protocol video steps next)
writes engine-delivered frames through the same physical edge: rebase
camera hardware ticks onto the host epoch, orient the raw camera array,
fit it to the recording geometry, and stamp per-frame TIFF metadata.
One home for that logic so the capture paths cannot drift on frame
identity or geometry.
"""

import datetime
import pathlib
import re

import numpy as np

import modules.common_utils as common_utils
import modules.image_utils as image_utils

# --- Frame filename contract ------------------------------------------------
#
# The recording engine's on-disk frame names, built AND parsed here, in one
# place. The names are load-bearing beyond the capture code: Create Video,
# post-processing discovery, and the hyperstack loader classify files by
# these tokens, and folders of existing user recordings must keep parsing
# forever. Changing a token means changing every consumer below plus a
# two-generation reader for old folders -- never change one in isolation.
#
# Protocol video steps write   <step_name>_Frame_<NNNN>.tiff   into a
# per-recording folder named <step_name>, where <step_name> ends with the
# 'video' post suffix (common_utils.build_step_name) -- so protocol frame
# names carry '_video_Frame_'. Manual "Frames" recordings write
# ManualVideo_Frame_<NNNN>_<ts>.tiff. The two vocabularies differ by case
# ('Video_Frame' vs 'video_Frame'), so the predicates below are disjoint;
# consumers must never re-derive them with ad-hoc substring checks.

_FRAME_TOKEN = '_Frame_'

# Protocol frame names juxtapose the step name's 'video' suffix with the
# frame token; the recording folder is the step name itself.
_VIDEO_RECORDING_DIR_SUFFIX = f'_{common_utils.POST_TOKEN_VIDEO}'
_PROTOCOL_FRAME_TOKEN = f'{_VIDEO_RECORDING_DIR_SUFFIX}{_FRAME_TOKEN}'

_MANUAL_FRAME_PREFIX = f'ManualVideo{_FRAME_TOKEN}'
MANUAL_HYPERSTACK_FILENAME = f'{_MANUAL_FRAME_PREFIX}HyperStack.ome.tiff'

# The producers pad the frame number (:04), so it grows to five digits at
# frame 10,000. Any fixed-width or lexical ordering therefore wraps there
# (frame 10000 collides with or sorts beside frame 1000); ordering must
# parse the number and compare numerically.
_FRAME_NUM_RE = re.compile(rf'{_FRAME_TOKEN}(\d+)')

# The digit after the prefix keeps the optional hyperstack container file
# out of the frame sequence.
_MANUAL_FRAME_RE = re.compile(rf'{re.escape(_MANUAL_FRAME_PREFIX)}\d')


def protocol_frame_filename_template(step_name: str) -> str:
    """Frame-file template for a protocol video step's recording."""
    return f'{step_name}{_FRAME_TOKEN}{{n:04d}}.tiff'


def manual_frame_filename_template() -> str:
    """Frame-file template for a manual "Frames" recording."""
    return f'{_MANUAL_FRAME_PREFIX}{{n:04d}}_{{ts}}.tiff'


def frame_number(filename: str | pathlib.Path) -> int:
    """Numeric frame index from a video-frame filename.

    Raises:
        ValueError: The name carries no frame-number token. A name that
            cannot be ordered must fail the build loudly -- a guessed
            key would scramble the output video silently.
    """
    match = _FRAME_NUM_RE.search(str(filename))
    if match is None:
        raise ValueError(f'No frame number in video frame filename {filename!r}')
    return int(match.group(1))


def is_manual_video_frame(filename: str) -> bool:
    """True for a manual recording's frame file (not its hyperstack)."""
    return _MANUAL_FRAME_RE.match(filename) is not None


def is_protocol_video_frame(filename: str | pathlib.Path) -> bool:
    """True for a frame file written by a protocol video step."""
    return _PROTOCOL_FRAME_TOKEN in str(filename)


def is_video_frame(filename: str | pathlib.Path) -> bool:
    """True for any recording-engine frame file, manual or protocol."""
    name = pathlib.PurePath(str(filename)).name
    return is_manual_video_frame(name) or is_protocol_video_frame(name)


def is_video_recording_dir_name(dirname: str) -> bool:
    """True when a directory NAME is a protocol video recording folder.

    Tests the final path component only -- a parent folder that happens
    to carry the token anywhere in its path must not classify.
    """
    return dirname.endswith(_VIDEO_RECORDING_DIR_SUFFIX)


class CameraTickRebaser:
    """Host-epoch seconds for camera frames; hardware ticks when usable.

    Camera hardware ticks are the frame's own clock, free of the OS
    scheduling jitter host arrival stamps carry; rebasing them onto the
    host epoch at the first tick-carrying frame keeps camera-grade
    intervals on the axis cadence selection runs on. Without usable
    ticks the host arrival time is used and the manifest's timestamp
    grade reports it.

    One instance per recording: the rebase offset anchors at the first
    tick-carrying frame and must never survive into the next recording.
    """

    def __init__(self, tick_freq_hz: float | None, clock):
        self._freq = tick_freq_hz
        self._clock = clock
        self._offset: float | None = None

    def frame_time_s(self, timestamp, chunks) -> float:
        """Host-epoch seconds for one frame delivered by the SDK callback."""
        if isinstance(timestamp, datetime.datetime):
            host_s = timestamp.timestamp()
        elif timestamp is not None:
            host_s = float(timestamp)
        else:
            host_s = self._clock()
        ticks = chunks.get('Timestamp') if chunks else None
        if self._freq and ticks is not None:
            if self._offset is None:
                self._offset = host_s - ticks / self._freq
            return self._offset + ticks / self._freq
        return host_s


def orient_and_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Flip a raw camera frame to display orientation and fit the geometry.

    Orientation and contiguity are paid here at the write edge, never in
    the per-frame ingest callback.

    Raises:
        ValueError: The frame cannot fit the recording geometry. Costs
            exactly that frame -- the engine counts it as a write
            failure and the recording continues.
    """
    image = np.flip(image, 0)
    target_shape = (height, width)
    if image.shape != target_shape:
        fitted = image_utils.fit_frame_to_shape(image, target_shape)
        if fitted is None:
            raise ValueError(
                f'frame shape {image.shape} incompatible with recording geometry {target_shape}'
            )
        image = fitted
    return image


def tiff_frame_metadata(
    timestamp_s: float,
    frame_number: int,
    chunks,
    tick_freq_hz: float | None,
) -> tuple[dict, str]:
    """Per-frame TIFF metadata plus a path-safe timestamp string.

    The timestamp travels in metadata, not pixels -- Create Video draws
    it at build time when the overlay is enabled. Camera chunk identity
    (hardware ticks, FrameID) is recorded when the frame carried it.

    Returns:
        ``(metadata, ts_filename)`` -- the metadata dict for the TIFF
        writer, and a colon-free millisecond-precision timestamp string
        safe for Windows filenames.
    """
    ts = datetime.datetime.fromtimestamp(timestamp_s)
    ts_filename = ts.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
    metadata = {
        'datetime': ts.strftime('%Y:%m:%d %H:%M:%S'),
        'timestamp': ts.strftime('%Y:%m:%d %H:%M:%S.%f'),
        'timestamp_iso': ts.isoformat(timespec='microseconds'),
        'frame_num': frame_number,
        # The writer reads this as a required key, so it is stated rather than
        # omitted: a recording frame has no scale source today, and an explicit
        # None makes the file declare no absolute unit instead of inheriting
        # tifffile's 1/1 default under a centimetre unit. Supplying a real
        # measured value here is all it takes to make these frames carry scale.
        'pixel_size_um': None,
    }
    if chunks is not None:
        ts_ticks = chunks.get('Timestamp')
        if ts_ticks is not None:
            metadata['timestamp_camera_ticks'] = int(ts_ticks)
        if tick_freq_hz is not None:
            metadata['timestamp_camera_tick_hz'] = int(tick_freq_hz)
        frame_id = chunks.get('FrameID')
        if frame_id is not None:
            metadata['frame_id'] = int(frame_id)
    return metadata, ts_filename
