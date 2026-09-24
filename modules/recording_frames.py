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

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from lvp_logger import logger

import modules.common_utils as common_utils
import modules.image_save as image_save
import modules.image_utils as image_utils
from modules.lumascope_api._constants import AxisState

if TYPE_CHECKING:
    from modules.lumascope_api import Lumascope

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


def resolve_recording_pixel_size(scope: 'Lumascope') -> float | None:
    """Effective um/pixel for a recording about to start, or None if unknown.

    Both recording legs call this once at start so their frames agree on scale
    and cannot drift from each other. Returns None when the scope cannot report
    its optics; the writer then omits the scale claim rather than inventing one,
    because a guessed scale is measured off the file forever and cannot be told
    from a real one.

    Args:
        scope: The Lumascope instance the recording is running against.
    """
    # Through the accessor, which reports "no objective selected yet" as None.
    # Subscripting the store directly raises instead, and a recording must not
    # die because the scope cannot yet say how big a pixel is.
    objective = scope.runtime_state.get_current_objective()
    if objective is None:
        logger.warning(
            'Recording is starting with no objective selected; its frames '
            'will carry no um/pixel scale claim.'
        )
        return None
    pixel_size_um = common_utils.get_pixel_size(
        focal_length=objective['focal_length'],
        binning_size=scope.imaging._binning_size,
        capabilities=scope.capabilities,
    )
    if pixel_size_um is None:
        return None
    return round(pixel_size_um, common_utils.max_decimal_precision('pixel_size'))


class FrameFact(NamedTuple):
    """What was true of the scope when one recorded frame arrived.

    Read on the camera callback and carried with the frame to its write,
    because the write runs later, behind the backlog, and a read then
    would describe a different moment. Positions are None where the scope
    did not know them: X and Y as a pair in plate millimetres, Z alone in
    micrometres. ``moving`` says an axis was MOVING or HOMING when the
    frame was delivered -- delivery follows exposure and readout, so this
    is the state at delivery, not during the exposure. ``channel`` is the
    channel that lit the frame, resolved the way a still's is.
    """

    plate_x_mm: float | None
    plate_y_mm: float | None
    z_um: float | None
    moving: bool
    channel: str


def frame_fact(
    scope: 'Lumascope',
    *,
    channel_tiebreak: str,
    to_plate: Callable[[float, float], tuple[float, float]] | None,
) -> FrameFact:
    """The fact for the frame arriving now; called on the camera callback.

    Every read is a lock-guarded memory read and nothing here can raise
    once a recording has started: the listener that calls this swallows a
    raise and drops the frame, and a recording that lost every frame that
    way would end blaming the camera.

    Args:
        scope: The scope the recording runs against.
        channel_tiebreak: The channel to record when no LED is lit -- the
            one the recording started on -- so luminescence frames are
            named, as a still's are.
        to_plate: The plate transform bound when the recording started
            (``runtime_state.plate_transform()``), or None when the scope
            had no labware or offset then; X and Y are then unknown.
    """
    positions = scope.motion.axis_positions()
    moving = any(p.state in (AxisState.MOVING, AxisState.HOMING) for p in positions.values())
    x = positions.get('X')
    y = positions.get('Y')
    z = positions.get('Z')
    plate_x_mm = plate_y_mm = None
    if (
        to_plate is not None
        and x is not None
        and y is not None
        and x.position is not None
        and y.position is not None
    ):
        plate_x_mm, plate_y_mm = to_plate(x.position, y.position)
    return FrameFact(
        plate_x_mm=plate_x_mm,
        plate_y_mm=plate_y_mm,
        z_um=z.position if z is not None else None,
        moving=moving,
        channel=common_utils.resolve_channel_identity(scope.illumination, channel_tiebreak),
    )


def tiff_frame_metadata(
    timestamp_s: float,
    frame_number: int,
    chunks: Mapping[str, Any] | None,
    tick_freq_hz: float | None,
    pixel_size_um: float | None,
    fact: FrameFact,
) -> tuple[dict, str]:
    """Per-frame TIFF metadata plus a path-safe timestamp string.

    The timestamp travels in metadata, not pixels -- Create Video draws
    it at build time when the overlay is enabled. Camera chunk identity
    (hardware ticks, FrameID) is recorded when the frame carried it. The
    frame's own fact -- where it was taken, whether the stage was moving,
    which channel lit it -- is recorded with the still capture's position
    keys, so the file says what it is with or without a hyperstack.

    Args:
        timestamp_s: Frame time, epoch seconds.
        frame_number: Ordinal within the recording.
        chunks: Camera chunk data, or None when the frame carried none.
        tick_freq_hz: Camera tick frequency, or None when unknown.
        pixel_size_um: Image scale measured once at recording start, or
            None when the scope cannot report its optics. Required rather
            than defaulted: this builder is the only scale source the
            recording legs have, so a caller that omits it would write
            unmeasurable files, and a default would hide that at the one
            place it must be visible.
        fact: What was true when the frame arrived (``frame_fact``).
            Required for the same reason, and refused as None: a frame
            file with no fact would claim nothing about where it was
            taken, silently.

    Returns:
        ``(metadata, ts_filename)`` -- the metadata dict for the TIFF
        writer, and a colon-free millisecond-precision timestamp string
        safe for Windows filenames.

    Raises:
        ValueError: If ``fact`` is None.
    """
    if fact is None:
        raise ValueError('a recorded frame needs its fact; none was given')
    ts = datetime.datetime.fromtimestamp(timestamp_s)
    ts_filename = ts.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
    metadata = {
        'datetime': ts.strftime('%Y:%m:%d %H:%M:%S'),
        'timestamp': ts.strftime('%Y:%m:%d %H:%M:%S.%f'),
        'timestamp_iso': ts.isoformat(timespec='microseconds'),
        'frame_num': frame_number,
        'channel': fact.channel,
        'stage_moving': fact.moving,
        **image_save.position_metadata_fields(fact.plate_x_mm, fact.plate_y_mm, fact.z_um),
        # The writer reads this as a required key. A real value makes the file
        # declare its scale; None makes it declare no absolute unit rather than
        # inherit tifffile's 1/1 default under a centimetre unit, which reads as
        # 1 px/cm -- a concrete and wildly wrong measurement.
        'pixel_size_um': pixel_size_um,
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
