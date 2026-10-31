# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Shared frame-edge helpers for video recording write lanes.

Every recording controller (the manual recording and the protocol video
step) writes engine-delivered frames through the same physical edge: rebase
camera hardware ticks onto the host epoch, build each frame's record as it
arrives, orient the raw camera array, fit it to the recording geometry, and
write it with its record through the still's builder and writer.
One home for that logic so the capture paths cannot drift on frame
identity or geometry.
"""

import datetime
import pathlib
import re

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import modules.common_utils as common_utils
import modules.image_save as image_save
import modules.image_utils as image_utils
from modules.lumascope_api._constants import AxisState
from modules.lumascope_api.frame_record import (
    FrameRecord,
    FrameTime,
    TimestampSource,
    delivered_frame_record,
)

if TYPE_CHECKING:
    from drivers.camera import Frame
    from modules.coord_transformations import PlatePosition
    from modules.image_mode import OutputFormat
    from modules.labware import Labware
    from modules.layer_record import ChannelColors
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
# Protocol video steps write   <step_name>_Frame_<NNNN><suffix>   into a
# per-recording folder named <step_name>, where <step_name> ends with the
# 'video' post suffix (common_utils.build_step_name) -- so protocol frame
# names carry '_video_Frame_'. Manual "Frames" recordings write
# ManualVideo_Frame_<NNNN>_<ts><suffix>. The suffix is the recording's
# format's (image_utils.derived_file_suffix): .ome.tiff for an OME-TIFF
# recording, .tiff for every other, and the writer takes its container from
# it. Readers find a frame by its token and a TIFF last suffix, which both
# satisfy. The two vocabularies differ by case
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


def protocol_frame_filename_template(step_name: str, output_format: 'OutputFormat') -> str:
    """Frame-file template for a protocol video step's recording in *output_format*, the run's."""
    return f'{step_name}{_FRAME_TOKEN}{{n:04d}}{image_utils.derived_file_suffix(output_format)}'


def manual_frame_filename_template(output_format: 'OutputFormat') -> str:
    """Frame-file template for a manual "Frames" recording in *output_format*, the Live one."""
    return f'{_MANUAL_FRAME_PREFIX}{{n:04d}}_{{ts}}{image_utils.derived_file_suffix(output_format)}'


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
    """True when a directory NAME is a protocol video step's folder name.

    A name only: the support report's census, which would rather list a
    recording that wrote nothing than miss it. A build asks
    ``is_video_recording_dir``, which reads the frames. Tests the final path
    component only -- a parent folder that happens to carry the token
    anywhere in its path must not classify.
    """
    return dirname.endswith(_VIDEO_RECORDING_DIR_SUFFIX)


def is_video_recording_dir(path: str | pathlib.Path) -> bool:
    """True when *path* is a protocol video recording folder: it holds a video step's frames.

    Read from what the folder holds, never from its name: a run folder a
    person renames ``plate1_video`` holds stills, and read as a recording it
    was scanned as one and every build on it found no images. Only the
    folder itself is read, so a parent that carries the token does not
    classify.
    """
    path = pathlib.Path(path)
    return path.is_dir() and any(is_protocol_video_frame(p.name) for p in path.iterdir())


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

    def frame_time_s(
        self,
        timestamp: datetime.datetime | float | None,
        chunks: Mapping[str, int | float] | None,
    ) -> tuple[float, TimestampSource]:
        """Host-epoch seconds for one frame delivered by the SDK callback, and
        what they are: the camera's ticks rebased, or the host's arrival time."""
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
            return self._offset + ticks / self._freq, TimestampSource.CAMERA_REBASED
        return host_s, TimestampSource.HOST_ARRIVAL


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


class FrameFact(NamedTuple):
    """What was true of the scope when one recorded frame arrived.

    Read on the camera callback and carried with the frame to its write,
    because the write runs later, behind the backlog, and a read then
    would describe a different moment. Positions are None where the scope
    did not know them: X and Y as a pair in plate millimetres, Z alone in
    micrometres. ``moving`` says an axis was MOVING or HOMING when the
    frame was delivered -- delivery follows exposure and readout, so this
    is the state at delivery, not during the exposure. What lit a recorded
    frame is read beside it (``arrival_facts``); a still's own record says
    what lit it.
    """

    plate_x_mm: float | None
    plate_y_mm: float | None
    z_um: float | None
    moving: bool

    def well_label(self, labware: 'Labware') -> str | None:
        """The well this frame's position lies in on ``labware``, the plate
        the position is stated on.

        None where X or Y is unknown, so a file names no well it cannot say
        it was over -- not the well a step planned, which a scope with no XY
        stage never reaches. ``''`` off the plate's wells, or on a plate
        with none.
        """
        if self.plate_x_mm is None or self.plate_y_mm is None:
            return None
        return labware.get_well_label(x=self.plate_x_mm, y=self.plate_y_mm)


def frame_fact(
    scope: 'Lumascope',
    *,
    to_plate: Callable[[float, float], 'PlatePosition'] | None,
) -> FrameFact:
    """The fact for the frame arriving now; called on the camera callback.

    A manual still asks it too, on the camera lane beside its grab, so a
    still and a recorded frame state their position from one reader.

    Every read is a lock-guarded memory read and nothing here can raise
    once a recording has started: the listener that calls this swallows a
    raise and drops the frame, and a recording that lost every frame that
    way would end blaming the camera.

    Args:
        scope: The scope the recording runs against.
        to_plate: The plate transform bound when the recording started
            (``runtime_state.plate_transform()``), or None when the caller
            states no plate position; X and Y are then unknown.
    """
    positions = scope.motion.axis_positions()
    turret = scope.motion.turret_status()
    states = [p.state for p in positions.values()] + ([turret.state] if turret else [])
    moving = any(state in (AxisState.MOVING, AxisState.HOMING) for state in states)
    x = positions.get('X')
    y = positions.get('Y')
    z = positions.get('Z')
    plate_x_mm = plate_y_mm = None
    if (
        to_plate is not None
        and x is not None
        and y is not None
        and x.position_um is not None
        and y.position_um is not None
    ):
        plate = to_plate(x.position_um, y.position_um)
        plate_x_mm, plate_y_mm = plate.x_mm, plate.y_mm
    return FrameFact(
        plate_x_mm=plate_x_mm,
        plate_y_mm=plate_y_mm,
        z_um=z.position_um if z is not None else None,
        moving=moving,
    )


class RecordingCamera(NamedTuple):
    """The camera facts every frame of one recording shares, read as it
    starts: the binning cannot change while a recording runs."""

    binning_size: int
    camera_model: str | None
    camera_tick_hz: int | None


def recording_camera(scope: 'Lumascope') -> RecordingCamera:
    """The camera facts a recording's frames share, read at its start.

    Raises:
        CameraReadFailedError: The binning is unknown and the camera does
            not answer its read.
    """
    camera = scope.capabilities
    return RecordingCamera(
        binning_size=scope.camera.binning or scope.camera.read_binning(),
        camera_model=camera.camera_model,
        camera_tick_hz=camera.camera_timestamp_tick_hz,
    )


class Arrival(NamedTuple):
    """What a recorded frame states, read as it arrived: where it was taken
    (``fact``), the channel it is named for (``light``), and the
    instrument's record of it (``record``)."""

    fact: FrameFact
    light: common_utils.Light
    record: FrameRecord


def arrival(
    scope: 'Lumascope',
    frame: 'Frame',
    *,
    frame_time: FrameTime,
    channel_tiebreak: str,
    to_plate: Callable[[float, float], 'PlatePosition'] | None,
    camera: RecordingCamera,
) -> Arrival:
    """What a recorded frame states, read as it arrives.

    The light is one read of the LEDs, so the channel a frame is named for
    and the currents its record states cannot disagree. Called on the camera
    callback, under ``frame_fact``'s rule that nothing here raises: the LED
    states and the held layer are memory reads.

    Args:
        scope: The scope the recording runs against.
        frame: The frame as the camera delivered it.
        frame_time: Its time, with what that time is.
        channel_tiebreak: The channel to record when no LED is lit -- the
            one the recording started on -- so luminescence frames are named.
        to_plate: As for ``frame_fact``.
        camera: The recording's camera facts (``recording_camera``).
    """
    from modules.lumascope_api.illumination import snapshot_lit_pairs

    readings = scope.illumination.get_led_states()
    return Arrival(
        fact=frame_fact(scope, to_plate=to_plate),
        light=common_utils.light_of(readings, channel_tiebreak),
        record=delivered_frame_record(
            captured_at=frame_time,
            chunks=frame.chunks,
            frame_significant_bits=frame.significant_bits,
            lit=dict(snapshot_lit_pairs(readings)),
            held_channel=scope.camera.get_held_layer(),
            binning_size=camera.binning_size,
            camera_model=camera.camera_model,
            camera_tick_hz=camera.camera_tick_hz,
        ),
    )


def write_frame(
    scope: 'Lumascope',
    image: np.ndarray,
    file_loc: pathlib.Path,
    arrived: Arrival,
    *,
    frame_number: int,
    objective_id: str,
    labware: 'Labware',
    fallback_focus: str | None,
    false_color_on: bool,
    channel_colors: 'ChannelColors',
    save_encoding: str,
) -> None:
    """Write one recorded frame through the still's builder and writer
    (``image_save.write_recorded_frame``), stating what it arrived with.

    Args:
        scope: The scope the recording runs against.
        image: The frame, oriented and fitted (``orient_and_fit``).
        file_loc: Output path, from the recording's template: its suffix
            names the container.
        arrived: What the frame states, read as it arrived (``arrival``).
        frame_number: Its ordinal within the recording.
        objective_id: The objective the recording started with.
        labware: The plate its positions are stated on, which names its well.
        fallback_focus: Why the step's autofocus did not focus, for a video
            step recorded at its fallback focus; None otherwise.
        false_color_on: Whether the channel's false-color toggle was on.
        channel_colors: Every layer's colour as the recording began.
        save_encoding: The resolved image-mode save encoding.
    """
    fact, light, record = arrived
    image_save.write_recorded_frame(
        scope,
        image,
        file_loc,
        channel=light.channel,
        plate_x_mm=fact.plate_x_mm,
        plate_y_mm=fact.plate_y_mm,
        stage_z_um=fact.z_um,
        stage_moving=fact.moving,
        frame_number=frame_number,
        objective_id=objective_id,
        frame_record=record,
        labware=labware,
        well_label=fact.well_label(labware),
        fallback_focus=fallback_focus,
        false_color_on=false_color_on,
        channel_colors=channel_colors,
        save_encoding=save_encoding,
    )


def filename_time(arrived: Arrival) -> str:
    """The frame's time for a file name: colon-free, to the millisecond,
    safe on Windows."""
    return arrived.record.captured_at.at.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
