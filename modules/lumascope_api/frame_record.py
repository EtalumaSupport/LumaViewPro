# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the instrument reported about one captured frame, taken with the frame.

A saved file is written after its frame is taken -- sometimes much later,
behind a backlog, a Stop or an error abort -- and by then the camera, the
LED and the stage have moved on. A record built when the file is written
describes the scope at the write, not the frame: the next step's gain, no
LED current once the run has turned the light off. So the capture takes
these facts in the same camera-lane task as the grab and hands them on with
the frame, and the writer reads nothing live.
"""

import dataclasses
import datetime
import enum
import types
from collections.abc import Callable, Mapping
from typing import NamedTuple

import numpy as np

from modules import image_utils
from modules.api_surface import api_fields
from modules.frame_validity import Judgment
from modules.lumascope_api.camera import AutoGainConvergence

# A frame with at least this fraction of its pixels saturated is blown -- a
# stale-gain or over-exposure symptom. Set high so a legitimately bright
# field does not trip it; a true blown-white frame saturates essentially
# every pixel. The capture retries a blown frame once against it, and a
# record answers ``blown`` by it.
BLOWN_FRACTION = 0.98


@api_fields('host_ns', 'camera_ticks', 'tick_hz', 'uncertainty_ns')
@dataclasses.dataclass(frozen=True)
class ClockMapping:
    """The camera's timestamp clock against the host's steady clock, measured once.

    The host latches the camera's clock between two reads of the scope's
    steady clock (``host_clock.steady_ns``); the latch fell somewhere inside
    that bracket, so its middle is the host time the latched value belongs to
    and half its width is how far off that can be. The read of the latched
    value comes after the bracket and is no part of it.

    Attributes:
        host_ns: The scope's steady clock (``host_clock.steady_ns``) at the
            middle of the bracket.
        camera_ticks: The camera's clock the latch latched.
        tick_hz: Ticks per second of the camera's clock.
        uncertainty_ns: Half the bracket.
    """

    host_ns: int
    camera_ticks: int
    tick_hz: int
    uncertainty_ns: int

    def camera_ticks_at(self, host_ns: int) -> int:
        """The camera's clock at the host's steady time ``host_ns``, through
        this mapping: off by ``uncertainty_ns`` and by whatever the clocks
        drifted between ``host_ns`` and the latch."""
        return self.camera_ticks + round((host_ns - self.host_ns) * self.tick_hz / 1e9)


class SettingSource(enum.StrEnum):
    """Where a record's exposure or gain came from.

    FRAME: the frame's own chunk value, stamped by the camera on the frame
        (Pylon ace 2 / dart, the simulated camera).
    READ: a read of the camera's setting beside the grab, for a frame that
        carries no chunk value (IDS, FX2, a Basler frame whose chunk was not
        delivered): the setting in force, not a measurement of the frame.
    """

    FRAME = 'frame'
    READ = 'read'


class TimestampSource(enum.StrEnum):
    """What a record's time is.

    HOST_WALLCLOCK: the host's wall clock read after the grab returned the
        frame (a still).
    HOST_ARRIVAL: the host's wall clock when the frame arrived from the
        camera (a recorded frame from a camera that stamps no ticks).
    CAMERA_REBASED: the camera's own clock for the frame, offset onto the
        host's wall clock at the recording's first frame that carried it.
    """

    HOST_WALLCLOCK = 'host_wallclock'
    HOST_ARRIVAL = 'host_arrival'
    CAMERA_REBASED = 'camera_rebased'


@api_fields('at', 'source')
@dataclasses.dataclass(frozen=True)
class FrameTime:
    """When a frame was taken, with what that time is.

    One value, so a time is never stated without its source: two clocks
    give a frame its time, and a reader comparing frames must know which.

    Attributes:
        at: The frame's time, host local.
        source: What ``at`` is (``TimestampSource``).
    """

    at: datetime.datetime
    source: TimestampSource


class LightTiming(enum.StrEnum):
    """When a record's LED currents were read.

    BEFORE_GRAB: before the grab that took the frame (a still): the LEDs
        lit for it.
    ON_ARRIVAL: as the frame arrived from the camera (a recorded frame):
        the LEDs lit then, which is after the frame's exposure.
    """

    BEFORE_GRAB = 'before_grab'
    ON_ARRIVAL = 'on_arrival'


@api_fields(
    'auto_gain',
    'black_level',
    'binning_size',
    'camera_model',
    'camera_tick_hz',
    'camera_timestamp_ticks',
    'captured_at',
    'clock_mapping',
    'dark',
    'exposure_ms',
    'exposure_source',
    'frame_id',
    'frame_significant_bits',
    'frames_summed',
    'gain_db',
    'gain_source',
    'held_channel',
    'illumination_ma',
    'illumination_timing',
    'judgments',
    'saturated_fraction',
)
@dataclasses.dataclass(frozen=True)
class FrameRecord:
    """The instrument's own account of one captured frame.

    A field the instrument did not report is None, never a stand-in: a
    stand-in written into a file reads downstream as a measurement.

    Built only by this module's two entries: ``grab_record`` for a still,
    ``delivered_frame_record`` for a recorded frame.

    Attributes:
        captured_at: When the frame was taken, with what that time is
            (``FrameTime``).
        exposure_ms: Per-frame exposure. For a summed capture this is each
            frame's exposure, not the total; ``frames_summed`` says how many.
            None for a recorded frame that carries no exposure of its own.
        exposure_source: Whether ``exposure_ms`` is the frame's own or the
            setting read beside the grab (``SettingSource``); None exactly
            when ``exposure_ms`` is.
        gain_db: Gain the frame was taken at; None as ``exposure_ms`` is.
        gain_source: Whether ``gain_db`` is the frame's own or the setting
            read beside the grab; None exactly when ``gain_db`` is.
        black_level: The camera's black level parameter the frame was taken
            at, in the camera's own units (``CameraAPI.read_black_level``).
            None for a recorded frame: no frame reports it.
        illumination_ma: Drive current of each LED address lit for the
            frame, by address. An address that was off is absent.
        illumination_timing: When those currents were read
            (``LightTiming``).
        frames_summed: Number of frames summed into the image.
        frame_significant_bits: The depth each summed frame was delivered at
            (the driver's per-frame stamp). With ``frames_summed`` it is what
            the sum's tag, its rendering and its saturation are worked out
            from (``image_utils.summed_significant_bits``, ``summed_full_scale``,
            ``convert_sum_to_8bit``).
        camera_timestamp_ticks: The camera's own clock for the frame
            (for a summed capture, the last frame's).
        camera_tick_hz: Ticks per second of that clock.
        frame_id: The camera's frame counter for the frame.
        binning_size: Binning the frame was delivered at.
        camera_model: The camera that took it.
        clock_mapping: The camera's clock mapped onto the host's, latched
            by the capture at the top of the cycle that grabbed the frame,
            before its drain (``ClockMapping``); None on a
            camera without a latch (``capabilities.camera_has_clock_latch``).
        judgments: The frame judged against each source's last change
            (``FrameValidity.judgments``): for a summed capture, the first
            frame summed. Empty when no change has been recorded; None for
            a recorded frame, which is never judged.
        saturated_fraction: The fraction of the frame's pixels at or near
            full scale (``ImagingAPI.saturated_fraction``), measured when the
            capture asked for the check (``all_ones_check``); for a summed
            capture, its most saturated frame. None when it did not ask.
        dark: True when the frame was kept with no pixel above the dark
            floor while illumination was commanded on, after the capture
            waited out its budget for a lit one; for a summed capture, when
            any frame was. None when the capture did not judge it (no LED
            on, or ``accept_dark``).
        held_channel: The channel whose stored settings the camera held for the
            grab (``CameraAPI.get_held_layer``); None when it held no
            layer's, after a raw camera write or with the settings a run's
            step wrote. The frame's own exposure and gain are the fields
            above either way.
        auto_gain: Where the standing auto-gain arm landed when the capture
            locked it (``AutoGainConvergence``); None when no arm stood.
            The lock's exposure and gain are ``exposure_ms`` and ``gain_db``.
    """

    captured_at: FrameTime
    exposure_ms: float | None
    exposure_source: SettingSource | None
    gain_db: float | None
    gain_source: SettingSource | None
    black_level: float | None
    illumination_ma: Mapping[int, float]
    illumination_timing: LightTiming
    frames_summed: int
    frame_significant_bits: int
    camera_timestamp_ticks: int | None
    camera_tick_hz: int | None
    frame_id: int | None
    binning_size: int
    camera_model: str | None
    clock_mapping: ClockMapping | None
    judgments: tuple[Judgment, ...] | None
    saturated_fraction: float | None
    dark: bool | None
    held_channel: str | None
    auto_gain: AutoGainConvergence | None

    def __post_init__(self):
        # A setting is stated with where it came from, or not at all.
        for value, source, name in (
            (self.exposure_ms, self.exposure_source, 'exposure'),
            (self.gain_db, self.gain_source, 'gain'),
        ):
            if (value is None) != (source is None):
                raise ValueError(f'a record states its {name} and its source together, or neither')
        # Frozen all the way down: the record travels to another thread and
        # is read there, so its mapping must not be shared with the capture's.
        object.__setattr__(
            self, 'illumination_ma', types.MappingProxyType(dict(self.illumination_ma))
        )

    @property
    def blown(self) -> bool | None:
        """Whether the frame is blown (``BLOWN_FRACTION``); None when its
        saturation was not measured."""
        if self.saturated_fraction is None:
            return None
        return self.saturated_fraction >= BLOWN_FRACTION


class _ChunkFacts(NamedTuple):
    """What a frame's own chunks state, in the record's units; None where
    the frame carried none."""

    exposure_ms: float | None
    gain_db: float | None
    ticks: int | None
    frame_id: int | None


def _chunk_facts(chunks: Mapping[str, int | float] | None) -> _ChunkFacts:
    """The one reading of a frame's chunks, for a still and a recorded frame
    alike: ``ExposureTime`` arrives in microseconds."""
    chunks = chunks or {}
    exposure_us = chunks.get('ExposureTime')
    ticks = chunks.get('Timestamp')
    frame_id = chunks.get('FrameID')
    return _ChunkFacts(
        exposure_ms=exposure_us / 1000.0 if exposure_us is not None else None,
        gain_db=chunks.get('Gain'),
        ticks=int(ticks) if ticks is not None else None,
        frame_id=int(frame_id) if frame_id is not None else None,
    )


def grab_record(
    *,
    captured_at: datetime.datetime,
    chunks: Mapping[str, int | float],
    read_exposure_ms: Callable[[], float],
    read_gain_db: Callable[[], float],
    black_level: float | None,
    lit: Mapping[int, float],
    frames_summed: int,
    frame_significant_bits: int,
    binning_size: int,
    camera_model: str | None,
    camera_tick_hz: int | None,
    clock_mapping: ClockMapping | None,
    judgments: tuple[Judgment, ...],
    saturated_fraction: float | None,
    dark: bool | None,
    held_channel: str | None,
    auto_gain: AutoGainConvergence | None,
) -> FrameRecord:
    """The record of a still, built on the grab's lane beside the grab.

    Exposure and gain are the frame's chunk values where the camera stamps
    them; a frame without them takes ``read_exposure_ms`` / ``read_gain_db``,
    the camera's setting read here, beside the grab -- the nearest the frame
    has to its own account. A read the camera does not answer raises and
    fails the capture. Its time is the host's wall clock after the grab
    returned.
    """
    facts = _chunk_facts(chunks)
    if facts.exposure_ms is not None:
        exposure_ms, exposure_source = facts.exposure_ms, SettingSource.FRAME
    else:
        exposure_ms, exposure_source = read_exposure_ms(), SettingSource.READ
    if facts.gain_db is not None:
        gain_db, gain_source = facts.gain_db, SettingSource.FRAME
    else:
        gain_db, gain_source = read_gain_db(), SettingSource.READ
    return FrameRecord(
        captured_at=FrameTime(at=captured_at, source=TimestampSource.HOST_WALLCLOCK),
        exposure_ms=exposure_ms,
        exposure_source=exposure_source,
        gain_db=gain_db,
        gain_source=gain_source,
        black_level=black_level,
        illumination_ma=lit,
        illumination_timing=LightTiming.BEFORE_GRAB,
        frames_summed=frames_summed,
        frame_significant_bits=frame_significant_bits,
        camera_timestamp_ticks=facts.ticks,
        camera_tick_hz=camera_tick_hz,
        frame_id=facts.frame_id,
        binning_size=binning_size,
        camera_model=camera_model,
        clock_mapping=clock_mapping,
        judgments=judgments,
        saturated_fraction=saturated_fraction,
        dark=dark,
        held_channel=held_channel,
        auto_gain=auto_gain,
    )


def delivered_frame_record(
    *,
    captured_at: FrameTime,
    chunks: Mapping[str, int | float] | None,
    frame_significant_bits: int,
    lit: Mapping[int, float],
    held_channel: str | None,
    binning_size: int,
    camera_model: str | None,
    camera_tick_hz: int | None,
) -> FrameRecord:
    """The record of a recorded frame, built on the camera callback as it arrives.

    Nothing here reads the camera: the callback must not raise, and a read
    there would be a second camera call per frame. So the frame states only
    what it carries -- exposure, gain, ticks and frame id from its chunks,
    none where it has no chunks, since a setting can change while the
    recording runs -- and what was true as it arrived: the LEDs lit and the
    layer the camera held. The binning, camera and tick rate are the
    recording's, fixed while it runs. No black level (no frame reports
    it), no judgments (a recorded frame is never judged), no clock
    mapping, saturation or auto-gain lock.
    """
    facts = _chunk_facts(chunks)
    return FrameRecord(
        captured_at=captured_at,
        exposure_ms=facts.exposure_ms,
        exposure_source=SettingSource.FRAME if facts.exposure_ms is not None else None,
        gain_db=facts.gain_db,
        gain_source=SettingSource.FRAME if facts.gain_db is not None else None,
        black_level=None,
        illumination_ma=lit,
        illumination_timing=LightTiming.ON_ARRIVAL,
        frames_summed=1,
        frame_significant_bits=frame_significant_bits,
        camera_timestamp_ticks=facts.ticks,
        camera_tick_hz=camera_tick_hz,
        frame_id=facts.frame_id,
        binning_size=binning_size,
        camera_model=camera_model,
        clock_mapping=None,
        judgments=None,
        saturated_fraction=None,
        dark=None,
        held_channel=held_channel,
        auto_gain=None,
    )


@api_fields('image', 'record')
@dataclasses.dataclass(frozen=True, eq=False)
class CapturedStill:
    """One captured still: its pixels and the instrument's record of them, together.

    Handed back by the grab that took them, so no caller reads a frame's
    record from a slot the next capture replaces. Equality is identity: a
    frozen dataclass's own would compare the arrays elementwise and raise.

    Attributes:
        image: The captured array, read-only: the frame, a sum, or its 8-bit
            rendering.
        record: The instrument's account of the frame (``FrameRecord``).
    """

    image: np.ndarray
    record: FrameRecord

    @property
    def significant_bits(self) -> int:
        """The depth of ``image``: 8 for an 8-bit array, else the bits its sum
        can reach (``image_utils.summed_significant_bits``); a single frame
        is its own depth. The one depth rule every save, evidence and display
        consumer shares, from the frame's own record: a depth read later from
        the camera describes the camera then, not this frame."""
        if self.image.dtype == np.uint8:
            return 8
        return image_utils.summed_significant_bits(
            self.record.frames_summed, self.record.frame_significant_bits
        )

    @property
    def full_scale(self) -> int:
        """The value at which ``image`` is saturated: 255 for an 8-bit array,
        else what its sum can hold (``image_utils.summed_full_scale``), never
        the depth's power of two."""
        if self.image.dtype == np.uint8:
            return 255
        return image_utils.summed_full_scale(
            self.record.frames_summed, self.record.frame_significant_bits
        )
