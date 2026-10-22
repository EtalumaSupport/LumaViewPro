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
import types
from collections.abc import Mapping

from modules.api_surface import api_fields
from modules.frame_validity import Judgment

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

    The host latches the camera's clock between two reads of its own
    ``time.monotonic_ns``; the latch fell somewhere inside that bracket, so
    its middle is the host time the latched value belongs to and half its
    width is how far off that can be. The read of the latched value comes
    after the bracket and is no part of it.

    Attributes:
        host_ns: ``time.monotonic_ns`` at the middle of the bracket.
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


@api_fields(
    'black_level',
    'binning_size',
    'camera_model',
    'camera_tick_hz',
    'camera_timestamp_ticks',
    'captured_at',
    'clock_mapping',
    'dark',
    'exposure_ms',
    'frame_id',
    'frame_significant_bits',
    'frames_summed',
    'gain_db',
    'held_layer',
    'illumination_ma',
    'judgments',
    'saturated_fraction',
)
@dataclasses.dataclass(frozen=True)
class FrameRecord:
    """The instrument's own account of one captured frame.

    A field the instrument did not report is None, never a stand-in: a
    stand-in written into a file reads downstream as a measurement.

    Attributes:
        captured_at: Host time when the grab returned the frame.
        exposure_ms: Per-frame exposure. For a summed capture this is each
            frame's exposure, not the total; ``frames_summed`` says how many.
        gain_db: Gain the frame was taken at.
        black_level: The camera's black level parameter the frame was taken
            at, in the camera's own units (``CameraAPI.get_black_level``).
        illumination_ma: Drive current of each LED address lit for the
            grab, by address. An address that was off is absent.
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
            frame summed. Empty when no change has been recorded.
        saturated_fraction: The fraction of the frame's pixels at or near
            full scale (``ImagingAPI.saturated_fraction``), measured when the
            capture asked for the check (``all_ones_check``); for a summed
            capture, its most saturated frame. None when it did not ask.
        dark: True when the frame was kept with no pixel above the dark
            floor while illumination was commanded on, after the capture
            waited out its budget for a lit one; for a summed capture, when
            any frame was. None when the capture did not judge it (no LED
            on, or ``accept_dark``).
        held_layer: The layer whose stored settings the camera held for the
            grab (``CameraAPI.get_held_layer``); None when it held no
            layer's, after a raw camera write or with the settings a run's
            step wrote. The frame's own exposure and gain are the fields
            above either way.
    """

    captured_at: datetime.datetime
    exposure_ms: float | None
    gain_db: float | None
    black_level: float | None
    illumination_ma: Mapping[int, float]
    frames_summed: int
    frame_significant_bits: int
    camera_timestamp_ticks: int | None
    camera_tick_hz: int | None
    frame_id: int | None
    binning_size: int
    camera_model: str | None
    clock_mapping: ClockMapping | None
    judgments: tuple[Judgment, ...]
    saturated_fraction: float | None
    dark: bool | None
    held_layer: str | None

    def __post_init__(self):
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
