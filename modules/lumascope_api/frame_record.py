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

from modules.frame_validity import Judgment


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
            at, in the camera's own units (``ImagingAPI.get_black_level``).
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
            by the capture beside its grab (``ClockMapping``); None on a
            camera without a latch (``capabilities.camera_has_clock_latch``).
        judgments: The frame judged against each source's last change
            (``FrameValidity.judgments``): for a summed capture, the first
            frame summed. Empty when no change has been recorded.
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

    def __post_init__(self):
        # Frozen all the way down: the record travels to another thread and
        # is read there, so its mapping must not be shared with the capture's.
        object.__setattr__(
            self, 'illumination_ma', types.MappingProxyType(dict(self.illumination_ma))
        )
