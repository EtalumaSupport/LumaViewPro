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
        illumination_ma: Drive current of each channel lit for the grab,
            by layer name. A channel that was off is absent.
        frames_summed: Number of frames summed into the image.
        camera_timestamp_ticks: The camera's own clock for the frame
            (for a summed capture, the last frame's).
        camera_tick_hz: Ticks per second of that clock.
        frame_id: The camera's frame counter for the frame.
        binning_size: Binning the frame was delivered at.
        camera_model: The camera that took it.
    """

    captured_at: datetime.datetime
    exposure_ms: float | None
    gain_db: float | None
    black_level: float | None
    illumination_ma: Mapping[str, float]
    frames_summed: int
    camera_timestamp_ticks: int | None
    camera_tick_hz: int | None
    frame_id: int | None
    binning_size: int
    camera_model: str | None

    def __post_init__(self):
        # Frozen all the way down: the record travels to another thread and
        # is read there, so its mapping must not be shared with the capture's.
        object.__setattr__(
            self, 'illumination_ma', types.MappingProxyType(dict(self.illumination_ma))
        )
