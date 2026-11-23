"""What a run writes to disk: the one estimate of a step's write and its inputs.

A saved frame is sized from the facts that decide its bytes on disk, never a
flat figure: the frame's width and height, the camera's depth and color, the
image mode's capture depth and save encoding, and, per step, its sum, its
channel and whether false color is on. The run's disk checks (``prepare()``'s
whole-run refusal, the per-scan backstop, the per-write and per-video-step
checks) and the size advisory all size through ``estimate_step_write_mb``, so
they cannot drift apart.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from _lumascope.modules import channel_record, image_mode
from _lumascope.modules.acquire_mode import AcquireMode
from _lumascope.modules.common_utils import read_video_config
from _lumascope.modules.image_mode import RecordingFormat

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from _lumascope.modules.channel_record import Color
    from _lumascope.modules.scope_capabilities import FrameSize, ScopeCapabilities

_MIB = 1024**2

ESTIMATED_VIDEO_STEP_MB = 50  # floor for a short compressed MP4
_MP4_COMPRESSION_FRACTION = 0.1  # MP4 ~ a tenth of the raw per-frame bytes


@dataclasses.dataclass(frozen=True)
class FrameSizing:
    """The run-wide facts that decide what one saved frame takes on disk.

    Built once for a run (``for_camera``) and held on its plan, so every disk
    check of the run sizes with the same facts.

    Attributes:
        frame: The frames the camera delivers, in pixels after binning.
        camera_bits: The most bits the camera delivers in any of its formats
            (8 on an FX2): a single frame at full depth stays one byte a pixel
            on an 8-bit camera.
        color_native: True for a camera that delivers 3-channel color.
        capture_depth: The image mode's capture depth (8, or 12 for a
            full-depth mode).
        save_encoding: The image mode's save encoding; ``'rgb'`` widens a
            16-bit frame shown in a color to three channels.
        channel_colors: Each channel's color, as the writers color a frame.
        stacked: True when the run's sequenced output is a hyperstack of its
            stills, built after the run beside the TIFFs it reads, which it
            does not remove.
    """

    frame: FrameSize
    camera_bits: int
    color_native: bool
    capture_depth: int
    save_encoding: str
    channel_colors: Mapping[str, Color]
    stacked: bool

    @classmethod
    def for_camera(
        cls,
        frame: FrameSize | None,
        capabilities: ScopeCapabilities,
        capture_config: image_mode.ImageCaptureConfig,
        channel_colors: Mapping[str, Color],
    ) -> FrameSizing | None:
        """The sizing for frames from this camera saved as ``capture_config``
        says, or None when the camera's frame size or depth is not known.

        Raises:
            Nothing.
        """
        if frame is None or capabilities.camera_max_significant_bits is None:
            return None
        return cls(
            frame=frame,
            camera_bits=capabilities.camera_max_significant_bits,
            color_native=capabilities.camera_is_color,
            capture_depth=capture_config.capture_depth,
            save_encoding=capture_config.save_encoding,
            channel_colors=dict(channel_colors),
            stacked=capture_config.output_format_sequenced == image_mode.OutputFormat.HYPERSTACK,
        )

    def _shown_in_color(self, channel: str, false_color_on: bool) -> bool:
        return (
            channel_record.shown_color(channel, false_color_on, self.channel_colors)
            != channel_record.GRAY
        )

    def saved_frame_bytes(self, *, channel: str, sum_count: int, false_color_on: bool) -> int:
        """The bytes one saved TIFF frame of ``channel`` holds, uncompressed.

        The writers' own rules: the image is 8-bit in the 8-bit mode, and for
        a single frame from an 8-bit camera; otherwise a 16-bit container,
        a sum on every camera included (``ImagingAPI`` capture). A 16-bit
        frame shown in a color is widened to RGB by the ``'rgb'`` encoding
        (``image_utils.maybe_apply_false_color``); a color camera's frame is
        RGB already.

        Raises:
            ArgumentRefusedError: ``channel`` is no channel of this release
                (``channel_record.output_color``), when its color decides
                the size: a 16-bit frame under the ``'rgb'`` encoding.
        """
        if self.capture_depth == 8:
            container = 1
        elif sum_count > 1 or self.camera_bits > 8:
            container = 2
        else:
            container = 1
        widened = (
            container == 2
            and self.save_encoding == image_mode.SAVE_ENCODING_RGB
            and self._shown_in_color(channel, false_color_on)
        )
        planes = 3 if self.color_native or widened else 1
        return self.frame.width * self.frame.height * container * planes

    def mp4_frame_bytes(self, *, channel: str, false_color_on: bool) -> int:
        """The raw bytes one 8-bit MP4 frame of ``channel`` encodes from:
        three planes in a color, one for a gray encode.

        Raises:
            ArgumentRefusedError: as ``saved_frame_bytes``.
        """
        planes = 3 if self.color_native or self._shown_in_color(channel, false_color_on) else 1
        return self.frame.width * self.frame.height * planes


def _coerce_positive_float(value) -> float:
    """Coerce a Video Config numeric to a float >= 0, defaulting bad input to 0.

    Video Config values come from a user-edited protocol and may be a
    non-numeric string or a NaN; float() raises on the former and NaN poisons
    the frame-count arithmetic. Both collapse to 0 (a missing dimension), which
    floors the estimate rather than raising and disabling the disk guard.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    # NaN != NaN; a NaN duration means "unknown", treated as 0.
    return result if result == result and result > 0 else 0.0


def estimate_step_write_mb(
    step: pd.Series | Mapping[str, object],
    *,
    sizing: FrameSizing,
    recording_format: RecordingFormat = RecordingFormat.MP4,
    video_fps: float | None,
) -> float:
    """Estimate the disk a single protocol step will write, in MB.

    One owner for write-size estimation, shared by every disk check of a run
    and the size advisory, so they cannot drift. An image step writes one
    saved frame (``FrameSizing.saved_frame_bytes`` at the step's sum). A
    video step scales with the recording: the frame count is duration_s *
    the rate the recording is sized at (``video_sizing_fps``: the global
    limit, else the camera's delivered rate), and each frame is one saved
    frame when the recording format writes frames, or a compressed fraction
    of its 8-bit frame in an MP4, never under the MP4 floor. A stack's
    second copy is not counted here: it is the run's, built after the run
    (``Protocol.estimate_write_mb``).

    Args:
        step: A protocol step; reads Acquire, channel, Sum, False_Color and
            Video Config.
        sizing: The run's frame sizing.
        recording_format: How the run saves a video step.
        video_fps: The rate a video step is sized at
            (``video_cadence.video_sizing_fps``); None when it is not known,
            which only a protocol with no video step may pass.

    Returns:
        Estimated megabytes the step will write to disk.

    Raises:
        ValueError: ``step`` is a video step and ``video_fps`` is None.
        ArgumentRefusedError: the step's channel is no channel of this
            release, when its color decides the size
            (``FrameSizing.saved_frame_bytes``).
    """
    channel = step['channel']
    false_color_on = bool(step['False_Color'])
    if step['Acquire'] != AcquireMode.VIDEO:
        return (
            sizing.saved_frame_bytes(
                channel=channel, sum_count=int(step['Sum']), false_color_on=false_color_on
            )
            / _MIB
        )
    if video_fps is None:
        raise ValueError('A video step cannot be sized without the rate it records at.')
    duration_s = _coerce_positive_float(read_video_config(step).get('duration'))
    frames = max(1, int(duration_s * video_fps))
    if recording_format.writes_frames:
        return (
            frames
            * sizing.saved_frame_bytes(channel=channel, sum_count=1, false_color_on=false_color_on)
            / _MIB
        )
    return max(
        ESTIMATED_VIDEO_STEP_MB,
        frames
        * sizing.mp4_frame_bytes(channel=channel, false_color_on=false_color_on)
        * _MP4_COMPRESSION_FRACTION
        / _MIB,
    )
