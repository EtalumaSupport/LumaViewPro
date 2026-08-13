"""VFR (per-frame pts) contract for the production VideoWriter.

The spike (test_video_vfr_encode_spike.py) proved the pinned av build
carries caller-assigned pts through an MP4 round trip; these tests pin
the same contract on the production class: a VideoWriter(vfr=True)
recording plays a delivery stall at its true duration, and the mode is
strictly opt-in -- the default writer keeps nominal-rate (CFR) timing.

Real encodes by design (the timing contract lives in the container, not
in any attribute a fake could capture). av is a hard dependency of the
video path, so an absent or broken build must FAIL here, never skip.
"""

import datetime
import math

import av
import numpy as np
import pytest

from modules.video_writer import VideoWriter

# A 5-frame recording with a 1.5 s delivery stall between the third and
# fourth frames. VFR means the stall must survive the encode round trip.
FRAME_TIMES_S = [0.0, 0.1, 0.2, 1.7, 1.8]
NOMINAL_FPS = 30


def _decode_frame_times(path):
    """Return each decoded frame's presentation time in seconds."""
    times = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            times.append(float(frame.pts * frame.time_base))
    return times


def _write_stalled_recording(path, *, vfr, timestamps=FRAME_TIMES_S):
    writer = VideoWriter(
        output_path=path,
        fps=NOMINAL_FPS,
        width=64,
        height=48,
        color='Red',
        vfr=vfr,
    )
    for i, ts in enumerate(timestamps):
        frame = np.full((48, 64), 40 * i, dtype=np.uint8)
        writer.add_frame(image=frame, timestamp=ts)
    writer.close()
    return writer


class TestVfrTiming:
    def test_per_frame_pts_survive_mp4_round_trip(self, tmp_path):
        out = tmp_path / 'vfr.mp4'
        _write_stalled_recording(out, vfr=True)

        decoded = _decode_frame_times(out)
        assert len(decoded) == len(FRAME_TIMES_S)
        for got, want in zip(decoded, FRAME_TIMES_S, strict=True):
            assert math.isclose(got, want, abs_tol=1e-3), f'{got} != {want}'

    def test_stall_plays_at_true_duration(self, tmp_path):
        out = tmp_path / 'vfr_stall.mp4'
        _write_stalled_recording(out, vfr=True)

        decoded = _decode_frame_times(out)
        span = decoded[-1] - decoded[0]
        true_span = FRAME_TIMES_S[-1] - FRAME_TIMES_S[0]
        assert math.isclose(span, true_span, abs_tol=1e-3)
        # The discriminator against constant-frame-rate collapse: at the
        # nominal rate, 5 frames would span ~0.13 s.
        assert span > 10 * (len(FRAME_TIMES_S) / NOMINAL_FPS)

    def test_container_duration_reports_true_time(self, tmp_path):
        out = tmp_path / 'vfr_duration.mp4'
        _write_stalled_recording(out, vfr=True)

        with av.open(str(out)) as container:
            duration_s = container.duration / av.time_base
        # Container duration = last pts + one frame's display time; the
        # muxer infers the tail frame duration, so allow half a second.
        assert duration_s >= FRAME_TIMES_S[-1] - 1e-3
        assert duration_s < FRAME_TIMES_S[-1] + 0.5

    def test_datetime_timestamps_encode_the_same_timeline(self, tmp_path):
        # Manual-path callers hold datetimes, the engine holds epoch
        # floats; pts is relative to the first frame either way.
        origin = datetime.datetime(2026, 8, 8, 12, 0, 0)
        stamps = [origin + datetime.timedelta(seconds=s) for s in FRAME_TIMES_S]
        out = tmp_path / 'vfr_datetime.mp4'
        _write_stalled_recording(out, vfr=True, timestamps=stamps)

        decoded = _decode_frame_times(out)
        for got, want in zip(decoded, FRAME_TIMES_S, strict=True):
            assert math.isclose(got, want, abs_tol=1e-3), f'{got} != {want}'


class TestVfrContract:
    def test_vfr_requires_a_timestamp_per_frame(self, tmp_path):
        writer = VideoWriter(
            output_path=tmp_path / 'vfr_missing_ts.mp4',
            fps=NOMINAL_FPS,
            width=64,
            height=48,
            color='Red',
            vfr=True,
        )
        frame = np.zeros((48, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match='timestamp'):
            writer.add_frame(image=frame)

    def test_default_writer_keeps_nominal_rate_timing(self, tmp_path):
        # The same stalled timestamps through a non-VFR writer must NOT
        # leak into presentation times: CFR callers keep today's timing.
        out = tmp_path / 'cfr.mp4'
        _write_stalled_recording(out, vfr=False)

        decoded = _decode_frame_times(out)
        span = decoded[-1] - decoded[0]
        assert span == pytest.approx((len(FRAME_TIMES_S) - 1) / NOMINAL_FPS, abs=1e-3)
