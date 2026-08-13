# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Headless proof that the shipped PyAV build encodes variable-frame-rate
MP4 with per-frame presentation timestamps.

The video MP4 leg derives its timing authority from real capture
timestamps (per-frame pts against a fine time_base), not from the
container's nominal rate -- a delivery stall must play at its true
duration. Everything downstream of that design assumes the pinned
``av`` build can (a) import, (b) accept caller-assigned pts, and
(c) preserve them through an MP4 mux/demux round trip. This module is
that proof. ``av`` is a hard dependency of the video path (no cv2/AVI
fallback), so an absent or broken build must FAIL here, never skip.
"""

import fractions
import math

import numpy as np

# Hard import by design: if av is missing the whole module errors loudly.
import av

# Camera timestamps for a 5-frame recording with a 1.5 s delivery stall
# between the third and fourth frames. VFR means the stall must survive
# the encode round trip as true time.
FRAME_TIMES_S = [0.0, 0.1, 0.2, 1.7, 1.8]
# 90 kHz is the conventional MPEG timescale: fine enough that sub-ms
# capture jitter survives quantization.
TIME_BASE = fractions.Fraction(1, 90000)


def _encode_vfr(path, frame_times_s):
    """Encode small gray frames at the given true timestamps.

    B-frames must be disabled for VFR: MP4 container duration sums
    dts-based sample durations, and x264's B-frame reordering compacts
    dts while the true times ride pts offsets -- frames decode at the
    right times but the container reports a fraction of the real
    duration (measured here: 0.43 s for a 1.8 s recording). With bf=0,
    dts == pts and the container duration is honest.
    """
    with av.open(str(path), mode='w') as container:
        stream = container.add_stream('h264', rate=30, options={'bf': '0'})
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = 'yuv420p'
        stream.codec_context.time_base = TIME_BASE

        for i, ts in enumerate(frame_times_s):
            gray = np.full((48, 64), 40 * i, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(gray, format='gray').reformat(format='yuv420p')
            frame.pts = round(ts / TIME_BASE)
            frame.time_base = TIME_BASE
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def _decode_frame_times(path):
    """Return each decoded frame's presentation time in seconds."""
    times = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            times.append(float(frame.pts * frame.time_base))
    return times


class TestPyAvVfrEncode:
    def test_h264_encoder_is_available(self):
        # The MP4 leg has no fallback encoder; the bundled FFmpeg must
        # carry h264 or the build is unshippable.
        codec = av.codec.Codec('h264', mode='w')
        assert codec is not None

    def test_per_frame_pts_survive_mp4_round_trip(self, tmp_path):
        out = tmp_path / 'vfr_spike.mp4'
        _encode_vfr(out, FRAME_TIMES_S)

        decoded = _decode_frame_times(out)
        assert len(decoded) == len(FRAME_TIMES_S)
        for got, want in zip(decoded, FRAME_TIMES_S, strict=True):
            # MP4 muxers may rescale the stream time_base; the times must
            # survive to millisecond fidelity regardless.
            assert math.isclose(got, want, abs_tol=1e-3), f'{got} != {want}'

    def test_stall_plays_at_true_duration(self, tmp_path):
        out = tmp_path / 'vfr_stall.mp4'
        _encode_vfr(out, FRAME_TIMES_S)

        decoded = _decode_frame_times(out)
        span = decoded[-1] - decoded[0]
        true_span = FRAME_TIMES_S[-1] - FRAME_TIMES_S[0]
        assert math.isclose(span, true_span, abs_tol=1e-3)
        # The discriminator against constant-frame-rate collapse: at the
        # nominal 30 fps init rate, 5 frames would span ~0.13 s.
        nominal_span = len(FRAME_TIMES_S) / 30
        assert span > 10 * nominal_span

    def test_container_duration_reports_true_time(self, tmp_path):
        out = tmp_path / 'vfr_duration.mp4'
        _encode_vfr(out, FRAME_TIMES_S)

        with av.open(str(out)) as container:
            duration_s = container.duration / av.time_base
        # Container duration = last pts + one frame's display time; the
        # muxer infers the tail frame duration, so allow half a second.
        assert duration_s >= FRAME_TIMES_S[-1] - 1e-3
        assert duration_s < FRAME_TIMES_S[-1] + 0.5
