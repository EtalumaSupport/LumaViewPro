"""Regression tests for build_session_manifest length/None handling.

The manual-record buffers are preallocated ``[None] * max_frames``, so a
recording shorter than the buffer leaves a None-padded tail, and a frame with
an incompatible shape is skipped, leaving an interior None. The manifest builder
must bound to captured_frames and treat any None as a frame with no host
timestamp -- never crash on ``None.isoformat()`` or ``None - datetime``.

Both cadence paths are exercised: cameras that deliver hardware timestamp ticks
(tick_freq_hz set -> the isoformat start/end + per-frame reads) and cameras with
none (the host-timestamp fps fallback).
"""

import datetime

from modules.recording_manifest import build_session_manifest

_TICK_HZ = 1_000_000_000
_T0 = datetime.datetime(2026, 7, 23, 12, 0, 0, 100000)
_T1 = datetime.datetime(2026, 7, 23, 12, 0, 0, 300000)
_T2 = datetime.datetime(2026, 7, 23, 12, 0, 0, 500000)


def _chunk(ts):
    return {'Timestamp': ts, 'FrameID': ts}


def test_manifest_ignores_none_padded_tail():
    # 2 captured frames in a 4-slot preallocated buffer: the tail is None.
    # Ticks present -> pre-fix this hit end_iso = timestamps[-1] = None.isoformat().
    manifest = build_session_manifest(
        timestamps=[_T0, _T1, None, None],
        chunks_per_frame=[_chunk(10), _chunk(20), None, None],
        tick_freq_hz=_TICK_HZ,
        captured_frames=2,
        video_duration=1.0,
    )
    rec = manifest['recording']
    assert rec['frames_captured'] == 2
    assert rec['start_iso'] == _T0.isoformat(timespec='microseconds')
    # end must come from the last REAL frame, not the None tail slot.
    assert rec['end_iso'] == _T1.isoformat(timespec='microseconds')
    assert len(manifest['frame_index']) == 2


def test_manifest_handles_interior_skipped_frame():
    # Slot 1 was skipped (incompatible-shape frame) -> interior None. Ticks
    # present -> pre-fix this hit the per-frame timestamps[1] = None.isoformat().
    manifest = build_session_manifest(
        timestamps=[_T0, None, _T2],
        chunks_per_frame=[_chunk(10), None, _chunk(30)],
        tick_freq_hz=_TICK_HZ,
        captured_frames=3,
        video_duration=1.0,
    )
    rec = manifest['recording']
    assert rec['frames_captured'] == 3
    assert rec['start_iso'] == _T0.isoformat(timespec='microseconds')
    assert rec['end_iso'] == _T2.isoformat(timespec='microseconds')
    # The skipped frame is represented honestly as a null timestamp, kept in place.
    index = manifest['frame_index']
    assert len(index) == 3
    assert index[1]['ts_host_iso'] is None
    assert index[0]['ts_host_iso'] == _T0.isoformat(timespec='microseconds')
    assert index[2]['ts_host_iso'] == _T2.isoformat(timespec='microseconds')


def test_manifest_none_tail_on_host_timestamp_fallback():
    # No camera ticks -> the fps fallback subtracts consecutive timestamps;
    # pre-fix the None tail hit None - datetime in compute_fps_stats.
    manifest = build_session_manifest(
        timestamps=[_T0, _T1, None, None],
        chunks_per_frame=[None, None, None, None],
        tick_freq_hz=None,
        captured_frames=2,
        video_duration=1.0,
        channel_color='Red',
    )
    rec = manifest['recording']
    assert rec['frames_captured'] == 2
    assert rec['end_iso'] == _T1.isoformat(timespec='microseconds')
    # The video builder reads recording.channel_color to false-color a mono
    # video; the crash previously lost the sidecar and the color with it.
    assert rec['channel_color'] == 'Red'
