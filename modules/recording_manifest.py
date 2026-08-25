# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Session manifest helpers for the manual-record path (issue #633 Stage 2B).

Pure functions for building the per-recording session_manifest.json file
that lives next to the saved TIFFs. Kept Kivy-free so it can be unit-
tested directly.

Schema mirrors the char tool's manifest provenance shape so
manifests across LVP recordings and char runs are comparable. The
frame_index gives downstream scripts a single
source of truth for frame ordering and per-frame timestamps without
having to read every TIFF.
"""

from __future__ import annotations

import platform
import socket


def gather_host_provenance() -> dict:
    """Return the host fingerprint dict.

    Mirrors the char tool's provenance schema. Failures fall back to
    empty strings so the manifest write doesn't abort.
    """
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = ''
    try:
        os_platform = platform.platform()
    except Exception:
        os_platform = ''
    try:
        cpu_model = platform.processor() or platform.machine()
    except Exception:
        cpu_model = ''
    try:
        py_version = platform.python_version()
    except Exception:
        py_version = ''
    return {
        'hostname': hostname,
        'os_platform': os_platform,
        'cpu_model': cpu_model,
        'python_version': py_version,
    }


def _fps_stats_from_interval_seconds(interval_seconds: list) -> dict:
    """Mean / min / max / sample-count FPS from interframe interval durations.

    The one interval-to-stats core shared by the host-timestamp and
    camera-tick paths, so both report FPS identically. Non-positive
    intervals (duplicate or out-of-order stamps) are dropped. Returns
    zeros when no positive interval remains.
    """
    rates = [1.0 / dt for dt in interval_seconds if dt > 0]
    if not rates:
        return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'samples': 0}
    return {
        'mean': sum(rates) / len(rates),
        'min': min(rates),
        'max': max(rates),
        'samples': len(rates),
    }


def compute_fps_stats(timestamps: list) -> dict:
    """Compute FPS stats from per-frame host wall-clock datetimes.

    Returns zeros when fewer than 2 timestamps are available (no
    intervals to measure). Host wall-clock carries OS scheduling jitter;
    prefer compute_fps_stats_from_ticks when the camera reports hardware
    timestamps.
    """
    if not timestamps or len(timestamps) < 2:
        return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'samples': 0}
    intervals = [
        (timestamps[i] - timestamps[i - 1]).total_seconds() for i in range(1, len(timestamps))
    ]
    return _fps_stats_from_interval_seconds(intervals)


def compute_fps_stats_from_ticks(ticks: list, tick_freq_hz: int | None) -> dict:
    """Compute FPS stats from the camera's own hardware timestamp ticks.

    Ticks are the frame's own clock, free of the OS scheduling jitter that
    host wall-clock stamps pick up between grab and callback, so they report
    the true frame cadence. Returns zeros (caller falls back to host time)
    when the tick frequency is unknown or fewer than 2 ticks are available.
    """
    if not tick_freq_hz or tick_freq_hz <= 0 or not ticks or len(ticks) < 2:
        return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'samples': 0}
    seconds = [t / tick_freq_hz for t in ticks]
    intervals = [seconds[i] - seconds[i - 1] for i in range(1, len(seconds))]
    return _fps_stats_from_interval_seconds(intervals)


def build_session_manifest(
    timestamps: list,
    chunks_per_frame: list,
    tick_freq_hz: int | None,
    captured_frames: int,
    video_duration: float,
    camera_model: str | None = None,
    camera_serial: str | None = None,
    lvp_version: str | None = None,
    channel_color: str | None = None,
) -> dict:
    """Build the session_manifest.json dict for one manual-video recording.

    Args:
        timestamps: per-frame host wall-clock datetimes from the grab
            callback. May be shorter than captured_frames if the camera
            dropped late frames; missing entries become None in the
            frame_index.
        chunks_per_frame: per-frame chunks dict from the camera SDK
            (keys 'Timestamp', 'FrameID', etc). May contain None for
            cameras without chunk support.
        tick_freq_hz: camera-side timestamp tick rate (1 GHz on Basler
            USB3; from GevTimestampTickFrequency on GigE; None when
            unknown).
        captured_frames: count of frames actually recorded -- only the
            first captured_frames slots of the record buffers hold real
            frames.
        video_duration: wall-clock seconds from record_init to
            record_complete.
        camera_model: e.g. 'a2A3536-31umBAS' or None.
        camera_serial: device serial string or None.
        lvp_version: LumaViewPro version (line 1 of version.txt).
        channel_color: active channel/layer color for the recording
            ('Red', 'Green', 'Blue', 'Lumi', ...), or None for mono /
            brightfield. Consumed by the Create-Video build so a
            false-colored recording renders in color, not grayscale.

    Returns:
        dict suitable for json.dump.
    """
    # Only the first captured_frames slots hold real frames: the record buffers
    # are preallocated [None] * max_frames, so the unused tail stays None, and a
    # frame whose ndim/channel-count is incompatible with the buffer is skipped,
    # leaving an interior None. Bound to the captured count and treat any None as
    # a frame that carries no host timestamp.
    timestamps = timestamps[:captured_frames]
    chunks_per_frame = chunks_per_frame[:captured_frames]
    valid_timestamps = [t for t in timestamps if t is not None]

    # Prefer the camera's hardware timestamp ticks for the FPS stats: the host
    # wall-clock stamps carry OS scheduling jitter between grab and callback,
    # which widens the min/max spread and misreports true cadence. Ticks are
    # the frame's own clock. Fall back to host time when the camera has no
    # timestamp chunk (tick_freq_hz None) or too few ticks landed.
    frame_ticks = [c['Timestamp'] for c in chunks_per_frame if c and c.get('Timestamp') is not None]
    if tick_freq_hz and len(frame_ticks) >= 2:
        fps_stats = compute_fps_stats_from_ticks(frame_ticks, tick_freq_hz)
    else:
        fps_stats = compute_fps_stats(valid_timestamps)
    recording_start_iso = (
        valid_timestamps[0].isoformat(timespec='microseconds') if valid_timestamps else None
    )
    recording_end_iso = (
        valid_timestamps[-1].isoformat(timespec='microseconds') if valid_timestamps else None
    )

    frame_index = []
    for i in range(captured_frames):
        ts = timestamps[i] if i < len(timestamps) else None
        ts_iso = ts.isoformat(timespec='microseconds') if ts is not None else None
        chunks = chunks_per_frame[i] if i < len(chunks_per_frame) else None
        ts_ticks = chunks.get('Timestamp') if chunks else None
        frame_id = chunks.get('FrameID') if chunks else None
        frame_index.append(
            {
                'i': i,
                'ts_host_iso': ts_iso,
                'ts_camera_ticks': int(ts_ticks) if ts_ticks is not None else None,
                'frame_id': int(frame_id) if frame_id is not None else None,
            }
        )

    return {
        'manifest_version': 1,
        'recording': {
            'start_iso': recording_start_iso,
            'end_iso': recording_end_iso,
            'frames_captured': captured_frames,
            'duration_s': float(video_duration),
            'actual_fps': fps_stats,
            'channel_color': channel_color,
        },
        'camera': {
            'model': camera_model,
            'serial': camera_serial,
            'timestamp_tick_hz': int(tick_freq_hz) if tick_freq_hz is not None else None,
        },
        'provenance': {
            'host': gather_host_provenance(),
            'software': {
                'lvp_version': lvp_version,
            },
        },
        'frame_index': frame_index,
    }
