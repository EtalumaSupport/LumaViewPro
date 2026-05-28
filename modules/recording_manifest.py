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


def compute_fps_stats(timestamps: list) -> dict:
    """Compute mean / min / max / sample-count FPS from frame timestamps.

    Returns zeros when fewer than 2 timestamps are available (no
    intervals to measure).
    """
    if not timestamps or len(timestamps) < 2:
        return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'samples': 0}
    intervals = []
    for i in range(1, len(timestamps)):
        dt = (timestamps[i] - timestamps[i - 1]).total_seconds()
        if dt > 0:
            intervals.append(1.0 / dt)
    if not intervals:
        return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'samples': 0}
    return {
        'mean': sum(intervals) / len(intervals),
        'min': min(intervals),
        'max': max(intervals),
        'samples': len(intervals),
    }


def build_session_manifest(
    timestamps: list,
    chunks_per_frame: list,
    tick_freq_hz: int | None,
    captured_frames: int,
    video_duration: float,
    camera_model: str | None = None,
    camera_serial: str | None = None,
    lvp_version: str | None = None,
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
        captured_frames: count of frames stored in the memmap.
        video_duration: wall-clock seconds from record_init to
            record_complete.
        camera_model: e.g. 'a2A3536-31umBAS' or None.
        camera_serial: device serial string or None.
        lvp_version: LumaViewPro version (line 1 of version.txt).

    Returns:
        dict suitable for json.dump.
    """
    fps_stats = compute_fps_stats(timestamps)
    recording_start_iso = timestamps[0].isoformat(timespec='microseconds') if timestamps else None
    recording_end_iso = timestamps[-1].isoformat(timespec='microseconds') if timestamps else None

    frame_index = []
    for i in range(captured_frames):
        ts_iso = timestamps[i].isoformat(timespec='microseconds') if i < len(timestamps) else None
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
