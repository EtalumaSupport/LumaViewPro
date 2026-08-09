# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Shared frame-edge helpers for video recording write lanes.

Every recording controller (manual today, protocol video steps next)
writes engine-delivered frames through the same physical edge: rebase
camera hardware ticks onto the host epoch, orient the raw camera array,
fit it to the recording geometry, and stamp per-frame TIFF metadata.
One home for that logic so the capture paths cannot drift on frame
identity or geometry.
"""

import datetime

import numpy as np

import modules.image_utils as image_utils


class CameraTickRebaser:
    """Host-epoch seconds for camera frames; hardware ticks when usable.

    Camera hardware ticks are the frame's own clock, free of the OS
    scheduling jitter host arrival stamps carry; rebasing them onto the
    host epoch at the first tick-carrying frame keeps camera-grade
    intervals on the axis cadence selection runs on. Without usable
    ticks the host arrival time is used and the manifest's timestamp
    grade reports it.

    One instance per recording: the rebase offset anchors at the first
    tick-carrying frame and must never survive into the next recording.
    """

    def __init__(self, tick_freq_hz: float | None, clock):
        self._freq = tick_freq_hz
        self._clock = clock
        self._offset: float | None = None

    def frame_time_s(self, timestamp, chunks) -> float:
        """Host-epoch seconds for one frame delivered by the SDK callback."""
        if isinstance(timestamp, datetime.datetime):
            host_s = timestamp.timestamp()
        elif timestamp is not None:
            host_s = float(timestamp)
        else:
            host_s = self._clock()
        ticks = chunks.get('Timestamp') if chunks else None
        if self._freq and ticks is not None:
            if self._offset is None:
                self._offset = host_s - ticks / self._freq
            return self._offset + ticks / self._freq
        return host_s


def orient_and_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Flip a raw camera frame to display orientation and fit the geometry.

    Orientation and contiguity are paid here at the write edge, never in
    the per-frame ingest callback.

    Raises:
        ValueError: The frame cannot fit the recording geometry. Costs
            exactly that frame -- the engine counts it as a write
            failure and the recording continues.
    """
    image = np.flip(image, 0)
    target_shape = (height, width)
    if image.shape != target_shape:
        fitted = image_utils.fit_frame_to_shape(image, target_shape)
        if fitted is None:
            raise ValueError(
                f'frame shape {image.shape} incompatible with recording geometry {target_shape}'
            )
        image = fitted
    return image


def tiff_frame_metadata(
    timestamp_s: float,
    frame_number: int,
    chunks,
    tick_freq_hz: float | None,
) -> tuple[dict, str]:
    """Per-frame TIFF metadata plus a path-safe timestamp string.

    The timestamp travels in metadata, not pixels -- Create Video draws
    it at build time when the overlay is enabled. Camera chunk identity
    (hardware ticks, FrameID) is recorded when the frame carried it.

    Returns:
        ``(metadata, ts_filename)`` -- the metadata dict for the TIFF
        writer, and a colon-free millisecond-precision timestamp string
        safe for Windows filenames.
    """
    ts = datetime.datetime.fromtimestamp(timestamp_s)
    ts_filename = ts.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
    metadata = {
        'datetime': ts.strftime('%Y:%m:%d %H:%M:%S'),
        'timestamp': ts.strftime('%Y:%m:%d %H:%M:%S.%f'),
        'timestamp_iso': ts.isoformat(timespec='microseconds'),
        'frame_num': frame_number,
    }
    if chunks is not None:
        ts_ticks = chunks.get('Timestamp')
        if ts_ticks is not None:
            metadata['timestamp_camera_ticks'] = int(ts_ticks)
        if tick_freq_hz is not None:
            metadata['timestamp_camera_tick_hz'] = int(tick_freq_hz)
        frame_id = chunks.get('FrameID')
        if frame_id is not None:
            metadata['frame_id'] = int(frame_id)
    return metadata, ts_filename
