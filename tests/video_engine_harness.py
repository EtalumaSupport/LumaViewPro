# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Test infrastructure for the video recording engine contract tests.

Three pieces: a fake clock the engine accepts as its injected time
source, a synthetic frame feed producing (image, timestamp, chunks)
triples without a camera, and a writer stub whose blocking/failure
behavior the tests script. None of them sleep on the wall clock, so
every timing contract runs deterministically at full speed.
"""

import pathlib
import threading

import numpy as np


class FakeClock:
    """Injectable time source: a float the test advances explicitly."""

    def __init__(self, start: float = 1000.0):
        self._now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


class FrameFeed:
    """Synthetic camera: produces (image, timestamp_s, chunks) triples.

    Frames are deliberately non-square so a width/height transposition
    in the engine or writer cannot pass unnoticed.
    """

    def __init__(self, width: int = 8, height: int = 6, dtype=np.uint8):
        self.width = width
        self.height = height
        self.dtype = dtype
        self._count = 0

    def frame(self, timestamp_s: float, with_camera_chunks: bool = True):
        """One delivered frame; pixel values encode the frame ordinal."""
        image = np.full((self.height, self.width), self._count % 256, dtype=self.dtype)
        chunks = {'ChunkTimestamp': int(timestamp_s * 1e9)} if with_camera_chunks else None
        self._count += 1
        return image, timestamp_s, chunks

    def frames(self, timestamps, with_camera_chunks: bool = True):
        """Triples for each timestamp in order."""
        return [self.frame(ts, with_camera_chunks) for ts in timestamps]


class WriterStub:
    """Scriptable writer edge: records writes; can block or fail on demand.

    Args:
        fail_frames: Frame numbers whose write raises (per-frame failure).
        blocked: Start with the gate closed so writes wedge until
            ``unblock`` -- the writer-lag scenarios.
        die_on_frame: Frame number whose write raises SystemExit,
            simulating writer-lane death (the fatal classification).
    """

    def __init__(
        self,
        out_dir: pathlib.Path,
        fail_frames=(),
        blocked: bool = False,
        die_on_frame: int | None = None,
    ):
        self.out_dir = pathlib.Path(out_dir)
        self.fail_frames = set(fail_frames)
        self.die_on_frame = die_on_frame
        self.written = []
        # Per-frame chunk metadata as delivered to the write edge, in
        # write order; parallel to written so existing tuple consumers
        # are untouched.
        self.written_chunks = []
        self._gate = threading.Event()
        if not blocked:
            self._gate.set()

    def unblock(self) -> None:
        self._gate.set()

    def block(self) -> None:
        self._gate.clear()

    def __call__(self, image, timestamp_s, frame_number, config, chunks=None) -> pathlib.Path:
        self._gate.wait()
        if self.die_on_frame is not None and frame_number == self.die_on_frame:
            raise SystemExit('writer lane death (scripted)')
        if frame_number in self.fail_frames:
            raise OSError(f'scripted write failure for frame {frame_number}')
        path = self.out_dir / f'frame_{frame_number:06d}.tiff'
        path.write_bytes(image.tobytes())
        self.written.append((frame_number, timestamp_s, path))
        self.written_chunks.append(chunks)
        return path


class ClaimStub:
    """Reference compare-and-claim: atomic, single owner, loud release."""

    def __init__(self):
        self._lock = threading.Lock()
        self.owner = None

    def try_claim(self, owner: str) -> bool:
        with self._lock:
            if self.owner is not None:
                return False
            self.owner = owner
            return True

    def release(self, owner: str) -> None:
        with self._lock:
            if self.owner != owner:
                raise RuntimeError(f'release by {owner!r} but owner is {self.owner!r}')
            self.owner = None


class NotifyRecorder:
    """Notification sink recording (severity, args, kwargs) calls."""

    def __init__(self):
        self.calls = []

    def _record(self, severity):
        def _call(*args, **kwargs):
            self.calls.append((severity, args, kwargs))

        return _call

    def __getattr__(self, name):
        if name in ('info', 'warning', 'error', 'critical'):
            return self._record(name)
        raise AttributeError(name)

    def severities(self):
        return [severity for severity, _, _ in self.calls]
