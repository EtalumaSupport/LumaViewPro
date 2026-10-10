# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The live view: the camera's newest frame, as JPEG, for any browser or MJPEG reader.

``GET /api/v1/live`` is ``multipart/x-mixed-replace``, one JPEG part per
frame the client is ready for; ``GET /api/v1/live.jpg`` is one frame. Both
are display -- 8-bit grey, lossy, no metadata; data is a capture, a file
in the live folder.

One frame listener serves every watcher, attached while anyone watches. It
keeps a copy of the newest frame and returns, on the camera's thread; each
client encodes the newest frame when it is ready for one, so a slow client
skips frames rather than falling behind, and the camera is never held up.
Each part carries ``X-Frame-Ordinal``, a count of the frames received
while anyone watched, never reset, so a client can tell what it skipped. A camera removed while
watched leaves the stream holding: its listener is put back when the
camera returns, and the frames resume.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import threading
from collections.abc import AsyncIterator

import numpy as np
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from modules import image_utils
from modules.scope_session import ScopeSession

# How long a snapshot nobody else is watching waits for the camera's next frame.
SNAPSHOT_WAIT_S = 2.0
# What a snapshot that found no frame tells the client to wait before asking again.
RETRY_AFTER_S = 1
BOUNDARY = 'frame'
MEDIA_TYPE = f'multipart/x-mixed-replace; boundary={BOUNDARY}'


@dataclasses.dataclass(frozen=True)
class Frame:
    """A frame the view holds: its own copy, and what it needs to be shown."""

    ordinal: int
    image: np.ndarray
    significant_bits: int
    timestamp: datetime.datetime

    def headers(self) -> dict[str, str]:
        return {
            'X-Frame-Ordinal': str(self.ordinal),
            'X-Frame-Timestamp': self.timestamp.isoformat(),
        }


def encode(frame: Frame, max_width: int | None) -> bytes:
    """The frame in 8-bit grey, as JPEG, no wider than *max_width* (half its width when None)."""
    image = image_utils.convert_to_8bit(frame.image, frame.significant_bits)
    height, width = image.shape[:2]
    target = max(1, width // 2) if max_width is None else max_width
    image = image_utils.decimate_for_preview(
        image, (target, max(1, round(height * target / width)))
    )
    return image_utils.encode_image(image, fmt='jpeg')


class LiveView:
    """The newest frame, for every client watching it."""

    def __init__(self, session: ScopeSession) -> None:
        self._imaging = session.scope.imaging
        # Guards the newest frame and its ordinal, written on the camera's thread.
        self._lock = threading.Lock()
        self._newest: Frame | None = None
        self._ordinal = 0
        self._watchers = 0
        self._finished = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._attaching: asyncio.Lock | None = None

    @property
    def watchers(self) -> int:
        """How many clients watch: a stream each, and each snapshot while it waits."""
        return self._watchers

    async def watch(self) -> None:
        """Join the watchers; the first attaches the frame listener.

        Raises what ``add_frame_listener`` raises -- a camera that is not
        connected is refused here, before anything is sent.
        """
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
            self._wake = asyncio.Event()
            self._attaching = asyncio.Lock()
        async with self._attaching:
            if self._watchers == 0:
                await asyncio.to_thread(
                    self._imaging.add_frame_listener, self._on_frame, 'REST live view'
                )
            self._watchers += 1

    async def leave(self) -> None:
        """Leave the watchers; the last removes the frame listener and lets go of its frame."""
        async with self._attaching:
            self._watchers -= 1
            if self._watchers == 0:
                await asyncio.to_thread(self._imaging.remove_frame_listener, self._on_frame)
                with self._lock:
                    # Not shown to the next watcher: the scope may have moved since.
                    self._newest = None

    def finish(self) -> None:
        """End every stream and snapshot: the server is closing. Called on the server's loop."""
        self._finished = True
        if self._wake is not None:
            self._woken()

    async def parts(self, max_width: int | None) -> AsyncIterator[bytes]:
        """One watcher's stream, joined by ``watch``: a part per frame it is ready for.

        Leaving is the caller's, after the stream ends however it ends: a
        stream never iterated -- its client gone before the first byte --
        runs no ``finally`` of its own.
        """
        sent = 0
        while (frame := await self._newer_than(sent)) is not None:
            sent = frame.ordinal
            jpeg = await asyncio.to_thread(encode, frame, max_width)
            head = ''.join(f'{k}: {v}\r\n' for k, v in frame.headers().items())
            yield (
                (
                    f'--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n'
                    f'Content-Length: {len(jpeg)}\r\n{head}\r\n'
                ).encode()
                + jpeg
                + b'\r\n'
            )

    async def snapshot(self, max_width: int | None) -> tuple[Frame, bytes] | None:
        """The newest frame and its JPEG, waiting up to ``SNAPSHOT_WAIT_S`` for one; None if none came.

        Raises what ``watch`` raises.
        """
        await self.watch()
        try:
            frame = await asyncio.wait_for(self._newer_than(0), SNAPSHOT_WAIT_S)
        except TimeoutError:
            return None
        finally:
            await self.leave()
        if frame is None:
            return None
        return frame, await asyncio.to_thread(encode, frame, max_width)

    async def _newer_than(self, ordinal: int) -> Frame | None:
        """The newest frame once it is newer than *ordinal*; None once the view has finished."""
        while True:
            wake = self._wake
            if self._finished:
                return None
            with self._lock:
                newest = self._newest
            if newest is not None and newest.ordinal > ordinal:
                return newest
            await wake.wait()

    def _on_frame(self, image: np.ndarray, timestamp: datetime.datetime, chunks: object) -> None:
        # On the camera's thread: copy and return. The depth is stamped
        # with the frame on this thread before its listeners are called.
        bits = self._imaging.last_significant_bits
        copy = image.copy()
        with self._lock:
            self._ordinal += 1
            self._newest = Frame(self._ordinal, copy, bits, timestamp)
        self._loop.call_soon_threadsafe(self._woken)

    def _woken(self) -> None:
        wake, self._wake = self._wake, asyncio.Event()
        wake.set()


class Watching(StreamingResponse):
    """A watcher's MJPEG response; it leaves the view when the response ends, on every exit."""

    def __init__(self, live: LiveView, max_width: int | None) -> None:
        super().__init__(
            live.parts(max_width), media_type=MEDIA_TYPE, headers={'Cache-Control': 'no-cache'}
        )
        self._live = live

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._live.leave()
