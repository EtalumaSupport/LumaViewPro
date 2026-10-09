# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The live view shows the camera's newest frame, to every client watching (``rest.live``).

``GET /api/v1/live`` is MJPEG and ``GET /api/v1/live.jpg`` one JPEG, each
half the frame's width unless ``max_width`` says otherwise; each frame
carries its ordinal, and a client ready for a frame gets the newest,
skipping the ones it was not ready for rather than falling behind. The frame listener is attached only while someone
watches. A snapshot the camera sends no frame for is answered 503
``no_frame_yet``; a camera that is not connected is refused before
anything is sent; the server's close refuses a new watch and ends the
open ones.

Read from a real uvicorn server on the simulator: an MJPEG response never
ends, and only a socket lets a client stop reading one.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import cv2
import httpx
import numpy as np
import pytest
import uvicorn
from starlette.requests import ClientDisconnect

from modules.scope_session import ScopeSession
from rest.app import build_app
from rest.live import LiveView, Watching
from tests.settings_fixtures import complete_settings
from tests.test_a_camera_command_with_no_camera_is_refused import _camera_less
from tests.test_a_command_for_absent_motion_hardware_is_refused import (
    make_session,  # noqa: F401 -- the fixture the camera-less test takes
)


@pytest.fixture
def session(tmp_path):
    live = tmp_path / 'live'
    live.mkdir()
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    yield s
    s.shutdown()


@contextlib.contextmanager
def _serving(session):
    app = build_app(session)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=0, log_level='warning'))
    thread = threading.Thread(target=server.run)
    thread.start()
    try:
        while not server.started:
            assert thread.is_alive(), 'the server ended before it started'
            thread.join(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=10) as client:
            yield app, server, client
    finally:
        server.should_exit = True
        thread.join(10)
        assert not thread.is_alive()


class _Parts:
    """An MJPEG response's parts, read one at a time: ``(headers, jpeg bytes)``."""

    def __init__(self, response: httpx.Response) -> None:
        self._bytes = response.iter_raw()
        self._buffer = b''

    def _fill(self) -> None:
        chunk = next(self._bytes, None)
        if chunk is None:
            raise AssertionError('the stream ended')
        self._buffer += chunk

    def next(self) -> tuple[dict[str, str], bytes]:
        while b'\r\n\r\n' not in self._buffer:
            self._fill()
        head, self._buffer = self._buffer.split(b'\r\n\r\n', 1)
        lines = head.decode().split('\r\n')
        assert lines[0] == '--frame', lines[0]
        headers = dict(line.split(': ', 1) for line in lines[1:])
        length = int(headers['Content-Length'])
        while len(self._buffer) < length + 2:
            self._fill()
        jpeg, self._buffer = self._buffer[:length], self._buffer[length + 2 :]
        return headers, jpeg


def _decoded(jpeg: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None, 'not a JPEG'
    return image


def _frame_width(session) -> int:
    image, _ = session.scope.imaging.get_image_from_buffer(force_to_8bit=True)
    return image.shape[1]


def _until(condition, what: str, timeout_s: float = 10.0) -> None:
    # The view's watchers change on the server's loop when it sees the
    # client go; nothing signals the test thread, so this waits, bounded.
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, f'never {what}'
        time.sleep(0.01)


def test_a_snapshot_is_the_newest_frame_at_half_its_width(session):
    width = _frame_width(session)
    with _serving(session) as (_app, _server, client):
        shot = client.get('/api/v1/live.jpg')
        narrow = client.get('/api/v1/live.jpg', params={'max_width': 64})

    assert shot.status_code == 200
    assert shot.headers['content-type'] == 'image/jpeg'
    assert int(shot.headers['x-frame-ordinal']) >= 1
    image = _decoded(shot.content)
    assert image.ndim == 2  # grey, as the screen shows it
    assert image.shape[1] == width // 2
    assert _decoded(narrow.content).shape[1] == 64


def test_a_snapshot_with_no_frame_is_refused_until_one_comes(session):
    session.scope.imaging.stop_streaming()
    with _serving(session) as (app, _server, client):
        refused = client.get('/api/v1/live.jpg')
        session.scope.imaging.start_streaming()
        shot = client.get('/api/v1/live.jpg')
        watchers = app.state.live.watchers

    assert refused.status_code == 503
    assert refused.json()['reason'] == 'no_frame_yet'
    assert refused.headers['retry-after'] == '1'
    assert shot.status_code == 200
    _decoded(shot.content)
    assert watchers == 0


@pytest.mark.parametrize(
    'query', [{'max_width': '0'}, {'max_width': 'wide'}, {'width': '64'}], ids=str
)
def test_a_width_that_is_not_a_positive_number_is_refused(session, query):
    with _serving(session) as (_app, _server, client):
        refused = [client.get(path, params=query) for path in ('/api/v1/live', '/api/v1/live.jpg')]

    assert [(r.status_code, r.json()['reason']) for r in refused] == [(422, 'invalid_request')] * 2


def test_the_listener_is_attached_only_while_someone_watches(session):
    with _serving(session) as (app, _server, client):
        live = app.state.live
        before = live.watchers
        with client.stream('GET', '/api/v1/live') as response:
            assert response.headers['content-type'] == 'multipart/x-mixed-replace; boundary=frame'
            _Parts(response).next()
            watching = live.watchers
        _until(lambda: live.watchers == 0, 'let go of the listener')

    assert (before, watching) == (0, 1)


def test_a_client_ready_for_a_frame_gets_the_newest_skipping_the_rest(session):
    # The view's own stream, read without a socket: a socket's buffers
    # would hold parts the client has not read yet, which hides the skip.
    skipped_by = 10
    delivered = threading.Semaphore(0)

    def counted(image, timestamp, chunks):
        delivered.release()

    async def watch() -> tuple[int, int]:
        live = LiveView(session)
        await live.watch()
        parts = live.parts(None)
        first = _ordinal(await anext(parts))
        session.scope.imaging.add_frame_listener(counted, 'test frame count')
        try:
            # The camera's own frames, counted as they are delivered.
            for _ in range(skipped_by + 1):
                assert await asyncio.to_thread(delivered.acquire, timeout=10)
        finally:
            session.scope.imaging.remove_frame_listener(counted)
        then = _ordinal(await anext(parts))
        await parts.aclose()
        await live.leave()
        return first, then

    first, then = asyncio.run(watch())

    assert then >= first + skipped_by


def _ordinal(part: bytes) -> int:
    head = part.split(b'\r\n\r\n', 1)[0].decode().split('\r\n')
    return int(dict(line.split(': ', 1) for line in head[1:])['X-Frame-Ordinal'])


def test_a_camera_that_is_not_connected_is_refused_before_any_frame(make_session, monkeypatch):
    session = _camera_less(make_session, monkeypatch, 'LS850')
    with _serving(session) as (app, _server, client):
        watched = client.get('/api/v1/live')
        shot = client.get('/api/v1/live.jpg')
        watchers = app.state.live.watchers

    for refused in (watched, shot):
        assert refused.headers['content-type'] == 'application/problem+json'
        assert refused.json()['reason'] == 'not_connected'
    assert watchers == 0


def test_the_close_refuses_a_new_watch_and_ends_the_open_ones(session):
    with _serving(session) as (app, server, client):
        with client.stream('GET', '/api/v1/live') as response:
            parts = _Parts(response)
            parts.next()
            app.state.closing.begin()
            refused = [client.get(path) for path in ('/api/v1/live', '/api/v1/live.jpg')]
            # The open stream keeps showing frames until the close finishes.
            parts.next()
            server.servers[0].get_loop().call_soon_threadsafe(app.state.closing.finish)
            with pytest.raises(AssertionError, match='the stream ended'):
                while True:
                    parts.next()
        _until(lambda: app.state.live.watchers == 0, 'let go of the listener')

    assert [(r.status_code, r.json()['reason']) for r in refused] == [(503, 'server_closing')] * 2


def test_a_client_gone_before_its_first_byte_leaves_no_watcher(session):
    # A stream whose client is gone before the response starts is never
    # iterated, so nothing inside it runs; the response itself must leave.
    async def send(message):
        raise OSError('the client has gone')

    async def watch_and_lose() -> tuple[int, int]:
        live = LiveView(session)
        await live.watch()
        joined = live.watchers
        with pytest.raises(ClientDisconnect):
            await Watching(live, None)(
                {'type': 'http', 'asgi': {'spec_version': '2.4'}}, None, send
            )
        return joined, live.watchers

    assert asyncio.run(watch_and_lose()) == (1, 0)
