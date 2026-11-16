# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A request's body is bounded before the server holds it.

``_lumascope.rest.app``'s one body reader: a body larger than its route takes
is refused ``413`` ``body_too_large`` before more of it than the ceiling is
held, nothing written and nothing run -- a ``Content-Length`` over the
ceiling before a byte is read, a chunked body at the first byte past it. A
JSON route takes up to the JSON ceiling, the file upload up to the upload
ceiling. An upload takes its live call and one of the upload places before
its body is read, so it is refused ``503`` ``overloaded`` before the client
sends it, never counts past the live limit while it arrives, and gives both
back on every exit. A client that goes away mid-body is one REST-log line.

Read from a real uvicorn server over raw sockets: only a socket can stall a
body halfway, or drop it. A request's end is waited on by its REST-log
line, and an upload's admission by its place being taken, never by timing.
"""

from __future__ import annotations

import json
import logging
import socket
import threading

import pytest
import uvicorn

import _lumascope.rest.app
import _lumascope.rest.jobs
from _lumascope.modules.protocol import PROTOCOL_FILE_CEILING_BYTES
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from _lumascope.rest.jobs import JobRegistry
from tests.settings_fixtures import complete_settings

# Small ceilings, so a body past one costs nothing to send.
_UPLOAD_CEILING = 64
_JSON_CEILING = 64
_WAIT_S = 10


@pytest.fixture
def live(tmp_path):
    folder = tmp_path / 'live'
    folder.mkdir()
    return folder


@pytest.fixture
def session(live):
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    yield s
    s.shutdown()


class _RestLog(logging.Handler):
    """The REST log's lines as they are written, so a test waits for a request's end."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self._written = threading.Condition()

    def emit(self, record: logging.LogRecord) -> None:
        with self._written:
            self.lines.append(record.getMessage())
            self._written.notify_all()

    def ended(self, path: str) -> list[str]:
        """The lines of the requests to *path*, once there is one."""
        with self._written:
            assert self._written.wait_for(lambda: self.of(path), _WAIT_S), f'{path} never ended'
            return self.of(path)

    def of(self, path: str) -> list[str]:
        return [line for line in self.lines if f' {path} -> ' in line]


@pytest.fixture
def rest_log():
    handler = _RestLog()
    logger = logging.getLogger('LVP.rest')
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)


@pytest.fixture
def places_taken(monkeypatch):
    """Released once per upload place taken: the server's own word that an upload is being read."""
    taken = threading.Semaphore(0)
    take_upload = JobRegistry.take_upload

    # A spy by design: it calls the real member and only signals the take.
    def counted(self):
        place = take_upload(self)
        taken.release()
        return place

    monkeypatch.setattr(JobRegistry, 'take_upload', counted)
    return taken


@pytest.fixture
def port(session, monkeypatch):
    monkeypatch.setattr(_lumascope.rest.app, 'UPLOAD_CEILING_BYTES', _UPLOAD_CEILING)
    monkeypatch.setattr(_lumascope.rest.app, 'JSON_BODY_CEILING_BYTES', _JSON_CEILING)
    # No log config of uvicorn's own, so its records reach the test's log capture.
    server = uvicorn.Server(
        uvicorn.Config(build_app(session), host='127.0.0.1', port=0, log_config=None)
    )
    thread = threading.Thread(target=server.run)
    thread.start()
    try:
        while not server.started:
            assert thread.is_alive(), 'the server ended before it started'
            thread.join(0.01)
        yield server.servers[0].sockets[0].getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(10)
        assert not thread.is_alive()


def _sent(port: int, head: str, body: bytes = b'') -> socket.socket:
    """A connection that has sent *head*, a request line and its headers, then *body*."""
    connection = socket.create_connection(('127.0.0.1', port), timeout=_WAIT_S)
    connection.sendall(head.replace('\n', '\r\n').encode() + b'\r\n' + body)
    return connection


def _answer(connection: socket.socket) -> tuple[int, dict | None]:
    """The status and JSON body of the answer *connection* is sent, then closed."""
    with connection:
        data = b''
        while b'\r\n\r\n' not in data:
            data += connection.recv(65536)
        head, _, body = data.partition(b'\r\n\r\n')
        lines = head.decode().split('\r\n')
        headers = dict(line.lower().split(': ', 1) for line in lines[1:])
        length = int(headers.get('content-length', '0'))
        while len(body) < length:
            body += connection.recv(65536)
    return int(lines[0].split()[1]), json.loads(body) if body else None


def _upload(port: int, name: str, *, length: int, body: bytes = b'') -> socket.socket:
    return _sent(
        port, f'PUT /api/v1/files/{name} HTTP/1.1\nHost: t\nContent-Length: {length}\n', body
    )


def _chunked(body: bytes) -> bytes:
    return f'{len(body):x}\r\n'.encode() + body + b'\r\n0\r\n\r\n'


def _call(port: int, path: str, body: bytes, *, chunked: bool = False) -> socket.socket:
    framing = 'Transfer-Encoding: chunked' if chunked else f'Content-Length: {len(body)}'
    return _sent(
        port,
        f'POST /api/v1/{path} HTTP/1.1\nHost: t\nContent-Type: application/json\n{framing}\n',
        _chunked(body) if chunked else body,
    )


def _stalled(port: int, count: int, places_taken: threading.Semaphore) -> list[socket.socket]:
    """*count* uploads, each sent part of its body and then nothing, each holding its place."""
    stalled = []
    for i in range(count):
        stalled.append(_upload(port, f'stalled_{i}.bin', length=_UPLOAD_CEILING, body=b'ab'))
        assert places_taken.acquire(timeout=_WAIT_S), f'upload {i} took no place'
    return stalled


def test_the_ceilings_are_64_kib_of_json_and_four_of_the_largest_protocol_for_an_upload():
    assert _lumascope.rest.app.JSON_BODY_CEILING_BYTES == 64 * 1024
    assert _lumascope.rest.app.UPLOAD_CEILING_BYTES == 4 * PROTOCOL_FILE_CEILING_BYTES


def test_the_description_names_both_ceilings_so_a_client_knows_them_before_it_sends(session):
    """The 4XX range every route declares states the ceilings the server refuses past."""
    routes = build_app(session).openapi()['paths']
    for path, method in (('/api/v1/take_control', 'post'), ('/api/v1/files/{name}', 'put')):
        words = routes[path][method]['responses']['4XX']['description']
        assert f'{_lumascope.rest.app.JSON_BODY_CEILING_BYTES} bytes for JSON' in words
        assert f'{_lumascope.rest.app.UPLOAD_CEILING_BYTES} for a file upload' in words


def test_an_upload_past_its_ceiling_is_refused_before_its_body_and_writes_nothing(port, live):
    """Only the headers are sent: the refusal comes before a byte of the body."""
    status, problem = _answer(_upload(port, 'big.bin', length=_UPLOAD_CEILING + 1))

    assert (status, problem['reason']) == (413, 'body_too_large')
    assert str(_UPLOAD_CEILING) in problem['detail']
    assert not (live / 'big.bin').exists()
    at_ceiling = b'x' * _UPLOAD_CEILING
    assert _answer(_upload(port, 'full.bin', length=_UPLOAD_CEILING, body=at_ceiling))[0] == 201
    assert (live / 'full.bin').read_bytes() == at_ceiling


def test_a_chunked_upload_past_its_ceiling_is_refused_and_writes_nothing(port, live):
    connection = _sent(
        port,
        'PUT /api/v1/files/big.bin HTTP/1.1\nHost: t\nTransfer-Encoding: chunked\n',
        _chunked(b'x' * (_UPLOAD_CEILING + 1)),
    )
    status, problem = _answer(connection)

    assert (status, problem['reason']) == (413, 'body_too_large')
    assert not (live / 'big.bin').exists()


# parametrize-justified: a length and a chunked body are two branches of the reader.
@pytest.mark.parametrize('chunked', [False, True], ids=['with a length', 'chunked'])
def test_a_json_body_past_its_ceiling_is_refused_and_the_member_never_called(
    port, session, chunked
):
    """Over the ceiling ``take_control`` is refused and takes nothing; within it, it is called."""
    over = json.dumps({'name': 'x' * _JSON_CEILING}).encode()
    status, problem = _answer(_call(port, 'take_control', over, chunked=chunked))

    assert (status, problem['reason']) == (413, 'body_too_large')
    assert session.control is None
    assert _answer(_call(port, 'take_control', b'{"name": "a"}', chunked=chunked))[0] == 200
    assert session.control is not None


def test_a_call_with_no_body_is_answered_as_before(port):
    status, _ = _answer(
        _sent(port, 'POST /api/v1/create_empty_protocol HTTP/1.1\nHost: t\nContent-Length: 0\n')
    )
    assert status == 200


def test_with_every_upload_place_held_an_upload_is_refused_before_its_body_and_a_call_served(
    port, places_taken
):
    stalled = _stalled(port, _lumascope.rest.jobs.UPLOAD_PLACES, places_taken)
    try:
        status, problem = _answer(_upload(port, 'fifth.bin', length=3))
        assert (status, problem['reason']) == (503, 'overloaded')
        assert 'uploads are being read' in problem['detail']
        assert _answer(_call(port, 'scope/motion/stop_motion', b'{}'))[0] == 200
    finally:
        for connection in stalled:
            connection.close()


def test_an_upload_being_read_counts_live_so_none_passes_the_live_limit(
    port, places_taken, monkeypatch
):
    """With the live limit below the places, the uploads being read hold it and the next is refused."""
    monkeypatch.setattr(_lumascope.rest.jobs, 'LIVE_LIMIT', 2)
    stalled = _stalled(port, 2, places_taken)
    try:
        status, problem = _answer(_upload(port, 'third.bin', length=3))
        assert (status, problem['reason']) == (503, 'overloaded')
        assert 'calls are running' in problem['detail']
    finally:
        for connection in stalled:
            connection.close()


def test_every_exit_gives_its_place_and_live_call_back_once(
    port, live, places_taken, rest_log, monkeypatch
):
    """After a refusal, a drop, a replay and a write, exactly the places are free: none kept, none given twice."""
    places = _lumascope.rest.jobs.UPLOAD_PLACES
    # The live limit at the places, so a live call kept refuses a place too.
    monkeypatch.setattr(_lumascope.rest.jobs, 'LIVE_LIMIT', places)
    assert _answer(_upload(port, 'big.bin', length=_UPLOAD_CEILING + 1))[0] == 413
    _upload(port, 'dropped.bin', length=_UPLOAD_CEILING, body=b'ab').close()
    rest_log.ended('/api/v1/files/dropped.bin')
    keyed = (
        'PUT /api/v1/files/keyed.bin HTTP/1.1\nHost: t\nIdempotency-Key: k1\nContent-Length: 3\n'
    )
    assert _answer(_sent(port, keyed, b'abc'))[0] == 201
    assert _answer(_sent(port, keyed, b'abc'))[0] == 201
    # A write's thread gives its place back before its answer is made.
    assert _answer(_upload(port, 'written.bin', length=3, body=b'abc'))[0] == 201
    # Only the places taken from here are counted.
    while places_taken.acquire(blocking=False):
        pass

    stalled = _stalled(port, places, places_taken)
    try:
        status, problem = _answer(_upload(port, 'one_more.bin', length=3))
        assert (status, problem['reason']) == (503, 'overloaded')
    finally:
        for connection in stalled:
            connection.close()


# parametrize-justified: the upload and a JSON route each read through the reader on their own path.
@pytest.mark.parametrize(
    'head',
    [
        'PUT /api/v1/files/dropped.bin HTTP/1.1\nHost: t\nContent-Length: 50\n',
        'POST /api/v1/take_control HTTP/1.1\nHost: t\nContent-Type: application/json\nContent-Length: 50\n',
    ],
    ids=['upload', 'json'],
)
def test_a_client_that_drops_mid_body_is_one_rest_log_line_and_no_traceback(
    port, rest_log, caplog, head
):
    caplog.set_level(logging.INFO)
    _sent(port, head, b'{"').close()

    (line,) = rest_log.ended(head.split()[1])
    assert 'the client went away after' in line
    assert [r.getMessage() for r in caplog.records if r.exc_info] == []
