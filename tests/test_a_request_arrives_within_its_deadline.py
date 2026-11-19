# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A REST request arrives within its deadlines, or its connection is ended.

The server's builder (``_lumascope.rest.server.build_server``) gives a
connection the header deadline for its request's headers, from its opening
and from each answer to the next request, whatever is left of a body the
answer did not read; a body its route reads (``_lumascope.rest.app``'s one
reader) arrives within the body deadline -- a floor plus its length at a
floor rate -- or is refused ``408`` ``body_too_slow`` and its connection
closed, an upload's place given back. A request that has arrived is never
ended by them, however long its answer takes.

Read from the builder's own server over raw sockets, the deadlines made
short: only a socket can fall silent halfway. A connection's end is its
EOF, or a reset while the client is still sending; a connection kept is one still open past the deadline, so those tests
wait out a short span by design.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

import _lumascope.rest.app
import _lumascope.rest.server
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from _lumascope.rest.server import build_server
from tests.settings_fixtures import complete_settings

_HEADER_S = 0.5
_BODY_FLOOR_S = 2.0
_BYTES_PER_S = 1000
# Past every deadline here, short of a hang.
_WAIT_S = 10
# How long a connection kept is watched: well past the header deadline.
_KEPT_S = 4 * _HEADER_S
_UPLOAD_LENGTH = 1000


@pytest.fixture
def live(tmp_path):
    folder = tmp_path / 'live'
    folder.mkdir()
    return folder


@pytest.fixture
def port(live, monkeypatch):
    monkeypatch.setattr(_lumascope.rest.server, 'HEADER_DEADLINE_S', _HEADER_S)
    monkeypatch.setattr(_lumascope.rest.app, 'BODY_DEADLINE_FLOOR_S', _BODY_FLOOR_S)
    monkeypatch.setattr(_lumascope.rest.app, 'BODY_FLOOR_BYTES_PER_S', _BYTES_PER_S)
    with socket.create_server(('127.0.0.1', 0)) as probe:
        free = probe.getsockname()[1]
    settings = complete_settings(live_folder=str(live))
    settings['rest_api'] = {
        'enabled': True,
        'address': '127.0.0.1',
        'port': free,
        'allowed_clients': [],
    }
    session = ScopeSession.create(settings, simulate=True)
    server = build_server(session, shut_down=session.shutdown)
    thread = threading.Thread(target=server.run)
    thread.start()
    try:
        while not server.serving:
            assert thread.is_alive(), 'the server ended before it served'
            thread.join(0.01)
        yield free
    finally:
        server.close()
        thread.join(_WAIT_S)
        assert not thread.is_alive()


# Where a request names the server: filled in with its address and port as
# it is sent, since the server answers no request addressed to another.
_SERVED = b'SERVED'


def _connected(port: int, sent: bytes = b'') -> socket.socket:
    connection = socket.create_connection(('127.0.0.1', port), timeout=_WAIT_S)
    if sent:
        connection.sendall(sent.replace(_SERVED, f'127.0.0.1:{port}'.encode()))
    return connection


def _until_closed(connection: socket.socket) -> bytes:
    """Everything the server sends *connection* until it ends it; a timeout if it never does.

    An end is a close, or a reset: a byte the client sent that lands after
    the server's last read makes its close a reset, which the kernel
    chooses and the deadline does not.
    """
    received = b''
    with connection:
        try:
            while chunk := connection.recv(65536):
                received += chunk
        except ConnectionResetError:
            pass
    return received


def _kept(connection: socket.socket) -> bool:
    """Whether the server keeps *connection* open for ``_KEPT_S``, whatever it sends meanwhile."""
    connection.settimeout(_KEPT_S)
    try:
        while connection.recv(65536):
            pass
    except TimeoutError:
        return True
    return False


def _head(answer: bytes) -> tuple[int, dict[str, str]]:
    """The status and headers of the first answer in *answer*."""
    head = answer.partition(b'\r\n\r\n')[0].decode().split('\r\n')
    fields = dict(line.split(': ', 1) for line in head[1:])
    return int(head[0].split()[1]), {k.lower(): v for k, v in fields.items()}


def _trickled(connection: socket.socket, data: bytes) -> threading.Thread:
    """A thread sending *data* a byte every tenth of a second, until the server closes *connection*."""

    def send() -> None:
        for byte in data:
            try:
                connection.sendall(bytes([byte]))
            except OSError:
                return
            time.sleep(0.1)

    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    return thread


def _upload_head(name: str, length: int = _UPLOAD_LENGTH, extra: str = '') -> bytes:
    return (
        f'PUT /api/v1/files/{name} HTTP/1.1\r\nHost: SERVED\r\n'
        f'Content-Type: application/octet-stream\r\nContent-Length: {length}\r\n{extra}\r\n'
    ).encode()


def test_the_description_states_the_body_deadline_so_a_client_knows_it_before_it_sends(live):
    session = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    try:
        routes = build_app(session).openapi()['paths']
    finally:
        session.shutdown()
    words = routes['/api/v1/files/{name}']['put']['responses']['4XX']['description']
    assert (
        f'408: a body that did not arrive within {_lumascope.rest.app.BODY_DEADLINE_FLOOR_S:g} s '
        f'plus its length at {_lumascope.rest.app.BODY_FLOOR_BYTES_PER_S} bytes a second'
    ) in words


def test_a_connection_that_sends_nothing_is_closed_at_the_header_deadline(port):
    assert _until_closed(_connected(port)) == b''


def test_headers_sent_a_byte_at_a_time_past_the_deadline_are_closed(port):
    connection = _connected(port)
    _trickled(connection, b'GET /api HTTP/1.1\r\nHost: x\r\n' + b'X-Slow: y\r\n' * 20)

    assert _until_closed(connection) == b''


@pytest.mark.parametrize(
    ('head', 'status'),
    [
        (b'POST /nothing HTTP/1.1\r\nHost: SERVED\r\nContent-Length: 100000\r\n\r\n', 404),
        (_upload_head('big.bin', length=_lumascope.rest.app.UPLOAD_CEILING_BYTES + 1), 413),
    ],
    ids=['not_found', 'body_too_large'],
)
def test_a_body_its_answer_did_not_read_is_closed_at_the_header_deadline(port, head, status):
    """Answered before its body is read; the rest trickled after: the next request's deadline still runs."""
    connection = _connected(port, head)
    _trickled(connection, b'z' * 200)

    assert _head(_until_closed(connection))[0] == status


@pytest.mark.parametrize(
    'head',
    [
        b'POST /api/v1/take_control HTTP/1.1\r\nHost: SERVED\r\n'
        b'Content-Type: application/json\r\nContent-Length: 50\r\n\r\n',
        _upload_head('stalled.bin'),
    ],
    ids=['json', 'upload'],
)
def test_a_body_that_falls_silent_is_refused_408_and_its_connection_closed(port, live, head):
    connection = _connected(port, head + b'{')

    status, fields = _head(_until_closed(connection))

    assert (status, fields['connection']) == (408, 'close')
    assert not (live / 'stalled.bin').exists()


def test_four_silent_uploads_give_their_places_back_at_their_deadlines(port, live):
    """The case the deadline is for: four clients gone silent mid-upload, every upload refused until they end."""
    stalled = [_connected(port, _upload_head(f's{i}.bin') + b'z' * 10) for i in range(4)]
    try:
        for connection in stalled:
            assert _head(_until_closed(connection))[0] == 408
    finally:
        # Closed by the client should the server never close them, so the
        # server's own close is not left waiting on a body.
        for connection in stalled:
            connection.close()

    connection = _connected(port, _upload_head('fifth.bin') + b'z' * _UPLOAD_LENGTH)

    assert _head(connection.recv(65536))[0] == 201
    connection.close()
    assert (live / 'fifth.bin').read_bytes() == b'z' * _UPLOAD_LENGTH


def test_a_body_s_deadline_is_its_own_length_s_not_its_route_s_ceiling(port):
    """At the ceiling's length, an upload's deadline would be far past the wait."""
    started = time.monotonic()
    status, _ = _head(_until_closed(_connected(port, _upload_head('small.bin'))))

    assert status == 408
    assert time.monotonic() - started < _BODY_FLOOR_S + _UPLOAD_LENGTH / _BYTES_PER_S + 1


def test_an_event_stream_open_past_both_deadlines_is_kept(port):
    connection = _connected(port, b'GET /api/v1/events HTTP/1.1\r\nHost: SERVED\r\n\r\n')

    assert _head(connection.recv(65536))[0] == 200
    assert _kept(connection)
    connection.close()


def test_an_event_stream_pipelined_after_an_answer_is_kept(port):
    """``on_response_complete`` starts the pipelined request itself: no timer may be left armed over it."""
    connection = _connected(
        port,
        b'GET /api HTTP/1.1\r\nHost: SERVED\r\n\r\nGET /api/v1/events HTTP/1.1\r\nHost: SERVED\r\n\r\n',
    )

    assert _kept(connection)
    connection.close()


def test_an_upload_longer_than_the_header_deadline_within_its_own_is_written(port, live):
    connection = _connected(port, _upload_head('slow.bin'))
    # Paced by design: the body arrives over three header deadlines.
    for _ in range(10):
        time.sleep(3 * _HEADER_S / 10)
        connection.sendall(b'z' * (_UPLOAD_LENGTH // 10))

    assert _head(connection.recv(65536))[0] == 201
    connection.close()
    assert (live / 'slow.bin').stat().st_size == _UPLOAD_LENGTH


def test_a_body_sent_after_100_continue_within_its_deadline_is_written(port, live):
    connection = _connected(port, _upload_head('later.bin', extra='Expect: 100-continue\r\n'))
    assert _head(connection.recv(65536))[0] == 100
    # Waited by design: the body comes after the header deadline, within its own.
    time.sleep(2 * _HEADER_S)
    connection.sendall(b'z' * _UPLOAD_LENGTH)

    assert _head(connection.recv(65536))[0] == 201
    connection.close()
    assert (live / 'later.bin').stat().st_size == _UPLOAD_LENGTH
