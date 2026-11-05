# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""REST served beside a host that keeps its own main loop -- the GUI (``_lumascope.rest.server.serve_beside``).

The port, the session's ``rest_api.port``, is bound on the host's thread,
so a port already taken is one ``RestServerNotStartedError`` raised where
the start is, and the host runs on without a server; otherwise the server
serves the host's own session, on ``127.0.0.1``, on a thread of its own,
and closes through the host's shutdown.
"""

from __future__ import annotations

import socket
import threading

import httpx
import pytest

from _lumascope.modules.exceptions import RestServerNotStartedError
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.server import serve_beside
from tests.settings_fixtures import complete_settings


HOST = '127.0.0.1'


def _free_port() -> int:
    with socket.create_server((HOST, 0)) as sock:
        return sock.getsockname()[1]


def _session_on(tmp_path, port: int) -> ScopeSession:
    live = tmp_path / 'live'
    live.mkdir()
    settings = complete_settings(live_folder=str(live))
    settings['rest_api']['port'] = port
    return ScopeSession.create(settings, simulate=True)


@pytest.fixture
def port() -> int:
    return _free_port()


@pytest.fixture
def session(tmp_path, port):
    s = _session_on(tmp_path, port)
    yield s
    s.shutdown()


def test_a_port_already_taken_is_raised_at_the_start_and_nothing_serves(tmp_path):
    shut = []
    with socket.create_server((HOST, 0)) as taken:
        port = taken.getsockname()[1]
        session = _session_on(tmp_path, port)
        threads_before = set(threading.enumerate())
        try:
            with pytest.raises(RestServerNotStartedError) as raised:
                serve_beside(session, shut_down=lambda: shut.append(1))
            threads_after = set(threading.enumerate())
        finally:
            session.shutdown()

    assert raised.value.reason == 'rest_server_not_started'
    assert f'{HOST}:{port}' in str(raised.value)
    assert isinstance(raised.value.__cause__, OSError)
    assert threads_after <= threads_before
    assert shut == []


def test_the_server_serves_the_hosts_session_and_closes_through_its_shutdown(session, port):
    shut = []

    def shut_down():
        shut.append(1)
        session.shutdown()

    server = serve_beside(session, shut_down=shut_down)
    try:
        # Bound before the call returned, so a client may connect at once;
        # it is answered once the loop is up.
        status = httpx.get(f'http://{HOST}:{port}/api/v1/status', timeout=10)
    finally:
        closed = server.close()

    assert status.status_code == 200
    assert (closed, shut) == (True, [1])
    assert session.live_work.closed


def test_the_servers_own_lines_reach_the_log_at_their_own_level(session, monkeypatch):
    """uvicorn's lines go to LumaViewPro's log as the records they are, not through stderr.

    The GUI captures stderr as warnings, so uvicorn's own stderr handler put
    "Application startup complete." in the errors log on every start.
    """
    import logging

    heard = []

    class _Catcher(logging.Handler):
        def emit(self, record):
            heard.append((record.name, record.levelname, record.getMessage()))

    # uvicorn's loggers as a process that never configured them has them: a
    # server another test built with uvicorn's own configuration left them
    # its handlers, no propagation and a WARNING floor.
    for name in ('uvicorn', 'uvicorn.error'):
        monkeypatch.setattr(logging.getLogger(name), 'handlers', [])
        monkeypatch.setattr(logging.getLogger(name), 'propagate', True)
        monkeypatch.setattr(logging.getLogger(name), 'level', logging.NOTSET)
    catcher = _Catcher(level=logging.DEBUG)
    root = logging.getLogger()
    # As lvp_logger sets the root logger, which the test process stands in for.
    level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(catcher)
    server = serve_beside(session, shut_down=session.shutdown)
    try:
        assert httpx.get(f'http://{HOST}:{server.config.port}/api/v1/status', timeout=10)
    finally:
        server.close()
        root.removeHandler(catcher)
        root.setLevel(level)

    assert ('uvicorn.error', 'INFO', 'Application startup complete.') in heard, heard
