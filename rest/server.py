# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The server's close, with clients connected.

uvicorn's own close re-raises the signal once serving stops, so the process
dies by it and nothing after the serve runs; and it waits for every open
connection, which an event stream never ends. ``Server`` closes in the
session's order instead, on SIGINT or SIGTERM or its host's ``close()``:

1. the listening sockets close, so a new connection is refused, and a
   member asked on an open connection is refused ``server_closing``; jobs,
   handles, files and the stream are still served;
2. the host's shutdown -- ``session.shutdown`` headless, the GUI's own
   in the GUI -- runs on a thread while the stream stays open, so it
   carries the run's end and the outcomes the close reports;
3. when it returns, ``closing`` goes to every stream and each one ends,
   the live views end, and every client still waiting is answered;
4. the calls' threads are joined, for up to ``JOIN_S``;
5. uvicorn stops, finding no long-lived connection.

The close runs once. A later signal while it runs is logged with what it
is waiting on, and changes nothing: the session's close has no time limit,
so forcing an exit could leave a file half-written. A server that is not
serving closes nothing: its host shuts its session down itself.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import signal
import socket
import threading
from collections.abc import Callable
from types import FrameType

import uvicorn

from modules.exceptions import RestServerNotStartedError
from modules.scope_session import ScopeSession
from rest.app import Closing, build_app

_log = logging.getLogger('lvp_logger.rest')

# The one address every host serves on: there are no credentials, so only
# this computer may reach the scope.
HOST = '127.0.0.1'

# Seconds the close waits for the calls' threads once the session has shut
# down; a call still running then is logged by name and the server stops.
JOIN_S = 10.0


class _Serving(enum.Enum):
    NOT_STARTED = 'not started'
    # Between run() and the loop's startup: a close here stops it unserved.
    STARTING = 'starting'
    SERVING = 'serving'
    # A close is running, through the host's shutdown.
    CLOSING = 'closing'
    # Told to stop before it served, or run() has returned.
    ENDED = 'ended'


class Server(uvicorn.Server):
    """A uvicorn server that closes its host's session before it stops.

    *shut_down* is the host's close of the session, run inside the server's
    close; the server never closes the session any other way.
    """

    def __init__(
        self,
        config: uvicorn.Config,
        session: ScopeSession,
        closing: Closing,
        *,
        shut_down: Callable[[], None],
    ) -> None:
        super().__init__(config)
        self._session = session
        self._closing = closing
        self._shut_down = shut_down
        self._lock = threading.Lock()
        self._state = _Serving.NOT_STARTED
        self._loop: asyncio.AbstractEventLoop | None = None
        self._close_task: asyncio.Task | None = None
        # Set when run() returns.
        self._ended = threading.Event()
        self.close_begun = False
        # What the host's shutdown raised, if it did.
        self.close_failed: BaseException | None = None

    @property
    def serving(self) -> bool:
        """Whether its loop is up and no close has begun."""
        return self._state is _Serving.SERVING

    async def serve(self, sockets: list[socket.socket] | None = None) -> None:
        with self._lock:
            if self._state is not _Serving.NOT_STARTED:
                # Closed before it ran: it serves nothing.
                self._ended.set()
                return
            self._state = _Serving.STARTING
        try:
            await super().serve(sockets)
        finally:
            # Ended as run() returns, so a later close hands off to no loop
            # and the host shuts its session down itself.
            with self._lock:
                self._state = _Serving.ENDED
                self._loop = None
            self._ended.set()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets)
        with self._lock:
            if self._state is _Serving.STARTING:
                self._loop = asyncio.get_running_loop()
                self._state = _Serving.SERVING

    def refuse_calls(self) -> None:
        """Refuse every member asked from now on (``server_closing``); the host's close follows.

        A host that acts before its close begins -- the GUI takes back
        control first -- calls this before it does, so no client's call
        lands in between.
        """
        self._closing.begin()

    def close(self) -> bool:
        """Close the server and, through the host's shutdown, the session; return once it has stopped.

        Called from any thread but the server's own. True to the call that
        closed the session this way; what the host's shutdown raised is
        raised here. False when this call closed nothing, and the host shuts
        its session down itself: the server is not serving (not started,
        its start failed, or it has stopped), it was still starting -- it is
        then told to stop, serves nothing, and is waited for -- or another
        close was running, which this call waited for.
        """
        with self._lock:
            state = self._state
            if state is _Serving.SERVING:
                self._begin_close_locked()
            elif state is _Serving.STARTING:
                self.should_exit = True
                self._state = _Serving.ENDED
            elif state is _Serving.NOT_STARTED:
                self._state = _Serving.ENDED
        if state is _Serving.SERVING:
            self._ended.wait()
            if self.close_failed is not None:
                raise self.close_failed
            return True
        if state in (_Serving.STARTING, _Serving.CLOSING):
            # Waited for, so nothing serves while the host shuts down.
            self._ended.wait()
        return False

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        """Begin the close; a later signal is logged. The signal is never re-raised."""
        with self._lock:
            state = self._state
            if state is _Serving.SERVING:
                self._begin_close_locked()
            elif state in (_Serving.NOT_STARTED, _Serving.STARTING):
                # Nothing is served yet: there is nothing to finish.
                self.should_exit = True
                self._state = _Serving.ENDED
        if state is _Serving.SERVING:
            _log.info(f'[REST     ] {signal.Signals(sig).name}: closing')
        elif state is _Serving.CLOSING:
            _log.warning(
                f'[REST     ] {signal.Signals(sig).name} while closing: the close is still '
                f'waiting on {self._session.live_work}'
            )

    def _begin_close_locked(self) -> None:
        self._state = _Serving.CLOSING
        self.close_begun = True
        self._loop.call_soon_threadsafe(self._begin_close)

    def _begin_close(self) -> None:
        # Held: the loop keeps only a weak reference to a task.
        self._close_task = asyncio.get_running_loop().create_task(self._close())

    async def _close(self) -> None:
        for server in self.servers:
            server.close()
        self._closing.begin()
        try:
            await asyncio.to_thread(self._shut_down)
        except Exception as e:
            # Reported where it is read: raised to the host's close(), or
            # by the headless launcher.
            self.close_failed = e
        self._closing.finish()
        running = await asyncio.to_thread(self._closing.join_jobs, JOIN_S)
        if running:
            _log.error(f'[REST     ] Closing with calls still running: {running}')
        self.should_exit = True


def serve_beside(session: ScopeSession, *, port: int, shut_down: Callable[[], None]) -> Server:
    """Serve *session*'s API on ``HOST``:*port* on a thread of its own, beside a host's own main loop.

    For a host that keeps the main thread -- the GUI. The port is bound
    here, on the caller's thread, so a port already taken is raised here and
    nothing serves; uvicorn binding it on the server's thread would end that
    thread instead, with the host never told. The server closes through
    *shut_down*, the host's close of the session, when the host calls
    ``close()``.

    Raises:
        RestServerNotStartedError: The port could not be taken.
    """
    try:
        sock = socket.create_server((HOST, port))
    except OSError as e:
        raise RestServerNotStartedError(HOST, port, e) from e
    app = build_app(session)
    server = Server(
        # No logging configuration of uvicorn's own: its records reach the
        # host's log at their own level. Its stderr handler would put them in
        # a GUI's stderr capture, which records every line as a warning.
        uvicorn.Config(app, host=HOST, port=port, log_config=None),
        session,
        app.state.closing,
        shut_down=shut_down,
    )
    threading.Thread(
        target=server.run, kwargs={'sockets': [sock]}, name='rest-server', daemon=True
    ).start()
    return server
