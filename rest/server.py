# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The server's close, with clients connected.

uvicorn's own close re-raises the signal once serving stops, so the process
dies by it and nothing after the serve runs; and it waits for every open
connection, which an event stream never ends. ``Server`` closes in the
session's order instead. On SIGINT or SIGTERM:

1. the listening sockets close, so a new connection is refused, and a
   member asked on an open connection is refused ``server_closing``; jobs,
   handles, files and the stream are still served;
2. ``session.shutdown()`` runs on a thread while the stream stays open, so
   it carries the run's end and the outcomes the close reports;
3. when it returns, ``closing`` goes to every stream and each one ends;
4. uvicorn stops, finding no long-lived connection.

The launcher then joins the calls' threads. A later signal while the close
runs is logged with what it is waiting on, and changes nothing: the
session's close has no time limit, so forcing an exit could leave a file
half-written.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import socket
from types import FrameType

import uvicorn

from modules.scope_session import ScopeSession
from rest.app import Closing

_log = logging.getLogger('lvp_logger.rest')


class Server(uvicorn.Server):
    """A uvicorn server whose SIGINT and SIGTERM close the session before it stops."""

    def __init__(self, config: uvicorn.Config, session: ScopeSession, closing: Closing) -> None:
        super().__init__(config)
        self._session = session
        self._closing = closing
        self._loop: asyncio.AbstractEventLoop | None = None
        self._close_task: asyncio.Task | None = None
        self.close_begun = False
        # What ``session.shutdown()`` raised, if it did.
        self.close_failed: BaseException | None = None

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets)
        self._loop = asyncio.get_running_loop()

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        """Begin the close; a later signal is logged. The signal is never re-raised."""
        if self._loop is None:
            # Nothing is served yet: there is nothing to finish.
            self.should_exit = True
            return
        if self.close_begun:
            _log.warning(
                f'[REST     ] {signal.Signals(sig).name} while closing: the close is still '
                f'waiting on {self._session.live_work}'
            )
            return
        self.close_begun = True
        _log.info(f'[REST     ] {signal.Signals(sig).name}: closing')
        self._loop.call_soon_threadsafe(self._begin_close)

    def _begin_close(self) -> None:
        # Held: the loop keeps only a weak reference to a task.
        self._close_task = asyncio.get_running_loop().create_task(self._close())

    async def _close(self) -> None:
        for server in self.servers:
            server.close()
        self._closing.begin()
        try:
            await asyncio.to_thread(self._session.shutdown)
        except Exception as e:
            self.close_failed = e
            _log.error(f'[REST     ] The session did not close cleanly: {e!r}')
        self._closing.finish()
        self.should_exit = True
