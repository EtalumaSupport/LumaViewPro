# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Who is calling the API on this thread: the GUI, a REST client, or code in the process.

The host stamps it where its calls enter -- the GUI on its own threads, the
REST server on each call's thread -- and nothing else writes it. A caller
never passes it, so it cannot claim to be another. A thread nobody stamped
is an in-process caller: a script, the SDK, a plugin.

It is read on the calling thread, where a run's provenance is fixed (the
run's ``prepare``) and where a stop runs, so it need not cross a lane.
"""

from __future__ import annotations

import contextlib
import enum
import threading
from collections.abc import Iterator

_stamp = threading.local()


class Caller(enum.StrEnum):
    """Which host a call came through."""

    GUI = 'gui'
    REST = 'rest'
    API = 'api'

    @property
    def words(self) -> str:
        """The caller as a sentence names it: 'LumaViewPro', 'a REST client', 'a script'."""
        return _WORDS[self]


_WORDS = {Caller.GUI: 'LumaViewPro', Caller.REST: 'a REST client', Caller.API: 'a script'}


def current_caller() -> Caller:
    """The caller this thread acts for; ``Caller.API`` where no host stamped it."""
    return getattr(_stamp, 'caller', Caller.API)


@contextlib.contextmanager
def calling(caller: Caller) -> Iterator[None]:
    """Act for *caller* on this thread for the length of a ``with`` block; nests."""
    previous = current_caller()
    _stamp.caller = caller
    try:
        yield
    finally:
        _stamp.caller = previous
