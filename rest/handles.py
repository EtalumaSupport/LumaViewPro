# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The live objects a client has been handed, each by one id.

A live object -- a protocol, a run, a move in flight, the session's
protocol runner -- crosses the wire as ``{"handle": <id>, "type": <class
name>}``; its members are at ``/api/v1/handles/<type>/<id>/<member>``, and
a client passes it as an argument by its id. One object has one id, so
two clients handed the same run hold the same id. An id stays until a
client forgets it or the session closes; forgetting an id lets go of the
id, never of the object: a run goes on, and its stop is its own member.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import fastapi

# The most handles held at once. A call whose answer can carry a new
# handle is refused while this many are held, before the member runs, so
# no answer loses the handle it carries.
LIMIT = 1000
# Seconds a client refused for the limit waits before asking again, by
# which time it, or a peer, may have forgotten handles it no longer needs.
RETRY_AFTER_S = 5


class HandleRegistry:
    """The ids of the live objects clients have been handed.

    Holds each object, so its id is never reused while the id is held.
    Thread-safe: a call's answer is encoded on the call's own thread.
    """

    def __init__(self, *, kept: Callable[[object], bool]) -> None:
        """Make an empty registry.

        Args:
            kept: Whether an object's id is never forgotten -- one every
                client shares and finds by asking again, such as the
                session's one protocol runner.
        """
        self._kept = kept
        self._lock = threading.Lock()
        self._objects: dict[str, object] = {}
        self._ids: dict[int, str] = {}
        self._next = 1

    def admit(self) -> None:
        """Refuse a call whose answer can carry a new handle while the limit is held.

        Raises:
            fastapi.HTTPException: 503, with ``Retry-After``.
        """
        with self._lock:
            full = len(self._objects) >= LIMIT
        if full:
            raise fastapi.HTTPException(
                503,
                f'{LIMIT} handles are held: forget the ones no longer needed '
                '(DELETE /api/v1/handles/<type>/<id>) and ask again.',
                headers={'Retry-After': str(RETRY_AFTER_S)},
            )

    def mint(self, obj: object) -> dict[str, str]:
        """*obj*'s wire form, giving it an id when it has none."""
        with self._lock:
            handle = self._ids.get(id(obj))
            if handle is None:
                handle = str(self._next)
                self._next += 1
                self._objects[handle] = obj
                self._ids[id(obj)] = handle
        return {'handle': handle, 'type': type(obj).__name__}

    def get(self, handle: str, cls: type) -> object:
        """The live object of class *cls* with id *handle*.

        Raises:
            fastapi.HTTPException: 404, no such id is held for that class.
        """
        with self._lock:
            obj = self._objects.get(handle)
        if obj is None or not isinstance(obj, cls):
            raise fastapi.HTTPException(404, f'No {cls.__name__} handle {handle} is held.')
        return obj

    def forget(self, handle: str, cls: type) -> None:
        """Let go of the id *handle*, never of its object.

        Raises:
            fastapi.HTTPException: 404, no such id is held for that class;
                409, the id is one every client shares.
        """
        obj = self.get(handle, cls)
        if self._kept(obj):
            raise fastapi.HTTPException(
                409, f'{cls.__name__} handle {handle} is shared by every client and is kept.'
            )
        with self._lock:
            if self._objects.pop(handle, None) is not None:
                del self._ids[id(obj)]

    def listing(self) -> list[dict[str, str]]:
        """Every held handle, oldest first."""
        with self._lock:
            held = sorted(self._objects.items(), key=lambda item: int(item[0]))
        return [{'handle': h, 'type': type(obj).__name__} for h, obj in held]
