# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import pathlib
import socket
import threading


class LvpLock:
    def __init__(self, lock_port: int = 0):
        """Create an instance lock.

        Args:
            lock_port: TCP port to bind for the lock. Pass 0 to let the OS
                       assign an ephemeral port (more secure -- avoids predictable
                       port that could be targeted for local DoS).
        """
        self._lock_port = lock_port
        self._lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Do NOT set SO_REUSEADDR: on Windows this has SO_REUSEPORT semantics
        # (allows live double-bind), which defeats the single-instance guard.
        # Two LVPs would both succeed here, then trample each other's serial
        # ports. See issue #559. Bind-only sockets release the port immediately
        # on process exit (no TIME_WAIT concern since we never listen()).

    def lock(self) -> bool:
        try:
            self._lock_socket.bind(('127.0.0.1', self._lock_port))
            return True
        except OSError:
            return False

    @property
    def port(self) -> int:
        """Return the actual bound port (useful when lock_port=0)."""
        try:
            return self._lock_socket.getsockname()[1]
        except Exception:
            return self._lock_port

    def close(self):
        try:
            self._lock_socket.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()


class _HeldPort:
    """A port's lock as this process holds it: the one bound socket, and the takes on it."""

    def __init__(self, socket_lock: LvpLock, key: int | None):
        self.socket_lock = socket_lock
        # The configured port it is recorded under; None for port 0, whose
        # every take binds an ephemeral port of its own and is shared by none.
        self.key = key
        self.takes = 0


# This process's held ports by configured port, and the one real scope's
# take. Two threads may build scopes at once, so both change under a lock.
_record_lock = threading.Lock()
_held: dict[int, _HeldPort] = {}
_scope_take: 'InstanceLock | None' = None


class InstanceLock:
    """One take of an installation's instance lock; ``close()`` gives it back.

    The port's socket is bound once per process, and each take is a handle
    on it, released when the last handle closes. A handle gives back its
    own take once: closing it again does nothing, so a teardown that runs
    twice cannot give back another holder's take. There is no finalizer:
    every take is given back by the ``close()`` of whoever took it.
    """

    def __init__(self, held: _HeldPort):
        self._held = held
        self._closed = False

    @property
    def port(self) -> int:
        """The bound port: the configured one, or the one the OS gave for port 0."""
        return self._held.socket_lock.port

    def close(self) -> None:
        global _scope_take
        with _record_lock:
            if self._closed:
                return
            self._closed = True
            if _scope_take is self:
                _scope_take = None
            self._held.takes -= 1
            if self._held.takes == 0:
                if self._held.key is not None:
                    del _held[self._held.key]
                self._held.socket_lock.close()


def _take(port: int) -> InstanceLock | None:
    """A handle on ``port``'s lock, bound now when this process does not hold it; None when another process does.

    Called under ``_record_lock``.
    """
    held = _held.get(port) if port != 0 else None
    if held is None:
        socket_lock = LvpLock(lock_port=port)
        if not socket_lock.lock():
            socket_lock.close()
            return None
        held = _HeldPort(socket_lock, port if port != 0 else None)
        if held.key is not None:
            _held[port] = held
    held.takes += 1
    return InstanceLock(held)


def take_instance_lock(source_path: str | pathlib.Path | None) -> InstanceLock | None:
    """A host's take of this installation's single-instance lock; None when another process holds it.

    The GUI and the REST server take it before they load anything that
    opens the scope, so a second launch exits before it opens a window or
    a port: two processes on one scope trample each other's serial ports.
    The scope the host then builds takes the lock too, as a second handle
    in the host's process (``take_scope_lock``). The returned lock is held
    until its ``close()``.
    """
    from modules.app_config import get_lvp_lock_port

    port = get_lvp_lock_port(source_path)
    with _record_lock:
        return _take(port)


def take_scope_lock(source_path: str | pathlib.Path | None) -> InstanceLock:
    """A real scope's take of its installation's instance lock.

    Every real scope takes it where its hardware opens, however it is built,
    so no scope drives the microscope beside the GUI, the REST server or a
    script, and a process drives one real scope. A host's own take in this
    process is no refusal: the scope's is a second handle on it.

    Raises:
        ScopeInUseError: another process holds the lock, or another real
            scope in this process holds a scope's take. Nothing was taken.
    """
    global _scope_take
    from modules.app_config import get_lvp_lock_port
    from modules.exceptions import ScopeInUseError

    port = get_lvp_lock_port(source_path)
    with _record_lock:
        taken = None if _scope_take is not None else _take(port)
        if taken is None:
            raise ScopeInUseError()
        _scope_take = taken
        return taken
