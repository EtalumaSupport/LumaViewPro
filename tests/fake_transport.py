# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""In-memory fake of `mpremote.transport.Transport` for unit tests.

Replaces pyserial-level `MagicMock` patches (previously ~48 uses in
`test_firmware_updater.py`) with a subclass that implements the
Transport API against a `{filename: bytes}` dict. Tests drive the
adapter (`drivers/mpremote_transport.py`) through this fake exactly
as they would drive a real `SerialTransport`, which means the test
surface matches the contract the production code relies on.

Scope:
  - All `fs_*` methods used by the LVP adapter
    (`fs_writefile`, `fs_readfile`, `fs_hashfile`, `fs_listdir`,
    `fs_exists`, `fs_stat`, `fs_rmfile`, `fs_rmdir`, `fs_mkdir`,
    `fs_touchfile`, `fs_isdir`).
  - `exec()` / `eval()` with a small command vocabulary
    (`os.rename`, `os.remove`, literal evaluation) — enough for the
    adapter's atomic-`.tmp` rename path. Unknown commands raise
    `TransportExecError` by default so tests that wander off the
    vetted vocabulary fail loudly.
  - `enter_raw_repl()` / `exit_raw_repl()` / `close()` are no-ops
    that track connection state.
  - Fault injection: configurable exceptions on any method (one-shot
    or permanent), write-failure count with auto-recovery, hash
    override for SHA-256 mismatch tests.

Out of scope (intentionally): simulating MicroPython runtime
semantics. Tests that need full device behavior are bench tests, not
unit tests.
"""

import hashlib
import os
import re
import stat as stat_mod
from typing import Callable, Dict, List, Optional, Union

from mpremote.transport import (
    Transport,
    TransportError,
    TransportExecError,
    listdir_result,
)


class FakeTransport(Transport):
    """In-memory Transport for unit tests.

    Args:
        initial_files: dict of {filename: bytes} to seed the fake
            filesystem. Defaults to empty.

    Attributes:
        files: mutable `{str: bytes}` dict representing the device
            filesystem. Tests inspect + mutate this directly.
        call_log: list of `(method_name, args)` tuples, appended on
            every method call. For test assertions about call order.
        exec_history: list of exec() command strings, for assertions
            about what the adapter sent.
        connected: True until `close()` is called.
        in_raw_repl: True while inside enter_raw_repl/exit_raw_repl
            pair. Adapter asserts this; tests can inspect it.

    Fault injection:
        fail_next_write: if > 0, decrement-and-raise on next
            fs_writefile call (TransportError). Enables retry tests.
        fs_hashfile_override: if bytes, returned instead of real hash
            on fs_hashfile. Enables SHA-256 mismatch tests.
        raise_on: `{method_name: exception}` — raised once per named
            method then cleared. For targeted failure injection.
    """

    def __init__(
        self, initial_files: Optional[Dict[str, bytes]] = None
    ) -> None:
        self.files: Dict[str, bytes] = dict(initial_files or {})
        self.call_log: List[tuple] = []
        self.exec_history: List[str] = []
        self.connected: bool = True
        self.in_raw_repl: bool = False

        self.fail_next_write: int = 0
        self.fs_hashfile_override: Optional[bytes] = None
        self.raise_on: Dict[str, Exception] = {}

    # ------------------------------------------------------------------
    # Fault-injection helper
    # ------------------------------------------------------------------
    def _maybe_raise(self, method: str) -> None:
        exc = self.raise_on.pop(method, None)
        if exc is not None:
            raise exc

    def _require_connected(self, method: str) -> None:
        if not self.connected:
            raise TransportError(
                f"FakeTransport.{method}: transport is closed"
            )

    # ------------------------------------------------------------------
    # SerialTransport-specific methods (no-ops for the fake)
    # ------------------------------------------------------------------
    def enter_raw_repl(
        self, soft_reset: bool = True, timeout_overall: int = 10
    ) -> None:
        self.call_log.append(("enter_raw_repl", soft_reset))
        self._require_connected("enter_raw_repl")
        self._maybe_raise("enter_raw_repl")
        self.in_raw_repl = True

    def exit_raw_repl(self) -> None:
        self.call_log.append(("exit_raw_repl",))
        self._maybe_raise("exit_raw_repl")
        self.in_raw_repl = False

    def close(self) -> None:
        self.call_log.append(("close",))
        self.connected = False
        self.in_raw_repl = False

    # ------------------------------------------------------------------
    # exec() / eval() with a whitelist command vocabulary
    # ------------------------------------------------------------------
    #
    # The adapter uses exec() for atomic-rename bookkeeping around
    # fs_writefile. Pattern is roughly:
    #     exec("import os")
    #     exec("os.remove('foo.bak')")           # ignore OSError
    #     exec("os.rename('foo', 'foo.bak')")    # ignore OSError
    #     exec("os.rename('foo.tmp', 'foo')")
    # so this fake recognises exactly those shapes. Anything else
    # raises TransportExecError, forcing test authors to either extend
    # the vocabulary here (preferred) or avoid that exec path entirely.

    _RE_OS_RENAME = re.compile(
        r"^\s*os\.rename\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)\s*$"
    )
    _RE_OS_REMOVE = re.compile(r"^\s*os\.remove\(\s*'([^']+)'\s*\)\s*$")
    _RE_IMPORT_OS = re.compile(r"^\s*import os\s*$")
    _RE_NOOP = re.compile(r"^\s*(pass|#.*)?\s*$")

    def exec(
        self, command: str, data_consumer: Optional[Callable] = None
    ) -> bytes:
        self.call_log.append(("exec", command))
        self.exec_history.append(command)
        self._require_connected("exec")
        self._maybe_raise("exec")

        for line in command.split("\n"):
            if self._RE_IMPORT_OS.match(line) or self._RE_NOOP.match(line):
                continue
            m = self._RE_OS_RENAME.match(line)
            if m:
                src, dst = m.group(1), m.group(2)
                if src not in self.files:
                    raise TransportExecError(
                        1, f"OSError: [Errno 2] ENOENT: {src}"
                    )
                self.files[dst] = self.files.pop(src)
                continue
            m = self._RE_OS_REMOVE.match(line)
            if m:
                name = m.group(1)
                if name not in self.files:
                    raise TransportExecError(
                        1, f"OSError: [Errno 2] ENOENT: {name}"
                    )
                del self.files[name]
                continue
            raise TransportExecError(
                1,
                f"FakeTransport.exec: unsupported command {line!r}. "
                f"Extend FakeTransport if the adapter needs this pattern.",
            )
        return b""

    def eval(self, expression: str, parse: bool = True):
        self.call_log.append(("eval", expression))
        self._require_connected("eval")
        self._maybe_raise("eval")
        raise TransportExecError(
            1,
            f"FakeTransport.eval: {expression!r} not implemented. "
            f"The adapter should not need eval for Phase 1 operations.",
        )

    # ------------------------------------------------------------------
    # fs_* overrides — operate on self.files dict directly
    # ------------------------------------------------------------------
    def fs_listdir(self, src: str = "") -> List[listdir_result]:
        self.call_log.append(("fs_listdir", src))
        self._require_connected("fs_listdir")
        self._maybe_raise("fs_listdir")
        results = []
        for name, data in self.files.items():
            results.append(listdir_result(name, stat_mod.S_IFREG, 0, len(data)))
        return results

    def fs_stat(self, src: str) -> os.stat_result:
        self.call_log.append(("fs_stat", src))
        self._require_connected("fs_stat")
        self._maybe_raise("fs_stat")
        if src not in self.files:
            raise OSError(2, f"ENOENT: {src}")
        # stat_result tuple: (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime)
        return os.stat_result(
            (stat_mod.S_IFREG | 0o644, 0, 0, 1, 0, 0, len(self.files[src]), 0, 0, 0)
        )

    def fs_exists(self, src: str) -> bool:
        self.call_log.append(("fs_exists", src))
        self._require_connected("fs_exists")
        return src in self.files

    def fs_isdir(self, src: str) -> bool:
        self.call_log.append(("fs_isdir", src))
        return False  # fake has no directories

    def fs_readfile(
        self,
        src: str,
        chunk_size: int = 256,
        progress_callback: Optional[Callable] = None,
    ) -> bytearray:
        self.call_log.append(("fs_readfile", src))
        self._require_connected("fs_readfile")
        self._maybe_raise("fs_readfile")
        if src not in self.files:
            raise OSError(2, f"ENOENT: {src}")
        data = self.files[src]
        if progress_callback is not None:
            # Fire once with completed=total to mirror progress reporting.
            progress_callback(len(data), len(data))
        return bytearray(data)

    def fs_writefile(
        self,
        dest: str,
        data: Union[bytes, bytearray],
        chunk_size: int = 256,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        self.call_log.append(("fs_writefile", dest, len(data)))
        self._require_connected("fs_writefile")
        if self.fail_next_write > 0:
            self.fail_next_write -= 1
            raise TransportError(
                f"FakeTransport.fs_writefile: injected failure (remaining={self.fail_next_write})"
            )
        self._maybe_raise("fs_writefile")
        self.files[dest] = bytes(data)
        if progress_callback is not None:
            progress_callback(len(data), len(data))

    def fs_hashfile(
        self, path: str, algo: str = "sha256", chunk_size: int = 256
    ) -> bytes:
        self.call_log.append(("fs_hashfile", path, algo))
        self._require_connected("fs_hashfile")
        self._maybe_raise("fs_hashfile")
        if self.fs_hashfile_override is not None:
            return self.fs_hashfile_override
        if path not in self.files:
            raise OSError(2, f"ENOENT: {path}")
        return hashlib.new(algo, self.files[path]).digest()

    def fs_rmfile(self, path: str) -> None:
        self.call_log.append(("fs_rmfile", path))
        self._require_connected("fs_rmfile")
        self._maybe_raise("fs_rmfile")
        if path not in self.files:
            raise OSError(2, f"ENOENT: {path}")
        del self.files[path]

    def fs_rmdir(self, path: str) -> None:
        self.call_log.append(("fs_rmdir", path))
        raise OSError(20, f"ENOTDIR: {path}")  # no dirs in fake

    def fs_mkdir(self, path: str) -> None:
        self.call_log.append(("fs_mkdir", path))
        raise OSError(17, f"EEXIST: {path}")  # stub

    def fs_touchfile(self, path: str) -> None:
        self.call_log.append(("fs_touchfile", path))
        self._require_connected("fs_touchfile")
        self.files.setdefault(path, b"")
