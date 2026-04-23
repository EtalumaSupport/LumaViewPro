# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Raw-REPL file transfer via the mpremote library.

Replaces the hand-rolled streaming / on-device `sys.stdin.buffer.read(N)`
helper in `drivers/raw_repl.py`. The old path had a structural flaw: a
single blocking read on the device with no per-chunk flow control,
which stalls and strands the board on MicroPython 1.19 USB CDC under
large-file pressure (bench-confirmed 2026-04-21 on a 75 KB push).

mpremote's raw-paste mode has device-advertised window-token flow
control, so the host cannot overrun the device. This adapter keeps
LVP's higher-level safety (atomic `.tmp` + SHA-256 verify + rename
with backup) while pushing the byte-streaming primitive down to
mpremote's `Transport.fs_writefile` / `fs_hashfile`.

Library-only rule (plan §1, per `docs/MPREMOTE_MIGRATION_PLAN.md` in
the Firmware repo): mpremote is used exclusively as a Python library.
Never shell out to the `mpremote` CLI from anywhere. If a capability
is missing, add it here.

Public surface mirrors `drivers/raw_repl.py` so Phase 2's
`SerialBoard` re-point is mechanical:

    with MpremoteSession(transport) as session:
        session.list_files()
        session.read_file('motorconfig.json')
        session.write_file('main.py', data)

`transport` is either a real `mpremote.transport_serial.SerialTransport`
(use `create_session(device_path)` for the common case) or a
`FakeTransport` from `tests/fake_transport.py` (for unit tests).

Pyserial ownership: `SerialTransport` calls `serial.serial_for_url(...)`
internally with `exclusive=True` by default, so the caller must have
closed its own pyserial handle on the same device before constructing
a session. `SerialBoard` already does this in the
`modules/lumascope_api.py` driver-swap pattern used during firmware
updates; Phase 2 wires that into `SerialBoard.enter_raw_repl` itself.

`verify_firmware_running` is intentionally NOT part of this adapter.
It operates on a raw pyserial port AFTER the raw-REPL window closes,
which is outside mpremote's Transport abstraction. It stays in
`drivers/raw_repl.py` (or gets folded into `SerialBoard` in Phase 2).
"""

import contextlib
import hashlib
import io
import logging
import pathlib
from typing import Optional

from mpremote.transport import Transport, TransportError, TransportExecError
from mpremote.transport_serial import SerialTransport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — conservative for field reliability
# ---------------------------------------------------------------------------

DEFAULT_BAUDRATE = 115200
WRITE_VERIFY_RETRIES = 3       # Attempts to write + verify a file
READ_VERIFY_RETRIES = 2        # Read file twice and compare (corruption check)


# ---------------------------------------------------------------------------
# Stdout capture shim (plan §4 R3)
# ---------------------------------------------------------------------------
#
# mpremote's transport module prints diagnostics to stdout via `print()`
# and `sys.stdout.write()` in several places (analysis §2 Q3 cites
# transport_serial.py:96, 99, 103, 106, 178, 188, 193, 278, 832 and
# transport.py:35, 39). In a PyInstaller windowed build LVP has no
# stdout, and in logged runs these messages would bypass the logging
# pipeline. Capture and drain to the logger instead.

@contextlib.contextmanager
def _capture_stdout_to_logger():
    """Redirect stdout during an mpremote call; drain lines to logger."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield
    text = buf.getvalue()
    if not text:
        return
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # mpremote uses "warning" / "error" / "Error" in its diagnostic
        # prints; escalate to warning in the logger when we see them.
        low = stripped.lower()
        if "error" in low or "warning" in low or "fail" in low:
            logger.warning(f"mpremote: {stripped}")
        else:
            logger.info(f"mpremote: {stripped}")


# ---------------------------------------------------------------------------
# MpremoteSession — raw-REPL window around a Transport
# ---------------------------------------------------------------------------

class MpremoteSession:
    """Holds a Transport across enter_raw_repl / exit_raw_repl.

    Accepts an already-constructed Transport (dependency injection).
    For a real serial port, use :func:`create_session` to build the
    SerialTransport and wrap it.

    Use as a context manager; `__enter__` calls `enter_raw_repl` and
    `__exit__` calls `exit_raw_repl`. The Transport itself is NOT
    closed on exit — the caller owns that lifecycle. `create_session`
    returns a ManagedSession that does close on exit.
    """

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self._in_raw_repl = False

    # --- lifecycle ---------------------------------------------------
    def enter(self, soft_reset: bool = True) -> None:
        """Enter raw REPL. Raises TransportError on failure."""
        with _capture_stdout_to_logger():
            self.transport.enter_raw_repl(soft_reset=soft_reset)
        self._in_raw_repl = True
        logger.info("mpremote raw REPL entered (soft_reset=%s)", soft_reset)

    def exit(self) -> None:
        """Exit raw REPL. Swallows errors during exit (best-effort)."""
        if not self._in_raw_repl:
            return
        try:
            with _capture_stdout_to_logger():
                self.transport.exit_raw_repl()
        except TransportError as e:
            logger.warning("exit_raw_repl: %s", e)
        finally:
            self._in_raw_repl = False

    def __enter__(self) -> "MpremoteSession":
        self.enter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exit()

    def _require_raw_repl(self, op: str) -> None:
        if not self._in_raw_repl:
            raise RuntimeError(
                f"MpremoteSession.{op}: not in raw REPL "
                f"(call enter() or use as a context manager)"
            )

    # --- file operations ---------------------------------------------
    def list_files(self) -> list:
        """List files on the board filesystem.

        Returns a list of filenames (strings). Empty list if the device
        returns no entries.
        """
        self._require_raw_repl("list_files")
        with _capture_stdout_to_logger():
            entries = self.transport.fs_listdir()
        return [e.name for e in entries]

    def read_file(self, filename: str, verify: bool = True) -> Optional[bytes]:
        """Read a file from the board.

        If ``verify`` is True (default), reads the file twice and
        compares — matches raw_repl.py's corruption-detection policy.

        Returns bytes, or None if all reads fail.
        """
        self._require_raw_repl("read_file")

        def _do_read() -> Optional[bytes]:
            try:
                with _capture_stdout_to_logger():
                    data = self.transport.fs_readfile(filename)
                return bytes(data)
            except (TransportError, TransportExecError, OSError) as e:
                logger.warning("read_file %s: %s", filename, e)
                return None

        data = _do_read()
        if data is None:
            return None
        if not verify:
            return data

        data2 = _do_read()
        if data2 is None:
            logger.warning(
                "read_file verify for %s: second read failed; "
                "using first read (%d bytes)", filename, len(data)
            )
            return data
        if data == data2:
            logger.debug("read_file verify: %s (%d bytes) OK", filename, len(data))
            return data
        # Disagreement — tiebreak with a third read.
        logger.error(
            "read_file verify FAILED for %s: "
            "read1=%d bytes, read2=%d bytes. Possible serial corruption.",
            filename, len(data), len(data2)
        )
        data3 = _do_read()
        if data3 == data:
            logger.info("read_file tiebreak: read3 matches read1")
            return data
        if data3 == data2:
            logger.info("read_file tiebreak: read3 matches read2")
            return data2
        logger.error(
            "read_file: all three reads differ for %s. "
            "Returning longest.", filename
        )
        return max([d for d in (data, data2, data3) if d], key=len)

    def write_file(self, filename: str, data: bytes) -> bool:
        """Write a file with SHA-256 verify, atomic rename, and backup.

        Safety layers (same policy as ``raw_repl.write_file``):
          1. Write to ``<name>.tmp`` first (via ``fs_writefile``).
          2. Compute SHA-256 on device (``fs_hashfile``), compare to
             local hash.
          3. Rename existing ``<name>`` to ``<name>.bak`` (ignoring
             ENOENT on first deploy).
          4. Rename ``<name>.tmp`` to ``<name>``.
          5. Retry up to ``WRITE_VERIFY_RETRIES`` times with full
             cleanup (remove stale .tmp on retry).

        Returns True on success, False on failure.
        """
        self._require_raw_repl("write_file")

        safe_name = pathlib.Path(filename).name
        if not safe_name:
            logger.error("write_file: invalid filename %r", filename)
            return False
        tmp_name = safe_name + ".tmp"
        bak_name = safe_name + ".bak"

        expected_hash = hashlib.sha256(data).digest()
        file_size = len(data)
        logger.info(
            "write_file %s (%d bytes, SHA256=%s)",
            safe_name, file_size, expected_hash.hex()[:16]
        )

        for attempt in range(1, WRITE_VERIFY_RETRIES + 1):
            logger.info("write_file attempt %d/%d", attempt, WRITE_VERIFY_RETRIES)
            try:
                # Step 1: stream bytes to .tmp (raw-paste, window-controlled).
                with _capture_stdout_to_logger():
                    self.transport.fs_writefile(tmp_name, data)

                # Step 2: device-side SHA-256.
                with _capture_stdout_to_logger():
                    device_hash = self.transport.fs_hashfile(tmp_name, "sha256")
                if device_hash != expected_hash:
                    logger.error(
                        "write_file SHA256 mismatch on %s: "
                        "expected %s, got %s",
                        safe_name,
                        expected_hash.hex()[:16],
                        device_hash.hex()[:16],
                    )
                    # Clean up corrupt .tmp before retry.
                    self._safe_remove(tmp_name)
                    continue

                # Step 3-4: atomic rename + backup. Each step tolerates
                # ENOENT (first deploy has no prior file, no prior .bak).
                self._safe_remove(bak_name)
                self._safe_rename(safe_name, bak_name)
                self._rename(tmp_name, safe_name)

                logger.info(
                    "write_file OK: %s (%d bytes, verified, backup=%s)",
                    safe_name, file_size, bak_name
                )
                return True

            except (TransportError, TransportExecError) as e:
                logger.warning("write_file attempt %d error: %s", attempt, e)
                self._safe_remove(tmp_name)

        logger.error(
            "write_file FAILED: %s after %d attempts",
            safe_name, WRITE_VERIFY_RETRIES
        )
        return False

    def raw_exec(self, code: str, timeout: int = 10):
        """Execute Python code on the device. Returns (stdout, stderr).

        Signature matches ``raw_repl.raw_exec``: returns a
        ``(stdout_bytes, stderr_bytes)`` tuple, or ``None`` on error.

        Under the Transport abstraction, stderr is delivered as a
        ``TransportExecError.error_output`` attribute. We normalize
        back to the tuple shape so Phase 2's SerialBoard swap needs
        no callsite changes.
        """
        self._require_raw_repl("raw_exec")
        try:
            with _capture_stdout_to_logger():
                stdout = self.transport.exec(code)
            return (stdout if isinstance(stdout, (bytes, bytearray)) else b"", b"")
        except TransportExecError as e:
            # Preserve device-side traceback per analysis §2 R7.
            stderr = e.error_output or b""
            if isinstance(stderr, str):
                stderr = stderr.encode("utf-8", errors="replace")
            return (b"", stderr)
        except TransportError as e:
            logger.warning("raw_exec error: %s", e)
            return None

    # --- internal helpers --------------------------------------------
    def _safe_remove(self, path: str) -> None:
        """Remove path on device, ignoring ENOENT."""
        try:
            with _capture_stdout_to_logger():
                self.transport.exec(f"import os\nos.remove('{path}')")
        except TransportExecError:
            pass  # ENOENT is fine
        except TransportError as e:
            logger.debug("_safe_remove(%s): %s", path, e)

    def _safe_rename(self, src: str, dst: str) -> None:
        """Rename src → dst, ignoring ENOENT (for optional .bak step)."""
        try:
            with _capture_stdout_to_logger():
                self.transport.exec(f"import os\nos.rename('{src}', '{dst}')")
        except TransportExecError:
            pass  # ENOENT is fine (no pre-existing file)
        except TransportError as e:
            logger.debug("_safe_rename(%s→%s): %s", src, dst, e)

    def _rename(self, src: str, dst: str) -> None:
        """Rename src → dst. Raises TransportExecError on failure."""
        with _capture_stdout_to_logger():
            self.transport.exec(f"import os\nos.rename('{src}', '{dst}')")


# ---------------------------------------------------------------------------
# Factory for real SerialTransport-backed sessions
# ---------------------------------------------------------------------------

class _ManagedSession(MpremoteSession):
    """MpremoteSession that owns its transport (closes on exit).

    Returned by :func:`create_session` for the common case where the
    caller wants a single end-to-end call. Tests that inject a
    FakeTransport use the base :class:`MpremoteSession` so they can
    inspect the transport after exit.
    """

    def exit(self) -> None:
        super().exit()
        try:
            self.transport.close()
        except Exception as e:
            logger.warning("transport.close: %s", e)


def create_session(
    device_path: str, baudrate: int = DEFAULT_BAUDRATE
) -> MpremoteSession:
    """Build a real SerialTransport-backed MpremoteSession.

    The caller must have closed any of its own pyserial handles on
    ``device_path`` before calling (SerialTransport claims the port
    with ``exclusive=True`` by default).

    Returns a session that owns the transport; `exit()` (or the
    context manager's `__exit__`) closes both the raw REPL and the
    underlying pyserial port.
    """
    transport = SerialTransport(device_path, baudrate=baudrate)
    return _ManagedSession(transport)
