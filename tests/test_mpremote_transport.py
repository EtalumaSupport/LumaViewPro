# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for drivers/mpremote_transport.py.

Drives the adapter through FakeTransport (tests/fake_transport.py)
so no real hardware or pyserial port is needed.
"""

import hashlib
import logging
from unittest.mock import MagicMock, call

import pytest

from mpremote.transport import TransportError, TransportExecError

from drivers.mpremote_transport import (
    MpremoteSession,
    WRITE_VERIFY_RETRIES,
    _CTRL_B,
    _CTRL_C,
    _CTRL_D,
    _capture_stdout_to_logger,
    _send_exit_sequence,
)
from tests.fake_transport import FakeTransport


# ---------------------------------------------------------------------------
# Context manager + lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_enter_then_exit_via_context_manager(self):
        ft = FakeTransport()
        with MpremoteSession(ft) as session:
            assert ft.in_raw_repl is True
            assert session.transport is ft
        assert ft.in_raw_repl is False

    def test_explicit_enter_and_exit(self):
        ft = FakeTransport()
        session = MpremoteSession(ft)
        session.enter()
        assert ft.in_raw_repl is True
        session.exit()
        assert ft.in_raw_repl is False

    def test_exit_is_idempotent(self):
        ft = FakeTransport()
        session = MpremoteSession(ft)
        session.enter()
        session.exit()
        session.exit()  # second exit is a no-op
        assert ft.in_raw_repl is False

    def test_exit_swallows_transport_errors(self):
        ft = FakeTransport()
        session = MpremoteSession(ft)
        session.enter()
        ft.raise_on["exit_raw_repl"] = TransportError("simulated")
        # Must not raise — best-effort exit.
        session.exit()
        assert session._in_raw_repl is False

    def test_operations_before_enter_raise(self):
        ft = FakeTransport()
        session = MpremoteSession(ft)
        with pytest.raises(RuntimeError, match="not in raw REPL"):
            session.list_files()
        with pytest.raises(RuntimeError, match="not in raw REPL"):
            session.read_file("x")
        with pytest.raises(RuntimeError, match="not in raw REPL"):
            session.write_file("x", b"y")
        with pytest.raises(RuntimeError, match="not in raw REPL"):
            session.raw_exec("print(1)")


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------

class TestListFiles:
    def test_list_returns_names(self):
        ft = FakeTransport({"a.py": b"1", "b.json": b"{}"})
        with MpremoteSession(ft) as session:
            names = session.list_files()
        assert sorted(names) == ["a.py", "b.json"]

    def test_list_empty(self):
        ft = FakeTransport()
        with MpremoteSession(ft) as session:
            assert session.list_files() == []


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_read_happy_path(self):
        ft = FakeTransport({"cal.json": b'{"k":1}'})
        with MpremoteSession(ft) as session:
            assert session.read_file("cal.json") == b'{"k":1}'

    def test_read_nonexistent_returns_none(self):
        ft = FakeTransport()
        with MpremoteSession(ft) as session:
            assert session.read_file("missing") is None

    def test_read_without_verify(self):
        ft = FakeTransport({"f": b"data"})
        with MpremoteSession(ft) as session:
            result = session.read_file("f", verify=False)
        assert result == b"data"
        # Should have read exactly once with verify=False
        reads = [c for c in ft.call_log if c[0] == "fs_readfile"]
        assert len(reads) == 1

    def test_read_with_verify_reads_twice(self):
        ft = FakeTransport({"f": b"x" * 500})
        with MpremoteSession(ft) as session:
            result = session.read_file("f", verify=True)
        assert result == b"x" * 500
        reads = [c for c in ft.call_log if c[0] == "fs_readfile"]
        assert len(reads) == 2


# ---------------------------------------------------------------------------
# write_file — the critical path
# ---------------------------------------------------------------------------

class TestWriteFile:
    def test_write_happy_path(self):
        ft = FakeTransport()
        payload = b"print('hello')\n" * 100
        with MpremoteSession(ft) as session:
            ok = session.write_file("main.py", payload)
        assert ok is True
        assert ft.files.get("main.py") == payload
        assert "main.py.tmp" not in ft.files
        # No .bak since there was no pre-existing file.
        assert "main.py.bak" not in ft.files

    def test_write_creates_bak_when_previous_exists(self):
        old = b"# v1\n"
        ft = FakeTransport({"main.py": old})
        new = b"# v2\n" * 1000
        with MpremoteSession(ft) as session:
            ok = session.write_file("main.py", new)
        assert ok is True
        assert ft.files["main.py"] == new
        assert ft.files["main.py.bak"] == old

    def test_write_removes_stale_bak_before_rename(self):
        ft = FakeTransport({
            "main.py": b"# v1\n",
            "main.py.bak": b"# OLD OLD\n",
        })
        new = b"# v2\n"
        with MpremoteSession(ft) as session:
            assert session.write_file("main.py", new) is True
        assert ft.files["main.py"] == new
        assert ft.files["main.py.bak"] == b"# v1\n"  # old .bak replaced

    def test_write_retries_on_transport_error(self):
        ft = FakeTransport()
        ft.fail_next_write = 2  # first two writes fail, third succeeds
        payload = b"retry data"
        with MpremoteSession(ft) as session:
            ok = session.write_file("main.py", payload)
        assert ok is True
        assert ft.files["main.py"] == payload
        # Three fs_writefile attempts total (2 failures + 1 success)
        writes = [c for c in ft.call_log if c[0] == "fs_writefile"]
        assert len(writes) == 3

    def test_write_gives_up_after_max_retries(self):
        ft = FakeTransport()
        ft.fail_next_write = WRITE_VERIFY_RETRIES + 5  # always fail
        with MpremoteSession(ft) as session:
            ok = session.write_file("main.py", b"data")
        assert ok is False
        writes = [c for c in ft.call_log if c[0] == "fs_writefile"]
        assert len(writes) == WRITE_VERIFY_RETRIES

    def test_write_rejects_sha256_mismatch(self):
        ft = FakeTransport()
        ft.fs_hashfile_override = b"\x00" * 32  # wrong hash every time
        with MpremoteSession(ft) as session:
            ok = session.write_file("main.py", b"data")
        assert ok is False
        # The bad .tmp got cleaned up each attempt.
        assert "main.py.tmp" not in ft.files
        assert "main.py" not in ft.files

    def test_write_verifies_exact_sha256(self):
        ft = FakeTransport()
        payload = b"some payload bytes"
        expected = hashlib.sha256(payload).digest()
        with MpremoteSession(ft) as session:
            assert session.write_file("x.bin", payload) is True
        # FakeTransport.fs_hashfile computes real sha256 over the
        # stored bytes — this asserts end-to-end integrity.
        stored = ft.files["x.bin"]
        assert hashlib.sha256(stored).digest() == expected

    def test_write_rejects_empty_filename(self):
        ft = FakeTransport()
        with MpremoteSession(ft) as session:
            assert session.write_file("", b"x") is False

    def test_write_uses_basename(self):
        """Path-like filenames get reduced to basename for the device."""
        ft = FakeTransport()
        with MpremoteSession(ft) as session:
            ok = session.write_file("/some/host/path/main.py", b"data")
        assert ok is True
        assert "main.py" in ft.files
        assert "/some/host/path/main.py" not in ft.files


# ---------------------------------------------------------------------------
# raw_exec
# ---------------------------------------------------------------------------

class TestRawExec:
    def test_exec_success_returns_stdout_stderr_tuple(self):
        ft = FakeTransport({"foo": b"hello"})
        with MpremoteSession(ft) as session:
            result = session.raw_exec("os.remove('foo')")
        assert result == (b"", b"")
        assert "foo" not in ft.files

    def test_exec_error_preserves_device_traceback(self):
        ft = FakeTransport()
        # FakeTransport.exec raises TransportExecError on unsupported
        # commands — use that to test the adapter's error-path wrapping.
        with MpremoteSession(ft) as session:
            result = session.raw_exec("completely.unsupported.command()")
        assert result is not None
        stdout, stderr = result
        assert stdout == b""
        assert b"unsupported" in stderr or b"Traceback" in stderr or stderr

    def test_exec_transport_error_returns_none(self):
        ft = FakeTransport()
        ft.raise_on["exec"] = TransportError("link down")
        with MpremoteSession(ft) as session:
            assert session.raw_exec("x") is None


# ---------------------------------------------------------------------------
# Stdout capture shim
# ---------------------------------------------------------------------------

class TestStdoutShim:
    def test_captures_info_text(self, caplog):
        caplog.set_level(logging.INFO, logger="drivers.mpremote_transport")
        with _capture_stdout_to_logger():
            print("hello from mpremote")
        assert any(
            "hello from mpremote" in r.message for r in caplog.records
        )

    def test_escalates_error_text_to_warning(self, caplog):
        caplog.set_level(logging.INFO, logger="drivers.mpremote_transport")
        with _capture_stdout_to_logger():
            print("Error: something failed")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("something failed" in r.message for r in warnings)

    def test_empty_stdout_is_silent(self, caplog):
        caplog.set_level(logging.INFO, logger="drivers.mpremote_transport")
        with _capture_stdout_to_logger():
            pass
        # No log records produced when nothing was printed.
        assert not any(
            "mpremote:" in r.message for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Ordering / call-log regressions
# ---------------------------------------------------------------------------

class TestExitSequence:
    """Full post-exit soft-reset sequence used by _ManagedSession.

    mpremote's Transport.exit_raw_repl is Ctrl-B only. raw_repl.py did
    Ctrl-C → Ctrl-B → Ctrl-D + boot wait + drain so the board resumed
    main.py after exit. _send_exit_sequence reproduces that. All
    SerialBoard callers depend on this contract.
    """

    def _build_mock_serial(self, pending_bytes=b""):
        """Mock pyserial Serial that reports `pending_bytes` as buffered."""
        m = MagicMock()
        m.in_waiting = len(pending_bytes)
        # Each read(n) consumes from the buffer, then reports empty.
        buf = [pending_bytes]

        def _read(n):
            data = buf[0][:n]
            buf[0] = buf[0][n:]
            m.in_waiting = len(buf[0])
            return data

        m.read.side_effect = _read
        return m

    def test_writes_ctrl_c_b_d_in_order(self, monkeypatch):
        # No-op sleep so the test runs instantly.
        monkeypatch.setattr(
            "drivers.mpremote_transport.time.sleep", lambda *_: None
        )
        ser = self._build_mock_serial()
        _send_exit_sequence(ser, boot_wait=0)
        # Expected write sequence: Ctrl-C, Ctrl-B, Ctrl-D.
        writes = [c.args[0] for c in ser.write.call_args_list]
        assert writes == [_CTRL_C, _CTRL_B, _CTRL_D]

    def test_drains_pending_bytes(self, monkeypatch):
        monkeypatch.setattr(
            "drivers.mpremote_transport.time.sleep", lambda *_: None
        )
        ser = self._build_mock_serial(b"boot noise from firmware\n")
        _send_exit_sequence(ser, boot_wait=0)
        # read() should have been called at least once to drain.
        assert ser.read.called
        # After draining, in_waiting is 0.
        assert ser.in_waiting == 0

    def test_tolerates_write_exception(self, monkeypatch):
        monkeypatch.setattr(
            "drivers.mpremote_transport.time.sleep", lambda *_: None
        )
        ser = MagicMock()
        ser.write.side_effect = OSError("port disappeared")
        # Must not raise — best-effort exit.
        _send_exit_sequence(ser, boot_wait=0)

    def test_tolerates_drain_exception(self, monkeypatch):
        monkeypatch.setattr(
            "drivers.mpremote_transport.time.sleep", lambda *_: None
        )
        ser = MagicMock()
        ser.in_waiting = 1
        ser.read.side_effect = OSError("read failure")
        _send_exit_sequence(ser, boot_wait=0)


class TestCallOrdering:
    def test_write_call_sequence(self):
        """write_file must: fs_writefile → fs_hashfile → exec(rename)."""
        ft = FakeTransport({"main.py": b"old"})
        with MpremoteSession(ft) as session:
            session.write_file("main.py", b"new")

        methods = [c[0] for c in ft.call_log]
        # Find the critical sequence: writefile then hashfile then renames.
        assert "fs_writefile" in methods
        write_idx = methods.index("fs_writefile")
        hash_idx = methods.index("fs_hashfile", write_idx)
        assert write_idx < hash_idx
        # Renames (via exec) come after the hash verify.
        exec_after_hash = [
            i for i, m in enumerate(methods) if m == "exec" and i > hash_idx
        ]
        assert exec_after_hash, "no rename exec() calls after hash verify"
