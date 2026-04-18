# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the opt-in profile_trace module."""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from modules import profile_trace


@pytest.fixture(autouse=True)
def _reset_profile_trace():
    """Ensure each test starts with profile_trace disabled + clean state."""
    profile_trace.disable()
    yield
    profile_trace.disable()


class TestDefaultOff:
    def test_default_disabled(self):
        assert profile_trace.ENABLE_PROFILE_TRACE is False

    def test_trace_is_noop_when_disabled(self, tmp_path):
        profile_trace._output_dir = tmp_path
        profile_trace.trace("x.csv", "a,b", [1, 2])
        assert not (tmp_path / "x.csv").exists()

    def test_timer_is_noop_when_disabled(self, tmp_path):
        profile_trace._output_dir = tmp_path
        with profile_trace.timer("x.csv", "a,b", lambda: [1]):
            pass
        assert not (tmp_path / "x.csv").exists()


class TestEnableDisable:
    def test_enable_creates_output_dir(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path / "profile_out")
        assert (tmp_path / "profile_out").is_dir()
        assert profile_trace.ENABLE_PROFILE_TRACE is True

    def test_enable_is_idempotent(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path / "p1")
        profile_trace.enable(output_dir=tmp_path / "p2")
        assert (tmp_path / "p1").is_dir()
        assert not (tmp_path / "p2").exists()

    def test_disable_flushes_and_closes(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace("t.csv", "a,b", [1, 2])
        profile_trace.disable()
        assert profile_trace.ENABLE_PROFILE_TRACE is False
        content = (tmp_path / "t.csv").read_text()
        assert "a,b" in content
        assert "1,2" in content


class TestTrace:
    def test_writes_header_on_first_row(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace("a.csv", "col1,col2", ["x", 42])
        content = (tmp_path / "a.csv").read_text()
        assert content.splitlines() == ["col1,col2", "x,42"]

    def test_does_not_duplicate_header(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace("a.csv", "col1,col2", ["x", 1])
        profile_trace.trace("a.csv", "col1,col2", ["y", 2])
        lines = (tmp_path / "a.csv").read_text().splitlines()
        assert lines == ["col1,col2", "x,1", "y,2"]


class TestTimer:
    def test_timer_writes_duration(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        with profile_trace.timer("t.csv", "ts_ms,duration_ms,label", lambda: ["work"]):
            time.sleep(0.01)
        lines = (tmp_path / "t.csv").read_text().splitlines()
        assert len(lines) == 2  # header + row
        row = lines[1].split(",")
        assert float(row[1]) >= 9  # ~10 ms, allow jitter
        assert row[2] == "work"

    def test_timer_extra_fn_not_called_when_disabled(self, tmp_path):
        calls = []

        def fn():
            calls.append(1)
            return ["x"]

        with profile_trace.timer("t.csv", "a,b,c", fn):
            pass
        assert calls == []

    def test_timer_handles_extra_fn_exception(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)

        def boom():
            raise RuntimeError("nope")

        with profile_trace.timer("t.csv", "ts_ms,duration_ms,label", boom):
            pass
        assert not (tmp_path / "t.csv").exists() or \
               (tmp_path / "t.csv").read_text().strip() == ""


class TestThreadSafety:
    def test_concurrent_writes_do_not_corrupt(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        threads = []
        for i in range(10):
            t = threading.Thread(
                target=lambda idx=i: [
                    profile_trace.trace("c.csv", "thread,n", [idx, n])
                    for n in range(50)
                ]
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        lines = (tmp_path / "c.csv").read_text().splitlines()
        # 500 data rows + 1 header
        assert len(lines) == 501
        # every data row has exactly 2 fields (no interleaving)
        for line in lines[1:]:
            assert len(line.split(",")) == 2


class TestEnvActivation:
    def test_env_var_enables_at_import(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LVP_PROFILE_TRACE", "1")
        monkeypatch.setenv("LVP_PROFILE_TRACE_DIR", str(tmp_path / "env_out"))
        profile_trace.disable()
        import importlib
        importlib.reload(profile_trace)
        assert profile_trace.ENABLE_PROFILE_TRACE is True
        assert (tmp_path / "env_out").is_dir()
        profile_trace.disable()
