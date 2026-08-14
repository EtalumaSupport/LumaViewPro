# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the opt-in profile_trace module."""

import ast
import csv
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from lib import profile_trace
from tests.ast_seams import iter_package_modules


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
        profile_trace.trace('x.csv', 'a,b', [1, 2])
        assert not (tmp_path / 'x.csv').exists()

    def test_timer_is_noop_when_disabled(self, tmp_path):
        profile_trace._output_dir = tmp_path
        with profile_trace.timer('x.csv', 'a,b', lambda: [1]):
            pass
        assert not (tmp_path / 'x.csv').exists()


class TestEnableDisable:
    def test_enable_creates_output_dir(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path / 'profile_out')
        assert (tmp_path / 'profile_out').is_dir()
        assert profile_trace.ENABLE_PROFILE_TRACE is True

    def test_enable_is_idempotent(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path / 'p1')
        profile_trace.enable(output_dir=tmp_path / 'p2')
        assert (tmp_path / 'p1').is_dir()
        assert not (tmp_path / 'p2').exists()

    def test_disable_flushes_and_closes(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('t.csv', 'a,b', [1, 2])
        profile_trace.disable()
        assert profile_trace.ENABLE_PROFILE_TRACE is False
        content = (tmp_path / 't.csv').read_text()
        assert 'a,b' in content
        assert '1,2' in content


class TestTrace:
    def test_writes_header_on_first_row(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('a.csv', 'col1,col2', ['x', 42])
        content = (tmp_path / 'a.csv').read_text()
        assert content.splitlines() == ['col1,col2', 'x,42']

    def test_does_not_duplicate_header(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('a.csv', 'col1,col2', ['x', 1])
        profile_trace.trace('a.csv', 'col1,col2', ['y', 2])
        lines = (tmp_path / 'a.csv').read_text().splitlines()
        assert lines == ['col1,col2', 'x,1', 'y,2']


def _read_rows(path):
    """Parse a trace CSV the way any consumer would.

    ``newline=''`` is csv's documented requirement on the READ side too --
    without it a quoted field containing a newline is split across records.
    """
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.reader(fh))


class TestFieldContentCannotShiftColumns:
    """A field's CONTENT must never change how many columns a row has.

    These rows were shifted, not hypothetically but in production:
    modules/lumascope_api/motion.py builds its `axis` field as
    ','.join(moving_axes), so every simultaneously-moving XY poll wrote a
    6-field row under a 5-column header. Two other call sites defended
    themselves with a per-call .replace(',', ';'); two did not. Quoting
    belongs to the writer, so no call site has to remember.
    """

    def test_comma_bearing_field_stays_one_column(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('c.csv', 'a,b,c', ['x', 'Well A1, BF', 'z'])
        rows = _read_rows(tmp_path / 'c.csv')
        assert rows[0] == ['a', 'b', 'c']
        assert rows[1] == ['x', 'Well A1, BF', 'z']

    def test_motion_trace_multi_axis_field_stays_one_column(self, tmp_path):
        # The exact shape motion.py emits: header of 5, axis field 'X,Y'.
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace(
            'motion_trace.csv',
            'ts_ms,duration_ms,event,axis,detail',
            [1234, '0.500', 'poll', 'X,Y', ''],
        )
        rows = _read_rows(tmp_path / 'motion_trace.csv')
        assert len(rows[1]) == len(rows[0]) == 5
        assert rows[1][3] == 'X,Y'

    def test_quote_bearing_field_round_trips(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('q.csv', 'a,b', ['say "hi"', 'end'])
        rows = _read_rows(tmp_path / 'q.csv')
        assert rows[1] == ['say "hi"', 'end']

    def test_newline_bearing_field_stays_one_record(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('n.csv', 'a,b', ['line1\nline2', 'end'])
        rows = _read_rows(tmp_path / 'n.csv')
        assert len(rows) == 2, 'embedded newline split the row into two records'
        assert rows[1] == ['line1\nline2', 'end']

    def test_rows_are_not_double_spaced(self, tmp_path):
        """csv.writer on a handle opened without newline='' doubles line
        endings on Windows. Assert no blank records survive the round trip."""
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('d.csv', 'a,b', [1, 2])
        profile_trace.trace('d.csv', 'a,b', [3, 4])
        rows = _read_rows(tmp_path / 'd.csv')
        assert rows == [['a', 'b'], ['1', '2'], ['3', '4']]


def _imports_profile_trace(node):
    """Whether an AST import statement pulls in profile_trace."""
    if isinstance(node, ast.Import):
        return any(a.name.endswith('profile_trace') for a in node.names)
    if isinstance(node, ast.ImportFrom):
        return any(a.name == 'profile_trace' for a in node.names)
    return False


class TestProfileTraceImportIsUnconditional:
    """No consumer may wrap its profile_trace import in a swallowing guard.

    lib/profile_trace imports only stdlib at module scope and its one logger
    import carries a real fallback, so `from lib import profile_trace` cannot
    legitimately raise. A guard around it therefore defends nothing and can
    only convert a wrong import path into permanent silent disablement -- which
    is exactly what happened: the protocol capture path imported it from the
    wrong package and emitted nothing at all from the day it was written until
    a packaging warning surfaced it. An ImportError here must be loud.
    """

    def test_no_consumer_guards_the_import(self):
        offenders = []
        for rel_path, tree in iter_package_modules(['drivers', 'lib', 'modules', 'ui']):
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                guarded = any(
                    _imports_profile_trace(stmt) for stmt in ast.walk(node) if stmt in node.body
                )
                if not guarded:
                    continue
                for handler in node.handlers:
                    names = (
                        [handler.type]
                        if not isinstance(handler.type, ast.Tuple)
                        else handler.type.elts
                    )
                    if any(getattr(n, 'id', None) == 'ImportError' for n in names if n):
                        offenders.append(f'{rel_path}:{node.lineno}')
        assert not offenders, (
            'profile_trace import wrapped in `except ImportError` at: '
            + ', '.join(offenders)
            + '. The import cannot legitimately fail; a guard here turns a '
            'wrong path into a permanently dead trace.'
        )


class TestTimer:
    def test_timer_writes_duration(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        with profile_trace.timer('t.csv', 'ts_ms,duration_ms,label', lambda: ['work']):
            time.sleep(0.01)
        lines = (tmp_path / 't.csv').read_text().splitlines()
        assert len(lines) == 2  # header + row
        row = lines[1].split(',')
        assert float(row[1]) >= 9  # ~10 ms, allow jitter
        assert row[2] == 'work'

    def test_timer_extra_fn_not_called_when_disabled(self, tmp_path):
        calls = []

        def fn():
            calls.append(1)
            return ['x']

        with profile_trace.timer('t.csv', 'a,b,c', fn):
            pass
        assert calls == []

    def test_timer_handles_extra_fn_exception(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)

        def boom():
            raise RuntimeError('nope')

        with profile_trace.timer('t.csv', 'ts_ms,duration_ms,label', boom):
            pass
        assert not (tmp_path / 't.csv').exists() or (tmp_path / 't.csv').read_text().strip() == ''


class TestThreadSafety:
    def test_concurrent_writes_do_not_corrupt(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        threads = []
        for i in range(10):
            t = threading.Thread(
                target=lambda idx=i: [
                    profile_trace.trace('c.csv', 'thread,n', [idx, n]) for n in range(50)
                ]
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        lines = (tmp_path / 'c.csv').read_text().splitlines()
        # 500 data rows + 1 header
        assert len(lines) == 501
        # every data row has exactly 2 fields (no interleaving)
        for line in lines[1:]:
            assert len(line.split(',')) == 2


class TestTimedLockInvariantThreshold:
    """warn_hold_threshold_ms fires a logger.warning when a TimedLock is
    held longer than the configured threshold. Runs regardless of the
    trace-CSV feature flag -- the invariant is a structural guard, not
    instrumentation. The LVP logger has propagate=False in production
    so caplog can't observe it directly; tests mock the logger.warning
    call instead."""

    def test_warns_when_hold_exceeds_threshold(self, monkeypatch):
        warnings_captured = []
        import lvp_logger

        monkeypatch.setattr(
            lvp_logger.logger,
            'warning',
            lambda msg, *a, **kw: warnings_captured.append(msg),
        )
        lock = profile_trace.TimedLock(
            threading.Lock(),
            name='test_invariant_lock',
            warn_hold_threshold_ms=1.0,
        )
        with lock:
            time.sleep(0.005)  # 5ms; well above 1ms threshold
        assert any('test_invariant_lock' in m and 'exceeded' in m for m in warnings_captured), (
            f'Expected hold-threshold warning; got: {warnings_captured}'
        )

    def test_no_warning_under_threshold(self, monkeypatch):
        warnings_captured = []
        import lvp_logger

        monkeypatch.setattr(
            lvp_logger.logger,
            'warning',
            lambda msg, *a, **kw: warnings_captured.append(msg),
        )
        lock = profile_trace.TimedLock(
            threading.Lock(),
            name='test_under_threshold',
            warn_hold_threshold_ms=100.0,
        )
        with lock:
            pass  # essentially zero hold
        assert not any('test_under_threshold' in m for m in warnings_captured)

    def test_default_no_threshold_no_warning(self, monkeypatch):
        # No warn_hold_threshold_ms means no invariant check; the TimedLock
        # is pure instrumentation (off when profile_trace_enabled is false).
        warnings_captured = []
        import lvp_logger

        monkeypatch.setattr(
            lvp_logger.logger,
            'warning',
            lambda msg, *a, **kw: warnings_captured.append(msg),
        )
        lock = profile_trace.TimedLock(threading.Lock(), name='no_threshold')
        with lock:
            time.sleep(0.01)  # 10ms; would exceed any reasonable threshold
        assert not any('no_threshold' in m for m in warnings_captured)

    def test_warning_active_when_trace_off(self, monkeypatch):
        """Threshold check fires regardless of ENABLE_PROFILE_TRACE state."""
        # _reset_profile_trace fixture leaves trace OFF.
        assert profile_trace.ENABLE_PROFILE_TRACE is False
        warnings_captured = []
        import lvp_logger

        monkeypatch.setattr(
            lvp_logger.logger,
            'warning',
            lambda msg, *a, **kw: warnings_captured.append(msg),
        )
        lock = profile_trace.TimedLock(
            threading.Lock(),
            name='trace_off_threshold_check',
            warn_hold_threshold_ms=0.5,
        )
        with lock:
            time.sleep(0.003)
        assert any('trace_off_threshold_check' in m for m in warnings_captured)


class TestSettingsActivation:
    def test_settings_key_enables_at_import(self, monkeypatch, tmp_path):
        out_dir = str(tmp_path / 'settings_out')
        monkeypatch.setattr(
            'modules.settings_init.load_profile_trace_setting',
            lambda directory: {'enabled': True, 'output_dir': out_dir},
        )
        profile_trace.disable()
        import importlib

        importlib.reload(profile_trace)
        assert profile_trace.ENABLE_PROFILE_TRACE is True
        assert (tmp_path / 'settings_out').is_dir()
        profile_trace.disable()

    def test_magicmock_settings_does_not_create_dirs(self, monkeypatch):
        """Bare MagicMock for `modules.settings_init` must not leak a
        real filesystem directory at import-time gate evaluation.

        Reproduces the condition surfaced 2026-05-28: test files
        register `sys.modules['modules.settings_init'] = MagicMock()`
        without configuring `load_profile_trace_setting`'s return value.
        The gate then dereferenced ['enabled'] (truthy MagicMock) and
        Path()-ified ['output_dir'] (another MagicMock), producing a
        stray `LumaViewPro/MagicMock/` directory at the repo root.
        """
        from unittest.mock import MagicMock

        monkeypatch.setattr(
            'modules.settings_init.load_profile_trace_setting',
            lambda directory: MagicMock(),
        )
        profile_trace.disable()
        import importlib

        importlib.reload(profile_trace)
        assert profile_trace.ENABLE_PROFILE_TRACE is False
        repo_root = Path(__file__).parent.parent
        assert not (repo_root / 'MagicMock').exists(), (
            'profile_trace gate leaked a real directory from a MagicMock '
            'load_profile_trace_setting() return value'
        )
