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
        profile_trace.trace('x.csv', 'a,b', [1, 2], recording_id=profile_trace.NO_RECORDING)
        assert not (tmp_path / 'x.csv').exists()

    def test_timer_is_noop_when_disabled(self, tmp_path):
        profile_trace._output_dir = tmp_path
        with profile_trace.timer('x.csv', 'a,b', lambda: [1]):
            pass
        assert not (tmp_path / 'x.csv').exists()


def _run_dir(base):
    """The single run directory enable(base) created under ``base``.

    enable() treats its argument as the BASE for timestamped per-run
    directories, so a test that wrote one run's rows finds them one level
    down rather than at the path it passed.
    """
    runs = [p for p in Path(base).iterdir() if p.is_dir()]
    assert len(runs) == 1, f'expected exactly one run dir under {base}, found {runs}'
    return runs[0]


class TestEnableDisable:
    def test_enable_creates_output_dir(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path / 'profile_out')
        assert _run_dir(tmp_path / 'profile_out').is_dir()
        assert profile_trace.ENABLE_PROFILE_TRACE is True

    def test_second_enable_starts_a_new_run(self, tmp_path):
        """Two runs in one process must not share a directory.

        enable() used to return early when tracing was already on, so a
        session that changed one measurement axis between recordings appended
        both to the same CSVs with nothing in the rows saying which
        configuration produced which.
        """
        profile_trace.enable(output_dir=tmp_path)
        first = profile_trace._output_dir
        profile_trace.enable(output_dir=tmp_path)
        second = profile_trace._output_dir
        assert first != second
        assert first.is_dir() and second.is_dir()

    def test_disable_flushes_and_closes(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        run = profile_trace._output_dir
        profile_trace.trace('t.csv', 'a,b', [1, 2], recording_id=profile_trace.NO_RECORDING)
        profile_trace.disable()
        assert profile_trace.ENABLE_PROFILE_TRACE is False
        content = (run / 't.csv').read_text()
        assert 'a,b' in content
        assert '1,2' in content


class TestTrace:
    def test_writes_header_on_first_row(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('a.csv', 'col1,col2', ['x', 42], recording_id='rec1')
        content = (_run_dir(tmp_path) / 'a.csv').read_text()
        assert content.splitlines() == ['recording_id,col1,col2', 'rec1,x,42']

    def test_does_not_duplicate_header(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('a.csv', 'col1,col2', ['x', 1], recording_id='r')
        profile_trace.trace('a.csv', 'col1,col2', ['y', 2], recording_id='r')
        lines = (_run_dir(tmp_path) / 'a.csv').read_text().splitlines()
        assert lines == ['recording_id,col1,col2', 'r,x,1', 'r,y,2']

    def test_rows_from_two_recordings_stay_attributable(self, tmp_path):
        """Overlapping recordings must not be averaged together.

        A protocol step's write runs on the file lane while the next step
        captures, so rows from two recordings interleave in one file. Without
        the identity column the only way to summarize them is to average
        across both, which yields a plausible wrong rate rather than a
        visible failure.
        """
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('m.csv', 'v', [1], recording_id='recA')
        profile_trace.trace('m.csv', 'v', [2], recording_id='recB')
        profile_trace.trace('m.csv', 'v', [3], recording_id='recA')
        rows = _read_rows(_run_dir(tmp_path) / 'm.csv')
        assert rows[0] == ['recording_id', 'v']
        assert [r[0] for r in rows[1:]] == ['recA', 'recB', 'recA']


class TestRequiredIdentity:
    """A row that cannot be attributed to a recording must be impossible.

    Not merely discouraged: the argument is keyword-only AND has no default,
    so a site that forgets it fails at the call rather than emitting a row
    that later gets averaged into someone else's numbers.
    """

    def test_trace_without_recording_id_raises(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        with pytest.raises(TypeError):
            profile_trace.trace('x.csv', 'a', [1])

    def test_batch_trace_without_recording_id_raises(self):
        with pytest.raises(TypeError):
            profile_trace.BatchTrace('x.csv', 'a')


class TestArityGuard:
    """A row whose field count disagrees with its header must raise.

    The sites that get this wrong are the ones a given run never exercises,
    so a header-shape check cannot cover it -- the check has to be per row.
    """

    def test_too_few_fields_raises(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        with pytest.raises(ValueError, match='row has 1 fields, header declares 2'):
            profile_trace.trace('x.csv', 'a,b', [1], recording_id='r')

    def test_too_many_fields_raises(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        with pytest.raises(ValueError):
            profile_trace.trace('x.csv', 'a,b', [1, 2, 3], recording_id='r')

    def test_batch_add_checks_arity(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        bt = profile_trace.BatchTrace('b.csv', 'a,b', 'r')
        with pytest.raises(ValueError):
            bt.add([1])


class TestBatchTrace:
    def test_rows_are_written_on_flush(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        bt = profile_trace.BatchTrace('b.csv', 'a,b', 'rec1')
        bt.add([1, 2])
        bt.add([3, 4])
        bt.flush()
        rows = _read_rows(_run_dir(tmp_path) / 'b.csv')
        assert rows == [['recording_id', 'a', 'b'], ['rec1', '1', '2'], ['rec1', '3', '4']]

    def test_batch_quotes_like_the_row_writer(self, tmp_path):
        """Both writers share one serialization path, so neither can drift.

        A second writer with its own escaping is how the column-shift defect
        would come back on the batched side alone.
        """
        profile_trace.enable(output_dir=tmp_path)
        bt = profile_trace.BatchTrace('b.csv', 'a,b', 'rec1')
        bt.add(['X,Y', 'end'])
        bt.flush()
        rows = _read_rows(_run_dir(tmp_path) / 'b.csv')
        assert rows[1] == ['rec1', 'X,Y', 'end']

    def test_disable_flushes_pending_batches(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        run = profile_trace._output_dir
        bt = profile_trace.BatchTrace('b.csv', 'a', 'rec1')
        bt.add([1])
        profile_trace.disable()
        rows = _read_rows(run / 'b.csv')
        assert rows[1] == ['rec1', '1']

    def test_batch_writes_nothing_when_disabled(self, tmp_path):
        profile_trace._output_dir = tmp_path
        bt = profile_trace.BatchTrace('b.csv', 'a', 'rec1')
        bt.add([1])
        bt.flush()
        assert not (tmp_path / 'b.csv').exists()


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
        profile_trace.trace('c.csv', 'a,b,c', ['x', 'Well A1, BF', 'z'], recording_id='r')
        rows = _read_rows(_run_dir(tmp_path) / 'c.csv')
        assert rows[0] == ['recording_id', 'a', 'b', 'c']
        assert rows[1] == ['r', 'x', 'Well A1, BF', 'z']

    def test_motion_trace_multi_axis_field_stays_one_column(self, tmp_path):
        # The exact shape motion.py emits: header of 5, axis field 'X,Y'.
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace(
            'motion_trace.csv',
            'ts_ms,duration_ms,event,axis,detail',
            [1234, '0.500', 'poll', 'X,Y', ''],
            recording_id=profile_trace.NO_RECORDING,
        )
        rows = _read_rows(_run_dir(tmp_path) / 'motion_trace.csv')
        assert len(rows[1]) == len(rows[0]) == 6
        assert rows[1][4] == 'X,Y'

    def test_quote_bearing_field_round_trips(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('q.csv', 'a,b', ['say "hi"', 'end'], recording_id='r')
        rows = _read_rows(_run_dir(tmp_path) / 'q.csv')
        assert rows[1] == ['r', 'say "hi"', 'end']

    def test_newline_bearing_field_stays_one_record(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('n.csv', 'a,b', ['line1\nline2', 'end'], recording_id='r')
        rows = _read_rows(_run_dir(tmp_path) / 'n.csv')
        assert len(rows) == 2, 'embedded newline split the row into two records'
        assert rows[1] == ['r', 'line1\nline2', 'end']

    def test_rows_are_not_double_spaced(self, tmp_path):
        """csv.writer on a handle opened without newline='' doubles line
        endings on Windows. Assert no blank records survive the round trip."""
        profile_trace.enable(output_dir=tmp_path)
        profile_trace.trace('d.csv', 'a,b', [1, 2], recording_id='r')
        profile_trace.trace('d.csv', 'a,b', [3, 4], recording_id='r')
        rows = _read_rows(_run_dir(tmp_path) / 'd.csv')
        assert rows == [['recording_id', 'a', 'b'], ['r', '1', '2'], ['r', '3', '4']]


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


class TestEveryTraceSitePassesIdentity:
    """The suite must exercise tracing ON, not merely tracing's existence.

    recording_id was made required while one caller -- TimedLock.__exit__ --
    still omitted it. Every test here had tracing enabled OR touched a lock,
    never both, so 5000+ tests passed over a startup crash: enabling the flag
    raised TypeError on the first lock release, which is inside
    illumination.leds_off() during scope initialize. A simulator launch found
    it in seconds. These two tests are the cheap standing version of that.
    """

    def test_timed_lock_writes_a_row_with_tracing_enabled(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        lock = profile_trace.TimedLock(threading.Lock(), name='unit.lock')
        with lock:
            pass
        profile_trace.disable()
        rows = _read_rows(_run_dir(tmp_path) / 'lock_trace.csv')
        assert rows[0][0] == 'recording_id'
        assert rows[1][0] == profile_trace.NO_RECORDING
        assert rows[1][3] == 'unit.lock'

    def test_no_trace_call_site_omits_recording_id(self):
        """Structural cluster guard over every call, however it is spelled.

        A grep for 'profile_trace.trace(' cannot see the bare 'trace(' call
        that shipped this defect, so the check walks call nodes instead.
        """
        offenders = []
        for rel_path, tree in iter_package_modules(['drivers', 'lib', 'modules', 'ui']):
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, 'id', None)
                if name != 'trace':
                    continue
                if not any(k.arg == 'recording_id' for k in node.keywords):
                    offenders.append(f'{rel_path}:{node.lineno}')
        assert not offenders, (
            'trace() called without recording_id at: '
            + ', '.join(offenders)
            + '. The argument is required; omitting it raises at runtime, '
            'which for a lock means every acquire/release.'
        )


class TestDiagnosticOutputIsWritable:
    """Diagnostic output must never default to the working directory.

    An installed build runs with its CWD in the install folder, where mkdir is
    denied. profile_trace's gate calls enable() at import, so a CWD-relative
    default does not merely fail to trace -- the PermissionError escapes a
    module-level call and the application does not start. The same shape sits
    in the cProfile helper, reached whenever its own flag is set.
    """

    def test_default_output_dir_is_under_appdata(self, tmp_path, monkeypatch):
        import lvp_logger

        monkeypatch.setattr(lvp_logger, 'lvp_appdata', str(tmp_path))
        profile_trace._base_dir = None  # forget any base a prior test supplied
        profile_trace.enable()
        run = profile_trace._output_dir
        assert run.is_absolute(), f'{run} is relative and would resolve against the CWD'
        assert tmp_path in run.parents
        assert run.parent == tmp_path / 'logs' / 'profile'

    def test_cprofile_default_path_is_under_appdata(self, tmp_path, monkeypatch):
        import lvp_logger

        from modules.profiling_utils import ProfilingHelper

        monkeypatch.setattr(lvp_logger, 'lvp_appdata', str(tmp_path))
        helper = ProfilingHelper()
        out = helper._profile_artifact_path
        assert out.is_absolute()
        assert tmp_path in out.parents
        assert out.parent == tmp_path / 'logs' / 'cprofile'

    def test_no_module_declares_a_cwd_relative_log_path(self):
        """The cluster guard: a literal './logs...' or 'logs/...' default.

        Two sites shipped this shape independently, so the fix is only durable
        if a third cannot be added quietly. Scans source rather than behaviour
        because the failure needs an unwritable CWD to reproduce, which no
        developer machine has.
        """
        offenders = []
        for rel_path, tree in iter_package_modules(['drivers', 'lib', 'modules', 'ui']):
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                # './logs...' is the CWD-relative marker -- both shipped
                # instances wrote it that way. A bare 'logs/...' is allowed
                # because it is only ever joined to an already-resolved root.
                if node.value.startswith('./logs'):
                    offenders.append(f'{rel_path}:{node.lineno} {node.value!r}')
        assert not offenders, (
            'CWD-relative diagnostic output path(s): '
            + ', '.join(offenders)
            + '. Anchor to profile_trace._appdata_root() -- an installed build '
            'cannot write to its working directory.'
        )


class TestTimer:
    def test_timer_writes_duration(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        with profile_trace.timer('t.csv', 'ts_ms,duration_ms,label', lambda: ['work']):
            time.sleep(0.01)
        lines = (_run_dir(tmp_path) / 't.csv').read_text().splitlines()
        assert len(lines) == 2  # header + row
        # Columns are recording_id, ts_ms, duration_ms, label.
        row = lines[1].split(',')
        assert row[0] == profile_trace.NO_RECORDING
        assert float(row[2]) >= 9  # ~10 ms, allow jitter
        assert row[3] == 'work'

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
        _r = _run_dir(tmp_path)
        assert not (_r / 't.csv').exists() or (_r / 't.csv').read_text().strip() == ''


class TestThreadSafety:
    def test_concurrent_writes_do_not_corrupt(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        threads = []
        for i in range(10):
            t = threading.Thread(
                target=lambda idx=i: [
                    profile_trace.trace('c.csv', 'thread,n', [idx, n], recording_id='r')
                    for n in range(50)
                ]
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        lines = (_run_dir(tmp_path) / 'c.csv').read_text().splitlines()
        # 500 data rows + 1 header
        assert len(lines) == 501
        # every data row has exactly 3 fields -- recording_id, thread, n --
        # so no two threads' rows interleaved within a line
        for line in lines[1:]:
            assert len(line.split(',')) == 3


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
        assert _run_dir(tmp_path / 'settings_out').is_dir()
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
