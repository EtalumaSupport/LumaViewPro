# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The expensive half of the metrics tick is opt-in, and says so in the log.

`psutil.open_files()` runs an `os.stat` per open handle, so it costs
O(handles) -- it is most expensive during the leak it exists to detect.
On Windows it enumerates through a C-extension call that holds the GIL,
stalling every thread in the process; it measured 134-306 ms per call
on healthy hosts. It also under-reports silently: a handle whose name
query times out is skipped, so a wedged tick logged `open_files=0`
against 34 immediately before.

So it is off unless asked for, and the flat-cost handle count carries
the leak trend instead.

The rendering matters as much as the gate. The metric has three states
-- measured, ran-and-failed, and never-asked-for -- and -1 was already
taken by the second. Letting the third fall through to -1 would print a
deliberate configuration as a malfunction in every log a reader opens.

The gate is read with `is True` rather than truthiness because nothing
validates the settings files: a hand-edited `"false"` is a truthy
string, and reading it loosely would switch the probe back on while its
operator believed it off.
"""

from collections import defaultdict
from unittest.mock import MagicMock

import pytest

from modules import app_context, common_utils, config_helpers


@pytest.fixture
def fake_proc(monkeypatch):
    """A process handle whose open_files() call is observable."""
    proc = MagicMock()
    proc.open_files.return_value = ['f1', 'f2', 'f3']
    monkeypatch.setattr(common_utils, '_self_process', lambda: proc)
    return proc


class _RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


def _drive_log_system_metrics(monkeypatch, settings, metrics, on_call=None):
    """Run log_system_metrics with its psutil-backed neighbours stubbed.

    It compares check_disk_space() against an int and subscripts the
    metrics mapping directly, so both need real values before the
    [HANDLE METRICS] block is reachable at all.
    """
    rec = _RecordingLogger()
    monkeypatch.setattr(config_helpers, 'metrics_logger', rec)
    monkeypatch.setattr(app_context, 'ctx', MagicMock())
    monkeypatch.setattr(config_helpers.common_utils, 'check_disk_space', lambda **k: 1.0e5)

    def _stub(**kwargs):
        if on_call is not None:
            on_call(kwargs)
        return metrics

    monkeypatch.setattr(config_helpers.common_utils, 'system_metrics', _stub)
    config_helpers.log_system_metrics(settings)
    return rec


def _routed_gate_value(monkeypatch, settings):
    """What collect_open_files does log_system_metrics pass down."""
    recorded = {}
    _drive_log_system_metrics(monkeypatch, settings, defaultdict(float), on_call=recorded.update)
    return recorded.get('collect_open_files')


class TestOpenFilesGate:
    def test_gate_off_does_not_call_open_files(self, fake_proc):
        common_utils.system_metrics(collect_open_files=False)
        fake_proc.open_files.assert_not_called()

    def test_gate_off_omits_the_key_entirely(self, fake_proc):
        metrics = common_utils.system_metrics(collect_open_files=False)
        # Absent, not -1: -1 already means "ran and failed".
        assert 'open_files_count' not in metrics

    def test_gate_on_measures(self, fake_proc):
        metrics = common_utils.system_metrics(collect_open_files=True)
        fake_proc.open_files.assert_called_once()
        assert metrics['open_files_count'] == 3

    def test_the_gate_argument_is_required(self, fake_proc):
        # A future caller must decide explicitly; a default is how the
        # ungated probe this file exists to fix came to be.
        with pytest.raises(TypeError):
            common_utils.system_metrics()

    @pytest.mark.parametrize('configured', ['false', 'true', 0, 1, None, '', 'yes', [], 'False'])
    def test_only_a_real_true_enables_the_probe(self, monkeypatch, configured, tmp_path):
        # 'true' and 'yes' are the dangerous half: truthy strings that a
        # loose read would honour.
        settings = {
            'live_folder': str(tmp_path),
            'profiling': {'open_files_enabled': configured},
        }
        assert _routed_gate_value(monkeypatch, settings) is False

    def test_a_real_true_enables_it(self, monkeypatch, tmp_path):
        settings = {
            'live_folder': str(tmp_path),
            'profiling': {'open_files_enabled': True},
        }
        assert _routed_gate_value(monkeypatch, settings) is True

    def test_missing_profiling_block_reads_as_off(self, monkeypatch, tmp_path):
        # Headless sessions can hold an empty settings dict, and the
        # default merge is best-effort -- absent must mean off.
        assert _routed_gate_value(monkeypatch, {'live_folder': str(tmp_path)}) is False


class TestHandleMetricsLine:
    def _emit(self, monkeypatch, tmp_path, metrics):
        # defaultdict(float): direct-subscript metric keys resolve to 0.0
        # while .get() still honours the caller's default for absent keys,
        # which is the distinction under test.
        backing = defaultdict(float)
        backing.update(metrics)
        rec = _drive_log_system_metrics(monkeypatch, {'live_folder': str(tmp_path)}, backing)
        return [m for m in rec.messages if '[HANDLE METRICS]' in m]

    def test_disabled_renders_as_off_not_as_a_failure(self, monkeypatch, tmp_path):
        lines = self._emit(monkeypatch, tmp_path, {'os_handles': 812})
        assert lines, 'no [HANDLE METRICS] line emitted'
        assert 'handles=812' in lines[0]
        assert 'open_files=off' in lines[0]
        assert 'open_files=-1' not in lines[0]

    def test_failure_still_renders_as_minus_one(self, monkeypatch, tmp_path):
        lines = self._emit(monkeypatch, tmp_path, {'os_handles': 812, 'open_files_count': -1})
        assert lines and 'open_files=-1' in lines[0]

    def test_measured_renders_the_count(self, monkeypatch, tmp_path):
        lines = self._emit(monkeypatch, tmp_path, {'os_handles': 812, 'open_files_count': 34})
        assert lines and 'open_files=34' in lines[0]

    def test_no_line_when_neither_metric_is_available(self, monkeypatch, tmp_path):
        # The disabled probe must not conjure a line on a platform where
        # the handle count is unavailable too.
        lines = self._emit(monkeypatch, tmp_path, {'os_handles': -1})
        assert not lines


_SCHEDULER = object()


class _EngineeringCtx:
    """Only what the cadence branch reads.

    A MagicMock cannot stand in here: scope construction reads other
    attributes off the context (objectives_loader wants source_path)
    and a MagicMock resolves those to bogus paths.
    """

    def __init__(self, engineering_mode):
        self.engineering_mode = engineering_mode


def _make_session(**kwargs):
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import spec_scope

    defaults = {
        'settings': {},
        'scope': spec_scope(),
        'io_executor': MagicMock(),
        'camera_executor': MagicMock(),
        'metrics_scheduler': _SCHEDULER,
    }
    defaults.update(kwargs)
    return ScopeSession(**defaults)


class TestCadence:
    def test_production_default_is_hourly(self):
        from modules.metrics_logger import DEFAULT_SYSTEM_METRICS_INTERVAL_S

        assert DEFAULT_SYSTEM_METRICS_INTERVAL_S == 3600.0

    def test_no_override_outside_engineering_mode_uses_the_default(self, monkeypatch):
        # ctx pinned explicitly: it is a process global, and a ctx left behind
        # by another test would otherwise decide this assertion.
        session = _make_session()
        monkeypatch.setattr(app_context, 'ctx', None)
        session.start_metrics()
        session.scope.metrics_logger.start.assert_called_once_with(_SCHEDULER)

    def test_engineering_mode_keeps_sixty_seconds(self, monkeypatch):
        # Build the scope before installing the ctx -- construction reads
        # other attributes off it.
        session = _make_session()
        monkeypatch.setattr(app_context, 'ctx', _EngineeringCtx(True))
        session.start_metrics()
        session.scope.metrics_logger.start.assert_called_once_with(
            _SCHEDULER, system_metrics_interval_s=60.0
        )

    def test_engineering_mode_false_uses_the_default(self, monkeypatch):
        session = _make_session()
        monkeypatch.setattr(app_context, 'ctx', _EngineeringCtx(False))
        session.start_metrics()
        session.scope.metrics_logger.start.assert_called_once_with(_SCHEDULER)

    def test_explicit_override_beats_engineering_mode(self, monkeypatch):
        # A bench operator naming an interval is honoured even on a machine
        # engineering mode would otherwise pin to 60 s.
        session = _make_session(settings={'profiling': {'metrics_interval_s': 42}})
        monkeypatch.setattr(app_context, 'ctx', _EngineeringCtx(True))
        session.start_metrics()
        session.scope.metrics_logger.start.assert_called_once_with(
            _SCHEDULER, system_metrics_interval_s=42.0
        )

    def test_unset_context_reads_as_production(self, monkeypatch):
        # Headless / REST / test contexts leave ctx None; that must resolve to
        # the production cadence rather than raising.
        session = _make_session()
        monkeypatch.setattr(app_context, 'ctx', None)
        session.start_metrics()
        session.scope.metrics_logger.start.assert_called_once_with(_SCHEDULER)
