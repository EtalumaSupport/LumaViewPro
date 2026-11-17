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
-- measured, ran-and-failed, and never-asked-for -- and the third reads
`off`, never as the second's `unavailable`: a deliberate configuration
printed as a malfunction would mislead every log a reader opens.

The gate is read with `is True` rather than truthiness because nothing
validates the settings files: a hand-edited `"false"` is a truthy
string, and reading it loosely would switch the probe back on while its
operator believed it off.
"""

import logging
import os
import re

from types import SimpleNamespace

import psutil
import pytest

from _lumascope.modules import common_utils, config_helpers
from _lumascope.modules.scheduler import ThreadingTimerScheduler
from tests.log_capture import records_of
from tests.scope_fakes import construct_session, real_executor_bundle
from tests.settings_fixtures import complete_settings


class _CountingProcess(psutil.Process):
    """This process, recording what each open_files() call answered."""

    def __init__(self):
        super().__init__(os.getpid())
        self.open_files_answers = []

    def open_files(self):
        answer = super().open_files()
        self.open_files_answers.append(answer)
        return answer


@pytest.fixture
def counting_proc(monkeypatch):
    """The metrics' process handle, one whose open_files() calls are observable."""
    proc = _CountingProcess()
    monkeypatch.setattr(common_utils, 'self_process', lambda: proc)
    return proc


def _metrics_lines(caplog, call):
    """The lines config_helpers writes to the metrics log while ``call`` runs."""
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=config_helpers.metrics_logger.name):
        call()
    return [
        r.getMessage()
        for r in records_of(caplog, config_helpers)
        if r.name == config_helpers.metrics_logger.name
    ]


def _tick_lines(caplog, scope, tmp_path, **profiling):
    settings = complete_settings(live_folder=str(tmp_path), profiling=profiling)
    return _metrics_lines(
        caplog,
        lambda: config_helpers.log_system_metrics(
            settings, scope=scope, executors=real_executor_bundle()
        ),
    )


def _handle_line(lines):
    (line,) = [m for m in lines if m.startswith('[HANDLE METRICS]')]
    return line


class TestOpenFilesGate:
    def test_gate_off_does_not_call_open_files_and_says_off(
        self, counting_proc, caplog, sim_scope, tmp_path
    ):
        line = _handle_line(_tick_lines(caplog, sim_scope, tmp_path))
        assert counting_proc.open_files_answers == []
        # "off", never a failure: a chosen configuration is not a broken probe.
        assert re.fullmatch(r'\[HANDLE METRICS\] handles=\d+ \| open_files=off', line), line

    def test_gate_on_measures(self, counting_proc, caplog, sim_scope, tmp_path):
        line = _handle_line(_tick_lines(caplog, sim_scope, tmp_path, open_files_enabled=True))
        (answer,) = counting_proc.open_files_answers
        assert line.endswith(f'| open_files={len(answer)}'), line

    @pytest.mark.parametrize('configured', ['false', 'true', 0, 1, None, '', 'yes', [], 'False'])
    def test_only_a_real_true_enables_the_probe(
        self, counting_proc, caplog, sim_scope, configured, tmp_path
    ):
        # 'true' and 'yes' are the dangerous half: truthy strings that a
        # loose read would honour.
        _tick_lines(caplog, sim_scope, tmp_path, open_files_enabled=configured)
        assert counting_proc.open_files_answers == []


class TestGcMetricsLine:
    def test_the_line_states_the_heap_block_count(self, caplog, sim_scope, tmp_path):
        lines = _tick_lines(caplog, sim_scope, tmp_path)
        (line,) = [m for m in lines if '[GC METRICS]' in m]
        assert re.search(r'\[GC METRICS\] blocks=[1-9]\d* \|', line), line


class TestUptimeOnTheSystemLine:
    def test_the_line_states_the_hosts_uptime(self, caplog, sim_scope, tmp_path):
        lines = _tick_lines(caplog, sim_scope, tmp_path)
        (line,) = [m for m in lines if '[SYSTEM METRICS]' in m]
        assert re.search(r'\| uptime_hr=\d+\.\d$', line), line


def test_an_unreadable_boot_time_is_raised_from_the_environment_line(monkeypatch):
    refusal = psutil.Error()

    def unreadable():
        raise refusal

    # a stand-in by design: a host whose boot time cannot be read is not one
    # this host can be made into.
    monkeypatch.setattr(psutil, 'boot_time', unreadable)
    with pytest.raises(psutil.Error) as raised:
        config_helpers.log_environment_once()
    assert raised.value is refusal


class _RecordingScheduler(ThreadingTimerScheduler):
    """The session's scheduler, recording each interval it is asked to keep."""

    def __init__(self):
        super().__init__()
        self.intervals = []

    def schedule_interval(self, callback, interval_s):
        self.intervals.append((callback, interval_s))
        return super().schedule_interval(callback, interval_s)


@pytest.fixture
def system_tick_interval(sim_scope_a_session_composes):
    """Start a session's metrics and answer the system tick's interval."""

    scheduler = _RecordingScheduler()

    def started(metrics_interval_s=None, **kwargs):
        session = construct_session(
            settings=complete_settings(profiling={'metrics_interval_s': metrics_interval_s}),
            scope=sim_scope_a_session_composes,
            executor_bundle=real_executor_bundle(),
            scheduler=scheduler,
            **kwargs,
        )
        session.enable_metrics()
        tick = session.metrics_logger.tick_system_metrics
        session.disable_metrics()
        (interval,) = [i for callback, i in scheduler.intervals if callback == tick]
        return interval

    yield started
    scheduler.shutdown()


class TestCadence:
    def test_production_default_is_hourly(self, system_tick_interval):
        assert system_tick_interval() == 3600.0

    def test_engineering_mode_keeps_sixty_seconds(self, system_tick_interval):
        # The session's own flag, the one a plugin turns on when it loads.
        assert system_tick_interval(engineering_mode=True) == 60.0

    def test_explicit_override_beats_engineering_mode(self, system_tick_interval):
        # A bench operator naming an interval is honoured even on a machine
        # engineering mode would otherwise pin to 60 s.
        assert system_tick_interval(42, engineering_mode=True) == 42.0


class TestEveryHostLogsTheExecutors:
    """The queue and futures lines come from the session's own executors, so a
    REST or headless host logs them too; only the display's rates need a
    display, which the GUI hands down."""

    def _lines(self, caplog, sim_scope, tmp_path, display=None):
        return _metrics_lines(
            caplog,
            lambda: config_helpers.log_system_metrics(
                complete_settings(live_folder=str(tmp_path)),
                scope=sim_scope,
                executors=real_executor_bundle(),
                display=display,
            ),
        )

    def test_a_host_with_no_display_logs_queues_and_futures(self, caplog, sim_scope, tmp_path):
        lines = self._lines(caplog, sim_scope, tmp_path)
        queues = [m for m in lines if m.startswith('[QUEUE METRICS]')]
        futures = [m for m in lines if m.startswith('[FUTURES METRICS]')]
        assert queues and 'IO=0' in queues[0]
        # The queueless threads are stated in words, never as a depth.
        assert 'PROTOCOL=idle' in queues[0] and 'SCOPEDISPLAY=off' in queues[0], queues[0]
        assert not re.search(r'=-\d', queues[0]), queues[0]
        # Named fields, not A/P/L, which read as Active/Pending/Leaked.
        assert futures and re.search(r'IO: alloc=\d+ pop=\d+ live=\d+', futures[0])
        assert not any('display_fps' in m or '[FRAME INTERVAL]' in m for m in lines)

    def test_a_display_handed_down_is_logged(self, caplog, sim_scope, tmp_path):
        # a stand-in by design: the display is the GUI's widget, which the
        # test process cannot build; the subject is the line it feeds.
        display = SimpleNamespace(
            display_fps=lambda: 24.0,
            frame_interval_percentiles_ms=lambda: {
                'p50': 40.0,
                'p95': 45.0,
                'p99': 50.0,
                'max': 60.0,
                'n': 24,
            },
        )
        lines = self._lines(caplog, sim_scope, tmp_path, display)
        assert any('display_fps=24.0' in m for m in lines)
        assert any(m.startswith('[FRAME INTERVAL] p50=40.0 ms') for m in lines)
