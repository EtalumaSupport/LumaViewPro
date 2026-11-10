# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A UI thread that stops running is reported once, with its stack, and again when it runs.

A field LS850T's GUI froze for four days and no log line said so: the
executor watchdog ticked on its own timer thread throughout but read only
the lanes' queues. The session's metrics now post a probe to the UI thread
each tick; a probe still waiting at the next tick is a hold, logged once
with the stack the thread is in, and its run logs that the thread ran again.
A hold shorter than that tick is said by the GUI's Clock, once, when the
thread runs again.
"""

from __future__ import annotations

import datetime
import logging
import threading
import types

import pytest

import _lumascope.modules.metrics_logger as metrics_logger_module
from _lumascope.modules.ui_queue import UiDispatcher, ui_backlog
from _lumascope.modules.scope_session import ScopeSession
from tests.log_capture import messages, records_of


def _said(caplog, level) -> list[str]:
    """The watch's lines at *level*: what the metrics module logged."""
    return [r.getMessage() for r in records_of(caplog, metrics_logger_module) if r.levelno == level]


def parked_in_a_named_function(release):
    release.wait(30)


@pytest.fixture
def held_ui_thread():
    """A UI thread parked in a named function, whose posts queue until the test runs them."""
    release = threading.Event()
    thread = threading.Thread(target=parked_in_a_named_function, args=(release,), daemon=True)
    thread.start()
    queue = []
    ScopeSession.set_ui_dispatcher(
        UiDispatcher(schedule=lambda func, timeout: queue.append(func), thread=thread)
    )
    yield queue
    ScopeSession.set_ui_dispatcher(None)
    release.set()
    thread.join(5)


def _run(queue):
    pending, queue[:] = list(queue), []
    for func in pending:
        func(0)


def test_a_held_ui_thread_is_reported_once_with_its_stack_then_its_return(
    sim_session, held_ui_thread, caplog
):
    watch = sim_session.metrics_logger
    watch.tick_ui_watch()
    assert _said(caplog, logging.WARNING) == []
    watch.tick_ui_watch()
    warnings = _said(caplog, logging.WARNING)
    assert len(warnings) == 1, warnings
    assert 'UI thread has run nothing' in warnings[0]
    assert 'parked_in_a_named_function' in warnings[0]

    watch.tick_ui_watch()
    assert len(_said(caplog, logging.WARNING)) == 1
    assert ui_backlog() == 1, 'a second probe was posted while one was outstanding'

    _run(held_ui_thread)
    ran_again = _said(caplog, logging.INFO)
    assert len(ran_again) == 1 and 'UI thread ran again' in ran_again[0], ran_again


def test_a_thread_that_keeps_up_is_never_reported(sim_session, held_ui_thread, caplog):
    watch = sim_session.metrics_logger
    for _ in range(3):
        watch.tick_ui_watch()
        _run(held_ui_thread)
    assert _said(caplog, logging.WARNING) == [] and _said(caplog, logging.INFO) == []


def test_a_host_with_no_ui_thread_posts_no_probe(sim_session, caplog):
    queue = []
    ScopeSession.set_ui_dispatcher(
        UiDispatcher(schedule=lambda func, timeout: queue.append(func), thread=None)
    )
    try:
        for _ in range(3):
            sim_session.metrics_logger.tick_ui_watch()
    finally:
        ScopeSession.set_ui_dispatcher(None)
    assert queue == []
    assert _said(caplog, logging.WARNING) == []


def test_a_probe_that_runs_after_stop_logs_nothing(sim_session, held_ui_thread, caplog):
    watch = sim_session.metrics_logger
    watch.tick_ui_watch()
    watch.tick_ui_watch()
    watch.stop()
    _run(held_ui_thread)
    assert _said(caplog, logging.INFO) == []


def test_a_failing_queue_snapshot_reaches_the_scheduler(sim_session, monkeypatch):
    # a stand-in by design: no lane's snapshot can be made to fail on the
    # simulator; the tick must let the failure reach the scheduler's error
    # hook, which reports it, rather than discard it.
    def broken():
        raise RuntimeError('snapshot broke')

    monkeypatch.setattr(sim_session.executor_bundle, 'snapshot', broken)
    with pytest.raises(RuntimeError, match='snapshot broke'):
        sim_session.metrics_logger.tick_executor_watchdog()


@pytest.fixture
def frame_started_before_the_line(monkeypatch):
    """The reporter's module, under a Clock whose frame started 0.3 s before the line is written."""
    import ui.ui_helpers as ui_helpers

    # a stand-in by design: Kivy never runs in the test process
    # (conftest install_mock_deps); the reporter reads only the Clock's now
    # and its frame's start, and this one puts them 0.3 s apart.
    monkeypatch.setattr(
        ui_helpers, 'Clock', types.SimpleNamespace(time=lambda: 100.0, get_time=lambda: 99.7)
    )
    return ui_helpers


def test_a_hold_shorter_than_the_watch_sees_is_said_by_the_clock_once_it_runs(
    frame_started_before_the_line, caplog
):
    """The Clock's gap past a second is said once; a gap within its tick says nothing.

    The probe above needs a hold of a tick or more (a minute) to see one; a
    field bundle's 9-10 s protocol loads froze the GUI and no line said so.
    """
    ui_helpers = frame_started_before_the_line
    caplog.set_level(logging.WARNING)
    ui_helpers.report_ui_thread_hold(ui_helpers.UI_HOLD_TICK_S)
    assert messages(records_of(caplog, ui_helpers)) == []

    ui_helpers.report_ui_thread_hold(9.04)

    said = messages(records_of(caplog, ui_helpers))
    assert len(said) == 1 and said[0].startswith('[UI        ] The UI thread was held 9.0 s, '), (
        said
    )


def test_a_hold_names_its_window_ending_when_its_frame_started_not_at_the_line(
    frame_started_before_the_line, caplog
):
    """A hold ends at its frame's start, which can be well before the line is written.

    A field bundle's two held windows, read back from their lines' stamps,
    seemed to overlap: the frame between them ran 1.65 s of other callbacks
    before the reporter wrote its line. The line states its window, so a
    reader never derives it from the stamp.
    """
    ui_helpers = frame_started_before_the_line
    caplog.set_level(logging.WARNING)

    ui_helpers.report_ui_thread_hold(9.04)

    (record,) = records_of(caplog, ui_helpers)
    began, ended = (
        datetime.datetime.combine(
            datetime.date.fromtimestamp(record.created), datetime.time.fromisoformat(clock_time)
        )
        for clock_time in record.getMessage().rsplit(', ', 1)[1].split(' - ')
    )
    stamped = datetime.datetime.fromtimestamp(record.created)
    assert 0.29 <= (stamped - ended).total_seconds() < 0.35, (ended, stamped)
    assert abs((ended - began).total_seconds() - 9.04) < 0.002, (began, ended)


class _Monotonic:
    """A monotonic clock the test moves; a stop is a jump no thread saw."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


@pytest.fixture
def monotonic(monkeypatch):
    """The metrics module's and the GUI reporter's clock, one the test moves."""
    import ui.ui_helpers as ui_helpers

    # a stand-in by design: no test can stop its own process for a stretch
    # and go on; the heartbeat and the reports read only time.monotonic,
    # and this one jumps as a standby does.
    clock = _Monotonic()
    fake_time = types.SimpleNamespace(monotonic=clock.monotonic)
    monkeypatch.setattr(metrics_logger_module, 'time', fake_time)
    monkeypatch.setattr(ui_helpers, 'time', fake_time)
    return clock


def _beat(watch, clock, until):
    """Run the heartbeat at its cadence up to *until*, as its timer thread does."""
    while clock.now < until:
        watch.tick_heartbeat()
        clock.now += metrics_logger_module.HEARTBEAT_INTERVAL_S
    clock.now = until
    watch.tick_heartbeat()


def test_a_stretch_in_which_no_thread_ran_is_said_once_with_its_length(
    sim_session, monotonic, caplog
):
    """A gap between two heartbeats past STOP_GAP_S is said once, naming both causes."""
    watch = sim_session.metrics_logger
    _beat(watch, monotonic, 1001.0)
    assert _said(caplog, logging.WARNING) == []

    monotonic.now += 3448.7
    _beat(watch, monotonic, monotonic.now + 1.0)

    warnings = _said(caplog, logging.WARNING)
    assert len(warnings) == 1, warnings
    assert warnings[0].startswith('[Watchdog  ] LVP ran no thread for 3448.7 s, '), warnings
    assert 'standby' in warnings[0] and 'held the interpreter' in warnings[0]
    assert watch.stopped_within(1000.0, monotonic.now) == pytest.approx(3448.7)


def test_a_probe_waiting_through_a_stop_is_not_a_hold_and_one_after_it_is(
    sim_session, held_ui_thread, monotonic, caplog
):
    """The watch blames the UI thread for no stretch the process did not run.

    Its timer can run before the heartbeat's after a resume, so the
    heartbeat's silence counts as the stop not yet said; a hold after the
    resume is still reported, from the probe posted then.
    """
    watch = sim_session.metrics_logger
    _beat(watch, monotonic, 1001.0)
    watch.tick_ui_watch()

    monotonic.now += 3448.7
    watch.tick_ui_watch()
    assert not any('UI thread' in line for line in _said(caplog, logging.WARNING))

    _beat(watch, monotonic, monotonic.now + 60.0)
    watch.tick_ui_watch()
    holds = [line for line in _said(caplog, logging.WARNING) if 'UI thread has run nothing' in line]
    assert len(holds) == 1 and 'for at least 60 s' in holds[0], holds


def test_the_gui_gives_the_ui_thread_only_the_part_of_its_window_the_process_ran(
    sim_session, monotonic, frame_started_before_the_line, monkeypatch, caplog
):
    """A window the process did not run says nothing; a hold beside a stop is the hold alone."""
    import _lumascope.modules.app_context as app_context

    ui_helpers = frame_started_before_the_line
    monkeypatch.setattr(app_context, 'ctx', app_context.AppContext.over_session(sim_session))
    caplog.set_level(logging.WARNING)
    watch = sim_session.metrics_logger
    _beat(watch, monotonic, 1001.0)
    monotonic.now += 3448.7
    _beat(watch, monotonic, monotonic.now + 0.3)

    ui_helpers.report_ui_thread_hold(3448.7 + 0.3)
    assert messages(records_of(caplog, ui_helpers)) == []

    _beat(watch, monotonic, monotonic.now + 5.0)
    ui_helpers.report_ui_thread_hold(3448.7 + 5.0 + 0.3)
    said = messages(records_of(caplog, ui_helpers))
    assert len(said) == 1, said
    assert said[0].startswith('[UI        ] The UI thread was held 5.'), said
    assert said[0].endswith('; for 3448.7 s of it no LVP thread ran'), said
