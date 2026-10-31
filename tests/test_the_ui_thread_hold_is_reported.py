# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A UI thread that stops running is reported once, with its stack, and again when it runs.

A field LS850T's GUI froze for four days and no log line said so: the
executor watchdog ticked on its own timer thread throughout but read only
the lanes' queues. The session's metrics now post a probe to the UI thread
each tick; a probe still waiting at the next tick is a hold, logged once
with the stack the thread is in, and its run logs that the thread ran again.
"""

from __future__ import annotations

import logging
import threading

import pytest

import modules.metrics_logger as metrics_logger_module
from modules.kivy_utils import UiDispatcher, ui_backlog
from modules.scope_session import ScopeSession
from tests.log_capture import records_of


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
