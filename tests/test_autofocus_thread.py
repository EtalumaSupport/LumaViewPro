# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for AutofocusThread.

Covers the public API contract:
  - run_autofocus(**kwargs) -> Future that resolves to result or carries
    an exception (including AutofocusAborted on caller-requested abort).
  - abort() unwinds the in-flight run; Future surfaces AutofocusAborted.
  - One AF at a time -- second concurrent run_autofocus() rejects via
    a Future that resolves immediately to RuntimeError.
  - is_running flips True during a run, False after the Future resolves.
  - start() / stop() are idempotent and process-exit-safe (daemon).
"""
from __future__ import annotations

import threading
import time

import pytest

from modules.autofocus_thread import AutofocusThread
from modules.exceptions import AutofocusAborted


class _FakeAFE:
    """Minimal AutofocusExecutor stand-in. The thread calls run(...) per
    request; this stub blocks until the abort_event is set OR returns
    after a configurable delay so tests can drive both completion paths.
    """

    def __init__(self, *, result=42.5, raise_exception=None, run_delay=0.0):
        self._result = result
        self._raise_exception = raise_exception
        self._run_delay = run_delay
        self.run_calls: list[dict] = []
        self.entered_run = threading.Event()
        self.exit_run = threading.Event()

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        abort_event = kwargs.get('abort_event')
        self.entered_run.set()

        if self._raise_exception is not None:
            raise self._raise_exception

        if abort_event is None:
            raise ValueError('AutofocusThread did not pass abort_event')

        # Honor abort if it fires during the run; otherwise sleep the
        # configured delay then return the canned result.
        deadline = time.monotonic() + self._run_delay
        while time.monotonic() < deadline:
            if abort_event.wait(timeout=0.01):
                raise AutofocusAborted('aborted via abort_event')

        self.exit_run.set()
        return self._result


@pytest.fixture
def afe():
    return _FakeAFE()


@pytest.fixture
def at(afe):
    thread = AutofocusThread(afe=afe)
    thread.start()
    yield thread
    thread.stop(timeout=2.0)


class TestLifecycle:
    def test_start_idempotent(self, afe):
        thread = AutofocusThread(afe=afe)
        thread.start()
        first = thread._thread
        thread.start()
        assert thread._thread is first
        thread.stop(timeout=1.0)

    def test_stop_joins(self, afe):
        thread = AutofocusThread(afe=afe)
        thread.start()
        thread.stop(timeout=2.0)
        assert thread._thread is None

    def test_stop_when_never_started_is_safe(self, afe):
        thread = AutofocusThread(afe=afe)
        thread.stop(timeout=1.0)


class TestRunAutofocus:
    def test_success_resolves_future_to_result(self, at, afe):
        future = at.run_autofocus(objective_id='4x')
        result = future.result(timeout=2.0)
        assert result == 42.5
        assert at.is_running is False
        assert at.current_future is None

    def test_run_passes_kwargs_to_afe(self, at, afe):
        at.run_autofocus(objective_id='10x', camera_gain=1.5).result(timeout=2.0)
        assert len(afe.run_calls) == 1
        call = afe.run_calls[0]
        assert call['objective_id'] == '10x'
        assert call['camera_gain'] == 1.5
        assert isinstance(call['abort_event'], threading.Event)

    def test_second_concurrent_call_rejects(self, afe):
        afe._run_delay = 1.0  # keep first run in-flight
        thread = AutofocusThread(afe=afe)
        thread.start()
        try:
            first = thread.run_autofocus(objective_id='4x')
            assert afe.entered_run.wait(timeout=1.0)
            second = thread.run_autofocus(objective_id='4x')
            with pytest.raises(RuntimeError, match='already in progress'):
                second.result(timeout=1.0)
            thread.abort()
            with pytest.raises(AutofocusAborted):
                first.result(timeout=2.0)
        finally:
            thread.stop(timeout=2.0)


class TestAbort:
    def test_abort_during_run_surfaces_AutofocusAborted(self, afe):
        afe._run_delay = 5.0  # ensures the run is in-flight when abort fires
        thread = AutofocusThread(afe=afe)
        thread.start()
        try:
            future = thread.run_autofocus(objective_id='4x')
            assert afe.entered_run.wait(timeout=1.0)
            thread.abort()
            with pytest.raises(AutofocusAborted):
                future.result(timeout=2.0)
        finally:
            thread.stop(timeout=2.0)

    def test_abort_when_idle_is_noop(self, at):
        at.abort()
        assert not at.is_running


class TestExceptionPropagation:
    def test_afe_exception_surfaces_via_future(self, afe):
        afe._raise_exception = RuntimeError('hardware fault')
        thread = AutofocusThread(afe=afe)
        thread.start()
        try:
            future = thread.run_autofocus(objective_id='4x')
            with pytest.raises(RuntimeError, match='hardware fault'):
                future.result(timeout=2.0)
        finally:
            thread.stop(timeout=2.0)
