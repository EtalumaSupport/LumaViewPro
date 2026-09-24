# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A sweep in flight names the run that dispatched it.

"An autofocus is running" and "whose autofocus" used to be separate
facts: the thread published only liveness, so a gate that wanted to name
the owner read the run's trigger from somewhere else and could pair a
live sweep with a run that had already been replaced. The thread now
records the dispatching run WITH the Future, and the run refusal that
fires on a live sweep passes that name to whoever was refused -- a
script and a REST caller included, neither of which has a popup to fall
back on.
"""

from __future__ import annotations

import threading

import pytest

from modules.autofocus_thread import AutofocusThread
from modules.exceptions import ProtocolRunRefusedError
from tests.protocol_drives import bare_capture_runner, scr_run_kwargs


class _BlockingAFE:
    """Runner stand-in that parks inside run() until released, so the
    sweep is observably in flight while the gate under test runs."""

    def __init__(self):
        self.entered_run = threading.Event()
        self.release = threading.Event()
        self.run_calls: list[dict] = []

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        self.entered_run.set()
        assert self.release.wait(timeout=10.0), 'test never released the sweep'
        return 5000.0


@pytest.fixture
def afe():
    return _BlockingAFE()


@pytest.fixture
def thread(afe):
    at = AutofocusThread(afe=afe)
    at.start()
    yield at
    afe.release.set()
    at.stop(timeout=2.0)


class TestTheSweepRecordsItsDispatcher:
    def test_the_in_flight_sweep_names_the_run_that_dispatched_it(self, thread, afe):
        thread.run_autofocus(run_trigger_source='protocol', objective_id='4x')
        assert afe.entered_run.wait(timeout=2.0)

        sweep = thread.in_flight_sweep
        assert sweep is not None
        assert sweep.run_trigger_source == 'protocol'

    def test_the_trigger_is_also_forwarded_to_the_runner(self, thread, afe):
        """Captured AND forwarded: the runner gates its own failure
        popups on the same value, so consuming it here would pop a modal
        on every protocol run's autofocus failure."""
        thread.run_autofocus(run_trigger_source='protocol', objective_id='4x')
        assert afe.entered_run.wait(timeout=2.0)

        assert afe.run_calls[0]['run_trigger_source'] == 'protocol'

    def test_the_sweep_is_gone_once_it_finishes(self, thread, afe):
        future = thread.run_autofocus(run_trigger_source='protocol', objective_id='4x')
        assert afe.entered_run.wait(timeout=2.0)

        afe.release.set()
        assert future.result(timeout=5.0) == 5000.0
        assert thread.in_flight_sweep is None


class TestTheRefusalNamesTheRun:
    def test_a_run_refused_for_a_live_sweep_is_told_whose_sweep(self, thread, afe):
        thread.run_autofocus(run_trigger_source='protocol', objective_id='4x')
        assert afe.entered_run.wait(timeout=2.0)

        runner = bare_capture_runner(autofocus_thread=thread)
        with pytest.raises(ProtocolRunRefusedError) as exc_info:
            runner.prepare(**scr_run_kwargs())

        refusal = exc_info.value
        assert refusal.reason == 'autofocus_running'
        assert refusal.holder == 'autofocus'
        assert refusal.holder_trigger == 'protocol'
        assert 'protocol' in refusal.message, 'the refusal must name the run that owns the sweep'

    def test_a_finished_sweep_refuses_nothing(self, thread, afe):
        """The gate reads the in-flight snapshot, so a sweep that has
        resolved stops blocking runs the instant it resolves -- it does
        not wait for the worker to clear its bookkeeping."""
        future = thread.run_autofocus(run_trigger_source='protocol', objective_id='4x')
        assert afe.entered_run.wait(timeout=2.0)
        afe.release.set()
        assert future.result(timeout=5.0) == 5000.0

        runner = bare_capture_runner(autofocus_thread=thread)
        plan = runner.prepare(**scr_run_kwargs())
        assert plan is not None
