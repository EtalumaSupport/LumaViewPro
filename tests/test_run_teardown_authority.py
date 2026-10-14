# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for run-teardown authority.

A stop is the RUN's -- the operation that run's start() returned --
never the caller's: who is asking is self-declared and protects nothing,
while a stale control naming an OLD run is the hole that once destroyed a
scan the control never started. The stop is the API's, so it holds for a
REST or SDK caller exactly as for the GUI.

The contract:

- stop() on the live run's operation unwinds the run, whoever holds it.
- stop() on the operation of a run that has ended does nothing, while
  another run is live: it raises nothing, notifies nothing AND LEAVES THE
  LIVE RUN RUNNING. This is the case that lost 170 of 192 protocol steps
  in the field: a stale autofocus toggle reached the teardown and
  destroyed a scan it never started.
- stop() with nothing live does nothing either: the run finished before
  the stop arrived (Eric, 2026-10-13: a stop once ended does nothing).
- force_reset(reason) is the named shutdown override: it tears down the
  live run without naming it, and says so at WARNING. It exists so app
  close, which holds no run's operation, still stops whatever is live.
"""

import threading
import time

from modules.run_events import RunEvents
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    COMPLETION_TIMEOUT,
    _make_autogain_settings,
    _make_image_capture_config,
    _make_multi_step_protocol,
    _make_tile_grid_steps,
    executor,
    executors,
    scope,
)

OWNER = 'scan'


def _start_run(executor, tmp_path, done):
    """Start a long-enough run triggered by OWNER, wait until it is live,
    and return the operation start() gave back."""
    protocol = _make_multi_step_protocol(_make_tile_grid_steps(rows=6, cols=8))

    plan = executor.prepare(
        protocol=protocol,
        run_trigger_source=OWNER,
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='teardown_authority',
        image_capture_config=_make_image_capture_config(),
        autogain_settings=_make_autogain_settings(),
        parent_dir=tmp_path / 'output',
        max_scans=1,
        events=RunEvents(run_ended=lambda *ended: done.set()),
    )
    run = executor.start(plan)

    deadline = time.monotonic() + 5.0
    while not executor.run_in_progress():
        assert time.monotonic() < deadline, 'run never reached in-progress'
        time.sleep(0.01)
    return run


def _an_ended_run(executor, tmp_path):
    """The operation of a run that has been stopped and has fully unwound."""
    done = threading.Event()
    run = _start_run(executor, tmp_path / 'ended', done)
    run.stop()
    assert done.wait(timeout=COMPLETION_TIMEOUT), 'the first run never ended'
    assert executor.wait_for_run_idle(COMPLETION_TIMEOUT), 'the first run never went idle'
    assert not run.progress().is_live
    # Its files drain after it ends, and the next start is refused until
    # they have landed -- the designed two-phase completion.
    assert executor.write_batch().wait_complete(COMPLETION_TIMEOUT), (
        "the first run's files never finished writing"
    )
    return run


def _notified_since(centre_posts, start):
    """The warnings and errors the centre posted after ``start``."""
    from modules.notification_center import Severity

    return [n for n in centre_posts[start:] if n.severity in (Severity.WARNING, Severity.ERROR)]


class TestTeardownAuthority:
    def test_a_stale_stop_does_nothing_and_the_live_run_survives(
        self, executor, scope, tmp_path, centre_posts
    ):
        """The reproduced defect: a stop naming an old run must not destroy
        the run that is live now."""
        stale = _an_ended_run(executor, tmp_path)
        start = len(centre_posts)
        done = threading.Event()
        live = _start_run(executor, tmp_path, done)

        assert stale.stop() is None

        notified = _notified_since(centre_posts, start)
        assert notified == [], f'a stale stop notified: {notified}'
        # The point of the whole slice: the live run is still running.
        assert executor.run_in_progress(), 'a stale stop still killed the run'
        assert live.progress().is_live
        assert not live.progress().is_stopping, 'a stale stop recorded a Stop on the live run'

        executor.force_reset(reason='test cleanup')
        assert done.wait(timeout=COMPLETION_TIMEOUT)

    def test_the_live_runs_own_handle_stops_it(self, executor, scope, tmp_path):
        done = threading.Event()
        run = _start_run(executor, tmp_path, done)

        run.stop()

        assert done.wait(timeout=COMPLETION_TIMEOUT), 'the stop did not unwind the run'
        assert (run.result(COMPLETION_TIMEOUT).status, run.result(0).reason) == (
            'aborted',
            'stopped',
        )

    def test_a_stop_with_no_run_live_does_nothing(self, executor, scope, tmp_path, centre_posts):
        """A stop that arrives after its run ended raises nothing, and nobody
        is notified: the run finished before the stop arrived."""
        stale = _an_ended_run(executor, tmp_path)
        start = len(centre_posts)

        assert stale.stop() is None

        notified = _notified_since(centre_posts, start)
        assert notified == [], f'a stop after the run ended notified: {notified}'

    def test_force_reset_overrides_ownership(self, executor, scope, tmp_path):
        """App close holds no run's handle and still stops the live run."""
        done = threading.Event()
        _start_run(executor, tmp_path, done)

        executor.force_reset(reason='app shutdown')

        assert done.wait(timeout=COMPLETION_TIMEOUT), 'force_reset did not unwind the run'

    def test_force_reset_unwinds_a_run_whose_protocol_thread_died(
        self, executor, scope, tmp_path, monkeypatch
    ):
        """With no run loop left to unwind the run, force_reset runs the
        cleanup itself, for the live run: app close is the last chance to
        release the scope, and a cleanup that did not know which run it
        ended would leave it held for good."""
        monkeypatch.setattr(executor, '_run_loop_under_claim', lambda run: None)
        done = threading.Event()
        _start_run(executor, tmp_path, done)
        deadline = time.monotonic() + 5.0
        while executor.protocol_thread.is_running:
            assert time.monotonic() < deadline, 'the run loop never ended'
            time.sleep(0.01)
        assert executor.run_in_progress(), 'the run ended without its cleanup'

        executor.force_reset(reason='app shutdown')

        assert executor.wait_for_run_idle(COMPLETION_TIMEOUT), 'force_reset left the run live'
        assert not executor.run_in_progress()
        assert done.is_set(), 'the run ended without run_ended'


class TestAStopControlCanSayTheRunIsStopping:
    """progress().is_stopping: live, and a Stop of it accepted -- what a stop control shows.

    A stopped run stays live through its teardown, so liveness alone makes a
    stop control say "running" until the LEDs, camera and lanes are put
    back. The run's recorded ending is the answer instead.
    """

    @staticmethod
    def _stopping(live, ending=None):
        """``is_stopping`` from the engine's progress read of a run that is *live* or not."""
        import types

        from modules.run_outcome import EndingLatch
        from modules.sequenced_capture_runner import SequencedCaptureRunner

        # A stand-in by design: the engine's progress read over its own
        # latch, with liveness and the counters given, so each ending's
        # answer is read without driving a run to that ending.
        runner = object.__new__(SequencedCaptureRunner)
        runner._run_lock = threading.RLock()
        runner._ending = EndingLatch()
        if ending is not None:
            runner._ending.set_if_unset(ending)
        runner._is_live_run_locked = lambda run: live
        runner._run_handle = None
        for read in ('_step_number', '_num_steps', '_remaining_scans', '_protocol_interval'):
            setattr(runner, read, lambda: None)
        return runner._progress_of(types.SimpleNamespace(_run_dir=None)).is_stopping

    def test_a_live_run_nobody_stopped_is_not_stopping(self):
        assert self._stopping(True) is False

    def test_a_live_run_with_an_accepted_stop_is_stopping(self):
        from modules.run_outcome import RunEnding

        assert self._stopping(True, RunEnding('aborted', 'stopped', 'Stopped')) is True

    def test_a_run_the_instrument_ended_is_not_stopping(self):
        from modules.run_outcome import RunEnding

        assert self._stopping(True, RunEnding('failed', 'position_lost', 'M')) is False

    def test_a_run_that_is_not_live_is_not_stopping(self):
        from modules.run_outcome import RunEnding

        assert self._stopping(False, RunEnding('aborted', 'stopped', 'Stopped')) is False

    def test_a_real_stop_reads_stopping_until_the_run_has_ended(self, executor, tmp_path):
        done = threading.Event()
        run = _start_run(executor, tmp_path, done)
        assert run.progress().is_stopping is False
        run.stop()
        progress = run.progress()
        assert progress.is_stopping or not progress.is_live
        assert done.wait(timeout=COMPLETION_TIMEOUT)
        assert executor.wait_for_run_idle(COMPLETION_TIMEOUT)
        assert run.progress().is_stopping is False


class TestTheRunsEndIsAnnounced:
    """After a run's cleanup puts the runner back to IDLE, its listener is told.

    The claim releases just before IDLE, so a listener woken by the claim
    alone can read the run as still live and draw it running with nothing
    to redraw it after. The runner announces the IDLE edge itself, on every
    exit, outside its cleanup lock.
    """

    @staticmethod
    def _listen(executor):
        heard = []
        executor._on_run_idle = lambda: heard.append(executor.run_in_progress())
        return heard

    @staticmethod
    def _heard_the_end(heard):
        deadline = time.monotonic() + COMPLETION_TIMEOUT
        while not heard and time.monotonic() < deadline:
            time.sleep(0.01)
        return bool(heard) and heard[-1] is False

    def test_a_run_that_ends_on_its_own_is_announced_after_it_ends(self, executor, tmp_path):
        from tests.test_run_refusal_contract import _make_single_step_protocol, _run_to_completion

        heard = self._listen(executor)
        _run_to_completion(executor, _make_single_step_protocol(), tmp_path)
        assert self._heard_the_end(heard), f'the end was not announced after IDLE: {heard}'

    def test_a_stopped_run_is_announced_after_it_ends(self, executor, tmp_path):
        heard = self._listen(executor)
        done = threading.Event()
        run = _start_run(executor, tmp_path, done)
        run.stop()
        assert done.wait(timeout=COMPLETION_TIMEOUT)
        assert self._heard_the_end(heard), f'the end was not announced after IDLE: {heard}'

    def test_a_run_that_failed_at_start_is_announced_after_it_ends(
        self, executor, tmp_path, monkeypatch
    ):
        heard = self._listen(executor)

        def _boom():
            raise OSError('save folder vanished')

        monkeypatch.setattr(executor, '_setup_run_dir', _boom)
        _start_run_and_let_it_fail(executor, tmp_path)
        assert self._heard_the_end(heard), f'the failed start was not announced: {heard}'


class TestHeldByOther:
    """A run control greys while the scope is held by anything but its run.

    Its own run leaves it live as that run's Stop; the run's operation, not
    the trigger, says which run is its own.
    """

    def test_the_live_run_is_not_other_to_itself(self, executor, tmp_path):
        done = threading.Event()
        live = _start_run(executor, tmp_path, done)
        try:
            assert executor.held_by_other(live) is False, (
                'a run control is greyed by its own run, and cannot be its Stop'
            )
            assert executor.held_by_other(None) is True
        finally:
            executor.force_reset(reason='test cleanup')
            assert executor.wait_for_run_idle(10.0)

    def test_an_ended_run_is_other_to_the_live_one(self, executor, tmp_path):
        stale = _an_ended_run(executor, tmp_path)
        done = threading.Event()
        _start_run(executor, tmp_path, done)
        try:
            assert executor.held_by_other(stale) is True
        finally:
            executor.force_reset(reason='test cleanup')
            assert executor.wait_for_run_idle(10.0)


def _start_run_and_let_it_fail(executor, tmp_path):
    """start() a run whose setup fails: the failed-at-start unwind runs inline."""
    protocol = _make_multi_step_protocol(_make_tile_grid_steps(rows=1, cols=1))
    plan = executor.prepare(
        protocol=protocol,
        run_trigger_source=OWNER,
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='failed_start',
        image_capture_config=_make_image_capture_config(),
        autogain_settings=_make_autogain_settings(),
        parent_dir=tmp_path / 'output',
        max_scans=1,
    )
    executor.start(plan)
    assert executor.wait_for_run_idle(COMPLETION_TIMEOUT)
