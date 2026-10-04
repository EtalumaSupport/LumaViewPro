# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for run-teardown authority.

A stop names the RUN it means to stop -- the handle that run's start()
returned -- never the caller: who is asking is self-declared and
protects nothing, while a stale control naming an OLD run is the hole
that once destroyed a scan the control never started. The refusal is
the API's, so it reaches a REST or SDK caller exactly as it reaches the
GUI -- a teardown that could only be refused by a widget would be no
refusal at all.

The contract:

- reset(run) with the live run's handle unwinds the run, whoever holds
  the handle.
- reset(run) with a handle naming a run that has ended, while another
  run is live, raises ProtocolRunRefusedError with reason
  'run_not_live' AND LEAVES THE LIVE RUN RUNNING. This is the case that
  lost 170 of 192 protocol steps in the field: a stale autofocus toggle
  reached reset() and destroyed a scan it never started.
- reset(run) with nothing live raises RunAlreadyEndedError -- not a
  refusal, and never notified: the run finished before the stop arrived.
- force_reset(reason) is the named shutdown override: it tears down the
  live run without naming it, and says so at WARNING. It exists so app
  close, which holds no run's handle, still stops whatever is live.
"""

import threading
import time

import pytest

from modules.exceptions import ProtocolRunRefusedError, RunAlreadyEndedError
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from tests.protocol_drives import autofocus_snapshot
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
    and return the handle start() gave back."""
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
        callbacks={
            'run_complete': lambda **kw: done.set(),
            'go_to_step': lambda **kw: None,
            'move_position': lambda axis: None,
        },
        leds_state_at_end='off',
        autofocus_snapshot=autofocus_snapshot(),
    )
    run = executor.start(plan)

    deadline = time.monotonic() + 5.0
    while not executor.run_in_progress():
        assert time.monotonic() < deadline, 'run never reached in-progress'
        time.sleep(0.01)
    return run


def _an_ended_run(executor, tmp_path):
    """The handle of a run that has been stopped and has fully unwound."""
    done = threading.Event()
    run = _start_run(executor, tmp_path / 'ended', done)
    executor.reset(run)
    assert done.wait(timeout=COMPLETION_TIMEOUT), 'the first run never ended'
    assert executor.wait_for_run_idle(COMPLETION_TIMEOUT), 'the first run never went idle'
    assert not executor.is_live_run(run)
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
    def test_a_stale_handle_is_refused_and_the_live_run_survives(
        self, executor, scope, tmp_path, centre_posts
    ):
        """The reproduced defect: a stop naming an old run must not destroy
        the run that is live now."""
        stale = _an_ended_run(executor, tmp_path)
        start = len(centre_posts)
        done = threading.Event()
        live = _start_run(executor, tmp_path, done)

        with pytest.raises(ProtocolRunRefusedError) as exc:
            executor.reset(stale)

        notified = _notified_since(centre_posts, start)
        assert exc.value.reason == 'run_not_live'
        # Notify-once, like every other refusal: the engine has already told
        # the user, so a caller reconciles its own state without re-notifying.
        assert len(notified) == 1, f'expected one notification, got {notified}'
        assert notified[0].category == 'Protocol'
        # The live run is named, so a caller can say WHOSE run it is.
        assert exc.value.holder == 'protocol'
        assert exc.value.holder_trigger == OWNER
        # The point of the whole slice: the live run is still running.
        assert executor.run_in_progress(), 'a refused teardown still killed the run'
        assert executor.is_live_run(live)

        executor.force_reset(reason='test cleanup')
        assert done.wait(timeout=COMPLETION_TIMEOUT)

    def test_a_handle_naming_no_run_is_refused_while_a_run_is_live(
        self, executor, scope, tmp_path, centre_posts
    ):
        """A control that never started anything holds None; its Stop must
        not reach the live run either."""
        start = len(centre_posts)
        done = threading.Event()
        live = _start_run(executor, tmp_path, done)

        with pytest.raises(ProtocolRunRefusedError) as exc:
            executor.reset(None)

        notified = _notified_since(centre_posts, start)
        assert exc.value.reason == 'run_not_live'
        assert len(notified) == 1, f'expected one notification, got {notified}'
        assert executor.is_live_run(live), 'a refused teardown still killed the run'

        executor.force_reset(reason='test cleanup')
        assert done.wait(timeout=COMPLETION_TIMEOUT)

    def test_the_live_runs_own_handle_stops_it(self, executor, scope, tmp_path):
        done = threading.Event()
        run = _start_run(executor, tmp_path, done)

        executor.reset(run)

        assert done.wait(timeout=COMPLETION_TIMEOUT), 'reset did not unwind the run'

    def test_anyone_holding_the_live_handle_may_stop_it(self, executor, scope, tmp_path):
        """The handle is the credential, not who started the run: a caller
        that fetched the live run's handle stops it like its starter would."""
        done = threading.Event()
        started = _start_run(executor, tmp_path, done)
        fetched = executor.run_outcome()
        assert fetched is started

        executor.reset(fetched)

        assert done.wait(timeout=COMPLETION_TIMEOUT), 'reset did not unwind the run'

    def test_teardown_with_no_run_raises_run_already_ended(
        self, executor, scope, tmp_path, centre_posts
    ):
        """A stop that arrives after its run ended is told so, and nobody is
        notified: nothing was refused."""
        stale = _an_ended_run(executor, tmp_path)
        start = len(centre_posts)

        with pytest.raises(RunAlreadyEndedError) as exc:
            executor.reset(stale)

        notified = _notified_since(centre_posts, start)
        assert not isinstance(exc.value, ProtocolRunRefusedError)
        assert notified == [], f'a stop after the run ended notified: {notified}'

    def test_teardown_before_any_run_raises_run_already_ended(self, executor, scope):
        with pytest.raises(RunAlreadyEndedError):
            executor.reset(None)

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
        assert done.is_set(), 'the run ended without run_complete'


class TestTheRunIsRequired:
    """A stop that does not name a run cannot be written: no default."""

    def test_reset_without_a_run_is_a_type_error(self, executor, scope, tmp_path):
        with pytest.raises(TypeError):
            executor.reset()


class TestAStopControlCanSayTheRunIsStopping:
    """is_stopping(run): live, and a Stop of it accepted -- what a stop control shows.

    A stopped run stays live through its teardown, so liveness alone makes a
    stop control say "running" until the LEDs, camera and lanes are put
    back. The run's recorded ending is the answer instead.
    """

    @staticmethod
    def _runner(live, monkeypatch):
        from modules.run_outcome import EndingLatch
        from modules.sequenced_capture_runner import SequencedCaptureRunner

        runner = object.__new__(SequencedCaptureRunner)
        runner._run_lock = threading.RLock()
        runner._ending = EndingLatch()
        monkeypatch.setattr(runner, '_is_live_run_locked', lambda run: live)
        return runner

    def test_a_live_run_nobody_stopped_is_not_stopping(self, monkeypatch):
        assert self._runner(True, monkeypatch).is_stopping(object()) is False

    def test_a_live_run_with_an_accepted_stop_is_stopping(self, monkeypatch):
        from modules.run_outcome import RunEnding

        runner = self._runner(True, monkeypatch)
        runner._ending.set_if_unset(RunEnding('aborted', 'stopped', 'Protocol Stopped', 'Stopped'))
        assert runner.is_stopping(object()) is True

    def test_a_run_the_instrument_ended_is_not_stopping(self, monkeypatch):
        from modules.run_outcome import RunEnding

        runner = self._runner(True, monkeypatch)
        runner._ending.set_if_unset(RunEnding('failed', 'motion_timeout', 'T', 'M'))
        assert runner.is_stopping(object()) is False

    def test_a_run_that_is_not_live_is_not_stopping(self, monkeypatch):
        from modules.run_outcome import RunEnding

        runner = self._runner(False, monkeypatch)
        runner._ending.set_if_unset(RunEnding('aborted', 'stopped', 'Protocol Stopped', 'Stopped'))
        assert runner.is_stopping(object()) is False

    def test_a_real_stop_reads_stopping_until_the_run_has_ended(self, executor, tmp_path):
        done = threading.Event()
        run = _start_run(executor, tmp_path, done)
        assert executor.is_stopping(run) is False
        executor.reset(run)
        assert executor.is_stopping(run) or not executor.is_live_run(run)
        assert done.wait(timeout=COMPLETION_TIMEOUT)
        assert executor.wait_for_run_idle(COMPLETION_TIMEOUT)
        assert executor.is_stopping(run) is False


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
        executor.reset(run)
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

    Its own run leaves it live as that run's Stop; the handle, not the
    trigger, says which run is its own.
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
        callbacks={'go_to_step': lambda **kw: None, 'move_position': lambda axis: None},
        leds_state_at_end='off',
        autofocus_snapshot=autofocus_snapshot(),
    )
    executor.start(plan)
    assert executor.wait_for_run_idle(COMPLETION_TIMEOUT)
