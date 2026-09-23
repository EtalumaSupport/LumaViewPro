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
    deadline = time.monotonic() + COMPLETION_TIMEOUT
    while executor.file_io_executor.is_protocol_queue_active():
        assert time.monotonic() < deadline, "the first run's files never finished writing"
        time.sleep(0.05)
    return run


def _capture_notifications(monkeypatch):
    notified = []
    from modules import notification_center

    for level in ('warning', 'error'):
        monkeypatch.setattr(
            notification_center.notifications,
            level,
            lambda *a, **kw: notified.append(a),
        )
    return notified


class TestTeardownAuthority:
    def test_a_stale_handle_is_refused_and_the_live_run_survives(
        self, executor, scope, tmp_path, monkeypatch
    ):
        """The reproduced defect: a stop naming an old run must not destroy
        the run that is live now."""
        stale = _an_ended_run(executor, tmp_path)
        notified = _capture_notifications(monkeypatch)
        done = threading.Event()
        live = _start_run(executor, tmp_path, done)

        with pytest.raises(ProtocolRunRefusedError) as exc:
            executor.reset(stale)

        assert exc.value.reason == 'run_not_live'
        # Notify-once, like every other refusal: the engine has already told
        # the user, so a caller reconciles its own state without re-notifying.
        assert len(notified) == 1, f'expected one notification, got {notified}'
        assert notified[0][0] == 'Protocol'
        # The live run is named, so a caller can say WHOSE run it is.
        assert exc.value.holder == 'protocol'
        assert exc.value.holder_trigger == OWNER
        # The point of the whole slice: the live run is still running.
        assert executor.run_in_progress(), 'a refused teardown still killed the run'
        assert executor.is_live_run(live)

        executor.force_reset(reason='test cleanup')
        assert done.wait(timeout=COMPLETION_TIMEOUT)

    def test_a_handle_naming_no_run_is_refused_while_a_run_is_live(
        self, executor, scope, tmp_path, monkeypatch
    ):
        """A control that never started anything holds None; its Stop must
        not reach the live run either."""
        notified = _capture_notifications(monkeypatch)
        done = threading.Event()
        live = _start_run(executor, tmp_path, done)

        with pytest.raises(ProtocolRunRefusedError) as exc:
            executor.reset(None)

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
        self, executor, scope, tmp_path, monkeypatch
    ):
        """A stop that arrives after its run ended is told so, and nobody is
        notified: nothing was refused."""
        stale = _an_ended_run(executor, tmp_path)
        notified = _capture_notifications(monkeypatch)

        with pytest.raises(RunAlreadyEndedError) as exc:
            executor.reset(stale)

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


class TestTheRunIsRequired:
    """A stop that does not name a run cannot be written: no default."""

    def test_reset_without_a_run_is_a_type_error(self, executor, scope, tmp_path):
        with pytest.raises(TypeError):
            executor.reset()
