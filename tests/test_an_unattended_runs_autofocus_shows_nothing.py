# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An unattended run's autofocus shows nothing, by the run's mute alone.

The run opens the notification centre's run scope at start and closes it
first thing in cleanup; an unattended scope is the one owner of "an
unattended run shows no popups". Autofocus once carried its own skip for
the 'protocol' trigger on top of it. These pin that the mute covers both
ways a sweep in a run ends badly, for every unattended autofocus -- a scan
of every step, from its button or the API, and a one-position autofocus a
diagnostic runs under its own claim:

- the sweep fails while the run is live: logged, the run captures at its
  fallback Z, nothing shown;
- the run is stopped while the sweep is live: cleanup closes the scope and
  then aborts the sweep, and an aborted sweep raises only AutofocusAborted,
  which notifies nothing.

Real simulated scope, real executors, a real AutofocusRunner on a real
AutofocusThread, listening on the real notification centre the runners
report to.

The one-position autofocus that takes the scope itself is attended, whoever
started it: its sweep failure is shown to a REST caller as to the button.
"""

from __future__ import annotations

import threading
import time

import pytest

from modules.exceptions import AutofocusAborted
from modules.notification_center import Severity, notifications
from modules.protocol_state_machine import SequencedCaptureRunMode
from tests.test_run_outcome_reports_autofocus_data import COMPLETION_TIMEOUT, _AfRig

# (run mode, trigger, under a diagnostic's claim)
UNATTENDED_AUTOFOCUS = [
    pytest.param(
        SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN, 'autofocus_scan', False, id='af_scan'
    ),
    pytest.param(
        SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN, 'api_autofocus_scan', False, id='api_af_scan'
    ),
    pytest.param(SequencedCaptureRunMode.SINGLE_AUTOFOCUS, 'api_autofocus', True, id='borrowed'),
]


@pytest.fixture
def shown():
    """What the real centre delivers to a listener, i.e. what a person sees."""
    delivered: list = []
    notifications.add_listener(
        (lambda n: n.shown and delivered.append(n)), min_severity=Severity.INFO
    )
    try:
        yield delivered
    finally:
        notifications.remove_listener(delivered.append)
        notifications.close_run_scope()


def _autofocus_titles(shown) -> list[str]:
    return [n.title for n in shown if n.category == 'Autofocus']


def _start(rig, tmp_path, run_mode, trigger, borrowed):
    borrowed_claim = None
    if borrowed:
        borrowed_claim = rig.runner._activity_claim.try_claim('diagnostic').lend()
    plan = rig.prepare_autofocus(
        tmp_path / 'af',
        save_data=False,
        run_trigger_source=trigger,
        run_mode=run_mode,
        borrowed_claim=borrowed_claim,
    )
    return rig.runner.start(plan)


def _fail_the_sweep_after_three_grabs(rig) -> list[int]:
    grabs = [0]
    real_grab = rig.scope.imaging.capture_and_wait

    def _grab_then_fail(*args, **kwargs):
        grabs[0] += 1
        if grabs[0] > 3:
            raise RuntimeError('camera fault mid-sweep')
        return real_grab(*args, **kwargs)

    rig.scope.imaging.capture_and_wait = _grab_then_fail
    return grabs


def _wait_for_live_sweep(rig) -> None:
    deadline = time.monotonic() + COMPLETION_TIMEOUT
    while time.monotonic() < deadline:
        future = rig.af_thread.current_future
        if future is not None and not future.done() and rig.af_runner.run_in_progress():
            return
        time.sleep(0.005)
    pytest.fail('the run never started its sweep')


@pytest.mark.parametrize(('run_mode', 'trigger', 'borrowed'), UNATTENDED_AUTOFOCUS)
def test_a_sweep_that_fails_in_the_run_shows_nothing(tmp_path, shown, run_mode, trigger, borrowed):
    rig = _AfRig()
    grabs = _fail_the_sweep_after_three_grabs(rig)
    try:
        pending = _start(rig, tmp_path, run_mode, trigger, borrowed)
        assert rig._done.wait(timeout=COMPLETION_TIMEOUT), 'the run did not complete'
        assert pending.wait(timeout_s=COMPLETION_TIMEOUT) is not None
    finally:
        rig.close()

    assert grabs[0] > 3, 'precondition: the sweep reached the injected fault'
    assert _autofocus_titles(shown) == [], (
        f'an unattended ({trigger!r}) run showed its sweep failure: {_autofocus_titles(shown)}'
    )


@pytest.mark.parametrize('trigger', ['autofocus', 'api_autofocus'])
def test_a_one_position_autofocus_shows_its_sweep_failure_whoever_started_it(
    tmp_path, shown, trigger
):
    """The REST autofocus was muted while the button's was not: the trigger
    decided. The same operation now answers the same."""
    rig = _AfRig()
    grabs = _fail_the_sweep_after_three_grabs(rig)
    try:
        pending = _start(rig, tmp_path, SequencedCaptureRunMode.SINGLE_AUTOFOCUS, trigger, False)
        assert rig._done.wait(timeout=COMPLETION_TIMEOUT), 'the run did not complete'
        assert pending.wait(timeout_s=COMPLETION_TIMEOUT) is not None
    finally:
        rig.close()

    assert grabs[0] > 3, 'precondition: the sweep reached the injected fault'
    assert _autofocus_titles(shown), (
        f'a one-position autofocus started by {trigger!r} showed no sweep failure'
    )


@pytest.mark.parametrize(('run_mode', 'trigger', 'borrowed'), UNATTENDED_AUTOFOCUS)
def test_a_sweep_the_stop_aborts_shows_nothing(tmp_path, shown, run_mode, trigger, borrowed):
    rig = _AfRig()
    sweep_ended = threading.Event()
    try:
        pending = _start(rig, tmp_path, run_mode, trigger, borrowed)
        _wait_for_live_sweep(rig)
        sweep = rig.af_thread.current_future
        sweep.add_done_callback(lambda _f: sweep_ended.set())
        rig.runner._reset(pending)
        rig.protocol_thread.abort()
        assert pending.wait(timeout_s=COMPLETION_TIMEOUT) is not None, 'the run never settled'
        assert sweep_ended.wait(timeout=COMPLETION_TIMEOUT), 'the sweep never ended'
    finally:
        rig.close()

    assert isinstance(sweep.exception(), AutofocusAborted), (
        f'precondition: cleanup aborted the live sweep; it ended with {sweep.exception()!r}'
    )
    assert _autofocus_titles(shown) == [], (
        f'a stopped unattended ({trigger!r}) run showed its aborted sweep: '
        f'{_autofocus_titles(shown)}'
    )
