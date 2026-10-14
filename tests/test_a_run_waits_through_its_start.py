# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each run's waiting form is its start, waited for.

``ProtocolRunner.run_X(..., timeout_s=None)`` is ``start_X(...)`` followed
by the started operation's ``result(timeout_s)``, in-process only: it takes
exactly its start's parameters and ``timeout_s``; it returns the run's
outcome however the run ended; a wait that runs out raises
``RunWaitExpiredError``, quiet, a ``TimeoutError``, holding the run's
operation, which is still running and can be waited on again or stopped.
"""

import inspect

import pytest

from modules.api_surface import IN_PROCESS, mark_of
from modules.exceptions import RunWaitExpiredError
from modules.notification_center import OutcomeKind, outcome_of
from modules.protocol_runner import ProtocolRunner
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.test_going_to_a_step_is_the_sessions_move import _PLATE, _protocol, _step

_RUNS = ('single_scan', 'protocol', 'autofocus', 'autofocus_all_steps', 'zstack', 'composite')
# Only a failure waits this long.
_RUN_S = 120.0


def test_each_run_x_takes_its_start_s_parameters_and_a_timeout():
    wrong = {}
    for run in _RUNS:
        start = inspect.signature(getattr(ProtocolRunner, f'start_{run}')).parameters
        waits = inspect.signature(getattr(ProtocolRunner, f'run_{run}')).parameters
        expected = [
            *start.values(),
            inspect.Parameter(
                'timeout_s',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation='float | None',
            ),
        ]
        if [(p.name, p.default) for p in waits.values()] != [(p.name, p.default) for p in expected]:
            wrong[run] = (list(start), list(waits))
        if mark_of(getattr(ProtocolRunner, f'run_{run}')) != IN_PROCESS:
            wrong[f'run_{run}'] = 'not marked in-process'
    assert not wrong, wrong


@pytest.fixture
def session_and_runner(tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as opened:
        yield opened


def _one_step(session):
    objective = session.scope.runtime_state.get_current_objective_id()
    return _protocol(_PLATE, _step(20.0, 20.0, 5000.0, objective))


def test_run_single_scan_returns_the_run_s_outcome(session_and_runner, tmp_path):
    session, runner = session_and_runner

    outcome = runner.run_single_scan(_one_step(session), parent_dir=str(tmp_path))

    assert (outcome.status, outcome.reason) == ('completed', 'completed')
    assert not session.run_in_progress


def test_a_wait_that_runs_out_hands_back_the_run_still_going(session_and_runner, tmp_path):
    session, runner = session_and_runner
    # Seconds of travel to the step, so the run is still going when the
    # wait runs out.
    session.scope._motion_driver.set_timing_mode('realistic')

    with pytest.raises(RunWaitExpiredError) as expired:
        runner.run_single_scan(_one_step(session), parent_dir=str(tmp_path), timeout_s=0.01)

    assert isinstance(expired.value, TimeoutError)
    assert expired.value.reason == 'still_running'
    assert outcome_of(expired.value).kind == OutcomeKind.QUIET
    run = expired.value.operation
    assert not run.done() and session.run_in_progress
    run.stop()
    assert (run.result(timeout=_RUN_S).status, run.result().reason) == ('aborted', 'stopped')
