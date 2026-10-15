# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A step whose autofocus did not focus is on the run's record.

An autofocus scan, and a capture protocol with autofocus steps, carry on at
the fallback Z when a step's autofocus fails -- and ended 'completed', "The
run finished normally.", with nothing on the outcome to say which steps
were captured, or left, at a focus nobody found. Now the outcome lists each
such step (``RunOutcome.focus_failures``) and the ending's words name them;
the status stays 'completed'.
"""

from __future__ import annotations

import pytest

import modules.autofocus_functions as autofocus_functions
from modules.exceptions import AutofocusFailedError
from modules.run_outcome import PendingRunOutcome, RunEnding
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from tests.test_a_late_write_records_its_frame import _step
from tests.test_a_run_reports_what_it_captured import _run
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.test_run_outcome_reports_autofocus_data import COMPLETION_TIMEOUT, _AfRig

FLAT = str(AutofocusFailedError('flat_focus_curve'))


@pytest.fixture
def every_score_zero(monkeypatch):
    """Every frame scores zero: every autofocus sweep's curve is flat."""
    monkeypatch.setattr(autofocus_functions, 'focus_function', lambda **kwargs: 0.0)


def test_an_autofocus_scan_names_the_step_whose_focus_was_not_updated(tmp_path, every_score_zero):
    rig = _AfRig()
    try:
        plan = rig.prepare_autofocus(
            tmp_path, save_data=False, run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN
        )
        outcome = rig.runner.start(plan).result(timeout=COMPLETION_TIMEOUT)
    finally:
        rig.close()

    assert outcome.status == 'completed', outcome
    assert outcome.message == (
        'The scan finished; autofocus did not focus 1 step, whose focus was not updated.'
    )
    assert [(f.scan, f.step_index, f.step_name, f.cause) for f in outcome.focus_failures] == [
        (0, 0, 'AF_test', FLAT)
    ]


def test_a_capture_protocol_captures_the_step_at_its_fallback_and_names_it(
    tmp_path, every_score_zero
):
    with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
        outcome = _run(
            runner,
            tmp_path / 'runs',
            [_step('C0', 0, x=20.0, gain=1.0, autofocus=True), _step('C1', 1, x=30.0, gain=1.0)],
        )

    assert outcome.status == 'completed', outcome
    assert (outcome.captures.asked, outcome.captures.captured) == (2, 2)
    assert outcome.message == (
        'The run finished; autofocus did not focus 1 step, which was captured at its '
        'fallback focus.'
    )
    assert [(f.step_index, f.step_name, f.cause) for f in outcome.focus_failures] == [
        (0, 'C0', FLAT)
    ]


def test_every_settle_path_carries_the_focus_failures():
    from modules.run_outcome import FocusFailure

    failures = (FocusFailure(scan=0, step_index=1, step_name='B', cause='flat'),)
    ending = RunEnding('completed', 'completed', 'm')
    for settle in (
        lambda pending: pending.resolve_if_pending(ending),
        lambda pending: pending.force_resolve('shutdown', fallback=ending),
        lambda pending: pending.resolve(
            pending.arm(ending), merged=False, artifact_path=None, merge_reason=''
        ),
    ):
        pending = PendingRunOutcome()
        pending.record_focus_failures(failures)
        assert settle(pending) is True
        assert pending.wait(timeout_s=1.0).focus_failures == failures
