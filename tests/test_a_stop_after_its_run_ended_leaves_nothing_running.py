# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Stop that arrives after its run ended leaves nothing running.

Every run control's Stop goes through reset_with_refusal_boundary, and
what it returns decides the widget's next move: True puts the control
back, False leaves it alone because another run is still live. The
engine answers a stop with nothing live with RunAlreadyEndedError -- not
a refusal, since the run finished on its own -- and the boundary must
read that as nothing left running, or a Stop pressed just as a run
finished strands its button mid-stop. The run_not_live refusal (another
run is live) must still read as False.
"""

from unittest.mock import MagicMock

from modules.exceptions import ProtocolRunRefusedError, RunAlreadyEndedError
from ui.ui_helpers import reset_with_refusal_boundary


def _runner_raising(exc):
    runner = MagicMock()
    runner.reset.side_effect = exc
    return runner


def test_a_stop_after_its_run_ended_reports_nothing_left_running():
    handle = object()
    runner = _runner_raising(RunAlreadyEndedError('That run has already ended; no run is live.'))
    assert reset_with_refusal_boundary(runner, handle) is True
    runner.reset.assert_called_once_with(handle)


def test_a_stale_stop_while_another_run_is_live_reports_the_refusal():
    runner = _runner_raising(
        ProtocolRunRefusedError(
            reason='run_not_live',
            title='Run Already Ended',
            message='That run has already ended.',
            holder='protocol',
            holder_trigger='scan',
        )
    )
    assert reset_with_refusal_boundary(runner, object()) is False


def test_a_stop_of_the_live_run_reports_it_stopped():
    runner = MagicMock()
    assert reset_with_refusal_boundary(runner, object()) is True
