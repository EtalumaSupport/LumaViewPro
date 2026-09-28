# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Stop that arrives after its run ended leaves nothing running, and says nothing.

Every run control's Stop goes through submit_reported, whose one reporter
reads the engine's answer by its type. The engine answers a stop with
nothing live with RunAlreadyEndedError -- not a refusal, since the run
finished on its own -- which is quiet: logged, never shown, so a Stop
pressed just as a run finished shows no dialog and the control's redraw
draws it idle. The run_not_live refusal (another run is live) is shown
once as a warning, and a Stop of the live run shows nothing at all.
"""

from unittest.mock import MagicMock

import pytest

import modules.app_context as _app_ctx
import ui.ui_helpers as ui_helpers
from modules.exceptions import ProtocolRunRefusedError, RunAlreadyEndedError
from modules.notification_center import Severity


@pytest.fixture
def stop(monkeypatch):
    """Submit a Stop of *handle* to a pool that runs it at once; return what was shown."""
    from types import SimpleNamespace

    from modules.sequential_io_executor import ENQUEUED
    from tests.shown_outcomes import capture_shown

    shown = capture_shown(monkeypatch)
    pool = MagicMock()

    def _run_now(task):
        task.action(*task.args, **task.kwargs)
        return ENQUEUED

    pool.put.side_effect = _run_now
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn, timeout=0: fn(0))
    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(worker_pool=pool))

    def _stop(runner, handle):
        redraws = []
        ui_helpers.submit_reported(
            lambda: runner.reset(handle), lambda: redraws.append(1), 'STOP', stop=True
        )
        return redraws

    return SimpleNamespace(submit=_stop, shown=shown)


def _runner_raising(exc):
    runner = MagicMock()
    runner.reset.side_effect = exc
    return runner


def test_a_stop_after_its_run_ended_shows_nothing_and_redraws(stop):
    handle = object()
    runner = _runner_raising(RunAlreadyEndedError('That run has already ended; no run is live.'))

    redraws = stop.submit(runner, handle)

    runner.reset.assert_called_once_with(handle)
    assert stop.shown == []
    assert redraws == [1], 'the control is drawn from the engine, now idle'


def test_a_stale_stop_while_another_run_is_live_is_one_warning(stop):
    runner = _runner_raising(
        ProtocolRunRefusedError(
            reason='run_not_live',
            title='Run Already Ended',
            message='That run has already ended.',
            holder='protocol',
            holder_trigger='scan',
        )
    )

    redraws = stop.submit(runner, object())

    assert [(n.title, n.severity) for n in stop.shown] == [('Run Already Ended', Severity.WARNING)]
    assert redraws == [1]


def test_a_stop_of_the_live_run_shows_nothing(stop):
    redraws = stop.submit(MagicMock(), object())

    assert stop.shown == []
    assert redraws == [1]
