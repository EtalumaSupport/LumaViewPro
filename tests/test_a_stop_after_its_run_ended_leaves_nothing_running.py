# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Stop that arrives after its run ended leaves nothing running, and says nothing.

Every run control's Stop goes through submit_reported, whose one reporter
reads the outcome by its type. A run's stop once its run has ended does
nothing and raises nothing (Eric, 2026-10-13: a stop once ended does
nothing), so a Stop pressed just as a run finished shows no dialog, and the
control's redraw draws it idle. A Stop of the live run shows nothing either.
"""

import concurrent.futures
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.app_context as _app_ctx
import ui.ui_helpers as ui_helpers
from modules.operation import Operation
from tests.pool_fakes import run_task_now


@pytest.fixture
def stop(monkeypatch, centre_posts):
    """Submit a Stop of *run* to a pool that runs it at once; return the redraws."""
    from modules.sequential_io_executor import ENQUEUED

    pool = MagicMock()

    def _run_now(task):
        run_task_now(task)
        return ENQUEUED

    pool.put.side_effect = _run_now
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn: fn(0))
    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(worker_pool=pool))

    def _stop(run):
        redraws = []
        ui_helpers.submit_reported(run.stop, lambda: redraws.append(1), 'STOP', stop=True)
        return redraws

    return SimpleNamespace(submit=_stop, posts=centre_posts)


def test_a_stop_after_its_run_ended_shows_nothing_stops_nothing_and_redraws(stop):
    stopped = []
    run = Operation(stop=lambda: stopped.append(1))
    concurrent.futures.Future.set_result(run, None)

    redraws = stop.submit(run)

    assert stopped == [], "an ended run's stop reached its work"
    assert stop.posts == []
    assert redraws == [1], 'the control is drawn from the engine, now idle'


def test_a_stop_of_the_live_run_stops_it_and_shows_nothing(stop):
    stopped = []
    run = Operation(stop=lambda: stopped.append(1))

    redraws = stop.submit(run)

    assert stopped == [1]
    assert stop.posts == []
    assert redraws == [1]
