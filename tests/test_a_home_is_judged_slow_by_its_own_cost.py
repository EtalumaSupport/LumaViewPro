# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A home is judged slow against the cost it declared, not a single move's.

Each home body declares how long it may legitimately run
(``@slow_task_budget``), and the lane warns "Slow task" only past it. When a
home began holding the scope, the lane was handed a wrapper that runs the
home and releases the claim; the wrapper carried the home's name but not its
cost, so the lane judged every home against a single move's 5 s and logged a
WARNING on every home that succeeded.
"""

from __future__ import annotations

import pytest

from modules.scope_session import ScopeSession
from modules.sequential_io_executor import IOTask
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session():
    s = ScopeSession.create(
        complete_settings(microscope='LS850T'), simulate=True, warn_pre_release=False
    )
    yield s
    s.shutdown()


@pytest.mark.parametrize('axis', ['ALL', 'Z', 'T'])
def test_the_home_the_lane_runs_carries_the_homes_declared_cost(session, monkeypatch, axis):
    lane = session.scope._io_executor
    budgets = []
    real_call = lane.call

    def call(task, member, *args, **kwargs):
        if member == 'home':
            budgets.append(task.declared_slow_task_budget())
        return real_call(task, member, *args, **kwargs)

    monkeypatch.setattr(lane, 'call', call)
    impl, _ = session.scope.motion._home_body(axis)
    declared = IOTask(action=impl).declared_slow_task_budget()
    assert declared is not None and declared > IOTask.DEFAULT_SLOW_TASK_THRESHOLD_SEC

    session.scope.motion.home(axis)

    assert budgets == [declared]
