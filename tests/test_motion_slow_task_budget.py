# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A motion command's declared cost reaches the task that times it.

`IOTask` warns when a task outruns a threshold written for a SINGLE
motion. `_submit_motion` could always declare a longer one -- homing does,
because it legitimately runs 10-60 s -- but its sibling `_dispatch_motion`
built its task with no threshold argument at all, so every caller routed
through it was pinned to the one-motion default regardless of cost.

`move_turret` is three physically-waited motions (Z park, turret, Z
restore) and tripped the warning on every successful move.

These pin the invariant, not the number: a declared budget must reach the
task, and the turret must declare one above the single-motion default.
Retuning the value does not red them.
"""

import inspect

from modules.lumascope_api.motion import MotionAPI
from modules.sequential_io_executor import IOTask


class TestDeclaredBudgetReachesTheTask:
    def test_dispatch_motion_accepts_a_declared_budget(self):
        # The structural gap: the helper could not express what its sibling
        # could, so no caller through it could ever declare a cost.
        params = inspect.signature(MotionAPI._dispatch_motion).parameters
        assert 'slow_task_threshold_sec' in params, (
            '_dispatch_motion must accept a slow-task budget; without it every '
            'caller is pinned to the single-motion default'
        )

    def test_dispatch_motion_forwards_the_budget_to_the_task(self):
        # Accepting it and dropping it would pass the signature check while
        # leaving the warning exactly as wrong.
        src = inspect.getsource(MotionAPI._dispatch_motion)
        assert 'slow_task_threshold_sec=slow_task_threshold_sec' in src, (
            '_dispatch_motion must pass the declared budget to the IOTask it builds'
        )

    def test_turret_declares_more_than_one_motion_costs(self):
        # Asserted as "above the default" rather than == 15.0 so retuning the
        # budget row does not red the test.
        assert MotionAPI._TURRET_MOVE_SLOW_TASK_S > IOTask.DEFAULT_SLOW_TASK_THRESHOLD_SEC, (
            'a turret move is three motions; its budget must exceed the '
            'single-motion default or it warns on every successful move'
        )

    def test_move_turret_passes_its_budget(self):
        src = inspect.getsource(MotionAPI.move_turret)
        assert 'slow_task_threshold_sec=self._TURRET_MOVE_SLOW_TASK_S' in src, (
            'move_turret must declare its budget at the dispatch site'
        )

    def test_default_stays_written_for_one_motion(self):
        # The fix must not be "raise the default": IO_WORKER is shared, and a
        # single Z move that took 14 s should still be reported.
        assert IOTask.DEFAULT_SLOW_TASK_THRESHOLD_SEC == 5.0, (
            'the shared-worker default describes ONE motion; multi-motion '
            'commands declare their own budget instead of raising it'
        )
