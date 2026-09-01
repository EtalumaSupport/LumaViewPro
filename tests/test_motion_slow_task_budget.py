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
        # budget row does not red the test. Read through the task rather than
        # off a class attribute: where the constant lives is not the contract,
        # and the previous version broke when it moved to module scope.
        declared = IOTask(action=MotionAPI._move_turret_impl).declared_slow_task_budget()
        assert declared is not None and declared > IOTask.DEFAULT_SLOW_TASK_THRESHOLD_SEC, (
            'a turret move is three motions; its budget must exceed the '
            'single-motion default or it warns on every successful move'
        )

    def test_default_stays_written_for_one_motion(self):
        # The fix must not be "raise the default": IO_WORKER is shared, and a
        # single Z move that took 14 s should still be reported.
        assert IOTask.DEFAULT_SLOW_TASK_THRESHOLD_SEC == 5.0, (
            'the shared-worker default describes ONE motion; multi-motion '
            'commands declare their own budget instead of raising it'
        )


class TestBudgetTravelsWithTheCommand:
    """The budget must survive a submission the API never sees.

    A turret budget declared on the dispatcher warned on every successful
    move anyway: the GUI does not call the public method. `turret_select`
    runs on the Kivy main thread and builds its own IOTask around
    `_move_turret_impl`, because the public member blocks on the future and
    would freeze the UI. Anything attached to one wrapper is absent from the
    others, so the cost is declared on the command instead.
    """

    HAND_ROLLED = (
        ('_move_turret_impl', 15.0),
        ('_home_turret_impl', 120.0),
        ('_home_impl', 120.0),
        ('_zhome_impl', 120.0),
    )

    def test_hand_rolled_task_gets_the_commands_budget(self):
        for name, expected in self.HAND_ROLLED:
            task = IOTask(action=getattr(MotionAPI, name))
            assert task.resolve_slow_task_threshold() == expected, (
                f'{name} submitted outside the dispatcher fell back to the '
                f'one-motion default; this is the shape that shipped a false '
                f'"Slow task" warning on every successful turret move'
            )

    def test_explicit_submission_value_still_wins(self):
        task = IOTask(action=MotionAPI._move_turret_impl, slow_task_threshold_sec=99.0)
        assert task.resolve_slow_task_threshold() == 99.0

    def test_undeclared_command_keeps_the_default(self):
        task = IOTask(action=lambda: None)
        assert task.resolve_slow_task_threshold() == IOTask.DEFAULT_SLOW_TASK_THRESHOLD_SEC

    def test_stall_bar_sees_a_command_declaration(self):
        # _stall_threshold_s raises the stuck-worker bar only on a real
        # declaration. Reading the submission kwarg alone would miss every
        # budget stated at the def site.
        assert IOTask(action=MotionAPI._move_turret_impl).declared_slow_task_budget() == 15.0

    def test_undeclared_command_does_not_move_the_stall_bar(self):
        assert IOTask(action=lambda: None).declared_slow_task_budget() is None
