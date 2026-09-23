"""Regression: a run's moves and turret moves route through the io
executor instead of calling the scope primitive directly.

The scan loop runs on protocol_thread, which is a DIFFERENT thread from
the io executor's single worker. If a move (or a turret move) is issued by
calling the scope primitive directly on protocol_thread, it races the
leds_off/led_on that the capture path queues on the io worker -- when the
move wins the race, the previous step's LED stays lit through the
well-to-well move (the red-LED-stuck-on report).

The run's step runner routes every move through io_executor.protocol_put
so it lands on the single io worker in FIFO order behind the step's
leds_off and ahead of the next led_on. The GUI has no run lane at all: a
run moves the scope itself on every host, and the GUI only displays.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

from modules.protocol_step_runner import ProtocolStepRunner

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _step_runner():
    fut = MagicMock()
    motion = MagicMock()
    parent = SimpleNamespace(
        _io_executor=SimpleNamespace(protocol_put=MagicMock(return_value=fut)),
        _scope=SimpleNamespace(motion=motion),
    )
    return ProtocolStepRunner(parent), parent, motion, fut


def test_a_run_axis_move_is_queued_on_the_io_worker():
    step_runner, parent, motion, fut = _step_runner()

    step_runner._move_axis_through_io('X', 1234.0)

    parent._io_executor.protocol_put.assert_called_once()
    task = parent._io_executor.protocol_put.call_args.args[0]
    assert task.action is motion._move_absolute_impl
    # The runner waits for completion: the next axis follows this one.
    fut.result.assert_called_once()
    # Never called on protocol_thread itself -- that is the bypass that
    # races leds_off.
    motion._move_absolute_impl.assert_not_called()


def test_a_run_turret_move_is_queued_on_the_io_worker():
    step_runner, parent, motion, fut = _step_runner()

    step_runner._move_turret_through_io(2)

    parent._io_executor.protocol_put.assert_called_once()
    task = parent._io_executor.protocol_put.call_args.args[0]
    assert task.action is motion._move_turret_impl
    assert task.kwargs == {'position': 2, 'restore_z': False}
    fut.result.assert_called_once()
    motion._move_turret_impl.assert_not_called()


def test_the_gui_queues_nothing_on_the_run_lane():
    """No production call to protocol_put in ui/: the run lane belongs to
    the run, and a GUI that queued its own moves there would be a second
    implementation of the run's motion, ordered by nothing the run owns."""
    offenders = []
    for path in sorted((REPO_ROOT / 'ui').rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('protocol_put', 'protocol_put_wait')
            ):
                offenders.append(f'{path.relative_to(REPO_ROOT)}:{node.lineno}')
    assert offenders == []
