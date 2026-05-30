"""Regression for #218: in a time-lapse, the stage should return to the first
step BETWEEN scans, so the next scan starts already positioned instead of
eating a last-step -> first-step move at scan start.

ProtocolRunLoop._return_to_first_step_between_scans pre-positions the stage at
step 0 after each scan when more scans remain (and not after the final scan,
and not while aborting). It uses a pure default_move -- not go_to_step -- so the
first step's LED is not powered during the idle inter-scan wait.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from modules.protocol_run_loop import ProtocolRunLoop


def _make_loop(remaining, aborted=False):
    p = SimpleNamespace()
    p.remaining_scans = lambda: remaining
    p._aborted = MagicMock()
    p._aborted.is_set.return_value = aborted
    p._protocol = MagicMock()
    p._protocol.step.return_value = {'X': 1.0, 'Y': 2.0, 'Z': 3.0}
    p._step_executor = MagicMock()
    return ProtocolRunLoop(p), p


def test_returns_to_first_step_when_scans_remain():
    loop, p = _make_loop(remaining=2)
    loop._return_to_first_step_between_scans()
    p._protocol.step.assert_called_once_with(idx=0)
    p._step_executor.default_move.assert_called_once_with(px=1.0, py=2.0, z=3.0)


def test_no_move_after_final_scan():
    loop, p = _make_loop(remaining=0)
    loop._return_to_first_step_between_scans()
    p._step_executor.default_move.assert_not_called()


def test_no_move_when_aborting():
    loop, p = _make_loop(remaining=2, aborted=True)
    loop._return_to_first_step_between_scans()
    p._step_executor.default_move.assert_not_called()


def test_move_failure_is_swallowed_not_raised():
    loop, p = _make_loop(remaining=2)
    p._step_executor.default_move.side_effect = RuntimeError('motor busy')
    # A failed return move must not propagate into the run loop.
    loop._return_to_first_step_between_scans()
