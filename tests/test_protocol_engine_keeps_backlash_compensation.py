# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The protocol engine must not disable backlash compensation on the
absolute moves that place capture Z.

The motion driver compensates downward Z moves by default: it overshoots
past the target and returns, so the stage lands where it was commanded
instead of short by the backlash distance. Every run kind that captures
at a stored focus reaches the stage through the engine's per-axis move,
so an engine-side override that turned compensation off silently put
every downward capture Z out of registration -- across full scans,
z-stacks, standalone autofocus and the composite run kind alike.

The invariant is pinned on what reaches the motion layer, not on the
engine's spelling: the engine may omit the parameter (driver default
applies) or pass True explicitly. Only an explicit False is the defect.
"""

from types import SimpleNamespace

import pytest

from modules.protocol_step_runner import ProtocolStepRunner


def _runner_capturing_moves(captured: list):
    def _move_absolute_impl(**kwargs):
        captured.append(kwargs)

    parent = SimpleNamespace(
        _io_executor=None,
        _scope=SimpleNamespace(motion=SimpleNamespace(_move_absolute_impl=_move_absolute_impl)),
    )
    return ProtocolStepRunner(parent)


@pytest.mark.parametrize('axis', ['X', 'Y', 'Z'])
def test_engine_absolute_move_leaves_backlash_compensation_on(axis):
    captured: list = []
    runner = _runner_capturing_moves(captured)

    runner._move_axis_through_io(axis, 1234.5)

    assert len(captured) == 1
    kwargs = captured[0]
    assert kwargs['axis'] == axis
    assert kwargs['position'] == 1234.5
    assert kwargs.get('overshoot_enabled', True) is True
