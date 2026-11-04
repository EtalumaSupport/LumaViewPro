# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The protocol engine leaves the approach of the moves that place capture Z to the API.

A downward Z move to a place overshoots past the target and returns, so the
stage lands where it was commanded instead of short by the backlash
distance. Every run kind that captures at a stored focus reaches the stage
through the engine's moves, so an engine-side override that turned the
approach off put every downward capture Z out of registration -- across full
scans, z-stacks, standalone autofocus and the composite run kind alike. The
approach is the API's (``MotionAPI.start_move_z``): the engine asks for Z at
a place and says nothing of how.
"""

from types import SimpleNamespace

from modules.protocol_step_runner import ProtocolStepRunner


def _runner_capturing_moves(captured: list):
    def start_move_xy(*args, **kwargs):
        captured.append(('xy', args, kwargs))

    def start_move_z(*args, **kwargs):
        captured.append(('z', args, kwargs))

    parent = SimpleNamespace(
        _io_executor=None,
        _scope=SimpleNamespace(
            motion=SimpleNamespace(start_move_xy=start_move_xy, start_move_z=start_move_z)
        ),
    )
    return ProtocolStepRunner(parent)


def test_the_engine_asks_for_z_at_a_place_and_leaves_its_approach_to_the_api():
    captured: list = []
    runner = _runner_capturing_moves(captured)

    runner._move_to_stage(100.0, 200.0, 1234.5)

    assert captured == [('xy', (100.0, 200.0), {}), ('z', (1234.5,), {})]
