# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run turns the turret to each step's objective itself, on every host.

A headless or REST run used to move only X, Y and Z: the turret move lived
in the GUI's step navigation, so a script's run captured every step through
whatever glass was in the light path and named each file for the objective
the step asked for. The run engine now issues the turret move before each
step's X, Y and Z, through the motion API under the run's taking, and a
host's step callback only displays. A move the lane refuses raises rather
than being skipped, so a step never captures at the position the last move left.
"""

from types import SimpleNamespace

import pytest

from modules.exceptions import HardwareCommandRefusedError
from modules.protocol_step_runner import ProtocolStepRunner
from tests.scope_fakes import TEST_TURRET_OBJECTIVES
from tests.test_a_saved_frame_keeps_its_objective import _two_objective_protocol
from tests.test_composite_run_e2e import headless_settings, open_composite_session


def test_a_headless_run_visits_each_steps_slot(tmp_path, monkeypatch):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        motion = session.scope.motion
        real_turret_move = motion._move_turret_impl
        visited = []

        def _turret_move(position, restore_z=True):
            real_turret_move(position=position, restore_z=restore_z)
            visited.append((position, restore_z))

        monkeypatch.setattr(motion, '_move_turret_impl', _turret_move)
        displayed = []

        outcome = runner.run_single_scan(
            protocol=_two_objective_protocol(),
            parent_dir=str(tmp_path / 'runs'),
            image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
            callbacks={'go_to_step': lambda **kw: displayed.append(kw['include_move'])},
        )
        result = outcome.wait(timeout_s=60.0)
        assert result is not None and result.status == 'completed', result

    slots = {obj: slot for slot, obj in TEST_TURRET_OBJECTIVES.items() if obj}
    protocol = _two_objective_protocol()
    expected = [slots[protocol.step(idx=i)['Objective']] for i in range(protocol.num_steps())]
    # Each step's slot, in order; Z is not restored after the safety park
    # because the step's own Z move follows.
    assert [slot for slot, _ in visited][: len(expected)] == expected
    assert all(restore_z is False for _, restore_z in visited)
    # The host's callback is told to display, never to move.
    assert displayed and not any(displayed)


def test_a_refused_move_raises_instead_of_being_skipped():
    refusal = HardwareCommandRefusedError('exclusive_activity_running', 'move', 'diagnostic')

    def _refuse(*args, **kwargs):
        raise refusal

    parent = SimpleNamespace(
        _io_executor=None,
        _scope=SimpleNamespace(motion=SimpleNamespace(move_absolute=_refuse, move_turret=_refuse)),
    )
    step_runner = ProtocolStepRunner(parent)

    with pytest.raises(HardwareCommandRefusedError) as excinfo:
        step_runner._move_axis_through_io('X', 1000.0)
    assert excinfo.value is refusal

    with pytest.raises(HardwareCommandRefusedError) as excinfo:
        step_runner._move_turret_through_io(2)
    assert excinfo.value is refusal
