# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The turret tells its own listeners its state and slot, whoever turned it.

The turret has a slot, not a position, so it is not on the position reads
or the position event (``MotionAPI.turret_status``, ``add_turret_listener``).
Its slot is recorded only once a turret command returns, after the turret
already says idle, so the event that carries the slot is the one sent when
it is recorded: a display that redrew on the idle state alone would show no
slot after every turn. A REST client's turn reaches the listener as a
person's does, which is how the GUI's turret buttons follow it.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from modules.lumascope_api import AxisState, TurretStatus
from rest.app import build_app


def test_a_rest_clients_turn_tells_the_listener_the_slot_it_reached(sim_turreted_session):
    session = sim_turreted_session
    motion = session.scope.motion
    motion.move_turret(1)
    heard = []
    motion.add_turret_listener(heard.append)
    try:
        with TestClient(build_app(session)) as client:
            answer = client.post('/api/v1/scope/motion/move_turret', json={'slot': 2})
    finally:
        motion.remove_turret_listener(heard.append)

    assert answer.status_code == 200, answer.text
    assert heard[-1] == TurretStatus(AxisState.IDLE, 2)
    assert motion.turret_status() == heard[-1]
    assert session.status.turret == heard[-1]


def test_the_position_reads_carry_the_stage_axes_alone(sim_turreted_session):
    motion = sim_turreted_session.scope.motion

    assert set(motion.axis_positions()) == {'X', 'Y', 'Z'}
    assert set(motion.get_target_positions()) == {'X', 'Y', 'Z'}
    assert set(sim_turreted_session.status.axes) == {'X', 'Y', 'Z'}
