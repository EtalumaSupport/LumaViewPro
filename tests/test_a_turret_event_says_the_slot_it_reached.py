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

from _lumascope.modules.lumascope_api import TurretStatus
from _lumascope.modules.axes import AxisState
from _lumascope.rest.app import build_app


def test_a_rest_clients_turn_tells_the_listener_the_slot_it_reached(sim_turreted_session):
    session = sim_turreted_session
    motion = session.scope.motion
    motion.move_turret(1)
    heard = []
    motion.add_turret_listener(heard.append)
    try:
        with TestClient(build_app(session)) as client:
            started = client.post('/api/v1/scope/motion/start_move_turret', json={'slot': 2})
            assert started.status_code == 202, started.text
            job = client.get(started.headers['Location'], headers={'Prefer': 'wait=60'}).json()
    finally:
        motion.remove_turret_listener(heard.append)

    assert job['status'] == 'completed', job
    assert heard[-1] == TurretStatus(AxisState.IDLE, 2)
    assert motion.turret_status() == heard[-1]
    assert session.status.turret == heard[-1]


def test_the_position_reads_carry_the_stage_axes_alone(sim_turreted_session):
    motion = sim_turreted_session.scope.motion

    assert set(motion.axis_positions()) == {'X', 'Y', 'Z'}
    assert set(motion.get_target_positions()) == {'X', 'Y', 'Z'}
    assert set(sim_turreted_session.status.axes) == {'X', 'Y', 'Z'}
