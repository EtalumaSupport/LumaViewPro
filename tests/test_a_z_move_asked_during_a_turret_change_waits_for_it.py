# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Z move asked while the turret changes waits for the whole change.

``MotionAPI.move_turret`` is three moves -- the Z park, the turret move and
the Z restore -- run as one task on the scope's one-worker io lane, and
every Z mover is a call on that lane (``Admission.LANE_CALL``). So a Z move
asked mid-change runs after the restore and Z ends where it was asked. Run
anywhere inside the change, the restore would put Z back where it was
before the park, and the asked move would be lost with nothing raised.
"""

from __future__ import annotations

import threading
import time

import pytest

_START_Z_UM = 3000.0
_ASKED_Z_UM = 5000.0
_WAIT_S = 10.0


def test_a_z_move_asked_mid_change_runs_after_the_restore(sim_turreted_session):
    scope = sim_turreted_session.scope
    motion = scope.motion
    motion.move_turret(1)
    motion.move_z(_START_Z_UM)

    outcome = {}

    def call(name, fn):
        try:
            fn()
            outcome[name] = 'returned'
        except BaseException as ex:
            outcome[name] = ex

    held = scope.simulation.hold_axis('T')
    try:
        changing = threading.Thread(target=call, args=('turret', lambda: motion.move_turret(2)))
        changing.start()
        held.result(_WAIT_S)

        lane = scope.io_lane()
        asking = threading.Thread(target=call, args=('z', lambda: motion.move_z(_ASKED_Z_UM)))
        asking.start()
        # The lane offers no event for "a call is queued"; its queue depth is
        # the one witness that the Z move was asked while the change held the
        # lane, not after it. Bounded and asserted, so a miss fails here.
        deadline = time.monotonic() + _WAIT_S
        while lane.queue.qsize() == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert lane.queue.qsize() == 1, 'the Z move never queued behind the turret change'

        # Still mid-change: Z is parked, and the asked move has not run.
        assert motion.get_current_position('Z') == pytest.approx(0.0, abs=0.1)
    finally:
        scope.simulation.clear()

    changing.join(_WAIT_S)
    asking.join(_WAIT_S)

    assert outcome == {'turret': 'returned', 'z': 'returned'}
    assert motion.turret_status().slot == 2
    assert motion.get_current_position('Z') == pytest.approx(_ASKED_Z_UM, abs=0.1)
