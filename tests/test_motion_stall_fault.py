# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A connected axis that never reaches its target must not stay MOVING
forever.

The disconnect fault already bounded a vanished board, but a CONNECTED
axis whose position_reached never fires (obstruction, firmware fault) had
no bound at all: the published motion timeout bounds explicit waiters
only, while every state-reader -- capture settle-checks, is_moving
pollers, the protocol runner -- wedged indefinitely. The monitor now
faults a stalled move to UNKNOWN at the published bound and notifies the
user once, the same terminal shape as the disconnect fault.
"""

import time
from tests.scope_fakes import build_scope


def _wait_until(predicate, timeout=3.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _errors_posted(centre_posts):
    from modules.notification_center import Severity

    return [(n.category, n.title, n.message) for n in centre_posts if n.severity == Severity.ERROR]


def test_stalled_move_faults_axis_and_notifies(monkeypatch, centre_posts):
    from modules.lumascope_api import AxisState

    scope = build_scope(simulate=True)
    motion = scope.motion
    # Fault on the next poll rather than after the 120 s production bound.
    motion._MOTION_SETTLE_TIMEOUT_S = 0.0
    # The stall condition: the board answers, but the move never arrives.
    monkeypatch.setattr(motion, 'get_target_status', lambda ax: False)

    # A move's start: MOVING, then armed once its target is written.
    assert motion._begin_move('Z', motion._stop_generation)
    motion._publish_drive('Z', True, 0.0, 0.0, None)

    assert _wait_until(lambda: not motion.is_moving()), (
        'a stalled axis must leave MOVING at the published bound, not hang forever'
    )
    assert motion._axis_state['Z'] == AxisState.UNKNOWN
    assert motion._arrival_events['Z'].is_set()
    errors = _errors_posted(centre_posts)
    assert len(errors) == 1, f'exactly one stall notification expected, got {errors}'
    assert 'stalled' in errors[0][1].lower()


def test_arriving_move_never_stall_faults(monkeypatch, centre_posts):
    """A move that reaches its target must complete IDLE with no fault and
    no notification, even with the stall bound at its most aggressive --
    arrival must always win over the stall clock."""
    from modules.lumascope_api import AxisState

    scope = build_scope(simulate=True)
    motion = scope.motion
    motion._MOTION_SETTLE_TIMEOUT_S = 0.0
    monkeypatch.setattr(motion, 'get_target_status', lambda ax: True)

    motion._begin_move('Z', motion._stop_generation)
    motion._publish_drive('Z', True, 0.0, 0.0, None)

    assert _wait_until(lambda: motion._axis_state['Z'] == AxisState.IDLE), (
        'an arriving move must transition to IDLE'
    )
    errors = _errors_posted(centre_posts)
    assert errors == [], f'no stall notification expected on arrival, got {errors}'
