"""Regression for #709: a motor board that disconnects mid-move must not
leave the axis stuck MOVING forever.

Before the fix the motion monitor skipped a disconnected axis on every
poll, so is_moving() never cleared and autofocus / the protocol runner
wedged silently with no notification. The monitor now bounds the
disconnect: after a short deadline it faults the axis to a terminal state
(UNKNOWN, which fires the arrival event so waiters unblock) and notifies
the user exactly once.
"""

import time


def _wait_until(predicate, timeout=3.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_disconnect_mid_move_faults_axis_and_notifies(monkeypatch):
    from modules.lumascope_api import AxisState, Lumascope

    errors = []
    import modules.notification_center as nc

    monkeypatch.setattr(
        nc.notifications,
        'error',
        lambda category, title, message, **k: errors.append((category, title, message)),
    )

    scope = Lumascope(simulate=True)
    motion = scope.motion
    # Fault on the next poll rather than after the 3 s production deadline.
    motion._DISCONNECT_FAULT_S = 0.0
    # Simulate the board vanishing mid-move.
    scope._motion_driver.is_connected = lambda: False

    # Drive an axis MOVING; the monitor wakes and now sees the disconnect.
    motion._set_axis_state('Z', AxisState.MOVING)
    assert motion.is_moving()

    assert _wait_until(lambda: not motion.is_moving()), (
        'a disconnected axis must leave MOVING within the fault deadline, not hang forever'
    )
    assert motion._axis_state['Z'] == AxisState.UNKNOWN
    assert motion._arrival_events['Z'].is_set()
    assert len(errors) == 1, f'exactly one disconnect notification expected, got {errors}'
    assert 'Motor' in errors[0][1]
