# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for issue #618 -- move_absolute race condition.

Original report: backlash characterization script's upwards pass captured
images at wildly wrong Z positions, intermittently. Same image (stddev=4.35)
returned for every dropout, downwards pass unaffected.

Root cause: `Lumascope.move_absolute` used to call
`_set_axis_state(axis, MOVING)` BEFORE `motion.move_abs_pos`. The state
change cleared the per-axis arrival event and woke the motion monitor
thread. The motion monitor then acquired the serial lock and polled
STATUS_R while `motion.move_abs_pos` was still doing its serial round-trips
(reading current_pos for the overshoot check, then writing TARGET_W).
During that ~50ms window, the hardware still had the PRIOR move's target
loaded, so STATUS_R returned `position_reached=True` (XACTUAL was matching
the prior XTARGET). The motion monitor concluded the move was done, called
`_set_axis_state(IDLE)`, and SET the arrival event.

When the main thread then called `wait_until_finished_moving()`, it found
the arrival event already set and returned immediately. The script captured
an image while the motor was actually still on its way to the new target,
producing the dropouts.

Fix: no verdict before the new target is on the board. The first fix
wrote the target before the MOVING transition; that left the axis IDLE
through a Z move's backlash leg, nearly the whole move, and a waiter
returned during it. Now the axis goes MOVING disarmed before its first
target write and is armed once its final target is written: the monitor
polls a disarmed axis but writes no verdict for it, so a reached bit read
before the arm -- the prior target's or the leg's -- ends nothing.

Side effect: the same race affected `AutofocusRunner._iterate()`, which
checks `scope.is_moving()` before capturing each focus-curve sample. AF
"noise" from sporadic bad data points was likely caused by this same
race. The fix resolves both #618 and the latent AF issue.
"""

from tests.scope_fakes import build_scope, home_sim_scope


# Heavy deps are mocked by tests/conftest.py at module-import time.


# ---------------------------------------------------------------------------
# Runtime ordering test -- uses real Lumascope(simulate=True) and reads the
# axis's state and arming at every target write.
# ---------------------------------------------------------------------------


class TestRuntimeOrder_618:
    """#618 runtime: every target write happens on a MOVING, disarmed axis."""

    def _track_writes(self, scope, axis):
        """Record, at each driver target write, the axis's state and whether it is armed."""
        writes = []
        motion = scope.motion
        orig_move_abs = scope._motion_driver.move_abs_pos

        def track_move_abs(ax, *args, **kwargs):
            if ax == axis:
                with motion._axis_state_lock:
                    writes.append((motion._axis_state[ax], motion._current_move[ax].armed))
            return orig_move_abs(ax, *args, **kwargs)

        scope._motion_driver.move_abs_pos = track_move_abs
        return writes

    def _assert_disarmed_at_every_write_and_armed_after(self, scope, writes):
        from modules.lumascope_api import AxisState

        assert writes, 'the move wrote no target'
        for state, armed in writes:
            assert state == AxisState.MOVING, f'a target was written to a {state} axis'
            assert armed is False, 'a target was written to an armed axis'
        with scope.motion._axis_state_lock:
            state = scope.motion._axis_state['Z']
            armed = scope.motion._current_move['Z'].armed
        assert state == AxisState.IDLE or armed, (
            'the axis was left disarmed after its final target was written'
        )

    def test_move_absolute_order_z(self):
        scope = home_sim_scope(build_scope(simulate=True))
        scope._motion_driver.set_timing_mode('fast')
        scope.motion.move_absolute('Z', 5000.0)
        writes = self._track_writes(scope, 'Z')
        # Downward, so the backlash leg is one of the writes.
        scope.motion.start_move_absolute('Z', 1000.0)
        assert len(writes) == 2, writes
        self._assert_disarmed_at_every_write_and_armed_after(scope, writes)

    def test_move_relative_order_z(self):
        scope = home_sim_scope(build_scope(simulate=True))
        scope._motion_driver.set_timing_mode('fast')
        writes = self._track_writes(scope, 'Z')
        scope.motion.start_move_relative('Z', 100.0)
        self._assert_disarmed_at_every_write_and_armed_after(scope, writes)


# ---------------------------------------------------------------------------
# Race simulation -- directly trigger the failure mode the old code had.
# ---------------------------------------------------------------------------


class TestRaceSimulation_618:
    """Simulate the exact race that caused #618: the monitor reads the prior
    target's reached bit while the new target is being written, and asks for
    the arrival verdict. The axis is disarmed then, so the verdict writes
    nothing."""

    def test_motion_monitor_cannot_falsely_set_idle_during_move(self):
        """A verdict asked for during the target write -- the monitor's IDLE
        on a reached bit -- leaves the axis MOVING with its arrival unset."""
        from modules.lumascope_api import AxisState

        scope = home_sim_scope(build_scope(simulate=True))
        scope._motion_driver.set_timing_mode('fast')
        motion = scope.motion

        # Prime: do one move to set Z to a known IDLE state
        motion.move_absolute('Z', 1000.0)

        orig_move_abs = scope._motion_driver.move_abs_pos
        observations = []

        def verdict_during_write(ax, *args, **kwargs):
            # What the monitor does on a reached bit: note the move, then
            # ask for IDLE for it, armed only.
            with motion._axis_state_lock:
                noted = motion._current_move[ax]
            wrote = motion._set_axis_state(ax, AxisState.IDLE, verdict_for=noted, armed_only=True)
            observations.append(
                (wrote, motion._axis_state[ax], motion._arrival_events[ax].is_set())
            )
            return orig_move_abs(ax, *args, **kwargs)

        scope._motion_driver.move_abs_pos = verdict_during_write
        # Upward, so the one write is the final target.
        motion.start_move_absolute('Z', 5000.0)

        assert len(observations) == 1, observations
        wrote, state, arrival_set = observations[0]
        assert wrote is False, 'a reached bit read during the target write ended the move'
        assert state == AxisState.MOVING
        assert arrival_set is False, 'a waiter would have returned before the move began'


# ---------------------------------------------------------------------------
# Integration smoke test -- back-to-back moves end up at the right place.
# ---------------------------------------------------------------------------


class TestBackToBackMoves_618:
    """Smoke test: rapid back-to-back wait_until_complete moves through
    the simulated motor must each leave the axis at the requested target.
    Catches gross regressions of the move_absolute contract."""

    def test_two_back_to_back_z_moves_end_at_correct_targets(self):

        scope = home_sim_scope(build_scope(simulate=True))
        scope._motion_driver.set_timing_mode('fast')

        scope.motion.move_absolute('Z', 2000.0)
        pos1 = scope._motion_driver.current_pos('Z')
        assert abs(pos1 - 2000.0) < 5.0, f'first move ended at {pos1}, expected ~2000'

        scope.motion.move_absolute('Z', 8000.0)
        pos2 = scope._motion_driver.current_pos('Z')
        assert abs(pos2 - 8000.0) < 5.0, f'second move ended at {pos2}, expected ~8000'

    def test_many_rapid_moves_end_at_correct_targets(self):

        scope = home_sim_scope(build_scope(simulate=True))
        scope._motion_driver.set_timing_mode('fast')

        # 20 rapid back-to-back moves, alternating direction
        targets = [3000.0, 7000.0] * 10
        for target in targets:
            scope.motion.move_absolute('Z', target)
            actual = scope._motion_driver.current_pos('Z')
            assert abs(actual - target) < 5.0, f'move to {target} ended at {actual}'
