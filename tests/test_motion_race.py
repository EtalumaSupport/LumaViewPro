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

import pytest

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
                    writes.append((motion._axis_state[ax], motion._armed_seq[ax]))
            return orig_move_abs(ax, *args, **kwargs)

        scope._motion_driver.move_abs_pos = track_move_abs
        return writes

    def _assert_disarmed_at_every_write_and_armed_after(self, scope, writes):
        from modules.lumascope_api import AxisState

        assert writes, 'the move wrote no target'
        for state, armed in writes:
            assert state == AxisState.MOVING, f'a target was written to a {state} axis'
            assert armed is None, 'a target was written to an armed axis'
        with scope.motion._axis_state_lock:
            state = scope.motion._axis_state['Z']
            armed = scope.motion._armed_seq['Z']
            seq = scope.motion._drive_seq['Z']
        assert state == AxisState.IDLE or armed == seq, (
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
            # What the monitor does on a reached bit: note the count, then
            # ask for IDLE at it, armed only.
            with motion._axis_state_lock:
                noted = motion._drive_seq[ax]
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


# ---------------------------------------------------------------------------
# Issue #674: move_relative must write _move_profile so the
# position predictor can animate the crosshair during relative moves.
# ---------------------------------------------------------------------------


class TestMoveRelProfile_674:
    """Regression for #674 crosshair-prediction during relative moves.

    Original bug (bench bundle SN12062-2026-05-22-182105.zip):
    `move_relative` did NOT write `_move_profile[axis]`. State
    still transitioned to MOVING, so `get_current_position` routed through
    `_predicted_position` -- which returned None for a missing profile and
    fell through to `_read_position_cache`. But `_pos_cache[axis]` had
    just been updated to the target, so the crosshair jumped to the
    target instead of animating along the ramp.

    Initial fix: mirror move_absolute's profile-write block.

    H3 refinement (bench 2026-05-26 -- this commit): profile-write must
    happen AFTER the driver call returns, not before. The serial round-
    trip to write the hardware target takes ~50 ms during which the motor
    has NOT begun physical motion. Capturing start_time before the driver
    call made _predicted_position's elapsed lead the motor by the full
    serial RT, producing a visible crosshair-outruns-stage effect on long
    moves. Profile-write precedes the arming, so the profile is in place
    before any verdict can end the move, and an arrival never leaves it on
    an IDLE axis.
    """

    def test_runtime_abs_profile_set_after_driver_returns(self):
        """ABSOLUTE-move path: bench evidence (LS850 click-on-plate)
        confirmed the visible bug is most dramatic on absolute moves
        through io_executor's task queue. Hooked at the driver call's
        RETURN moment, profile must be UNSET -- proves the write follows
        the driver call so start_time captures post-serial-RT timing."""

        scope = home_sim_scope(build_scope(simulate=True))
        scope._motion_driver.set_timing_mode('fast')

        scope.motion.move_absolute('X', 1000.0)

        observed = {}
        orig_move_abs = scope._motion_driver.move_abs_pos

        def snapshot_at_driver_return(axis, um, *args, **kwargs):
            result = orig_move_abs(axis, um, *args, **kwargs)
            with scope.motion._move_profile_lock:
                profile = scope.motion._move_profile.get(axis)
            observed['profile_at_driver_return'] = None if profile is None else dict(profile)
            return result

        scope._motion_driver.move_abs_pos = snapshot_at_driver_return

        scope.motion.start_move_absolute('X', 1400.0)

        assert observed.get('profile_at_driver_return') is None, (
            'profile must be UNSET when move_abs_pos returns -- the outer '
            'move_absolute writes it AFTER the driver returns. '
            f'Observed: {observed.get("profile_at_driver_return")!r}'
        )
        with scope.motion._move_profile_lock:
            profile = scope.motion._move_profile.get('X')
        assert profile is not None, (
            '_move_profile[X] must be written by the time move_absolute returns'
        )
        assert profile['target_pos'] == pytest.approx(1400.0, abs=5.0)

    @pytest.mark.parametrize('move', ['absolute', 'relative'])
    def test_runtime_profile_present_at_the_arrival(self, move):
        """The profile must already be written when the move's arrival is
        written -- otherwise it lands after the arrival cleared it and sits
        on an IDLE axis as the target of a move that has ended."""
        import threading

        from modules.lumascope_api import AxisState

        scope = home_sim_scope(build_scope(simulate=True))
        scope._motion_driver.set_timing_mode('fast')

        scope.motion.move_absolute('X', 1000.0)

        observed = {}
        arrived = threading.Event()
        orig_set_state = scope.motion._set_axis_state

        def snapshot_at_arrival(ax, state, **kwargs):
            first = ax == 'X' and state == AxisState.IDLE and 'profile_at_arrival' not in observed
            if first:
                with scope.motion._move_profile_lock:
                    profile = scope.motion._move_profile.get('X')
                observed['profile_at_arrival'] = None if profile is None else dict(profile)
            wrote = orig_set_state(ax, state, **kwargs)
            if first:
                arrived.set()
            return wrote

        scope.motion._set_axis_state = snapshot_at_arrival

        if move == 'absolute':
            scope.motion.start_move_absolute('X', 1400.0)
        else:
            scope.motion.start_move_relative('X', 400.0)

        assert arrived.wait(10.0), 'the move never arrived'
        assert observed['profile_at_arrival'] is not None, (
            'the profile must be written before the axis is armed, so it is '
            'in place before the arrival'
        )
        with scope.motion._move_profile_lock:
            assert scope.motion._move_profile.get('X') is None, (
                'a profile was left on the IDLE axis'
            )

    def test_runtime_profile_set_after_driver_returns(self):
        """Production path: profile must be present + populated correctly
        right after move_relative returns. Hooked at the driver
        call's RETURN moment (still inside the driver's move, before the outer
        method writes profile), profile should be UNSET -- proves the
        write is positioned after the driver call returns."""

        scope = home_sim_scope(build_scope(simulate=True))
        scope._motion_driver.set_timing_mode('fast')

        # Prime: move to a known non-zero start; wait_until_complete clears profile.
        scope.motion.move_absolute('X', 1000.0)
        with scope.motion._move_profile_lock:
            assert scope.motion._move_profile.get('X') is None, (
                'profile should be cleared after IDLE transition'
            )

        observed = {}
        orig_move = scope._motion_driver.move_abs_pos

        def snapshot_at_driver_return(axis, um, *args, **kwargs):
            result = orig_move(axis, um, *args, **kwargs)
            # Snapshot RIGHT before returning to the outer method. With
            # the H3 fix, profile is still None here -- the outer method
            # writes it after this returns. With the pre-H3 code, profile
            # would already be set here.
            with scope.motion._move_profile_lock:
                profile = scope.motion._move_profile.get(axis)
            observed['profile_at_driver_return'] = None if profile is None else dict(profile)
            return result

        scope._motion_driver.move_abs_pos = snapshot_at_driver_return

        delta = 300.0
        scope.motion.start_move_relative('X', delta)

        # H3 invariant: profile not yet written at driver-return.
        assert observed.get('profile_at_driver_return') is None, (
            'profile must be UNSET when the driver call returns -- the outer '
            'move_relative writes it AFTER the driver returns so '
            'start_time captures post-serial-RT timing (H3 refinement). '
            f'Observed: {observed.get("profile_at_driver_return")!r}'
        )

        # Sanity: profile IS set by the time move_relative returns.
        with scope.motion._move_profile_lock:
            profile = scope.motion._move_profile.get('X')
        assert profile is not None, (
            '_move_profile[X] must be written by the time move_relative '
            'returns (still required for the predictor when state is MOVING)'
        )
        assert profile['start_pos'] == pytest.approx(1000.0, abs=5.0)
        assert profile['target_pos'] == pytest.approx(1300.0, abs=5.0)
        assert profile['ramp'] and profile['ramp'].get('vmax', 0) > 0

    def test_runtime_predictor_returns_non_none_during_move(self):
        """End-to-end: after move_relative returns and state is
        MOVING, _predicted_position must return a value. This is the
        crosshair-animation precondition (regardless of pre-H3 vs post-H3
        positioning of the profile-write -- by the time the outer method
        returns, profile must be set)."""
        from modules.lumascope_api import AxisState

        scope = home_sim_scope(build_scope(simulate=True))
        scope._motion_driver.set_timing_mode('fast')

        scope.motion.move_absolute('X', 1000.0)
        scope.motion.start_move_relative('X', 500.0)

        # Observation AFTER the move method returns: profile must be set,
        # state must be MOVING (or just transitioned), and predictor must
        # return a valid interpolated position.
        predicted = scope.motion._predicted_position('X')
        assert predicted is not None, (
            '_predicted_position must return a value once profile is set -- '
            'None means the crosshair will fall through to cache'
        )
        with scope.motion._axis_state_lock:
            state = scope.motion._axis_state.get('X')
        assert state == AxisState.MOVING, (
            f'state should be MOVING immediately after move_rel returns '
            f'(wait_until_complete=False); got {state!r}'
        )

    def test_h3_start_time_captured_after_driver_delay(self):
        """H3 timing invariant: profile.start_time must be captured AFTER
        the driver call returns. Inject a known delay into the driver and
        verify start_time > (t_before_call + delay). Without the H3 fix,
        start_time would be < (t_before_call + delay) because it was
        captured BEFORE the driver call."""
        import time as _time

        scope = home_sim_scope(build_scope(simulate=True))
        scope._motion_driver.set_timing_mode('fast')

        scope.motion.move_absolute('X', 1000.0)

        DELAY_S = (
            0.040  # 40 ms -- well above scheduler jitter; below an arrow's perception threshold
        )
        orig_move = scope._motion_driver.move_abs_pos

        def slow_driver(axis, um, *args, **kwargs):
            _time.sleep(DELAY_S)
            return orig_move(axis, um, *args, **kwargs)

        scope._motion_driver.move_abs_pos = slow_driver

        t_before = _time.monotonic()
        scope.motion.start_move_relative('X', 300.0)
        t_after = _time.monotonic()

        with scope.motion._move_profile_lock:
            profile = scope.motion._move_profile.get('X')
        assert profile is not None, 'profile must be set after move returns'
        start_time = profile['start_time']
        # H3 invariant: start_time > t_before + DELAY_S (was captured AFTER the driver call).
        # Pre-H3 code: start_time <= t_before + small-margin (captured BEFORE the driver call).
        assert start_time >= t_before + DELAY_S * 0.9, (
            f'profile.start_time={start_time:.6f} must be at least t_before+90%*delay '
            f'(={t_before + DELAY_S * 0.9:.6f}); H3 fix captures start_time AFTER '
            f'the driver call. Pre-H3, start_time was captured BEFORE the call.'
        )
        assert start_time <= t_after, (
            f'profile.start_time={start_time:.6f} must precede move-method return '
            f't_after={t_after:.6f}'
        )
