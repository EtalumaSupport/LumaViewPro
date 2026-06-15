# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for issue #618 -- move_absolute_position race condition.

Original report: backlash characterization script's upwards pass captured
images at wildly wrong Z positions, intermittently. Same image (stddev=4.35)
returned for every dropout, downwards pass unaffected.

Root cause: `Lumascope.move_absolute_position` used to call
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

Fix: write the hardware target first, THEN transition the axis to MOVING.
By the time `_set_axis_state(MOVING)` clears the arrival event, the new
XTARGET is already on the hardware, so any subsequent `position_reached`
poll reflects the new (correct) target -- guaranteed False until real
arrival. The same fix was applied to `move_relative_position`.

Side effect: the same race affected `AutofocusRunner._iterate()`, which
checks `scope.is_moving()` before capturing each focus-curve sample. AF
"noise" from sporadic bad data points was likely caused by this same
race. The fix resolves both #618 and the latent AF issue.
"""

import pytest

# Heavy deps are mocked by tests/conftest.py at module-import time.


# ---------------------------------------------------------------------------
# Runtime ordering test -- uses real Lumascope(simulate=True) and traces
# the actual call sequence. (The hardware-write-before-MOVING invariant
# is proven here behaviorally; there is no separate source-text pin.)
# ---------------------------------------------------------------------------


class TestRuntimeOrder_618:
    """#618 runtime: instrument the methods involved and verify call order."""

    def _track_calls(self, scope, axis):
        """Wrap motion.move_abs_pos / move_rel_pos and _set_axis_state to
        record the order in which they're called. The _set_axis_state wrap
        targets scope.motion._set_axis_state (the canonical surface) because
        intra-motion calls reference self._set_axis_state directly after
        the 2c band-aid revert."""
        from modules.lumascope_api import AxisState

        call_order = []
        orig_move_abs = scope._motion_driver.move_abs_pos
        orig_move_rel = scope._motion_driver.move_rel_pos
        orig_set_state = scope.motion._set_axis_state

        def track_move_abs(*args, **kwargs):
            call_order.append('motion.move_abs_pos')
            return orig_move_abs(*args, **kwargs)

        def track_move_rel(*args, **kwargs):
            call_order.append('motion.move_rel_pos')
            return orig_move_rel(*args, **kwargs)

        def track_set_state(ax, state):
            if ax == axis and state == AxisState.MOVING:
                call_order.append('set_state_MOVING')
            elif ax == axis and state == AxisState.IDLE:
                call_order.append('set_state_IDLE')
            return orig_set_state(ax, state)

        scope._motion_driver.move_abs_pos = track_move_abs
        scope._motion_driver.move_rel_pos = track_move_rel
        scope.motion._set_axis_state = track_set_state
        return call_order

    def test_move_absolute_position_order_z(self):
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        scope._motion_driver.set_timing_mode('fast')
        call_order = self._track_calls(scope, 'Z')
        scope.motion.move_absolute_position('Z', 5000.0, wait_until_complete=False)
        # The hardware write must come before the MOVING transition
        assert 'motion.move_abs_pos' in call_order
        assert 'set_state_MOVING' in call_order
        move_idx = call_order.index('motion.move_abs_pos')
        state_idx = call_order.index('set_state_MOVING')
        assert move_idx < state_idx, (
            f'motion.move_abs_pos must precede _set_axis_state(MOVING). Got order: {call_order}'
        )

    def test_move_relative_position_order_z(self):
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        scope._motion_driver.set_timing_mode('fast')
        call_order = self._track_calls(scope, 'Z')
        scope.motion.move_relative_position('Z', 100.0, wait_until_complete=False)
        assert 'motion.move_rel_pos' in call_order
        assert 'set_state_MOVING' in call_order
        move_idx = call_order.index('motion.move_rel_pos')
        state_idx = call_order.index('set_state_MOVING')
        assert move_idx < state_idx, (
            f'motion.move_rel_pos must precede _set_axis_state(MOVING). Got order: {call_order}'
        )


# ---------------------------------------------------------------------------
# Race simulation -- directly trigger the failure mode the old code had.
# ---------------------------------------------------------------------------


class TestRaceSimulation_618:
    """Simulate the exact race that caused #618 by injecting a 'motion
    monitor' that polls during motion.move_abs_pos. With the fix, the
    monitor's premature IDLE transition cannot happen because the new
    target is already on the hardware before the axis is marked MOVING."""

    def test_motion_monitor_cannot_falsely_set_idle_during_move(self):
        """The motion monitor (or any caller) inspecting axis state during
        motion.move_abs_pos must not see the axis as MOVING with an
        already-set arrival event -- that's the race signature."""
        from modules.lumascope_api import Lumascope, AxisState

        scope = Lumascope(simulate=True)
        scope._motion_driver.set_timing_mode('fast')

        # Hook motion.move_abs_pos to inspect state during the call
        orig_move_abs = scope._motion_driver.move_abs_pos
        observations = []

        def observe_during_move(*args, **kwargs):
            # Inspect axis state and arrival event BEFORE the new target
            # is actually written. With the fix, _set_axis_state(MOVING)
            # has not yet been called, so:
            #   - axis state should still be IDLE (or UNKNOWN)
            #   - arrival event should still be SET (from prior move)
            # That means the motion monitor would NOT poll Z (state != MOVING)
            # and could not falsely conclude arrival.
            state = scope.motion._axis_state['Z']
            arrival_set = scope.motion._arrival_events['Z'].is_set()
            observations.append((state, arrival_set))
            return orig_move_abs(*args, **kwargs)

        scope._motion_driver.move_abs_pos = observe_during_move

        # Prime: do one move to set Z to a known IDLE state
        scope.motion.move_absolute_position('Z', 1000.0, wait_until_complete=True)
        observations.clear()  # reset after the priming move

        # Now do a back-to-back move
        scope.motion.move_absolute_position('Z', 5000.0, wait_until_complete=False)

        assert len(observations) == 1, (
            f'motion.move_abs_pos should be called once, got {len(observations)}'
        )
        state_during_move, arrival_during_move = observations[0]
        assert state_during_move != AxisState.MOVING, (
            f'Axis state must NOT be MOVING when motion.move_abs_pos starts. '
            f'Got {state_during_move}. The fix is to write hardware first.'
        )
        # Arrival event was set at the end of the priming move and
        # should still be set when the new motion.move_abs_pos starts.
        assert arrival_during_move is True, (
            'Arrival event from the prior move should still be set. The fix '
            'delays the clear until AFTER the new TARGET_W is written.'
        )


# ---------------------------------------------------------------------------
# Integration smoke test -- back-to-back moves end up at the right place.
# ---------------------------------------------------------------------------


class TestBackToBackMoves_618:
    """Smoke test: rapid back-to-back wait_until_complete moves through
    the simulated motor must each leave the axis at the requested target.
    Catches gross regressions of the move_absolute_position contract."""

    def test_two_back_to_back_z_moves_end_at_correct_targets(self):
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        scope._motion_driver.set_timing_mode('fast')

        scope.motion.move_absolute_position('Z', 2000.0, wait_until_complete=True)
        pos1 = scope._motion_driver.current_pos('Z')
        assert abs(pos1 - 2000.0) < 5.0, f'first move ended at {pos1}, expected ~2000'

        scope.motion.move_absolute_position('Z', 8000.0, wait_until_complete=True)
        pos2 = scope._motion_driver.current_pos('Z')
        assert abs(pos2 - 8000.0) < 5.0, f'second move ended at {pos2}, expected ~8000'

    def test_many_rapid_moves_end_at_correct_targets(self):
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        scope._motion_driver.set_timing_mode('fast')

        # 20 rapid back-to-back moves, alternating direction
        targets = [3000.0, 7000.0] * 10
        for target in targets:
            scope.motion.move_absolute_position('Z', target, wait_until_complete=True)
            actual = scope._motion_driver.current_pos('Z')
            assert abs(actual - target) < 5.0, f'move to {target} ended at {actual}'


# ---------------------------------------------------------------------------
# Issue #674: move_relative_position must write _move_profile so the
# position predictor can animate the crosshair during relative moves.
# ---------------------------------------------------------------------------


class TestMoveRelProfile_674:
    """Regression for #674 crosshair-prediction during relative moves.

    Original bug (bench bundle SN12062-2026-05-22-182105.zip):
    `move_relative_position` did NOT write `_move_profile[axis]`. State
    still transitioned to MOVING, so `get_current_position` routed through
    `_predicted_position` -- which returned None for a missing profile and
    fell through to `_read_position_cache`. But `_pos_cache[axis]` had
    just been updated to the target, so the crosshair jumped to the
    target instead of animating along the ramp.

    Initial fix: mirror move_absolute_position's profile-write block.

    H3 refinement (bench 2026-05-26 -- this commit): profile-write must
    happen AFTER the driver call returns, not before. The serial round-
    trip to write the hardware target takes ~50 ms during which the motor
    has NOT begun physical motion. Capturing start_time before the driver
    call made _predicted_position's elapsed lead the motor by the full
    serial RT, producing a visible crosshair-outruns-stage effect on long
    moves. Profile-write still precedes _set_axis_state(MOVING) so the
    predictor is ready by the time the move is observable as in-progress.
    """

    def test_runtime_abs_profile_set_after_driver_returns(self):
        """ABSOLUTE-move path: bench evidence (LS850 click-on-plate)
        confirmed the visible bug is most dramatic on absolute moves
        through io_executor's task queue. Hooked at the driver call's
        RETURN moment, profile must be UNSET -- proves the write follows
        the driver call so start_time captures post-serial-RT timing."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        scope._motion_driver.set_timing_mode('fast')

        scope.motion.move_absolute_position('X', 1000.0, wait_until_complete=True)

        observed = {}
        orig_move_abs = scope._motion_driver.move_abs_pos

        def snapshot_at_driver_return(axis, um, *args, **kwargs):
            result = orig_move_abs(axis, um, *args, **kwargs)
            with scope.motion._move_profile_lock:
                profile = scope.motion._move_profile.get(axis)
            observed['profile_at_driver_return'] = None if profile is None else dict(profile)
            return result

        scope._motion_driver.move_abs_pos = snapshot_at_driver_return

        scope.motion.move_absolute_position('X', 1400.0, wait_until_complete=False)

        assert observed.get('profile_at_driver_return') is None, (
            'profile must be UNSET when move_abs_pos returns -- the outer '
            'move_absolute_position writes it AFTER the driver returns. '
            f'Observed: {observed.get("profile_at_driver_return")!r}'
        )
        with scope.motion._move_profile_lock:
            profile = scope.motion._move_profile.get('X')
        assert profile is not None, (
            '_move_profile[X] must be written by the time move_absolute_position returns'
        )
        assert profile['target_pos'] == pytest.approx(1400.0, abs=5.0)

    @pytest.mark.parametrize('move', ['absolute', 'relative'])
    def test_runtime_profile_present_at_moving_transition(self, move):
        """The profile must already be written when the axis transitions
        to MOVING -- otherwise the predictor reads None for an observably
        moving axis and the crosshair falls through to the cache."""
        from modules.lumascope_api import AxisState, Lumascope

        scope = Lumascope(simulate=True)
        scope._motion_driver.set_timing_mode('fast')

        scope.motion.move_absolute_position('X', 1000.0, wait_until_complete=True)

        observed = {}
        orig_set_state = scope.motion._set_axis_state

        def snapshot_at_moving(ax, state):
            if ax == 'X' and state == AxisState.MOVING:
                with scope.motion._move_profile_lock:
                    profile = scope.motion._move_profile.get('X')
                observed['profile_at_moving'] = None if profile is None else dict(profile)
            return orig_set_state(ax, state)

        scope.motion._set_axis_state = snapshot_at_moving

        if move == 'absolute':
            scope.motion.move_absolute_position('X', 1400.0, wait_until_complete=False)
        else:
            scope.motion.move_relative_position('X', 400.0, wait_until_complete=False)

        assert 'profile_at_moving' in observed, 'the move must transition X to MOVING'
        assert observed['profile_at_moving'] is not None, (
            'profile must be written BEFORE _set_axis_state(MOVING) so the '
            'predictor is ready when the axis becomes observably MOVING'
        )

    def test_runtime_profile_set_after_driver_returns(self):
        """Production path: profile must be present + populated correctly
        right after move_relative_position returns. Hooked at the driver
        call's RETURN moment (still inside move_rel_pos, before the outer
        method writes profile), profile should be UNSET -- proves the
        write is positioned after the driver call returns."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        scope._motion_driver.set_timing_mode('fast')

        # Prime: move to a known non-zero start; wait_until_complete clears profile.
        scope.motion.move_absolute_position('X', 1000.0, wait_until_complete=True)
        with scope.motion._move_profile_lock:
            assert scope.motion._move_profile.get('X') is None, (
                'profile should be cleared after IDLE transition'
            )

        observed = {}
        orig_move_rel = scope._motion_driver.move_rel_pos

        def snapshot_at_driver_return(axis, um, *args, **kwargs):
            result = orig_move_rel(axis, um, *args, **kwargs)
            # Snapshot RIGHT before returning to the outer method. With
            # the H3 fix, profile is still None here -- the outer method
            # writes it after this returns. With the pre-H3 code, profile
            # would already be set here.
            with scope.motion._move_profile_lock:
                profile = scope.motion._move_profile.get(axis)
            observed['profile_at_driver_return'] = None if profile is None else dict(profile)
            return result

        scope._motion_driver.move_rel_pos = snapshot_at_driver_return

        delta = 300.0
        scope.motion.move_relative_position('X', delta, wait_until_complete=False)

        # H3 invariant: profile not yet written at driver-return.
        assert observed.get('profile_at_driver_return') is None, (
            'profile must be UNSET when the driver call returns -- the outer '
            'move_relative_position writes it AFTER the driver returns so '
            'start_time captures post-serial-RT timing (H3 refinement). '
            f'Observed: {observed.get("profile_at_driver_return")!r}'
        )

        # Sanity: profile IS set by the time move_relative_position returns.
        with scope.motion._move_profile_lock:
            profile = scope.motion._move_profile.get('X')
        assert profile is not None, (
            '_move_profile[X] must be written by the time move_relative_position '
            'returns (still required for the predictor when state is MOVING)'
        )
        assert profile['start_pos'] == pytest.approx(1000.0, abs=5.0)
        assert profile['target_pos'] == pytest.approx(1300.0, abs=5.0)
        assert profile['ramp'] and profile['ramp'].get('vmax', 0) > 0

    def test_runtime_predictor_returns_non_none_during_move(self):
        """End-to-end: after move_relative_position returns and state is
        MOVING, _predicted_position must return a value. This is the
        crosshair-animation precondition (regardless of pre-H3 vs post-H3
        positioning of the profile-write -- by the time the outer method
        returns, profile must be set)."""
        from modules.lumascope_api import AxisState, Lumascope

        scope = Lumascope(simulate=True)
        scope._motion_driver.set_timing_mode('fast')

        scope.motion.move_absolute_position('X', 1000.0, wait_until_complete=True)
        scope.motion.move_relative_position('X', 500.0, wait_until_complete=False)

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

        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        scope._motion_driver.set_timing_mode('fast')

        scope.motion.move_absolute_position('X', 1000.0, wait_until_complete=True)

        DELAY_S = (
            0.040  # 40 ms -- well above scheduler jitter; below an arrow's perception threshold
        )
        orig_move_rel = scope._motion_driver.move_rel_pos

        def slow_driver(axis, um, *args, **kwargs):
            _time.sleep(DELAY_S)
            return orig_move_rel(axis, um, *args, **kwargs)

        scope._motion_driver.move_rel_pos = slow_driver

        t_before = _time.monotonic()
        scope.motion.move_relative_position('X', 300.0, wait_until_complete=False)
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
