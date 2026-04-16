# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for issue #618 — move_absolute_position race condition.

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
poll reflects the new (correct) target — guaranteed False until real
arrival. The same fix was applied to `move_relative_position`.

Side effect: the same race affected `AutofocusExecutor._iterate()`, which
checks `scope.is_moving()` before capturing each focus-curve sample. AF
"noise" from sporadic bad data points was likely caused by this same
race. The fix resolves both #618 and the latent AF issue.
"""

import ast
import pathlib
import threading
from unittest.mock import MagicMock

import pytest

# Heavy deps are mocked by tests/conftest.py at module-import time.


REPO = pathlib.Path(__file__).parent.parent
LUMASCOPE_API = REPO / "modules" / "lumascope_api.py"


# ---------------------------------------------------------------------------
# Source-level pin (catches reverts, runs without importing Lumascope)
# ---------------------------------------------------------------------------

def _find_method_source(class_name: str, method_name: str) -> str:
    tree = ast.parse(LUMASCOPE_API.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.unparse(child)
    raise AssertionError(f"{class_name}.{method_name} not found")


class TestSourceOrder_618:
    """#618 source pin: motion.move_abs_pos / move_rel_pos must appear
    BEFORE _set_axis_state(MOVING) in their respective methods."""

    def test_move_absolute_position_writes_hardware_first(self):
        body = _find_method_source("Lumascope", "move_absolute_position")
        move_idx = body.find("self.motion.move_abs_pos(")
        state_idx = body.find("self._set_axis_state(axis, AxisState.MOVING)")
        assert move_idx != -1, "move_abs_pos call missing from move_absolute_position"
        assert state_idx != -1, "_set_axis_state(MOVING) call missing"
        assert move_idx < state_idx, (
            "move_abs_pos must be called BEFORE _set_axis_state(MOVING) "
            "to avoid the #618 race"
        )

    def test_move_relative_position_writes_hardware_first(self):
        body = _find_method_source("Lumascope", "move_relative_position")
        move_idx = body.find("self.motion.move_rel_pos(")
        state_idx = body.find("self._set_axis_state(axis, AxisState.MOVING)")
        assert move_idx != -1, "move_rel_pos call missing from move_relative_position"
        assert state_idx != -1, "_set_axis_state(MOVING) call missing"
        assert move_idx < state_idx, (
            "move_rel_pos must be called BEFORE _set_axis_state(MOVING) "
            "to avoid the #618 race"
        )


# ---------------------------------------------------------------------------
# Runtime ordering test — uses real Lumascope(simulate=True) and traces
# the actual call sequence.
# ---------------------------------------------------------------------------

class TestRuntimeOrder_618:
    """#618 runtime: instrument the methods involved and verify call order."""

    def _track_calls(self, scope, axis):
        """Wrap motion.move_abs_pos / move_rel_pos and _set_axis_state to
        record the order in which they're called."""
        from modules.lumascope_api import AxisState

        call_order = []
        orig_move_abs = scope.motion.move_abs_pos
        orig_move_rel = scope.motion.move_rel_pos
        orig_set_state = scope._set_axis_state

        def track_move_abs(*args, **kwargs):
            call_order.append("motion.move_abs_pos")
            return orig_move_abs(*args, **kwargs)

        def track_move_rel(*args, **kwargs):
            call_order.append("motion.move_rel_pos")
            return orig_move_rel(*args, **kwargs)

        def track_set_state(ax, state):
            if ax == axis and state == AxisState.MOVING:
                call_order.append("set_state_MOVING")
            elif ax == axis and state == AxisState.IDLE:
                call_order.append("set_state_IDLE")
            return orig_set_state(ax, state)

        scope.motion.move_abs_pos = track_move_abs
        scope.motion.move_rel_pos = track_move_rel
        scope._set_axis_state = track_set_state
        return call_order

    def test_move_absolute_position_order_z(self):
        from modules.lumascope_api import Lumascope
        scope = Lumascope(simulate=True)
        scope.motion.set_timing_mode("fast")
        call_order = self._track_calls(scope, "Z")
        scope.move_absolute_position("Z", 5000.0, wait_until_complete=False)
        # The hardware write must come before the MOVING transition
        assert "motion.move_abs_pos" in call_order
        assert "set_state_MOVING" in call_order
        move_idx = call_order.index("motion.move_abs_pos")
        state_idx = call_order.index("set_state_MOVING")
        assert move_idx < state_idx, (
            f"motion.move_abs_pos must precede _set_axis_state(MOVING). "
            f"Got order: {call_order}"
        )

    def test_move_relative_position_order_z(self):
        from modules.lumascope_api import Lumascope
        scope = Lumascope(simulate=True)
        scope.motion.set_timing_mode("fast")
        call_order = self._track_calls(scope, "Z")
        scope.move_relative_position("Z", 100.0, wait_until_complete=False)
        assert "motion.move_rel_pos" in call_order
        assert "set_state_MOVING" in call_order
        move_idx = call_order.index("motion.move_rel_pos")
        state_idx = call_order.index("set_state_MOVING")
        assert move_idx < state_idx, (
            f"motion.move_rel_pos must precede _set_axis_state(MOVING). "
            f"Got order: {call_order}"
        )


# ---------------------------------------------------------------------------
# Race simulation — directly trigger the failure mode the old code had.
# ---------------------------------------------------------------------------

class TestRaceSimulation_618:
    """Simulate the exact race that caused #618 by injecting a 'motion
    monitor' that polls during motion.move_abs_pos. With the fix, the
    monitor's premature IDLE transition cannot happen because the new
    target is already on the hardware before the axis is marked MOVING."""

    def test_motion_monitor_cannot_falsely_set_idle_during_move(self):
        """The motion monitor (or any caller) inspecting axis state during
        motion.move_abs_pos must not see the axis as MOVING with an
        already-set arrival event — that's the race signature."""
        from modules.lumascope_api import Lumascope, AxisState
        scope = Lumascope(simulate=True)
        scope.motion.set_timing_mode("fast")

        # Hook motion.move_abs_pos to inspect state during the call
        orig_move_abs = scope.motion.move_abs_pos
        observations = []

        def observe_during_move(*args, **kwargs):
            # Inspect axis state and arrival event BEFORE the new target
            # is actually written. With the fix, _set_axis_state(MOVING)
            # has not yet been called, so:
            #   - axis state should still be IDLE (or UNKNOWN)
            #   - arrival event should still be SET (from prior move)
            # That means the motion monitor would NOT poll Z (state != MOVING)
            # and could not falsely conclude arrival.
            state = scope._axis_state["Z"]
            arrival_set = scope._arrival_events["Z"].is_set()
            observations.append((state, arrival_set))
            return orig_move_abs(*args, **kwargs)

        scope.motion.move_abs_pos = observe_during_move

        # Prime: do one move to set Z to a known IDLE state
        scope.move_absolute_position("Z", 1000.0, wait_until_complete=True)
        observations.clear()  # reset after the priming move

        # Now do a back-to-back move
        scope.move_absolute_position("Z", 5000.0, wait_until_complete=False)

        assert len(observations) == 1, (
            f"motion.move_abs_pos should be called once, got {len(observations)}"
        )
        state_during_move, arrival_during_move = observations[0]
        assert state_during_move != AxisState.MOVING, (
            f"Axis state must NOT be MOVING when motion.move_abs_pos starts. "
            f"Got {state_during_move}. The fix is to write hardware first."
        )
        # Arrival event was set at the end of the priming move and
        # should still be set when the new motion.move_abs_pos starts.
        assert arrival_during_move is True, (
            "Arrival event from the prior move should still be set. The fix "
            "delays the clear until AFTER the new TARGET_W is written."
        )


# ---------------------------------------------------------------------------
# Integration smoke test — back-to-back moves end up at the right place.
# ---------------------------------------------------------------------------

class TestBackToBackMoves_618:
    """Smoke test: rapid back-to-back wait_until_complete moves through
    the simulated motor must each leave the axis at the requested target.
    Catches gross regressions of the move_absolute_position contract."""

    def test_two_back_to_back_z_moves_end_at_correct_targets(self):
        from modules.lumascope_api import Lumascope
        scope = Lumascope(simulate=True)
        scope.motion.set_timing_mode("fast")

        scope.move_absolute_position("Z", 2000.0, wait_until_complete=True)
        pos1 = scope.motion.current_pos("Z")
        assert abs(pos1 - 2000.0) < 5.0, f"first move ended at {pos1}, expected ~2000"

        scope.move_absolute_position("Z", 8000.0, wait_until_complete=True)
        pos2 = scope.motion.current_pos("Z")
        assert abs(pos2 - 8000.0) < 5.0, f"second move ended at {pos2}, expected ~8000"

    def test_many_rapid_moves_end_at_correct_targets(self):
        from modules.lumascope_api import Lumascope
        scope = Lumascope(simulate=True)
        scope.motion.set_timing_mode("fast")

        # 20 rapid back-to-back moves, alternating direction
        targets = [3000.0, 7000.0] * 10
        for target in targets:
            scope.move_absolute_position("Z", target, wait_until_complete=True)
            actual = scope.motion.current_pos("Z")
            assert abs(actual - target) < 5.0, (
                f"move to {target} ended at {actual}"
            )
