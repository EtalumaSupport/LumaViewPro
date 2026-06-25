# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for the LED State Authority skeleton (target_leds + apply).

These pin the decision function and the diff-and-emit behavior that the LED
authority is built on, in isolation -- before any decider is migrated onto it.
target_leds is a pure function of (transition, ctx), so most cases need no
hardware. apply drives a real Lumascope(simulate=True) and asserts the LED
command substream a driver listener records, exactly as the end-to-end lifecycle
test does (test_led_lifecycle_sequence.py); the difference is that this test
calls the authority methods directly rather than through a protocol run.

The authority's two guarantees under test:
  1. apply diffs against the cached state and emits only what changed, so
     re-applying an already-correct target produces no off-then-on blink.
  2. apply(RUN_START) reconciles the cache to a known baseline, so a run that
     starts after a write that bypassed the cache (a stimulation pulse, a prior
     session) re-asserts the correct channels instead of trusting a stale cache.
"""

from __future__ import annotations

import threading

import pytest

from modules.lumascope_api import Lumascope
from modules.lumascope_api.illumination import (
    LedEndPolicy,
    LedTransition,
    LedTransitionCtx,
)


class LedSubstream:
    """Thread-safe recorder of the LED-only command substream.

    Records (color, enabled, mA, owner) for every driver command, in order.
    Mirrors the recorder in test_led_lifecycle_sequence.py.
    """

    def __init__(self) -> None:
        self._events: list[tuple] = []
        self._lock = threading.Lock()

    def __call__(self, color, enabled, mA, owner) -> None:
        with self._lock:
            self._events.append((color, bool(enabled), mA, owner))

    @property
    def events(self) -> list[tuple]:
        with self._lock:
            return list(self._events)

    def on_events(self) -> list[tuple]:
        return [(c, m) for c, e, m, _o in self.events if e]

    def transitions(self, color: str) -> list[bool]:
        out: list[bool] = []
        for c, e, _m, _o in self.events:
            if c != color:
                continue
            if not out or out[-1] != e:
                out.append(e)
        return out

    def lit_transitions(self, color: str) -> list[bool]:
        trans = self.transitions(color)
        while trans and trans[0] is False:
            trans.pop(0)
        return trans

    def final_lit(self) -> set:
        lit: set[str] = set()
        for c, e, _m, _o in self.events:
            if e:
                lit.add(c)
            else:
                lit.discard(c)
        return lit

    def lit_at_most_one(self) -> bool:
        """Replay the stream; True iff at most one color is ever lit at any
        intermediate point. final_lit() only checks the end-state and so cannot
        catch a transient double-illumination -- this enforces the
        mutual-exclusion invariant at every step (a switch that lights the new
        channel before extinguishing the old would fail here but pass final_lit).
        """
        lit: set[str] = set()
        for c, e, _m, _o in self.events:
            if e:
                lit.add(c)
            else:
                lit.discard(c)
            if len(lit) > 1:
                return False
        return True

    def render(self) -> str:
        lines = [
            f'  {"ON " if e else "OFF"} {c:<6} mA={m} owner={o!r}' for c, e, m, o in self.events
        ]
        return '\n'.join(lines) if lines else '  (no LED events)'


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    yield s
    s.disconnect()


def _ch(scope, color: str) -> int:
    return scope.illumination.color2ch(color)


def _ctx(scope, color: str, mA: float, **kwargs) -> LedTransitionCtx:
    """Build a ctx for a single primary channel, plus any overrides."""
    return LedTransitionCtx(channel=_ch(scope, color), mA=mA, **kwargs)


# ---------------------------------------------------------------------------
# target_leds -- the pure decision function. No hardware: a channel int and an
# mA are the only inputs, so these assert the policy directly.
# ---------------------------------------------------------------------------

GREEN_CH, GREEN_MA = 1, 250.0
RED_CH, RED_MA = 0, 350.0
BLUE_CH, BLUE_MA = 2, 120.0

SNAPSHOT = frozenset({(BLUE_CH, BLUE_MA)})


def _td(transition, **ctx_kwargs) -> frozenset:
    """Call the pure decision function (a staticmethod -- no instance needed)."""
    from modules.lumascope_api.illumination import LedLease

    return LedLease.target_leds(transition, LedTransitionCtx(**ctx_kwargs))


def test_target_step_light_is_exclusive_primary():
    assert _td(LedTransition.STEP_LIGHT, channel=GREEN_CH, mA=GREEN_MA) == frozenset(
        {(GREEN_CH, GREEN_MA)}
    )


def test_target_af_enter_off_when_no_color():
    assert _td(LedTransition.AF_ENTER, channel=None, mA=None) == frozenset()
    assert _td(LedTransition.AF_ENTER, channel=GREEN_CH, mA=GREEN_MA) == frozenset(
        {(GREEN_CH, GREEN_MA)}
    )


def test_target_af_to_capture_holds_or_restores():
    # keep_led_on: hold the AF channel for the capture.
    assert _td(
        LedTransition.AF_TO_CAPTURE, channel=GREEN_CH, mA=GREEN_MA, keep_led_on=True
    ) == frozenset({(GREEN_CH, GREEN_MA)})
    # interactive AF: restore the pre-AF snapshot instead.
    assert (
        _td(
            LedTransition.AF_TO_CAPTURE,
            channel=GREEN_CH,
            mA=GREEN_MA,
            keep_led_on=False,
            snapshot_lit=SNAPSHOT,
        )
        == SNAPSHOT
    )


def test_target_step_boundary_zstack_always_holds():
    # Unconditional hold within a z-stack group, regardless of the across-move flag.
    assert _td(
        LedTransition.STEP_BOUNDARY,
        channel=GREEN_CH,
        mA=GREEN_MA,
        same_zstack_group=True,
        keep_led_across_moves=False,
    ) == frozenset({(GREEN_CH, GREEN_MA)})


def test_target_step_boundary_across_move_is_opt_in():
    base = {'channel': GREEN_CH, 'mA': GREEN_MA, 'same_color': True}
    # Default (opt-in OFF): extinguish on the stage move (photobleaching-safe).
    assert _td(LedTransition.STEP_BOUNDARY, **base, keep_led_across_moves=False) == frozenset()
    # Opt-in ON + same color: hold across the move.
    assert _td(LedTransition.STEP_BOUNDARY, **base, keep_led_across_moves=True) == frozenset(
        {(GREEN_CH, GREEN_MA)}
    )
    # Opt-in ON but a color switch: still extinguish (the hold is same-color only).
    assert (
        _td(
            LedTransition.STEP_BOUNDARY,
            channel=GREEN_CH,
            mA=GREEN_MA,
            same_color=False,
            keep_led_across_moves=True,
        )
        == frozenset()
    )


def test_target_run_end_policy():
    assert _td(LedTransition.RUN_END, end_policy=LedEndPolicy.OFF) == frozenset()
    assert (
        _td(
            LedTransition.RUN_END,
            end_policy=LedEndPolicy.RETURN_TO_ORIGINAL,
            snapshot_lit=SNAPSHOT,
        )
        == SNAPSHOT
    )


def test_target_run_start_is_snapshot():
    assert _td(LedTransition.RUN_START, snapshot_lit=SNAPSHOT) == SNAPSHOT
    assert _td(LedTransition.RUN_START) == frozenset()


def test_target_manual_step_preview_gate():
    assert _td(
        LedTransition.MANUAL_STEP, channel=GREEN_CH, mA=GREEN_MA, preview_on=True
    ) == frozenset({(GREEN_CH, GREEN_MA)})
    assert (
        _td(LedTransition.MANUAL_STEP, channel=GREEN_CH, mA=GREEN_MA, preview_on=False)
        == frozenset()
    )


def test_target_unknown_transition_raises():
    from modules.lumascope_api.illumination import LedLease

    with pytest.raises((ValueError, TypeError)):
        LedLease.target_leds('not-a-transition', LedTransitionCtx())


# ---------------------------------------------------------------------------
# apply -- diff-and-emit against a real driver. The lease owns the LEDs; apply
# drives them through the proven led_on / led_off primitives.
# ---------------------------------------------------------------------------


def test_apply_step_light_lights_exclusively_and_holds_idempotently(scope):
    ill = scope.illumination
    sub = LedSubstream()
    ill.add_led_listener(sub)
    lease = ill.acquire_led_lease('protocol')

    lease.apply(LedTransition.STEP_LIGHT, _ctx(scope, 'Green', GREEN_MA))
    assert ill.led_enabled('Green')
    assert sub.on_events() == [('Green', GREEN_MA)], sub.render()

    # Re-apply the identical target: idempotent, no off-then-on blink.
    lease.apply(LedTransition.STEP_LIGHT, _ctx(scope, 'Green', GREEN_MA))
    assert sub.lit_transitions('Green') == [True], sub.render()
    assert sub.on_events() == [('Green', GREEN_MA)], sub.render()

    # Switch color: Green off, Red on, never two lit at once.
    lease.apply(LedTransition.STEP_LIGHT, _ctx(scope, 'Red', RED_MA))
    assert sub.lit_transitions('Green') == [True, False], sub.render()
    assert sub.lit_transitions('Red') == [True], sub.render()
    assert sub.final_lit() == {'Red'}, sub.render()
    assert sub.lit_at_most_one(), f'double illumination during switch\n{sub.render()}'

    lease.release(leave_on=False)


def test_apply_step_boundary_hold_vs_off(scope):
    ill = scope.illumination
    sub = LedSubstream()
    ill.add_led_listener(sub)
    lease = ill.acquire_led_lease('protocol')

    lease.apply(LedTransition.STEP_LIGHT, _ctx(scope, 'Green', GREEN_MA))
    # z-stack boundary: held, zero new commands.
    lease.apply(
        LedTransition.STEP_BOUNDARY,
        _ctx(scope, 'Green', GREEN_MA, same_zstack_group=True),
    )
    assert sub.lit_transitions('Green') == [True], sub.render()
    assert ill.led_enabled('Green')

    # Plain boundary, opt-in off: extinguish.
    lease.apply(
        LedTransition.STEP_BOUNDARY,
        _ctx(scope, 'Green', GREEN_MA, same_color=True, keep_led_across_moves=False),
    )
    assert sub.lit_transitions('Green') == [True, False], sub.render()
    assert not ill.led_enabled('Green')

    lease.release(leave_on=False)


def test_apply_run_end_off_leaves_all_dark(scope):
    ill = scope.illumination
    sub = LedSubstream()
    ill.add_led_listener(sub)
    lease = ill.acquire_led_lease('protocol')

    lease.apply(LedTransition.STEP_LIGHT, _ctx(scope, 'Green', GREEN_MA))
    lease.apply(LedTransition.RUN_END, LedTransitionCtx(end_policy=LedEndPolicy.OFF))
    assert sub.lit_transitions('Green') == [True, False], sub.render()
    assert sub.final_lit() == set(), sub.render()

    lease.release(leave_on=False)


def test_apply_run_end_return_to_original_relights_snapshot(scope):
    ill = scope.illumination
    sub = LedSubstream()
    ill.add_led_listener(sub)
    lease = ill.acquire_led_lease('protocol')

    lease.apply(LedTransition.STEP_LIGHT, _ctx(scope, 'Green', GREEN_MA))
    snap = frozenset({(_ch(scope, 'Blue'), BLUE_MA)})
    lease.apply(
        LedTransition.RUN_END,
        LedTransitionCtx(end_policy=LedEndPolicy.RETURN_TO_ORIGINAL, snapshot_lit=snap),
    )
    # The run's Green is off before the snapshot's Blue is restored -- never two lit.
    assert sub.lit_transitions('Green') == [True, False], sub.render()
    assert sub.lit_transitions('Blue') == [True], sub.render()
    assert sub.final_lit() == {'Blue'}, sub.render()
    assert sub.lit_at_most_one(), f'double illumination during restore\n{sub.render()}'

    lease.release(leave_on=False)


def test_apply_run_start_reconcile_reasserts_after_cache_bypass(scope):
    """A stimulation-style write that bypasses the cache (led_on_fast updates the
    driver but not _led_state) leaves the cache claiming a channel is off while
    the hardware has it lit. A plain diff toward a new target would trust that
    stale cache and never turn the channel off. apply(RUN_START) clears to a
    known baseline first, so the channel is actually extinguished and the
    intended targets re-asserted."""
    ill = scope.illumination
    # Bypass the cache: driver lit, cache unaware.
    ill.led_on_fast(_ch(scope, 'Green'), GREEN_MA)
    assert not ill.led_enabled('Green'), 'precondition: cache is unaware of the fast write'

    sub = LedSubstream()
    ill.add_led_listener(sub)
    lease = ill.acquire_led_lease('protocol')

    snap = frozenset({(_ch(scope, 'Blue'), BLUE_MA)})
    lease.apply(LedTransition.RUN_START, LedTransitionCtx(snapshot_lit=snap))

    # The reconcile drove the hardware off (Green off fired) and re-asserted Blue.
    assert ('Green', False, 0.0, '') in sub.events, sub.render()
    assert sub.final_lit() == {'Blue'}, sub.render()
    assert ill.led_enabled('Blue')
    assert not ill.led_enabled('Green')

    lease.release(leave_on=False)


def test_apply_run_start_empty_snapshot_leaves_all_dark(scope):
    ill = scope.illumination
    ill.led_on(_ch(scope, 'Green'), GREEN_MA, owner='ui')
    sub = LedSubstream()
    ill.add_led_listener(sub)
    lease = ill.acquire_led_lease('protocol')

    lease.apply(LedTransition.RUN_START, LedTransitionCtx())
    assert sub.final_lit() == set(), sub.render()
    assert not ill.led_enabled('Green')

    lease.release(leave_on=False)


def test_apply_on_released_lease_is_a_noop(scope):
    """A transition on an already-released lease drives no LED. A queued apply
    can outlive its run; acting then would light or off a channel out of turn
    (worse, a new run may hold the lease under the same owner name)."""
    ill = scope.illumination
    lease = ill.acquire_led_lease('protocol')
    lease.apply(LedTransition.STEP_LIGHT, _ctx(scope, 'Green', GREEN_MA))
    lease.release(leave_on=False)

    sub = LedSubstream()
    ill.add_led_listener(sub)
    lease.apply(LedTransition.STEP_LIGHT, _ctx(scope, 'Red', RED_MA))
    assert sub.events == [], sub.render()
    assert not ill.led_enabled('Red')
