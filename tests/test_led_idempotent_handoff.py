# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the idempotent LED ownership handoff.

Autofocus and protocol stepping both need to make a single channel the only
lit LED. Doing that with a nuclear leds_off followed by led_on blinks a channel
that is already lit at the target current off->on -- a visible flicker at every
autofocus scan boundary and on every same-color protocol step (Z-stack slice).

The diff-based restore_led_state and the apply_transition manual-nav path leave
an already-correct channel untouched, so re-asserting it does not flicker. LED
listeners fire only when a command actually reaches the driver (a self-skipped
no-op does not fire), so counting listener events is a direct measure of "did
the LED blink".
"""

import threading

import pytest

from modules.lumascope_api import Lumascope
from modules.lumascope_api.illumination import LedTransition, LedTransitionCtx


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    yield s


def _color(scope, ch):
    return scope.illumination.ch2color(ch)


def test_restore_does_not_blink_channel_already_at_target(scope):
    """restore_led_state leaves a channel already at its snapshot target lit,
    without an off->on blink -- the autofocus scan-end case."""
    scope.illumination.led_on(channel=0, mA=100, owner='autofocus')
    snapshot = scope.illumination.save_led_state('autofocus')

    events = []
    scope.illumination.add_led_listener(lambda c, e, m, o: events.append((c, e)))
    scope.illumination.restore_led_state(snapshot, owner='autofocus')

    assert events == [], f'restore blinked a channel already at target: {events}'
    assert scope.illumination.led_enabled(_color(scope, 0))


def test_restore_still_relights_a_channel_that_was_turned_off(scope):
    """The graceful path stays correct: a snapshot channel that is currently
    off is turned back on by restore."""
    scope.illumination.led_on(channel=0, mA=100, owner='autofocus')
    snapshot = scope.illumination.save_led_state('autofocus')
    scope.illumination.leds_off_owned('autofocus')
    assert not scope.illumination.led_enabled(_color(scope, 0))

    scope.illumination.restore_led_state(snapshot, owner='autofocus')
    assert scope.illumination.led_enabled(_color(scope, 0))


def test_restore_owner_scoped_leaves_other_channels_alone(scope):
    """restore with an owner only clears that owner's channels; another
    subsystem's channel is left untouched (preserves the existing contract)."""
    scope.illumination.led_on(channel=0, mA=100, owner='ui')
    scope.illumination.led_on(channel=1, mA=50, owner='autofocus')
    snapshot = scope.illumination.save_led_state('autofocus')
    scope.illumination.leds_off_owned('autofocus')

    scope.illumination.restore_led_state(snapshot, owner='autofocus')
    assert scope.illumination.led_enabled(_color(scope, 0))  # ui's channel untouched
    assert scope.illumination.led_enabled(_color(scope, 1))  # autofocus's restored


@pytest.fixture
def scope_io(scope):
    """Simulated scope with a started io_executor registered, so the
    X_async LED methods (which dispatch IOTasks) run end to end. Manual
    step navigation reaches the LED through apply_transition_async."""
    from modules.sequential_io_executor import SequentialIOExecutor

    ex = SequentialIOExecutor(name='TEST_LED_IO')
    ex.start()
    scope.register_executors(io_executor=ex)
    yield scope
    ex.shutdown(wait=True)


def _run_async(fn, *args, timeout=5, **kwargs):
    """Submit an X_async LED call and block until the io_executor runs it."""
    done = threading.Event()
    fn(*args, callback=lambda *a, **k: done.set(), **kwargs)
    assert done.wait(timeout), 'async LED task did not complete in time'


def _preview(scope_io, ch, mA):
    """Drive the production manual-nav preview path (apply_transition_async)."""
    _run_async(
        scope_io.illumination.apply_transition_async,
        LedTransition.MANUAL_STEP,
        LedTransitionCtx(channel=ch, mA=mA, preview_on=True),
    )


def test_manual_preview_async_skips_already_lit_channel(scope_io):
    """Manual-nav preview on a channel already at the target current emits
    no driver command -- the manual same-color step no longer blinks."""
    scope_io.illumination.led_on(channel=3, mA=200, owner='ui')

    events = []
    scope_io.illumination.add_led_listener(lambda c, e, m, o: events.append((c, e)))
    _preview(scope_io, 3, 200)

    assert events == [], f'already-lit channel was re-commanded (flicker): {events}'
    assert scope_io.illumination.led_enabled(_color(scope_io, 3))


def test_manual_preview_async_turns_off_other_channels(scope_io):
    """Manual-nav preview offs other lit channels and lights the target --
    the manual switch-to-a-new-color step."""
    scope_io.illumination.led_on(channel=0, mA=100, owner='ui')
    _preview(scope_io, 3, 200)

    assert not scope_io.illumination.led_enabled(_color(scope_io, 0))
    assert scope_io.illumination.led_enabled(_color(scope_io, 3))
