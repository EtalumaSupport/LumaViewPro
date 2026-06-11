# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the idempotent LED ownership handoff.

Autofocus and protocol stepping both need to make a single channel the only
lit LED. Doing that with a nuclear leds_off followed by led_on blinks a channel
that is already lit at the target current off->on -- a visible flicker at every
autofocus scan boundary and on every same-color protocol step (Z-stack slice).

leds_exclusive and the diff-based restore_led_state leave an already-correct
channel untouched, so re-asserting it does not flicker. LED listeners fire only
when a command actually reaches the driver (a self-skipped no-op does not fire),
so counting listener events is a direct measure of "did the LED blink".
"""

import pytest

from modules.lumascope_api import Lumascope


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    yield s


def _color(scope, ch):
    return scope.illumination.ch2color(ch)


def test_leds_exclusive_skips_already_lit_channel(scope):
    """A channel already on at the target current is not re-commanded."""
    scope.illumination.led_on(channel=3, mA=200, owner='protocol')

    events = []
    scope.illumination.add_led_listener(lambda c, e, m, o: events.append((c, e)))
    scope.illumination.leds_exclusive(channel=3, mA=200, owner='protocol')

    assert events == [], f'already-lit channel was re-commanded (flicker): {events}'
    assert scope.illumination.led_enabled(_color(scope, 3))


def test_leds_exclusive_turns_off_other_channels(scope):
    """Other lit channels are turned off; the target channel ends up lit."""
    scope.illumination.led_on(channel=0, mA=100)
    scope.illumination.leds_exclusive(channel=3, mA=200, owner='protocol')

    assert not scope.illumination.led_enabled(_color(scope, 0))
    assert scope.illumination.led_enabled(_color(scope, 3))


def test_leds_exclusive_lights_a_dark_channel(scope):
    """With nothing lit, leds_exclusive turns the target channel on."""
    scope.illumination.leds_exclusive(channel=3, mA=200, owner='protocol')
    assert scope.illumination.led_enabled(_color(scope, 3))


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
