# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: disconnect() leaves no LED channel driving current.

Bug
---
`disconnect()` stopped the motors, closed the serial port and cleared the
state caches -- but never sent an off. Nothing downstream covered it: the
driver's `_safety_leds_off()` fires on CONNECT (guarding against a previous
session that died with channels lit), and the atexit hook only runs when the
interpreter exits. So a caller that tore the scope down without exiting the
process -- a GUI Disconnect button, switching scopes mid-session, an SDK
script closing one scope to open another -- got a closed port and a sample
still being illuminated, until the next connect or a power cycle.

Fix
---
The shutoff moves into `disconnect()` itself, above the motor stop so the
ordering the atexit path already relied on is unchanged. It is the same
defense-in-depth argument the motor stop carries in that function: every
teardown path benefits without the caller having to remember. That also
means no LED-specific member has to be public for an app author to shut a
scope down safely -- `disconnect()` is the whole contract.

These drive a real `Lumascope(simulate=True)` and read the SIMULATED BOARD's
channel state, not the API's cache: the cache is what a broken shutoff would
still report correctly while the hardware stayed lit.
"""

from __future__ import annotations

import pytest

from modules.lumascope_api import Lumascope


@pytest.fixture
def scope():
    """A simulated scope with the atexit hook OFF.

    The hook would darken the LEDs at interpreter exit and mask whether
    `disconnect()` did it, which is the whole question here.
    """
    s = Lumascope(simulate=True, register_atexit=False, register_metrics=False)
    yield s
    s.disconnect()


def _lit_channels(driver):
    return {ch for ch, ma in driver._channel_states.items() if ma and ma > 0}


def test_disconnect_darkens_a_lit_channel(scope):
    # The driver reference has to be captured first: disconnect() swaps the
    # scope's slot to a NullLEDBoard, so reading it afterwards would inspect
    # a fresh object that was never lit and pass no matter what.
    driver = scope._led_driver
    scope.illumination.led_on(channel=0, mA=50)
    assert _lit_channels(driver), 'precondition: the channel must actually be lit'

    scope.disconnect()

    assert not _lit_channels(driver), (
        'disconnect() must leave no channel driving current -- a closed port '
        'with a lit sample is the failure this guards'
    )


def test_disconnect_darkens_every_lit_channel(scope):
    # One channel passing does not prove the command was a leds-off rather
    # than an off aimed at whichever channel the code happened to track.
    driver = scope._led_driver
    scope.illumination.led_on(channel=0, mA=30)
    scope.illumination.led_on(channel=1, mA=40)
    assert len(_lit_channels(driver)) >= 2, 'precondition: two channels lit'

    scope.disconnect()

    assert not _lit_channels(driver)


def test_disconnect_completes_when_nothing_is_lit(scope):
    # The shutoff runs unconditionally, so the already-dark case has to be a
    # clean no-op rather than something that raises on the way down.
    driver = scope._led_driver
    assert not _lit_channels(driver)

    assert scope.disconnect() is True
