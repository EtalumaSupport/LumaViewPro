# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression for #695: a live LED apply must not turn off an AF-owned channel.

Bug
---
Clicking the AF button defocuses the exposure field, whose apply chain
(exp_text -> apply_settings -> update_led_state -> set_led_state) issued a
led_off on the channel autofocus was using -- about 50 ms after autofocus
turned it on. Autofocus then scanned dark frames yet reported success (the
focus score collapsed).

Fix
---
The protection is structural, not a UI guard. Autofocus holds the LED
ownership lease for the duration of a scan, and led_on/led_off refuse any
write whose owner is not the lease holder. A bare UI led_off (empty owner)
arriving mid-scan is therefore rejected at the API and the AF channel stays
lit. The former update_led_state is_focusing early-return duplicated this
refusal and has been retired -- the lease is the single structural guard.

Test approach
-------------
Real-path API test (no Kivy widget needed): autofocus takes the lease and
lights its channel; a live UI led_off is refused; the channel stays lit.
This fails if the lease enforcement that replaced the guard is removed.
"""

from __future__ import annotations

import pytest

from modules.lumascope_api import Lumascope


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    yield s


def _lit(scope, ch):
    color = scope.illumination.ch2color(ch)
    return scope.illumination.led_enabled(color)


def test_live_ui_off_cannot_dark_an_af_owned_channel(scope):
    """An unleased UI led_off must not turn off the channel autofocus owns."""
    scope.illumination.acquire_led_lease('autofocus')
    scope.illumination.led_on(channel=3, mA=200, owner='autofocus')
    assert _lit(scope, 3)

    # The #695 shape: the AF button click defocuses the exposure field, whose
    # apply chain issues a bare UI led_off. While AF holds the lease this is
    # refused, so the AF channel stays lit and the scan does not go dark.
    scope.illumination.led_off(channel=3, owner='')
    assert _lit(scope, 3), 'live UI off darkened an AF-owned channel (#695 regression)'
