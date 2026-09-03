# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""save_live_image's off-after-capture promise holds when the capture raises.

Nothing below the host extinguishes an LED on its own -- no firmware
watchdog, no driver auto-off -- so a caller that passed
``turn_off_all_leds_after`` is relying on this function to end the
illumination. If the promise held only on the success path, the failure
that most needs it would be the one that skipped it, and the sample sits
under a lit LED with no owner left to turn it off.

The lit-before-fault assertion is load-bearing: an extinguish test that
never proves the LED was lit passes just as green over a fix that
silently refuses the on-command as over a working one.
"""

from unittest.mock import patch

import pytest

from modules import image_save
from modules.lumascope_api import Lumascope


LAYER = 'Blue'
ILLUMINATION_MA = 10.0


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    return s


def _lit(scope):
    return scope.illumination.get_led_state(LAYER)['enabled']


def test_save_live_image_off_flag_holds_on_capture_raise(scope, tmp_path):
    scope.illumination._led_on_impl(LAYER, ILLUMINATION_MA)
    assert _lit(scope), 'precondition: the LED must be lit before the fault'

    with (
        patch.object(scope.imaging, '_capture_and_wait_impl', side_effect=RuntimeError('boom')),
        pytest.raises(RuntimeError),
    ):
        image_save.save_live_image(
            scope,
            save_folder=tmp_path,
            file_root='ext_',
            turn_off_all_leds_after=True,
            channel=LAYER,
            false_color_on=False,
            save_encoding='raw',
        )

    assert not _lit(scope), (
        'turn_off_all_leds_after promises the off; it must hold when the capture raises'
    )
