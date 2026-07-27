# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#749 regression: colours a scope cannot drive must not light other LEDs.

Bug class: an undrivable colour represented in-band. `color2ch` returned a
real channel number (3 on RP2040-family boards, -1 on FX2) for colours the
scope has no LED for, so a luminescence acquire silently lit the brightfield
LED on EL-0940 boards and raised a range error on FX2 -- and the dark-floor
capture guard, keyed on `illumination > 0` alone, would reject a genuinely
dark luminescence frame as a failed capture once the LED correctly stays
dark.

This module grows with the fix stages:
- Stage 1 (here): the three dark-floor sites key on "the scope drives an
  LED for this channel", not illumination alone, and the composite
  luminescence grab gates its led_on on the same predicate.
- Stage 2 adds the driver contract tests (color2ch -> None) and the
  named-colour seam errors.
- Stage 3 adds the task-failure notification wording tests.
"""

from __future__ import annotations

import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from modules import common_utils


class TestLayersWithLedSemantics:
    """The predicate source of truth: which channels drive an LED."""

    def test_luminescence_layers_have_no_led(self):
        with_led = common_utils.get_layers_with_led()
        for lumi_layer in common_utils.get_luminescence_layers():
            assert lumi_layer not in with_led

    def test_transmitted_and_fluorescence_layers_have_leds(self):
        with_led = common_utils.get_layers_with_led()
        for layer in (
            *common_utils.get_transmitted_layers(),
            *common_utils.get_fluorescence_layers(),
        ):
            assert layer in with_led


class TestDarkFloorKeysOnLedDrivability:
    """Seam guards: all three capture sites must key dark-floor rejection
    (and, for the composite loop, the led_on itself) on LED drivability,
    never on illumination alone. A source pin per site -- the behavioral
    seam tests ride the contract stage where the seam is directly drivable.
    """

    def test_protocol_writer_predicate(self):
        src = (REPO / 'modules' / 'protocol_image_writer.py').read_text()
        assert (
            "dark_floor_check=step['Illumination'] > 0\n"
            "                        and step['Color'] in common_utils.get_layers_with_led()"
        ) in src, 'protocol capture must exempt LED-less channels from dark-floor rejection'

    def test_composite_loop_gates_led_and_dark_floor_together(self):
        src = (REPO / 'ui' / 'composite_capture.py').read_text()
        assert (
            'led_driven = layer in common_utils.get_layers_with_led() and illumination > 0'
        ) in src, 'composite loop must compute LED drivability once'
        assert 'dark_floor_check=led_driven' in src, (
            'composite grab must expect a dark frame exactly when no LED was driven'
        )
        assert 'if layer not in common_utils.get_transmitted_layers():' not in src, (
            'the vacuously-true not-transmitted guard must not gate led_on'
        )

    def test_live_capture_predicate(self):
        src = (REPO / 'ui' / 'composite_capture.py').read_text()
        assert (
            'layer in common_utils.get_layers_with_led()\n'
            "            and layer_configs[layer]['illumination_ma'] > 0"
        ) in src, 'manual live capture must exempt LED-less channels from dark-floor rejection'
