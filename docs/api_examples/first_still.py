#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A first saved still, on the simulator.

Brings a session up from the user's settings, runs the startup motion,
moves over a well, lights brightfield, and saves one still the way
LumaViewPro's Capture button does. Prints the file and what its record says.

Run it in a LumaViewPro source folder (how, step by step: README.md here):

    python3 docs/api_examples/first_still.py

The same still over REST is ``first_still.sh``.
"""

import pathlib
import sys

# The installation folder, importable when the script is run by its path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from modules.scope_session import ScopeSession


def main():
    session = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)
    try:
        # The startup motion: every axis homed, then the turret. A move
        # before it is refused, because the position is unknown.
        session.run_application_session()
        scope = session.scope
        scope.motion.move_to_well('B2')

        # 5 mA lights the simulator's specimen without clipping; brightfield
        # saturates from about 20 mA at the shipped exposure.
        scope.illumination.led_on('BF', 5)

        # The still is named for the lit LED, and the camera holds that
        # channel's stored gain and exposure for its frame.
        saved = session.manual_capture.capture(false_color_on=False).result(timeout=30)
        scope.illumination.leds_off()

        still = saved[0]
        full_scale = (1 << still.significant_bits) - 1
        print(f'Saved {still.file}')
        print(f'  channel {still.channel}, well {still.well_label}')
        print(f'  exposure {still.frame.exposure_ms} ms, gain {still.frame.gain_db} dB')
        print(f'  lit: {still.frame.dark is False}')
        print(f'  brightest pixel {still.statistics.maximum} of {full_scale}')
        return still
    finally:
        # Finishes what the session is doing, then turns the LEDs off,
        # stops motion and disconnects.
        session.shutdown()


if __name__ == '__main__':
    main()
