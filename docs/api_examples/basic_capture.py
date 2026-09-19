#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Basic capture example using Lumascope API in simulate mode.

Demonstrates:
- Initializing the scope in simulate mode
- Setting LED illumination via scope.illumination
- Moving the Z axis via scope.motion (positions in micrometers)
- Capturing an image via scope.imaging
"""

import sys
import pathlib

# Make the repo root importable when run standalone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# This example runs the SAME code path two ways:
#   standalone: python3 docs/api_examples/basic_capture.py  (the real installed deps)
#   in-suite:   tests/test_api_examples.py runs main() under the heavy-dep
#               mocks the test conftest installs before collection
# The sys.path line serves the standalone form; in-suite it is a no-op.

from modules.exceptions import ConfigError
from modules.scope_session import ScopeSession


def main():
    # create_headless() is the supported factory for a simulated session: it
    # wires the simulated drivers, configures the scope from settings and
    # releases the camera start gate, so there is no separate bring-up and no
    # start_streaming() call to make here.
    #
    # source_path defaults to the working directory, which must be an LVP
    # installation root. ConfigError is caught and printed rather than left to
    # propagate because an uncaught exception in a process that imports
    # lvp_logger is written to the log file and never to the terminal: this
    # message is the only thing that would tell you what went wrong.
    try:
        session = ScopeSession.create_headless()
    except ConfigError as exc:
        print(f'Could not create a headless session: {exc}')
        print(
            'Run this from a LumaViewPro installation root -- a directory '
            'holding data/settings.json.'
        )
        raise SystemExit(1) from exc

    scope = session.scope
    print('Headless session created (simulate=True)')

    # Home before commanding any move. Until an axis has been homed its
    # position is unknown, and a move against an unknown reference frame
    # is refused with AxisStateUnknownError rather than driven blind.
    if not scope.motion.move_home_and_wait('ALL'):
        print('Homing failed -- cannot move safely')
        session.shutdown()
        return

    # Set the brightfield channel to 100 mA. Channels are named, not
    # numbered: the name is the portable identity, and which number it maps
    # to differs by board (an FX2 board carries four channels, an RP2040
    # board six), so a literal channel number is not portable.
    scope.illumination.led_on(channel='BF', illumination_ma=100)
    print('BF LED set to 100 mA')

    # Move Z axis to 5000 um and wait for the move to complete
    scope.motion.move_absolute('Z', 5000, wait_until_complete=True)

    # Read the target Z position (returns um). Zero serial I/O --
    # the API serves this from the push-based position cache.
    z_target = scope.motion.get_target_position('Z')
    print(f'Z target position: {z_target} um')

    # Capture an image. capture_and_wait drains stale frames and
    # returns a frame valid for the current LED + exposure state. The
    # LED is on, so a frame with no lit pixel is rejected as a capture
    # fault -- the dark-floor expectation is derived from commanded state.
    image = scope.imaging.capture_and_wait(force_to_8bit=True)
    if image is None:
        print('Capture failed')
    else:
        print(f'Captured image: shape={image.shape}, dtype={image.dtype}')
        print(f'  Min={image.min()}, Max={image.max()}, Mean={image.mean():.1f}')

    # Turn off LEDs
    scope.illumination.leds_off()
    print('All LEDs off')

    # shutdown() tears down everything the factory built: the LEDs drain
    # through the io lane while its worker is still alive, motion stops, the
    # scope disconnects, and the consumer threads stop before the lanes they
    # consume. A bare scope.disconnect() would leave the lanes running.
    session.shutdown()
    print('Session shut down')


if __name__ == '__main__':
    main()
