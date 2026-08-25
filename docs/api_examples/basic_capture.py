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

from modules.lumascope_api import Lumascope


def main():
    # Create scope in simulate mode -- no hardware required
    scope = Lumascope(simulate=True)
    print('Scope initialized (simulate=True)')

    # Begin the live camera feed (required before capture on every backend).
    scope.imaging.start_streaming()

    # Home before commanding any move. Until an axis has been homed its
    # position is unknown, and a move against an unknown reference frame
    # is refused with AxisStateUnknownError rather than driven blind.
    if not scope.motion.move_home_and_wait('ALL'):
        print('Homing failed -- cannot move safely')
        scope.disconnect()
        return

    # Set LED channel 0 (BF) to 100 mA
    scope.illumination.led_on(channel=0, mA=100)
    print('LED 0 set to 100 mA')

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

    # Disconnect
    scope.disconnect()
    print('Scope disconnected')


if __name__ == '__main__':
    main()
