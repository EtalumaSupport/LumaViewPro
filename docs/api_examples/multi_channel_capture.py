#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Multi-channel fluorescence capture example.

Demonstrates:
- Capturing images across multiple fluorescence channels (Blue, Green, Red)
- Setting per-channel LED illumination (mA) and exposure (ms)
- Reading the captured frame via scope.imaging.capture_and_wait
"""

import sys
import pathlib

# Make the repo root importable when run standalone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# This example runs the SAME code path two ways:
#   standalone: python3 docs/api_examples/multi_channel_capture.py  (the real installed deps)
#   in-suite:   tests/test_api_examples.py runs main() under the heavy-dep
#               mocks the test conftest installs before collection
# The sys.path line serves the standalone form; in-suite it is a no-op.

from modules.lumascope_api import Lumascope


# Channel configurations: color name, LED current (mA), exposure time (ms)
CHANNELS = [
    {'color': 'Blue', 'mA': 50, 'exposure_ms': 200},
    {'color': 'Green', 'mA': 80, 'exposure_ms': 150},
    {'color': 'Red', 'mA': 100, 'exposure_ms': 100},
]


def main():
    # Create scope in simulate mode -- no hardware required
    scope = Lumascope(simulate=True)
    print('Scope initialized (simulate=True)')

    # Begin the live camera feed (required before capture on every backend).
    scope.imaging.start_streaming()

    # Capture each fluorescence channel
    for ch in CHANNELS:
        color = ch['color']
        print(f'\n--- Channel: {color} ---')

        # Configure LED illumination for this channel (mA)
        scope.illumination.led_on(channel=color, mA=ch['mA'])
        print(f'  LED on: {ch["mA"]} mA')

        # Set exposure time (ms)
        scope.imaging.set_exposure_ms(ch['exposure_ms'])
        print(f'  Exposure: {ch["exposure_ms"]} ms')

        # Capture a frame valid for the current LED + exposure state.
        # This channel's LED is driven, so a frame with no lit pixel is
        # rejected as a capture fault -- derived from commanded state.
        image = scope.imaging.capture_and_wait(force_to_8bit=True)
        if image is None:
            print(f'  ERROR: Failed to capture {color} channel')
            continue

        print(f'  Captured: shape={image.shape}, dtype={image.dtype}')
        print(f'  Pixel stats: min={image.min()}, max={image.max()}, mean={image.mean():.1f}')

        # Turn off this channel before switching colors
        scope.illumination.led_off(channel=color)

    # Turn off all LEDs and disconnect
    scope.illumination.leds_off()
    print('\nAll LEDs off')

    scope.disconnect()
    print('Scope disconnected')

    # NOTE: To save images, import from modules.image_save:
    #     from modules.image_save import save_image, save_live_image
    #     save_image(scope, array=image, save_folder='./out', ...)
    # These require setting an objective, labware, and stage offset
    # for metadata generation. See protocol_execution.py for a more
    # complete workflow.


if __name__ == '__main__':
    main()
