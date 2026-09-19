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

from modules.exceptions import ConfigError
from modules.scope_session import ScopeSession


# Channel configurations: channel name, LED current (mA), exposure time (ms)
CHANNELS = [
    {'channel': 'Blue', 'illumination_ma': 50, 'exposure_ms': 200},
    {'channel': 'Green', 'illumination_ma': 80, 'exposure_ms': 150},
    {'channel': 'Red', 'illumination_ma': 100, 'exposure_ms': 100},
]


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

    # Capture each fluorescence channel
    for ch in CHANNELS:
        channel = ch['channel']
        print(f'\n--- Channel: {channel} ---')

        # Configure LED illumination for this channel (mA)
        scope.illumination.led_on(channel=ch['channel'], illumination_ma=ch['illumination_ma'])
        print(f'  LED on: {ch["illumination_ma"]} mA')

        # Set exposure time (ms)
        scope.imaging.set_exposure_ms(ch['exposure_ms'])
        print(f'  Exposure: {ch["exposure_ms"]} ms')

        # Capture a frame valid for the current LED + exposure state.
        # This channel's LED is driven, so a frame with no lit pixel is
        # rejected as a capture fault -- derived from commanded state.
        image = scope.imaging.capture_and_wait(force_to_8bit=True)
        if image is None:
            print(f'  ERROR: Failed to capture {channel} channel')
            continue

        print(f'  Captured: shape={image.shape}, dtype={image.dtype}')
        print(f'  Pixel stats: min={image.min()}, max={image.max()}, mean={image.mean():.1f}')

        # Turn off this channel before switching channels
        scope.illumination.led_off(channel=channel)

    # Turn off all LEDs and disconnect
    scope.illumination.leds_off()
    print('\nAll LEDs off')

    # shutdown() tears down everything the factory built: the LEDs drain
    # through the io lane while its worker is still alive, motion stops, the
    # scope disconnects, and the consumer threads stop before the lanes they
    # consume. A bare scope.disconnect() would leave the lanes running.
    session.shutdown()
    print('Session shut down')

    # NOTE: To save images, import from modules.image_save:
    #     from modules.image_save import save_image, save_live_image
    #     save_image(scope, array=image, save_folder='./out', ...)
    # These require an objective, labware, and stage offset for metadata
    # generation -- what session.configure_scope() (or the factories that
    # call it) applies. See protocol_execution.py for a complete workflow.


if __name__ == '__main__':
    main()
