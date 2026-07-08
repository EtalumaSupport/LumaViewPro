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
from unittest.mock import MagicMock

# Add parent directory to path so we can import lumascope_api
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Mock modules not needed for headless simulate mode
_mock_logger = MagicMock()
_mock_lvp_logger = MagicMock()
_mock_lvp_logger.logger = _mock_logger
_mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
_mock_lvp_logger.unpause_thread = MagicMock()
_mock_lvp_logger.pause_thread = MagicMock()

sys.modules.setdefault('lvp_logger', _mock_lvp_logger)
sys.modules.setdefault('platformdirs', MagicMock())
sys.modules.setdefault('requests', MagicMock())
sys.modules.setdefault('requests.structures', MagicMock())
sys.modules.setdefault('pypylon', MagicMock())
sys.modules.setdefault('pypylon.pylon', MagicMock())
sys.modules.setdefault('pypylon.genicam', MagicMock())
sys.modules.setdefault('ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak_ipl_extension', MagicMock())
sys.modules.setdefault('ids_peak_ipl', MagicMock())

from modules.lumascope_api import Lumascope


def main():
    # Create scope in simulate mode -- no hardware required
    scope = Lumascope(simulate=True)
    print('Scope initialized (simulate=True)')

    # Begin the live camera feed (required before capture on every backend).
    scope.imaging.start_streaming()

    # Set LED channel 0 (BF) to 100 mA
    scope.illumination.led_on(channel=0, mA=100)
    print('LED 0 set to 100 mA')

    # Move Z axis to 5000 um and wait for the move to complete
    scope.motion.move_absolute_position('Z', 5000, wait_until_complete=True)

    # Read the target Z position (returns um). Zero serial I/O --
    # the API serves this from the push-based position cache.
    z_target = scope.motion.get_target_position('Z')
    print(f'Z target position: {z_target} um')

    # Capture an image. capture_and_wait drains stale frames and
    # returns a frame valid for the current LED + exposure state.
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
