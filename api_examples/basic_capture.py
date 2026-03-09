#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Basic capture example using Lumascope API in simulate mode.

Demonstrates:
- Initializing the scope in simulate mode
- Setting LED illumination
- Moving the Z axis
- Capturing and saving an image
"""

import sys
import pathlib
from unittest.mock import MagicMock

# Add parent directory to path so we can import lumascope_api
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Mock modules not needed for headless simulate mode
_mock_logger = MagicMock()
_mock_lvp_logger = MagicMock()
_mock_lvp_logger.logger = _mock_logger
_mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
_mock_lvp_logger.unpause_thread = MagicMock()
_mock_lvp_logger.pause_thread = MagicMock()

sys.modules.setdefault('lvp_logger', _mock_lvp_logger)
sys.modules.setdefault('userpaths', MagicMock())
sys.modules.setdefault('requests', MagicMock())
sys.modules.setdefault('requests.structures', MagicMock())
sys.modules.setdefault('pypylon', MagicMock())
sys.modules.setdefault('pypylon.pylon', MagicMock())
sys.modules.setdefault('pypylon.genicam', MagicMock())
sys.modules.setdefault('ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak_ipl_extension', MagicMock())
sys.modules.setdefault('ids_peak_ipl', MagicMock())

from lumascope_api import Lumascope


def main():
    # Create scope in simulate mode — no hardware required
    scope = Lumascope(simulate=True)
    print("Scope initialized (simulate=True)")
    print(f"  LED board: {type(scope.led).__name__}")
    print(f"  Motor board: {type(scope.motion).__name__}")
    print(f"  Camera: {type(scope.camera).__name__}")

    # Set LED channel 0 (BF) to 100 mA
    scope.led.led_on(channel=0, mA=100)
    print("\nLED 0 set to 100 mA")

    # Move Z axis (steps as unsigned 32-bit integer)
    scope.motion.move(axis='Z', steps=50000)
    print("Z axis moved by 50000 steps")

    # Read current Z target position (returns um)
    z_pos = scope.motion.target_pos(axis='Z')
    print(f"Z target position: {z_pos} um")

    # Capture an image (start grabbing, grab, access .array)
    scope.camera.start_grabbing()
    success, timestamp = scope.camera.grab()
    image = scope.camera.array
    print(f"\nCaptured image: success={success}, shape={image.shape}, dtype={image.dtype}")
    print(f"  Min={image.min()}, Max={image.max()}, Mean={image.mean():.1f}")
    scope.camera.stop_grabbing()

    # Turn off LEDs
    scope.led.leds_off()
    print("\nAll LEDs off")

    # Disconnect
    scope.disconnect()
    print("Scope disconnected")


if __name__ == '__main__':
    main()
