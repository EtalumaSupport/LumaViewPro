#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Multi-channel fluorescence capture example.

Demonstrates:
- Capturing images across multiple fluorescence channels (Blue, Green, Red)
- Setting per-channel LED illumination and exposure
- Saving individual channel images as TIFF files
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


# Channel configurations: color name, LED current (mA), exposure time (ms)
CHANNELS = [
    {"color": "Blue",  "mA": 50,  "exposure_ms": 200},
    {"color": "Green", "mA": 80,  "exposure_ms": 150},
    {"color": "Red",   "mA": 100, "exposure_ms": 100},
]


def main():
    # Create scope in simulate mode -- no hardware required
    scope = Lumascope(simulate=True)
    print("Scope initialized (simulate=True)")

    # Start camera grabbing
    scope.camera.start_grabbing()

    # Capture each fluorescence channel
    for ch in CHANNELS:
        color = ch["color"]
        print(f"\n--- Channel: {color} ---")

        # Configure LED illumination for this channel
        scope.led_on(channel=color, mA=ch["mA"])
        print(f"  LED on: {ch['mA']} mA")

        # Set exposure time
        scope.set_exposure_time(ch["exposure_ms"])
        print(f"  Exposure: {ch['exposure_ms']} ms")

        # Capture image through the API (returns numpy array)
        image = scope.get_image(force_to_8bit=True)
        if image is False:
            print(f"  ERROR: Failed to capture {color} channel")
            continue

        print(f"  Captured: shape={image.shape}, dtype={image.dtype}")
        print(f"  Pixel stats: min={image.min()}, max={image.max()}, mean={image.mean():.1f}")

        # Turn off LED before switching channels
        scope.led_off(channel=color)

    # Turn off all LEDs
    scope.leds_off()
    print("\nAll LEDs off")

    # Stop camera and disconnect
    scope.camera.stop_grabbing()
    scope.disconnect()
    print("Scope disconnected")

    # NOTE: To save images, you would use scope.save_image() or
    # scope.save_live_image(). These require setting an objective,
    # labware, and stage offset for metadata generation. See the
    # protocol_execution.py example for a more complete workflow.


if __name__ == '__main__':
    main()
