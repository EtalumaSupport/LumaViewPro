#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Z-stack capture example.

Demonstrates:
- Moving the Z axis through a range of positions
- Capturing an image at each Z slice
- Building a Z-stack for 3D analysis or extended depth of focus

Note: The Lumascope API also provides a built-in autofocus method
(scope.autofocus()) that sweeps Z and finds the best focus plane
automatically. This example shows manual Z stepping for cases where
you want full control over the Z-stack.
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


# Z-stack parameters (all values in micrometers)
Z_START = 4000.0    # Starting Z position (um)
Z_END = 6000.0      # Ending Z position (um)
Z_STEP = 200.0      # Step size between slices (um)

# Illumination settings
LED_COLOR = "BF"    # Brightfield
LED_MA = 100        # LED current (mA)
EXPOSURE_MS = 50    # Exposure time (ms)


def main():
    # Create scope in simulate mode -- no hardware required
    scope = Lumascope(simulate=True)
    print("Scope initialized (simulate=True)")

    # Configure illumination
    scope.led_on(channel=LED_COLOR, mA=LED_MA)
    scope.set_exposure_time(EXPOSURE_MS)
    print(f"LED: {LED_COLOR} at {LED_MA} mA, exposure: {EXPOSURE_MS} ms")

    # Start camera
    scope.camera.start_grabbing()

    # Calculate the number of slices
    num_slices = int((Z_END - Z_START) / Z_STEP) + 1
    print(f"\nZ-stack: {Z_START} to {Z_END} um, step={Z_STEP} um ({num_slices} slices)")

    # Capture Z-stack
    z_stack_images = []
    z_pos = Z_START

    for i in range(num_slices):
        # Move Z to target position and wait for completion
        scope.move_absolute_position('Z', pos=z_pos, wait_until_complete=True)

        # Read back the actual position
        actual_z = scope.get_current_position(axis='Z')

        # Capture image at this Z position
        image = scope.get_image(force_to_8bit=True)
        if image is False:
            print(f"  Slice {i:3d}: FAILED at Z={z_pos:.1f} um")
            z_pos += Z_STEP
            continue

        z_stack_images.append(image)
        print(f"  Slice {i:3d}: Z={actual_z:.1f} um, "
              f"shape={image.shape}, mean={image.mean():.1f}")

        z_pos += Z_STEP

    print(f"\nCaptured {len(z_stack_images)} / {num_slices} slices")

    # NOTE: To save each slice as a file, you would call:
    #   scope.save_image(image, save_folder='./zstack', append=f'_Z{i:03d}', ...)
    # This requires setting objective, labware, and stage offset first.

    # NOTE: For autofocus, you can use the built-in method:
    #   scope.autofocus(AF_min=10, AF_max=100, AF_range=500)
    # This automatically sweeps Z and moves to the best focus position.

    # Clean up
    scope.leds_off()
    scope.camera.stop_grabbing()
    scope.disconnect()
    print("Scope disconnected")


if __name__ == '__main__':
    main()
