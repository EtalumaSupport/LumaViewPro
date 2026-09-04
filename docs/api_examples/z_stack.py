#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Z-stack capture example.

Demonstrates:
- Moving the Z axis through a range of positions (all in um)
- Capturing an image at each Z slice via scope.imaging.capture_and_wait
- Building a Z-stack for 3D analysis or extended depth of focus

Note: autofocus runs as a RUN through the session's protocol runner
(see LumascopeSkills.md, Running protocols) -- it sweeps Z and settles
on the best focus plane under the run-exclusivity claim. This example
shows manual Z stepping for cases where you want full control over
the Z-stack.
"""

import sys
import pathlib

# Make the repo root importable when run standalone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# This example runs the SAME code path two ways:
#   standalone: python3 docs/api_examples/z_stack.py  (the real installed deps)
#   in-suite:   tests/test_api_examples.py runs main() under the heavy-dep
#               mocks the test conftest installs before collection
# The sys.path line serves the standalone form; in-suite it is a no-op.

from modules.lumascope_api import Lumascope


# Z-stack parameters (all values in micrometers)
Z_START_UM = 4000.0  # Starting Z position (um)
Z_END_UM = 6000.0  # Ending Z position (um)
Z_STEP_UM = 200.0  # Step size between slices (um)

# Illumination settings
LED_COLOR = 'BF'  # Brightfield
LED_MA = 100  # LED current (mA)
EXPOSURE_MS = 50  # Exposure time (ms)


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

    # Configure illumination
    scope.illumination.led_on(channel=LED_COLOR, illumination_ma=LED_MA)
    scope.imaging.set_exposure_ms(EXPOSURE_MS)
    print(f'LED: {LED_COLOR} at {LED_MA} mA, exposure: {EXPOSURE_MS} ms')

    # Calculate the number of slices
    num_slices = int((Z_END_UM - Z_START_UM) / Z_STEP_UM) + 1
    print(f'\nZ-stack: {Z_START_UM} to {Z_END_UM} um, step={Z_STEP_UM} um ({num_slices} slices)')

    # Capture Z-stack
    z_stack_images = []
    z_pos_um = Z_START_UM

    for i in range(num_slices):
        # Move Z to target position (um) and wait for completion
        scope.motion.move_absolute(
            'Z',
            z_pos_um,
            wait_until_complete=True,
        )

        # Read back the actual position (um, from the push-based cache)
        actual_z_um = scope.motion.get_current_position('Z')

        # Capture a frame valid for the current Z + LED + exposure state.
        image = scope.imaging.capture_and_wait(force_to_8bit=True)
        if image is None:
            print(f'  Slice {i:3d}: FAILED at Z={z_pos_um:.1f} um')
            z_pos_um += Z_STEP_UM
            continue

        z_stack_images.append(image)
        print(
            f'  Slice {i:3d}: Z={actual_z_um:.1f} um, shape={image.shape}, mean={image.mean():.1f}'
        )

        z_pos_um += Z_STEP_UM

    print(f'\nCaptured {len(z_stack_images)} / {num_slices} slices')

    # NOTE: To save each slice as a file, import from modules.image_save:
    #   from modules.image_save import save_image
    #   save_image(scope, array=image, save_folder='./zstack',
    #              append=f'_Z{i:03d}', ...)
    # This requires objective, labware, and stage offset first -- what
    # session.configure_scope() applies; see protocol_execution.py.

    # NOTE: for automatic focusing, run autofocus as a run through the
    # session's protocol runner (LumascopeSkills.md, Running protocols).

    # Clean up
    scope.illumination.leds_off()
    scope.disconnect()
    print('Scope disconnected')


if __name__ == '__main__':
    main()
