# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated frame shows the LED current and the stage position it was made under.

The simulator rendered a lit field the same at any current, and the same at
any X and Y, so a frame made before an LED or XY change could not be told
from one made after it. The light now scales with the current, and the
field follows the stage.
"""

from __future__ import annotations

import numpy as np

from drivers.simulated_camera import SimulatedCamera
from tests.camera_fakes import grab_a_frame_made_after_now


def test_the_light_scales_with_the_led_current():
    current = [SimulatedCamera.REFERENCE_ILLUMINATION_MA]
    cam = SimulatedCamera(width=64, height=48, illumination_func=lambda: current[0])
    # The focus target follows the light exactly; the specimen is floored.
    cam.set_test_pattern(enabled=True, pattern='focus_target')
    cam.exposure_t(10.0)
    cam.gain(0.0)
    cam.open_and_start()
    try:
        at_reference = float(grab_a_frame_made_after_now(cam).image.mean())
        current[0] = 2 * SimulatedCamera.REFERENCE_ILLUMINATION_MA
        doubled = float(grab_a_frame_made_after_now(cam).image.mean())
        current[0] = 0.0
        unlit = float(grab_a_frame_made_after_now(cam).image.max())
    finally:
        cam.stop_grabbing()

    assert 1.9 < doubled / at_reference < 2.1, (at_reference, doubled)
    assert unlit == 0


def _shift_between(before: np.ndarray, after: np.ndarray, reach: int) -> tuple[int, int]:
    """The (dy, dx) that best carries ``before`` onto ``after``, within ``reach`` pixels."""
    a = before.astype(np.float32)
    b = after.astype(np.float32)
    best = min(
        (float(np.abs(np.roll(a, (dy, dx), axis=(0, 1)) - b).mean()), (dy, dx))
        for dy in range(-reach, reach + 1)
        for dx in range(-reach, reach + 1)
    )
    return best[1]


def test_the_field_follows_the_stage():
    stage = [0.0, 0.0]
    cam = SimulatedCamera(width=96, height=64, xy_position_func=lambda: (stage[0], stage[1]))
    cam.exposure_t(10.0)
    cam.open_and_start()
    try:
        before = grab_a_frame_made_after_now(cam).image
        stage[0] = 20.0 / SimulatedCamera._SPECIMEN_PX_PER_UM  # 20 px of travel in X
        after = grab_a_frame_made_after_now(cam).image
    finally:
        cam.stop_grabbing()

    dy, dx = _shift_between(before, after, reach=24)
    # The field moves against the stage; the specimen's own drift is a pixel.
    assert abs(dy) <= 1 and -21 <= dx <= -19, (dy, dx)
