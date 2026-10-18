# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated frame shows the LED current and the stage position it was made under.

The simulator rendered a lit field the same at any current, and the same at
any X and Y, so a frame made before an LED or XY change could not be told
from one made after it. The light now scales with the current, and the
field follows the stage.

Then every layer got the same light, floored at half the field's grey
levels and capped at them, so below the floor and above the cap the
exposure, gain and LED sliders changed nothing. Each layer now renders the
light the catalogue gives it, its shipped settings show the field at its
own grey levels, and each pixel saturates on its own, on both simulated
cameras.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.scope_session import ScopeSession
from tests.camera_fakes import grab_a_frame_made_after_now
from tests.settings_fixtures import complete_settings


def test_the_light_scales_with_the_led_current():
    current = [SimulatedCamera.REFERENCE_ILLUMINATION_MA]
    cam = SimulatedCamera(width=64, height=48, illumination_func=lambda: current[0])
    # Short enough that doubled light stays below full scale, where each
    # pixel would saturate on its own.
    cam.set_test_pattern(enabled=True, pattern='focus_target')
    cam.exposure_t(2.0)
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


# ---------------------------------------------------------------------------
# On a simulated scope, each layer's light, exposure and gain move the image
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _scope(model: str):
    """A simulated ``model``, its camera focused where Z stands."""
    session = ScopeSession.create(
        complete_settings(microscope=model), simulate=True, warn_pre_release=False
    )
    try:
        camera = session.scope._camera_driver
        if hasattr(camera, 'set_focal_z'):
            camera.set_focal_z(session.scope.motion.get_current_position('Z'))
        yield session.scope
    finally:
        session.shutdown()


def _shown(scope, layer: str, ma: float, exposure_ms: float, gain_db: float) -> np.ndarray:
    """The frame the camera delivers with only ``layer`` lit at these settings."""
    scope.illumination.leds_off()
    scope.illumination.led_on(layer, ma)
    scope.camera.set_exposure_ms(exposure_ms)
    scope.camera.set_gain_db(gain_db)
    return scope.imaging.capture_and_wait(accept_dark=True)


# Each layer's shipped (mA, ms, dB): what makes it look right on a scope.
_SHIPPED = {'BF': (5.0, 2.0, 1.0), 'Green': (250.0, 100.0, 10.0)}


# The two simulated cameras, and a bright modality and a dim one.
@pytest.mark.parametrize('model', ['LS850', 'LS620'])
@pytest.mark.parametrize('layer', ['BF', 'Green'])
def test_each_slider_moves_the_image_from_the_layer_s_shipped_settings(model, layer):
    ma, ms, db = _SHIPPED[layer]
    with _scope(model) as scope:
        at_shipped = float(_shown(scope, layer, ma / 2, ms / 2, db).mean())
        brighter = {
            'current': float(_shown(scope, layer, ma, ms / 2, db).mean()),
            'exposure': float(_shown(scope, layer, ma / 2, ms, db).mean()),
            'gain': float(_shown(scope, layer, ma / 2, ms / 2, db + 6.0).mean()),
        }
    assert at_shipped > 0, f'{layer} at half its shipped light renders black'
    dimmer = {slider: mean for slider, mean in brighter.items() if mean <= at_shipped * 1.3}
    assert not dimmer, f'raising each slider from {at_shipped:.1f} did not brighten: {brighter}'


# A bright modality and a dim one.
@pytest.mark.parametrize('layer', ['BF', 'Green'])
def test_a_layer_at_its_shipped_settings_shows_the_field_unsaturated(layer):
    with _scope('LS850') as scope:
        frame = _shown(scope, layer, *_SHIPPED[layer])
    assert 0.75 * 255 < frame.max() < 255, (layer, int(frame.min()), int(frame.max()))


def test_more_light_saturates_the_brightest_pixels_first():
    ma, ms, db = _SHIPPED['BF']
    with _scope('LS850') as scope:
        frame = _shown(scope, 'BF', ma, ms * 1.5, db)
    saturated = float((frame == 255).mean())
    assert 0 < saturated < 0.5, f'{saturated:.1%} of the field is at full scale'
