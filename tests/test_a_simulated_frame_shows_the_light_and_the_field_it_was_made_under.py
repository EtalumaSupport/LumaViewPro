# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated frame shows the LED current and the stage position it was made under.

The simulator rendered a lit field the same at any current, and the same at
any X and Y, so a frame made before an LED or XY change could not be told
from one made after it. The light now scales with the current, and the
field follows the stage.

Then the field followed the stage at one fixed scale, the same through
every objective, and wrapped every frame width, so a stitch on the
simulator had no known answer. The sample now lies on the stage: a move
shifts the image by the distance over the pixel size every saved image
carries, an objective change rescales it, it never repeats, and both
simulated cameras image the same sample.

Then every layer got the same light, floored at half the field's grey
levels and capped at them, so below the floor and above the cap the
exposure, gain and LED sliders changed nothing. Each layer now renders the
light the catalogue gives it, its shipped settings show the field at its
own grey levels, and each pixel saturates on its own, on both simulated
cameras.
"""

from __future__ import annotations

import contextlib

import cv2
import numpy as np
import pytest

from drivers.simulated_camera import SimulatedCamera
from drivers.simulated_specimen import NOMINAL_UM_PER_PX
from modules import common_utils
from modules.scope_session import ScopeSession
from tests.camera_fakes import grab_a_frame_made_after_now
from tests.settings_fixtures import complete_settings


def test_the_light_scales_with_the_led_current():
    cam = SimulatedCamera(width=64, height=48)
    cam.set_illumination(SimulatedCamera.REFERENCE_ILLUMINATION_MA)
    # Short enough that doubled light stays below full scale, where each
    # pixel would saturate on its own.
    cam.set_test_pattern(enabled=True, pattern='focus_target')
    cam.exposure_t(2.0)
    cam.gain(0.0)
    cam.open_and_start()
    try:
        at_reference = float(grab_a_frame_made_after_now(cam).image.mean())
        cam.set_illumination(2 * SimulatedCamera.REFERENCE_ILLUMINATION_MA)
        doubled = float(grab_a_frame_made_after_now(cam).image.mean())
        cam.set_illumination(0.0)
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
    # Big enough to hold a few cells: a sensor smaller than a cell sees
    # only the medium, which a shift barely changes.
    cam = SimulatedCamera(width=192, height=128)
    cam.set_stage_xy(0.0, 0.0)
    cam.exposure_t(10.0)
    cam.open_and_start()
    try:
        before = grab_a_frame_made_after_now(cam).image
        # 20 px of travel in X at the scale of a camera no objective is known to.
        cam.set_stage_xy(20.0 * NOMINAL_UM_PER_PX, 0.0)
        after = grab_a_frame_made_after_now(cam).image
    finally:
        cam.stop_grabbing()

    dy, dx = _shift_between(before, after, reach=24)
    # The field moves against the stage; the specimen's own drift is a pixel.
    assert abs(dy) <= 1 and -21 <= dx <= -19, (dy, dx)


# ---------------------------------------------------------------------------
# On a simulated scope, each layer's light, exposure and gain move the image
# ---------------------------------------------------------------------------


def test_one_frame_width_of_travel_does_not_bring_the_same_field_back():
    """The sample does not repeat: a field one frame away is other cells.

    One frame width at the nominal scale is where a field that wrapped at
    the sensor's edge showed the same cells again, which a stitch or a
    likeness judge reads as a match one period away.
    """
    cam = SimulatedCamera(width=192, height=128)
    cam.set_stage_xy(0.0, 0.0)
    cam.exposure_t(10.0)
    cam.open_and_start()
    try:
        here = grab_a_frame_made_after_now(cam).image
        cam.set_stage_xy(192 * NOMINAL_UM_PER_PX, 0.0)
        a_frame_away = grab_a_frame_made_after_now(cam).image
    finally:
        cam.stop_grabbing()

    # Near zero for the same field, a pixel of drift apart; about 24 grey
    # levels for unrelated fields of this sample.
    assert _least_difference(here, a_frame_away, reach=2) > 10.0


def test_a_frame_after_a_move_is_drawn_where_the_stage_went_in_focus_or_not():
    """No frame after a move is served from the old position's cut or blur."""

    def camera(x_um: float):
        cam = SimulatedCamera(width=192, height=128, depth_of_field_func=lambda: 4.0)
        cam.set_stage_xy(x_um, 0.0)
        cam.set_focal_z(5000.0)
        cam.exposure_t(10.0)
        cam.open_and_start()
        return cam

    for z in (5000.0, 5006.0):  # in focus, then a depth and a half of field off
        moved = camera(0.0)
        moved.set_z_position(z)
        try:
            for _ in range(4):  # every drift step's frame cut and blurred here
                grab_a_frame_made_after_now(moved)
            moved.set_stage_xy(300.0, 0.0)
            after = grab_a_frame_made_after_now(moved).image
        finally:
            moved.stop_grabbing()
        fresh = camera(300.0)
        fresh.set_z_position(z)
        try:
            there = grab_a_frame_made_after_now(fresh).image
        finally:
            fresh.stop_grabbing()
        assert _least_difference(after, there, reach=2) < 0.5, z


def _least_difference(a: np.ndarray, b: np.ndarray, reach: int) -> float:
    """The mean difference of ``a`` and ``b`` at the shift within ``reach`` pixels that best aligns them."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    h, w = a.shape
    return min(
        float(
            np.abs(
                a[reach + dy : h - reach + dy, reach + dx : w - reach + dx]
                - b[reach : h - reach, reach : w - reach]
            ).mean()
        )
        for dy in range(-reach, reach + 1)
        for dx in range(-reach, reach + 1)
    )


@contextlib.contextmanager
def _session(model: str, homed: bool = False):
    """A simulated ``model``'s session, homed when a test moves the stage, its camera focused where Z stands."""
    session = ScopeSession.create(
        complete_settings(microscope=model), simulate=True, warn_pre_release=False
    )
    try:
        if homed:
            session.scope.motion.home()
        camera = session.scope._camera_driver
        if hasattr(camera, 'set_focal_z'):
            camera.set_focal_z(session.scope.motion.get_current_position('Z'))
        yield session
    finally:
        session.shutdown()


@contextlib.contextmanager
def _scope(model: str):
    """A simulated ``model``, its camera focused where Z stands."""
    with _session(model) as session:
        yield session.scope


def _pixel_size(scope) -> float:
    """The um per pixel every image this scope saves now carries."""
    objective = scope.objectives[scope.runtime_state.get_current_objective_id()]
    return common_utils.get_pixel_size(
        objective.focal_length_mm,
        scope.camera.read_binning(),
        capabilities=scope.capabilities,
    )


def _lit_frame(scope) -> np.ndarray:
    """A frame with the brightfield lit at its shipped settings, as floats."""
    return _shown(scope, 'BF', *_SHIPPED['BF']).astype(np.float32)


def _image_shift(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """The (rows, columns) the content of ``before`` moved by to land in ``after``."""
    (dx, dy), _ = cv2.phaseCorrelate(before, after)
    return dy, dx


def test_a_stage_move_shifts_the_image_by_the_distance_over_the_pixel_size_against_the_stage():
    """A move of D um shifts the image D over the saved pixel size, toward lower column and row.

    So two tiles whose fields overlap on the stage overlap in their images
    where the stitcher's placement puts them, and the direction is the one
    its placement and the save's row flip expect.
    """
    with _session('LS850', homed=True) as session:
        scope = session.scope
        session.select_objective('10x Oly')
        um_per_px = _pixel_size(scope)
        x, y = (scope.motion.get_current_position(axis) for axis in ('X', 'Y'))
        here = _lit_frame(scope)
        # About a third of a field each way, so the frames overlap.
        scope.motion.move_xy(x_um=x + 600 * um_per_px)
        right = _lit_frame(scope)
        scope.motion.move_xy(x_um=x)
        scope.motion.move_xy(y_um=y + 450 * um_per_px)
        down = _lit_frame(scope)

    # Up to two pixels either way is the drift: each frame stands a pixel
    # off centre, and two frames can stand on opposite sides.
    rows, cols = _image_shift(here, right)
    assert abs(rows) <= 2.5 and abs(cols + 600) <= 2.5, (rows, cols)
    rows, cols = _image_shift(here, down)
    assert abs(rows + 450) <= 2.5 and abs(cols) <= 2.5, (rows, cols)


def _centre(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape
    return frame[(h - size) // 2 : (h + size) // 2, (w - size) // 2 : (w + size) // 2]


def _likeness(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised correlation of two same-sized frames: 1 the same picture, about 0 unrelated."""
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float((a * b).mean())


def _seen_at(frame: np.ndarray, um_per_px: float, at_um_per_px: float, size: int) -> np.ndarray:
    """The centre of ``frame``, ``size`` pixels square at ``at_um_per_px``."""
    zoom = um_per_px / at_um_per_px
    scaled = cv2.resize(frame, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
    return _centre(scaled, size)


def test_an_objective_change_rescales_the_image_by_the_ratio_of_the_pixel_sizes():
    with _session('LS850') as session:
        scope = session.scope
        session.select_objective('4x Oly')
        wide, wide_scale = _lit_frame(scope), _pixel_size(scope)
        session.select_objective('10x Oly')
        close, close_scale = _lit_frame(scope), _pixel_size(scope)

    # The 10x frame shrunk to the 4x scale is the 4x frame's centre.
    size = 600
    shrunk = _seen_at(close, close_scale, wide_scale, size)
    assert _likeness(shrunk, _centre(wide, size)) > 0.8


def test_the_classic_and_the_ls850_image_one_sample():
    """A simulated LS720 (the FX2 sensor) and LS850 at one position and objective see the same cells."""
    frames = {}
    for model in ('LS850', 'LS720'):
        with _session(model, homed=True) as session:
            scope = session.scope
            session.select_objective('10x Oly')
            scope.motion.move_xy(x_um=20000.0)
            scope.motion.move_xy(y_um=15000.0)
            frames[model] = (_lit_frame(scope), _pixel_size(scope))

    (ls850, ls850_scale), (ls720, ls720_scale) = frames['LS850'], frames['LS720']
    size = 600
    assert _likeness(_seen_at(ls720, ls720_scale, ls850_scale, size), _centre(ls850, size)) > 0.8


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
