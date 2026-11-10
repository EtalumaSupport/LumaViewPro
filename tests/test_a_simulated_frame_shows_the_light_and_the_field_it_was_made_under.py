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

Then the bench measured a camera's move count on a Ronchi ruling, and the
simulator had none: its cells saturated where the bench's Ronchi read in
band, so the measurement could not be run as on the bench. A simulated
scope can now be brought up with a Ronchi ruling at the bench's light.

Then the ruling was the bench's 40 lines/mm only, so a run at another
pitch had nothing to read. A record now gives the ruling its pitch, and
refuses one that is not a ruling's.
"""

from __future__ import annotations

import contextlib

import cv2
import numpy as np
import pytest

from _lumascope.drivers.simulated_camera import SimulatedCamera
from _lumascope.drivers.simulated_specimen import NOMINAL_UM_PER_PX, Slide
from _lumascope.modules.exceptions import ArgumentRefusedError
from _lumascope.modules import common_utils
from _lumascope.modules.channel_record import SIM_LIGHT_MAGNIFICATION
from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from _lumascope.modules.scope_session import ScopeSession
from tests.camera_fakes import grab_a_frame_made_after_now
from tests.scope_fakes import build_scope
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED1


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
    # The field moves against the stage. The specimen's own drift puts it a
    # pixel off centre in one of four directions (DRIFT_STEPS), so two frames
    # differ by up to two pixels along an axis, whichever frames they are.
    assert abs(dy) <= 2 and -22 <= dx <= -18, (dy, dx)


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
def _session(
    model: str,
    homed: bool = False,
    slide: Slide = Slide.CELLS,
    ronchi_lines_per_mm: float | None = None,
):
    """A simulated ``model``'s session with ``slide`` on its stage, a ruling at ``ronchi_lines_per_mm``, homed when a test moves the stage, its camera focused where Z stands."""
    session = ScopeSession.create(
        complete_settings(microscope=model),
        simulate=True,
        simulated=SimulatedScope(slide=slide, ronchi_lines_per_mm=ronchi_lines_per_mm),
        warn_pre_release=False,
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
    """A brightfield frame as bright as the shipped settings make it on the 20x, as floats.

    Brightfield falls as the square of the magnification, so the shipped
    light clips the field on a 4x or a 10x; the exposure is set for the
    objective, as a person sets it, so the picture is the sample's. The
    exposure, not the current: the LED is set in whole mA.
    """
    ma, ms, db = _SHIPPED['BF']
    magnification = scope.runtime_state.get_current_objective().magnification
    ms *= (magnification / SIM_LIGHT_MAGNIFICATION) ** 2
    return _shown(scope, 'BF', ma, ms, db).astype(np.float32)


def _image_shift(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """The (rows, columns) the content of ``before`` moved by to land in ``after``."""
    (dx, dy), _ = cv2.phaseCorrelate(before, after)
    return dy, dx


def test_a_stage_move_shifts_the_image_by_the_distance_over_the_pixel_size_against_the_stage():
    """A move of D um shifts the image D over the saved pixel size, toward lower column and higher row.

    Row 0 is the top of the picture, so a move to higher Y carries the
    sample toward its bottom. So two tiles whose fields overlap on the stage
    overlap in their images where the stitcher's placement puts them, and
    the direction is the one its placement expects.
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
    assert abs(rows - 450) <= 2.5 and abs(cols) <= 2.5, (rows, cols)


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


def test_brightfield_falls_as_the_square_of_the_magnification_and_fluorescence_does_not():
    """BF at its shipped settings is a quarter as bright on a 40x as on the 20x, a ninth on a 60x.

    As a clear field does on the LS850T (/3.84 and /3.99 per doubling), and
    as the catalogue's BF power says; a layer the bench has not read, here a
    fluorescence one, renders the same through each. Each frame is lit after
    its objective is chosen, so the LED's write times the light it shows.
    """
    means = {}
    with _session('LS850') as session:
        scope = session.scope
        for objective in ('20x Oly', '40x w/collar', '60x w/collar'):
            session.select_objective(objective)
            for layer in ('BF', LED1):
                means[objective, layer] = float(_shown(scope, layer, *_SHIPPED[layer]).mean())

    at_20x = means['20x Oly', 'BF']
    # The sample is drawn at each objective's scale, so its mean moves a few
    # per cent with the scale alone.
    for objective, fraction in (('40x w/collar', 1 / 4), ('60x w/collar', 1 / 9)):
        assert 0.85 < means[objective, 'BF'] / (at_20x * fraction) < 1.15, means
        assert 0.85 < means[objective, LED1] / means['20x Oly', LED1] < 1.15, means


@pytest.mark.slow
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
    return scope.imaging.capture_still(accept_dark=True).image


# Each layer's shipped (mA, ms, dB): what makes it look right on a scope.
_SHIPPED = {'BF': (5.0, 2.0, 1.0), LED1: (250.0, 100.0, 10.0)}


# The two simulated cameras, and a bright modality and a dim one.
@pytest.mark.parametrize('model', ['LS850', 'LS620'])
@pytest.mark.parametrize('layer', ['BF', LED1])
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
@pytest.mark.parametrize('layer', ['BF', LED1])
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


def test_the_ronchi_reads_at_the_move_count_s_start_as_the_bench_s_did():
    """At the transition stage's start (brightfield 25 mA, 25 ms, 0 dB) the Ronchi is in its 20 to 100 DN start band on the 20x and the 40x, and above it on the 10x.

    As the LS850T's Ronchi read on the bench: 72.7 on the 20x, 31.5 on the
    40x, 116.5 on the 10x, which the stage refuses.
    """
    means = {}
    with _session('LS850', slide=Slide.RONCHI) as session:
        for objective in ('10x Oly', '20x Oly', '40x w/collar'):
            session.select_objective(objective)
            means[objective] = float(_shown(session.scope, 'BF', 25.0, 25.0, 0.0).mean())

    assert 20 < means['20x Oly'] < 100, means
    assert 20 < means['40x w/collar'] < 100, means
    assert means['10x Oly'] > 100, means


# Each simulated camera images the stage's slide its own way: the stand-in
# camera and the FX2's sensor.
@pytest.mark.parametrize('model', ['LS850', 'LS620'])
def test_the_ronchi_shows_its_25_um_bars_along_the_rows(model):
    with _session(model, slide=Slide.RONCHI) as session:
        frame, um_per_px = _ruling_on_the_20x(session)

    across_each_row = float(np.ptp(frame, axis=1).max())
    assert across_each_row <= 1, f'a row varies {across_each_row} DN across its columns'
    period_um = _bar_period_um(frame, um_per_px)
    # The spectrum's bins are a few per cent apart at a 20x frame's height.
    assert 0.95 < period_um / 25.0 < 1.05, f'the bars repeat every {period_um:.1f} um'


def test_a_ruling_given_its_pitch_shows_its_bars_at_that_pitch():
    """A 100 lines/mm ruling, given on the record, repeats every 10 um where the bench's 40 lines/mm repeats every 25."""
    with _session('LS850', slide=Slide.RONCHI, ronchi_lines_per_mm=100.0) as session:
        period_um = _bar_period_um(*_ruling_on_the_20x(session))

    assert 0.95 < period_um / 10.0 < 1.05, f'the bars repeat every {period_um:.1f} um'


def test_a_pitch_is_refused_unless_it_is_a_ruling_s_positive_pitch():
    """A pitch of 0, a negative, a NaN, an infinity, or any pitch with the cells on the stage: no record is built."""
    given = [
        (Slide.RONCHI, 0.0, 'fault_bound'),
        (Slide.RONCHI, -40.0, 'fault_bound'),
        (Slide.RONCHI, float('nan'), 'not_a_number'),
        (Slide.RONCHI, float('inf'), 'not_a_number'),
        (Slide.CELLS, 40.0, 'fault_bound'),
    ]
    wrong = []
    for slide, pitch, reason in given:
        try:
            SimulatedScope(slide=slide, ronchi_lines_per_mm=pitch)
        except ArgumentRefusedError as refused:
            if (refused.reason, refused.argument) != (reason, 'ronchi_lines_per_mm'):
                wrong.append((slide, pitch, refused.reason, refused.argument))
        else:
            wrong.append((slide, pitch, 'built'))
    assert not wrong, wrong


def _ruling_on_the_20x(session) -> tuple[np.ndarray, float]:
    """The session's brightfield frame of its ruling through the 20x, and the frame's pixel size in um."""
    session.select_objective('20x Oly')
    frame = _shown(session.scope, 'BF', 25.0, 25.0, 0.0).astype(np.float64)
    return frame, _pixel_size(session.scope)


def _bar_period_um(frame: np.ndarray, um_per_px: float) -> float:
    """How often the bars along ``frame``'s rows repeat, in um, from the rows' strongest frequency."""
    rows = frame.mean(axis=1)
    spectrum = np.abs(np.fft.rfft(rows - rows.mean()))
    return len(rows) / (int(np.argmax(spectrum[1:])) + 1) * um_per_px


def test_a_real_scope_refuses_a_slide():
    with pytest.raises(ArgumentRefusedError) as refused:
        build_scope(simulate=False, sim_slide=Slide.RONCHI)
    assert (refused.value.reason, refused.value.argument) == ('needs_simulated_scope', 'sim_slide')
