# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated scope brought up with an image slide shows each lit channel's own picture.

The manual's figures of the live view showed the simulator's synthetic
cells, the same grey scene in every channel, so a figure of a fluorescence
channel looked like brightfield. A simulated scope can now be brought up
with a folder of JPGs, one per channel, named for the channel's key: a lit
channel shows its own picture filling the frame, upright, and on the
stand-in camera at its own levels at the channel's shipped settings on the
20x; the settings and the focus act
on it as on the cells; a channel with no picture shows black; and a folder
the cameras could not show is refused as the record is built.
"""

from __future__ import annotations

import contextlib
import pathlib

import cv2
import numpy as np
import pytest

from _lumascope.drivers.simulated_specimen import Slide
from _lumascope.modules.exceptions import ArgumentRefusedError
from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED1, LED2

# Each channel's shipped (mA, ms, dB), which show its picture at its own levels on the 20x.
_SHIPPED = {'BF': (5.0, 2.0, 1.0), LED1: (250.0, 100.0, 10.0)}

# Two pictures each channel's frame is told apart by: large blocks, so a
# JPG's rounding and the scaling to the frame keep them, and neither is
# the same upside down.
_H, _W = 120, 160


def _bf_picture() -> np.ndarray:
    picture = np.full((_H, _W), 60, dtype=np.uint8)
    picture[: _H // 3, : _W // 2] = 180
    picture[_H // 2 :, _W * 3 // 4 :] = 120
    return picture


def _led1_picture() -> np.ndarray:
    columns = np.linspace(30, 150, _W, dtype=np.float32)
    picture = np.repeat(columns[None, :], _H, axis=0)
    picture[_H * 2 // 3 :, :] = 20
    return picture.astype(np.uint8)


def _write_set(folder: pathlib.Path, pictures: dict[str, np.ndarray]) -> str:
    folder.mkdir()
    for name, picture in pictures.items():
        assert cv2.imwrite(str(folder / f'{name}.jpg'), picture, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(folder)


def _covering(picture: np.ndarray, h: int, w: int) -> np.ndarray:
    """``picture`` as it covers an ``h`` x ``w`` frame: its centre cut to the frame's shape, then scaled."""
    ph, pw = picture.shape
    if pw / ph > w / h:
        keep = round(ph * w / h)
        picture = picture[:, (pw - keep) // 2 : (pw - keep) // 2 + keep]
    else:
        keep = round(pw * h / w)
        picture = picture[(ph - keep) // 2 : (ph - keep) // 2 + keep, :]
    return cv2.resize(picture, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def _likeness(a: np.ndarray, b: np.ndarray) -> float:
    """The correlation of two frames' broad shapes."""
    small = [
        cv2.resize(x.astype(np.float32), (64, 48), interpolation=cv2.INTER_AREA) for x in (a, b)
    ]
    return float(np.corrcoef(small[0].ravel(), small[1].ravel())[0, 1])


@contextlib.contextmanager
def _session(model: str, folder: str):
    """A simulated ``model`` with the pictures in ``folder`` on its stage, on the 20x, focused."""
    session = ScopeSession.create(
        complete_settings(microscope=model),
        simulate=True,
        simulated=SimulatedScope(slide=Slide.IMAGES, slide_images=folder),
        warn_pre_release=False,
    )
    try:
        session.select_objective('20x Oly')
        camera = session.scope._camera_driver
        if hasattr(camera, 'set_focal_z'):
            camera.set_focal_z(session.scope.motion.get_current_position('Z'))
        yield session.scope
    finally:
        session.shutdown()


def _shown(scope, lit: dict[str, float], exposure_ms: float, gain_db: float) -> np.ndarray:
    """The frame the camera delivers with ``lit`` channels lit at their mA, at these settings."""
    scope.illumination.leds_off()
    for layer, ma in lit.items():
        scope.illumination.led_on(layer, ma)
    scope.camera.set_exposure_ms(exposure_ms)
    scope.camera.set_gain_db(gain_db)
    return scope.imaging.capture_still(accept_dark=True).image.astype(np.float32)


@pytest.fixture
def pictures(tmp_path) -> str:
    return _write_set(tmp_path / 'set', {'BF': _bf_picture(), LED1: _led1_picture()})


def _frames_at_shipped(model: str, folder: str) -> dict[str, np.ndarray]:
    with _session(model, folder) as scope:
        return {
            layer: _shown(scope, {layer: ma}, ms, db) for layer, (ma, ms, db) in _SHIPPED.items()
        }


# The two simulated cameras: the stand-in and the simulated FX2.
@pytest.mark.parametrize('model', ['LS850', 'LS720'])
def test_each_lit_channel_shows_its_own_picture_upright_filling_the_frame(model, pictures):
    frames = _frames_at_shipped(model, pictures)
    for layer, own, other in (
        ('BF', _bf_picture(), _led1_picture()),
        (LED1, _led1_picture(), _bf_picture()),
    ):
        frame = frames[layer]
        expected = _covering(own, *frame.shape)
        assert _likeness(frame, expected) > 0.95, layer
        assert _likeness(frame, _covering(other, *frame.shape)) < 0.5, layer
        assert _likeness(frame, expected[::-1]) < _likeness(frame, expected) - 0.3, (
            f'{layer} upside down'
        )


def test_a_channel_at_its_shipped_settings_on_the_20x_shows_its_picture_at_its_own_levels(pictures):
    # The stand-in camera alone: the simulated FX2 renders the cells brighter
    # than the stand-in at the same settings too, so its levels are not the file's.
    frames = _frames_at_shipped('LS850', pictures)
    for layer, own in (('BF', _bf_picture()), (LED1, _led1_picture())):
        frame = frames[layer]
        assert frame.mean() == pytest.approx(_covering(own, *frame.shape).mean(), rel=0.05), layer


@pytest.mark.parametrize('model', ['LS850', 'LS720'])
def test_an_unlit_scope_and_a_lit_channel_with_no_picture_show_black(model, pictures):
    with _session(model, pictures) as scope:
        unlit = _shown(scope, {}, *_SHIPPED[LED1][1:])
        no_picture = _shown(scope, {LED2: _SHIPPED[LED1][0]}, *_SHIPPED[LED1][1:])
    assert unlit.max() == 0
    assert no_picture.max() == 0


def test_two_lit_channels_show_their_pictures_summed_each_by_its_own_light(pictures):
    # At these settings BF's 1 mA shows its picture near its own levels and
    # LED1's 500 mA a small share of its own: each by its own light.
    ms, db = 10.0, 1.0
    with _session('LS850', pictures) as scope:
        bf = _shown(scope, {'BF': 1.0}, ms, db)
        led1 = _shown(scope, {LED1: 500.0}, ms, db)
        both = _shown(scope, {'BF': 1.0, LED1: 500.0}, ms, db)
    assert 0 < led1.mean() < 0.2 * bf.mean()
    assert both.mean() == pytest.approx(bf.mean() + led1.mean(), rel=0.02)


def test_a_colour_picture_shows_at_its_brightest_colour(tmp_path):
    grey = _led1_picture()
    blue = np.zeros((_H, _W, 3), dtype=np.uint8)
    blue[:, :, 0] = grey
    folder = _write_set(tmp_path / 'blue', {LED1: blue})
    ma, ms, db = _SHIPPED[LED1]
    with _session('LS850', folder) as scope:
        frame = _shown(scope, {LED1: ma}, ms, db)
    assert frame.mean() == pytest.approx(_covering(grey, *frame.shape).mean(), rel=0.1)


def test_the_exposure_scales_the_picture_and_the_focus_blurs_it(pictures):
    ma, ms, db = _SHIPPED['BF']
    with _session('LS850', pictures) as scope:
        at_shipped = _shown(scope, {'BF': ma}, ms / 2, db)
        doubled = _shown(scope, {'BF': ma}, ms, db)
        camera = scope._camera_driver
        camera.set_focal_z(scope.motion.get_current_position('Z') + 200.0)
        defocused = _shown(scope, {'BF': ma}, ms, db)

    assert doubled.mean() == pytest.approx(2 * at_shipped.mean(), rel=0.03)

    def edges(frame):
        # The squared step: a blur spreads an edge, which keeps its summed step.
        return float(np.square(np.diff(frame, axis=0)).mean())

    assert edges(defocused) < 0.5 * edges(doubled)


def test_binning_shows_the_same_picture_at_fewer_pixels(pictures):
    ma, ms, db = _SHIPPED['BF']
    with _session('LS850', pictures) as scope:
        full = _shown(scope, {'BF': ma}, ms, db)
        scope.camera.set_binning(2)
        binned = _shown(scope, {'BF': ma}, ms, db)
    assert binned.shape[0] < full.shape[0]
    assert _likeness(binned, _covering(_bf_picture(), *binned.shape)) > 0.95


def test_a_folder_the_cameras_could_not_show_is_refused_as_the_record_is_built(tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    # A hidden file is not a picture: the folder still holds none.
    (empty / '.DS_Store').write_bytes(b'\0')
    unknown = _write_set(tmp_path / 'unknown', {'NotAChannel': _bf_picture()})
    broken = tmp_path / 'broken'
    broken.mkdir()
    (broken / 'BF.jpg').write_bytes(b'not a jpeg')

    refused = {}
    for case, folder in {
        'missing': str(tmp_path / 'missing'),
        'empty': str(empty),
        'unknown': unknown,
        'broken': str(broken),
        'no folder': None,
    }.items():
        with pytest.raises(ArgumentRefusedError) as caught:
            SimulatedScope(slide=Slide.IMAGES, slide_images=folder)
        refused[case] = caught.value.reason
    with pytest.raises(ArgumentRefusedError) as without_images:
        SimulatedScope(slide=Slide.CELLS, slide_images=str(empty))
    refused['without IMAGES'] = without_images.value.reason

    assert refused == {
        'missing': 'fault_bound',
        'empty': 'fault_bound',
        'unknown': 'channel_unknown',
        'broken': 'fault_bound',
        'no folder': 'fault_bound',
        'without IMAGES': 'fault_bound',
    }
