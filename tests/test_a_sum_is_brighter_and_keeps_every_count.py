"""A sum is brighter and keeps every count (#786).

Summing is for a brighter image, primarily in luminescence. A sum of N frames
of b bits is tagged with the bits it can reach, N x (2^b - 1); every 8-bit
rendering of it is drawn against one frame's white, so it looks N times
brighter; and its saturation is measured against what it can hold, not
against its tag's power of two. The count and the per-frame depth are the
capture's own, recorded with the frame, so no caller restates them.

Before this, a 12-bit sum was tagged 16 (rendered at about a quarter of one
frame's brightness), and ``capture_frame_depth(image)`` after a sum answered
one frame's depth, which the sum's values overran. A sum z-projection of
uint8 slices was clipped at 255, and one of 12-bit slices was tagged 12 and
failed its read-back.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from modules import image_utils, zprojection
from modules.exceptions import FrameDepthError
from modules.protocol_image_writer import ProtocolImageWriter
from modules.zprojector import ZProjector
from tests.scope_fakes import build_scope

WHITE = {'Mono8': 255, 'Mono12': 4095}


@pytest.fixture
def sim_camera():
    """A simulated scope whose camera delivers one uniform white frame format."""

    def _make(pixel_format):
        scope = build_scope(simulate=True)
        scope._camera_driver.profile.pixel_formats = [pixel_format]
        cam = scope._camera_driver
        cam.set_timing_mode('fast')
        cam.set_pixel_format(pixel_format)
        cam.set_test_pattern(True, 'White')
        scope.imaging.start_streaming()
        return scope

    return _make


# --- the rule -----------------------------------------------------------------


@pytest.mark.parametrize(
    ('frames', 'bits', 'tag'),
    [(1, 8, 8), (1, 12, 12), (3, 8, 10), (4, 8, 10), (4, 12, 14), (9, 12, 16), (17, 12, 16)],
)
def test_a_sum_is_tagged_with_the_bits_it_can_reach(frames, bits, tag):
    assert image_utils.summed_significant_bits(frames, bits) == tag


@pytest.mark.parametrize(
    ('frames', 'bits', 'full_scale'),
    [(1, 8, 255), (3, 8, 765), (4, 12, 16380), (30, 12, 65535)],
)
def test_a_sum_saturates_at_what_it_can_hold(frames, bits, full_scale):
    assert image_utils.summed_full_scale(frames, bits) == full_scale


def test_a_sum_renders_against_one_frames_white():
    three_8bit_frames = np.array([[0, 64, 255, 765]], dtype=np.uint16)
    rendered = image_utils.convert_sum_to_8bit(three_8bit_frames, 3, 8)
    assert rendered.tolist() == [[0, 64, 255, 255]]


def test_a_single_frame_renders_as_it_always_has():
    frame = np.array([[0, 2048, 4095]], dtype=np.uint16)
    assert np.array_equal(
        image_utils.convert_sum_to_8bit(frame, 1, 12), image_utils.convert_to_8bit(frame, 12)
    )


def test_a_value_past_what_the_sum_can_reach_is_still_refused():
    with pytest.raises(FrameDepthError):
        image_utils.convert_sum_to_8bit(np.array([[1024]], dtype=np.uint16), 4, 8)


# --- the capture --------------------------------------------------------------


def test_a_12bit_sum_renders_brighter_not_dimmer(sim_camera):
    scope = sim_camera('Mono12')
    one = scope.imaging.get_image(force_to_8bit=True, sum_count=1)
    two = scope.imaging.get_image(force_to_8bit=True, sum_count=2)
    assert int(one.max()) == 255
    assert int(two.max()) == 255


def test_the_depth_of_a_sum_needs_no_count(sim_camera):
    scope = sim_camera('Mono12')
    image = scope.imaging.get_image(force_to_8bit=False, sum_count=4)
    assert int(image.max()) == 4 * WHITE['Mono12']
    assert scope.imaging.capture_frame_depth(image) == 14
    # The depth answered is one the frame honours: no value lies past it.
    assert int(image.max()) <= (1 << scope.imaging.capture_frame_depth(image)) - 1


def test_the_record_carries_the_count_and_each_frames_depth(sim_camera):
    scope = sim_camera('Mono12')
    image = scope.imaging.capture_and_wait(
        force_to_8bit=False, sum_count=3, accept_dark=True, timeout_s=2.0
    )
    assert image is not None
    record = scope.imaging.last_capture_info['frame_record']
    assert record.frames_summed == 3
    assert record.frame_significant_bits == 12


@pytest.mark.parametrize(('pixel_format', 'frames'), [('Mono12', 3)])
def test_a_sum_of_blown_frames_reads_saturated(sim_camera, pixel_format, frames):
    scope = sim_camera(pixel_format)
    image = scope.imaging.get_image(force_to_8bit=False, sum_count=frames)
    writer = ProtocolImageWriter.__new__(ProtocolImageWriter)
    writer._scope = scope
    evidence = writer._capture_evidence(image, scope.imaging.capture_frame_full_scale(image))
    assert 'sat=100.0%' in evidence


# --- the products -------------------------------------------------------------


def _write(path: pathlib.Path, array: np.ndarray, bits: int, encoding: str, color: str = 'Blue'):
    image_utils.write_tiff(
        data=array,
        file_loc=path,
        metadata={
            'datetime': '2026-10-04T12:00:00',
            'plate_pos_mm': {'x': 0.0, 'y': 0.0},
            'z_pos_um': 0.0,
            'exposure_time_ms': 50.0,
            'gain_db': 0.0,
            'illumination_ma': 0.0,
            'pixel_size_um': 0.5,
            'channel': color,
            'objective': {
                'model': 'PlanFluor20x',
                'manufacturer': 'Nikon',
                'magnification': 20,
                'aperture': 0.45,
                'working_distance': 8.1,
                'immersion': 'Air',
            },
            'instrument': {
                'manufacturer': 'Etaluma',
                'model': 'LS620',
                'serial_number': 'SN0',
                'camera_model': 'MT9P031',
            },
            'plate': {'name': '96-well', 'rows': 8, 'columns': 12},
            'well_label': 'A1',
        },
        ome=False,
        color=color,
        significant_bits=bits,
        save_encoding=encoding,
    )


def test_a_sum_projection_of_8bit_slices_keeps_its_counts():
    slices = [np.full((4, 4), 200, dtype=np.uint8) for _ in range(3)]
    projected = zprojection.zproject(slices, zprojection.ZProjectMethod.Sum)
    assert projected.dtype == np.uint16
    assert int(projected.max()) == 600


def test_a_colour_sum_projection_does_not_wrap():
    slices = []
    for _ in range(3):
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[:, :, 2] = 200
        slices.append(rgb)
    result = ZProjector(has_turret=False)._zproject_for_multi_channel(
        slices, zprojection.ZProjectMethod.Sum
    )
    assert result['image'].dtype == np.uint16
    assert int(result['image'][:, :, 2].max()) == 600


def test_a_12bit_sum_projection_is_tagged_by_its_reach_and_reads_back(tmp_path):
    rows = []
    for z in range(3):
        name = f'A1_Blue_Z{z}.tiff'
        _write(tmp_path / name, np.full((8, 8), 4000, dtype=np.uint16), 12, 'right_aligned')
        rows.append({'Color': 'Blue', 'Filepath': name})
    result = ZProjector(has_turret=False)._zproject(
        path=tmp_path,
        df=pd.DataFrame(rows),
        method='Sum',
        output_file_loc=pathlib.Path('projected.tiff'),
    )
    assert result['significant_bits'] == 14
    pixels, bits = image_utils.load_pixels(tmp_path / 'projected.tiff')
    assert bits == 14
    assert int(pixels.max()) == 12000
