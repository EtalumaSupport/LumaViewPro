"""A sum is brighter and keeps every count, on every camera (#786).

Summing is for a brighter image, primarily in luminescence. A sum of N frames
of b bits is stored in a uint16 array on every camera, an 8-bit one included,
tagged with the bits it can reach, N x (2^b - 1); every 8-bit rendering of it
is drawn against one frame's white, so it looks N times brighter; and its
saturation is measured against what it can hold, not against its tag's power
of two. The count and the per-frame depth are the capture's own, recorded with
the frame, so no caller restates them.

Before this, an 8-bit camera's sum was clipped to 255 (the counts lost from
every file), a 12-bit sum was tagged 16 (rendered at about a quarter of one
frame's brightness), ``capture_frame_depth(image)`` after a sum answered one
frame's depth, a sum z-projection of 12-bit slices was tagged 12 and failed
its read-back, a hyperstack mixing 8-bit and 16-bit planes was refused, and
an 8-bit camera was offered only the 8-bit image mode.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
import tifffile as tf

from modules import image_mode, image_utils, zprojection
from modules.exceptions import FrameDepthError
from modules.protocol_image_writer import ProtocolImageWriter
from modules.scope_session import ScopeSession
from modules.stack_builder import StackBuilder
from modules.zprojector import ZProjector
from tests.scope_fakes import build_scope, bind_settings_like_a_session
from tests.settings_fixtures import complete_settings

WHITE = {'Mono8': 255, 'Mono12': 4095}


@pytest.fixture
def sim_camera():
    """A simulated scope whose camera delivers one uniform white frame format."""

    def _make(pixel_format):
        scope = build_scope(simulate=True)
        bind_settings_like_a_session(scope)
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


def test_an_8bit_cameras_sum_keeps_every_count(sim_camera):
    scope = sim_camera('Mono8')
    image = scope.imaging.get_image(force_to_8bit=False, sum_count=4)
    assert image.dtype == np.uint16
    assert int(image.max()) == 4 * WHITE['Mono8']
    assert scope.imaging.capture_frame_depth(image) == 10
    assert scope.imaging.capture_frame_full_scale(image) == 4 * WHITE['Mono8']


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


@pytest.mark.parametrize(('pixel_format', 'frames'), [('Mono8', 3), ('Mono8', 4), ('Mono12', 3)])
def test_a_sum_of_blown_frames_reads_saturated(sim_camera, pixel_format, frames):
    scope = sim_camera(pixel_format)
    image = scope.imaging.get_image(force_to_8bit=False, sum_count=frames)
    writer = ProtocolImageWriter.__new__(ProtocolImageWriter)
    writer._scope = scope
    evidence = writer._capture_evidence(image, scope.imaging.capture_frame_full_scale(image))
    assert 'sat=100.0%' in evidence


@pytest.fixture(scope='module')
def ls620_session():
    settings = complete_settings(microscope='LS620')
    settings['image_mode'] = image_mode.IMAGE_MODE_12BIT_SCIENTIFIC
    session = ScopeSession.create(settings, simulate=True, warn_pre_release=False)
    yield session
    session.shutdown()


def test_an_ls620_sum_is_16bit_and_tagged_by_its_reach(ls620_session):
    imaging = ls620_session.scope.imaging
    image = imaging.capture_and_wait(
        force_to_8bit=False, sum_count=4, accept_dark=True, timeout_s=2.0
    )
    assert image is not None
    assert image.dtype == np.uint16
    assert imaging.capture_frame_depth(image) == 10
    assert imaging.last_capture_info['frame_record'].frame_significant_bits == 8


def test_every_camera_is_offered_every_mode():
    assert image_mode.available_modes() == [
        image_mode.IMAGE_MODE_8BIT,
        image_mode.IMAGE_MODE_12BIT_SCIENTIFIC,
        image_mode.IMAGE_MODE_12BIT_SCALED,
        image_mode.IMAGE_MODE_12BIT_FALSE_COLOR_RGB,
    ]


@pytest.mark.parametrize(
    ('mode', 'formats', 'warned'),
    [
        ('12bit_scientific', ('JPG', 'TIFF'), True),
        ('12bit_false_color_rgb', ('TIFF', 'JPG'), True),
        ('12bit_scaled', ('TIFF', 'OME-TIFF'), False),
        ('8bit', ('JPG', 'JPG'), False),
    ],
)
def test_a_jpg_beside_a_full_depth_mode_is_warned(mode, formats, warned):
    assert image_mode.jpg_depth_warning_active(mode, formats) is warned


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


MIXED_PLANES = pytest.mark.parametrize(
    ('encoding', 'unsummed', 'summed'),
    [
        # Scientific keeps counts: the single frame beside the 4-sum, 1:4.
        ('right_aligned', 200, 800),
        # Scaled fills each plane's container, as each plane's own file does.
        ('msb_aligned', 200 << 8, 800 << 6),
    ],
)


def _mixed_planes(path: pathlib.Path, encoding: str) -> pd.DataFrame:
    """An unsummed 8-bit Blue plane beside a 4-sum 10-bit Lumi plane, on disk."""
    _write(path / 'blue.tiff', np.full((4, 4), 200, dtype=np.uint8), 8, encoding)
    _write(path / 'lumi.tiff', np.full((4, 4), 800, dtype=np.uint16), 10, encoding, 'Lumi')
    return pd.DataFrame(
        [
            {'Filepath': 'blue.tiff', 'Color': 'Blue', 'Scan Count': 0, 'Z-Slice': 0},
            {'Filepath': 'lumi.tiff', 'Color': 'Lumi', 'Scan Count': 0, 'Z-Slice': 0},
        ]
    ).assign(X=0.0, Y=0.0, Z=0.0)


def _plane_maxima(stack_file: pathlib.Path) -> list[int]:
    stack = tf.imread(str(stack_file))
    assert stack.dtype == np.uint16
    return sorted(int(plane.max()) for plane in stack.reshape(-1, *stack.shape[-2:]))


@MIXED_PLANES
def test_a_hyperstack_of_8bit_and_16bit_planes_is_built_at_16bit(
    tmp_path, encoding, unsummed, summed
):
    result = StackBuilder._create_stack(
        path=tmp_path,
        df=_mixed_planes(tmp_path, encoding),
        output_file_loc=pathlib.Path('stack.ome.tiff'),
        save_encoding=encoding,
    )
    assert result['status'], result.get('error')
    assert _plane_maxima(tmp_path / 'stack.ome.tiff') == sorted([unsummed, summed])


@MIXED_PLANES
def test_a_runs_stack_is_built_in_the_runs_encoding(tmp_path, encoding, unsummed, summed):
    # The run's post-run build reaches the stack through the group algorithm.
    result = StackBuilder(has_turret=False)._group_algorithm(
        path=tmp_path,
        df=_mixed_planes(tmp_path, encoding),
        output_file_loc=pathlib.Path('stack.ome.tiff'),
        save_encoding=encoding,
    )
    assert result.status, result.error
    assert _plane_maxima(tmp_path / 'stack.ome.tiff') == sorted([unsummed, summed])


@MIXED_PLANES
def test_a_manual_recordings_stack_is_built_in_its_encoding(tmp_path, encoding, unsummed, summed):
    # One recording is one channel over time: the two planes are its frames.
    df = _mixed_planes(tmp_path, encoding).assign(Color='Lumi', **{'Scan Count': [0, 1]})
    result = StackBuilder(has_turret=False).create_single_recording_stack(
        df=df,
        path=tmp_path,
        output_file_loc=tmp_path / 'stack.ome.tiff',
        save_encoding=encoding,
    )
    assert result['status'], result.get('error')
    assert _plane_maxima(tmp_path / 'stack.ome.tiff') == sorted([unsummed, summed])


# --- the renderings and the hand-offs -----------------------------------------


def test_a_sums_jpg_is_rendered_against_one_frames_white(tmp_path):
    """A JPG is the frame as it is shown: three 8-bit frames summed to 255 are
    one frame's white, so the JPG is white, not 255 of the sum's 1023."""
    import cv2

    from modules import image_save
    from tests.frame_records import frame_record, plate
    from tests.test_jpg_export import _scope_with_depth

    path = image_save.save_image(
        _scope_with_depth(),
        np.full((32, 32), 255, dtype=np.uint16),
        save_folder=str(tmp_path),
        file_root='snap_',
        append='BF',
        channel='BF',
        false_color_on=False,
        tail_id_mode=None,
        output_format='JPG',
        jpeg_quality=95,
        save_encoding='right_aligned',
        significant_bits=10,
        objective_id='4x Oly',
        frame_record=frame_record(frames_summed=3, frame_significant_bits=8),
        labware=plate(),
        well_label=None,
    )
    jpg = cv2.imdecode(np.frombuffer(pathlib.Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert int(jpg.min()) >= 250


def test_the_depth_before_any_capture_is_the_cameras_stamp(sim_camera):
    scope = sim_camera('Mono12')
    assert scope.imaging.capture_frame_depth(np.zeros((4, 4), dtype=np.uint16)) == 12
    assert scope.imaging.capture_frame_full_scale(np.zeros((4, 4), dtype=np.uint16)) == 4095


def test_the_post_run_build_is_told_the_runs_encoding(monkeypatch, tmp_path):
    from modules import stack_builder
    from modules.notification_center import notifications

    asked = {}

    def _load_folder(self, **kwargs):
        asked.update(kwargs)
        # Only what it was asked matters here; the build's answer is reported.
        raise RuntimeError('not built in this test')

    monkeypatch.setattr(notifications, 'report_outcome', lambda *a, **k: None)
    monkeypatch.setattr(stack_builder.StackBuilder, 'load_folder', _load_folder)
    stack_builder.build_hyperstacks_for_run(
        tmp_path,
        False,
        tmp_path / 'tiling.json',
        wait_for_images=lambda: None,
        save_encoding='msb_aligned',
    )
    assert asked['save_encoding'] == 'msb_aligned'
