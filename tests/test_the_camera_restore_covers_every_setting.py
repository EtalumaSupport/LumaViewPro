"""The one camera-state restore puts back every camera setting with a getter.

``save_camera_state`` / ``restore_camera_state`` covered gain, exposure and
the auto-gain arm; a caller that also changed the pixel format, frame size,
binning or black level restored those itself, or not at all (the engineering
plugin restored format and frame size inline and the black level nowhere).
The snapshot now holds all four: binning where the camera offers more than one
size, the black level where the camera's can be set, and a failed read of an
offered setting raises at the save instead of leaving it out. The restore
writes a setting only where it differs from the camera's value now (a frame
size, format or binning write restarts grabbing), in the order the camera
needs: binning (which resets the frame), frame size, format, black level, then
gain and exposure and the arm.
"""

from dataclasses import replace
from unittest.mock import patch

import pytest

from drivers.exceptions import HardwareError


def _changed(imaging):
    imaging.set_binning_size(2)
    imaging.set_frame_size(800, 600)
    imaging.set_pixel_format('Mono12')
    imaging.set_black_level(4.0)


def _state(sim_scope):
    driver = sim_scope._camera_driver
    return {
        'binning': driver.get_binning_size(),
        'frame_size': dict(driver.get_frame_size()),
        'pixel_format': driver.get_pixel_format(),
        'black_level': driver.get_black_level(),
    }


def test_every_setting_a_caller_changed_comes_back(sim_scope):
    imaging = sim_scope.imaging
    imaging.set_pixel_format('Mono8')
    imaging.set_frame_size(1920, 1080)
    before = _state(sim_scope)
    snapshot = imaging.save_camera_state('test')

    _changed(imaging)
    assert _state(sim_scope) != before
    imaging.restore_camera_state(snapshot)

    assert _state(sim_scope) == before


def test_an_unchanged_setting_is_not_written(sim_scope):
    imaging = sim_scope.imaging
    driver = sim_scope._camera_driver
    snapshot = imaging.save_camera_state('test')

    with (
        patch.object(driver, 'set_binning_size') as binning,
        patch.object(driver, 'set_frame_size') as frame,
        patch.object(driver, 'set_pixel_format') as fmt,
        patch.object(driver, 'set_black_level') as black,
    ):
        imaging.restore_camera_state(snapshot)

    binning.assert_not_called()
    frame.assert_not_called()
    fmt.assert_not_called()
    black.assert_not_called()


def test_the_snapshot_names_the_four_settings(sim_scope):
    snapshot = sim_scope.imaging.save_camera_state('test')
    assert {'binning', 'frame_size', 'pixel_format', 'black_level'} <= set(snapshot)


def test_a_camera_whose_black_level_cannot_be_set_has_none_to_restore(sim_scope, monkeypatch):
    monkeypatch.setattr(
        sim_scope,
        'capabilities',
        replace(sim_scope.capabilities, camera_supports_black_level=False),
    )
    assert 'black_level' not in sim_scope.imaging.save_camera_state('test')


def test_a_camera_with_one_binning_size_has_none_to_restore(sim_scope, monkeypatch):
    monkeypatch.setattr(
        sim_scope,
        'capabilities',
        replace(sim_scope.capabilities, camera_binning_sizes=(1,)),
    )
    assert 'binning' not in sim_scope.imaging.save_camera_state('test')


def test_a_failed_black_level_read_raises_at_the_save(sim_scope):
    with (
        patch.object(
            sim_scope._camera_driver,
            'get_black_level',
            side_effect=HardwareError('BlackLevel read failed'),
        ),
        pytest.raises(HardwareError, match='BlackLevel'),
    ):
        sim_scope.imaging.save_camera_state('test')


@pytest.mark.parametrize(
    ('reader', 'setting'),
    [
        ('get_frame_size', 'frame size'),
        ('get_pixel_format', 'pixel format'),
        ('get_binning_size', 'binning'),
    ],
)
def test_a_failed_read_of_an_offered_setting_raises_at_the_save(sim_scope, reader, setting):
    with (
        patch.object(sim_scope._camera_driver, reader, side_effect=RuntimeError('node read')),
        pytest.raises(HardwareError, match=setting),
    ):
        sim_scope.imaging.save_camera_state('test')
