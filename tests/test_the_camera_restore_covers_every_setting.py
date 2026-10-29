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
gain and exposure and the arm. The conversion-gain mode and line-noise
reduction, which no driver reads back, are held in the snapshot as the camera
last took them and come back before the black level, their saved settings with
them.
"""

from dataclasses import replace
from unittest.mock import patch

import pytest

from drivers.exceptions import CameraReadFailedError, HardwareError


def _changed(camera):
    camera.set_binning(2)
    camera.set_frame_size(800, 600)
    # A pixel format is set only through the image mode; 12-bit scientific
    # selects Mono12 on the simulated camera.
    camera.set_image_mode('12bit_scientific')
    camera.set_black_level(4.0)


def _state(sim_scope):
    driver = sim_scope._camera_driver
    return {
        'binning': driver.get_binning_size(),
        'frame_size': dict(driver.get_frame_size()),
        'pixel_format': driver.get_pixel_format(),
        'black_level': driver.get_black_level(),
    }


def test_every_setting_a_caller_changed_comes_back(sim_scope):
    camera = sim_scope.camera
    camera.set_image_mode('8bit')
    camera.set_frame_size(1920, 1080)
    before = _state(sim_scope)
    snapshot = camera.save_camera_state('test')

    _changed(camera)
    assert _state(sim_scope) != before
    camera.restore_camera_state(snapshot)

    assert _state(sim_scope) == before


def test_an_unchanged_setting_is_not_written(sim_scope):
    camera = sim_scope.camera
    driver = sim_scope._camera_driver
    snapshot = camera.save_camera_state('test')

    with (
        patch.object(driver, 'set_binning_size') as binning,
        patch.object(driver, 'set_frame_size') as frame,
        patch.object(driver, 'set_pixel_format') as fmt,
        patch.object(driver, 'set_black_level') as black,
    ):
        camera.restore_camera_state(snapshot)

    binning.assert_not_called()
    frame.assert_not_called()
    fmt.assert_not_called()
    black.assert_not_called()


def test_the_snapshot_names_the_four_settings(sim_scope):
    snapshot = sim_scope.camera.save_camera_state('test')
    assert {'binning', 'frame_size', 'pixel_format', 'black_level'} <= set(snapshot)


def test_a_camera_whose_black_level_cannot_be_set_has_none_to_restore(sim_scope, monkeypatch):
    monkeypatch.setattr(
        sim_scope,
        'capabilities',
        replace(sim_scope.capabilities, camera_supports_black_level=False),
    )
    assert 'black_level' not in sim_scope.camera.save_camera_state('test')


def test_a_camera_with_one_binning_size_has_none_to_restore(sim_scope, monkeypatch):
    monkeypatch.setattr(
        sim_scope,
        'capabilities',
        replace(sim_scope.capabilities, camera_binning_sizes=(1,)),
    )
    assert 'binning' not in sim_scope.camera.save_camera_state('test')


def test_a_failed_black_level_read_raises_at_the_save(sim_scope):
    with (
        patch.object(
            sim_scope._camera_driver,
            'get_black_level',
            side_effect=CameraReadFailedError('BlackLevel read failed'),
        ),
        pytest.raises(HardwareError, match='BlackLevel'),
    ):
        sim_scope.camera.save_camera_state('test')


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
        sim_scope.camera.save_camera_state('test')


def test_the_conversion_gain_and_line_noise_reduction_come_back_and_are_saved(
    sim_scope, monkeypatch
):
    # a stand-in by design: the simulated camera has neither toggle (a
    # simulator gap), so the scope is given both and the driver's two writes
    # record what reached it.
    monkeypatch.setattr(
        sim_scope,
        'capabilities',
        replace(
            sim_scope.capabilities,
            camera_supports_conversion_gain_mode=True,
            camera_supports_line_noise_reduction=True,
        ),
    )
    reached = []
    driver = sim_scope._camera_driver
    driver.set_conversion_gain_mode = lambda mode: reached.append(mode) or True
    driver.set_line_noise_reduction = lambda enabled: reached.append(enabled) or True
    camera = sim_scope.camera
    camera.set_high_conversion_gain(False)
    camera.set_line_noise_reduction(False)
    snapshot = camera.save_camera_state('test')

    camera.set_high_conversion_gain(True)
    camera.set_line_noise_reduction(True)
    reached.clear()
    camera.restore_camera_state(snapshot)

    assert reached == ['Low', False]
    assert (camera.high_conversion_gain, camera.line_noise_reduction) == (False, False)
    assert sim_scope.read_setting('camera.high_conversion_gain') is False
    assert sim_scope.read_setting('camera.line_noise_reduction') is False

    # Held as the snapshot holds them, neither is written again: on Pylon
    # each write restarts the grab.
    reached.clear()
    camera.restore_camera_state(snapshot)
    assert reached == []
