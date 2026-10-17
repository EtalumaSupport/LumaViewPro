# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A connect-time camera configuration step the camera did not take is named, and the rest run.

``init_camera_config`` used to stop at its first failed step, so every later
setting -- the image orientation among them, which nothing re-applies --
stayed at the camera's own default, with only an ERROR in the log. Each
step now runs whatever an earlier one did; the ones not taken are named on
the driver (``config_not_applied``) and bring-up reports them once
(``CameraConfigNotAppliedNotice``; Eric 2026-10-16, *"Run all, report once"*).

The drivers run over the suite's SDK stand-ins; the setters each step fans
out to are stubbed (``init_configurable_pylon_camera``), so the step under
test is the only one that fails.
"""

from unittest.mock import MagicMock

from drivers.simulated_camera import SimulatedCamera
from tests.camera_fakes import bare_ids_camera, init_configurable_pylon_camera
from tests.test_bring_up_is_a_record import _bring_up


def test_a_pylon_step_not_taken_is_named_and_the_rest_run():
    cam = init_configurable_pylon_camera()
    cam.active.ReverseX.SetValue.side_effect = Exception('ReverseX not writable')
    cam.set_pixel_format = MagicMock(return_value=False)

    cam.init_camera_config()

    assert cam.config_not_applied == (
        'the Mono8 pixel format',
        'the image orientation (ReverseX)',
    )
    cam.exposure_t.assert_called_once_with(exposure_ms=10)
    cam.set_frame_size.assert_called_once_with(w=1900, h=1900)
    cam._mark_disconnected.assert_not_called()


def test_a_pylon_step_that_loses_comms_is_named_and_the_rest_run():
    """One RuntimeException is that step's failure, not the camera's removal.

    The camera is marked removed only on a definite signal (the SDK's removal
    callback, DEVICE_NOT_FOUND, the failed-grab cascade); bring-up used to stop
    every later step on this exception and mark the camera removed.
    """
    from pypylon import genicam

    cam = init_configurable_pylon_camera()
    del cam._mark_disconnected  # the real latch
    cam._device_removed = False
    cam._use_camera_emulation = False  # a real camera runs the auto-gain step
    cam.active.ReverseX.SetValue.side_effect = genicam.RuntimeException('comms lost')
    cam.init_auto_gain_focus = MagicMock(side_effect=genicam.RuntimeException('comms lost'))

    cam.init_camera_config()

    assert cam.config_not_applied == (
        'the image orientation (ReverseX)',
        'the auto-gain and focus defaults',
    )
    cam.exposure_t.assert_called_once_with(exposure_ms=10)
    cam.set_frame_size.assert_called_once_with(w=1900, h=1900)
    assert not cam.is_device_removed()


def test_an_ids_step_not_taken_is_named_and_the_rest_run():
    cam = bare_ids_camera()
    cam.profile = MagicMock(pixel_formats=['Mono8'])
    cam.remote_nodemap.FindNode.return_value.Execute.side_effect = Exception('UserSetLoad busy')
    for name in (
        'set_pixel_format',
        'exposure_t',
        'set_frame_size',
        '_configure_free_run',
        '_log_free_run_state',
    ):
        setattr(cam, name, MagicMock())
    cam.get_supported_pixel_formats = MagicMock(return_value=('Mono8',))
    cam._diag_probe_node = MagicMock(return_value={'access': 'ReadWrite'})

    cam.init_camera_config()

    assert cam.config_not_applied == ('the Default user set',)
    cam.set_pixel_format.assert_called_once()
    cam.set_frame_size.assert_called_once_with(1900, 1900)
    cam._configure_free_run.assert_called_once_with()


def test_bring_up_reports_the_steps_not_taken_once(monkeypatch, tmp_path, centre_posts):
    # A stand-in by design: the simulated camera's connect-time configuration
    # cannot be made to fail (a simulator gap), so its account is set.
    monkeypatch.setattr(
        SimulatedCamera, 'config_not_applied', ('the image orientation (ReverseX)',)
    )

    s = _bring_up(monkeypatch, tmp_path, microscope='LS850T')
    try:
        notices = [n for n in centre_posts if n.reason == 'camera_config_not_applied']
        assert len(notices) == 1, centre_posts
        assert notices[0].title == 'Camera configuration incomplete'
        assert 'the image orientation (ReverseX)' in notices[0].message
    finally:
        s.shutdown()
        s.scope.disconnect()
