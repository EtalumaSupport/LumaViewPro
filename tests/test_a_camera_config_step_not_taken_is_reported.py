# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A connect-time camera configuration step the camera did not take is named, and the rest run.

``init_camera_config`` used to stop at its first failed step, so every later
setting, which nothing re-applies, stayed at the camera's own default, with
only an ERROR in the log. Each step now runs whatever an earlier one did;
the ones not taken are named on the driver (``config_not_applied``) and
bring-up reports them once (``CameraConfigNotAppliedNotice``; Eric
2026-10-16, *"Run all, report once"*). The image orientation is not such a
step: a camera whose orientation cannot be known does not come up
(``tests/test_a_camera_whose_orientation_cannot_be_known_does_not_come_up.py``).

The drivers run over the suite's SDK stand-ins; the setters each step fans
out to are stubbed (``init_configurable_pylon_camera``), so the step under
test is the only one that fails.
"""

from unittest.mock import MagicMock

from _lumascope.drivers.simulated_camera import SimulatedCamera
from tests.camera_fakes import bare_ids_camera, init_configurable_pylon_camera
from tests.test_bring_up_is_a_record import _bring_up


def test_a_pylon_step_not_taken_is_named_and_the_rest_run():
    cam = init_configurable_pylon_camera()
    cam.set_pixel_format = MagicMock(return_value=False)
    cam.gain = MagicMock(side_effect=Exception('Gain not writable'))

    cam.init_camera_config()

    assert cam.config_not_applied == ('the Mono8 pixel format', 'gain 0 dB')
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
    cam.auto_gain = MagicMock(side_effect=genicam.RuntimeException('comms lost'))
    cam.init_auto_gain_focus = MagicMock(side_effect=genicam.RuntimeException('comms lost'))

    cam.init_camera_config()

    assert cam.config_not_applied == ('auto-gain off', 'the auto-gain and focus defaults')
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

    cam.init_camera_config()

    assert cam.config_not_applied == ('the Default user set',)
    cam.set_pixel_format.assert_called_once()
    cam.set_frame_size.assert_called_once_with(1900, 1900)
    cam._configure_free_run.assert_called_once_with()


def test_bring_up_reports_the_steps_not_taken_once(monkeypatch, tmp_path, centre_posts):
    # A stand-in by design: the simulated camera's connect-time configuration
    # cannot be made to fail (a simulator gap), so its account is set.
    monkeypatch.setattr(SimulatedCamera, 'config_not_applied', ('gain 0 dB',))

    s = _bring_up(monkeypatch, tmp_path, microscope='LS850T')
    try:
        notices = [n for n in centre_posts if n.reason == 'camera_config_not_applied']
        assert len(notices) == 1, centre_posts
        assert notices[0].title == 'Camera configuration incomplete'
        assert 'gain 0 dB' in notices[0].message
    finally:
        s.shutdown()
        s.scope.disconnect()


def test_a_config_guard_that_cannot_read_the_grab_state_runs_no_step_and_raises_the_read():
    """The guard's own grab-state read is no step: it fails the configuration
    whole, before any step runs, and reaches connect()'s handler (Pylon's and
    IDS's report a found camera whose configuration failed) instead of being
    swallowed with every step skipped and nothing named."""
    from _lumascope.drivers.exceptions import CameraReadFailedError

    def unreadable():
        raise CameraReadFailedError('The camera did not answer its grab state: timeout')

    pylon = init_configurable_pylon_camera()
    ids = bare_ids_camera()
    ids.profile = MagicMock(pixel_formats=['Mono8'])
    for name in ('set_pixel_format', 'exposure_t', 'set_frame_size', '_configure_free_run'):
        setattr(ids, name, MagicMock())
    for cam in (pylon, ids):
        cam.is_grabbing = unreadable
        try:
            cam.init_camera_config()
        except CameraReadFailedError:
            pass
        else:
            raise AssertionError(f'{type(cam).__name__}: configured past an unread grab state')
        cam.exposure_t.assert_not_called()
        cam.set_frame_size.assert_not_called()
        assert cam.config_not_applied == ()


def test_an_ids_connect_whose_configuration_cannot_read_the_grab_state_fails_naming_the_read(
    monkeypatch,
):
    """Through ``connect``: the guard's read reaches connect's own handler, a
    camera found whose configuration could not read its grab state, raised
    to the registry as that camera's with the step and the read in its
    words, and no step run. Pylon's connect reads the grab state raw before
    its configuration and fails there (its own handler); its configuration
    guard is the case above."""
    import pytest
    from ids_peak import ids_peak

    from _lumascope.drivers import idscamera
    from _lumascope.drivers.exceptions import CameraReadFailedError, CameraUsbOpenFailedError

    cam = bare_ids_camera()
    cam.active = None
    opened = MagicMock()
    stream = opened.DataStreams.return_value[0].OpenDataStream.return_value
    stream.IsGrabbing.side_effect = ids_peak.TimeoutException('GC_ERR_TIMEOUT')
    cam._find_device = MagicMock()
    cam._open_device_with_retry = lambda descriptor, refresh: (opened, descriptor)
    # Stand-ins by design: the subject is connect's handler around the
    # configuration; the device, its profile and its handler are not.
    monkeypatch.setattr(idscamera, 'ImageHandler', MagicMock())
    for name in (
        '_register_device_callbacks',
        '_load_profile',
        '_query_dynamic_capabilities',
        '_reapply_frame_callbacks',
        '_cleanup_partial_connect',
        'set_pixel_format',
        'exposure_t',
        'set_frame_size',
        '_configure_free_run',
    ):
        setattr(cam, name, MagicMock())
    cam.profile = MagicMock(pixel_formats=['Mono8'])

    with pytest.raises(CameraUsbOpenFailedError, match='its configuration failed') as raised:
        cam.connect()
    assert isinstance(raised.value.__cause__, CameraReadFailedError)
    assert 'grab state' in str(raised.value)
    cam.set_frame_size.assert_not_called()
    cam._cleanup_partial_connect.assert_called_once()
