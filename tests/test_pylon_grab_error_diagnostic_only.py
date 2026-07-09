# Copyright Etaluma, Inc.
"""Regression net: OnGrabError is a diagnostic signal, not a disconnect owner.

The pylon SDK fires OnGrabError when a grab-thread exception is caught -- for
example a GEV chunk-parse exception raised right after an in-stream SetValue.
Per the SDK doc it is the last diagnostic log line before the grab loop stops;
it is NOT evidence the device was physically removed. A benign, recoverable
grab-thread exception must not tear the camera down.

Disconnect authority lives only with the definitive-evidence owners: the SDK
removal callback (OnCameraDeviceRemoved), the DEVICE_NOT_FOUND grab path, and
the consecutive-failure cascade. OnGrabError logs and returns; it never latches.
"""

from unittest.mock import patch

from drivers import pyloncamera
from drivers.pyloncamera import _CameraRemovalHandler
from tests.camera_fakes import bare_pylon_camera


def test_on_grab_error_logs_but_does_not_latch():
    cam = bare_pylon_camera()
    handler = _CameraRemovalHandler(cam)

    with patch.object(pyloncamera, '_cam_log') as mock_log:
        handler.OnGrabError(camera=None, errorMessage='transient grab-thread exception')

    # A grab-thread exception is diagnostic, not proof of removal: the camera
    # must NOT be marked disconnected off a single OnGrabError.
    cam._mark_disconnected.assert_not_called()
    # ...but the error must still surface in the log for post-mortem.
    assert mock_log.error.called
    assert any('OnGrabError' in str(call) for call in mock_log.error.call_args_list)
