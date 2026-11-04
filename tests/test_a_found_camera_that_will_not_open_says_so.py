# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera on the bus that cannot be opened is reported as that, not as no camera.

A camera driver's ``connect`` returns False only when no camera of its kind
is reachable, so the registry tries the next driver and, with none, the
bring-up says "Camera not detected". A camera that is found and cannot be
opened or set up raises ``CameraUsbOpenFailedError``, which the registry
outranks over every "not found" and the bring-up record tells as "Could not
connect to the camera". On 2026-10-20 an IDS U3-34L whose reopen failed on
``DataStreams()[0]`` after a successful open was told "a camera driver did
not load ... Reinstall".

The SDKs are stand-ins by design: the simulator has neither IDS peak nor
pylon, and the faults are the SDKs' own.
"""

from unittest.mock import MagicMock

import pytest

from drivers.exceptions import CameraUsbOpenFailedError
from modules.exceptions import CameraNotAvailableError
from modules.lumascope_api._lumascope import _camera_failure_cause
from tests.camera_fakes import bare_ids_camera, bare_pylon_camera


def _ids_camera(monkeypatch, devices):
    from drivers import idscamera

    dm = MagicMock()
    dm.Devices.return_value = devices
    monkeypatch.setattr(idscamera.ids_peak.DeviceManager, 'Instance', lambda: dm, raising=False)
    cam = bare_ids_camera()
    cam._device_lost_callback = None
    cam._device_lost_callback_handle = None
    cam._async_teardown_started = False
    cam.active = None
    return cam, dm


def _one_ids_device():
    devices = MagicMock()
    devices.empty.return_value = False
    descriptor = MagicMock()
    descriptor.SerialNumber.return_value = 'SN-OURS'
    devices.__getitem__.return_value = descriptor
    devices.__iter__.return_value = iter([descriptor])
    return devices, descriptor


def test_an_ids_camera_whose_data_stream_fails_after_the_open_raises(monkeypatch):
    devices, descriptor = _one_ids_device()
    device = descriptor.OpenDevice.return_value
    device.DataStreams.return_value = []  # the bench's IndexError on [0]
    cam, _dm = _ids_camera(monkeypatch, devices)

    with pytest.raises(CameraUsbOpenFailedError) as raised:
        cam.connect()

    assert isinstance(raised.value.__cause__, IndexError)
    assert cam.active is None
    assert cam.data_stream is None


def test_an_ids_camera_that_another_program_holds_raises(monkeypatch):
    devices, descriptor = _one_ids_device()
    descriptor.OpenDevice.side_effect = type('BadAccessException', (Exception,), {})('open already')
    # The open's retries ride out a held camera; with no backoff they run at once.
    monkeypatch.setattr('drivers.idscamera._OPEN_CONTROL_BACKOFF_S', 0.0)
    cam, _dm = _ids_camera(monkeypatch, devices)

    with pytest.raises(CameraUsbOpenFailedError):
        cam.connect()

    assert cam.active is None


def test_an_ids_host_whose_device_update_fails_has_no_camera(monkeypatch):
    devices, _descriptor = _one_ids_device()
    cam, dm = _ids_camera(monkeypatch, devices)
    dm.Update.side_effect = RuntimeError('no GenTL producer')

    assert cam.connect() is False
    assert cam.found is False


def _pylon_tl(monkeypatch, devices):
    from drivers import pyloncamera

    tl = MagicMock()
    tl.GetInstance.return_value.EnumerateDevices.return_value = devices
    monkeypatch.setattr(pyloncamera.pylon, 'TlFactory', tl)
    return tl.GetInstance.return_value


def test_a_basler_that_is_listed_and_will_not_open_raises(monkeypatch):
    from drivers import pyloncamera

    _pylon_tl(monkeypatch, [MagicMock()])
    # a stand-in by design: pylon refusing an open is a fault the simulator cannot yet produce (it has no pylon)
    sdk_camera = MagicMock()
    sdk_camera.Open.side_effect = pyloncamera.genicam.RuntimeException(
        'The device is controlled by another application'
    )
    monkeypatch.setattr(pyloncamera.pylon, 'InstantCamera', lambda _device: sdk_camera)
    cam = bare_pylon_camera()
    cam.active = None

    with pytest.raises(CameraUsbOpenFailedError):
        cam.connect()

    assert cam.active is None


def test_a_host_with_no_basler_has_no_camera(monkeypatch):
    from drivers import pyloncamera

    factory = _pylon_tl(monkeypatch, [])
    factory.CreateFirstDevice.side_effect = pyloncamera.genicam.RuntimeException(
        'No device is available'
    )
    cam = bare_pylon_camera()
    cam.active = None

    assert cam.connect() is False
    assert cam.found is False


def test_the_bring_up_tells_an_open_failure_as_could_not_connect():
    fault = CameraUsbOpenFailedError('IDS camera found, its data stream failed: IndexError')

    assert _camera_failure_cause(fault) == 'camera_not_initialized'
    assert 'Could not connect to the camera' in str(
        CameraNotAvailableError('camera_not_initialized')
    )


def test_the_bring_up_tells_its_own_failure_as_lumaviewpros_not_the_cables():
    """A camera bring-up that raised outside the hardware's family is our own
    defect: its cause and words name LumaViewPro, never the USB cable."""
    fault = TypeError('scripted defect')

    assert _camera_failure_cause(fault) == 'camera_bring_up_failed'
    words = str(CameraNotAvailableError('camera_bring_up_failed'))
    assert 'LumaViewPro failed' in words
    assert 'USB' not in words
