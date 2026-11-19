# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Basler USB camera is reset at every connect, before it is configured.

A 1.1.0 daA3840 left stalling its stops 11 s by an earlier session stays so
through a reconnect, and only a power cycle or a ``DeviceReset`` clears it.
So ``PylonCamera.connect`` resets the USB device it created, waits for that
serial to leave the bus and come back, and connects to it; a camera without
``DeviceReset`` connects as created. Each failure is typed by how far the
connect got: the open before the reset, the reset once open, the camera's
return to the bus. The emulator and GigE bodies are not reset.

The SDK is a stand-in by design: the simulator has no pylon, and the subject
is what connect asks of it. Each test ends the connect where its post-open
configuration starts, by a scripted ``IsGrabbing`` failure, so the camera
connect goes on with is the one it opened.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from _lumascope.drivers.exceptions import (
    CameraSetupFailedError,
    CameraUsbNotFoundError,
    CameraUsbOpenFailedError,
)
from tests.camera_fakes import bare_pylon_camera

_SERIAL = '40159444'

# The scripted end of every connect here: its post-open configuration's first read.
_SCRIPTED_END = 'its post-open configuration failed: RuntimeException: scripted end'


def _info(device_class='BaslerUsb', serial=_SERIAL):
    info = MagicMock()
    info.GetDeviceClass.return_value = device_class
    info.GetSerialNumber.return_value = serial
    return info


class _Bus:
    """pylon's transport-layer factory over a bus whose listing changes.

    Each ``EnumerateDevices`` answers the next listing, the last repeating.
    """

    def __init__(self, *listings):
        self._listings = list(listings)
        self.recreated_from = None

    def EnumerateDevices(self):
        if len(self._listings) > 1:
            return self._listings.pop(0)
        return self._listings[0]

    def CreateFirstDevice(self, *_filter):
        return 'created'

    def CreateDevice(self, info):
        self.recreated_from = info
        return 'recreated'


def _sdk_camera(info, *, reset_node=None):
    # a stand-in by design: pylon's InstantCamera, which the simulator has no pylon to give; the subject is what connect asks of it
    camera = MagicMock()
    camera.GetDeviceInfo.return_value = info
    nodemap = MagicMock()
    nodemap.Contains.side_effect = lambda name: name == 'DeviceReset' and reset_node is not None
    nodemap.GetNode.return_value = reset_node
    camera.GetNodeMap.return_value = nodemap
    return camera


def _ends_at_the_post_open_configuration(camera, pyloncamera):
    camera.IsGrabbing.side_effect = pyloncamera.genicam.RuntimeException('scripted end')


def _connect(monkeypatch, bus, cameras, *, emulation=False):
    """Connect a bare PylonCamera on ``bus``; ``cameras`` maps each device to its InstantCamera."""
    from _lumascope.drivers import pyloncamera

    monkeypatch.setattr(
        pyloncamera.pylon, 'TlFactory', SimpleNamespace(GetInstance=lambda: bus), raising=False
    )
    monkeypatch.setattr(pyloncamera.pylon, 'InstantCamera', cameras.__getitem__, raising=False)
    monkeypatch.setattr(pyloncamera.pylon, 'DeviceInfo', MagicMock, raising=False)
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(
        pyloncamera,
        'time',
        SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep, perf_counter=lambda: clock.now),
    )
    cam = bare_pylon_camera()
    cam.active = None
    cam._use_camera_emulation = emulation
    cam.connect()


def test_a_usb_camera_is_reset_before_anything_is_registered_and_its_serial_reconnected(
    monkeypatch,
):
    from _lumascope.drivers import pyloncamera

    created, returned = _info(), _info()
    reset_node = MagicMock()
    first = _sdk_camera(created, reset_node=reset_node)
    second = _sdk_camera(returned)
    _ends_at_the_post_open_configuration(second, pyloncamera)
    bus = _Bus([created], [created], [], [returned])

    with pytest.raises(CameraSetupFailedError, match=_SCRIPTED_END):
        _connect(monkeypatch, bus, {'created': first, 'recreated': second})

    reset_node.Execute.assert_called_once_with()
    first.DestroyDevice.assert_called_once_with()
    first.RegisterConfiguration.assert_not_called()
    assert bus.recreated_from is returned
    second.Open.assert_called_once_with()


def test_a_usb_camera_without_device_reset_connects_as_created(monkeypatch):
    from _lumascope.drivers import pyloncamera

    created = _info()
    first = _sdk_camera(created)
    _ends_at_the_post_open_configuration(first, pyloncamera)
    bus = _Bus([created])

    with pytest.raises(CameraSetupFailedError, match=_SCRIPTED_END):
        _connect(monkeypatch, bus, {'created': first})

    first.DestroyDevice.assert_not_called()
    assert bus.recreated_from is None
    first.RegisterConfiguration.assert_called()


def test_a_usb_camera_whose_open_before_its_reset_fails_raises_the_open(monkeypatch):
    from _lumascope.drivers import pyloncamera

    created = _info()
    first = _sdk_camera(created, reset_node=MagicMock())
    first.Open.side_effect = pyloncamera.genicam.RuntimeException(
        'The device is controlled by another application'
    )

    with pytest.raises(CameraUsbOpenFailedError, match='the open before its reset failed'):
        _connect(monkeypatch, _Bus([created]), {'created': first})


def test_a_usb_camera_whose_reset_fails_raises_its_set_up(monkeypatch):
    from _lumascope.drivers import pyloncamera

    created = _info()
    reset_node = MagicMock()
    reset_node.Execute.side_effect = pyloncamera.genicam.RuntimeException('access denied')
    first = _sdk_camera(created, reset_node=reset_node)

    with pytest.raises(CameraSetupFailedError, match='its reset failed'):
        _connect(monkeypatch, _Bus([created]), {'created': first})


@pytest.mark.parametrize(
    ('after_the_reset', 'words'),
    [
        ((), 'never left the bus'),
        (([],), 'left the bus and did not come back'),
    ],
    ids=['never-left', 'left-and-not-back'],
)
def test_a_usb_camera_not_back_by_the_deadline_is_not_found(monkeypatch, after_the_reset, words):
    """Never connected as it was: it may reboot under the session."""
    created = _info()
    first = _sdk_camera(created, reset_node=MagicMock())
    bus = _Bus([created], [created], *after_the_reset)

    with pytest.raises(CameraUsbNotFoundError, match=f'reset it: it {words} in the 90 s'):
        _connect(monkeypatch, bus, {'created': first})

    first.RegisterConfiguration.assert_not_called()


@pytest.mark.parametrize(
    ('device_class', 'emulation'),
    [('BaslerCamEmu', True), ('BaslerGigE', False)],
    ids=['emulator', 'gige'],
)
def test_the_emulator_and_a_gige_camera_are_not_reset(monkeypatch, device_class, emulation):
    from _lumascope.drivers import pyloncamera

    created = _info(device_class)
    reset_node = MagicMock()
    first = _sdk_camera(created, reset_node=reset_node)
    _ends_at_the_post_open_configuration(first, pyloncamera)

    with pytest.raises(CameraSetupFailedError, match=_SCRIPTED_END):
        _connect(monkeypatch, _Bus([created]), {'created': first}, emulation=emulation)

    reset_node.Execute.assert_not_called()
    first.RegisterConfiguration.assert_called()
