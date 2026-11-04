# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Lumascope Classic bound to a driver LumaViewPro cannot use says which.

Windows binds the FX2's bootloader (8613) and its running firmware (EA17)
each on its own, and pyusb sees a device only through a driver libusb can
drive, so a scope on no driver, or on one the stream cannot use, came up as
"Camera not found. Check USB cable and power." The connect now reads both
bindings, logs them, and refuses a present device whose driver cannot serve
what LumaViewPro does with it, by what the code needs, not by whose package
it is: the upload reaches the bootloader through libusb (WinUSB, libusbK or
libusb0), the stream reads the running firmware through WinUSB.

The simulated FX2 is the bus, its binding a fault door; the Windows
transport's own read is stood in where SetupAPI is absent.
"""

from types import SimpleNamespace

import pytest

from drivers import fx2driver
from drivers.exceptions import CameraUsbDriverUnusableError
from drivers.registry import DriverRegistry
from drivers.simulated_fx2 import SimulatedFX2Device, SimulatedFX2Transport
from drivers.usb_driver_binding import FX2_BOOTLOADER, FX2_FIRMWARE, DeviceBinding
from modules.exceptions import CameraNotAvailableError
from modules.lumascope_api._lumascope import _camera_failure_cause


class _AbsentCamera:
    """Another camera driver, its camera not attached."""

    found = False


def _simulated(identity, service) -> SimulatedFX2Transport:
    transport = SimulatedFX2Transport(SimulatedFX2Device())
    transport.bind(identity, service)
    return transport


def test_a_bootloader_on_no_driver_is_refused_naming_it():
    with pytest.raises(
        CameraUsbDriverUnusableError, match=r'Lumascope Classic \(Bootloader\) .*no driver'
    ):
        fx2driver._FX2Connection(_simulated(FX2_BOOTLOADER, None))


def test_a_running_firmware_on_libusbk_is_refused_once_it_re_enumerates():
    transport = _simulated(FX2_FIRMWARE, 'libusbK')

    with pytest.raises(CameraUsbDriverUnusableError, match=r'libusbK.*needs one of: winusb'):
        fx2driver._FX2Connection(transport)
    assert transport.device.uploads == 1


def test_a_bootloader_on_libusbk_connects():
    transport = _simulated(FX2_BOOTLOADER, 'libusbK')

    fx2driver._FX2Connection(transport)

    # Uploaded through the bootloader and running its firmware.
    assert transport.device.pid == fx2driver.PID_APP


def test_bring_up_names_the_driver_not_an_absent_camera(monkeypatch):
    fx2driver._FX2Connection._reset_for_test()
    monkeypatch.setattr(
        fx2driver, '_platform_transport', lambda: _simulated(FX2_BOOTLOADER, 'usbccgp')
    )
    registry = DriverRegistry('camera')
    registry.register('pylon', priority=100)(_AbsentCamera)
    registry.register('fx2', priority=80)(fx2driver.FX2Camera)
    try:
        with pytest.raises(CameraUsbDriverUnusableError) as caught:
            registry.create('auto')
    finally:
        fx2driver._FX2Connection._reset_for_test()

    cause = _camera_failure_cause(caught.value)
    notice = CameraNotAvailableError(cause)
    assert cause == 'camera_driver_unusable'
    assert notice.title == 'Camera driver not usable'
    assert 'install a LumaViewPro build that carries it' in str(notice)


def _binding(identity, *, present, service):
    return DeviceBinding(
        instance_id=f'{identity}\\5&1&0&1',
        present=present,
        service=service,
        provider=None,
        inf=None,
        version=None,
        date=None,
        problem_code=None,
    )


def test_the_windows_transport_refuses_a_present_device_and_not_a_past_one(monkeypatch):
    seen = {
        FX2_BOOTLOADER: [_binding(FX2_BOOTLOADER, present=False, service=None)],
        FX2_FIRMWARE: [_binding(FX2_FIRMWARE, present=True, service=None)],
    }
    # a stand-in by design: SetupAPI, absent off Windows.
    monkeypatch.setattr('drivers.winusb_iso.device_bindings', lambda identity: seen[identity])
    # a stand-in by design: the USB bus, on which nothing is found.
    monkeypatch.setattr(
        fx2driver,
        'usb',
        SimpleNamespace(
            core=SimpleNamespace(find=lambda **kw: None),
            util=SimpleNamespace(dispose_resources=lambda dev: None),
        ),
        raising=False,
    )

    with pytest.raises(CameraUsbDriverUnusableError) as refused:
        fx2driver._FX2Connection(fx2driver._WinUsbTransport())
    seen[FX2_FIRMWARE] = []
    with pytest.raises(fx2driver._FX2AbsentError):
        fx2driver._FX2Connection(fx2driver._WinUsbTransport())

    assert 'Lumascope Classic USB\\VID_04B4&PID_EA17' in str(refused.value)
    assert 'Bootloader' not in str(refused.value)


class _RunningFirmware:
    """A pyusb device: the FX2 running its firmware, on WinUSB."""

    def is_kernel_driver_active(self, interface):
        return False

    def set_configuration(self):
        pass


def test_a_running_scope_is_not_refused_for_an_unbound_bootloader_identity(monkeypatch):
    # 8613 is every EEPROM-less Cypress FX2's bootloader: an unbound one may
    # be another vendor's board, and says nothing about a Lumascope already
    # running its firmware, as it is after an earlier launch.
    seen = {
        FX2_BOOTLOADER: [_binding(FX2_BOOTLOADER, present=True, service=None)],
        FX2_FIRMWARE: [_binding(FX2_FIRMWARE, present=True, service='WinUSB')],
    }
    # a stand-in by design: SetupAPI, absent off Windows.
    monkeypatch.setattr('drivers.winusb_iso.device_bindings', lambda identity: seen[identity])
    # a stand-in by design: the USB bus, holding the running firmware.
    monkeypatch.setattr(
        fx2driver,
        'usb',
        SimpleNamespace(
            core=SimpleNamespace(
                find=lambda idVendor, idProduct: (
                    _RunningFirmware() if idProduct == fx2driver.PID_APP else None
                ),
                USBError=OSError,
            ),
            util=SimpleNamespace(
                claim_interface=lambda dev, i: None, dispose_resources=lambda dev: None
            ),
        ),
        raising=False,
    )

    fx2driver._FX2Connection(fx2driver._WinUsbTransport())
