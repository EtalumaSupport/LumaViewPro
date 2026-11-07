# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Lumascope Classic not on LumaViewPro's driver is refused, saying which.

Windows binds the FX2's bootloader (8613) and its running firmware (EA17)
each on its own, and pyusb sees a device only through a driver libusb can
drive, so a scope on no driver, or on one the stream cannot use, came up as
"Camera not found. Check USB cable and power." The connect reads both
bindings, logs them, and refuses a present device on anything but
LumaViewPro's package, on both identities, whatever else could drive it:
LumaView's libusbK and a Zadig WinUSB binding alike, so a package installed
after ours is caught and Repair takes the identity back.

The simulated FX2 is the bus, its binding a fault door; the Windows
transport's own read is stood in where SetupAPI is absent.
"""

from types import SimpleNamespace

import pytest

from _lumascope.drivers import fx2driver
from _lumascope.drivers.exceptions import CameraUsbDriverUnusableError
from _lumascope.drivers.registry import DriverRegistry
from _lumascope.drivers.simulated_fx2 import SimulatedFX2Device, SimulatedFX2Transport
from _lumascope.drivers.usb_driver_binding import (
    FX2_BOOTLOADER,
    FX2_FIRMWARE,
    DeviceBinding,
    refuse_unusable,
)
from _lumascope.modules.exceptions import CameraNotAvailableError
from _lumascope.modules.lumascope_api._lumascope import _camera_failure_cause


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

    with pytest.raises(CameraUsbDriverUnusableError, match=r"libusbK.*not LumaViewPro's"):
        fx2driver._FX2Connection(transport)
    assert transport.device.uploads == 1


def test_a_bootloader_on_libusbk_is_refused_before_the_upload():
    # libusb could upload through it, and it is still LumaView's package.
    transport = _simulated(FX2_BOOTLOADER, 'libusbK')

    with pytest.raises(
        CameraUsbDriverUnusableError, match=r"Bootloader\) .*libusbK.*not LumaViewPro's"
    ):
        fx2driver._FX2Connection(transport)
    assert transport.device.uploads == 0


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

    cause = _camera_failure_cause(caught.value, False)
    notice = CameraNotAvailableError(cause)
    assert cause == 'camera_driver_unusable'
    assert notice.title == 'Camera driver not usable'
    assert 'install a LumaViewPro build that carries it' in str(notice)


def _binding(identity, *, present, service, ours=False):
    return DeviceBinding(
        instance_id=f'{identity}\\5&1&0&1',
        present=present,
        service=service,
        provider=None,
        inf=None,
        version=None,
        date=None,
        problem_code=None,
        ours=ours,
    )


def test_a_winusb_binding_from_another_package_is_refused():
    # A Zadig binding is WinUSB, which the stream could read, and not ours.
    zadig = _binding(FX2_FIRMWARE, present=True, service='WinUSB')

    with pytest.raises(CameraUsbDriverUnusableError, match=r"WinUSB.*not LumaViewPro's"):
        refuse_unusable({FX2_FIRMWARE: [zadig]})


def test_the_windows_transport_refuses_a_present_device_and_not_a_past_one(monkeypatch):
    seen = {
        FX2_BOOTLOADER: [_binding(FX2_BOOTLOADER, present=False, service=None)],
        FX2_FIRMWARE: [_binding(FX2_FIRMWARE, present=True, service=None)],
    }
    # a stand-in by design: SetupAPI, absent off Windows.
    monkeypatch.setattr(
        '_lumascope.drivers.winusb_iso.device_bindings', lambda identity: seen[identity]
    )
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
        FX2_FIRMWARE: [_binding(FX2_FIRMWARE, present=True, service='WinUSB', ours=True)],
    }
    # a stand-in by design: SetupAPI, absent off Windows.
    monkeypatch.setattr(
        '_lumascope.drivers.winusb_iso.device_bindings', lambda identity: seen[identity]
    )
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
