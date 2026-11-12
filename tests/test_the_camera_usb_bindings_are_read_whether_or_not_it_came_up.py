# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every client reads which driver Windows has bound to a Lumascope Classic.

``DiagnosticsAPI.get_camera_usb_bindings`` answers both USB identities, the
bootloader (8613) and the running firmware (EA17), from the same read the
camera's connect checks, so REST, the SDK and the support report see what
the connect saw, also when the camera did not come up. A present device on
no driver, or on a package not LumaViewPro's, is unusable.

The simulated FX2 answers on a simulated LS620, its binding the simulation's
fault door; the Windows read is stood in where SetupAPI is absent.
"""

import pytest

from _lumascope.drivers import fx2driver
from _lumascope.drivers.usb_driver_binding import FX2_BOOTLOADER, FX2_FIRMWARE, DeviceBinding
from _lumascope.modules.lumascope_api.diagnostics import UsbDeviceBinding, UsbIdentityBindings
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def fx2_session():
    session = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    yield session
    session.shutdown()


def test_a_simulated_classic_reports_its_running_firmware_on_winusb(fx2_session):
    bootloader, firmware = fx2_session.scope.diagnostics.get_camera_usb_bindings()

    # Uploaded at bring-up, so only the running firmware is on the bus.
    assert bootloader == UsbIdentityBindings(
        hardware_id=FX2_BOOTLOADER,
        name='Lumascope Classic (Bootloader)',
        usable=True,
        devices=(),
    )
    assert (firmware.hardware_id, firmware.usable) == (FX2_FIRMWARE, True)
    assert [(d.present, d.service, d.ours) for d in firmware.devices] == [(True, 'WinUSB', True)]


def test_a_running_firmware_rebound_to_libusbk_reads_unusable(fx2_session):
    fx2_session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, 'libusbK', problem_code=None)

    _, firmware = fx2_session.scope.diagnostics.get_camera_usb_bindings()

    assert firmware.usable is False
    assert [d.service for d in firmware.devices] == ['libusbK']


def test_a_simulated_scope_never_reads_the_hosts_bus(sim_scope, monkeypatch):
    def _read_the_host(identity):
        raise AssertionError('a simulated scope read the host bus')

    # a stand-in by design: the host's device list, which a simulated scope
    # must never reach; on this Mac it would answer None either way.
    monkeypatch.setattr(fx2driver, 'host_bindings', _read_the_host)

    assert sim_scope.diagnostics.get_camera_usb_bindings() is None


def test_windows_reads_each_device_present_or_past_through_setupapi(monkeypatch):
    seen = DeviceBinding(
        instance_id=f'{FX2_BOOTLOADER}\\5&1&0&1',
        present=False,
        service='libusbK',
        provider='libwdi',
        inf='oem7.inf',
        version='6.1.7600.16385',
        date='2024-01-01',
        problem_code=None,
        problem_status=None,
        location='Port_#0002.Hub_#0001',
        parent='USB\\ROOT_HUB30\\4&1&0&0',
        interface_guids=('{479cda5e-5386-49e7-817a-e7839ff46d2d}',),
        ours=False,
        unreadable=(),
    )
    # a stand-in by design: SetupAPI, absent off Windows.
    monkeypatch.setattr('_lumascope.drivers.winusb_iso.device_bindings', lambda identity: [seen])

    assert fx2driver._WinUsbTransport.bindings(FX2_BOOTLOADER) == [seen]
    # The published record carries every field the driver's does.
    assert UsbDeviceBinding(**vars(seen)).inf == 'oem7.inf'


def test_a_host_that_binds_no_driver_to_a_usb_device_reads_none():
    assert fx2driver._LibusbTransport.bindings(FX2_FIRMWARE) is None
