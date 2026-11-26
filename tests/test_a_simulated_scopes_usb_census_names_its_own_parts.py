# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated scope's USB census names the scope's own parts, with their faults.

The simulated bus is built from the simulated scope, never the host: a
Classic holds its FX2 under the driver Windows has it on, so a code 31
reads in the census as it does on the build PC; an EL-0940 scope holds its
LED board and its motor board under the IDs their drivers find them by;
a device whose descriptor Windows could not read is listed with its
problem code, as Windows lists one.
"""

import dataclasses

from _lumascope.drivers.simulated_fx2 import SimulatedFX2Device, SimulatedFX2Transport
from _lumascope.drivers.simulated_usb_bus import SimulatedUsbBus
from _lumascope.drivers.usb_driver_binding import FX2_BOOTLOADER


def test_a_classic_on_our_driver_windows_did_not_start_reads_as_code_31():
    fx2 = SimulatedFX2Transport(SimulatedFX2Device())
    fx2.bind(FX2_BOOTLOADER, 'WinUSB', 31)

    census = SimulatedUsbBus(
        fx2=fx2, led_board=False, motor_board=False, unidentified_devices=0
    ).census()

    (classic,) = census.devices
    assert (classic.vendor_id, classic.product_id) == ('04B4', '8613')
    assert classic.driver.ours and not classic.driver.started
    assert 'problem code 31' in classic.describe()


def test_an_el_0940_scope_holds_its_two_boards_and_an_unidentified_device_with_its_code():
    census = SimulatedUsbBus(
        fx2=None, led_board=True, motor_board=True, unidentified_devices=1
    ).census()

    assert [(d.vendor_id, d.product_id) for d in census.devices] == [
        ('0424', '704C'),
        ('2E8A', '0005'),
        ('0000', '0002'),
    ]
    unidentified = census.devices[2]
    assert unidentified.vendor is None and unidentified.driver.problem_code == 43
    assert census.source == 'the simulated bus'


def test_each_devices_census_line_names_its_identity_once_beside_its_driver():
    """A device's census line takes its identity from the device and its driver's words from its driver.

    The driver carries no identity of its own, so the device's instance ID,
    presence and port are stated once: a Classic on a driver, a board on
    none, and a device Windows could not identify each read as the line a
    bring-up logs.
    """
    fx2 = SimulatedFX2Transport(SimulatedFX2Device())
    fx2.bind(FX2_BOOTLOADER, 'WinUSB', 31)

    devices = (
        SimulatedUsbBus(fx2=fx2, led_board=True, motor_board=False, unidentified_devices=1)
        .census()
        .devices
    )

    assert [device.describe() for device in devices] == [
        'Cypress (the Lumascope Classic) 04B4:8613: USB\\VID_04B4&PID_8613\\SIMULATED (present): '
        "WinUSB from simulated None None dated None, LumaViewPro's, not started by Windows, "
        'problem code 31 (This device is not working properly because Windows cannot load the '
        'drivers required for this device.), at Port_#0001.Hub_#0001 on '
        'USB\\ROOT_HUB30\\SIMULATED, interface GUIDs {SIMULATED}, 480 Mb/s',
        'Microchip (the EL-0940 LED board) 0424:704C: USB\\VID_0424&PID_704C\\SIMULATED&2 '
        '(present) at Port_#0002.Hub_#0001',
        'not a census vendor 0000:0002: USB\\VID_0000&PID_0002\\SIMULATED&3 (present): no driver, '
        'problem code 43 (Windows has stopped this device because it has reported problems.), '
        'at Port_#0003.Hub_#0001 on USB\\ROOT_HUB30\\SIMULATED',
    ]
    driven = [device.driver for device in devices if device.driver is not None]
    assert len(driven) == 2
    assert all(
        {'instance_id', 'present', 'location'}.isdisjoint(f.name for f in dataclasses.fields(d))
        for d in driven
    )
