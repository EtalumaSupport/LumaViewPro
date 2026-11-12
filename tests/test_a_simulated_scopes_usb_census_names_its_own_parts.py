# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated scope's USB census names the scope's own parts, with their faults.

The simulated bus is built from the simulated scope, never the host: a
Classic holds its FX2 under the driver Windows has it on, so a code 31
reads in the census as it does on the build PC; an EL-0940 scope holds its
LED board and its motor board under the IDs their drivers find them by;
a device whose descriptor Windows could not read is listed with its
problem code, as Windows lists one.
"""

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
    assert classic.binding.ours and not classic.binding.started
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
    assert unidentified.vendor is None and unidentified.binding.problem_code == 43
    assert census.source == 'the simulated bus'
