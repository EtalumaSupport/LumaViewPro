# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated scope's USB bus, as the USB census reads it.

Built from the simulated scope's parts, so a simulated scope's census
names what the scope is made of, never the host's own devices: a Classic
holds its FX2, under the driver Windows has each of its identities on
(``SimulatedFX2Transport.bindings``, its faults the simulator's
``camera_usb_bindings``); an EL-0940 scope holds its LED board and, when it
has motion, its motor board, under the IDs their drivers find them by. A
device whose descriptor Windows could not read can be added, as Windows
lists one. The bus is read as Windows reads it, port and hub with each
device; the boards' Windows drivers and every speed but the FX2's
(a USB 2.0 high-speed chip) are not simulated, and read as not read.
The simulated camera of an EL-0940 scope is not on this bus: its USB
identity and driver are a Basler's or an IDS's, which the bench read of
the USB census plan gives before it is simulated.
"""

from __future__ import annotations

from _lumascope.drivers.simulated_fx2 import SimulatedFX2Transport
from _lumascope.drivers.usb_census import CensusDevice, UsbCensus, ids_of
from _lumascope.drivers.usb_driver_binding import (
    CENSUS_VENDORS,
    FX2_BOOTLOADER,
    FX2_FIRMWARE,
    DeviceBinding,
)

# The IDs each EL-0940 board's driver finds it by (``ledboard.py``,
# ``motorboard.py``).
_LED_BOARD = ('0424', '704C')
_MOTOR_BOARD = ('2E8A', '0005')
# What Windows lists for a device whose descriptor request failed: Device
# Manager's "Unknown USB Device (Device Descriptor Request Failed)".
_UNIDENTIFIED = ('0000', '0002')
_DESCRIPTOR_FAILED = 43
# The FX2 (CY7C68013) is a USB 2.0 high-speed device.
_FX2_SPEED_MBPS = 480.0


class SimulatedUsbBus:
    """The USB bus of one simulated scope.

    Args:
        fx2: The simulated FX2's transport on a Classic; None on an EL-0940 scope.
        led_board: Whether the scope's EL-0940 LED board is on the bus.
        motor_board: Whether the scope's EL-0940 motor board is on the bus.
        unidentified_devices: How many devices Windows could not read the
            descriptor of are on the bus.
    """

    def __init__(
        self,
        *,
        fx2: SimulatedFX2Transport | None,
        led_board: bool,
        motor_board: bool,
        unidentified_devices: int,
    ):
        self._fx2 = fx2
        self._led_board = led_board
        self._motor_board = motor_board
        self._unidentified_devices = unidentified_devices

    def census(self) -> UsbCensus:
        """The census of the bus now, ports numbered in the order the parts are listed."""
        devices: list[CensusDevice] = []
        if self._fx2 is not None:
            for identity in (FX2_BOOTLOADER, FX2_FIRMWARE):
                vendor_id, product_id = ids_of(identity)
                for binding in self._fx2.bindings(identity):
                    devices.append(
                        CensusDevice(
                            vendor_id=vendor_id,
                            product_id=product_id,
                            vendor=CENSUS_VENDORS[vendor_id],
                            instance_id=binding.instance_id,
                            location=binding.location,
                            present=binding.present,
                            binding=binding,
                            speed_mbps=_FX2_SPEED_MBPS,
                            port_best_mbps=None,
                            unreadable=(),
                        )
                    )
        boards = [_LED_BOARD] * self._led_board + [_MOTOR_BOARD] * self._motor_board
        for port, ids in enumerate(boards, start=len(devices) + 1):
            devices.append(_device(ids, port, binding=None))
        for port in range(len(devices) + 1, len(devices) + 1 + self._unidentified_devices):
            instance_id = _instance_id(_UNIDENTIFIED, port)
            failed = DeviceBinding(
                instance_id=instance_id,
                present=True,
                service=None,
                provider=None,
                inf=None,
                version=None,
                date=None,
                problem_code=_DESCRIPTOR_FAILED,
                problem_status=None,
                location=_location(port),
                parent=_ROOT_HUB,
                interface_guids=None,
                ours=False,
                unreadable=(),
            )
            devices.append(_device(_UNIDENTIFIED, port, binding=failed))
        return UsbCensus(source='the simulated bus', devices=tuple(devices))


_ROOT_HUB = 'USB\\ROOT_HUB30\\SIMULATED'


def _location(port: int) -> str:
    return f'Port_#{port:04d}.Hub_#0001'


def _instance_id(ids: tuple[str, str], port: int) -> str:
    return f'USB\\VID_{ids[0]}&PID_{ids[1]}\\SIMULATED&{port}'


def _device(ids: tuple[str, str], port: int, *, binding: DeviceBinding | None) -> CensusDevice:
    return CensusDevice(
        vendor_id=ids[0],
        product_id=ids[1],
        vendor=CENSUS_VENDORS.get(ids[0]),
        instance_id=_instance_id(ids, port),
        location=_location(port),
        present=True,
        binding=binding,
        speed_mbps=None,
        port_best_mbps=None,
        unreadable=(),
    )
