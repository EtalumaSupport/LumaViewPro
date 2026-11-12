# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What is on the USB bus of each vendor a Lumascope is built from, layer by layer.

A camera or board that does not come up is told and logged by what the
host itself sees, never by its own driver's search alone: a Basler that
pylon does not find may be on the bus at USB 2 speed, a Classic on a
driver Windows did not start. The census reads, for each device of a
vendor in ``CENSUS_VENDORS``, present or seen before:

- on Windows, through the one walk of its USB device list
  (``windows_usb.usb_devices``): the device, its port and hub, the driver
  bound to it and whose package, whether Windows started it, and from its
  hub the speed it negotiated and the best its port can do; and every
  other present USB device Windows has a problem code for, since a device
  whose descriptor could not be read reports a generic vendor and product;
- off Windows, through the bundled USB library (``fx2driver.usb_library_census``,
  the one module that loads it): each present device, its bus and ports,
  and its negotiated speed; the host binds no driver to a USB device there.

The census decides nothing and starts nothing: it is read and logged. A
host whose device list cannot be read gives a census that says why.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from _lumascope.drivers.exceptions import CameraUsbLookupFailedError
from _lumascope.drivers.usb_driver_binding import (
    CENSUS_VENDORS,
    FX2_BOOTLOADER,
    FX2_FIRMWARE,
    DeviceBinding,
    DeviceProperties,
    binding_from,
)

_IDS = re.compile(r'VID_([0-9A-F]{4})&PID_([0-9A-F]{4})', re.IGNORECASE)

# USB_DEVICE_SPEED (usbspec.h), as a hub's connection information gives it.
_HUB_SPEED_MBPS = {0: 1.5, 1: 12.0, 2: 480.0, 3: 5000.0}
# USB_NODE_CONNECTION_INFORMATION_EX_V2 (usbioctl.h): the port's protocols
# and the device's operating flags, each a bit.
_PORT_USB110 = 0x1
_PORT_USB200 = 0x2
_PORT_USB300 = 0x4
_OPERATING_AT_SUPER_SPEED_PLUS = 0x4


@dataclass(frozen=True)
class HubPort:
    """A device's connection as its hub reports it, unparsed.

    ``speed`` is ``USB_DEVICE_SPEED``; ``protocols`` the port's
    ``SupportedUsbProtocols`` bits; ``flags`` the connection's
    ``USB_NODE_CONNECTION_INFORMATION_EX_V2_FLAGS`` bits.
    """

    speed: int
    protocols: int
    flags: int

    @property
    def speed_mbps(self) -> float | None:
        """The speed the device negotiated, in Mb/s; None for a value Windows does not define."""
        if self.flags & _OPERATING_AT_SUPER_SPEED_PLUS:
            return 10000.0
        return _HUB_SPEED_MBPS.get(self.speed)

    @property
    def port_best_mbps(self) -> float | None:
        """The fastest speed the port supports, in Mb/s; None when it names no protocol."""
        if self.protocols & _PORT_USB300:
            return 5000.0
        if self.protocols & _PORT_USB200:
            return 480.0
        if self.protocols & _PORT_USB110:
            return 12.0
        return None


@dataclass(frozen=True)
class CensusDevice:
    """One device on the USB bus, each layer the host could read.

    Attributes:
        vendor_id: The USB vendor ID, four hex digits upper case; None
            where Windows names none (a hub).
        product_id: The USB product ID, as ``vendor_id``.
        vendor: Its name in ``CENSUS_VENDORS``; None for a device of another
            vendor, listed for its problem code.
        instance_id: Windows' instance ID; None off Windows.
        location: Where it is: Windows' port and hub, or the library's bus
            and port path; None when neither is given.
        present: False for a device Windows saw before and does not see now.
        binding: The driver Windows bound to it and whether Windows started
            it; None off Windows, or when Windows gave a property the
            binding is decided by in a type this does not read (named in
            ``unreadable``).
        speed_mbps: The speed it negotiated, in Mb/s; None when not read.
        port_best_mbps: The fastest speed its port supports, in Mb/s; None
            when not read (always off Windows).
        unreadable: Each layer the host could not read for it, with why.
    """

    vendor_id: str | None
    product_id: str | None
    vendor: str | None
    instance_id: str | None
    location: str | None
    present: bool
    binding: DeviceBinding | None
    speed_mbps: float | None
    port_best_mbps: float | None
    unreadable: tuple[str, ...]

    def describe(self) -> str:
        """One line for the log: who it is, where, its driver, its speed."""
        ids = f'{self.vendor_id or "----"}:{self.product_id or "----"}'
        who = f'{self.vendor or "not a census vendor"} {ids}'
        if self.binding is not None:
            what = self.binding.describe()
        else:
            where = f' at {self.location}' if self.location else ''
            state = 'present' if self.present else 'not present'
            what = f'{self.instance_id or "device"} ({state}){where}'
        speed = ''
        if self.speed_mbps is not None:
            speed = f', {self.speed_mbps:g} Mb/s'
            if self.port_best_mbps is not None:
                speed += f' on a port of {self.port_best_mbps:g} Mb/s'
        unread = f', unreadable: {"; ".join(self.unreadable)}' if self.unreadable else ''
        return f'{who}: {what}{speed}{unread}'


@dataclass(frozen=True)
class UsbCensus:
    """The census of one read: what the host read the bus through, and what it found.

    A host whose bus cannot be read gives a census that says so: what is on
    the bus is then not known, which is a fact about the host, told with
    the census and never as an empty bus.

    Attributes:
        source: ``Windows' device list``, or the USB library and its version.
        devices: Each device of a census vendor, then each other present
            device with a problem code, in the order the host listed them;
            empty when ``unreadable``.
        unreadable: Why the host's device list could not be read (Windows'
            device list failed, or off Windows the USB library is not in
            use); None when it was read.
    """

    source: str
    devices: tuple[CensusDevice, ...]
    unreadable: str | None


def ids_of(instance_id: str) -> tuple[str | None, str | None]:
    """A Windows instance ID's vendor and product IDs, upper case; Nones when it names none."""
    match = _IDS.search(instance_id)
    if match is None:
        return None, None
    return match.group(1).upper(), match.group(2).upper()


def census_from(
    devices: Iterable[DeviceProperties],
    ours: set[str],
    hub_port: Callable[[DeviceProperties], HubPort | None],
) -> tuple[CensusDevice, ...]:
    """The census of Windows' USB devices, from what the one walk read of each.

    Args:
        devices: Every USB device the walk read, present or seen before.
        ours: The published INF names, lower case, of our packages.
        hub_port: The device's connection as its hub reports it; None for a
            device not on a hub's port (not present, or a function of a
            composite device). Raises ``CameraUsbLookupFailedError`` when
            the hub could not be asked, which is recorded on the device.
    """
    census = []
    for device in devices:
        vendor_id, product_id = ids_of(device.instance_id)
        vendor = CENSUS_VENDORS.get(vendor_id) if vendor_id else None
        present = bool(device.values.get('present'))
        if vendor is None and not (present and device.values.get('problem_code')):
            continue
        # A binding read carries its own unreadable properties; one refused
        # leaves them to the device's record, with why it was refused.
        unreadable = []
        try:
            binding = binding_from(device, ours)
        except CameraUsbLookupFailedError as e:
            binding = None
            unreadable.extend(f'{name} ({why})' for name, why in device.unreadable.items())
            unreadable.append(f'driver ({e})')
        port = None
        if present:
            try:
                port = hub_port(device)
            except CameraUsbLookupFailedError as e:
                unreadable.append(f'speed ({e})')
        census.append(
            CensusDevice(
                vendor_id=vendor_id,
                product_id=product_id,
                vendor=vendor,
                instance_id=device.instance_id,
                location=device.values.get('location'),
                present=present,
                binding=binding,
                speed_mbps=port.speed_mbps if port else None,
                port_best_mbps=port.port_best_mbps if port else None,
                unreadable=tuple(unreadable),
            )
        )
    return tuple(census)


def read_census() -> UsbCensus:
    """This host's census, read now; a host whose bus cannot be read gives one that says why."""
    if sys.platform == 'win32':
        return _windows_census()
    # local-import: the FX2 driver is the one module that loads the USB library.
    from _lumascope.drivers.fx2driver import usb_library_census

    return usb_library_census(int(vid, 16) for vid in CENSUS_VENDORS)


def _windows_census() -> UsbCensus:
    from _lumascope.drivers.usb_driver_binding import store_inf_folder, store_packages
    from _lumascope.drivers.windows_usb import hub_port, usb_devices  # local-import: Windows only

    source = "Windows' device list"
    try:
        ours = {
            p.inf.lower()
            for identity in (FX2_BOOTLOADER, FX2_FIRMWARE)
            for p in store_packages(identity, store_inf_folder())
            if p.ours
        }
        devices = usb_devices(lambda _instance_id: True)
    except (CameraUsbLookupFailedError, OSError) as e:
        # Told in the census itself: the bus is not known, not empty.
        return UsbCensus(source=source, devices=(), unreadable=str(e))
    return UsbCensus(source=source, devices=census_from(devices, ours, hub_port), unreadable=None)


# cfg.h: the device node's driver is started.
_DN_STARTED = 0x8


def device_list_lines(devices: Iterable[DeviceProperties]) -> list[str]:
    """Device Manager's whole list as lines: each device's name, class, status and problem code.

    A property Windows did not give is said as not given; one in a type this
    does not read is named with its type.
    """
    lines = []
    for device in devices:
        status = device.values.get('status')
        if status is None:
            state = _said(device, 'status')
        else:
            state = f'{status:#010x}, ' + ('started' if status & _DN_STARTED else 'not started')
        code = device.values.get('problem_code')
        problem = f', problem code {code}' if code else ''
        lines.append(
            f'{device.instance_id}: {_said(device, "name")}, class {_said(device, "class")}, '
            f'status {state}{problem}'
        )
    return lines


def _said(device: DeviceProperties, name: str) -> str:
    """A property of the device in words: its value, not given, or unreadable with its type."""
    if name in device.unreadable:
        return f'unreadable ({device.unreadable[name]})'
    return str(device.values[name]) if name in device.values else 'not given'


def read_device_list() -> list[str]:
    """Device Manager's whole list on this Windows host, as lines; the first says so when it cannot be read."""
    from _lumascope.drivers.windows_usb import present_devices  # local-import: Windows only

    try:
        devices = present_devices()
    except CameraUsbLookupFailedError as e:
        # Told in the list itself: the devices are not known, not none.
        return [f"Windows' device list could not be read: {e}"]
    return device_list_lines(devices)
