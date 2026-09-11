# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests: a bad accessor cannot cost a device-identity line.

Bug shape: each camera driver logged a couple of hand-picked identity
fields about the device it found -- Pylon four of the ~36 a CDeviceInfo
carries, all inside one try whose except logged at debug; IDS only model
and serial; FX2 nothing but the PID it matched on. Everything describing
the host-side transport was therefore absent from every support bundle,
and one failing accessor dropped the rest silently. Two machines running
byte-identical SDK, binding and kernel driver behaved differently, and
no log we had could tell them apart.

Only the degradation path is pinned here, for all three drivers. That a
given SDK populates a given field on real hardware is a bench fact, not
a unit-test one, and a passing bundle shows it directly; a silently
truncated line is the failure no bundle can show, because the line
simply is not there.

The three describers deliberately do not share an implementation: Pylon
answers "what do you carry?" generically, peak requires a known accessor
per field, and libusb exposes plain descriptor attributes. Only the
rendering convention is common, so bundles stay greppable alike.
"""

from drivers.fx2driver import describe_usb_device
from drivers.idscamera import describe_device_descriptor
from drivers.pyloncamera import describe_device_info


class _StubDeviceInfo:
    """Stand-in for the SDK's generic property interface.

    Mirrors the shape verified against an enumerated device:
    ``GetPropertyNames()`` returns ``(count, names)`` and
    ``GetPropertyValue(name)`` returns ``(ok, value)``.
    """

    def __init__(self, props, unavailable=(), raising=()):
        self._props = dict(props)
        self._unavailable = set(unavailable)
        self._raising = set(raising)

    def GetPropertyNames(self):
        names = tuple(self._props)
        return len(names), names

    def GetPropertyValue(self, name):
        if name in self._raising:
            raise RuntimeError(f'accessor exploded for {name}')
        if name in self._unavailable:
            return False, ''
        return True, self._props[name]


def test_one_raising_accessor_costs_one_field_not_the_line():
    dev = _StubDeviceInfo(
        {'ModelName': 'daA3840-45um', 'PortID': 'USB_VID_2676', 'UsbDriverType': 'pylon'},
        raising=('PortID',),
    )

    described = describe_device_info(dev)

    assert 'daA3840-45um' in described and 'pylon' in described, (
        f'a single failing accessor must not suppress the properties that '
        f'read fine; got: {described}'
    )
    assert 'PortID' in described, (
        f'the failing property must still be named so the gap is visible '
        f'rather than silent; got: {described}'
    )


def test_unreadable_device_info_names_its_reason_instead_of_raising():
    class _Hostile:
        def GetPropertyNames(self):
            raise RuntimeError('node map detached')

    described = describe_device_info(_Hostile())

    assert 'node map detached' in described, (
        f'an unusable device info must record why rather than vanish or '
        f'raise into connect(); got: {described}'
    )


def test_ids_descriptor_survives_an_accessor_the_transport_lacks():
    """peak exposes typed accessors; not every TL implements all of them."""

    class _Descriptor:
        def __getattr__(self, name):
            if name == 'Version':

                def _raise():
                    raise RuntimeError('not supported by this transport layer')

                return _raise
            return lambda: f'<{name}>'

    described = describe_device_descriptor(_Descriptor())

    assert '<ModelName>' in described and '<SerialNumber>' in described, (
        f'one unsupported accessor must not suppress the readable ones; got: {described}'
    )
    assert 'Version' in described, (
        f'the unsupported accessor must still be named so the gap is visible; got: {described}'
    )


def test_fx2_device_description_places_the_device_on_its_controller():
    """bus / address / port chain / speed are the attribution fields."""

    class _Dev:
        address = 7
        bus = 1
        port_number = 3
        port_numbers = (3, 2)
        speed = 2
        bcdUSB = 0x0200
        idVendor = 0x04B4
        idProduct = 0x8613

        def __getattr__(self, name):
            raise AttributeError(name)

    described = describe_usb_device(_Dev())

    for field in ('bus=1', 'address=7', 'port_numbers=(3, 2)', 'speed=2'):
        assert field in described, (
            f'{field} must be recorded so a bundle can place the device on a '
            f'host controller; got: {described}'
        )
    assert 'bcdUSB=0x0200' in described and 'idVendor=0x04B4' in described, (
        f'IDs and BCD fields must render in hex to stay readable; got: {described}'
    )
