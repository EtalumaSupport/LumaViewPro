# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The USB census names each Lumascope vendor's device on the bus, layer by layer.

A camera or board that does not come up is told by what the host sees:
on Windows, each census vendor's device, present or seen before, its port,
its driver and whether Windows started it, and the speed it negotiated
against the best its port can do, with every other present device Windows
has a problem code for (a device whose descriptor could not be read reports
a generic vendor and product); off Windows, each device the bundled USB
library sees, with its bus, ports and speed. A layer the host could not
read is named on its device; the census goes on. A host whose bus cannot
be read gives a census that says why, never an empty bus.

Windows' walk is built here from recorded property sets, the hub's answer
from its fields; the walk and the hub query themselves run only on Windows.
"""

from types import SimpleNamespace

from _lumascope.drivers import fx2driver
from _lumascope.drivers.exceptions import CameraUsbLookupFailedError
from _lumascope.drivers.usb_census import HubPort, census_from
from _lumascope.drivers.usb_driver_binding import DeviceProperties

_USB2_PORT = 0x3  # USB 1.1 and 2.0
_USB3_PORT = 0x7  # USB 1.1, 2.0 and 3.x
_HIGH_SPEED = 2
_SUPER_SPEED = 3


def _device(instance_id, **values) -> DeviceProperties:
    return DeviceProperties(instance_id=instance_id, values=values, unreadable={})


_BASLER = _device(
    'USB\\VID_2676&PID_BA02\\24123456',
    present=True,
    service='BaslerUsb',
    provider='Basler',
    inf='oem40.inf',
    version='6.3.0.1',
    date='2024-05-02',
    location='Port_#0003.Hub_#0002',
    parent='USB\\ROOT_HUB30\\4&1C2D3E&0&0',
    address=3,
)
_UNIDENTIFIED = _device(
    'USB\\VID_0000&PID_0002\\5&2B3C&0&4', present=True, problem_code=43, address=4
)
_A_MOUSE = _device('USB\\VID_046D&PID_C077\\5&2B3C&0&1', present=True, service='HidUsb')
_A_PAST_CLASSIC = _device('USB\\VID_04B4&PID_8613\\5&1A2B&0&2', present=False)


def _hub(answers):
    def hub_port(device):
        answer = answers[device.instance_id]
        if isinstance(answer, Exception):
            raise answer
        return answer

    return hub_port


def test_a_basler_on_a_usb3_port_at_high_speed_is_named_with_both_speeds():
    (basler,) = census_from(
        [_BASLER], set(), _hub({_BASLER.instance_id: HubPort(_HIGH_SPEED, _USB3_PORT, 0)})
    )

    assert (basler.vendor, basler.vendor_id, basler.product_id) == ('Basler', '2676', 'BA02')
    assert (basler.speed_mbps, basler.port_best_mbps) == (480.0, 5000.0)
    assert basler.binding.service == 'BaslerUsb' and basler.binding.started
    assert '480 Mb/s on a port of 5000 Mb/s' in basler.describe()


def test_every_census_vendor_and_every_present_device_with_a_problem_code_is_listed():
    census = census_from(
        [_A_MOUSE, _BASLER, _UNIDENTIFIED, _A_PAST_CLASSIC],
        set(),
        _hub(
            {
                _BASLER.instance_id: HubPort(_SUPER_SPEED, _USB3_PORT, 0),
                _UNIDENTIFIED.instance_id: HubPort(_HIGH_SPEED, _USB2_PORT, 0),
            }
        ),
    )

    assert [d.instance_id for d in census] == [
        _BASLER.instance_id,
        _UNIDENTIFIED.instance_id,
        _A_PAST_CLASSIC.instance_id,
    ]
    unidentified = census[1]
    assert unidentified.vendor is None and unidentified.binding.problem_code == 43
    assert 'not a census vendor 0000:0002' in unidentified.describe()
    past = census[2]
    assert not past.present and past.speed_mbps is None


def test_a_hub_that_cannot_be_asked_is_named_on_its_device_and_the_census_goes_on():
    refused = CameraUsbLookupFailedError('CreateFileW(USB\\ROOT_HUB30\\4&1C2D3E&0&0) failed: 5')

    (basler,) = census_from([_BASLER], set(), _hub({_BASLER.instance_id: refused}))

    assert basler.speed_mbps is None
    assert basler.unreadable == (f'speed ({refused})',)


def test_a_driver_property_of_an_unread_type_leaves_the_device_listed_without_its_driver():
    odd = DeviceProperties(
        instance_id=_BASLER.instance_id,
        values=dict(_BASLER.values),
        unreadable={'service': 'type 0x19'},
    )

    (basler,) = census_from([odd], set(), _hub({odd.instance_id: None}))

    assert basler.binding is None
    assert basler.unreadable[0] == 'service (type 0x19)'
    assert 'could not be read: service (type 0x19)' in basler.unreadable[1]


def test_the_hub_names_a_super_speed_plus_device_and_a_usb2_port():
    assert HubPort(_SUPER_SPEED, _USB3_PORT, 0x4).speed_mbps == 10000.0
    assert HubPort(_HIGH_SPEED, _USB2_PORT, 0).port_best_mbps == 480.0


class _LibraryDevice(SimpleNamespace):
    """A pyusb device: the descriptor fields the census reads."""


def _bundled_libusb_in_use(monkeypatch):
    # a stand-in by design: the bundled libusb, which the tests' stub pyusb
    # does not load, so the gate decided at import reads it not in use.
    monkeypatch.setattr(
        fx2driver,
        '_FX2_GATE',
        fx2driver._FX2_GATE._replace(library_path='/bundled/libusb-1.0.dylib'),
    )


def _library(monkeypatch, devices):
    _bundled_libusb_in_use(monkeypatch)
    # a stand-in by design: the USB bus, which this host's tests never reach.
    monkeypatch.setattr(
        fx2driver,
        'usb',
        SimpleNamespace(
            core=SimpleNamespace(
                find=lambda find_all, custom_match: filter(custom_match, devices),
                NoBackendError=ValueError,
            )
        ),
        raising=False,
    )


def test_off_windows_each_census_vendors_device_the_library_sees_is_listed(monkeypatch):
    basler = _LibraryDevice(idVendor=0x2676, idProduct=0xBA02, bus=2, port_numbers=(1, 3), speed=3)
    a_mouse = _LibraryDevice(idVendor=0x046D, idProduct=0xC077, bus=1, port_numbers=(2,), speed=2)
    _library(monkeypatch, [a_mouse, basler])

    census = fx2driver.usb_library_census([0x2676, 0x04B4])

    assert census.source.startswith('libusb ') and census.unreadable is None
    (seen,) = census.devices
    assert (seen.vendor, seen.location, seen.speed_mbps) == ('Basler', 'bus 2 port 1.3', 480.0)
    assert seen.binding is None and seen.port_best_mbps is None


def test_off_windows_a_library_that_cannot_list_the_bus_gives_a_census_saying_so(monkeypatch):
    def no_backend(find_all, custom_match):
        raise ValueError('No backend available')

    _bundled_libusb_in_use(monkeypatch)
    # a stand-in by design: a bundled libusb that cannot list the bus.
    monkeypatch.setattr(
        fx2driver,
        'usb',
        SimpleNamespace(core=SimpleNamespace(find=no_backend, NoBackendError=ValueError)),
        raising=False,
    )

    census = fx2driver.usb_library_census([0x2676])

    assert census.devices == ()
    assert census.unreadable == 'could not list the bus: No backend available'


def test_off_windows_a_usb_library_not_in_use_gives_a_census_saying_why(monkeypatch):
    # a stand-in by design: a host whose bundled libusb was refused at import.
    monkeypatch.setattr(
        fx2driver,
        '_FX2_GATE',
        fx2driver._FX2_GATE._replace(
            library_path=None, library_refusal='libusb-package is not installed'
        ),
    )

    census = fx2driver.usb_library_census([0x2676])

    assert census.devices == ()
    assert census.unreadable == 'the USB library is not in use: libusb-package is not installed'
