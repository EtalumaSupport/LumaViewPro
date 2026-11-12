# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An FX2 not found logs every Cypress device the USB library can see, and the libusb it sees through.

Windows can list a Classic that LumaViewPro's libusb cannot open: on no
driver, on a driver libusb does not reach, or on one Windows did not start.
"Not found" alone cannot tell that from a scope that is not plugged in; the
line beside it can, when the support bundle is read with Device Manager's
view of the same PC.
"""

from types import SimpleNamespace

from _lumascope.drivers import fx2driver
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from tests.settings_fixtures import complete_settings


def test_a_simulated_classic_off_the_bus_logs_that_the_bus_holds_none(caplog):
    with caplog.at_level('INFO', logger='LVP'):
        session = ScopeSession.create(
            complete_settings(microscope='LS620'),
            simulate=True,
            simulated=SimulatedScope(camera_present=False),
            warn_pre_release=False,
        )
        session.shutdown()

    assert any(
        'no FX2 found; the Cypress devices the simulated bus holds: none' in r.getMessage()
        for r in caplog.records
    )


class _Device(SimpleNamespace):
    """A pyusb device: the descriptor fields ``describe_usb_device`` reads."""


def test_the_usb_library_names_each_cypress_device_and_its_speed(monkeypatch):
    other_board = _Device(idVendor=0x04B4, idProduct=0x00F1, speed=3, bus=1, address=7)
    # a stand-in by design: the USB bus, which this host's tests never reach.
    monkeypatch.setattr(
        fx2driver,
        'usb',
        SimpleNamespace(
            core=SimpleNamespace(
                find=lambda find_all, custom_match: filter(custom_match, [other_board]),
                NoBackendError=ValueError,
            )
        ),
        raising=False,
    )

    seen = fx2driver._PyusbTransport().vendor_devices()

    assert seen.startswith('the Cypress (0x04B4) devices libusb ')
    assert 'idProduct=0x00F1' in seen and 'speed=3' in seen


def test_a_usb_library_that_cannot_list_says_so(monkeypatch):
    def no_backend(find_all, custom_match):
        raise ValueError('No backend available')

    # a stand-in by design: a host whose libusb did not load.
    monkeypatch.setattr(
        fx2driver,
        'usb',
        SimpleNamespace(core=SimpleNamespace(find=no_backend, NoBackendError=ValueError)),
        raising=False,
    )

    assert 'unreadable: No backend available' in fx2driver._PyusbTransport().vendor_devices()
