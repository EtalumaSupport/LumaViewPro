# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Bring-up logs the USB census once, and every client reads it through the API.

The census is read before the LED board's connect touches a Classic's FX2,
so the log holds the bus as bring-up found it, one line per device;
``DiagnosticsAPI.read_usb_census`` reads it again for REST, the SDK and the
support report. A simulated scope's census is its own parts. A bus that
cannot be read is logged as that, and the scope comes up as it would
without a census.
"""

import pytest

from _lumascope.drivers.simulated_usb_bus import SimulatedUsbBus
from _lumascope.drivers.usb_census import UsbCensus
from _lumascope.drivers.usb_driver_binding import FX2_BOOTLOADER
from _lumascope.modules.exceptions import ArgumentRefusedError
from _lumascope.modules.lumascope_api.simulation import SimulatedScope, SimulatedUsbBinding
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


def _session(model, simulated=SimulatedScope()):
    return ScopeSession.create(
        complete_settings(microscope=model),
        simulate=True,
        simulated=simulated,
        warn_pre_release=False,
    )


def test_bring_up_logs_a_classic_on_our_driver_windows_did_not_start(caplog):
    with caplog.at_level('INFO', logger='LVP'):
        session = _session(
            'LS620',
            SimulatedScope(
                camera_usb_bindings=(SimulatedUsbBinding(FX2_BOOTLOADER, 'WinUSB', 31),)
            ),
        )
    try:
        (classic,) = session.scope.diagnostics.read_usb_census().devices
    finally:
        session.shutdown()

    census_lines = [r.getMessage() for r in caplog.records if '[USB census]' in r.getMessage()]
    assert census_lines[0] == '[USB census] 1 device(s), read through the simulated bus'
    assert 'Cypress (the Lumascope Classic) 04B4:8613' in census_lines[1]
    assert 'problem code 31' in census_lines[1]
    assert classic.binding.problem_code == 31 and classic.binding.ours


def test_an_el_0940_scopes_census_names_its_boards_and_an_unidentified_device():
    session = _session('LS850T', SimulatedScope(unidentified_usb_devices=1))
    try:
        census = session.scope.diagnostics.read_usb_census()
    finally:
        session.shutdown()

    assert census.unreadable is None
    assert [(d.vendor_id, d.product_id) for d in census.devices] == [
        ('0424', '704C'),
        ('2E8A', '0005'),
        ('0000', '0002'),
    ]
    assert census.devices[2].binding.problem_code == 43


def test_a_bus_that_cannot_be_read_is_logged_as_that_and_the_scope_comes_up(monkeypatch, caplog):
    # a stand-in by design: a host whose device list cannot be read, which
    # the simulated bus does not yet produce.
    monkeypatch.setattr(
        SimulatedUsbBus,
        'census',
        lambda self: UsbCensus(
            source='the simulated bus', devices=(), unreadable='SetupDiGetClassDevsW failed: 5'
        ),
    )
    with caplog.at_level('INFO', logger='LVP'):
        session = _session('LS850T')
    try:
        assert session.scope.diagnostics.read_usb_census().unreadable == (
            'SetupDiGetClassDevsW failed: 5'
        )
        assert all(part.up for part in session.bring_up_record().parts)
    finally:
        session.shutdown()

    assert (
        '[USB census] unreadable through the simulated bus: SetupDiGetClassDevsW failed: 5'
        in caplog.text
    )


def test_a_negative_or_fractional_count_of_unidentified_devices_is_refused():
    for count in (-1, 1.5):
        with pytest.raises(ArgumentRefusedError) as refused:
            SimulatedScope(unidentified_usb_devices=count)
        assert refused.value.reason == 'fault_bound'
