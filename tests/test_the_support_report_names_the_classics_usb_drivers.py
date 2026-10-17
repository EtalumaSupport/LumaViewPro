# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report names the driver Windows bound to each Lumascope Classic identity.

A Classic on a wrong or missing driver does not come up, and the report is
what support reads. It writes ``diagnostics.get_camera_usb_bindings``, the
camera connect's own read, as text and as JSON, and says when the host binds
no driver at all.

The simulated FX2 is the bus, its binding the simulation's fault door.
"""

import json

import pytest

from drivers.usb_driver_binding import FX2_BOOTLOADER, FX2_FIRMWARE
from modules.scope_session import ScopeSession
from modules.tech_support_report import TechSupportReport
from tests.settings_fixtures import complete_settings


@pytest.fixture
def fx2_session():
    session = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    yield session
    session.shutdown()


def test_a_classic_on_libusbk_is_written_unusable_naming_its_driver(fx2_session, tmp_path):
    fx2_session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, 'libusbK')

    TechSupportReport(session=fx2_session)._step_camera_usb_bindings(tmp_path)

    written = (tmp_path / 'camera_usb_bindings' / 'camera_usb_bindings.txt').read_text()
    assert f'Lumascope Classic ({FX2_FIRMWARE}): NOT USABLE; needs one of winusb' in written
    assert f'{FX2_FIRMWARE}\\SIMULATED (present): libusbK from simulated' in written
    assert f'Lumascope Classic (Bootloader) ({FX2_BOOTLOADER}): usable' in written
    recorded = json.loads(
        (tmp_path / 'camera_usb_bindings' / 'camera_usb_bindings.json').read_text()
    )
    assert [(i['hardware_id'], i['usable']) for i in recorded] == [
        (FX2_BOOTLOADER, True),
        (FX2_FIRMWARE, False),
    ]


def test_a_scope_with_no_fx2_says_the_bindings_were_not_read(sim_scope, tmp_path):
    TechSupportReport(scope=sim_scope)._step_camera_usb_bindings(tmp_path)

    written = (tmp_path / 'camera_usb_bindings' / 'camera_usb_bindings.txt').read_text()
    assert 'Not read: this host binds no driver to a USB device.' in written
    assert not (tmp_path / 'camera_usb_bindings' / 'camera_usb_bindings.json').exists()
