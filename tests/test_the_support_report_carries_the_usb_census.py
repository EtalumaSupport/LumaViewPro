# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report carries the USB census and, on Windows, Device Manager's whole list.

Support reads the report on a PC it cannot inspect, so the report writes
the census bring-up logs, read again through ``get_usb_census``, as text
and JSON; and on Windows every present device, its class, status and
problem code, from the same walk of Windows' device list the census
reads, in place of the ``wmic`` reads current Windows 11 does not have
(#775).
"""

import json
import subprocess

import pytest

from _lumascope.drivers import usb_census
from _lumascope.drivers.usb_driver_binding import DeviceProperties
from _lumascope.modules import tech_support_report
from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.tech_support_report import TechSupportReport
from tests.settings_fixtures import complete_settings


@pytest.fixture
def el_0940_session():
    session = ScopeSession.create(
        complete_settings(microscope='LS850T'),
        simulate=True,
        simulated=SimulatedScope(unidentified_usb_devices=1),
        warn_pre_release=False,
    )
    yield session
    session.shutdown()


def test_the_report_writes_the_census_as_text_and_json(el_0940_session, tmp_path):
    TechSupportReport(el_0940_session)._step_usb_census(tmp_path)

    text = (tmp_path / 'usb_census' / 'usb_census.txt').read_text()
    assert text.startswith('USB census, read through the simulated bus')
    assert 'Microchip (the EL-0940 LED board) 0424:704C' in text
    assert 'not a census vendor 0000:0002' in text and 'problem code 43' in text
    body = json.loads((tmp_path / 'usb_census' / 'usb_census.json').read_text())
    assert body['unreadable'] is None
    assert [d['vendor_id'] for d in body['devices']] == ['0424', '2E8A', '0000']


def test_device_managers_list_names_each_device_its_class_and_whether_windows_started_it():
    lines = usb_census.device_list_lines(
        [
            DeviceProperties(
                'USB\\VID_2676&PID_BA02\\24123456',
                {'name': 'Basler ace', 'class': 'USB', 'status': 0x0180200A},
                {},
            ),
            DeviceProperties(
                'USB\\VID_0000&PID_0002\\5&2B3C&0&4',
                {'class': 'USB', 'status': 0x01806400, 'problem_code': 43},
                {'name': 'type 0x19'},
            ),
        ]
    )

    assert lines == [
        'USB\\VID_2676&PID_BA02\\24123456: Basler ace, class USB, status 0x0180200a, started',
        'USB\\VID_0000&PID_0002\\5&2B3C&0&4: unreadable (type 0x19), class USB, '
        'status 0x01806400, not started, problem code 43',
    ]


def test_on_windows_the_usb_devices_are_listed_from_the_walk_and_no_wmic_runs(monkeypatch):
    # a stand-in by design: Windows and its device list, which this host has not.
    monkeypatch.setattr(tech_support_report.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(
        tech_support_report, 'read_device_list', lambda: ['USB\\ROOT_HUB30\\4&1: root hub']
    )

    def no_subprocess(*args, **kwargs):
        raise AssertionError(f'a subprocess ran: {args}')

    monkeypatch.setattr(subprocess, 'run', no_subprocess)
    # a stand-in by design: the host's serial ports, which no test may touch.
    monkeypatch.setattr('serial.tools.list_ports.comports', lambda include_links: [])

    collected = dict(tech_support_report._collect_usb_devices())

    assert collected['device_list'] == 'USB\\ROOT_HUB30\\4&1: root hub\n'
    assert set(collected) == {'device_list', 'pyserial_ports'}
