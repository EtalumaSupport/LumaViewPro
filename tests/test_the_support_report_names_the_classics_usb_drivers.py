# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report names the driver Windows bound to each Lumascope Classic identity.

A Classic on a wrong or missing driver does not come up, and the report is
what support reads. It writes ``diagnostics.read_camera_usb_bindings``, the
camera connect's own read, as text and as JSON, and says when the host binds
no driver at all.

The simulated FX2 is the bus, its binding the simulation's fault door.
"""

import json

import pytest

from _lumascope.drivers.usb_driver_binding import (
    FX2_BOOTLOADER,
    FX2_FIRMWARE,
    SETUPAPI_LOG,
    SETUPAPI_SECTIONS,
    setupapi_log_excerpt,
    signing_facts,
)
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.tech_support_report import TechSupportReport
from tests.settings_fixtures import complete_settings


@pytest.fixture
def fx2_session():
    session = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    yield session
    session.shutdown()


def test_a_classic_on_libusbk_is_written_not_ours_naming_its_driver(fx2_session, tmp_path):
    fx2_session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, 'libusbK', problem_code=None)

    TechSupportReport(session=fx2_session)._step_camera_usb_bindings(tmp_path)

    written = (tmp_path / 'camera_usb_bindings' / 'camera_usb_bindings.txt').read_text()
    assert f'Lumascope Classic ({FX2_FIRMWARE}): NOT usable' in written
    assert f'{FX2_FIRMWARE}\\SIMULATED (present): libusbK from simulated' in written
    assert (
        f"Lumascope Classic (Bootloader) ({FX2_BOOTLOADER}): on LumaViewPro's driver, started"
        in written
    )
    recorded = json.loads(
        (tmp_path / 'camera_usb_bindings' / 'camera_usb_bindings.json').read_text()
    )
    assert [(i['hardware_id'], i['usable']) for i in recorded] == [
        (FX2_BOOTLOADER, True),
        (FX2_FIRMWARE, False),
    ]


def test_our_driver_windows_did_not_start_is_written_with_its_problem_code(fx2_session, tmp_path):
    fx2_session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, 'WinUSB', problem_code=31)

    TechSupportReport(session=fx2_session)._step_camera_usb_bindings(tmp_path)

    written = (tmp_path / 'camera_usb_bindings' / 'camera_usb_bindings.txt').read_text()
    assert f'Lumascope Classic ({FX2_FIRMWARE}): NOT usable' in written
    assert (
        "LumaViewPro's, not started by Windows, problem code 31 (This device is not working "
        'properly because Windows cannot load the drivers required for this device.)'
    ) in written
    # Where it is, and the interface GUID its package registered: the code
    # 31 of 2026-11-11 was the INF's GUID.
    assert (
        'at Port_#0001.Hub_#0001 on USB\\ROOT_HUB30\\SIMULATED, interface GUIDs {SIMULATED}'
    ) in written


def test_a_scope_with_no_fx2_says_the_bindings_were_not_read(sim_session, tmp_path):
    TechSupportReport(sim_session)._step_camera_usb_bindings(tmp_path)

    written = (tmp_path / 'camera_usb_bindings' / 'camera_usb_bindings.txt').read_text()
    assert 'Not read: this host binds no driver to a USB device.' in written
    assert not (tmp_path / 'camera_usb_bindings' / 'camera_usb_bindings.json').exists()


def _section(device: str, *body: str) -> str:
    return '\n'.join([f'>>>  [Device Install (Hardware initiated) - {device}]', *body])


def test_the_install_log_keeps_the_newest_sections_naming_a_lumascope_vendor(tmp_path):
    # The log is the one place Windows says why a package bound, did not,
    # or bound and did not start.
    sections = [
        _section('USB\\VID_046D&PID_C52B\\1', '     a mouse'),
        _section('USB\\VID_04B4&PID_8613\\5&1', '!!!  dvi: Device not started: code 31'),
        _section('USB\\VID_2676&PID_BA03\\2', '     a Basler'),
        *(_section(f'USB\\vid_04b4&pid_ea17\\{n}', '     ok') for n in range(25)),
    ]
    (tmp_path / SETUPAPI_LOG).write_text('preamble\n' + '\n'.join(sections) + '\n')

    excerpt = setupapi_log_excerpt(str(tmp_path))

    assert 'a mouse' not in excerpt and 'preamble' not in excerpt
    # The newest twenty, oldest first: the code 31 and the Basler are older.
    assert 'code 31' not in excerpt and 'a Basler' not in excerpt
    assert excerpt.count('>>>  [') == SETUPAPI_SECTIONS
    assert excerpt.rstrip().endswith('     ok') and 'vid_04b4&pid_ea17\\24' in excerpt


def test_an_unreadable_install_log_is_said_so(tmp_path):
    assert 'unreadable' in setupapi_log_excerpt(str(tmp_path))


def test_the_signing_facts_say_each_one_windows_did_not_give():
    registry = {
        'CurrentBuild': '22631',
        'UBR': 4460,
        'ProductName': 'Windows 10 Pro',
        'DisplayVersion': '23H2',
        'SystemStartOptions': ' NOEXECUTE=OPTIN  TESTSIGNING',
    }

    def read_value(path, name):
        # a stand-in by design: the Windows registry, absent off Windows.
        if name not in registry:
            raise FileNotFoundError(2, 'The system cannot find the file specified')
        return registry[name]

    facts = signing_facts(read_value)

    assert (
        facts['Windows']
        == 'ProductName Windows 10 Pro, DisplayVersion 23H2, CurrentBuild 22631, UBR 4460'
    )
    assert facts['Test signing'] == 'on'
    assert (
        facts['Secure Boot'].startswith('unreadable (')
        and 'UEFISecureBootEnabled' in facts['Secure Boot']
    )


def test_off_windows_the_report_says_the_windows_facts_were_not_read(fx2_session, tmp_path):
    TechSupportReport(session=fx2_session)._step_windows_drivers(tmp_path)

    written = (tmp_path / 'windows_drivers' / 'windows_drivers.txt').read_text()
    assert written == 'Not read: this host is not Windows.\n'
