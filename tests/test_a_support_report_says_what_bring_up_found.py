# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every support ZIP says what bring-up found and which plugins loaded.

The report reported presence from live probes and never the cause, and
nothing read the plugin health ledger. Now both ZIPs, the logs zip
included, carry ``bring_up.json`` (each part, why it is not up, the
substitutions, the settings file set aside, or why there is no record) and
``plugins.json`` (each namespace's health and the plugins that did not
load, or that this host has no registry).
"""

from __future__ import annotations

import json
import zipfile

import pytest

from drivers.registry import DriverFallback
from modules.lumascope_api.bring_up import MOTOR
from modules.plugins import PluginRegistry
from modules.scope_session import ScopeSession
from modules.tech_support_report import TechSupportReport
from tests.settings_fixtures import complete_settings
from tests.test_bring_up_is_a_record import _bring_up


@pytest.fixture
def session(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield session
    session.shutdown()


def _read(zip_path, name):
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(name))


def test_the_logs_zip_carries_the_bring_up_record(session, tmp_path):
    saved = session.make_logs_zip(output_dir=tmp_path / 'out')
    body = _read(saved.path, 'bring_up.json')
    record = session.bring_up_record()
    assert [part['part'] for part in body['parts']] == [part.part for part in record.parts]
    assert [part['up'] for part in body['parts']] == [part.up for part in record.parts]


def test_a_part_that_did_not_come_up_is_in_the_zip_with_its_cause(monkeypatch, tmp_path):
    session = _bring_up(
        monkeypatch,
        tmp_path,
        motor=DriverFallback('port_in_use', ('MotorBoard',)),
        microscope='LS850T',
    )
    try:
        saved = session.make_logs_zip(output_dir=tmp_path / 'out')
    finally:
        session.shutdown()
        session.scope.disconnect()
    motor = next(
        part for part in _read(saved.path, 'bring_up.json')['parts'] if part['part'] == MOTOR
    )
    assert (motor['up'], motor['cause'], motor['cause_words']) == (
        False,
        'port_in_use',
        'port in use',
    )


def test_a_report_with_no_scope_says_why_it_has_no_record(tmp_path):
    report = TechSupportReport()
    report.diag.build_failure = FileNotFoundError('scopes.json is missing')
    zip_path = report.generate_logs_only(output_dir=tmp_path / 'out')
    body = _read(zip_path, 'bring_up.json')
    assert body['record'] is None
    assert 'scopes.json is missing' in body['why']


def test_a_plugin_that_did_not_load_is_in_the_zip(tmp_path):
    registry = PluginRegistry()
    registry.record_load_failure('broken', '1.0', 'it needs numpy2')
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path)), simulate=True, plugin_health=registry.health
    )
    try:
        saved = session.make_logs_zip(output_dir=tmp_path / 'out')
    finally:
        session.shutdown()
    body = _read(saved.path, 'plugins.json')
    assert body['not_loaded'] == [{'name': 'broken', 'version': '1.0', 'reason': 'it needs numpy2'}]
    assert sorted(ns['namespace'] for ns in body['namespaces']) == [
        'live_processing',
        'post_processing',
        'rest',
        'ui',
    ]


def test_a_host_with_no_plugins_says_so(session, tmp_path):
    saved = session.make_logs_zip(output_dir=tmp_path / 'out')
    assert _read(saved.path, 'plugins.json') == {
        'plugins': None,
        'why': 'no plugin registry on this host',
    }


def test_a_full_report_on_the_simulator_is_saved_with_both_files(session, tmp_path, monkeypatch):
    # The whole report, every step, as the panel's button makes it: a step
    # that raises out of the report (not into it) costs the ZIP. The host's
    # USB inventory lists this machine's real serial ports, which no test
    # may touch; it is the one host read stood in for.
    from modules import tech_support_report

    monkeypatch.setattr(tech_support_report, '_collect_usb_devices', lambda: [])
    saved = session.make_support_report(output_dir=tmp_path / 'out')
    assert saved.title == 'Support Report Saved'
    with zipfile.ZipFile(saved.path) as zf:
        names = zf.namelist()
    assert {'bring_up.json', 'plugins.json'} <= set(names)
    assert not [name for name in names if name.endswith('_ERROR.txt')]
