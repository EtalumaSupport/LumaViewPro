# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A support report collects from the folders of the session that made it.

The captures come from the session's live folder, the data folder from the
session's installation (``scope.source_path``). When the live folder is not
there (an unplugged drive, a deleted folder), each section that reads it
names the folder and why, and collects nothing in its place: the report
once re-read ``current.json`` from the process's installation and, failing
that, collected the first ``Documents/EtalumaCaptures``, ``Etaluma`` or
``LumaViewPro`` it found, as if it were the person's captures. The report
still saves. Beside the data folder's settings file, the report carries
the settings the session is running on, its ``api_key`` values blanked.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import zipfile

import pytest

from _lumascope.modules.protocol import Protocol
from _lumascope.modules.tech_support_report import TechSupportReport
from tests.installation_fixtures import process_logs_at, write_installation

NOT_READ = 'LIVE_FOLDER_NOT_READ.txt'


def _logs_zip(session, tmp_path, monkeypatch) -> zipfile.ZipFile:
    # The process's own logs are not this test's subject: keep the
    # checkout's out of the ZIP.
    process_logs_at(monkeypatch, tmp_path / 'process')
    return zipfile.ZipFile(session.make_logs_zip(output_dir=tmp_path / 'out').path)


@pytest.fixture
def documents(tmp_path, monkeypatch):
    """A ``Documents/EtalumaCaptures`` holding a protocol, where the report once guessed."""
    import platformdirs

    docs = tmp_path / 'Documents'
    run = docs / 'EtalumaCaptures' / 'run_A'
    run.mkdir(parents=True)
    (run / 'guessed_protocol.tsv').write_text(f'{Protocol.PROTOCOL_FILE_HEADER}\n')
    # a stand-in by design: the user's Documents, which platformdirs answers
    # and the suite's conftest replaces.
    monkeypatch.setattr(platformdirs, 'user_documents_dir', lambda: str(docs))
    return docs


def test_a_live_folder_that_is_not_there_is_named_and_nothing_is_collected_in_its_place(
    sim_session_on_its_own_installation, documents, tmp_path, monkeypatch
):
    session = sim_session_on_its_own_installation
    live = pathlib.Path(session.setting('live_folder'))
    shutil.rmtree(live)

    report = _logs_zip(session, tmp_path, monkeypatch)

    for section in ('recent_protocols', 'video_receipts'):
        assert str(live) in report.read(f'{section}/{NOT_READ}').decode(), section
    assert not [name for name in report.namelist() if 'guessed_protocol' in name]


def test_the_disk_speed_is_not_tested_without_the_live_folder(
    sim_session_on_its_own_installation, tmp_path
):
    session = sim_session_on_its_own_installation
    live = pathlib.Path(session.setting('live_folder'))
    shutil.rmtree(live)
    bundle = tmp_path / 'bundle'
    bundle.mkdir()

    TechSupportReport(session)._step_disk_speed(bundle)

    note = (bundle / 'disk_speed' / NOT_READ).read_text()
    assert note.startswith('Disk speed not tested: ')
    assert str(live) in note
    assert not (bundle / 'disk_speed' / 'disk_speed.json').exists()


@pytest.mark.skipif(
    os.name == 'nt' or os.geteuid() == 0, reason='needs POSIX permission bits that bind'
)
def test_a_live_folder_under_one_that_cannot_be_read_is_named_and_the_report_saves(
    sim_session_on_its_own_installation, tmp_path, monkeypatch
):
    session = sim_session_on_its_own_installation
    share = tmp_path / 'share'
    live = share / 'capture'
    live.mkdir(parents=True)
    session.set_live_folder(str(live))
    share.chmod(0o000)
    try:
        report = _logs_zip(session, tmp_path, monkeypatch)
    finally:
        share.chmod(0o700)

    assert str(live) in report.read(f'video_receipts/{NOT_READ}').decode()


def test_the_data_folder_is_the_sessions_installations_without_its_icons(
    sim_session_on_its_own_installation, tmp_path, monkeypatch
):
    session = sim_session_on_its_own_installation
    data = session.scope.source_path / 'data'
    (data / 'this_installation.txt').write_text('mine\n')
    (data / 'icons').mkdir()
    (data / 'icons' / 'button.png').write_bytes(b'\x89PNG')

    names = set(_logs_zip(session, tmp_path, monkeypatch).namelist())

    assert {'data/this_installation.txt', 'data/labware.json'} <= names
    assert not [name for name in names if name.startswith('data/icons')]


def test_the_report_carries_the_settings_the_session_runs_on(tmp_path, monkeypatch):
    from _lumascope.modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    # A script's session: its settings handed in and saved nowhere, so no
    # settings file holds them; a plugin's own section carries a key.
    root = write_installation(tmp_path / 'installation')
    live = tmp_path / 'live'
    live.mkdir()
    settings = complete_settings(
        live_folder=str(live), plugins={'x': {'api_key': 's3cr3t', 'marker': 'live'}}
    )
    session = ScopeSession.create(settings, simulate=True, source_path=str(root))
    try:
        report = _logs_zip(session, tmp_path, monkeypatch)
    finally:
        session.shutdown()

    carried = json.loads(report.read('live_settings.json'))
    assert carried['plugins'] == {'x': {'api_key': '', 'marker': 'live'}}
    assert carried['live_folder'] == str(live)
    assert 'data/current.json' not in report.namelist()
