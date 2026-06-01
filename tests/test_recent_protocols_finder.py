# Copyright Etaluma, Inc.
"""Regression test: the tech-support protocol finder must include TSV.

LVP protocols are TSV-native -- Protocol.to_file() writes a TSV whose first
row is the PROTOCOL_FILE_HEADER banner, and a protocol run in progress
auto-writes its protocol (as unsaved_protocol.tsv) into the run directory.
The finder previously globbed *.json only, so a tech-support bundle reported
"No protocol files found" even when a protocol was running and on disk.

These tests assert the finder surfaces TSV protocols by their header banner,
still accepts legacy JSON, and ignores non-protocol files of either type.
"""

import pytest

from modules import tech_support_report
from modules.protocol import Protocol


@pytest.fixture
def protocol_dir(tmp_path, monkeypatch):
    """Point all three finder search roots at one temp directory."""
    for name in ('_get_protocol_dir', '_get_lvp_data_dir', '_get_capture_dir'):
        monkeypatch.setattr(tech_support_report, name, lambda: tmp_path)
    return tmp_path


def test_finds_running_unsaved_tsv_protocol(protocol_dir):
    (protocol_dir / 'unsaved_protocol.tsv').write_text(
        f'{Protocol.PROTOCOL_FILE_HEADER}\nVersion\t5\n', encoding='utf-8'
    )
    found = tech_support_report.get_recent_protocols()
    assert 'unsaved_protocol' in {p['name'] for p in found}


def test_finds_legacy_json_protocol(protocol_dir):
    (protocol_dir / 'legacy.json').write_text(
        '{"steps": [], "period": 0}', encoding='utf-8'
    )
    found = tech_support_report.get_recent_protocols()
    assert 'legacy' in {p['name'] for p in found}


def test_ignores_non_protocol_files(protocol_dir):
    (protocol_dir / 'notes.tsv').write_text('col_a\tcol_b\n1\t2\n', encoding='utf-8')
    (protocol_dir / 'config.json').write_text('{"foo": "bar"}', encoding='utf-8')
    found = tech_support_report.get_recent_protocols()
    assert all(p['name'] not in ('notes', 'config') for p in found)
