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

from _lumascope.modules import tech_support_report
from _lumascope.modules.protocol import Protocol


def test_finds_running_unsaved_tsv_protocol(tmp_path):
    (tmp_path / 'unsaved_protocol.tsv').write_text(
        f'{Protocol.PROTOCOL_FILE_HEADER}\nVersion\t5\n', encoding='utf-8'
    )
    found = tech_support_report.get_recent_protocols([tmp_path])
    assert 'unsaved_protocol' in {p['name'] for p in found}


def test_finds_legacy_json_protocol(tmp_path):
    (tmp_path / 'legacy.json').write_text('{"steps": [], "period": 0}', encoding='utf-8')
    found = tech_support_report.get_recent_protocols([tmp_path])
    assert 'legacy' in {p['name'] for p in found}


def test_ignores_non_protocol_files(tmp_path):
    (tmp_path / 'notes.tsv').write_text('col_a\tcol_b\n1\t2\n', encoding='utf-8')
    (tmp_path / 'config.json').write_text('{"foo": "bar"}', encoding='utf-8')
    found = tech_support_report.get_recent_protocols([tmp_path])
    assert all(p['name'] not in ('notes', 'config') for p in found)
