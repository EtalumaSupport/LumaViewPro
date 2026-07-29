# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests: the support bundle keeps every recent protocol record.

Bug shape: _step_protocols copied recent protocols into the bundle by
bare basename. Same-named files are the NORMAL case (every protocol run
folder emits protocol_record.tsv / protocol_post_record.tsv), and the
copy loop runs newest-first with overwrite semantics -- so the OLDEST
same-named file won while _index.txt and metadata claimed the newest was
captured. A 2026-07-29 support bundle lost the exact run records its
issue was filed about.

Contract under test: every listed protocol lands in the bundle under a
name derived once and shared by the copied file, the _index.txt entry,
and the metadata mirror -- so the three can never disagree.
"""

from __future__ import annotations

import os
import pathlib
import re
import time

from modules import tech_support_report
from modules.protocol import Protocol
from modules.tech_support_report import TechSupportReport


def _write_protocol(path: pathlib.Path, marker: str, mtime: float) -> None:
    path.write_text(f'{Protocol.PROTOCOL_FILE_HEADER}\n{marker}\n')
    os.utime(path, (mtime, mtime))


def _make_two_same_named_runs(tmp_path: pathlib.Path):
    """Two run folders, each holding a protocol_record.tsv; newer has marker B."""
    run_old = tmp_path / 'capture' / 'run_old'
    run_new = tmp_path / 'capture' / 'run_new'
    run_old.mkdir(parents=True)
    run_new.mkdir(parents=True)
    now = time.time()
    _write_protocol(run_old / 'protocol_record.tsv', 'marker-OLD', now - 3600)
    _write_protocol(run_new / 'protocol_record.tsv', 'marker-NEW', now)
    return tmp_path / 'capture'


def _run_step_protocols(tmp_path, monkeypatch):
    capture_dir = _make_two_same_named_runs(tmp_path)
    monkeypatch.setattr(tech_support_report, '_get_capture_dir', lambda: capture_dir)
    monkeypatch.setattr(tech_support_report, '_get_protocol_dir', lambda: None)
    monkeypatch.setattr(tech_support_report, '_get_lvp_data_dir', lambda: None)

    report = TechSupportReport()
    bundle_tmp = tmp_path / 'bundle'
    bundle_tmp.mkdir()
    report._step_protocols(bundle_tmp)
    return report, bundle_tmp / 'recent_protocols'


def test_same_named_protocols_all_survive_in_the_bundle(tmp_path, monkeypatch):
    _, dest = _run_step_protocols(tmp_path, monkeypatch)

    copied = sorted(p.name for p in dest.iterdir() if p.suffix == '.tsv')
    assert len(copied) == 2, (
        f'both same-named protocol records must survive; bundle holds only: {copied}'
    )
    contents = {p.read_text() for p in dest.iterdir() if p.suffix == '.tsv'}
    assert any('marker-NEW' in c for c in contents), 'the newer record was lost'
    assert any('marker-OLD' in c for c in contents), 'the older record was lost'


def test_index_and_metadata_name_the_files_actually_on_disk(tmp_path, monkeypatch):
    report, dest = _run_step_protocols(tmp_path, monkeypatch)

    on_disk = {p.name for p in dest.iterdir() if p.suffix == '.tsv'}
    index_text = (dest / '_index.txt').read_text()
    entry_names = set(re.findall(r'^\s*\d+\.\s+(\S+)\s*$', index_text, re.MULTILINE))
    assert entry_names == on_disk, (
        f'each numbered _index.txt entry must name the exact bundled filename; '
        f'entries: {sorted(entry_names)} vs on disk: {sorted(on_disk)}'
    )
    meta_names = [entry['name'] for entry in report._meta['recent_protocols']]
    assert len(set(meta_names)) == len(meta_names), (
        f'metadata mirror must carry collision-free names; got: {meta_names}'
    )
    for filename in on_disk:
        assert pathlib.Path(filename).stem in meta_names, (
            f'metadata mirror must carry the bundled name {filename!r}; meta: {meta_names}'
        )
