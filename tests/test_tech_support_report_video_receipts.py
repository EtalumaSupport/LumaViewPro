# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Support-bundle video receipts: what a future bundle can PROVE.

The customer's truncated-recordings defect arrived via a support bundle
that carried no video artifacts at all -- no manifest, no frame census,
no execution record -- so the truncation was invisible until the
customer described it. These tests pin the receipts step: every recent
recording folder yields its manifests, a frames inventory whose
count/min/max expose truncation and gaps, and the owning run's
execution record, with no pixel data shipped.
"""

import json
import time

from modules import recording_frames, tech_support_report
from modules.protocol_execution_record import ProtocolExecutionRecord
from modules.tech_support_report import (
    TechSupportReport,
    find_execution_record,
    find_video_recording_dirs,
    video_recording_inventory,
)
from modules.video_recording import MANIFEST_FILENAME


def _make_protocol_recording(
    run_root, well='A1', color='BF', scan=0, frame_numbers=range(5), manifest=True
):
    step_name = f'{well}_{color}_{scan:04}_video'
    rec = run_root / color / step_name
    rec.mkdir(parents=True)
    template = recording_frames.protocol_frame_filename_template(step_name)
    for n in frame_numbers:
        (rec / template.format(n=n)).write_bytes(b'x' * 10)
    if manifest:
        (rec / MANIFEST_FILENAME).write_text('{}')
    return rec


def _make_manual_recording(manual_root, ts='20260810_120000', frames=3):
    rec = manual_root / f'Video_{ts}'
    rec.mkdir(parents=True)
    template = recording_frames.manual_frame_filename_template()
    for n in range(frames):
        (rec / template.format(n=n, ts=ts)).write_bytes(b'x' * 10)
    (rec / MANIFEST_FILENAME).write_text('{}')
    return rec


def test_finds_protocol_manual_and_manifestless_recordings(tmp_path):
    run = tmp_path / 'run_A'
    with_manifest = _make_protocol_recording(run, color='BF')
    manifestless = _make_protocol_recording(run, color='Green', manifest=False)
    manual = _make_manual_recording(tmp_path / 'Manual')
    (tmp_path / 'unrelated').mkdir()

    found = find_video_recording_dirs(tmp_path, limit=10)

    assert with_manifest in found
    assert manifestless in found, 'a manifest-lost folder is exactly the interesting case'
    assert manual in found
    assert tmp_path / 'unrelated' not in found


def test_newest_first_and_capped(tmp_path):
    run = tmp_path / 'run_A'
    old = _make_protocol_recording(run, well='A1')
    new = _make_protocol_recording(run, well='B2')
    past = time.time() - 1000
    import os

    os.utime(old, (past, past))

    found = find_video_recording_dirs(tmp_path, limit=1)

    assert found == [new]


def test_inventory_exposes_truncation_and_gaps(tmp_path):
    run = tmp_path / 'run_A'
    rec = _make_protocol_recording(run, frame_numbers=[0, 1, 3, 4])
    (rec / 'clip.mp4').write_bytes(b'm' * 44)
    (rec / 'notes.txt').write_text('x')

    inventory = video_recording_inventory(rec)

    assert inventory['frame_count'] == 4
    assert (inventory['frame_number_min'], inventory['frame_number_max']) == (0, 4)
    assert inventory['frame_count'] < inventory['frame_number_max'] + 1, (
        'count below the number span is how a gap shows in the receipt'
    )
    assert inventory['frame_total_bytes'] == 40
    assert inventory['manifests'] == [MANIFEST_FILENAME]
    assert inventory['mp4s'] == [{'name': 'clip.mp4', 'bytes': 44}]
    assert inventory['other_file_count'] == 1


def test_execution_record_found_walking_up_and_absent_for_manual(tmp_path):
    run = tmp_path / 'run_A'
    rec = _make_protocol_recording(run)
    record_path = run / ProtocolExecutionRecord.DEFAULT_FILENAME
    record_path.write_text('rows')
    manual = _make_manual_recording(tmp_path / 'Manual')

    assert find_execution_record(rec) == record_path
    assert find_execution_record(manual) is None


def test_step_writes_receipt_bundles(tmp_path, monkeypatch):
    capture = tmp_path / 'capture'
    run = capture / 'run_A'
    rec = _make_protocol_recording(run)
    (run / ProtocolExecutionRecord.DEFAULT_FILENAME).write_text('rows')
    monkeypatch.setattr(tech_support_report, '_get_capture_dir', lambda: capture)
    report = TechSupportReport()
    out = tmp_path / 'bundle'
    out.mkdir()

    report._step_video_receipts(out)

    receipts = out / 'video_receipts'
    bundle = receipts / f'01_{rec.name}'
    inventory = json.loads((bundle / 'inventory.json').read_text())
    assert inventory['frame_count'] == 5
    assert (bundle / MANIFEST_FILENAME).is_file()
    assert (bundle / ProtocolExecutionRecord.DEFAULT_FILENAME).is_file()
    assert (receipts / '_index.txt').is_file()
    assert report._meta['video_receipts'] == [
        {
            'folder': rec.name,
            'frame_count': 5,
            'manifest_count': 1,
            'has_execution_record': True,
        }
    ]
    frame_copies = [p for p in bundle.iterdir() if recording_frames.is_video_frame(p.name)]
    assert frame_copies == [], 'receipts must not ship pixel data'


def test_step_records_none_found(tmp_path, monkeypatch):
    capture = tmp_path / 'capture'
    capture.mkdir()
    monkeypatch.setattr(tech_support_report, '_get_capture_dir', lambda: capture)
    report = TechSupportReport()
    out = tmp_path / 'bundle'
    out.mkdir()

    report._step_video_receipts(out)

    assert (out / 'video_receipts' / 'none_found.txt').is_file()


def test_logs_only_bundle_carries_video_receipts(tmp_path, monkeypatch):
    """A video complaint usually arrives via the quick 'zip logs' bundle,
    not the full report -- the receipts must ride BOTH."""
    import zipfile

    capture = tmp_path / 'capture'
    rec = _make_protocol_recording(capture / 'run_A')
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'lumaviewpro.log').write_text('log line\n')
    monkeypatch.setattr(tech_support_report, '_get_capture_dir', lambda: capture)
    monkeypatch.setattr(tech_support_report, '_get_lvp_logs_dir', lambda: logs)
    monkeypatch.setattr(tech_support_report, '_get_lvp_data_dir', lambda: None)
    report = TechSupportReport()

    zip_path = report.generate_logs_only(output_dir=tmp_path / 'out')

    assert zip_path is not None
    names = zipfile.ZipFile(zip_path).namelist()
    assert any(f'video_receipts/01_{rec.name}/inventory.json' in n for n in names)
    assert any('lumaviewpro.log' in n for n in names)
