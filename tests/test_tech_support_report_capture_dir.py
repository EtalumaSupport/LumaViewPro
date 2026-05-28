# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: tech support report finds the configured capture dir.

Bug
---
_get_capture_dir read settings.get('capture_path', '') -- but capture_path
is not a real settings key (it is in no schema). The canonical capture
directory is live_folder. The read therefore always returned '' and the
report fell through to guessing among EtalumaCaptures / Etaluma / LumaViewPro
under Documents, so the operator's configured capture folder was unreachable.

Fix
---
Read live_folder via the canonical _resolve_settings_path resolver
(current.json-first): current.json carries the absolute path resolved at
runtime, while settings.json may still hold the unresolved './capture'
default.
"""

from __future__ import annotations

import json
import pathlib

from modules import tech_support_report


def _write(path: pathlib.Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def test_reads_live_folder_from_current_json(monkeypatch, tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    capture = tmp_path / 'MyCaptures'
    capture.mkdir()
    _write(data_dir / 'current.json', {'live_folder': str(capture)})

    monkeypatch.setattr(tech_support_report, '_get_lvp_data_dir', lambda: data_dir)
    assert tech_support_report._get_capture_dir() == capture.resolve()


def test_current_json_wins_over_settings_json(monkeypatch, tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    runtime = tmp_path / 'RuntimeResolved'
    runtime.mkdir()
    stale = tmp_path / 'StaleDefault'
    stale.mkdir()
    # settings.json holds the stale default; current.json the resolved path.
    _write(data_dir / 'settings.json', {'live_folder': str(stale)})
    _write(data_dir / 'current.json', {'live_folder': str(runtime)})

    monkeypatch.setattr(tech_support_report, '_get_lvp_data_dir', lambda: data_dir)
    assert tech_support_report._get_capture_dir() == runtime.resolve()


def test_falls_back_when_live_folder_missing(monkeypatch, tmp_path):
    # No live_folder configured -> the documents fallback, not a crash.
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    _write(data_dir / 'current.json', {})
    docs = tmp_path / 'Docs'
    docs.mkdir()

    monkeypatch.setattr(tech_support_report, '_get_lvp_data_dir', lambda: data_dir)
    monkeypatch.setattr(tech_support_report, '_get_user_documents', lambda: docs)
    # No EtalumaCaptures/Etaluma/LumaViewPro subdir exists, so docs itself.
    assert tech_support_report._get_capture_dir() == docs


def test_no_longer_reads_dead_capture_path_key():
    src = pathlib.Path(tech_support_report.__file__).read_text()
    assert "settings.get('capture_path'" not in src, (
        'capture_path is not a real settings key; read live_folder instead'
    )
