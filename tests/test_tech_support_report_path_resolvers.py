# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: tech_support_report uses platformdirs for Documents/Desktop.

Bug
---
tech_support_report._get_user_documents returned Path.home() / "Documents"
and _get_desktop returned Path.home() / "Desktop". Both diverged from
the platformdirs-based resolvers in modules/path_utils.get_source_root
and modules/app_environment.init_environment. On a localized Windows
install where Documents is renamed (e.g. German "Dokumente"),
platformdirs returns the localized path while the hardcoded English
name does not -- the tech support report would silently mis-locate the
data root and bundle the wrong files. Rule-35 semantic-duplicate audit
2026-05-19, finding 5.

Fix
---
Both helpers now wrap platformdirs.user_documents_dir() /
platformdirs.user_desktop_dir().

Test approach
-------------
Behavioral exec: monkeypatch platformdirs and confirm the helpers
delegate. Also confirms the helpers return pathlib.Path (platformdirs
returns str), preserving the call-site contract.
"""

from __future__ import annotations

import pathlib

import pytest

from modules import tech_support_report


def test_get_user_documents_uses_platformdirs(monkeypatch):
    """_get_user_documents honors platformdirs (localized-Windows safe)."""
    sentinel = '/tmp/.test_docs_locale'
    monkeypatch.setattr(tech_support_report.platformdirs, 'user_documents_dir', lambda: sentinel)
    result = tech_support_report._get_user_documents()
    assert isinstance(result, pathlib.Path)
    assert str(result) == sentinel


def test_get_desktop_uses_platformdirs_when_exists(monkeypatch, tmp_path):
    """_get_desktop returns the platformdirs path when it exists."""
    monkeypatch.setattr(tech_support_report.platformdirs, 'user_desktop_dir', lambda: str(tmp_path))
    result = tech_support_report._get_desktop()
    assert isinstance(result, pathlib.Path)
    assert result == tmp_path


def test_get_desktop_falls_back_to_home_when_platformdirs_missing(monkeypatch):
    """_get_desktop falls back to home() when platformdirs path is absent."""
    monkeypatch.setattr(
        tech_support_report.platformdirs,
        'user_desktop_dir',
        lambda: '/tmp/.this_directory_does_not_exist_for_test',
    )
    result = tech_support_report._get_desktop()
    assert isinstance(result, pathlib.Path)
    assert result == pathlib.Path.home()


def test_user_documents_no_hardcoded_documents_string():
    """Source-text guard: catch a re-introduction of Path.home() / 'Documents'."""
    src = pathlib.Path(tech_support_report.__file__).read_text()
    needle = "Path.home() / 'Documents'"
    assert needle not in src, (
        f'tech_support_report.py contains {needle!r}; resolution must go '
        'through platformdirs.user_documents_dir() to honor localized '
        'Windows installs'
    )


def test_user_desktop_no_hardcoded_desktop_string():
    """Source-text guard: catch a re-introduction of Path.home() / 'Desktop'."""
    src = pathlib.Path(tech_support_report.__file__).read_text()
    needle = "Path.home() / 'Desktop'"
    assert needle not in src, (
        f'tech_support_report.py contains {needle!r}; resolution must go '
        'through platformdirs.user_desktop_dir()'
    )
