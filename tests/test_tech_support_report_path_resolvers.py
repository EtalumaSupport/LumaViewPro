# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression guard: tech_support_report names no English-only Documents or Desktop.

Bug
---
tech_support_report built Path.home() / "Documents" and Path.home() /
"Desktop". On a localized Windows install where Documents is renamed
(e.g. German "Dokumente"), the hardcoded English name is not the folder,
and the report silently mis-located what it collected. Rule-35
semantic-duplicate audit 2026-05-19, finding 5.

Fix
---
The report reads no Documents folder at all: every folder it collects
from is its session's (the live folder, the installation). The Desktop a
host names for the ZIP is path_utils.desktop_folder, the one Desktop for
every caller; its tests are in test_a_report_names_its_folder.py.
"""

from __future__ import annotations

import pathlib

from _lumascope.modules import tech_support_report


def test_user_documents_no_hardcoded_documents_string():
    """Source-text guard: catch a re-introduction of Path.home() / 'Documents'."""
    src = pathlib.Path(tech_support_report.__file__).read_text(encoding='utf-8')
    needle = "Path.home() / 'Documents'"
    assert needle not in src, (
        f'tech_support_report.py contains {needle!r}; the report collects '
        "from its session's folders, never a guessed Documents folder"
    )


def test_user_desktop_no_hardcoded_desktop_string():
    """Source-text guard: catch a re-introduction of Path.home() / 'Desktop'."""
    src = pathlib.Path(tech_support_report.__file__).read_text(encoding='utf-8')
    needle = "Path.home() / 'Desktop'"
    assert needle not in src, (
        f'tech_support_report.py contains {needle!r}; resolution must go '
        'through platformdirs.user_desktop_dir()'
    )
