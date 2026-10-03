# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The macOS open-file picker filters by type identifier, asked of macOS.

`choose file of type` takes type identifiers. The picker handed it bare
extensions, which happened to work for tif, tsv and csv but greyed out every
.json file, so on a Mac no cell-count method file could be loaded (Eric's
census 2026-10-02: "json" returned nothing; "public.json", "tsv", "csv" and
their identifiers each returned the file). The picker now asks macOS for each
extension's identifier in a lookup of its own, and the dialog script stays
plain AppleScript: run in the same AppleScriptObjC script, the dialog let
nothing be selected (Eric, the same day).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import ui.file_dialogs as file_dialogs

_IDENTIFIERS = {
    'json': 'public.json',
    'tsv': 'public.tab-separated-values-text',
    'csv': 'public.comma-separated-values-text',
    'tif': 'public.tiff',
    'tiff': 'public.tiff',
}


def _dialog_script(monkeypatch, filetypes, initial_dir=None) -> str:
    seen = {}

    def run(args, **kwargs):
        seen['script'] = args[-1]
        return SimpleNamespace(returncode=0, stdout='/picked/file\n')

    monkeypatch.setattr(
        file_dialogs, '_macos_type_identifiers', lambda exts: [_IDENTIFIERS[e] for e in exts]
    )
    monkeypatch.setattr(file_dialogs.subprocess, 'run', run)
    assert (
        file_dialogs._macos_open_file(initial_dir=initial_dir, filetypes=filetypes)
        == '/picked/file'
    )
    return seen['script']


def test_the_filter_is_the_identifiers_macos_names(monkeypatch):
    script = _dialog_script(monkeypatch, [('JSON', '.json')], initial_dir='/data')
    assert script == (
        'set theFile to choose file of type {"public.json"} '
        'default location POSIX file "/data"\nPOSIX path of theFile'
    )


def test_every_extension_of_a_filter_is_asked_for(monkeypatch):
    script = _dialog_script(monkeypatch, [('TIFF', '.tif .tiff'), ('TSV', '.tsv')])
    assert 'of type {"public.tiff", "public.tiff", "public.tab-separated-values-text"}' in script


def test_the_dialog_script_is_plain_applescript(monkeypatch):
    script = _dialog_script(monkeypatch, [('CSV', '.csv')])
    assert 'use framework' not in script


def test_no_filter_asks_for_no_types(monkeypatch):
    script = _dialog_script(monkeypatch, None)
    assert script == 'set theFile to choose file\nPOSIX path of theFile'


@pytest.mark.skipif(sys.platform != 'darwin', reason='asks macOS for the identifiers')
def test_macos_names_the_identifiers_the_filters_use():
    extensions = ['json', 'tsv', 'csv', 'tif', 'tiff']
    assert file_dialogs._macos_type_identifiers(extensions) == [_IDENTIFIERS[e] for e in extensions]
