# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests: tile overlap is a persisted system setting.

Tile overlap used to live only in the protocol panel's overlap spinner
and was read straight off the widget at scan time, so it reset to 0%
every launch. It is now promoted to settings['tiling_overlap_percent']:
the spinner is just the editor (writes the setting via
update_tiling_overlap), and scan/apply read the persisted value through
ProtocolSettings.get_tiling_overlap_percent().

These guard the two halves of that contract:
  - the key ships in the tracked settings.json schema default, so the
    settings_init default-merge backfills it into every user's
    current.json and the bare read never raises KeyError;
  - get_tiling_overlap_percent reads the setting, not the widget, and
    the scan-config builder goes through that single accessor.

current.json is gitignored runtime state, not the schema source, so it
is intentionally not asserted here.
"""

from __future__ import annotations

import ast
import json
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'
CONFIG_UI_GETTERS_SRC = REPO / 'modules' / 'config_ui_getters.py'


def _method_source(path: pathlib.Path, class_name: str, method_name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.get_source_segment(path.read_text(), child)
    raise AssertionError(f'{class_name}.{method_name} not found in {path}')


def _function_source(path: pathlib.Path, func_name: str) -> str:
    src = path.read_text()
    tree = ast.parse(src)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f'{func_name} not found in {path}')


def test_tiling_overlap_percent_in_settings_schema():
    """The tracked settings.json default carries the key.

    settings_init merges missing settings.json keys into current.json at
    load, so shipping the default here backfills every existing user and
    the bare read in get_tiling_overlap_percent never raises KeyError.
    """
    data = json.loads((REPO / 'data' / 'settings.json').read_text())
    assert 'tiling_overlap_percent' in data, (
        'data/settings.json must define tiling_overlap_percent so the '
        'default-merge backfills it and the persisted read does not KeyError'
    )


def test_get_tiling_overlap_percent_reads_setting_not_widget():
    """The accessor reads the persisted setting, not the spinner widget."""
    source = _method_source(PROTOCOL_SETTINGS_SRC, 'ProtocolSettings', 'get_tiling_overlap_percent')
    assert "settings['tiling_overlap_percent']" in source, (
        'get_tiling_overlap_percent must read the persisted setting'
    )
    assert 'tiling_overlap_spinner' not in source, (
        'get_tiling_overlap_percent must not read the spinner widget; the '
        'setting is the source of truth'
    )


def test_scan_config_uses_overlap_accessor():
    """The scan-config builder reads overlap through the single accessor."""
    source = _function_source(CONFIG_UI_GETTERS_SRC, 'get_sequenced_capture_config_from_ui')
    assert 'get_tiling_overlap_percent()' in source, (
        'get_sequenced_capture_config_from_ui must read overlap via '
        'ProtocolSettings.get_tiling_overlap_percent, not the spinner widget'
    )
    assert 'tiling_overlap_spinner' not in source, (
        'overlap must no longer be read off the spinner widget at scan time'
    )
