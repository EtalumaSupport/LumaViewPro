# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: loading a protocol restores the tiling selection, and
re-applying tiling to an already-tiled protocol is blocked.

Bug history
-----------
Bench report 2026-06-03: loading a protocol saved with 2x2 tiling came
back showing 1x1 in the tiling spinner. Tiling is baked into the steps
as expanded tile positions (one row per tile, named ..._T<gridlabel>)
rather than stored as a scalar, so the spinner stayed at its 1x1 default
and misrepresented the protocol. The scan itself was correct (the tiled
steps round-trip through save/load), but the display lied -- and because
apply_tiling APPENDS tile groups with no un-tile path, trusting the wrong
"1x1" and re-applying would compound the tiles (2x2 on a 2x2 -> 16).

Fix
---
- ProtocolSettings.load_protocol infers the tiling label back from the
  step names (TilingConfig.determine_tiling_label_from_names) and sets
  the spinner.
- ProtocolSettings.apply_tiling refuses when the protocol is already
  tiled, directing the user to reload the untiled base first.

These tests pin (a) the inference workhorse the restore relies on, and
(b) that the wiring is present in both UI methods (source-inspection,
since the Kivy/ctx-bound widget is not live-instantiable here -- the
same convention as the other test_protocol_settings_*.py files).
"""

from __future__ import annotations

import ast
import pathlib

from modules.tiling_config import TilingConfig


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'
TILING_JSON = REPO / 'data' / 'tiling.json'


# --- (a) the inference workhorse the restore relies on ---

def _tiling_config() -> TilingConfig:
    return TilingConfig(tiling_configs_file_loc=TILING_JSON)


def test_infers_2x2_from_tiled_step_names():
    tc = _tiling_config()
    names = [
        'A1_BF_TA1', 'A1_BF_TA2', 'A1_BF_TB1', 'A1_BF_TB2',
        'A1_Green_TA1', 'A1_Green_TA2', 'A1_Green_TB1', 'A1_Green_TB2',
    ]
    assert tc.determine_tiling_label_from_names(names) == '2x2'


def test_infers_3x3_from_tiled_step_names():
    tc = _tiling_config()
    names = [
        f'A1_BF_T{row}{col}'
        for row in ('A', 'B', 'C')
        for col in (1, 2, 3)
    ]
    assert tc.determine_tiling_label_from_names(names) == '3x3'


def test_untiled_protocol_falls_back_to_no_tiling():
    tc = _tiling_config()
    # Untiled step names have no _T<gridlabel> segment.
    names = ['A1_BF', 'A2_BF', 'A3_Green']
    inferred = tc.determine_tiling_label_from_names(names)
    # The restore site uses `inferred or no_tiling_label()`, so either a
    # falsy result or the explicit no-tiling label is acceptable -- both
    # land the spinner on 1x1.
    assert (inferred or tc.no_tiling_label()) == '1x1'


# --- (b) the wiring is present in both UI methods ---

def _method(name: str) -> ast.FunctionDef:
    tree = ast.parse(PROTOCOL_SETTINGS_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'ProtocolSettings':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return child
    raise AssertionError(f'ProtocolSettings.{name} not found')


def _calls_determine_tiling(method: ast.FunctionDef) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == 'determine_tiling_label_from_names'
        for n in ast.walk(method)
    )


def test_load_protocol_restores_tiling_spinner():
    load = _method('load_protocol')
    assert _calls_determine_tiling(load), (
        'load_protocol must infer the tiling label from the loaded steps '
        'so the spinner reflects an already-tiled protocol.'
    )
    src = ast.get_source_segment(PROTOCOL_SETTINGS_SRC.read_text(), load)
    assert "tiling_size_spinner" in src, (
        'load_protocol must set the tiling_size_spinner from the inferred label.'
    )


def test_apply_tiling_guards_against_recompounding():
    apply = _method('apply_tiling')
    assert _calls_determine_tiling(apply), (
        'apply_tiling must detect an already-tiled protocol (via the step '
        'names) and refuse, since it appends tile groups with no un-tile path.'
    )
