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
import json
import pathlib

from modules.common_utils import build_step_name, parse_step_name
from modules.tiling_config import TilingConfig, _row_label, _split_row_col


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'
TILING_JSON = REPO / 'data' / 'tiling.json'


# --- (a) the inference workhorse the restore relies on ---


def _tiling_config() -> TilingConfig:
    return TilingConfig(tiling_configs_file_loc=TILING_JSON)


def test_infers_2x2_from_tiled_step_names():
    tc = _tiling_config()
    names = [
        'A1_BF_TA1',
        'A1_BF_TA2',
        'A1_BF_TB1',
        'A1_BF_TB2',
        'A1_Green_TA1',
        'A1_Green_TA2',
        'A1_Green_TB1',
        'A1_Green_TB2',
    ]
    assert tc.determine_tiling_label_from_names(names) == '2x2'


def test_infers_3x3_from_tiled_step_names():
    tc = _tiling_config()
    names = [f'A1_BF_T{row}{col}' for row in ('A', 'B', 'C') for col in (1, 2, 3)]
    assert tc.determine_tiling_label_from_names(names) == '3x3'


def test_turret_token_does_not_raise_and_infers_tiling():
    # A turret token ('Turret<n>') shares its leading 'T' with the tile token.
    # When the parser misclassified it as a tile ('urret<n>'), this site did
    # int(label[1:]) -> int('rret<n>') and raised ValueError, taking down the
    # tiling inference for any protocol whose step names carry a turret token.
    tc = _tiling_config()
    names = [f'A1_BF_T{row}{col}_Turret3' for row in ('A', 'B') for col in (1, 2)]
    assert tc.determine_tiling_label_from_names(names) == '2x2'


def test_turret_only_names_have_no_tile():
    # Turret tokens with no tile must not be mistaken for tiles (which would
    # raise on the trailing-number parse); they simply contribute no tile.
    tc = _tiling_config()
    names = ['A1_BF_Turret1', 'A1_Green_Turret2']
    assert (tc.determine_tiling_label_from_names(names) or tc.no_tiling_label()) == '1x1'


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
    assert 'tiling_size_spinner' in src, (
        'load_protocol must set the tiling_size_spinner from the inferred label.'
    )


def test_apply_tiling_guards_against_recompounding():
    apply = _method('apply_tiling')
    assert _calls_determine_tiling(apply), (
        'apply_tiling must detect an already-tiled protocol (via the step '
        'names) and refuse, since it appends tile groups with no un-tile path.'
    )


# --- (c) row labels stay alphabetic past 26 rows so large mosaics round-trip ---


def test_row_label_is_bijective_base26():
    # A naive chr(i + ord('A')) leaves the alphabet past row 25 into '[', '\\',
    # ... and '_' (the step-name delimiter) at row 30; the label must instead
    # carry over to 'AA', 'AB', ... so it stays parseable.
    assert _row_label(0) == 'A'
    assert _row_label(25) == 'Z'
    assert _row_label(26) == 'AA'
    assert _row_label(27) == 'AB'
    assert _row_label(51) == 'AZ'
    assert _row_label(52) == 'BA'
    # Every label is uppercase letters only -- never punctuation or the '_'
    # token separator -- for a generous row count.
    for i in range(1000):
        assert _row_label(i).isalpha() and _row_label(i).isupper()


def test_split_row_col_round_trips_multiletter():
    for label in ('A1', 'Z9', 'AA1', 'AB12', 'BZ3'):
        letters, col = _split_row_col(label)
        assert f'{letters}{col}' == label


def _large_config(tmp_path, m, n):
    label = f'{m}x{n}'
    cfg = {
        'metadata': {'name': 'tiling', 'version': 1, 'default': '1x1'},
        'data': {'1x1': {'m': 1, 'n': 1}, label: {'m': m, 'n': n}},
    }
    loc = tmp_path / 'tiling.json'
    loc.write_text(json.dumps(cfg))
    return TilingConfig(tiling_configs_file_loc=loc), label


def test_large_mosaic_tile_labels_round_trip_through_step_names(tmp_path):
    # End-to-end on the reachable path: a 32-row config (past the chr() break,
    # including row 30 that would have rendered the '_' token separator) produces
    # tile labels that parse back as tiles and round-trip through the step name.
    tc, label = _large_config(tmp_path, m=32, n=32)
    tiles = tc.get_tile_centers(
        config_label=label,
        focal_length=50.0,
        frame_size={'width': 1936, 'height': 1216},
        fill_factor=1,
        binning_size=1,
    )
    assert len(tiles) == 32 * 32
    for tile_label in tiles:
        name = build_step_name(parse_step_name(f'A1_BF_T{tile_label}'))
        assert name == f'A1_BF_T{tile_label}'
        assert parse_step_name(name).tile == tile_label
    # The inference workhorse recovers the same MxN from those names.
    names = [f'A1_BF_T{t}' for t in tiles]
    assert tc.determine_tiling_label_from_names(names) == label
