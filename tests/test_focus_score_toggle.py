# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Focus-score toggle gates the costly Vollath compute on the preview path.

The engineering-mode preview computes a Vollath focus score once per 0.5s.
That convolution dominates the per-frame engineering-stats cost, while the
caller (with the plugin installed) cannot easily leave engineering mode. The
"Focus Score" engineering-tab toggle suppresses it; mean and std stay. The
flag is read LIVE each frame -- caching it on the first frame would freeze
the value the same way an earlier debug gate did.

ScopeDisplay is a Kivy widget that cannot be imported headless, so the seam
is locked by parsing the source. The settings default is checked directly.
"""

from __future__ import annotations

import ast
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC_PATH = REPO / 'ui' / 'scope_display.py'
SRC = SRC_PATH.read_text()
TREE = ast.parse(SRC)


def _find_func(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_focus_score_enabled_predicate_exists():
    assert _find_func('_focus_score_enabled') is not None


def test_predicate_indexes_the_live_setting():
    fn = _find_func('_focus_score_enabled')
    body = ast.get_source_segment(SRC, fn)
    # Indexes the live settings dict, which carries every shipped key.
    assert "ctx.settings['focus_score_enabled']" in body


def test_focus_function_gated_by_the_toggle():
    # The expensive call must be guarded by the predicate; when off the label
    # shows 'off' instead of recomputing.
    assert 'if self._focus_score_enabled(ctx):' in SRC
    assert "af_score = 'off'" in SRC


def test_setting_seeded_in_defaults():
    # pin-justified: the shipped default in data/settings.json is the
    # contract a fresh install receives.
    settings = json.loads((REPO / 'data' / 'settings.json').read_text())
    # Seeded so _deep_merge_defaults supplies it to existing current.json;
    # default OFF.
    assert settings.get('focus_score_enabled') is False
