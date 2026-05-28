# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""[PERF] gating reads settings.debug_mode live, not cached on the first frame.

The perf instrumentation used to cache debug_mode on the first frame
(self._debug_perf, lazy-resolved once). If the first frame arrived before
settings finished loading it froze at False and [PERF] never logged even
after debug_mode went on -- the same divergent-cache shape that caused a
multi-hour debug session. The gate now reads ctx.settings live each frame
via _debug_perf_enabled.

ScopeDisplay is a Kivy widget that cannot be imported headless, so this
locks the invariant by parsing the source: the predicate exists and reads
the setting, and no cached self._debug_perf assignment survives.
"""

from __future__ import annotations

import ast
import pathlib

SRC_PATH = pathlib.Path(__file__).resolve().parent.parent / 'ui' / 'scope_display.py'
SRC = SRC_PATH.read_text()
TREE = ast.parse(SRC)


def _find_func(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_predicate_exists_and_reads_setting_live():
    fn = _find_func('_debug_perf_enabled')
    assert fn is not None
    body = ast.get_source_segment(SRC, fn)
    assert "ctx.settings.get('debug_mode', False)" in body


def test_no_cached_debug_perf_assignment():
    # The first-frame cache is the bug being removed. Any assignment to the
    # cached attribute (other than via the live predicate) reintroduces it.
    for node in ast.walk(TREE):
        if isinstance(node, ast.Attribute) and node.attr == '_debug_perf':
            raise AssertionError(
                'self._debug_perf cache reintroduced; gate must read '
                'debug_mode live via _debug_perf_enabled'
            )


def test_no_first_frame_none_check():
    assert 'self._debug_perf is None' not in SRC
