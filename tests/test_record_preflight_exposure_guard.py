# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: record preflight must refuse an unknown camera exposure.

Bug
---
MainDisplay.record_init derives the recording frame rate as
1.0 / (exposure / 1000) from the cached camera exposure. The cache
seeds 0.0 at construction and keeps the prior value when a read
fails, so a camera whose exposure was never successfully read
reports 0.0 -- and the divide raised ZeroDivisionError inside the
record preflight, after the recording flag was already claimed.

Fix
---
record_init refuses loudly when exposure is None or <= 0: logs,
notifies the user, clears the recording claim, and returns before
the divide. No fabricated fallback rate -- the derived frame rate
sizes the recording memmap, so a stand-in value would misallocate
the buffer.

Test approach
-------------
MainDisplay is a Kivy widget (ids, _app_ctx, scope wiring), so the
guard is locked structurally via AST in the same style as the other
UI call-site gates: the exposure guard must exist, must clear the
recording claim and return, and must run BEFORE the divide.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent


def _record_init_node() -> ast.FunctionDef:
    tree = ast.parse((REPO / 'ui' / 'main_display.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'MainDisplay':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == 'record_init':
                    return child
    raise AssertionError('MainDisplay.record_init not found in ui/main_display.py')


def test_record_init_guards_zero_exposure_before_divide():
    method = _record_init_node()
    src = ast.unparse(method)

    guards = [
        s
        for s in ast.walk(method)
        if isinstance(s, ast.If)
        and 'exposure <= 0' in ast.unparse(s.test)
        and any(isinstance(inner, ast.Return) for inner in ast.walk(s))
        and 'recording.clear' in ast.unparse(s)
    ]
    assert guards, (
        'record_init must refuse (clear the recording claim and return) '
        'when the cached camera exposure is unknown (None or <= 0); '
        'the 0.0 cache seed otherwise reaches the frame-rate divide as '
        'a ZeroDivisionError'
    )

    guard_pos = src.index('exposure <= 0')
    divide_pos = src.index('1.0 / (exposure / 1000)')
    assert guard_pos < divide_pos, (
        'The exposure guard must run BEFORE the frame-rate divide that sizes the recording buffer'
    )
