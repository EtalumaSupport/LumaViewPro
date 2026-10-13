# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Whether the live picture is frozen has one store, and the button is drawn from it.

The display thread's pause is the store. A second flag beside it (the
retired ``ScopeDisplay.play``) drifted from it whenever the thread was
restarted, and a Freeze button written by Kivy alone showed paused on a
press no handler saw (a touch ending off the button fires no
``on_release``). The button acts on press and ``draw_live_button`` is its
one Python writer.
"""

from __future__ import annotations

import ast
import re

from tests.ast_seams import REPO_ROOT, production_modules

_DRAWER = ('ui/main_display.py', 'draw_live_button')


def _is_the_freeze_button(node) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == 'live_btn'
    )


def _freeze_button_writers():
    """``(file, enclosing def)`` of every write to the Freeze button's state."""
    found = []
    for rel, tree in production_modules():
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == 'state'
                        and _is_the_freeze_button(target.value)
                    ):
                        found.append((rel, fn.name, node.lineno))
    return found


def test_only_draw_live_button_writes_the_freeze_button():
    writers = _freeze_button_writers()
    others = [w for w in writers if w[:2] != _DRAWER]
    assert writers, 'draw_live_button no longer writes the Freeze button; the guard is stale'
    assert not others, (
        'the Freeze button is drawn from the display thread by draw_live_button alone; '
        f'other writers: {others}'
    )


def test_no_production_code_reads_or_writes_a_play_flag():
    sites = [
        f'{rel}:{node.lineno}'
        for rel, tree in production_modules()
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == 'play'
    ]
    assert not sites, (
        "whether the picture is frozen is the display thread's is_paused; "
        f'a play flag beside it is a second store: {sites}'
    )


def _kv_block(control_id: str) -> str:
    kv = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text()
    start = kv.index(f'id: {control_id}')
    following = re.search(r'\n\s+[A-Z][A-Za-z_]*:\s*\n', kv[start:])
    return kv[start : start + following.start()]


def test_the_freeze_button_acts_on_press_and_stays_live_during_a_run():
    block = _kv_block('live_btn')
    assert 'on_press: root.cam_toggle()' in block
    assert 'on_release' not in block
    assert 'disabled:' not in block, (
        "a freeze is the display's alone, so the button is live during a run"
    )
