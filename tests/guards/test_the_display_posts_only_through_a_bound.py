# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the display posts to the UI thread from off it goes through a bound.

The display thread posted each frame's validity dot as a fresh Clock
callback, so a UI thread held up for four days held every one: 8.7 GB.
Its posts, and a run's saved frame, now go through ``schedule_ui_latest``
(one pending per key) or ``_schedule_blit`` (one pending frame). This walk
follows ``_render_one_frame`` and ``hold_protocol_saved_image`` through
every ``ScopeDisplay`` method they call and refuses any other post. A
lambda's body is not followed: it is what runs on the UI thread.
"""

from __future__ import annotations

import ast

from tests.ast_seams import parse_module

_MODULE = 'ui/scope_display.py'
_ROOTS = ('_render_one_frame', 'hold_protocol_saved_image')
# The two bounded ways to post; the walk does not enter them.
_BOUNDED = {'schedule_ui_latest', '_schedule_blit'}
_RAW_POSTS = {'schedule_once', 'schedule_interval', 'create_trigger', 'schedule_ui'}


def _methods() -> dict[str, ast.FunctionDef]:
    tree = parse_module(_MODULE)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'ScopeDisplay':
            return {d.name: d for d in node.body if isinstance(d, ast.FunctionDef)}
    raise AssertionError(f'ScopeDisplay is gone from {_MODULE}; re-pin the walk')


def _calls(node: ast.AST):
    """Every call in ``node``, lambda bodies left out."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Lambda):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _calls(child)


def _called_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def test_the_display_posts_only_through_a_bound():
    methods = _methods()
    missing = [root for root in _ROOTS if root not in methods]
    assert missing == [], f'{missing} gone from ScopeDisplay; re-pin the walk'
    seen: set[str] = set()
    todo = list(_ROOTS)
    raw = []
    while todo:
        name = todo.pop()
        if name in seen or name in _BOUNDED:
            continue
        seen.add(name)
        for call in _calls(methods[name]):
            called = _called_name(call)
            if called in _RAW_POSTS:
                raw.append(f'{_MODULE}:{call.lineno} ScopeDisplay.{name} posts by {called}')
            func = call.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == 'self'
                and func.attr in methods
            ):
                todo.append(func.attr)
    assert raw == [], (
        'the display posts to the UI thread without a bound; use schedule_ui_latest '
        '(one pending per key) or _schedule_blit:\n' + '\n'.join(raw)
    )
