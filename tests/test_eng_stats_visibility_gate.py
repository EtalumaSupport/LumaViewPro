# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Engineering preview stats compute only while the panel is on-screen.

The per-0.5s mean / std / focus-score compute in the scope-display loop is
expensive (a full pass over the frame plus a Vollath focus convolution). It
must not run when every layer accordion is collapsed -- in that state the
stat labels are off-screen and the result is discarded. open_layer is None
exactly when no accordion is expanded, so the gate keys on it.

ScopeDisplay is a Kivy widget that cannot be imported headless (its class
body evaluates Kivy properties), so this locks the decision seam by parsing
the source: the _eng_stats_due predicate carries the open_layer-None gate,
and the render loop routes the compute through it -- not through a bare
time-interval check that runs while the panel is closed.
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


def test_eng_stats_due_predicate_exists():
    assert _find_func('_eng_stats_due') is not None, (
        '_eng_stats_due seam removed; visibility gate no longer isolated/testable'
    )


def test_predicate_gates_on_open_layer_none():
    fn = _find_func('_eng_stats_due')
    body = ast.get_source_segment(SRC, fn)
    # The off-screen case (no expanded accordion) must short-circuit to False.
    assert 'open_layer is None' in body
    assert 'return False' in body


def _calls_named(node, names):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in names
    ]


def test_render_loop_routes_compute_through_predicate():
    # The expensive compute (the stats and the focus score) must be reached
    # only inside the one `if self._eng_stats_due(...)` block of the render
    # loop, so a closed panel skips it. Asserted on the AST, not on a string:
    # every call to the stats seam, the focus score or a raw numpy reduction
    # in _render_one_frame lies inside that If's subtree.
    render = _find_func('_render_one_frame')
    assert render is not None
    gates = [
        n
        for n in ast.walk(render)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and isinstance(n.test.func, ast.Attribute)
        and n.test.func.attr == '_eng_stats_due'
    ]
    assert len(gates) == 1, 'exactly one _eng_stats_due gate in the render loop'
    inside = {id(n) for n in ast.walk(gates[0])}
    compute = _calls_named(render, {'_engineering_stats', 'focus_function', 'mean', 'std'})
    assert compute, 'the render loop no longer computes the engineering stats at all'
    escaped = [c for c in compute if id(c) not in inside]
    assert not escaped, f'{len(escaped)} compute call(s) outside the _eng_stats_due gate'
