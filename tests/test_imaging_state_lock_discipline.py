# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""ImagingAPI state guarded by ``_state_lock`` must never be touched without it.

``_scale_bar`` was written by the GUI thread from ``set_scale_bar`` and read by
the capture and live-view threads inside ``get_image`` /
``get_image_from_buffer``, with none of those four sites holding the lock that
the three public accessors take. Worse than a plain race: each reader pulled
``enabled`` and ``color`` in two separate unlocked reads, so a toggle landing
between them drew the bar in the previous colour.

The check is structural rather than behavioural on purpose. A timing test for a
data race is flaky by nature and only exercises the sites it happens to call;
this asserts the invariant over every attribute the class guards, so the next
attribute to grow an unguarded access fails here instead of in the field.

``__init__`` is exempt: the object is not reachable from another thread yet.
"""

from __future__ import annotations

import ast

import pytest

from tests.ast_seams import parse_module

IMAGING = 'modules/lumascope_api/imaging.py'
LOCK = '_state_lock'
EXEMPT_METHODS = {'__init__'}


def _imaging_class() -> ast.ClassDef:
    for node in ast.walk(parse_module(IMAGING)):
        if isinstance(node, ast.ClassDef) and node.name == 'ImagingAPI':
            return node
    pytest.fail(f'ImagingAPI not found in {IMAGING}')


def _self_attribute(node: ast.AST) -> str | None:
    """Base ``self.<name>`` of an expression, through subscripts and calls."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == 'self'
        ):
            return node.attr
        node = node.value
    return None


def _holds_state_lock(with_node: ast.With) -> bool:
    for item in with_node.items:
        expr = item.context_expr
        if isinstance(expr, ast.Attribute) and expr.attr == LOCK:
            return True
    return False


def _guarded_attributes_and_lines(cls: ast.ClassDef) -> tuple[set[str], set[int]]:
    """Attributes touched under the lock, and every line inside a lock block."""
    attributes: set[str] = set()
    lines: set[int] = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.With) and _holds_state_lock(node):
            for sub in ast.walk(node):
                lines.add(getattr(sub, 'lineno', -1))
                if isinstance(sub, ast.Attribute):
                    name = _self_attribute(sub)
                    if name and name != LOCK:
                        attributes.add(name)
    return attributes, lines


def _method_of(cls: ast.ClassDef, lineno: int) -> str:
    best = None
    for node in ast.walk(cls):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            if node.lineno <= lineno <= end and (best is None or node.lineno > best.lineno):
                best = node
    return best.name if best else '<class body>'


def test_state_lock_attributes_are_never_touched_unguarded():
    cls = _imaging_class()
    guarded, guarded_lines = _guarded_attributes_and_lines(cls)

    assert guarded, (
        f'no attributes found under `with self.{LOCK}` in ImagingAPI -- the '
        f'lock or its usage was renamed and this guard is no longer checking '
        f'anything'
    )

    violations = []
    for node in ast.walk(cls):
        if not isinstance(node, ast.Attribute):
            continue
        name = _self_attribute(node)
        if name not in guarded or node.lineno in guarded_lines:
            continue
        method = _method_of(cls, node.lineno)
        if method in EXEMPT_METHODS:
            continue
        violations.append(f'{IMAGING}:{node.lineno} self.{name} in {method}()')

    assert not violations, (
        'ImagingAPI state guarded by '
        + LOCK
        + ' is accessed without it:\n  '
        + '\n  '.join(sorted(set(violations)))
        + '\n\nRead it through the locked accessor (e.g. scale_bar_config) or '
        'wrap the access, and take ONE snapshot rather than reading related '
        'fields one at a time.'
    )


def test_scale_bar_readers_use_a_single_snapshot():
    """The capture paths must not re-read the config for each field."""
    cls = _imaging_class()
    # The capture body is _get_image_impl (public get_image is a thin
    # ungated forwarder with no config reads of its own).
    for method_name in ('_get_image_impl', 'get_image_from_buffer'):
        method = next(
            (
                n
                for n in ast.walk(cls)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method_name
            ),
            None,
        )
        assert method is not None, f'{method_name} not found on ImagingAPI'

        snapshots = [
            n
            for n in ast.walk(method)
            if isinstance(n, ast.Attribute) and n.attr == 'scale_bar_config'
        ]
        assert len(snapshots) == 1, (
            f'{method_name}() reads scale_bar_config {len(snapshots)} times; '
            f'it must take exactly one snapshot so that enabled and color come '
            f'from the same configuration even if the GUI toggles mid-frame'
        )
