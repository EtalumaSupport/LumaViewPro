# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The record of each layer's last auto-gain lock has one writer, the turning off.

``ScopeSession.channel_auto_gain_lock`` answers what the camera reached the last
time a layer's auto-gain was turned off, from ``_auto_gain_locks``. Only
``set_channel_auto_gain`` takes that lock, so a write anywhere else would make
the read answer a lock no turning off took. Every reference that writes the
record -- an item stored or deleted, the name rebound, a mutating method
called on it -- is in ``set_channel_auto_gain``, or is the session's
construction of the empty record.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, MemberNames, Words

_RECORD = '_auto_gain_locks'
_WRITERS = MemberNames('modules/scope_session.py', {'set_channel_auto_gain', '__init__'})
_MUTATORS = Words({'pop', 'popitem', 'clear', 'update', 'setdefault', '__setitem__', '__delitem__'})


def _record_writes(tree: ast.AST):
    """(line, enclosing function) of every write to the record."""
    found = []

    def is_record(node):
        return isinstance(node, ast.Attribute) and node.attr == _RECORD

    def walk(node, function):
        if function is None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        targets = []
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
        for target in targets:
            if is_record(target) or (isinstance(target, ast.Subscript) and is_record(target.value)):
                found.append((node.lineno, function))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATORS
            and is_record(node.func.value)
        ):
            found.append((node.lineno, function))
        for child in ast.iter_child_nodes(node):
            walk(child, function)

    walk(tree, None)
    return found


def test_only_the_turning_off_writes_the_lock_record():
    roots = [REPO_ROOT / name for name in ('modules', 'ui', 'rest')]
    files = [REPO_ROOT / 'lumaviewpro.py']
    files += [path for root in roots for path in sorted(root.rglob('*.py'))]
    writes = [
        (path.relative_to(REPO_ROOT).as_posix(), line, function)
        for path in files
        for line, function in _record_writes(ast.parse(path.read_text()))
    ]
    found = {function for _path, _line, function in writes}
    assert found >= _WRITERS, (
        'the walk did not find both the construction and the turning off: it is not reading them'
    )
    offenders = [
        f'{path}:{line} in {function}'
        for path, line, function in writes
        if path != 'modules/scope_session.py' or function not in _WRITERS
    ]
    assert offenders == []
