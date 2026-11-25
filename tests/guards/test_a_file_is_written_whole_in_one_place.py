# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A file is written whole in one place: ``path_utils.written_whole``.

Two routines and six hand-rolled temp-then-replace copies had drifted into
eight behaviours: a temporary ending in its target's kind that a later
count took for an image, a copy whose failure left its partial file, a
hidden name the support report would not have blanked. So nothing in
``src/``, ``ui/`` or ``lumaviewpro.py`` renames a file into place but the
routine itself and the three functions that move an existing file aside.

The walk refuses ``os.replace``, ``os.rename``, ``shutil.move``,
``tempfile.mkstemp`` and ``tempfile.NamedTemporaryFile``, and the ``Path``
spelling: a ``.replace`` or ``.rename`` call with exactly one positional
argument and no keyword (``str.replace`` takes two; ``dataclasses.replace``,
``datetime.replace`` and ``DataFrame.rename`` take keywords). It does not
see a direct write onto an existing name (``open(target, 'w')``), since
whether the target exists is not in the source.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules, parse_module

_REFUSED = {
    ('os', 'replace'),
    ('os', 'rename'),
    ('shutil', 'move'),
    ('tempfile', 'mkstemp'),
    ('tempfile', 'NamedTemporaryFile'),
}

# By enclosing function, never by module: a second copy beside one of these
# is still refused.
_EXEMPT = {
    ('src/_lumascope/modules/path_utils.py', 'written_whole'),
    ('src/_lumascope/modules/path_utils.py', 'move_aside'),
    ('src/_lumascope/modules/settings_init.py', 'retire_rejected_current_json'),
    ('src/_lumascope/modules/app_environment.py', '_fold_flat_captures'),
}


def _is_a_rename(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if isinstance(func.value, ast.Name) and (func.value.id, func.attr) in _REFUSED:
        return True
    return func.attr in ('replace', 'rename') and len(call.args) == 1 and not call.keywords


def _renames(tree: ast.Module):
    """Yield ``(enclosing qualname, line)`` for every rename call in *tree*."""

    def visit(node: ast.AST, qualname: str):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                yield from visit(child, f'{qualname}.{child.name}' if qualname else child.name)
                continue
            if isinstance(child, ast.Call) and _is_a_rename(child):
                yield qualname, child.lineno
            yield from visit(child, qualname)

    yield from visit(tree, '')


def _production():
    yield from iter_package_modules(('src', 'ui'))
    yield 'lumaviewpro.py', parse_module('lumaviewpro.py')


def test_nothing_renames_a_file_into_place_but_the_routine_and_the_moves_aside():
    offenders = []
    exempt_used = set()
    for rel_path, tree in _production():
        for qualname, line in _renames(tree):
            function = qualname.rsplit('.', 1)[-1]
            if (rel_path, function) in _EXEMPT:
                exempt_used.add((rel_path, function))
            else:
                offenders.append(f'{rel_path}:{line} ({qualname or "module level"})')
    assert offenders == [], (
        'a file is written whole through path_utils.written_whole, and an '
        'existing file is moved aside through path_utils.move_aside; these '
        f'rename a file themselves: {offenders}'
    )
    assert exempt_used == _EXEMPT, f'exempted, but renames nothing: {sorted(_EXEMPT - exempt_used)}'
