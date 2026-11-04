# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every guard's table of names says what it holds, and a member name still names a member.

A guard that searches production for the names in a hardcoded table goes
quiet when one of them is renamed: the search finds no site, and no site
reads as no offender. ``MOTION_ONLY_METHODS`` and its siblings had a presence
check of their own; ``_READERS`` did not, and renaming ``get_actual_position``
left the position-reader guard green while it stopped looking at every new
read. ``_DOOR_NAMES`` listed a door no simulator has, ``_CATALOGUES`` a
loader since deleted, and two scans walked three directories that are gone.

So the contract is held over the class, not per table: every module-level
table of identifiers written out in a ``tests/guards`` module is built as
``MemberNames`` (each name defined in the modules it names), ``Packages``
(each a directory or module that exists), ``RetiredNames`` (held absent by
the guard that lists it) or ``Words`` (none of those), from
``tests/ast_seams``. A table computed from its owner -- the
reasons of a production table, the fields of a record -- reads the owner and
cannot go stale, so it is not one. A new table fails here until someone says
which it is.
"""

import ast
import importlib

from tests.ast_seams import (
    REPO_ROOT,
    MemberNames,
    Packages,
    RetiredNames,
    Words,
    defined_names,
)

_GUARDS = REPO_ROOT / 'tests' / 'guards'
_KINDS = {kind.__name__: kind for kind in (MemberNames, Packages, RetiredNames, Words)}
_COLLECTIONS = Words({'frozenset', 'set', 'tuple', 'list'})


def _identifier(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isidentifier()
    )


def _literal_names(node: ast.AST) -> bool:
    """A set, tuple or list written out of identifier strings, or of those and named constants."""
    return (
        isinstance(node, (ast.Set, ast.Tuple, ast.List))
        and any(_identifier(e) for e in node.elts)
        and all(_identifier(e) or isinstance(e, ast.Name) for e in node.elts)
    )


def _tables():
    """``{(guard module, table name): kind name or None}`` for every written-out table."""
    found = {}
    for path in sorted(_GUARDS.glob('*.py')):
        for node in ast.parse(path.read_text(encoding='utf-8')).body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if not isinstance(target, ast.Name) or not target.id.lstrip('_').isupper():
                continue
            value, kind = node.value, None
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.args:
                if value.func.id in _KINDS:
                    kind = value.func.id
                elif value.func.id not in _COLLECTIONS:
                    continue
                value = value.args[-1]
            if _literal_names(value):
                found[(path.stem, target.id)] = kind
    return found


def test_every_name_table_says_what_it_holds():
    tables = _tables()
    assert ('test_every_position_reader_is_classified', '_READERS') in tables
    unclassified = sorted(f'{module}.{name}' for (module, name), kind in tables.items() if not kind)
    assert not unclassified, (
        'Build each table of names as MemberNames(owners, names), Packages(names), '
        'RetiredNames(names) or Words(names), from tests.ast_seams:\n  ' + '\n  '.join(unclassified)
    )


def test_every_member_name_is_defined_by_its_owner():
    stale = []
    for (module, name), kind in _tables().items():
        if kind != MemberNames.__name__:
            continue
        table = getattr(importlib.import_module(f'tests.guards.{module}'), name)
        defined = set().union(*(defined_names(owner) for owner in table.owners))
        stale.extend(
            f'{module}.{name}: {member!r} not in {table.owners}'
            for member in sorted(table - defined)
        )
    assert not stale, (
        'A guard lists a member its owner no longer defines -- the guard has stopped '
        'looking for it. Rename it in the table, or drop it if the member is gone:\n  '
        + '\n  '.join(stale)
    )


def test_every_package_exists():
    gone = []
    for (module, name), kind in _tables().items():
        if kind != Packages.__name__:
            continue
        table = getattr(importlib.import_module(f'tests.guards.{module}'), name)
        root = REPO_ROOT / table.under
        gone.extend(
            f'{module}.{name}: {package!r} under {table.under or "the repo root"!r}'
            for package in sorted(table)
            if not ((root / package).is_dir() or (root / f'{package}.py').is_file())
        )
    assert not gone, (
        'A guard scans a directory or module that does not exist -- it scans nothing '
        'there. Rename it in the table, or drop it if it is gone:\n  ' + '\n  '.join(gone)
    )
