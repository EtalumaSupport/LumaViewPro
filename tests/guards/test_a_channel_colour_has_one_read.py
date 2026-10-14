# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A channel's colour has one read: ``ScopeSession.channel_colors``.

Every capture, recording, build and view is coloured from the mapping the
Session hands it. A module that read the release's colour itself would
colour its output past a per-channel override, so one capture or build
would disagree with the rest about what colour a channel is. This walk over
every production module, drivers, tools and the REST host included, refuses
a release-colour read outside the catalogue's own module and the Session's
read.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules, production_modules, walk_defs

_CATALOGUE_MODULE = 'modules/layer_record.py'
_OWNER_MODULE = 'modules/scope_session.py'
_OWNER_DEF = 'ScopeSession.channel_colors'
_READS = ('release_color', 'release_colors')


def _all_production_modules():
    yield from production_modules()
    yield from iter_package_modules(('drivers', 'tools', 'rest'))


def _read(node: ast.AST) -> str | None:
    """The colour read ``node`` makes: a release-colour call, or a record's ``display_color``."""
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
        return name if name in _READS else None
    if isinstance(node, ast.Attribute) and node.attr == 'display_color':
        return 'display_color'
    return None


def _reads(tree: ast.Module):
    """Yield (qualname or '<module>', lineno, read) for every colour read."""
    inside = set()
    for qualname, node in walk_defs(tree.body):
        for inner in ast.walk(node):
            read = _read(inner)
            if read is not None and id(inner) not in inside:
                inside.add(id(inner))
                yield qualname, inner.lineno, read
    for inner in ast.walk(tree):
        read = _read(inner)
        if read is not None and id(inner) not in inside:
            yield '<module>', inner.lineno, read


def test_no_module_but_the_session_reads_a_channels_colour():
    found = []
    for rel_path, tree in _all_production_modules():
        if rel_path == _CATALOGUE_MODULE:
            continue
        for qualname, lineno, read in _reads(tree):
            if rel_path == _OWNER_MODULE and qualname == _OWNER_DEF:
                continue
            found.append(f'{rel_path}:{lineno} {qualname} reads {read}')
    assert found == [], (
        "only ScopeSession.channel_colors reads a channel's colour; take the "
        f'mapping it hands you instead: {found}'
    )


def test_the_guard_sees_the_read_it_exempts():
    # The exemption must name the real owner, or a rename would leave the
    # guard exempting nothing while the other check still passed.
    trees = dict(_all_production_modules())
    owned = {
        read for qualname, _lineno, read in _reads(trees[_OWNER_MODULE]) if qualname == _OWNER_DEF
    }
    assert owned == {'release_colors'}
