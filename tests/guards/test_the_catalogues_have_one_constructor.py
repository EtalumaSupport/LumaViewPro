# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Only the scope builds the labware and objective catalogues.

Every module that built its own catalogue read the installation's folder,
not the folder the session was started on, so one session held several
catalogues that disagreed about which objectives and plates exist. The scope
now reads both once, in ``Lumascope._read_catalogues``, and every consumer
reads its copy. A construction anywhere else is a second store; this walk
over every production module, drivers and tools included, refuses it at
commit rather than in review.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules, production_modules, walk_defs

_OWNER_MODULE = 'modules/lumascope_api/_lumascope.py'
_OWNER_DEF = 'Lumascope._read_catalogues'
_CATALOGUES = ('WellPlateLoader', 'ObjectiveLoader', 'LabwareLoader')


def _all_production_modules():
    yield from production_modules()
    yield from iter_package_modules(('drivers', 'tools'))


def _constructed(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
    return name if name in _CATALOGUES else None


def _constructions(tree: ast.Module):
    """Yield (qualname or '<module>', lineno, class) for every catalogue construction."""
    inside = set()
    for qualname, node in walk_defs(tree.body):
        for inner in ast.walk(node):
            name = _constructed(inner)
            if name is not None and id(inner) not in inside:
                inside.add(id(inner))
                yield qualname, inner.lineno, name
    for inner in ast.walk(tree):
        name = _constructed(inner)
        if name is not None and id(inner) not in inside:
            yield '<module>', inner.lineno, name


def test_no_module_but_the_scope_builds_a_catalogue():
    found = []
    for rel_path, tree in _all_production_modules():
        for qualname, lineno, name in _constructions(tree):
            if rel_path == _OWNER_MODULE and qualname == _OWNER_DEF:
                continue
            found.append(f'{rel_path}:{lineno} {qualname} builds {name}')
    assert found == [], (
        'only Lumascope._read_catalogues may build a labware or objective '
        f"catalogue; read the scope's copy instead: {found}"
    )


def test_the_guard_sees_the_owner_it_exempts():
    # The exemption must name the real owner, or a rename would leave the
    # guard exempting nothing while every other check still passed.
    trees = dict(_all_production_modules())
    owned = {
        name
        for qualname, _lineno, name in _constructions(trees[_OWNER_MODULE])
        if qualname == _OWNER_DEF
    }
    assert owned == {'WellPlateLoader', 'ObjectiveLoader'}
