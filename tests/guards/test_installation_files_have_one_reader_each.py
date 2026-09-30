# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each file the installation ships is named in production code only by its owner.

scopes.json was read by four modules and the GUI, and the motor defaults by
three sites, each from the installation's default folder, so a scope
started on another folder read its catalogues from one folder and its
models and motor defaults from another. The scope now reads them once, in
``Lumascope._read_catalogues``, and everything else reads its copy. A file
name in any other production call is a second reader; this walk refuses it
at commit.

Two owners stay outside the scope by design: the release layer vocabulary
(``layer_record.release_catalogue``), which the settings bring-up needs
before any scope exists, and the firmware bench tool, which drives a motor
board with no scope at all.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules, production_modules, walk_defs

_FILES = ('scopes.json', 'motorconfig_defaults.json', 'labware.json', 'objectives.json')

# (module, qualname) -> the files that def may name.
_OWNERS = {
    ('modules/lumascope_api/_lumascope.py', 'Lumascope._read_catalogues'): {
        'scopes.json',
        'motorconfig_defaults.json',
    },
    ('modules/labware_loader.py', 'LabwareLoader.__init__'): {'labware.json'},
    ('modules/objectives_loader.py', 'ObjectiveLoader.__init__'): {'objectives.json'},
    ('modules/layer_record.py', 'release_catalogue'): {'scopes.json'},
    ('modules/layer_record.py', 'load_scope_models'): {'scopes.json'},
    ('tools/firmware_tools.py', '_connect_motor_board'): {'motorconfig_defaults.json'},
}


def _all_production_modules():
    yield from production_modules()
    yield from iter_package_modules(('drivers', 'tools'))


def _named_in_calls(tree: ast.Module):
    """Yield (qualname or '<module>', lineno, file) for every file name passed to a call."""
    seen = set()

    def names(node):
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            for arg in [*call.args, *(k.value for k in call.keywords)]:
                if isinstance(arg, ast.Constant) and arg.value in _FILES and id(arg) not in seen:
                    seen.add(id(arg))
                    yield arg.lineno, arg.value

    for qualname, node in walk_defs(tree.body):
        for lineno, name in names(node):
            yield qualname, lineno, name
    for lineno, name in names(tree):
        yield '<module>', lineno, name


def _load_scope_models_calls(tree: ast.Module):
    for qualname, node in walk_defs(tree.body):
        for call in ast.walk(node):
            if isinstance(call, ast.Call):
                func = call.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
                if name == 'load_scope_models':
                    yield qualname, call.lineno


def test_only_the_owners_name_an_installation_file():
    found = []
    for rel_path, tree in _all_production_modules():
        for qualname, lineno, name in _named_in_calls(tree):
            if name in _OWNERS.get((rel_path, qualname), set()):
                continue
            found.append(f'{rel_path}:{lineno} {qualname} names {name}')
    assert found == [], (
        "a second reader of an installation file; read the scope's copy "
        f'(scope.scope_models, scope.wellplate_loader, scope.objective_helper) instead: {found}'
    )


def test_only_the_scope_reads_the_model_catalogue():
    found = [
        f'{rel_path}:{lineno} {qualname}'
        for rel_path, tree in _all_production_modules()
        for qualname, lineno in _load_scope_models_calls(tree)
        if (rel_path, qualname)
        != ('modules/lumascope_api/_lumascope.py', 'Lumascope._read_catalogues')
    ]
    assert found == [], f'read scope.scope_models instead of the file: {found}'


def test_the_guard_sees_every_owner_it_exempts():
    # Each exemption must name a real def that names its files, or a rename
    # would leave the guard exempting nothing while its main check passed.
    trees = dict(_all_production_modules())
    for (rel_path, qualname), files in _OWNERS.items():
        named = {name for q, _lineno, name in _named_in_calls(trees[rel_path]) if q == qualname}
        assert named == files, f'{rel_path} {qualname} names {named}, expected {files}'
