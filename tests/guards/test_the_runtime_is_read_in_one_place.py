# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each fact of how the process was launched is read in one module.

Three modules each read the installer's marker their own way and each placed
the data folder from it; a PyInstaller build was asked about at five sites.
They agreed on the shipped layout only by accident, and a bundle run from
``dist`` was logged as a source run. ``path_utils.app_runtime`` is now the
one answer. It reads the marker itself, and asks
``_lumascope.lib.build_files.frozen`` whether the process is a PyInstaller
build: the build root needs that answer too, and ``src/_lumascope/lib/`` may not import
``src/_lumascope/modules/``. This walk refuses a marker read or a ``sys.frozen`` read
anywhere else in production code, drivers, tools and the REST server
included.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules, parse_module, production_modules

# Each read, and the one module that makes it.
_OWNERS = {
    'marker': 'src/_lumascope/modules/path_utils.py',
    'frozen': 'src/_lumascope/lib/build_files.py',
}
_MARKER = 'marker.lvpinstalled'
_ALLOWED: set[tuple[str, str]] = set()


def _all_production_modules():
    yield from production_modules()
    yield from iter_package_modules(
        ('src/_lumascope/drivers', 'tools', 'src/_lumascope/lib', 'src/_lumascope/rest')
    )
    yield 'src/_lumascope/lvp_logger.py', parse_module('src/_lumascope/lvp_logger.py')


def _is_frozen_read(node: ast.AST) -> bool:
    """``sys.frozen``, or ``getattr(sys, 'frozen', ...)``."""
    if isinstance(node, ast.Attribute):
        return node.attr == 'frozen' and isinstance(node.value, ast.Name) and node.value.id == 'sys'
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'getattr'
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == 'sys'
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == 'frozen'
    )


def _reads(tree: ast.Module):
    """Yield (lineno, 'marker' | 'frozen') for every runtime read in the module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == _MARKER:
            yield node.lineno, 'marker'
        elif _is_frozen_read(node):
            yield node.lineno, 'frozen'


def test_only_the_owners_ask_how_the_process_was_launched():
    found = []
    for rel_path, tree in _all_production_modules():
        for lineno, kind in _reads(tree):
            if rel_path != _OWNERS[kind] and (rel_path, kind) not in _ALLOWED:
                found.append(f'{rel_path}:{lineno} reads the {kind}')
    assert found == [], (
        'ask _lumascope.modules.path_utils.app_runtime() how the process was launched '
        f'(app_runtime().frozen for a PyInstaller build): {found}'
    )


def test_the_guard_sees_the_owners_and_the_allowed_read():
    # Each owner and exemption must still name a real read, or a rename would
    # leave the guard exempting nothing while its other check still passed.
    trees = dict(_all_production_modules())
    for kind, owner in _OWNERS.items():
        assert {k for _lineno, k in _reads(trees[owner])} == {kind}
    for rel_path, kind in _ALLOWED:
        assert kind in {k for _lineno, k in _reads(trees[rel_path])}


def test_the_guard_refuses_a_planted_read():
    planted = ast.parse(
        "import sys\nif getattr(sys, 'frozen', False) or sys.frozen:\n"
        "    open('marker.lvpinstalled')\n"
    )
    assert sorted(kind for _lineno, kind in _reads(planted)) == ['frozen', 'frozen', 'marker']
