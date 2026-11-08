# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""No first-party module reads another first-party module's private name.

A leading underscore is a module's statement that the name is its own
mechanism, free to change with it. A module that reads one through the
other's import (``wire_encoding._field_text``) depends on that mechanism,
so the owner can no longer change it without breaking a caller it cannot
see. Where a module needs what another's private name does, the owner
publishes it.

The walk reads every module of ``src/_lumascope`` for an attribute read
``<module>._<name>`` whose receiver is a first-party module bound by an
import in the reading file (``import a.b as m``, ``from a import m``),
resolved from the tracked files, never by importing them. A name imported
directly (``from a import _x``) and a read through a local alias are not
followed. The reads below predate the guard, each awaiting its owner's
public member; the pin only shrinks.
"""

import ast

from tests.ast_seams import iter_package_modules, module_name, tracked_files

_PINNED = frozenset(
    {
        'src/_lumascope/modules/config_helpers.py: common_utils._IS_WINDOWS',
        'src/_lumascope/modules/scope_session.py: ui_queue._set_ui_dispatcher',
        'src/_lumascope/modules/tech_support_report.py: settings_init._resolve_settings_path',
    }
)


def _first_party_modules() -> frozenset[str]:
    return frozenset(module_name(path) for path in tracked_files(['src/_lumascope']))


def _module_bindings(tree: ast.Module, first_party: frozenset[str]) -> set[str]:
    """The local names this file binds to a first-party module."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and alias.name in first_party:
                    bound.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                if f'{node.module}.{alias.name}' in first_party:
                    bound.add(alias.asname or alias.name)
    return bound


def _private_reads(path: str, tree: ast.Module, first_party: frozenset[str]) -> set[str]:
    modules = _module_bindings(tree, first_party)
    return {
        f'{path}: {node.value.id}.{node.attr}'
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in modules
        and node.attr.startswith('_')
        and not node.attr.startswith('__')
    }


def test_no_module_reads_a_private_name_through_another_modules_import():
    first_party = _first_party_modules()
    assert '_lumascope.modules.wire_encoding' in first_party, 'the module set lost its names'
    reads = set()
    for path, tree in iter_package_modules(['src/_lumascope']):
        reads |= _private_reads(path, tree, first_party)
    new = sorted(reads - _PINNED)
    gone = sorted(_PINNED - reads)
    assert new == [], "reads of another module's private name; publish it:\n" + '\n'.join(new)
    assert gone == [], 'pinned reads no longer made; drop them from the pin:\n' + '\n'.join(gone)
