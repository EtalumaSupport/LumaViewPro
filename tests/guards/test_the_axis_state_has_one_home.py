# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``AxisState`` is defined in the leaf ``axes`` module and imported from there alone.

It lived in the API package's ``_constants``, re-exported by the
composition root, the package and the motion API, so four import paths
named one class and the refusals' module, which the package imports,
spelled the homing state as a bare string to dodge the cycle. Beside
``Axis`` in the leaf module, every module reaches it with no cycle; the
published ``lumascope`` package is the one other door, for callers
outside the project. An AST walk sees every import at any nesting depth.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, tracked_files

_HOMES = {'_lumascope.modules.axes', 'lumascope'}


def test_the_axis_state_is_defined_in_the_axes_module():
    from _lumascope.modules.axes import AxisState

    assert AxisState.__module__ == '_lumascope.modules.axes'


def test_every_import_of_the_axis_state_names_its_home():
    found = []
    for rel_path in tracked_files():
        tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module not in _HOMES
                and any(alias.name == 'AxisState' for alias in node.names)
            ):
                found.append(f'{rel_path}:{node.lineno} imports AxisState from {node.module}')
    assert found == [], (
        'AxisState is imported from _lumascope.modules.axes (or the published lumascope '
        'package), never from a module that only re-imports it:\n' + '\n'.join(found)
    )
