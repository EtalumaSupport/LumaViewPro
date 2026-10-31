# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every rotating log handler is made by the one builder that reads LOG_CAPS.

A log's cap -- its size and its number of backups -- has one row in
``lvp_logger.LOG_CAPS``, and the table's sum is held under the support
bundle's budget. A ``RotatingFileHandler`` built anywhere else would spell
its own cap, outside the table and outside the budget, as every handler
did before the table existed (the logs then reached about 595 MB).
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, tracked_python_files

_BUILDER = ('lvp_logger.py', '_rotating_handler')


def _builds_a_rotating_handler(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
    return name == 'RotatingFileHandler'


def test_no_rotating_handler_is_built_outside_the_cap_table_builder():
    offenders = []
    for rel_path in tracked_python_files():
        if rel_path.startswith('tests/'):
            continue
        tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding='utf-8'), filename=rel_path)
        builders = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and (rel_path, node.name) == _BUILDER
        ]
        allowed = {id(n) for builder in builders for n in ast.walk(builder)}
        offenders += [
            f'{rel_path}:{node.lineno}'
            for node in ast.walk(tree)
            if _builds_a_rotating_handler(node) and id(node) not in allowed
        ]

    assert not offenders, (
        'a RotatingFileHandler built outside lvp_logger._rotating_handler spells its '
        f'own cap, outside LOG_CAPS and the bundle budget: {offenders}'
    )
