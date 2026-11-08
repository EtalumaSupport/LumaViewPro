# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Nothing LumaViewPro ships forces a garbage collection.

``gc.collect()`` walks the whole heap holding the GIL, so every thread stops
for it, the live view's among them. A protocol run's scan loop forced one every
60 s: on a field run's heap of about 2.59M blocks each took about 240 ms, and
851 of the run's 942 slow frames fell within 0.4 s after one. Python's own
collector runs without being asked; a full collection comes when the long-lived
population has grown by a quarter. Tests may force one; the app may not.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, tracked_files


def _forced_collections(tree: ast.Module) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == 'collect'
            and isinstance(node.value, ast.Name)
            and node.value.id == 'gc'
        ):
            found.append((node.lineno, 'gc.collect'))
        elif isinstance(node, ast.ImportFrom) and node.module == 'gc':
            found.extend(
                (node.lineno, 'from gc import collect')
                for alias in node.names
                if alias.name == 'collect'
            )
    return found


def test_no_shipped_module_forces_a_collection():
    offenders = []
    for rel_path in tracked_files():
        if rel_path.startswith('tests/'):
            continue
        source = (REPO_ROOT / rel_path).read_text(encoding='utf-8')
        tree = ast.parse(source, filename=rel_path)
        offenders.extend(f'{rel_path}:{line} {name}' for line, name in _forced_collections(tree))
    assert not offenders, (
        'A forced collection stops every thread, the live view included, for '
        "as long as the heap takes to walk; leave collection to Python's own "
        'collector:\n' + '\n'.join(offenders)
    )
