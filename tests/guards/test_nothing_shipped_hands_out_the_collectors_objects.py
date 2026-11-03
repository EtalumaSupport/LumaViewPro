# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Nothing LumaViewPro ships asks the garbage collector for its objects.

``gc.get_objects()``, ``gc.get_referrers()`` and ``gc.get_referents()`` hand
the calling thread references to objects the collector tracks, a tuple
another thread is still building among them. When that thread's generator
released the GIL, its ``tuple()`` found a second reference and raised
``SystemError: bad argument to internal function``: the metrics tick's object
count ended a protocol run that way mid-move. A count or a collection
(``sys.getallocatedblocks()``, ``gc.collect()``, ``gc.get_stats()``) hands out
nothing and is not refused. Tests may call them; the app may not.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, RetiredNames, tracked_python_files

_HANDS_OUT = RetiredNames({'get_objects', 'get_referrers', 'get_referents'})


def _calls_that_hand_out(tree: ast.Module) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _HANDS_OUT
            and isinstance(node.value, ast.Name)
            and node.value.id == 'gc'
        ):
            found.append((node.lineno, f'gc.{node.attr}'))
        elif isinstance(node, ast.ImportFrom) and node.module == 'gc':
            found.extend(
                (node.lineno, f'from gc import {alias.name}')
                for alias in node.names
                if alias.name in _HANDS_OUT
            )
    return found


def test_no_shipped_module_asks_the_collector_for_its_objects():
    offenders = []
    for rel_path in tracked_python_files():
        if rel_path.startswith('tests/'):
            continue
        source = (REPO_ROOT / rel_path).read_text(encoding='utf-8')
        tree = ast.parse(source, filename=rel_path)
        offenders.extend(f'{rel_path}:{line} {name}' for line, name in _calls_that_hand_out(tree))
    assert not offenders, (
        "These hand a thread the collector's objects, a half-built tuple among them; "
        'count with sys.getallocatedblocks() instead:\n' + '\n'.join(offenders)
    )
