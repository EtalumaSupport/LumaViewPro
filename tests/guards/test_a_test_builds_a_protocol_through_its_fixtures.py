# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A test builds or reads a protocol through ``tests/protocol_fixtures.py``.

A test that called ``Protocol(...)`` or ``Protocol.from_file(...)`` itself
handed in the tiling grids and the unit at every site, so each new argument
a protocol needs was an edit in every test that built one: the tiling-grids
change touched 73 test files, the unit 18. ``protocol_of`` and
``protocol_from_file`` hand them in once.
"""

import ast

from tests.ast_seams import REPO_ROOT, tracked_python_files

_THE_FIXTURES = 'tests/protocol_fixtures.py'


def _builds_a_protocol(node) -> bool:
    if not isinstance(node, ast.Call):
        return False
    called = node.func
    if isinstance(called, ast.Name):
        return called.id == 'Protocol'
    return (
        isinstance(called, ast.Attribute)
        and called.attr == 'from_file'
        and isinstance(called.value, ast.Name)
        and called.value.id == 'Protocol'
    )


def test_no_test_calls_the_protocol_constructor_or_reader_itself():
    offenders = [
        f'{rel}:{node.lineno}'
        for rel in tracked_python_files()
        if rel.startswith('tests/') and rel != _THE_FIXTURES
        for node in ast.walk(ast.parse((REPO_ROOT / rel).read_text(encoding='utf-8')))
        if _builds_a_protocol(node)
    ]
    assert offenders == [], (
        'a test builds a protocol with protocol_of and reads one with '
        f'protocol_from_file (tests/protocol_fixtures.py): {offenders}'
    )
