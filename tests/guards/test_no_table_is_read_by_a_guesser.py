# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Guard: no production code reads a table with pandas' readers.

Every LumaViewPro table is written by csv.writer, each cell as its text.
pd.read_csv guesses: it read a step named NA as a missing value, cut a cell
at a NUL, and resolved quotes its own way, so a protocol, a post-processing
record and a results file each read back as other values. The one reader is
common_utils.read_table, and each reader types its own columns.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT

_PANDAS_READERS = {'read_csv', 'read_table', 'read_fwf', 'read_excel'}


def _production_sources():
    for directory in ('modules', 'drivers', 'ui', 'tools', 'scripts', 'lib'):
        yield from (REPO_ROOT / directory).rglob('*.py')
    yield from REPO_ROOT.glob('*.py')


def _pandas_reads(source: str) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and node.attr in _PANDAS_READERS
        and isinstance(node.value, ast.Name)
        and node.value.id in ('pd', 'pandas')
    ]


def test_the_guard_sees_a_pandas_reader():
    assert _pandas_reads('import pandas as pd\ntable = pd.read_csv(path)\n') == [2]
    assert _pandas_reads('header, rows = read_table(text, sep=",")\n') == []


def test_no_production_table_is_read_by_pandas():
    offenders = [
        f'{path.relative_to(REPO_ROOT)}:{line}'
        for path in _production_sources()
        for line in _pandas_reads(path.read_text(encoding='utf-8'))
    ]
    assert not offenders, (
        'a table is read with a pandas reader, which guesses at its cells; read it with '
        'common_utils.read_table and type its columns:\n' + '\n'.join(offenders)
    )
