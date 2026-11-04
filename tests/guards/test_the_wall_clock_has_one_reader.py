# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The wall clock is read as a datetime in one place: ``modules/wall_time.py``.

Every time LVP saves or publishes states its UTC offset, so two saved times
compare however many daylight-saving changes lie between them. A read of
the wall clock as a datetime anywhere else (``datetime.now()``,
``datetime.fromtimestamp(...)``, and their spellings) makes a time whose form
the call chose, most often naive local, and nothing stops it reaching a
record beside times that state their offset. So production reads "now"
through ``wall_time.now()``, a file's time through ``wall_time.local_time``,
and the wall clock's integer through ``drivers/host_clock.py``, which reads
no datetime. A file-name label formats the time ``wall_time`` gives it; a
driver, which imports no module, labels with ``time.strftime``.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, Words, tracked_python_files

# The production trees, and the root's own modules.
_PRODUCTION = ('modules/', 'drivers/', 'rest/', 'ui/')
_THE_ONE_READER = 'modules/wall_time.py'

# The datetime and date members that read the wall clock.
_WALL_READS = Words({'now', 'utcnow', 'today', 'fromtimestamp', 'utcfromtimestamp'})


def _names_a_time_type(node: ast.expr) -> bool:
    """``datetime``, ``date``, or any ``<...>.datetime`` / ``<...>.date``."""
    if isinstance(node, ast.Name):
        return node.id in ('datetime', 'date')
    return isinstance(node, ast.Attribute) and node.attr in ('datetime', 'date')


def offences(tree: ast.AST) -> list[int]:
    """The lines in ``tree`` that read the wall clock as a datetime."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _WALL_READS
        and _names_a_time_type(node.func.value)
    ]


def _production_files() -> list[str]:
    return [
        path
        for path in tracked_python_files()
        if path != _THE_ONE_READER and (path.startswith(_PRODUCTION) or '/' not in path)
    ]


def test_every_spelling_of_a_wall_read_is_found():
    """The guard's own known positives, so its clean result below means none."""
    for source in (
        'datetime.now()',
        'datetime.datetime.now()',
        '_dt.datetime.now().astimezone()',
        'datetime.datetime.now(datetime.UTC)',
        'datetime.datetime.fromtimestamp(stat.st_mtime)',
        'date.today()',
        'datetime.utcnow()',
    ):
        assert offences(ast.parse(source)) == [1], source
    assert offences(ast.parse('wall_time.now()')) == []


def test_the_wall_clock_is_read_as_a_datetime_only_through_wall_time():
    offenders = []
    for path in _production_files():
        source = (REPO_ROOT / path).read_text(encoding='utf-8')
        for lineno in offences(ast.parse(source, filename=path)):
            offenders.append(f'{path}:{lineno}')
    assert not offenders, (
        'these read the wall clock as a datetime outside modules/wall_time.py; read '
        'wall_time.now() or wall_time.local_time(wall_ns), so the time states its '
        'offset: ' + ', '.join(offenders)
    )
