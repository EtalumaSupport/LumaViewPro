# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every timestamped log format is made by _lumascope.modules.log_stamp.

Each writer once spelled its own stamp, and LumaViewPro's logs carried five
forms in three layouts: serial.log, camera.log and api.log had no date, so a
support bundle's trace lines could only be placed on a day by counting
midnight rollovers. A string naming ``%(asctime)`` (a format, wherever it is
held before it reaches a formatter), a ``datefmt``, or a ``Formatter``
subclass anywhere else in production code spells a second stamp. A console
format with no stamp passes.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, tracked_files

_OWNER = 'src/_lumascope/modules/log_stamp.py'


def _name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else getattr(node, 'attr', None)


def _spells_a_stamp(node: ast.AST) -> bool:
    # A Formatter subclass passes its format to __init__, where no call names
    # Formatter: lvp_logger's five formatter classes did.
    if isinstance(node, ast.ClassDef):
        return any(_name(base) == 'Formatter' for base in node.bases)
    # A format held in a module constant reaches its formatter by name, so the
    # string itself is what is refused.
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and '%(asctime)' in node.value
    return isinstance(node, ast.keyword) and node.arg == 'datefmt'


def offenders_in(rel_path: str, source: str) -> list[str]:
    tree = ast.parse(source, filename=rel_path)
    lines = sorted(node.lineno for node in ast.walk(tree) if _spells_a_stamp(node))
    return [f'{rel_path}:{line}' for line in lines]


def test_no_log_stamp_is_spelled_outside_log_stamp():
    offenders = []
    for rel_path in tracked_files():
        if rel_path.startswith('tests/') or rel_path == _OWNER:
            continue
        offenders += offenders_in(rel_path, (REPO_ROOT / rel_path).read_text(encoding='utf-8'))

    assert not offenders, (
        'a log format that spells its own timestamp, outside src/_lumascope/modules/log_stamp.py '
        f'(use make_formatter, or now_stamp for a line written by hand): {offenders}'
    )


def test_the_guard_sees_a_stamped_format_and_passes_a_stampless_console():
    planted = (
        'import logging\n'
        "a = logging.Formatter('%(asctime)s %(message)s')\n"
        "b = logging.Formatter(fmt='%(message)s', datefmt='%H:%M:%S')\n"
        "logging.basicConfig(format='%(asctime)s %(message)s')\n"
        "c = logging.Formatter('[%(levelname)s %(name)s] %(message)s')\n"
        'class Trace(logging.Formatter):\n'
        '    pass\n'
        "_FORMAT = '%(asctime)s %(levelname)s %(message)s'\n"
        'd = logging.Formatter(_FORMAT)\n'
    )

    assert offenders_in('planted.py', planted) == [
        'planted.py:2',
        'planted.py:3',
        'planted.py:4',
        'planted.py:6',
        'planted.py:8',
    ]
