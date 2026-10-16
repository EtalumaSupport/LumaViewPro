# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""No production code makes an array writeable again.

The camera marks every delivered frame read-only before any consumer has
it, and a saved frame is read-only from capture to file, so whoever holds
one -- the preview, a recording, a capture, a listener -- cannot change
the pixels another holds. A frame that owns its data can still be flipped
back with ``flags.writeable = True`` or ``setflags(write=True)``; that
undoes the mark for every holder at once. A consumer that needs to draw
draws on its own copy.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules, production_modules


def _is_false(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _writeable_flag(target: ast.AST) -> bool:
    """``<x>.flags.writeable`` or ``<x>.flags['WRITEABLE']`` (or ``'W'``)."""
    if isinstance(target, ast.Attribute):
        return target.attr == 'writeable' and _is_flags(target.value)
    if isinstance(target, ast.Subscript) and _is_flags(target.value):
        key = target.slice
        return isinstance(key, ast.Constant) and key.value in ('WRITEABLE', 'W')
    return False


def _is_flags(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == 'flags'


def _writeable_again(tree: ast.Module) -> list[int]:
    """Lines that set an array's write flag to anything but ``False``."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(_writeable_flag(t) for t in node.targets) and not _is_false(node.value):
                lines.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'setflags'
        ):
            write = [kw.value for kw in node.keywords if kw.arg == 'write']
            write += node.args[:1]
            if any(not _is_false(value) for value in write):
                lines.append(node.lineno)
    return sorted(lines)


def _all_production_modules():
    yield from production_modules()
    yield from iter_package_modules(('drivers', 'rest', 'tools'))


def test_no_production_module_makes_an_array_writeable_again():
    found = [
        f'{rel_path}:{lineno}'
        for rel_path, tree in _all_production_modules()
        for lineno in _writeable_again(tree)
    ]
    assert found == [], (
        'a delivered or saved frame is read-only for every holder; draw on '
        f'a copy instead of making it writeable again: {found}'
    )


def test_the_guard_sees_each_way_of_making_an_array_writeable():
    # Each form the guard refuses, and the read-only mark it allows.
    refused = ast.parse(
        'a.flags.writeable = True\n'
        "a.flags['WRITEABLE'] = 1\n"
        'a.setflags(write=True)\n'
        'a.setflags(flag)\n'
        'a.flags.writeable = keep\n'
    )
    allowed = ast.parse('a.flags.writeable = False\na.setflags(write=False)\n')
    assert _writeable_again(refused) == [1, 2, 3, 4, 5]
    assert _writeable_again(allowed) == []
