# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A frame's record is built by its module's two entries and nowhere else.

A still's record and a recorded frame's were built by two builders: the
grab's in the imaging API and a flat dict in the recording legs, which
read the same chunks two ways and stated different facts. A record is now
built only by ``frame_record.grab_record`` and
``frame_record.delivered_frame_record``, which share the one reading of a
frame's chunks. A construction anywhere else is a third builder; this walk
over every tracked module refuses it at commit. ``tests/frame_records.py``
is the tests' one description of a record, so a field added to it is added
once.
"""

from __future__ import annotations

import ast

from tests.ast_seams import MemberNames, parse_module, tracked_python_files, walk_defs

_OWNER_MODULE = 'modules/lumascope_api/frame_record.py'
_OWNER_DEFS = MemberNames(_OWNER_MODULE, {'grab_record', 'delivered_frame_record'})
_TEST_DESCRIPTION = 'tests/frame_records.py'


def _constructions(tree: ast.Module):
    """Yield (qualname or '<module>', lineno) for every ``FrameRecord(...)`` call."""
    seen = set()
    for qualname, node in walk_defs(tree.body):
        for inner in ast.walk(node):
            if _is_construction(inner) and id(inner) not in seen:
                seen.add(id(inner))
                yield qualname, inner.lineno
    for inner in ast.walk(tree):
        if _is_construction(inner) and id(inner) not in seen:
            yield '<module>', inner.lineno


def _is_construction(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
    return name == 'FrameRecord'


def test_a_frame_record_is_built_only_by_its_two_entries():
    found = []
    for rel_path in tracked_python_files():
        if rel_path == _TEST_DESCRIPTION:
            continue
        for qualname, lineno in _constructions(parse_module(rel_path)):
            if rel_path == _OWNER_MODULE and qualname in _OWNER_DEFS:
                continue
            found.append(f'{rel_path}:{lineno} {qualname}')
    assert found == [], (
        'a FrameRecord is built only by frame_record.grab_record and '
        'frame_record.delivered_frame_record (tests: tests/frame_records.py): '
        f'{found}'
    )


def test_the_guard_sees_the_entries_it_exempts():
    # The exemption must name the real builders, or a rename would leave the
    # guard exempting nothing while every other check still passed.
    built_in = {qualname for qualname, _ in _constructions(parse_module(_OWNER_MODULE))}
    assert built_in == _OWNER_DEFS
