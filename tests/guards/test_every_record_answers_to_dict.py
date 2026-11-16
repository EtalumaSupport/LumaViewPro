# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every published record answers ``to_dict()``, and its ``to_dict()`` is API in-process.

``api_fields`` gives each record it marks the method, so a record added
later has it with no line to remember. The method is the one marked
function every record shares, so a record that defines a ``to_dict`` of
its own, unmarked or marked otherwise, reads as private or as another
contract than the reference states. This walks every record under
``src/_lumascope/modules/`` that ``api_fields`` marks and names each one
without the method or whose method is not marked in-process.
"""

from __future__ import annotations

from _lumascope.modules import wire_encoding
from _lumascope.modules.api_surface import FIELDS_ATTRIBUTE, IN_PROCESS, is_record, mark_of


def test_every_published_record_answers_to_dict_marked_in_process():
    published = [
        cls
        for cls in wire_encoding.project_classes().values()
        if is_record(cls) and FIELDS_ATTRIBUTE in vars(cls)
    ]
    without = sorted(
        f'{c.__module__}.{c.__qualname__}'
        for c in published
        if not callable(getattr(c, 'to_dict', None))
    )
    unmarked = sorted(
        f'{c.__module__}.{c.__qualname__}'
        for c in published
        if callable(getattr(c, 'to_dict', None)) and mark_of(c.to_dict) != IN_PROCESS
    )

    assert published, 'no published record was found: the walk itself is broken'
    assert not without, 'A published record has no to_dict():\n  ' + '\n  '.join(without)
    assert not unmarked, (
        "A published record's to_dict() is not marked in-process:\n  " + '\n  '.join(unmarked)
    )
