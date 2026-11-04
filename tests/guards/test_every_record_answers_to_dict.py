# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every published record answers ``to_dict()``.

``api_fields`` gives each record it marks the method, so a record added
later has it with no line to remember. This walks every record under
``modules/`` that ``api_fields`` marks and names each one without it.
"""

from __future__ import annotations

from modules import wire_encoding
from modules.api_surface import FIELDS_ATTRIBUTE, is_record


def test_every_published_record_answers_to_dict():
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

    assert published, 'no published record was found: the walk itself is broken'
    assert not without, 'A published record has no to_dict():\n  ' + '\n  '.join(without)
