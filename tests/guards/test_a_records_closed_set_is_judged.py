# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every record field typed with a closed set is judged as the record is built.

``api_fields`` finds a dataclass's closed-set fields from the record's
module when it marks the record -- a member, a tuple, list or frozenset of
them, or a mapping keyed by them -- and judges them at each build, so a
caller's plain string becomes the member and a value outside the set is
refused ``not_offered`` (``modules.closed_sets``). A field it cannot find
-- its set imported only for type checking, defined below the record, held
in some other form, or on a NamedTuple, which has no build to judge in --
would hold whatever string a caller built it with, and a comparison with a
member would quietly go the wrong way. This resolves every published
record's annotations in full and names each such field.
"""

import dataclasses
import enum
import typing

from modules import api_arguments, api_surface, wire_encoding
from modules.lumascope_api.illumination import LedReading, LedState


def _names_a_closed_set(annotation: object) -> bool:
    if isinstance(annotation, type) and issubclass(annotation, enum.StrEnum):
        return True
    return any(_names_a_closed_set(arm) for arm in typing.get_args(annotation))


def _unjudged() -> dict[str, str]:
    found = {}
    for name, cls in sorted(wire_encoding.project_classes().items()):
        if not api_surface.is_record(cls) or not api_surface.fields_of(cls):
            continue
        hints = typing.get_type_hints(cls, localns=api_arguments._names())
        judged = api_surface.closed_fields_of(cls) if dataclasses.is_dataclass(cls) else {}
        for field, annotation in hints.items():
            if typing.get_origin(annotation) is typing.ClassVar:
                continue
            if _names_a_closed_set(annotation) and field not in judged:
                found[f'{name}.{field}'] = str(annotation)
    return found


def test_the_guard_sees_a_judged_field():
    assert api_surface.closed_fields_of(LedReading) == {
        'state': api_surface.ClosedField(LedState, False)
    }


def test_every_records_closed_set_field_is_judged():
    assert _unjudged() == {}
