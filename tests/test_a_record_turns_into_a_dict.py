# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A record's ``to_dict()`` is its data as plain Python, for a caller in the same process.

MATLAB's ``struct()`` takes a dict and not a dataclass, so every published
record answers ``to_dict()``. It walks the record as the wire does -- a
nested record a dict, an enum its wire form, a sequence a list -- but keeps
what only a wire needs converted: a path, a handle and a running call stay
the live objects the caller can use, a NaN stays a NaN, and a key keeps its
type.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import pathlib

from modules.api_surface import api_fields
from modules.lumascope_api.protocols import StepTargets
from modules.operation import Operation


class _Shade(enum.StrEnum):
    DEEP = 'deep'


@api_fields('depth_um')
@dataclasses.dataclass(frozen=True)
class _Inner:
    depth_um: float


@api_fields('inner', 'shade', 'path', 'handle', 'running', 'level', 'by_channel', 'steps')
@dataclasses.dataclass(frozen=True)
class _Outer:
    inner: _Inner
    shade: _Shade
    path: pathlib.Path
    handle: object
    running: Operation
    level: float
    by_channel: dict
    steps: tuple


def test_a_record_answers_plain_data_keeping_the_live_objects():
    path = pathlib.Path('/somewhere/image.tiff')
    handle = object()
    running = Operation()
    record = _Outer(
        inner=_Inner(depth_um=2.5),
        shade=_Shade.DEEP,
        path=path,
        handle=handle,
        running=running,
        level=math.nan,
        by_channel={3: 'Blue'},
        steps=(1, 2),
    )

    answer = record.to_dict()

    assert answer['inner'] == {'depth_um': 2.5}
    assert answer['shade'] == 'deep'
    assert answer['path'] is path
    assert answer['handle'] is handle
    assert answer['running'] is running
    assert math.isnan(answer['level'])
    assert answer['by_channel'] == {3: 'Blue'}
    assert answer['steps'] == [1, 2]


def test_a_published_record_answers_its_fields():
    targets = StepTargets(turret_slot=2, x=1000.0, y=None, z=5000.0)

    assert targets.to_dict() == {'turret_slot': 2, 'x': 1000.0, 'y': None, 'z': 5000.0}
