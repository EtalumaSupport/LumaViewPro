# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run may only ask for motion on axes this scope has.

A move on an axis the scope lacks does nothing and says nothing, so a run
whose steps sit at different places on that axis completes: every image is
taken at one place and saved under its own step's name and coordinates. A
manual scope (an LS620 or LS560) lacks every axis; a Z-only scope (an LS820)
lacks X and Y. prepare() refuses such a run before anything is captured.

Steps at one place on a missing axis ask for nothing there and are admitted:
a single-location time lapse on a manual scope is the run this keeps.
Autofocus moves Z, so it needs a Z axis.
"""

from __future__ import annotations

import dataclasses

import pytest

from modules.exceptions import ProtocolRunRefusedError
from tests.test_a_protocol_needs_its_objectives_on_the_turret import _prepare
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_multi_step_protocol,
    executor,
    executors,
    scope,
)

UNREACHABLE = 'positions_unreachable'
HERE = {'x': 10.0, 'y': 20.0, 'z': 5000.0}
ELSEWHERE_XY = {'x': 30.0, 'y': 20.0, 'z': 5000.0}
ELSEWHERE_Z = {'x': 10.0, 'y': 20.0, 'z': 6000.0}
AUTOFOCUS = {**HERE, 'auto_focus': True}

# The axes the scope has, the steps' positions, and the answer: None is
# admitted, a string is the reason the refusal must carry. Written out
# rather than computed, so the table cannot agree with the code by
# sharing its predicate.
RULE_TABLE = [
    # A manual scope: one place and no autofocus, or nothing.
    ((), (HERE, HERE), None),
    ((), (HERE, ELSEWHERE_XY), UNREACHABLE),
    ((), (HERE, ELSEWHERE_Z), UNREACHABLE),
    ((), (AUTOFOCUS,), UNREACHABLE),
    # A Z-only scope: Z and autofocus, but one X/Y place.
    (('Z',), (HERE, HERE), None),
    (('Z',), (HERE, ELSEWHERE_XY), UNREACHABLE),
    (('Z',), (HERE, ELSEWHERE_Z), None),
    (('Z',), (AUTOFOCUS,), None),
    # A motorized scope: everything.
    (('X', 'Y', 'Z'), (HERE, ELSEWHERE_XY), None),
    (('X', 'Y', 'Z'), (HERE, ELSEWHERE_Z), None),
    (('X', 'Y', 'Z'), (AUTOFOCUS,), None),
]


@pytest.mark.parametrize(('axes', 'positions', 'refused'), RULE_TABLE)
def test_the_rule(executor, scope, monkeypatch, tmp_path, axes, positions, refused):
    # ScopeCapabilities is frozen: the axes are swapped by replacing the
    # whole record, which is what a scope built with those axes carries.
    monkeypatch.setattr(scope, 'capabilities', dataclasses.replace(scope.capabilities, axes=axes))
    protocol = _make_multi_step_protocol(
        [{**position, 'name': f'step_{i}'} for i, position in enumerate(positions)]
    )
    if refused is None:
        _prepare(executor, protocol, tmp_path)
    else:
        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, protocol, tmp_path)
        assert refusal.value.reason == refused
