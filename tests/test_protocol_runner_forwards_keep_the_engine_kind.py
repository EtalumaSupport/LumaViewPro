# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The L2 protocol runner forwards each engine member in the engine's own kind.

`ProtocolRunner` wraps the session's `SequencedCaptureRunner` member by
member so scripts, REST and the GUI observe one run through one shape. Two
of its forwards called engine members that are properties, so
`runner.video_drain_busy()` raised "'bool' object is not callable" on every
use and `runner.video_pending_writes()` the same on an int -- a facade pair
that could not be called at all, unnoticed because no production caller and
no test had reached them. The engine's spelling is the canonical one: the
app-close gate and the session read both as attributes.

The first test pins the behaviour; the second pins the class, by comparing
every same-named forward in the facade against the engine member's kind, so
a future forward cannot ship with the wrong one.
"""

import ast

from modules.protocol_runner import ProtocolRunner
from modules.sequenced_capture_runner import SequencedCaptureRunner
from tests.ast_seams import parse_module


class _Engine:
    """The two answers the facade must forward, spelled as the engine spells them."""

    def __init__(self, busy: bool, pending: int):
        self._busy = busy
        self._pending = pending

    @property
    def video_drain_busy(self) -> bool:
        return self._busy

    @property
    def video_pending_writes(self) -> int:
        return self._pending


def _facade_over(engine) -> ProtocolRunner:
    runner = ProtocolRunner.__new__(ProtocolRunner)
    runner._executor = engine
    return runner


def test_the_facade_answers_the_engine_by_attribute_read():
    """What an L2 caller sees: the engine's values, not a bound method."""
    runner = _facade_over(_Engine(busy=True, pending=7))
    assert runner.video_pending_writes == 7
    runner = _facade_over(_Engine(busy=False, pending=0))
    assert runner.video_pending_writes == 0


def _member_kinds(cls: type) -> dict[str, str]:
    """'property' or 'method' for every function-like member declared on cls."""
    kinds = {}
    for name, member in vars(cls).items():
        if isinstance(member, property):
            kinds[name] = 'property'
        elif callable(member):
            kinds[name] = 'method'
    return kinds


def _same_named_forwards() -> set[str]:
    """Facade members whose body reaches `self._executor.<same name>`."""
    tree = parse_module('modules/protocol_runner.py')
    forwards = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == 'ProtocolRunner'):
            continue
        for fn in node.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            for sub in ast.walk(fn):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Attribute)
                    and sub.value.attr == '_executor'
                    and sub.attr == fn.name
                ):
                    forwards.add(fn.name)
    return forwards


def test_every_same_named_forward_keeps_the_engine_members_kind():
    """The class, not the instance: a forward is a property exactly when the
    engine member is, so calling one can never raise on its own kind."""
    facade = _member_kinds(ProtocolRunner)
    engine = _member_kinds(SequencedCaptureRunner)
    forwards = _same_named_forwards()
    assert forwards, 'the source read found no forwards; the instrument is broken, not the facade'
    mismatched = {
        name: (facade[name], engine[name]) for name in forwards if facade[name] != engine[name]
    }
    assert mismatched == {}, (
        f'facade members whose kind differs from the engine member they forward '
        f'(facade, engine): {mismatched}'
    )
