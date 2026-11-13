# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A long action has one form on the wire: its start form.

In Python, "x waits, start_x returns an Operation": both forms are API. On
the wire only the start form is served, as a job a client reads and stops
(``DELETE /api/v1/jobs/<id>``). A waiting form served beside it was a
second door to the same action whose job could not be stopped -- held as a
job past the client's wait, it holds no operation, and its cancel is
refused ``not_cancellable`` -- and the docs taught the waiting route. So a
class whose wire member is ``start_x`` has no wire member ``x``, nor
``run_x`` (the protocol runner's waiting forms). A waiting form is marked
``@api(in_process=True)``.

A static read of the marks: no session.
"""

from __future__ import annotations

from _lumascope.modules import wire_encoding
from _lumascope.modules.api_surface import API, mark_of

_START = 'start_'


def _members(cls):
    """Every name on *cls* and its bases, the nearest definition winning."""
    return {name: value for base in reversed(cls.__mro__) for name, value in vars(base).items()}


def test_no_waiting_form_is_served_beside_its_start_form():
    served_twice = []
    for owner, cls in sorted(wire_encoding.project_classes().items()):
        members = _members(cls)
        for name, member in sorted(members.items()):
            if not name.startswith(_START) or mark_of(member) != API:
                continue
            action = name[len(_START) :]
            for waiting in (action, f'run_{action}'):
                if mark_of(members.get(waiting)) == API:
                    served_twice.append(f'{owner}.{waiting} (beside {owner}.{name})')
    assert served_twice == [], (
        'a waiting form is served beside its start form; mark it @api(in_process=True):\n'
        + '\n'.join(served_twice)
    )


def test_the_guard_reads_the_start_forms_it_judges():
    # The walk finds the start forms it is about, so its empty answer above
    # is the marks', not an empty walk's.
    found = {
        f'{owner}.{name}'
        for owner, cls in wire_encoding.project_classes().items()
        for name, member in _members(cls).items()
        if name.startswith(_START) and mark_of(member) == API
    }
    assert {'MotionAPI.start_move_z', 'ScopeSession.start_go_to_step'} <= found
