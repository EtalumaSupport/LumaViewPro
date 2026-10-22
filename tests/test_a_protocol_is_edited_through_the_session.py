# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every edit of a protocol a caller makes is a Session member taking the protocol.

The protocol panel set the schedule and the capture root through the
``Protocol`` object's own writers, and a REST caller changed a step's
autofocus the same way: a second GUI or a script had to know the data
object's writers to do what the panel does. Each edit is now a member of
the Session, and ``Protocol`` is data a caller reads.
"""

from __future__ import annotations


import pytest

from modules.coord_transformations import PlateLocation
from modules.protocol import ProtocolScheduleRefusedError, StepNotFoundError
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture


def _two_steps(session):
    protocol = session.create_empty_protocol()
    session.add_step(
        protocol, layers=('BF', 'Blue'), at=PlateLocation(x_mm=20.0, y_mm=15.0, z_um=6000.0)
    )
    return protocol


def test_the_schedule_is_set_and_an_unrunnable_one_refused_leaving_it(session):
    protocol = session.create_empty_protocol()

    session.set_protocol_schedule(protocol, period_s=5 * 60, duration_s=2 * 3600)
    assert (protocol.period_s(), protocol.duration_s()) == (5 * 60, 2 * 3600)

    with pytest.raises(ProtocolScheduleRefusedError) as refused:
        session.set_protocol_schedule(protocol, period_s=0.5, duration_s=3600)
    assert (refused.value.reason, refused.value.key) == ('schedule_not_runnable', 'period')
    assert (protocol.period_s(), protocol.duration_s()) == (5 * 60, 2 * 3600)


def test_the_capture_root_is_kept_as_given(session):
    protocol = session.create_empty_protocol()

    session.set_protocol_capture_root(protocol, 'plate 7')

    assert protocol.capture_root() == 'plate 7'


def test_one_steps_autofocus_or_every_steps_is_set(session):
    protocol = _two_steps(session)

    session.set_step_autofocus(protocol, enabled=True, step_index=1)
    assert protocol.steps()['Auto_Focus'].tolist() == [False, True]

    session.set_step_autofocus(protocol, enabled=True)
    assert protocol.steps()['Auto_Focus'].tolist() == [True, True]


def test_an_autofocus_edit_of_a_step_the_protocol_lacks_is_refused_and_changes_nothing(session):
    protocol = _two_steps(session)

    with pytest.raises(StepNotFoundError) as refused:
        session.set_step_autofocus(protocol, enabled=True, step_index=2)

    assert refused.value.reason == 'no_such_step'
    assert protocol.steps()['Auto_Focus'].tolist() == [False, False]
