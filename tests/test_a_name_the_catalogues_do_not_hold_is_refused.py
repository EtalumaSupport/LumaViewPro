# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A labware or objective name the installation's catalogues do not hold is refused as the request's.

Each catalogue's owner -- ``WellPlateLoader.resolve_plate_key`` for a plate,
``ObjectiveLoader.get_objective_info`` for an objective -- refused a name it
does not hold with a bare ``ConfigError``, which a wire client read as a
fault (500) and no caller could tell from any other settings failure. It is
now ``CatalogueNameRefusedError``, still a ``ConfigError`` for the callers
that recover from a bad stored name, with a reason, the owner's parameter as
``argument`` and the catalogue's keys as ``offered``, refused before anything
is stored. A null objective is the scope's "nothing selected"
(``ObjectiveUnknownError('none_selected')``) where a stored objective is read,
and a turret slot is refused by one check wherever it is taken. A null asked
of a member that names an objective to act on, or a slot that is no whole
number, is the ``@api`` door's
(``tests/test_an_argument_of_another_type_is_refused_at_the_door.py``). The
slot members are asked on a turret scope: with no turret they refuse the
absent turret first (``tests/test_turret_objective_assignment.py``).
"""

import threading
import traceback

import pytest
from fastapi.testclient import TestClient

from modules.exceptions import (
    ArgumentTypeRefusedError,
    CatalogueNameRefusedError,
    ConfigError,
    ObjectiveUnknownError,
    PositionOutOfRangeError,
    RefusalCause,
)
from rest.app import build_app
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session


@pytest.fixture(scope='module')
def z_changes():
    """Each change to the session's Z after its home, with the stack of the thread that made it."""
    return []


@pytest.fixture(scope='module')
def session(tmp_path_factory, z_changes):
    s = _session(tmp_path_factory.mktemp('c6'), 'LS850')

    def record(axis, at):
        if axis == 'Z':
            z_changes.append(
                (at, threading.current_thread().name, ''.join(traceback.format_stack()))
            )

    s.scope.motion.add_position_listener(record)
    yield s
    s.shutdown()


@pytest.fixture(scope='module')
def turret_session(tmp_path_factory):
    s = _session(tmp_path_factory.mktemp('c6t'), 'LS850T')
    yield s
    s.shutdown()


# The members that write a turret slot, asked of the turret scope.
_SLOT_WRITERS = {'confirm_objective at a slot', 'assign_turret_objective'}


def _selection(session):
    return {
        'labware': session.settings['protocol']['labware'],
        'objective_id': session.settings['objective_id'],
        'turret_objectives': dict(session.settings['turret_objectives']),
    }


LABWARE_MEMBERS = {
    'resolve_plate_key': lambda s, v: s.wellplate_loader.resolve_plate_key(v),
    'get_plate': lambda s, v: s.wellplate_loader.get_plate(v),
    'select_labware': lambda s, v: s.select_labware(v),
    'set_protocol_labware': lambda s, v: s.set_protocol_labware(s.create_empty_protocol(), v),
}

OBJECTIVE_MEMBERS = {
    'get_objective_info': lambda s, v: s.objective_helper.get_objective_info(objective_id=v),
    'select_objective': lambda s, v: s.select_objective(v),
    'confirm_objective': lambda s, v: s.confirm_objective(v),
    'confirm_objective at a slot': lambda s, v: s.confirm_objective(v, slot=2),
    'assign_turret_objective': lambda s, v: s.assign_turret_objective(2, v),
}


@pytest.mark.parametrize('name', ['Acme 1536 Ultra Plate', '', '96 well'])
@pytest.mark.parametrize('member', LABWARE_MEMBERS)
def test_a_plate_the_catalogue_lacks_is_refused_offering_its_plates(session, member, name):
    before = _selection(session)
    with pytest.raises(CatalogueNameRefusedError) as refused:
        LABWARE_MEMBERS[member](session, name)
    assert refused.value.reason == 'labware_unknown'
    assert refused.value.cause == RefusalCause.REQUEST
    assert isinstance(refused.value, ConfigError)
    assert refused.value.argument == 'labware_id'
    assert refused.value.value == name
    assert refused.value.offered == tuple(session.wellplate_loader.get_plate_list())
    assert _selection(session) == before


@pytest.mark.parametrize('name', ['zzz-no-such-objective', '10x', '10xOly'])
@pytest.mark.parametrize('member', OBJECTIVE_MEMBERS)
def test_an_objective_the_catalogue_lacks_is_refused_offering_its_keys(
    session, turret_session, member, name
):
    # Both are arguments, not fetched with request.getfixturevalue: a module
    # session built inside a test is built after the conftest's teardown mark,
    # so that test's teardown shut it down under the tests after it.
    session = turret_session if member in _SLOT_WRITERS else session
    before = _selection(session)
    with pytest.raises(CatalogueNameRefusedError) as refused:
        OBJECTIVE_MEMBERS[member](session, name)
    assert refused.value.reason == 'objective_not_in_catalogue'
    assert refused.value.cause == RefusalCause.REQUEST
    assert isinstance(refused.value, ConfigError)
    assert refused.value.argument == 'objective_id'
    assert refused.value.offered == tuple(session.objective_helper.get_objectives_list())
    assert _selection(session) == before


def test_no_objective_is_none_selected_where_a_stored_one_is_read(session):
    before = _selection(session)
    with pytest.raises(ObjectiveUnknownError) as refused:
        OBJECTIVE_MEMBERS['get_objective_info'](session, None)
    assert refused.value.reason == 'none_selected'
    assert refused.value.cause == RefusalCause.STATE
    assert _selection(session) == before


SLOT_MEMBERS = {
    'assign_turret_objective': lambda s, v: s.assign_turret_objective(v, '10x Oly'),
    'clear_turret_objective': lambda s, v: s.clear_turret_objective(v),
}


@pytest.mark.parametrize('slot', [0, 5, -1])
@pytest.mark.parametrize('member', SLOT_MEMBERS)
def test_a_slot_that_is_not_one_is_refused_as_the_turrets_range(turret_session, member, slot):
    before = _selection(turret_session)
    with pytest.raises(PositionOutOfRangeError) as refused:
        SLOT_MEMBERS[member](turret_session, slot)
    assert refused.value.reason == 'position_out_of_range'
    assert refused.value.bound == 'turret slots'
    assert (refused.value.low, refused.value.high) == (1, 4)
    assert _selection(turret_session) == before


def test_a_retired_plate_name_resolves_and_is_stored_under_its_key(session, z_changes):
    session.set_channel_acquire('BF', 'image')
    # The config carries Z only while Z holds its reference, and no test in
    # this file moves an axis; under the push suite Z has been seen without
    # one here, so a loss says what Z holds and who changed it.
    z = session.scope.motion.axis_positions()['Z']
    seen = '\n'.join(f'{at} on {thread}:\n{stack}' for at, thread, stack in z_changes[-5:])
    assert z.position_um is not None, f'Z is {z} after the home; its changes since:\n{seen}'
    config = session.get_sequenced_capture_config()
    config['labware_id'] = 'Center Dish'
    protocol = session.scope.protocols.create_protocol(input_config=config)
    assert protocol.labware_id() == 'Center Plate'


@pytest.mark.parametrize(
    ('route', 'body', 'reason', 'argument'),
    [
        (
            'labware_id',
            {'labware_id': 'Acme 1536 Ultra Plate'},
            'labware_unknown',
            'labware_id',
        ),
        (
            'objective_id',
            {'objective_id': '10xOly'},
            'objective_not_in_catalogue',
            'objective_id',
        ),
    ],
)
def test_a_wire_client_is_refused_as_the_request_naming_what_the_catalogue_offers(
    session, route, body, reason, argument
):
    with TestClient(build_app(session)) as client:
        answer = client.put(f'/api/v1/{route}', json=body)
    assert answer.status_code == 422
    problem = answer.json()
    assert problem['reason'] == reason
    assert problem['argument'] == argument
    assert problem['offered']


def test_a_wire_slot_out_of_range_is_the_requests(turret_session):
    with TestClient(build_app(turret_session)) as client:
        answer = client.put('/api/v1/turret_objective', json={'slot': 5, 'objective_id': '10x Oly'})
    assert answer.status_code == 422
    assert answer.json()['reason'] == 'position_out_of_range'


def test_a_reason_without_words_is_refused_at_the_raise():
    with pytest.raises(TypeError):
        CatalogueNameRefusedError('no_such_reason', argument='labware_id', value='x', offered=())


def test_none_is_refused_when_the_active_objective_is_already_unknown(tmp_path):
    """An unknown active objective is None, so a None asked for once matched it.

    The no-op compare ran before the owner, so ``select_objective(None)``
    returned False and ``confirm_objective(None)`` recorded the objective as
    confirmed with none named. The door now refuses the None first.
    """
    s = _session(tmp_path, 'LS850')
    try:
        s.settings['objective_id'] = None
        s.settings['objective_confirmed'] = None
        for call in (lambda: s.select_objective(None), lambda: s.confirm_objective(None)):
            with pytest.raises(ArgumentTypeRefusedError) as refused:
                call()
            assert refused.value.argument == 'objective_id'
        assert s.settings['objective_confirmed'] is None
    finally:
        s.shutdown()
