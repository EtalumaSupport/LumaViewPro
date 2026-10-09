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
(``ObjectiveUnknownError('none_selected')``) at every door, and a turret slot
is refused by one check wherever it is taken.
"""

import pytest
from fastapi.testclient import TestClient

from modules.exceptions import (
    CatalogueNameRefusedError,
    ConfigError,
    ObjectiveUnknownError,
    PositionOutOfRangeError,
    RefusalCause,
)
from rest.app import build_app
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    s = _session(tmp_path_factory.mktemp('c6'), 'LS850')
    yield s
    s.shutdown()


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
    'runtime_state.get_objective_info': lambda s, v: s.scope.runtime_state.get_objective_info(v),
    'select_objective': lambda s, v: s.select_objective(v),
    'confirm_objective': lambda s, v: s.confirm_objective(v),
    'confirm_objective at a slot': lambda s, v: s.confirm_objective(v, turret_position=2),
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
    assert refused.value.argument == 'plate_key'
    assert refused.value.value == name
    assert refused.value.offered == tuple(session.wellplate_loader.get_plate_list())
    assert _selection(session) == before


@pytest.mark.parametrize('name', ['zzz-no-such-objective', '10x', '10xOly'])
@pytest.mark.parametrize('member', OBJECTIVE_MEMBERS)
def test_an_objective_the_catalogue_lacks_is_refused_offering_its_keys(session, member, name):
    before = _selection(session)
    with pytest.raises(CatalogueNameRefusedError) as refused:
        OBJECTIVE_MEMBERS[member](session, name)
    assert refused.value.reason == 'objective_not_in_catalogue'
    assert refused.value.cause == RefusalCause.REQUEST
    assert isinstance(refused.value, ConfigError)
    assert refused.value.argument == 'objective_id'
    assert refused.value.offered == tuple(session.objective_helper.get_objectives_list())
    assert _selection(session) == before


@pytest.mark.parametrize('member', OBJECTIVE_MEMBERS)
def test_no_objective_is_none_selected_at_every_door(session, member):
    before = _selection(session)
    with pytest.raises(ObjectiveUnknownError) as refused:
        OBJECTIVE_MEMBERS[member](session, None)
    assert refused.value.reason == 'none_selected'
    assert refused.value.cause == RefusalCause.STATE
    assert _selection(session) == before


SLOT_MEMBERS = {
    'assign_turret_objective': lambda s, v: s.assign_turret_objective(v, '10x Oly'),
    'clear_turret_objective': lambda s, v: s.clear_turret_objective(v),
}


@pytest.mark.parametrize('slot', [0, 5, -1, True, 1.5, '2', None])
@pytest.mark.parametrize('member', SLOT_MEMBERS)
def test_a_slot_that_is_not_one_is_refused_as_the_turrets_range(session, member, slot):
    before = _selection(session)
    with pytest.raises(PositionOutOfRangeError) as refused:
        SLOT_MEMBERS[member](session, slot)
    assert refused.value.reason == 'position_out_of_range'
    assert refused.value.bound == 'turret slots'
    assert (refused.value.low, refused.value.high) == (1, 4)
    assert _selection(session) == before


def test_a_retired_plate_name_resolves_and_is_stored_under_its_key(session):
    session.set_layer_acquire('BF', 'image')
    config = session.get_sequenced_capture_config()
    config['labware_id'] = 'Center Dish'
    protocol = session.scope.protocols.create_protocol(input_config=config)
    assert protocol.labware() == 'Center Plate'


@pytest.mark.parametrize(
    ('route', 'body', 'reason', 'argument'),
    [
        (
            'select_labware',
            {'labware_name': 'Acme 1536 Ultra Plate'},
            'labware_unknown',
            'plate_key',
        ),
        (
            'select_objective',
            {'objective_id': '10xOly'},
            'objective_not_in_catalogue',
            'objective_id',
        ),
        (
            'scope/runtime_state/get_objective_info',
            {'objective_id': 'zzz'},
            'objective_not_in_catalogue',
            'objective_id',
        ),
    ],
)
def test_a_wire_client_is_refused_as_the_request_naming_what_the_catalogue_offers(
    session, route, body, reason, argument
):
    with TestClient(build_app(session)) as client:
        answer = client.post(f'/api/v1/{route}', json=body)
    assert answer.status_code == 422
    problem = answer.json()
    assert problem['reason'] == reason
    assert problem['argument'] == argument
    assert problem['offered']


def test_a_wire_slot_out_of_range_is_the_requests(session):
    with TestClient(build_app(session)) as client:
        answer = client.post(
            '/api/v1/assign_turret_objective', json={'position': 5, 'objective_id': '10x Oly'}
        )
    assert answer.status_code == 422
    assert answer.json()['reason'] == 'position_out_of_range'


def test_a_reason_without_words_is_refused_at_the_raise():
    with pytest.raises(TypeError):
        CatalogueNameRefusedError('no_such_reason', argument='plate_key', value='x', offered=())


def test_none_is_refused_when_the_active_objective_is_already_unknown(tmp_path):
    """An unknown active objective is None, so a None asked for once matched it.

    The no-op compare ran before the owner, so ``select_objective(None)``
    returned False and ``confirm_objective(None)`` recorded the objective as
    confirmed with none named.
    """
    s = _session(tmp_path, 'LS850')
    try:
        s.settings['objective_id'] = None
        s.settings['objective_confirmed'] = False
        for call in (lambda: s.select_objective(None), lambda: s.confirm_objective(None)):
            with pytest.raises(ObjectiveUnknownError) as refused:
                call()
            assert refused.value.reason == 'none_selected'
        assert s.settings['objective_confirmed'] is False
    finally:
        s.shutdown()


@pytest.mark.parametrize(
    ('name', 'refusal', 'reason'),
    [
        (None, ObjectiveUnknownError, 'none_selected'),
        ('zzz', CatalogueNameRefusedError, 'objective_not_in_catalogue'),
    ],
)
def test_the_slot_lookup_asks_the_catalogue_first(session, name, refusal, reason):
    # None would otherwise match a slot with no objective assigned.
    with pytest.raises(refusal) as refused:
        session.scope.motion.get_turret_position_for_objective_id(name)
    assert refused.value.reason == reason
