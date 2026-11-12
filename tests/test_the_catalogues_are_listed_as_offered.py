# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each catalogue is listed whole, in the order it is offered, as the records its lookup answers.

``scope.all_objectives`` and ``scope.all_labware`` answer every record of
the installation's objective and labware catalogues, in order: the
objectives as the picker offers them (each maker's together, rising by
magnification; ``test_the_objective_catalogue_lists_each_makers_glass_together``
pins that order), the plates as the catalogue file lists them. Each record
is the one its lookup, ``scope.objective`` or ``scope.plate``, answers for
the name it carries, so a caller that lists the catalogue needs no second
read. Over REST each list is one ``GET``, an array, so the order crosses
the wire with it.
"""

import json

import pytest
from fastapi.testclient import TestClient

from _lumascope.modules.path_utils import resolve_data_file
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.settings_fixtures import complete_settings


@pytest.fixture(scope='module')
def session():
    built = ScopeSession.create(complete_settings(), simulate=True)
    yield built
    built.shutdown()


def test_the_objectives_are_listed_in_the_order_offered_each_its_lookups_answer(session):
    scope = session.scope
    listed = scope.all_objectives
    assert [o.id for o in listed] == scope.objective_helper.get_objectives_list()
    assert all(scope.objective(o.id) is o for o in listed)


def test_the_plates_are_listed_in_the_catalogues_order_each_its_lookups_answer(session):
    scope = session.scope
    in_file = list(json.loads(resolve_data_file('labware.json').read_text('utf-8'))['Wellplate'])
    listed = scope.all_labware
    assert [p.name for p in listed] == in_file
    assert all(scope.plate(p.name) is p for p in listed)


def test_a_wire_client_lists_each_catalogue_in_one_request_in_order(session):
    with TestClient(build_app(session)) as client:
        objectives = client.get('/api/v1/scope/all_objectives')
        plates = client.get('/api/v1/scope/all_labware')
        one = client.get('/api/v1/scope/objective', params={'objective_id': '10x Oly'})
    assert objectives.status_code == plates.status_code == one.status_code == 200
    assert [o['id'] for o in objectives.json()] == [o.id for o in session.scope.all_objectives]
    assert [p['name'] for p in plates.json()] == [p.name for p in session.scope.all_labware]
    assert one.json() == next(o for o in objectives.json() if o['id'] == '10x Oly')
