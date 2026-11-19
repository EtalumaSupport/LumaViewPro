# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each catalogue is listed whole, in the order it is offered, as the records its lookup answers.

``scope.all_objectives``, ``scope.all_labware`` and ``scope.all_scope_models``
answer every record of the installation's objective, labware and scope
model catalogues, in order: the objectives as the picker offers them (each
maker's together, rising by magnification;
``test_the_objective_catalogue_lists_each_makers_glass_together`` pins that
order), the plates and the models as their catalogue files list them. Each
record is the one its lookup, ``scope.objective``, ``scope.plate`` or
``scope.scope_model``, answers for the name it carries, so a caller that
lists the catalogue needs no second read. Over REST each list is one
``GET``, an array, so the order crosses the wire with it; the model
catalogue is no longer served whole as an object keyed by name.
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


def test_the_models_are_listed_in_the_catalogues_order_each_its_lookups_answer(session):
    scope = session.scope
    in_file = list(json.loads(resolve_data_file('scopes.json').read_text('utf-8'))['Models'])
    listed = scope.all_scope_models
    assert [m.id for m in listed] == in_file
    assert all(scope.scope_model(m.id) is m for m in listed)


def test_a_wire_client_lists_each_catalogue_in_one_request_in_order(session):
    with TestClient(build_app(session)) as client:
        objectives = client.get('/api/v1/scope/all_objectives')
        plates = client.get('/api/v1/scope/all_labware')
        models = client.get('/api/v1/scope/all_scope_models')
        one = client.get('/api/v1/scope/objective', params={'objective_id': '10x Oly'})
        whole_mapping = client.get('/api/v1/scope/scope_models')
    assert objectives.status_code == plates.status_code == models.status_code == 200
    assert one.status_code == 200
    assert [o['id'] for o in objectives.json()] == [o.id for o in session.scope.all_objectives]
    assert [p['name'] for p in plates.json()] == [p.name for p in session.scope.all_labware]
    assert [m['id'] for m in models.json()] == [m.id for m in session.scope.all_scope_models]
    assert one.json() == next(o for o in objectives.json() if o['id'] == '10x Oly')
    assert whole_mapping.status_code == 404
