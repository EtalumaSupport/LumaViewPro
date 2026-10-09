# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A wire client holds a live object by its handle, and reaches its members through it.

A protocol, a run, a move in flight and the session's protocol runner cross
the wire as ``{"handle": <id>, "type": <class name>}``. Their members are at
``/api/v1/handles/<type>/<id>/<member>``, and a client passes one as an
argument by its id. One object has one id. ``GET /api/v1/handles`` lists the
held ids, so a client whose answer was lost finds its handle; ``DELETE``
forgets an id and never the object, and the runner every client shares is
kept. A call whose answer can carry a new handle is refused while the
limit is held, before the member runs, so no answer loses its handle.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import rest.handles
from modules.api_surface import API, fields_of, mark_of
from modules.lumascope_api.motion import MoveInFlight
from modules.protocol import Protocol
from modules.protocol_runner import ProtocolRunner
from modules.scope_session import ScopeSession
from modules.sequenced_capture_runner import RunHandle
from rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def live(tmp_path):
    folder = tmp_path / 'live'
    folder.mkdir()
    return folder


@pytest.fixture
def session(live):
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    yield s
    s.shutdown()


@pytest.fixture
def client(session):
    # One event loop for every request, as the server runs.
    with TestClient(build_app(session)) as client:
        yield client


def _members(cls) -> set[tuple[str, str]]:
    """``(method, member)`` for each wire member *cls* marks, from the marks themselves."""
    found = {('get', n) for k in cls.__mro__ for n in fields_of(k)}
    for klass in cls.__mro__:
        for name, value in vars(klass).items():
            if mark_of(value) == API:
                found.add(('get' if isinstance(value, property) else 'post', name))
    return found


def test_the_live_objects_a_client_can_be_handed_each_have_their_members(client):
    paths = client.get('/api/v1/openapi.json').json()['paths']
    served = {
        (method, path.removeprefix('/api/v1/handles/'))
        for path, ops in paths.items()
        for method in ops
        if path.startswith('/api/v1/handles/')
    }

    types = {path.split('/')[0] for _, path in served}
    assert types == {'MoveInFlight', 'Protocol', 'ProtocolRunner', 'RunHandle'}
    for cls in (MoveInFlight, Protocol, ProtocolRunner, RunHandle):
        assert {
            (method, path.split('/', 2)[2])
            for method, path in served
            if path.startswith(f'{cls.__name__}/{{handle_id}}/')
        } == _members(cls), cls.__name__
        assert ('delete', f'{cls.__name__}/{{handle_id}}') in served


def test_one_object_has_one_id(client):
    first = client.post('/api/v1/create_protocol_runner').json()
    second = client.post('/api/v1/create_protocol_runner').json()

    assert first == second == {'handle': first['handle'], 'type': 'ProtocolRunner'}
    assert client.get('/api/v1/handles').json() == [first]


def test_a_handles_members_are_reached_through_it(client):
    protocol = client.post('/api/v1/create_empty_protocol').json()

    steps = client.post(f'/api/v1/handles/Protocol/{protocol["handle"]}/num_steps')

    assert steps.json() == 0


def test_a_handle_is_passed_as_an_argument_by_its_id(client, live):
    protocol = client.post('/api/v1/create_empty_protocol').json()

    saved = client.post(
        '/api/v1/save_protocol', json={'protocol': protocol['handle'], 'file_path': 'plate'}
    )

    assert saved.json()['name'] == 'plate.tsv'
    assert (live / 'plate.tsv').is_file()


def test_a_move_in_flight_is_waited_on_through_its_handle(client, session):
    home_sim_scope(session.scope)
    move = client.post(
        '/api/v1/scope/motion/start_move_absolute', json={'axis': 'Z', 'position': 100.0}
    ).json()

    assert move['type'] == 'MoveInFlight'
    assert client.post(f'/api/v1/handles/MoveInFlight/{move["handle"]}/wait').status_code == 200
    assert session.scope.motion.axis_positions()['Z'].position == pytest.approx(100.0, abs=1)


def test_an_id_not_held_for_its_type_is_404(client):
    protocol = client.post('/api/v1/create_empty_protocol').json()

    assert client.post('/api/v1/handles/Protocol/999/num_steps').status_code == 404
    wrong_type = client.post(
        f'/api/v1/handles/RunHandle/{protocol["handle"]}/wait', json={'timeout_s': 1}
    )
    assert wrong_type.status_code == 404
    as_argument = client.post('/api/v1/save_protocol', json={'protocol': '999', 'file_path': 'x'})
    assert as_argument.status_code == 404


def test_forgetting_an_id_lets_go_of_the_id_and_the_shared_runner_is_kept(client):
    protocol = client.post('/api/v1/create_empty_protocol').json()
    runner = client.post('/api/v1/create_protocol_runner').json()

    assert client.delete(f'/api/v1/handles/Protocol/{protocol["handle"]}').status_code == 204
    assert client.delete(f'/api/v1/handles/Protocol/{protocol["handle"]}').status_code == 404
    assert client.post(f'/api/v1/handles/Protocol/{protocol["handle"]}/num_steps').status_code == (
        404
    )
    assert client.delete(f'/api/v1/handles/ProtocolRunner/{runner["handle"]}').status_code == 409
    assert client.get('/api/v1/handles').json() == [runner]


def test_a_call_that_would_hand_out_past_the_limit_is_refused_before_it_runs(client, monkeypatch):
    monkeypatch.setattr(rest.handles, 'LIMIT', 1)
    runner = client.post('/api/v1/create_protocol_runner').json()

    refused = client.post('/api/v1/create_empty_protocol')

    assert refused.status_code == 503
    assert refused.headers['Retry-After'] == str(rest.handles.RETRY_AFTER_S)
    assert client.get('/api/v1/handles').json() == [runner]
    # A call that hands nothing out is not refused.
    assert client.get('/api/v1/status').status_code == 200


def test_a_parameter_no_client_can_fill_is_not_on_the_wire(client):
    paths = client.get('/api/v1/openapi.json').json()
    operation = paths['paths']['/api/v1/handles/ProtocolRunner/{handle_id}/run_autofocus']['post']
    schema_ref = operation['requestBody']['content']['application/json']['schema']
    name = (schema_ref.get('$ref') or schema_ref['anyOf'][0]['$ref']).rsplit('/', 1)[1]

    assert 'claim' not in paths['components']['schemas'][name]['properties']
