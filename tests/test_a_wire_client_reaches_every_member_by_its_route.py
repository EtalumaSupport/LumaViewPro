# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A wire client reaches every member marked for the wire, at the route its Python name gives.

The REST server serves exactly the marked set: the Session's member ``m``
at ``/api/v1/m``, a sub-object's member below the sub-object's name. A read
is ``GET`` and answers its encoded value; a method is ``POST`` and takes its
arguments as a JSON object by parameter name, checked before the member is
reached, so a body that does not fit is 422 and never a half-made call. A
path argument is a live-folder name, reaching the member resolved, and a
name outside the live folder is refused before the member runs.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modules.api_surface import API, fields_of, is_record, mark_of
from modules.exceptions import LiveFolderPathRefusedError
from modules.scope_session import ScopeSession
from rest.app import build_app
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
    return TestClient(build_app(session))


def _wire_routes(session) -> set[tuple[str, str]]:
    """``(method, path)`` for every wire member under the live session.

    Walked from the marks and the live objects, not from the annotations the
    server reads: a read that holds a live object with members of its own is
    a sub-object, every other read a ``GET``, every marked method a ``POST``.
    """
    found = set()

    def walk(obj, prefix):
        cls = type(obj)
        reads = [n for k in cls.__mro__ for n in fields_of(k)]
        for klass in cls.__mro__:
            for name, value in vars(klass).items():
                if mark_of(value) != API:
                    continue
                if isinstance(value, property):
                    reads.append(name)
                else:
                    found.add(('post', f'{prefix}/{name}'))
        for name in reads:
            value = getattr(obj, name)
            if type(value).__module__.startswith('modules') and not is_record(type(value)):
                walk(value, f'{prefix}/{name}')
            else:
                found.add(('get', f'{prefix}/{name}'))

    walk(session, '/api/v1')
    return found


def test_the_routes_are_exactly_the_wire_members(client, session):
    paths = client.get('/api/v1/openapi.json').json()['paths']
    served = {
        (method, path)
        for path, ops in paths.items()
        for method in ops
        if not path.startswith('/api/v1/handles')
    }

    assert served - {('get', '/api')} == _wire_routes(session)
    # The sub-objects are reached, not only the Session's own members.
    assert ('post', '/api/v1/scope/motion/move_absolute') in served
    assert ('get', '/api/v1/manual_recording/is_recording') in served


def test_the_versions_are_answered_unversioned(client):
    assert client.get('/api').json() == {'versions': ['v1']}


def test_a_read_answers_its_encoded_value(client, session):
    assert client.get('/api/v1/app_version').json() == session.app_version
    assert client.get('/api/v1/engineering_mode').json() is False
    assert set(client.get('/api/v1/status').json()) == {
        'live_work',
        'axes',
        'parts',
        'camera_streaming',
    }


def test_a_sub_objects_method_is_called_with_its_arguments_by_name(client, session):
    on = client.post(
        '/api/v1/scope/illumination/led_on',
        json={'channel': 'Red', 'illumination_ma': 10.0, 'block': True},
    )
    assert on.status_code == 200
    assert on.json() == 10.0
    assert session.scope.illumination.get_led_state('Red') == {
        'enabled': True,
        'illumination_ma': 10.0,
    }

    state = client.post('/api/v1/scope/illumination/get_led_state', json={'channel': 'Red'})
    assert state.json() == {'enabled': True, 'illumination_ma': 10.0}


@pytest.mark.parametrize('body', [None, {}])
def test_a_method_with_no_arguments_takes_an_empty_body_or_an_empty_object(client, session, body):
    session.scope.illumination.led_on('Red', 10.0, block=True)

    answer = client.post('/api/v1/scope/illumination/leds_off', json=body)

    assert answer.status_code == 200
    assert session.scope.illumination.get_led_state('Red')['enabled'] is False


@pytest.mark.parametrize(
    'body',
    [
        {},
        {'channel': 'Red'},
        {'channel': 'Red', 'illumination_ma': '10'},
        {'channel': 'Red', 'illumination_ma': 10.0, 'block': 1},
        {'channel': 'Red', 'illumination_ma': 10.0, 'brightness': 3},
    ],
    ids=[
        'empty',
        'missing',
        'a string for a number',
        'a number for a bool',
        'an unknown key',
    ],
)
def test_a_body_that_does_not_fit_is_422_and_reaches_nothing(client, session, body):
    answer = client.post('/api/v1/scope/illumination/led_on', json=body)

    assert answer.status_code == 422
    assert session.scope.illumination.get_led_state('Red')['enabled'] is False


@pytest.mark.parametrize('number', ['NaN', 'Infinity', '1e400'])
def test_a_number_that_is_not_finite_is_422_and_reaches_nothing(client, session, number):
    # Python's JSON reader takes these tokens, though JSON has none of them.
    answer = client.post(
        '/api/v1/scope/illumination/led_on',
        content=f'{{"channel": "Red", "illumination_ma": {number}}}',
        headers={'content-type': 'application/json'},
    )

    assert answer.status_code == 422
    assert session.scope.illumination.get_led_state('Red')['enabled'] is False


def test_a_member_route_takes_no_query_string(client):
    assert client.get('/api/v1/status?verbose=1').status_code == 422


def test_a_path_argument_is_a_live_folder_name_and_reaches_the_member_resolved(client, live):
    answer = client.post('/api/v1/make_logs_zip', json={'output_dir': 'reports'})

    assert answer.status_code == 200
    path = answer.json()['path']
    assert path['name'].startswith('reports/')
    assert (live / path['name']).is_file()


def test_a_path_outside_the_live_folder_is_refused_before_the_member_runs(client, tmp_path):
    with pytest.raises(LiveFolderPathRefusedError):
        client.post('/api/v1/make_logs_zip', json={'output_dir': '../outside'})

    assert not (tmp_path / 'outside').exists()


def test_a_call_that_ends_its_thread_is_answered_as_a_fault(session, monkeypatch):
    def leaves(*_args, **_kwargs):
        raise SystemExit(3)

    monkeypatch.setattr(session.scope.illumination, 'leds_off', leaves)
    client = TestClient(build_app(session), raise_server_exceptions=False)

    assert client.post('/api/v1/scope/illumination/leds_off').status_code == 500
