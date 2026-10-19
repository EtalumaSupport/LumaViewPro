# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A wire client reaches every member marked for the wire, at the route its Python name gives.

The REST server serves exactly the marked set: the Session's member ``m``
at ``/api/v1/m``, a sub-object's member below the sub-object's name. A read
-- a property, or a method marked ``reads`` -- is ``GET`` and answers its
encoded value, a read method's arguments in the query string; any other
method is ``POST`` and takes its arguments as a JSON object by parameter
name. Either is checked before the member is reached, so arguments that do
not fit are 422 and never a half-made call. A path argument is a
live-folder name, reaching the member resolved, and a name outside the
live folder is refused before the member runs.
"""

from __future__ import annotations

import enum

import pytest
from fastapi.testclient import TestClient

from modules.api_surface import (
    API,
    IN_PROCESS,
    fields_of,
    is_engineering,
    is_record,
    mark_of,
    reads_of,
)
from modules.coord_transformations import StageOffset
from modules.lumascope_api.illumination import LedReading, LedState
from modules.scope_session import ScopeSession
from rest.app import SERVER_SEGMENTS, build_app
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


def _wire_routes(session) -> set[tuple[str, str]]:
    """``(method, path)`` for every wire member under the live session.

    Walked from the marks and the live objects, not from the annotations the
    server reads: a read that holds a live object with members of its own is
    a sub-object, every other read a ``GET`` (an enum is a value, wherever it
    is declared), every method marked ``reads`` a ``GET``, and every other
    marked method a ``POST``.
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
                    found.add(('get' if reads_of(value) else 'post', f'{prefix}/{name}'))
        for name in reads:
            value = getattr(obj, name)
            if (
                type(value).__module__.startswith('modules')
                and not is_record(type(value))
                and not isinstance(value, enum.Enum)
            ):
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
        if path.removeprefix('/api/v1/').split('/')[0] not in SERVER_SEGMENTS
    }

    assert served - {('get', '/api')} == _wire_routes(session)
    # The sub-objects are reached, not only the Session's own members.
    assert ('post', '/api/v1/scope/motion/move_absolute') in served
    assert ('get', '/api/v1/manual_recording/is_recording') in served


def test_every_engineering_member_is_in_process_and_served_nowhere(client, session):
    """A bench tool is the engineering plugin's, called in-process; a wire
    client reaches none (C6).

    The test above holds the routes equal to the members marked for the wire,
    so a bench tool marked onto the wire would pass there; this one walks
    every object a session reaches and names each engineering member that is
    not in-process or is served.
    """
    engineering = {}
    seen = set()

    def walk(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        cls = type(obj)
        reads = [n for k in cls.__mro__ for n in fields_of(k)]
        for klass in cls.__mro__:
            for name, value in vars(klass).items():
                if mark_of(value) is None:
                    continue
                if is_engineering(value):
                    engineering[f'{cls.__name__}.{name}'] = mark_of(value)
                if isinstance(value, property) and mark_of(value) == API:
                    reads.append(name)
        for name in reads:
            child = getattr(obj, name)
            if type(child).__module__.startswith('modules') and not is_record(type(child)):
                walk(child)

    walk(session)
    # The camera's four (W3b) and diagnostics' ten: a walk that reached
    # neither object would pass the two checks below on nothing.
    assert len(engineering) >= 14
    assert {n: m for n, m in engineering.items() if m != IN_PROCESS} == {}

    paths = client.get('/api/v1/openapi.json').json()['paths']
    served = {path.rsplit('/', 1)[-1] for path in paths}
    assert served & {n.split('.')[-1] for n in engineering} == set()


def test_a_wire_client_reads_settings_by_path_or_whole_not_by_document(client, session):
    """The Session's slices of the settings store are not served: a wire
    client reads one setting by its path, or every setting at once from the
    snapshot (census C's row B9: retire the documents, keep the one
    whole-store read)."""
    documents = {
        'get_layer_configs',
        'get_stim_configs',
        'get_enabled_stim_configs',
        'get_auto_gain_settings',
        'capture_settings_snapshot',
        'get_zstack_params',
    }
    paths = client.get('/api/v1/openapi.json').json()['paths']
    assert {path.rsplit('/', 1)[-1] for path in paths} & documents == set()

    whole = client.get('/api/v1/get_settings_snapshot')
    assert whole.status_code == 200, whole.text
    assert whole.json()['protocol']['autogain'] == session.get_setting('protocol.autogain')
    one = client.get('/api/v1/get_setting', params={'path': 'BF.exposure_ms'})
    assert one.status_code == 200, one.text
    assert one.json() == whole.json()['BF']['exposure_ms']


def test_the_versions_are_answered_unversioned(client):
    assert client.get('/api').json() == {'versions': ['v1']}


def test_a_read_answers_its_encoded_value(client, session):
    assert client.get('/api/v1/app_version').json() == session.app_version
    assert client.get('/api/v1/app_runtime').json() == 'source'
    assert client.get('/api/v1/engineering_mode').json() is False
    assert set(client.get('/api/v1/status').json()) == {
        'live_work',
        'axes',
        'parts',
        'camera_streaming',
    }


def test_a_read_method_is_a_get_its_arguments_converted_from_the_query(client, session):
    """A read method is asked with GET (B2): each argument a query key, its
    text converted to the parameter's type, a list its key repeated, a
    record one key a field; a key it does not take, a value that will not
    convert or half a record is 422 naming the key, and POST is 405."""
    limits = client.get('/api/v1/scope/motion/get_axis_limits', params={'axis': 'Z'})
    assert limits.json() == session.scope.motion.get_axis_limits('Z').to_dict()
    assert (
        client.post('/api/v1/scope/motion/get_axis_limits', json={'axis': 'Z'}).status_code == 405
    )

    frame = client.get('/api/v1/scope/camera/frame_at_binning', params={'size': '2'})
    assert frame.status_code == 200, frame.text
    sources = client.get(
        '/api/v1/scope/imaging/frames_until_valid', params=[('exclude_sources', 'a')] * 2
    )
    assert sources.status_code == 200, sources.text

    offset = {'x_mm': 10, 'y_mm': 20, 'stage_offset.x_um': 3, 'stage_offset.y_um': 4}
    stage = client.get('/api/v1/scope/runtime_state/plate_to_stage', params=offset)
    expected = session.scope.runtime_state.plate_to_stage(
        10.0, 20.0, stage_offset=StageOffset(x_um=3.0, y_um=4.0)
    )
    assert stage.json() == expected.to_dict()

    for params, key in (
        ({'axis': 'Z', 'bogus': '1'}, 'query/bogus'),
        ({'size': 'two'}, 'query/size'),
        ({**offset, 'stage_offset.y_um': None}, 'query/stage_offset.y_um'),
    ):
        path = (
            '/api/v1/scope/runtime_state/plate_to_stage'
            if 'x_mm' in params
            else '/api/v1/scope/camera/frame_at_binning'
            if 'size' in params
            else '/api/v1/scope/motion/get_axis_limits'
        )
        refused = client.get(path, params={k: v for k, v in params.items() if v is not None})
        assert refused.status_code == 422, refused.text
        assert key in refused.json()['detail'], refused.text


def test_a_sub_objects_method_is_called_with_its_arguments_by_name(client, session):
    on = client.post(
        '/api/v1/scope/illumination/led_on',
        json={'channel': 'Red', 'illumination_ma': 10.0},
    )
    assert on.status_code == 200
    assert on.json() == 10.0
    assert session.scope.illumination.get_led_state('Red') == LedReading(
        2, 'Red', LedState.LIT, 10.0
    )

    state = client.get('/api/v1/scope/illumination/get_led_state', params={'channel': 'Red'})
    assert state.json() == {'address': 2, 'name': 'Red', 'state': 'lit', 'illumination_ma': 10.0}


def test_a_snapshot_read_over_the_wire_is_taken_back_by_restore(client, session):
    # The snapshot holds its readings as records inside an array; the
    # wire answers it as JSON and restore_led_state decodes it back.
    session.scope.illumination.led_on('Red', 10.0)
    snapshot = client.get('/api/v1/scope/illumination/save_led_state', params={'tag': 'before'})
    assert snapshot.status_code == 200
    assert snapshot.json()['tag'] == 'before'
    assert {'address': 2, 'name': 'Red', 'state': 'lit', 'illumination_ma': 10.0} in (
        snapshot.json()['readings']
    )
    session.scope.illumination.leds_off()

    restored = client.post(
        '/api/v1/scope/illumination/restore_led_state', json={'snapshot': snapshot.json()}
    )

    assert restored.status_code == 200
    assert session.scope.illumination.get_led_state('Red') == LedReading(
        2, 'Red', LedState.LIT, 10.0
    )


@pytest.mark.parametrize('body', [None, {}])
def test_a_method_with_no_arguments_takes_an_empty_body_or_an_empty_object(client, session, body):
    session.scope.illumination.led_on('Red', 10.0)

    answer = client.post('/api/v1/scope/illumination/leds_off', json=body)

    assert answer.status_code == 200
    assert session.scope.illumination.get_led_state('Red').state is LedState.DARK


@pytest.mark.parametrize(
    'body',
    [
        {},
        {'channel': 'Red'},
        {'channel': 'Red', 'illumination_ma': '10'},
        {'channel': 'Red', 'illumination_ma': 10.0, 'brightness': 3},
    ],
    ids=[
        'empty',
        'missing',
        'a string for a number',
        'an unknown key',
    ],
)
def test_a_body_that_does_not_fit_is_422_and_reaches_nothing(client, session, body):
    answer = client.post('/api/v1/scope/illumination/led_on', json=body)

    assert answer.status_code == 422
    assert session.scope.illumination.get_led_state('Red').state is LedState.DARK


def test_a_number_for_a_bool_is_422_and_reaches_nothing(client, session):
    motion = session.scope.motion
    before = motion.get_current_position('Z')

    answer = client.post(
        '/api/v1/scope/motion/move_absolute',
        json={'axis': 'Z', 'position': before + 50.0, 'overshoot_enabled': 1},
    )

    assert answer.status_code == 422
    assert motion.get_current_position('Z') == before


@pytest.mark.parametrize('number', ['NaN', 'Infinity', '1e400'])
def test_a_number_that_is_not_finite_is_422_and_reaches_nothing(client, session, number):
    # Python's JSON reader takes these tokens, though JSON has none of them.
    answer = client.post(
        '/api/v1/scope/illumination/led_on',
        content=f'{{"channel": "Red", "illumination_ma": {number}}}',
        headers={'content-type': 'application/json'},
    )

    assert answer.status_code == 422
    assert session.scope.illumination.get_led_state('Red').state is LedState.DARK


def test_a_member_route_takes_no_query_string(client):
    assert client.get('/api/v1/status?verbose=1').status_code == 422


def test_a_path_argument_is_a_live_folder_name_and_reaches_the_member_resolved(client, live):
    answer = client.post('/api/v1/make_logs_zip', json={'output_dir': 'reports'})

    assert answer.status_code == 200
    path = answer.json()['path']
    assert path['name'].startswith('reports/')
    assert (live / path['name']).is_file()


def test_a_path_outside_the_live_folder_is_refused_before_the_member_runs(client, tmp_path):
    refused = client.post('/api/v1/make_logs_zip', json={'output_dir': '../outside'})

    assert refused.status_code == 422
    assert refused.json()['reason'] == 'outside_live_folder'
    assert not (tmp_path / 'outside').exists()


def test_a_call_that_ends_its_thread_is_answered_as_a_fault(session, monkeypatch):
    def leaves(*_args, **_kwargs):
        raise SystemExit(3)

    monkeypatch.setattr(session.scope.illumination, 'leds_off', leaves)
    with TestClient(build_app(session)) as client:
        answer = client.post('/api/v1/scope/illumination/leds_off')

    assert answer.status_code == 500
    assert answer.headers['content-type'] == 'application/problem+json'
    assert (answer.json()['kind'], answer.json()['title']) == ('fault', 'RuntimeError')
