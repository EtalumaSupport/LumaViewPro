# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the REST server sends matches what its OpenAPI description declares.

The guard (``tests/guards/test_every_answer_is_described``) holds every
route and event to a declared schema; this holds the bodies to it. Over a
simulated session, every read and every call that needs no argument and
asks rather than acts (``get_``, ``is_``, ``list``...), and every read of
a protocol's handle, is answered and checked against its route's 200 with
``jsonschema``, formats checked, so a time sent without its offset fails a
declared ``date-time``. The events a home and a run send are checked
against their components. The server's own answers -- a job while it runs
and once it has ended, the handles held, the versions -- and a member's
handle and path answers are checked against their models on a real
server, since those models are written beside the code that makes each
body rather than read from it.
"""

from __future__ import annotations

import re
import threading

import jsonschema
import pytest
from fastapi.testclient import TestClient

from modules import wire_encoding
from modules.scope_session import ScopeSession
from rest import routes as rest_routes
from rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_the_event_stream_carries_what_the_scope_tells import _reading, _serving

# A call that only asks, by its name: safe to make with no argument.
_ASKS = re.compile(
    r'(get_|status|bring_up_record|capabilities|list|live_folder_listing|frame_size|'
    r'min_frame|read_|scale_bar|scope_models|settings_template|resolve_current|'
    r'plugin_health|is_|has_)'
)


@pytest.fixture
def session(tmp_path):
    live = tmp_path / 'live'
    live.mkdir()
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    yield s
    s.shutdown()


def _checker(described: dict, schema: dict) -> jsonschema.Draft202012Validator:
    """A validator of *schema*, its references read from the description's components."""
    rooted = {**schema, 'components': described['components']}
    return jsonschema.Draft202012Validator(
        rooted, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    )


def _answer(described: dict, path: str, method: str) -> dict:
    return described['paths'][path][method]['responses']['200']['content']['application/json'][
        'schema'
    ]


def test_a_time_without_its_offset_fails_a_declared_date_time():
    checker = jsonschema.Draft202012Validator(
        {'type': 'string', 'format': 'date-time'},
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )

    assert checker.is_valid('2026-10-09T12:00:00+02:00')
    assert not checker.is_valid('2026-10-09T12:00:00')


@pytest.mark.slow
def test_every_answer_a_simulated_session_gives_matches_its_route(session):
    app = build_app(session)
    described = app.openapi()
    handed_out = wire_encoding.handed_out(
        ScopeSession, wire_encoding.project_classes(), wire_encoding.project_aliases()
    )
    checked, mismatched = 0, []

    def check(body: object, schema: dict, where: str) -> None:
        nonlocal checked
        if schema == {}:
            # Anything matches a schema that says nothing.
            mismatched.append(f'{where}: declares nothing')
        errors = list(_checker(described, schema).iter_errors(body))
        if errors:
            mismatched.append(f'{where}: {errors[0].message[:200]}')
        checked += 1

    with TestClient(app) as client:
        for route in rest_routes.routes(ScopeSession, handed_out=handed_out):
            member = route.member
            asks = not any(p.required for p in member.parameters) and _ASKS.match(member.name)
            if not (member.read or asks):
                continue
            path = f'/api/v1/{route.path}'
            if member.read:
                response = client.get(path)
            else:
                response = client.post(path, json={})
            if response.status_code != 200:
                continue
            method = 'get' if member.read else 'post'
            check(response.json(), _answer(described, path, method), path)

        from modules.protocol import Protocol

        protocol = client.post('/api/v1/create_empty_protocol', json={}).json()['handle']
        for route in rest_routes.routes(Protocol, handed_out=handed_out):
            if not route.member.read:
                continue
            response = client.get(f'/api/v1/handles/Protocol/{protocol}/{route.path}')
            assert response.status_code == 200, route.path
            template = f'/api/v1/handles/Protocol/{{handle_id}}/{route.path}'
            check(response.json(), _answer(described, template, 'get'), template)

    assert not mismatched, '\n'.join(mismatched)
    # 97 answers at this writing.
    assert checked >= 90, checked


@pytest.mark.slow
def test_every_event_of_a_home_and_a_run_matches_its_component(session):
    from rest import events as event_stream

    app = build_app(session)
    described = app.openapi()
    components = {
        name: f'#/components/schemas/{model.__name__}'
        for name, model in event_stream.published(
            session,
            wire_encoding.handed_out(
                ScopeSession, wire_encoding.project_classes(), wire_encoding.project_aliases()
            ),
        ).items()
    }
    session.settings['BF']['acquire'] = 'image'
    with _serving(session, app) as client, _reading(client) as stream:
        seen = stream.until('status')
        assert (
            client.post('/api/v1/scope/motion/home', headers={'Prefer': 'wait=60'}).status_code
            == 200
        )
        home_sim_scope(session.scope)
        protocol = client.post('/api/v1/create_empty_protocol').json()['handle']
        assert client.post('/api/v1/add_step', json={'protocol': protocol}).status_code == 200
        runner = client.post('/api/v1/create_protocol_runner').json()['handle']
        assert (
            client.post(
                f'/api/v1/handles/ProtocolRunner/{runner}/run_single_scan',
                json={'protocol': protocol, 'enable_image_saving': False},
                headers={'Prefer': 'wait=60'},
            ).status_code
            == 200
        )
        seen += stream.until('run_ended')

    mismatched = []
    for event in seen:
        if 'comment' in event:
            continue
        checker = _checker(described, {'$ref': components[event['event']]})
        errors = list(checker.iter_errors(event['data']))
        if errors:
            mismatched.append(f'{event["event"]}: {errors[0].message[:200]}')
    names = {e['event'] for e in seen if 'event' in e}
    assert {'status', 'position', 'scan_started', 'step_started', 'run_ended'} <= names, names
    assert not mismatched, '\n'.join(mismatched)


def _component(described: dict, name: str) -> jsonschema.Draft202012Validator:
    return _checker(described, {'$ref': f'#/components/schemas/{name}'})


def _matches(checker: jsonschema.Draft202012Validator, body: object) -> None:
    errors = [e.message for e in checker.iter_errors(body)]
    assert not errors, (body, errors)


@pytest.mark.slow
def test_the_servers_own_answers_and_a_handle_and_a_path_match_their_models(session, monkeypatch):
    app = build_app(session)
    described = app.openapi()
    job = _component(described, 'Job')
    release = threading.Event()
    real_leds_off = session.scope.illumination.leds_off
    real_zip = session.make_logs_zip

    def leds_off():
        assert release.wait(10)
        return real_leds_off()

    def zips_once_released(*args, **kwargs):
        assert release.wait(10)
        return real_zip(*args, **kwargs)

    monkeypatch.setattr(session.scope.illumination, 'leds_off', leds_off)
    monkeypatch.setattr(session, 'make_logs_zip', zips_once_released)
    with _serving(session, app) as client:
        _matches(_checker(described, _answer(described, '/api', 'get')), client.get('/api').json())

        accepted = client.post('/api/v1/scope/illumination/leds_off', headers={'Prefer': 'wait=0'})
        assert accepted.status_code == 202
        _matches(job, accepted.json())
        running = client.get(accepted.headers['Location'], headers={'Prefer': 'wait=0'}).json()
        assert running['status'] in ('pending', 'running')
        _matches(job, running)

        zipping = client.post(
            '/api/v1/make_logs_zip', json={'output_dir': 'reports'}, headers={'Prefer': 'wait=0'}
        )
        assert zipping.status_code == 202
        release.set()
        ended = client.get(accepted.headers['Location'], headers={'Prefer': 'wait=10'}).json()
        assert ended['status'] == 'completed'
        _matches(job, ended)
        zipped = client.get(zipping.headers['Location'], headers={'Prefer': 'wait=30'}).json()
        assert zipped['status'] == 'completed' and zipped['progress'] is not None
        _matches(job, zipped)
        # A job's result is what the member's 200 would have been.
        _matches(
            _checker(described, _answer(described, '/api/v1/make_logs_zip', 'post')),
            zipped['result'],
        )
        _matches(
            _checker(described, _answer(described, '/api/v1/jobs', 'get')),
            client.get('/api/v1/jobs').json(),
        )

        handed = client.post('/api/v1/create_empty_protocol').json()
        _matches(
            _checker(described, _answer(described, '/api/v1/create_empty_protocol', 'post')),
            handed,
        )
        held = client.get('/api/v1/handles').json()
        assert handed in held
        _matches(_checker(described, _answer(described, '/api/v1/handles', 'get')), held)

        saved = client.post(
            '/api/v1/save_protocol', json={'protocol': handed['handle'], 'file_path': 'p.tsv'}
        )
        assert saved.status_code == 200, saved.json()
        assert saved.json()['name'] == 'p.tsv'
        _matches(
            _checker(described, _answer(described, '/api/v1/save_protocol', 'post')),
            saved.json(),
        )
