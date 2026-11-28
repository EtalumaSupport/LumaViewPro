# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The OpenAPI document is what a generated client and an agent read: one name for each thing.

Read from the document an application over a simulated session builds
(``build_app``, its ``openapi()``):

- each record is one schema, every component's title its own, so a
  generated client has one type per record and drops no answer for a
  second one (``<Record>-Input`` beside ``<Record>-Output``);
- every operation has the name the server gives it, none of FastAPI's
  made-up ones (``read_api_v1_jobs__job_id__get``), each name once;
- the problem every refusal and failure is answered with is one
  component, ``Problem``, and a failed job's ``error`` is it;
- a body whose every field has a default is its record's schema, not
  "it or null", which a generated client cannot send;
- what a route reads by hand is declared: ``Control-Session`` on every
  write, ``max_width`` on the live view and a preview, ``Last-Event-ID``
  on the event stream, ``Range`` on a file; a 202 says where its job is
  (``Location``) and when to read it (``Retry-After``);
- every summary and description is Markdown, with no reStructuredText
  double backtick left, and the document says first what every route
  shares (``Prefer``, ``Idempotency-Key``, ``X-Request-ID``).
"""

from __future__ import annotations

import json

import fastapi.routing
import pytest

from _lumascope.modules import wire_encoding
from _lumascope.modules.exceptions import ConfigurationDefectError
from _lumascope.modules.scope_session import ScopeSession
import _lumascope.rest.app
from _lumascope.rest import routes as rest_routes
from _lumascope.rest.app import PREFIX, build_app, served
from tests.settings_fixtures import complete_settings

_OPERATIONS = ('get', 'put', 'post', 'delete', 'patch')


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    """The application over a simulated session, and the document it publishes."""
    live = tmp_path_factory.mktemp('live')
    session = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    try:
        app = build_app(session)
        yield app, app.openapi()
    finally:
        session.shutdown()


def _operations(document: dict) -> list[tuple[str, str, dict]]:
    return [
        (path, method, operation)
        for path, methods in document['paths'].items()
        for method, operation in methods.items()
        if method in _OPERATIONS
    ]


def _parameters(document: dict, operation: dict) -> list[dict]:
    """An operation's parameters, each shared one read where it is declared."""
    shared = document['components'].get('parameters', {})
    return [
        shared[p['$ref'].rsplit('/', 1)[-1]] if '$ref' in p else p
        for p in operation.get('parameters', [])
    ]


def test_every_record_is_one_schema_and_every_title_its_own(built):
    _app, document = built
    schemas = document['components']['schemas']
    titles: dict[str, list[str]] = {}
    for name, schema in schemas.items():
        titles.setdefault(schema.get('title', name), []).append(name)
    twice = {title: names for title, names in titles.items() if len(names) > 1}
    assert 'PlateLocation' in schemas
    assert twice == {}


def test_a_record_described_two_ways_fails_the_document_as_a_configuration_defect():
    # The merge read alone: no record the release declares differs, so no
    # document built from one can reach this.
    side = {'properties': {'x': {'type': 'number'}}, 'required': ['x']}
    described = {
        'components': {
            'schemas': {
                'Point-Input': side,
                'Point-Output': {**side, 'required': []},
            }
        }
    }

    with pytest.raises(ConfigurationDefectError) as refused:
        _lumascope.rest.app._one_schema_per_record(described)

    assert refused.value.reason == 'record_schemas_differ'


def test_every_operation_is_named_by_the_server_once(built):
    app, document = built
    unnamed = sorted(
        f'{sorted(route.methods)} {route.path}'
        for route in app.routes
        if isinstance(route, fastapi.routing.APIRoute)
        and route.include_in_schema
        and route.operation_id is None
    )
    names = [operation['operationId'] for _, _, operation in _operations(document)]
    assert unnamed == []
    assert len(names) == len(set(names))
    assert {'jobs.read', 'events.stream', 'still'} <= set(names)


def test_every_problem_and_a_failed_job_s_error_is_the_one_problem(built):
    _app, document = built
    problem = {'$ref': '#/components/schemas/Problem'}
    error = document['components']['schemas']['Job']['properties']['error']
    inline = sorted(
        f'{method} {path} {code}'
        for path, method, operation in _operations(document)
        for code, response in operation['responses'].items()
        for media, content in response.get('content', {}).items()
        if media == 'application/problem+json' and content['schema'] != problem
    )
    assert error['anyOf'] == [problem, {'type': 'null'}]
    assert inline == []


def test_a_body_of_defaults_is_its_record_not_it_or_null(built):
    _app, document = built
    or_null = sorted(
        f'{method} {path}'
        for path, method, operation in _operations(document)
        if {'type': 'null'}
        in operation.get('requestBody', {})
        .get('content', {})
        .get('application/json', {})
        .get('schema', {})
        .get('anyOf', [])
    )
    still = document['paths']['/api/v1/still']['post']['requestBody']
    assert or_null == []
    assert still.get('required', False) is False
    assert '$ref' in still['content']['application/json']['schema']


def test_what_a_route_reads_by_hand_is_declared(built):
    _app, document = built
    paths = document['paths']

    def names(path: str, method: str) -> set[str]:
        return {p['name'] for p in _parameters(document, paths[path][method])}

    # A member's call writes unless it reads or stops (a stop is anyone's);
    # each of the server's own writes takes the header too.
    handed_out = wire_encoding.handed_out(
        ScopeSession, wire_encoding.project_classes(), wire_encoding.project_aliases()
    )
    member_routes = rest_routes.routes(ScopeSession, handed_out=handed_out)
    for cls in sorted(handed_out, key=lambda c: c.__name__):
        member_routes += rest_routes.routes(cls, handed_out=handed_out)
    forgets = {
        (path, 'delete')
        for path in paths
        if path.startswith('/api/v1/handles/') and path.endswith('}') and 'delete' in paths[path]
    }
    writes = {
        (f'{PREFIX}/{served(route)[0]}', route.method)
        for route in member_routes
        if not (route.member.read or route.member.reads or route.member.is_stop)
    } | {
        ('/api/v1/files/{name}', 'put'),
        ('/api/v1/still', 'post'),
        ('/api/v1/jobs/{job_id}', 'delete'),
        *forgets,
    }
    named = {
        (path, method)
        for path, method, _ in _operations(document)
        if 'Control-Session' in names(path, method)
    }
    assert len(writes) > 100
    assert named == writes, sorted(named ^ writes)
    assert 'Control-Session' in names('/api/v1/scope/motion/start_move_xy', 'post')
    assert 'Control-Session' not in names('/api/v1/scope/motion/axis_limits', 'get')
    for path in ('/api/v1/live', '/api/v1/live.jpg', '/api/v1/previews/{name}'):
        assert 'max_width' in names(path, 'get'), path
    assert 'Last-Event-ID' in names('/api/v1/events', 'get')
    assert 'Range' in names('/api/v1/files/{name}', 'get')
    jobs_without = sorted(
        f'{method} {path}'
        for path, method, operation in _operations(document)
        if '202' in operation['responses']
        and path != '/api/v1/jobs/{job_id}'
        and set(operation['responses']['202'].get('headers', {})) != {'Location', 'Retry-After'}
    )
    assert jobs_without == []


def test_the_words_are_markdown_and_the_document_says_first_what_routes_share(built):
    _app, document = built
    in_rst = sorted(
        f'{method} {path}'
        for path, method, operation in _operations(document)
        if '``' in (operation.get('summary') or '') + (operation.get('description') or '')
    )
    components = sorted(
        name
        for name, schema in document['components']['schemas'].items()
        if '``' in json.dumps(schema)
    )
    info = document['info']['description']
    assert in_rst == []
    assert components == []
    for header in ('Prefer', 'Idempotency-Key', 'X-Request-ID', 'Control-Session', 'Location'):
        assert f'`{header}' in info, header
