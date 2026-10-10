# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every answer the REST server sends is described in its OpenAPI description.

A client that reads ``/api/v1/openapi.json`` -- Swagger, a generated client,
an importer -- learns what each route answers on success from the member's
declared return type, read into its wire form by ``wire_encoding.outbound``
from the table ``encode`` follows. Read from the description an application
over a simulated session builds:

- every member route's 200 is its answer's schema, never ``{}``, and bare
  (an object with no keys, an array of nothing said) only where the
  member's own type says no more (a bare ``dict``); its 202 is the job;
- a route that is not JSON declares what it is: a file, a JPEG, an MJPEG
  stream, an event stream; the server's own JSON routes their models;
- each event the stream sends has a component, listed by ``/events``;
- the untyped places in what answers and events carry -- a keyless
  object, an array of nothing said, a value of no type -- are counted, a
  ratchet, so they only fall as their owners type them;
- no two classes an answer or argument reaches share a name, and no
  component name is one FastAPI made up for a clash, since the last class
  of a name would silently stand for every annotation that names it;
- every enum an answer or argument reaches crosses as its name, a
  ``StrEnum`` as its value: what ``encode`` sends is what is declared.
"""

from __future__ import annotations

import enum
import inspect
import json
import pathlib
import sys
import typing

import pytest

from modules import wire_encoding
from rest import routes as rest_routes

# Keyless objects, arrays of nothing said and values of no type, over every
# route's 200 and 202 and every event, each component counted once.
UNTYPED_PIN = 52

_SERVER_MODELS = ('Handle', 'Job', 'JobProgress', 'PathAnswer', 'Versions')
# A JSON Schema keyword that says what a value is.
_TYPING = {'type', '$ref', 'anyOf', 'oneOf', 'allOf', 'enum', 'const'}


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    """The application's description, its member routes, its events' models, and their records."""
    from modules.scope_session import ScopeSession
    from rest import events
    from rest.app import build_app
    from tests.settings_fixtures import complete_settings

    live = tmp_path_factory.mktemp('live')
    session = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    classes = wire_encoding.project_classes()
    aliases = wire_encoding.project_aliases()
    handed_out = wire_encoding.handed_out(ScopeSession, classes, aliases)
    try:
        described = build_app(session).openapi()
        published = events.published(session, handed_out)
        records = events.records(session)
    finally:
        session.shutdown()
    member_routes = rest_routes.routes(ScopeSession, handed_out=handed_out)
    for cls in sorted(handed_out, key=lambda c: c.__name__):
        member_routes += rest_routes.routes(cls, handed_out=handed_out)
    return described, member_routes, published, records


def _url(route: rest_routes.Route) -> str:
    from modules.scope_session import ScopeSession

    if route.root is ScopeSession:
        return f'/api/v1/{route.path}'
    return f'/api/v1/handles/{route.root.__name__}/{{handle_id}}/{route.path}'


def _operation(described: dict, route: rest_routes.Route) -> dict:
    return described['paths'][_url(route)]['get' if route.member.read else 'post']


def _json(response: dict) -> dict:
    return response['content']['application/json']['schema']


def _bare(schema: dict) -> bool:
    """A value that says nothing of itself: no type, an object with no keys, an array of nothing."""
    if not _TYPING & set(schema):
        return True
    kinds = schema.get('type')
    kinds = kinds if isinstance(kinds, list) else [kinds]
    if 'object' in kinds and 'properties' not in schema:
        values = schema.get('additionalProperties')
        return not (isinstance(values, dict) and _TYPING & set(values))
    if 'array' in kinds:
        return 'prefixItems' not in schema and not _TYPING & set(schema.get('items', {}))
    return False


def _untyped(described: dict, events: dict) -> list[str]:
    """Each untyped place in what the routes answer and the events carry, by where it is."""
    schemas = described['components']['schemas']
    found, followed = [], set()

    def walk(node: object, where: str) -> None:
        if isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f'{where}/{i}')
            return
        if not isinstance(node, dict):
            return
        if '$ref' in node:
            name = node['$ref'].rsplit('/', 1)[-1]
            if name not in followed:
                followed.add(name)
                walk(schemas[name], name)
            return
        if _bare(node):
            found.append(where)
            return
        for key in ('anyOf', 'oneOf', 'allOf', 'prefixItems'):
            walk(node.get(key), f'{where}/{key}')
        for key in ('items', 'additionalProperties'):
            if isinstance(node.get(key), dict):
                walk(node[key], f'{where}/{key}')
        for name, field in node.get('properties', {}).items():
            walk(field, f'{where}.{name}')

    for path, operations in described['paths'].items():
        for method, operation in operations.items():
            for status in ('200', '202'):
                response = operation['responses'].get(status, {})
                if 'application/json' in response.get('content', {}):
                    walk(_json(response), f'{method.upper()} {path} {status}')
    for model in events.values():
        walk({'$ref': f'#/components/schemas/{model.__name__}'}, model.__name__)
    return found


def test_every_member_route_declares_its_answer_and_its_job(built):
    described, member_routes, _events, _records = built
    wrong = []
    for route in member_routes:
        responses = _operation(described, route)['responses']
        answer = _json(responses['200'])
        if answer == {}:
            wrong.append(f'{_url(route)}: 200 is {{}}')
            continue
        top = answer.get('anyOf', [answer])
        said_nothing = [
            a
            for a in route.member.returns
            if a.form in (wire_encoding.ARRAY, wire_encoding.OBJECT) and not (a.items or a.parts)
        ]
        bare = [s for s in top if _bare(s)]
        if len(bare) != len(said_nothing):
            wrong.append(f'{_url(route)}: 200 {json.dumps(answer)} for {route.member.returns}')
        if _json(responses['202']) != {'$ref': '#/components/schemas/Job'}:
            wrong.append(f'{_url(route)}: 202 is not the job')
    assert len(member_routes) > 250
    assert not wrong, '\n'.join(wrong)


def test_a_route_that_is_not_json_declares_what_it_is(built):
    described, _routes, events, _records = built
    sent = {
        '/api/v1/files/{name}': 'application/octet-stream',
        '/api/v1/live': 'multipart/x-mixed-replace',
        '/api/v1/live.jpg': 'image/jpeg',
        '/api/v1/events': 'text/event-stream',
    }
    declared = {
        path: list(described['paths'][path]['get']['responses']['200']['content']) for path in sent
    }
    assert declared == {path: [media] for path, media in sent.items()}
    listing = described['paths']['/api/v1/events']['get']['responses']['200']['description']
    for name, model in events.items():
        assert f'`{name}`: `#/components/schemas/{model.__name__}`' in listing
        assert model.__name__ in described['components']['schemas']


def test_the_servers_own_routes_declare_their_models(built):
    described, _routes, _events, _records = built
    ref = '#/components/schemas/'
    assert _json(described['paths']['/api']['get']['responses']['200']) == {
        '$ref': f'{ref}Versions'
    }
    for path, model in (
        ('/api/v1/handles', 'Handle'),
        ('/api/v1/jobs', 'Job'),
    ):
        listing = _json(described['paths'][path]['get']['responses']['200'])
        assert listing['items'] == {'$ref': f'{ref}{model}'}, path
    job = _json(described['paths']['/api/v1/jobs/{job_id}']['get']['responses']['200'])
    assert job == {'$ref': f'{ref}Job'}


def test_the_untyped_places_in_answers_and_events_do_not_grow(built):
    described, _routes, events, _records = built
    untyped = _untyped(described, events)
    _counted.append(len(untyped))
    assert len(untyped) <= UNTYPED_PIN, (
        f'{len(untyped)} untyped places in answers and events, over the pin of '
        f'{UNTYPED_PIN}: type the new one (a record, a typed array or mapping), or '
        f'raise the pin in this commit and say why.\n  ' + '\n  '.join(untyped)
    )


def _reached(member_routes: list[rest_routes.Route], records: dict[str, type]) -> set[type]:
    """Every project class an answer, an argument or an event reaches."""
    found: set[type] = set()

    def out(alternatives: tuple[wire_encoding.Outbound, ...]) -> None:
        for a in alternatives:
            if a.cls is not None and a.cls not in found:
                found.add(a.cls)
                for _f, alts in a.fields:
                    out(alts)
            out(a.items)
            for part in a.parts:
                out(part)

    def into(alternatives: tuple[wire_encoding.Inbound, ...]) -> None:
        for a in alternatives:
            if a.cls is not None:
                found.add(a.cls)
            into(a.items)
            for _f, alts in a.fields:
                into(alts)

    classes = wire_encoding.project_classes()
    aliases = wire_encoding.project_aliases()
    for record in records.values():
        out(wire_encoding.outbound(record.__name__, classes, aliases))
    for route in member_routes:
        found.add(route.owner)
        out(route.member.returns)
        for parameter in route.member.parameters:
            into(parameter.alternatives)
    return found


def _same_named(reached: set[type], every: dict[str, set[type]]) -> list[str]:
    """Each reached class whose name another project class also has."""
    return sorted(
        f'{cls.__name__}: {sorted(f"{c.__module__}.{c.__qualname__}" for c in every[cls.__name__])}'
        for cls in reached
        if len(every.get(cls.__name__, ())) > 1
    )


def _every_project_class() -> dict[str, set[type]]:
    wire_encoding.project_classes()
    every: dict[str, set[type]] = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith('modules'):
            continue
        for cname, obj in vars(module).items():
            if inspect.isclass(obj) and obj.__module__ == name:
                every.setdefault(cname, set()).add(obj)
    return every


def test_no_two_classes_on_the_wire_share_a_name(built):
    described, member_routes, events, records = built
    reached = _reached(member_routes, records)
    assert _same_named(reached, _every_project_class()) == []
    event_models = {
        m.__name__ for m in events.values() if m not in rest_routes._ANSWER_RECORDS.values()
    }
    clashing = sorted({c.__name__ for c in reached} & {*_SERVER_MODELS, *event_models})
    assert clashing == [], 'a project class is named as one of the server models'
    made_up = [n for n in described['components']['schemas'] if '__' in n]
    assert made_up == [], f'FastAPI renamed clashing components: {made_up}'


def test_a_second_class_of_a_reached_name_is_caught():
    from modules.lumascope_api._constants import AxisPosition

    other = type('AxisPosition', (), {'__module__': 'modules.elsewhere'})
    every = {'AxisPosition': {AxisPosition, other}}

    assert _same_named({AxisPosition}, every) == [
        "AxisPosition: ['modules.elsewhere.AxisPosition', "
        "'modules.lumascope_api._constants.AxisPosition']"
    ]


def test_every_enum_on_the_wire_crosses_in_its_declared_form(built, tmp_path):
    _described, member_routes, _events, records = built
    enums = sorted(
        (c for c in _reached(member_routes, records) if issubclass(c, enum.Enum)),
        key=lambda c: c.__name__,
    )

    def no_handle(obj):
        raise AssertionError(obj)

    drifted = []
    for cls in enums:
        declared = [m.value if issubclass(cls, enum.StrEnum) else m.name for m in cls]
        sent = [
            json.loads(
                json.dumps(
                    wire_encoding.encode(
                        m, live_folder=pathlib.Path(tmp_path), handle=no_handle, job=no_handle
                    )
                )
            )
            for m in cls
        ]
        described = list(
            typing.get_args(
                rest_routes.answer_type((wire_encoding.Outbound('enum', cls.__name__, cls),))
            )
        )
        if not sent == described == declared:
            drifted.append(
                f'{cls.__name__}: declared {declared}, described {described}, sent {sent}'
            )
    assert enums, 'no enum reached: the walk is broken'
    assert not drifted, '\n'.join(drifted)


# Announced at the end of every run (tests/ratchets.py).
from tests import ratchets as _ratchets


# The count the ratchet test measured; the summary prints it when the test ran.
_counted: list[int] = []


def _measure() -> int:
    if not _counted:
        raise RuntimeError('the ratchet test did not run')
    return _counted[-1]


_ratchets.register('rest: untyped places in answers and events', _measure, UNTYPED_PIN)
