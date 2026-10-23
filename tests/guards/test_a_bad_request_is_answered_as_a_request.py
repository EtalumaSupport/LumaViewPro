# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A request the API cannot use is answered as the request's fault, never as the scope's.

A wire client that sends an argument a member cannot use -- an empty
dictionary where keys are read, a zero where a divisor is, a name that is
no file -- is told so: a 4xx whose ``type`` names the reason. Before this
guard, members read such arguments where they were first used, so the
client got a 500 titled ``KeyError`` or ``ZeroDivisionError`` with
``type`` ``about:blank``, or the camera's rejection of a change it was
never sent: the census of 2026-10-11 found six such routes behind one
another, each hidden by the last.

The walk posts every POST route of the OpenAPI document three times, and
asks every read that takes arguments three times through its query string,
each argument given a value of its schema's kind that is wrong for most members
(empty, negative, huge, an unknown name, a path that leaves the live
folder), against the simulated scope, and names every route that answers
500 or ``about:blank``. Each is asked with the longest wait, so a blocking
call is answered its end; a 202 started an operation, so the request was
accepted, and the operation is stopped.
"""

from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

# Routes the walk does not post. Each is a slow hardware or file operation
# whose bad arguments are refused by the same checks a walked route reaches
# (the path checks, the request reads), so posting it adds minutes and no
# member: the support report and the logs zip (a whole bundle), the camera
# bandwidth test, the file writer's recovery and the close's drain (each a
# step of a session ending), and the engineering members (raw board commands).
_NOT_WALKED = re.compile(
    r'discard_close_drain|recover_file_writer|bandwidth|support_report|logs_zip|engineering'
)

_WRONG = {
    'string': ('bogus', '', ' ../x'),
    'number': (-1e9, 1e12, -1.0),
    'integer': (-1, 999999, 0),
    'boolean': (True, True, True),
    'object': ({}, {}, {}),
    'array': ([], [], []),
}


def _schema(schema: dict, document: dict) -> dict:
    """``schema`` with its reference followed and its null alternative dropped, in either order."""
    if 'anyOf' in schema:
        kinds = [each for each in schema['anyOf'] if each.get('type') != 'null']
        schema = kinds[0] if kinds else {}
    if '$ref' in schema:
        schema = document['components']['schemas'][schema['$ref'].split('/')[-1]]
    return schema


def _wrong_value(schema: dict, document: dict, variant: int) -> object:
    schema = _schema(schema, document)
    if 'enum' in schema:
        return 'bogus_enum'
    return _WRONG.get(schema.get('type'), ('bogus',) * 3)[variant]


def _as_query(sent: dict) -> dict:
    """*sent* as query text: a list as its key repeated, a bool as ``true``, the rest as text."""
    query = {}
    for name, value in sent.items():
        values = value if isinstance(value, list) else [value]
        query[name] = [str(v).lower() if isinstance(v, bool) else str(v) for v in values]
    return query


def _ending(client, answer) -> tuple[int, dict]:
    """The status and problem a call was answered.

    Each call is asked with the longest wait, so a blocking call is answered
    its end. A 202 is a call that started an operation: the request was
    accepted, and the operation, which may run until it is stopped (a
    recording), is stopped rather than waited for.
    """
    if answer.status_code == 202:
        client.delete(answer.headers['Location'])
        return 202, {}
    return answer.status_code, answer.json() if answer.status_code >= 400 else {}


@pytest.fixture
def session(tmp_path):
    live = tmp_path / 'live'
    live.mkdir()
    (live / 'a_file.txt').write_text('x')
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    home_sim_scope(s.scope)
    # A protocol has steps, so a step's members reach their own checks.
    s.set_channel_acquire('BF', 'image')
    yield s
    s.shutdown()


# Slow: it asks every route three times (4 to 6 s), so it runs in the
# whole suite, which the trunk push gate requires every tenth commit.
@pytest.mark.slow
def test_no_route_answers_a_bad_request_as_a_fault(session):
    wrong = []
    with TestClient(build_app(session), raise_server_exceptions=False) as client:
        document = client.get('/api/v1/openapi.json').json()
        protocol = client.post('/api/v1/new_protocol').json()['handle']
        runner = client.post('/api/v1/create_protocol_runner').json()['handle']
        handles = {'/Protocol/': protocol, '/ProtocolRunner/': runner}
        walked = 0
        paths = document['paths']
        for route, method in sorted((p, m) for p in paths for m in paths[p]):
            operations = paths[route]
            asked = [p for p in operations[method].get('parameters', []) if p['in'] == 'query']
            if not (method in ('post', 'put') or asked) or _NOT_WALKED.search(route):
                continue
            walked += 1
            handle = next((h for part, h in handles.items() if part in route), '1')
            path = route.replace('{handle_id}', handle)
            if method in ('post', 'put'):
                body = (
                    operations[method]
                    .get('requestBody', {})
                    .get('content', {})
                    .get('application/json', {})
                    .get('schema', {})
                )
                fields = _schema(body, document).get('properties', {}) if body else {}
            else:
                # A read's arguments are its query string's keys.
                fields = {p['name']: p['schema'] for p in asked}
            for variant in range(3):
                sent = {
                    name: protocol if name == 'protocol' else _wrong_value(field, document, variant)
                    for name, field in fields.items()
                }
                if method in ('post', 'put'):
                    answer = client.request(
                        method.upper(),
                        path,
                        content=json.dumps(sent),
                        headers={'content-type': 'application/json', 'Prefer': 'wait=60'},
                    )
                else:
                    answer = client.get(path, params=_as_query(sent), headers={'Prefer': 'wait=60'})
                status, problem = _ending(client, answer)
                if status >= 500 or problem.get('type') == 'about:blank':
                    wrong.append(
                        f'{route} {json.dumps(sent)[:100]}: {status} '
                        f'{problem.get("title")!r} {problem.get("type")}'
                    )
            # A run a route started is ended before the next route is posted,
            # so no route is refused for an activity the walk itself left.
            session.create_protocol_runner().sequenced_capture_runner.wait_for_run_idle(1)
    # Each operation: the POSTs, the PUTs, and the reads that take arguments.
    assert walked >= 120, f'the walk reached only {walked} routes'
    assert not wrong, '\n'.join(wrong)
