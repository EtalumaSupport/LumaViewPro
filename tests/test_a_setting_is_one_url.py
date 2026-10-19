# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A setting is one URL, its read's: GET reads it, PUT writes it and answers the read.

A setter marked ``sets`` names the read whose setting it writes, and the
server serves the pair as one resource: ``PUT`` on the read's path calls the
setter with the JSON body and answers the read after the write, the value the
scope took. The setter has no route of its own, and a setter that does not
pair with its read is refused when the routes are built, so it is never
served.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from modules.api_surface import api
from modules.scope_capabilities import FrameSize
from modules.scope_session import ScopeSession
from rest.app import build_app
from rest.routes import routes
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

# Each setting with a URL on a simulated LS850: the body its PUT is sent, the
# key its GET is asked with, and what both then answer; or the reason the
# simulated scope refuses it -- its camera lacks the feature, the LS850 has
# no turret -- which leaves the read answering as it did.
_WRITES = {
    'model': ({'model': 'LS560'}, None, 'LS560'),
    'objective_id': ({'objective_id': '10x Oly'}, None, '10x Oly'),
    'labware': ({'labware_name': '384 well microplate'}, None, '384 well microplate'),
    'scale_bar_enabled': ({'enabled': True}, None, True),
    'acceleration_limit_pct': ({'val_pct': 40}, None, 40),
    'layer_acquire': ({'layer': 'Blue', 'mode': 'video'}, {'layer': 'Blue'}, 'video'),
    'layer_auto_gain': ({'layer': 'Green', 'enabled': True}, {'layer': 'Green'}, True),
    'saved_focus': ({'layer': 'Blue', 'z_um': 5000.0}, {'layer': 'Blue'}, 5000.0),
    'scope/camera/image_mode': ({'mode': '12bit_scientific'}, None, '12bit_scientific'),
    'scope/camera/binning': ({'size': 2}, None, 2),
    'scope/camera/frame_size': ({'width': 600, 'height': 400}, None, {'width': 600, 'height': 400}),
    'scope/camera/gain_db': ({'gain_db': 5.0}, None, 5.0),
    'scope/camera/exposure_ms': ({'exposure_ms': 20.0}, None, 20.0),
    'scope/camera/black_level': ({'value': 4.0}, None, 4.0),
    'scope/camera/auto_exposure': ({'state': True}, None, True),
}
_REFUSED = {
    'turret_objective': (
        {'position': 2, 'objective_id': '10x Oly'},
        {'position': 2},
        'axis_absent',
    ),
    'scope/camera/high_conversion_gain': (
        {'enabled': True},
        None,
        'conversion_gain_mode_unsupported',
    ),
    'scope/camera/line_noise_reduction': (
        {'enabled': True},
        None,
        'line_noise_reduction_unsupported',
    ),
}


def _shutdown(session):
    try:
        session.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


@pytest.fixture
def session(tmp_path):
    built = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850'), simulate=True
    )
    home_sim_scope(built.scope)
    yield built
    _shutdown(built)


@pytest.fixture
def client(session):
    with TestClient(build_app(session)) as client:
        yield client


def _put_paths(client) -> set[str]:
    paths = client.get('/api/v1/openapi.json').json()['paths']
    return {path.removeprefix('/api/v1/') for path, ops in paths.items() if 'put' in ops}


def _read_of(answer) -> tuple[int, object]:
    """A read's status and value, or its refusal's reason: a problem's instance is new each time."""
    body = answer.json()
    return answer.status_code, body if answer.status_code == 200 else body.get('reason')


def test_each_setting_is_written_by_put_and_answered_its_read(client):
    assert _put_paths(client) == set(_WRITES) | set(_REFUSED)
    wrong = {}
    for path, (body, key, expected) in _WRITES.items():
        put = client.put(f'/api/v1/{path}', json=body)
        read = client.get(f'/api/v1/{path}', params=key)
        answers = (put.status_code, put.json(), read.status_code, read.json())
        if answers != (200, expected, 200, expected):
            wrong[path] = answers
    for path, (body, key, reason) in _REFUSED.items():
        before = client.get(f'/api/v1/{path}', params=key)
        put = client.put(f'/api/v1/{path}', json=body)
        after = client.get(f'/api/v1/{path}', params=key)
        answers = (put.json().get('reason'), _read_of(after))
        if answers != (reason, _read_of(before)):
            wrong[path] = answers
    assert wrong == {}


def test_a_setter_has_no_route_of_its_own(client):
    documented = client.get('/api/v1/openapi.json').json()['paths']
    setters = {
        f'/api/v1/{ops["put"]["operationId"].replace(".", "/")}'
        for ops in documented.values()
        if 'put' in ops
    }
    assert '/api/v1/scope/camera/set_binning' in setters
    served = {path: client.post(path, json={}).status_code for path in sorted(setters)}
    assert served == dict.fromkeys(sorted(setters), 404)
    assert client.post('/api/v1/scope/camera/binning', json={'size': 1}).status_code == 405


def test_a_put_that_outlives_its_wait_is_a_job_whose_result_is_the_read(tmp_path):
    # On a turret scope choosing an objective turns the turret to it, which
    # outlives a respond-async wait of none.
    settings = complete_settings(live_folder=str(tmp_path), microscope='LS850T')
    settings['turret_objectives'] = {1: '4x Oly', 2: '10x Oly', 3: None, 4: None}
    turreted = ScopeSession.create(settings, simulate=True)
    try:
        home_sim_scope(turreted.scope)
        with TestClient(build_app(turreted)) as client:
            asked = client.put(
                '/api/v1/objective_id',
                json={'objective_id': '10x Oly'},
                headers={'Prefer': 'respond-async'},
            )
            assert asked.status_code == 202
            ended = client.get(asked.headers['Location'], headers={'Prefer': 'wait=30'}).json()
            assert (ended['status'], ended['result']) == ('completed', '10x Oly')
            assert client.get('/api/v1/objective_id').json() == '10x Oly'
    finally:
        _shutdown(turreted)


class _NoSuchRead:
    @api(sets='level')
    def set_level(self, value: int) -> None:
        pass


class _NotARead:
    @api
    def level(self) -> int:
        return 0

    @api(sets='level')
    def set_level(self, value: int) -> None:
        pass


class _AnotherKind:
    @api
    @property
    def level(self) -> int:
        return 0

    @api(sets='level')
    def set_level(self, value: str) -> None:
        pass


class _AnotherKey:
    @api(reads=True)
    def level(self, channel: str) -> int:
        return 0

    @api(sets='level')
    def set_level(self, layer: str, value: int) -> None:
        pass


class _NotTheRecordsFields:
    @api
    @property
    def frame(self) -> FrameSize:
        return FrameSize(0, 0)

    @api(sets='frame')
    def set_frame(self, height: int, width: int) -> None:
        pass


class _Pairs:
    @api(reads=True)
    def level(self, channel: str) -> int | None:
        return 0

    @api(sets='level')
    def set_level(self, channel: str, value: int) -> None:
        pass

    @api
    @property
    def frame(self) -> FrameSize:
        return FrameSize(0, 0)

    @api(sets='frame')
    def set_frame(self, width: int, height: int) -> None:
        pass


def test_a_setter_that_does_not_pair_with_its_read_is_refused_when_built():
    """Each refusal is the routes' build's, so no such setter is ever served:
    its read is missing or is not a read, writes another kind than the read
    answers, takes another key, or sets a record's fields under other names."""
    unpaired = (_NoSuchRead, _NotARead, _AnotherKind, _AnotherKey, _NotTheRecordsFields)
    wrong = []
    for cls in unpaired:
        try:
            routes(cls, handed_out=frozenset())
        except TypeError as e:
            if f'{cls.__name__}.set_' not in str(e):
                wrong.append(f'{cls.__name__}: refused for another reason: {e}')
            continue
        wrong.append(f'{cls.__name__}: built')
    assert wrong == []
    built = {(r.method, r.path, r.name) for r in routes(_Pairs, handed_out=frozenset())}
    assert {('put', 'level', 'set_level'), ('put', 'frame', 'set_frame')} <= built
