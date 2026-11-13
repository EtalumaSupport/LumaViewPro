# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A wire client takes a still at the camera's own depth, record and pixels in one answer.

``POST /api/v1/still`` reaches ``ScopeSession.capture_still`` with no 8-bit
conversion and the member's arguments its body sends, the camera as set
when it sends none, as every write is reached: on a call
thread as a REST client, in the control session its header names. The
answer's body is the still's record as JSON, ``X-Record-Length`` bytes,
then its pixels, read back by ``X-Image-Shape``, ``X-Image-Dtype`` and
``X-Significant-Bits``. It is answered when the still is taken, never as a
job, since a job's answer cannot carry the pixels. The route is the
server's own, so the guard over the members' routes does not reach it.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield s
    s.shutdown()


@pytest.fixture
def client(session):
    with TestClient(build_app(session)) as client:
        yield client


def _still(answer):
    """The answer's record and pixels, read back by its headers alone."""
    length = int(answer.headers['X-Record-Length'])
    record = json.loads(answer.content[:length])
    shape = tuple(int(n) for n in answer.headers['X-Image-Shape'].split(','))
    dtype = np.dtype(answer.headers['X-Image-Dtype']).newbyteorder('<')
    pixels = np.frombuffer(answer.content[length:], dtype=dtype).reshape(shape)
    return record, pixels


def _sources(record):
    """The changes the still's frame was judged against, by source."""
    return {judgment['change']['source'] for judgment in record['judgments']}


def test_a_twelve_bit_still_is_its_record_then_its_pixels_at_twelve_bits(client, session):
    session.scope.camera.set_image_mode('12bit_scientific')
    # Lit, so the pixels have values to read back; unlit, every one is 0.
    session.scope.simulation.light_outside(40.0)
    in_process = session.capture_still(None, force_to_8bit=False).image

    answer = client.post('/api/v1/still', json={'exposed_after_call': True})

    assert answer.status_code == 200, answer.text
    assert answer.headers['content-type'] == 'application/octet-stream'
    record, pixels = _still(answer)
    assert (pixels.dtype, pixels.shape) == (np.uint16, in_process.shape)
    assert answer.headers['X-Significant-Bits'] == '12'
    assert record['frame_significant_bits'] == 12
    # Pixels read in the wrong order or at the wrong type pass 12 bits.
    assert 0 < int(pixels.max()) < 2**12


def test_exposed_after_call_reaches_the_capture_and_its_record_says_so(client):
    plain, _ = _still(client.post('/api/v1/still'))
    exposed, _ = _still(client.post('/api/v1/still', json={'exposed_after_call': True}))

    assert 'caller' not in _sources(plain)
    assert 'caller' in _sources(exposed)


def test_the_still_takes_every_argument_of_its_member_but_three(client):
    """A parameter added to ``capture_still`` reaches the wire or is named
    here as one it never takes: the 8-bit conversion (the route answers the
    camera's own depth), the changes a frame may skip (a frame handed out to
    be measured never skips one) and the per-frame callback (no wire form)."""
    function = inspect.unwrap(ScopeSession.capture_still)
    never = {'self', 'force_to_8bit', 'exclude_sources', 'sum_iteration_callback'}
    document = client.get('/api/v1/openapi.json').json()
    body = document['paths']['/api/v1/still']['post']['requestBody']
    schema = body['content']['application/json']['schema']
    schema = schema['anyOf'][0] if 'anyOf' in schema else schema
    if '$ref' in schema:
        schema = document['components']['schemas'][schema['$ref'].rsplit('/', 1)[1]]

    assert set(schema['properties']) == set(inspect.signature(function).parameters) - never


def test_a_summed_still_is_its_count_of_frames_at_the_depth_it_reaches(client, session):
    session.scope.camera.set_image_mode('12bit_scientific')
    session.scope.simulation.light_outside(40.0)

    answer = client.post('/api/v1/still', json={'sum_count': 4, 'sum_delay_s': 0.0})

    assert answer.status_code == 200, answer.text
    record, pixels = _still(answer)
    assert record['frames_summed'] == 4
    # Four 12-bit frames reach 14 bits, held in 16.
    assert answer.headers['X-Significant-Bits'] == '14'
    assert pixels.dtype == np.uint16


def test_a_still_of_the_shown_channel_holds_its_settings(client, session):
    shown = session.shown_channel.channel
    other = 'PC' if shown != 'PC' else 'BF'

    held = client.post('/api/v1/still', json={'channel': shown, 'accept_dark': True})
    refused = client.post('/api/v1/still', json={'channel': other})

    assert held.status_code == 200, held.text
    assert _still(held)[0]['held_channel'] == shown
    assert refused.json()['reason'] == 'channel_not_shown', refused.json()


def test_a_sum_of_no_frame_is_the_members_refusal_not_a_server_fault(client):
    refused = client.post('/api/v1/still', json={'sum_count': 0})

    assert refused.headers['content-type'] == 'application/problem+json'
    assert refused.json()['reason'] == 'wrong_kind', refused.json()
    assert refused.json()['argument'] == 'sum_count'


def test_a_key_the_still_does_not_take_is_refused(client):
    refused = client.post('/api/v1/still', json={'exposed_after_cal': True})

    assert refused.status_code == 422
    assert refused.json()['reason'] == 'invalid_request'


def test_a_still_the_scope_refuses_is_answered_the_members_problem(client, session):
    session.scope.simulation.remove_camera()

    refused = client.post('/api/v1/still')

    assert refused.headers['content-type'] == 'application/problem+json'
    assert refused.json()['reason'] == 'not_connected', refused.json()


def test_under_a_held_session_only_its_holder_takes_a_still(client):
    session_id = client.post('/api/v1/take_control', json={'name': 'acquisition engine'}).json()

    refused = client.post('/api/v1/still')
    taken = client.post('/api/v1/still', headers={'Control-Session': session_id})

    assert refused.status_code == 409
    assert refused.json()['reason'] == 'control_session_held'
    assert taken.status_code == 200, taken.text


def test_a_still_is_answered_with_its_pixels_whatever_the_client_will_wait(client):
    """A member that outlives the client's wait is answered 202 and a job;
    the still is answered when it is taken, since a job carries no pixels."""
    answer = client.post('/api/v1/still', headers={'Prefer': 'respond-async'})

    assert answer.status_code == 200, answer.text
    record, pixels = _still(answer)
    assert pixels.size > 0
    assert record['frame_id'] is not None
