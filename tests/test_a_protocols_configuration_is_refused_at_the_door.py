# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol configuration the build cannot read is refused at ``create_protocol``, as the request's.

A client's ``input_config`` or ``empty_config`` was read wherever the build
first used each key, so a key left out was a ``KeyError`` and a value of the
wrong kind a ``TypeError`` from ``round()`` -- a 500 over REST -- while
``True`` built as 1 ms and NaN built a tile grid. ``create_protocol`` now
judges the whole configuration first, every key the build reads, and names
the key's path in ``argument``. What only the build can see -- no objective,
no focus, an overlap outside 0-50 -- is refused by the build and reported
once.
"""

import copy

import pytest
from fastapi.testclient import TestClient

from modules.exceptions import ArgumentRefusedError, ProtocolRunRefusedError, RefusalCause
from rest.app import build_app
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    s = _session(tmp_path_factory.mktemp('c7'), 'LS850')
    s.set_layer_acquire('BF', 'image')
    yield s
    s.shutdown()


@pytest.fixture
def config(session):
    return copy.deepcopy(session.get_sequenced_capture_config())


@pytest.fixture
def reported(monkeypatch):
    calls = []
    monkeypatch.setattr(
        'modules.notification_center.notifications.report_outcome',
        lambda exc, **kw: calls.append(exc),
    )
    return calls


def _refused(session, reason, **sources):
    with pytest.raises(ArgumentRefusedError) as refused:
        session.scope.protocols.create_protocol(**sources)
    assert refused.value.reason == reason
    assert refused.value.cause == RefusalCause.REQUEST
    return refused.value


def test_a_configuration_as_the_session_builds_it_builds(session, config):
    assert session.scope.protocols.create_protocol(input_config=config).num_steps() > 0


@pytest.mark.parametrize(
    ('path', 'argument'),
    [
        (('labware_id',), "input_config['labware_id']"),
        (('frame_dimensions',), "input_config['frame_dimensions']"),
        (('frame_dimensions', 'width'), "input_config['frame_dimensions']['width']"),
        (('binning_size',), "input_config['binning_size']"),
        (('use_zstacking',), "input_config['use_zstacking']"),
        (('stim_config',), "input_config['stim_config']"),
        (('layer_configs', 'BF', 'gain_db'), "input_config['layer_configs']['BF']['gain_db']"),
        (('layer_configs', 'BF', 'acquire'), "input_config['layer_configs']['BF']['acquire']"),
    ],
)
def test_a_key_left_out_is_refused_naming_its_path(session, config, path, argument):
    holder = config
    for key in path[:-1]:
        holder = holder[key]
    del holder[path[-1]]
    assert _refused(session, 'missing_key', input_config=config).argument == argument


@pytest.mark.parametrize(
    ('path', 'value', 'reason'),
    [
        (('layer_configs', 'BF', 'exposure_ms'), True, 'not_a_number'),
        (('layer_configs', 'BF', 'gain_db'), float('nan'), 'not_a_number'),
        (('layer_configs', 'BF', 'illumination_ma'), '50', 'not_a_number'),
        (('layer_configs', 'BF', 'sum'), 2.5, 'wrong_kind'),
        (('layer_configs', 'BF', 'autofocus'), 'yes', 'wrong_kind'),
        (('use_zstacking',), 'text', 'wrong_kind'),
        (('frame_dimensions',), 'big', 'wrong_kind'),
        (('binning_size',), True, 'wrong_kind'),
        (('tiling_overlap_percent',), float('nan'), 'not_a_number'),
        (('layer_configs', 'BF', 'acquire'), 'imagee', 'acquire_mode_unknown'),
        (('layer_configs',), {'Purple': {'acquire': None}}, 'layer_unknown'),
    ],
)
def test_a_value_of_the_wrong_kind_is_refused_before_anything_is_built(
    session, config, path, value, reason
):
    holder = config
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    _refused(session, reason, input_config=config)


def test_an_unknown_zstack_reference_is_refused_offering_the_three(session, config):
    config['use_zstacking'] = True
    config['zstack_params'] = {'range': 20.0, 'step_size': 5.0, 'z_reference': 'middle'}
    refusal = _refused(session, 'zstack_reference_unknown', input_config=config)
    assert refusal.offered == ('top', 'center', 'bottom')


def test_an_empty_protocol_judges_its_five_keys(session):
    empty = {
        'labware_id': '96 well microplate',
        'period': None,
        'duration': None,
        'frame_dimensions': 'big',
        'binning_size': 1,
    }
    assert _refused(session, 'wrong_kind', empty_config=empty).argument == (
        "empty_config['frame_dimensions']"
    )


@pytest.mark.parametrize('sources', [{}, {'input_config': {}, 'empty_config': {}}])
def test_not_exactly_one_source_is_refused(session, sources):
    _refused(session, 'protocol_source_ambiguous', **sources)


def test_steps_with_no_objective_are_refused_and_reported_once(session, config, reported):
    config['objective_id'] = None
    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.scope.protocols.create_protocol(input_config=config)
    assert refused.value.reason == 'objective_not_given'
    assert refused.value.cause == RefusalCause.REQUEST
    assert reported == [refused.value]


def test_a_layer_with_no_focus_and_no_current_z_is_refused_and_reported_once(
    session, config, reported
):
    config['layer_configs']['BF']['focus'] = None
    config['positions'] = [{'x': 10.0, 'y': 10.0, 'z': None, 'name': 'A1'}]
    config.pop('current_z', None)
    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.scope.protocols.create_protocol(input_config=config)
    assert refused.value.reason == 'focus_not_given'
    assert reported == [refused.value]


@pytest.mark.parametrize('overlap', [-1.0, 51.0])
def test_an_overlap_outside_0_to_50_is_refused_and_reported_once(
    session, config, reported, overlap
):
    config['tiling_overlap_percent'] = overlap
    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.scope.protocols.create_protocol(input_config=config)
    assert refused.value.reason == 'overlap_out_of_range'
    assert reported == [refused.value]


def test_a_wire_client_is_refused_as_the_request_naming_the_key(session, config):
    del config['layer_configs']['BF']['gain_db']
    # The schedule crosses the wire in seconds, as the server sends it.
    config['period'] = config['period'].total_seconds()
    config['duration'] = config['duration'].total_seconds()
    with TestClient(build_app(session)) as client:
        answer = client.post(
            '/api/v1/scope/protocols/create_protocol', json={'input_config': config}
        )
    assert answer.status_code == 422
    problem = answer.json()
    assert problem['reason'] == 'missing_key'
    assert problem['argument'] == "input_config['layer_configs']['BF']['gain_db']"


def test_the_zstack_keys_are_not_read_when_not_stacking(session, config):
    # A composite or a plain scan carries no stack; its zstack_params may be empty.
    config['use_zstacking'] = False
    config['zstack_params'] = {}
    assert session.scope.protocols.create_protocol(input_config=config).num_steps() > 0


def test_a_whole_number_stored_as_a_float_is_a_whole_number(session, config):
    # The settings store admits a count of 2.0 (its rule: a whole number),
    # so the build takes what the store holds.
    config['layer_configs']['BF']['sum'] = 2.0
    assert session.scope.protocols.create_protocol(input_config=config).num_steps() > 0


def test_an_overlap_of_50_percent_builds(session, config):
    config['tiling_overlap_percent'] = 50.0
    assert session.scope.protocols.create_protocol(input_config=config).num_steps() > 0


@pytest.mark.parametrize('acquire', [{}, [], 3])
def test_an_acquire_mode_of_another_kind_is_refused(session, config, acquire):
    config['layer_configs']['BF']['acquire'] = acquire
    _refused(session, 'acquire_mode_unknown', input_config=config)


@pytest.mark.parametrize(
    ('tuned', 'reason'), [(True, 'wrong_kind'), ({('A1', 'BF'): 'x'}, 'not_a_number')]
)
def test_a_tuned_z_map_is_judged_when_given(session, config, tuned, reason):
    config['previous_well_z'] = tuned
    _refused(session, reason, input_config=config)


def test_an_unknown_reference_is_refused_on_a_protocol_with_no_steps(session):
    protocol = session.create_empty_protocol()
    with pytest.raises(ArgumentRefusedError) as refused:
        session.scope.protocols.apply_zstacking(
            protocol, range_um=20.0, step_size_um=5.0, z_reference='middle'
        )
    assert refused.value.reason == 'zstack_reference_unknown'


@pytest.mark.parametrize(
    ('overlap', 'reason'),
    [(float('nan'), 'not_a_number'), ('10', 'not_a_number'), (51.0, 'overlap_out_of_range')],
)
def test_an_overlap_no_grid_can_take_is_refused_at_one_tile_too(session, overlap, reason):
    protocol = session.create_empty_protocol()
    with pytest.raises((ArgumentRefusedError, ProtocolRunRefusedError)) as refused:
        session.scope.protocols.apply_tiling(
            protocol,
            session.scope.protocols.tiling_config().no_tiling_label(),
            frame_dimensions={'width': 1900, 'height': 1900},
            binning_size=1,
            overlap_percent=overlap,
        )
    assert refused.value.reason == reason
