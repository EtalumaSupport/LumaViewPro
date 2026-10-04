# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A step index outside the protocol is refused by every writer.

Each ``Protocol`` writer checked its index on its own, or not at all.
``modify_autofocus`` and ``modify_step_z_height`` wrote through pandas'
``.at``, which answers an index past the end -- or -1 -- by appending a
phantom row: a step nobody added, no step-list revision, run like any other.
``apply_zstack_group_focus`` raised a bare ``KeyError``; ``insert_step``
placed a step past the end without a word; ``delete_step`` on an empty
protocol did nothing. Every writer now refuses with ``ProtocolError`` and
leaves the steps as they were.

``after_step=-1`` is not out of range: it is "after no step", how an empty
protocol takes its first step through the Session and through the GUI.
"""

import datetime

import pandas as pd
import pytest

from modules import config_helpers
from modules.exceptions import ProtocolError
from modules.protocol import Protocol
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.test_protocol_roundtrip import TILING_CONFIGS, _build_protocol, _make_step

NUM_STEPS = 3
OUTSIDE = [NUM_STEPS, NUM_STEPS + 3, -1]

LAYER_CONFIG = {
    'autofocus': False,
    'false_color': False,
    'illumination_ma': 50.0,
    'gain_db': 1.0,
    'auto_gain': False,
    'exposure_ms': 10.0,
    'sum': 1,
    'acquire': 'image',
    'video_config': {'duration': 5, 'fps': 5},
    'focus': None,
}
PLATE_POSITION = {'x': 10.0, 'y': 20.0, 'z': 5000.0}


def _protocol():
    return _build_protocol(
        [_make_step(name=f'A{i + 1}_BF', well=f'A{i + 1}') for i in range(NUM_STEPS)]
    )


WRITERS = {
    'modify_autofocus': lambda p, i: p.modify_autofocus(step_idx=i, enabled=True),
    'modify_step_z_height': lambda p, i: p.modify_step_z_height(step_idx=i, z=1.0),
    'apply_zstack_group_focus': lambda p, i: p.apply_zstack_group_focus(
        reference_step_idx=i, z=1.0
    ),
    'modify_name': lambda p, i: p.modify_name(step_idx=i, step_name='renamed'),
    'modify_step': lambda p, i: p.modify_step(
        step_idx=i,
        layer='BF',
        layer_config=LAYER_CONFIG,
        plate_position=PLATE_POSITION,
        objective_id='10x Oly',
        stim_configs={},
    ),
    'delete_step': lambda p, i: p.delete_step(step_idx=i),
}


def _insert(protocol, **place):
    return protocol.insert_step(
        step_name='added',
        layer='BF',
        layer_config=LAYER_CONFIG,
        plate_position=PLATE_POSITION,
        objective_id='10x Oly',
        stim_configs={},
        **place,
    )


def _assert_refused_and_unchanged(protocol, write):
    before = protocol.steps()
    revision = protocol.step_list_revision

    with pytest.raises(ProtocolError):
        write(protocol)

    pd.testing.assert_frame_equal(protocol.steps(), before)
    assert protocol.num_steps() == len(before)
    assert protocol.step_list_revision == revision


@pytest.mark.parametrize('idx', OUTSIDE)
@pytest.mark.parametrize('writer', WRITERS)
def test_every_writer_refuses_a_step_index_outside_the_protocol(writer, idx):
    _assert_refused_and_unchanged(_protocol(), lambda p: WRITERS[writer](p, idx))


@pytest.mark.parametrize('writer', WRITERS)
def test_every_writer_takes_the_last_step(writer):
    # The positive control: the same call one index lower is admitted.
    protocol = _protocol()
    WRITERS[writer](protocol, NUM_STEPS - 1)


def test_a_frame_with_its_own_index_is_written_by_position():
    # A caller's frame (create_protocol(config=)) may carry any index. The
    # writers address a step by label and step() by position, so the frame
    # is renumbered 0..n-1 when it is taken, or index 1 passes the range
    # check and the write appends a row labelled 1.
    frame = pd.DataFrame(
        [_make_step(name=f'A{i + 1}_BF', well=f'A{i + 1}') for i in range(NUM_STEPS)],
        index=[5, 7, 9],
    )
    protocol = Protocol(
        tiling_configs_file_loc=TILING_CONFIGS,
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': frame,
            'period': datetime.timedelta(minutes=1),
            'duration': datetime.timedelta(hours=1),
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        },
    )

    protocol.modify_autofocus(step_idx=1, enabled=True)

    assert protocol.num_steps() == NUM_STEPS
    assert [protocol.step(idx=i)['Auto_Focus'] for i in range(NUM_STEPS)] == [False, True, False]


def test_deleting_from_an_empty_protocol_is_refused():
    _assert_refused_and_unchanged(_build_protocol([]), lambda p: p.delete_step(step_idx=0))


@pytest.mark.parametrize(
    'place',
    [
        {'after_step': NUM_STEPS},
        {'after_step': NUM_STEPS + 3},
        {'after_step': -2},
        {'before_step': NUM_STEPS + 1},
        {'before_step': -1},
    ],
    ids=str,
)
def test_an_insert_outside_the_protocol_is_refused(place):
    _assert_refused_and_unchanged(_protocol(), lambda p: _insert(p, **place))


@pytest.mark.parametrize(
    ('place', 'lands_at'),
    [
        ({'after_step': -1}, 0),
        ({'after_step': NUM_STEPS - 1}, NUM_STEPS),
        ({'before_step': 0}, 0),
        ({'before_step': NUM_STEPS}, NUM_STEPS),
    ],
    ids=str,
)
def test_an_insert_at_either_end_lands_there(place, lands_at):
    protocol = _protocol()
    name = _insert(protocol, **place)
    assert protocol.num_steps() == NUM_STEPS + 1
    assert protocol.step(idx=lands_at)['Name'] == name


@pytest.fixture
def session(tmp_path):
    # A manual scope: no axis to home before a step can record its place.
    settings = complete_settings(live_folder=str(tmp_path), microscope='LS620')
    for layer in config_helpers.get_layer_configs(settings):
        settings[layer]['acquire'] = None
    settings['BF']['acquire'] = 'image'
    session = ScopeSession.create(settings, simulate=True)
    yield session
    session.shutdown()
    session.scope.disconnect()


def _empty_protocol(session):
    return session.scope.protocols.create_protocol(
        empty_config=session.get_sequenced_capture_config()
    )


@pytest.mark.parametrize(
    'place',
    [{}, {'after_step': -1}],
    ids=['a script names no place', 'the GUI adds after no step'],
)
def test_the_first_step_of_an_empty_protocol_is_added(session, place):
    protocol = _empty_protocol(session)
    names = session.add_step(protocol, **place)
    assert protocol.num_steps() == 1
    assert list(protocol.steps()['Name']) == names
