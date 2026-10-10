# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A step added with no place given goes after the last step.

``ScopeSession.add_step(protocol)`` offers ``before_step`` and ``after_step``
as optional, but called with neither it raised "Must specify after_step or
before_step": a signature whose defaults were always refused. The GUI always
names a place, so only a script met it. A script building a protocol step by
step means "add" as "add at the end"; the steps land after the last one, in
channel order.
"""

import pytest

from modules import config_helpers
from modules.exceptions import ProtocolError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    # A manual scope: no axis to home before a step can record its place.
    settings = complete_settings(live_folder=str(tmp_path), microscope='LS620')
    for layer in config_helpers.get_layer_configs(settings):
        settings[layer]['acquire'] = None
    session = ScopeSession.create(settings, simulate=True)
    yield session
    session.shutdown()
    session.scope.disconnect()


def _protocol(session):
    return session.scope.protocols.create_protocol(
        empty_config=session.get_sequenced_capture_config()
    )


def test_steps_added_with_no_place_follow_the_last_step_in_channel_order(session):
    protocol = _protocol(session)
    session.settings['BF']['acquire'] = 'image'
    first = session.add_step(protocol)
    session.settings['Blue']['acquire'] = 'image'
    session.settings['step_channel_order'] = ['Blue', 'BF']

    second = session.add_step(protocol)

    assert list(protocol.steps()['Name']) == first + second
    assert [protocol.step(idx=i)['Color'] for i in range(3)] == ['BF', 'Blue', 'BF']


def test_the_protocols_own_insert_names_its_place(session):
    # The placement primitive has no default of its own: one that put a
    # step first would contradict the API's "add goes last".
    protocol = _protocol(session)
    session.settings['BF']['acquire'] = 'image'
    session.add_step(protocol)
    step = protocol.step(idx=0)

    with pytest.raises(ProtocolError):
        protocol.insert_step(
            step_name=None,
            layer='BF',
            layer_config=session.get_layer_configs()['BF'],
            plate_position={'x': step['X'], 'y': step['Y'], 'z': step['Z']},
            objective_id=step['Objective'],
            stim_configs=session.get_stim_configs(),
        )

    assert protocol.num_steps() == 1


def test_a_place_on_both_sides_is_still_refused(session):
    protocol = _protocol(session)
    session.settings['BF']['acquire'] = 'image'
    session.add_step(protocol)

    with pytest.raises(ProtocolError):
        session.add_step(protocol, before_step=0, after_step=0)

    assert protocol.num_steps() == 1
