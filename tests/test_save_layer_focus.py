# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Z a caller holds is saved as a layer's focus through one door, and only one Z can reach.

The Autofocus button wrote the Z its run chose straight into the GUI's
settings dict, past the Session, with no check that the number was one this
scope's Z could reach, though every new step of the layer is born at it.
``ScopeSession.set_channel_settings`` is the door for a caller holding a number;
``save_focus`` and ``apply_focus_to_channel_steps`` write through the same
path.
"""

import math

import pytest

from _lumascope.modules.exceptions import (
    HardwareCommandRefusedError,
    MissingPart,
    PositionOutOfRangeError,
    ProtocolRunRefusedError,
    SettingRefusedError,
)
from _lumascope.modules.scope_parts import Part
from _lumascope.modules.scope_session import ScopeSession
from tests.ast_seams import direct_call_names, find_def
from tests.settings_fixtures import complete_settings


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path_factory.mktemp('live'))), simulate=True
    )
    yield s
    s.shutdown()


def _z_limits(session):
    limits = session.scope.motion.get_axis_limits('Z')
    return limits.min_um, limits.max_um


def test_a_z_inside_the_travel_is_stored_as_the_layers_focus(session):
    low, high = _z_limits(session)
    z = (low + high) / 2

    session.set_channel_settings('BF', focus_z_um=z)

    assert session.settings['BF']['focus'] == z


@pytest.mark.parametrize('kind', ['below', 'above'])
def test_a_z_the_focus_cannot_reach_is_refused_and_nothing_written(session, kind):
    low, high = _z_limits(session)
    session.set_channel_settings('BF', focus_z_um=(low + high) / 2)
    before = session.settings['BF']['focus']
    z = {'below': low - 1.0, 'above': high + 1.0}[kind]

    with pytest.raises(PositionOutOfRangeError) as refused:
        session.set_channel_settings('BF', focus_z_um=z)

    assert refused.value.axis == 'Z'
    assert session.settings['BF']['focus'] == before


@pytest.mark.parametrize('z', [math.nan, math.inf])
def test_a_z_that_is_not_a_number_is_refused_as_one_and_nothing_written(session, z):
    low, high = _z_limits(session)
    session.set_channel_settings('BF', focus_z_um=(low + high) / 2)
    before = session.settings['BF']['focus']

    with pytest.raises(SettingRefusedError) as refused:
        session.set_channel_settings('BF', focus_z_um=z)

    assert (refused.value.reason, refused.value.path) == ('wrong_kind', 'BF.focus')
    assert session.settings['BF']['focus'] == before


def test_a_layer_this_scope_does_not_have_is_refused(session):
    with pytest.raises(HardwareCommandRefusedError) as refused:
        session.set_channel_settings('Lumi', focus_z_um=1000.0)
    assert refused.value.missing == MissingPart(Part.CHANNEL, channel='Lumi')


def test_a_scope_with_no_z_refuses_a_focus(tmp_path):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS620'), simulate=True
    )
    try:
        assert s.scope.capabilities.has_focus is False
        with pytest.raises(ProtocolRunRefusedError) as refused:
            s.set_channel_settings('BF', focus_z_um=1000.0)
        assert refused.value.reason == 'positions_unreachable'
    finally:
        s.shutdown()


def test_save_focus_and_apply_to_steps_write_through_the_one_path():
    for member in ('save_focus', 'apply_focus_to_channel_steps'):
        node = find_def(
            'src/_lumascope/modules/scope_session.py', member, class_name='ScopeSession'
        )
        assert '_store_channel_focus' in direct_call_names(node), member
