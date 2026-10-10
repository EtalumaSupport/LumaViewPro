# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Z a caller holds is saved as a layer's focus through one door, and only one Z can reach.

The Autofocus button wrote the Z its run chose straight into the GUI's
settings dict, past the Session, with no check that the number was one this
scope's Z could reach, though every new step of the layer is born at it.
``ScopeSession.save_layer_focus`` is the door for a caller holding a number;
``save_focus`` and ``apply_focus_to_layer_steps`` write through the same
path, and ``update_settings`` on a focus names it.
"""

import ast
import math

import pytest

from modules.exceptions import (
    ArgumentRefusedError,
    HardwareCommandRefusedError,
    MissingPart,
    PositionOutOfRangeError,
    ProtocolRunRefusedError,
)
from modules.scope_session import ScopeSession
from tests.ast_seams import direct_call_names, find_def, parse_module, walk_defs
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
    return limits['min'], limits['max']


def test_a_z_inside_the_travel_is_stored_as_the_layers_focus(session):
    low, high = _z_limits(session)
    z = (low + high) / 2

    session.save_layer_focus('BF', z)

    assert session.settings['BF']['focus'] == z


@pytest.mark.parametrize('kind', ['below', 'above'])
def test_a_z_the_focus_cannot_reach_is_refused_and_nothing_written(session, kind):
    low, high = _z_limits(session)
    session.save_layer_focus('BF', (low + high) / 2)
    before = session.settings['BF']['focus']
    z = {'below': low - 1.0, 'above': high + 1.0}[kind]

    with pytest.raises(PositionOutOfRangeError) as refused:
        session.save_layer_focus('BF', z)

    assert refused.value.axis == 'Z'
    assert session.settings['BF']['focus'] == before


@pytest.mark.parametrize('z', [math.nan, math.inf])
def test_a_z_that_is_not_a_number_is_refused_as_one_and_nothing_written(session, z):
    low, high = _z_limits(session)
    session.save_layer_focus('BF', (low + high) / 2)
    before = session.settings['BF']['focus']

    with pytest.raises(ArgumentRefusedError) as refused:
        session.save_layer_focus('BF', z)

    assert (refused.value.reason, refused.value.argument) == ('not_a_number', 'z_um')
    assert session.settings['BF']['focus'] == before


def test_a_layer_this_scope_does_not_have_is_refused(session):
    with pytest.raises(HardwareCommandRefusedError) as refused:
        session.save_layer_focus('Lumi', 1000.0)
    assert refused.value.missing == MissingPart.layer('Lumi')


def test_a_scope_with_no_z_refuses_a_focus(tmp_path):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS620'), simulate=True
    )
    try:
        assert s.scope.capabilities.has_focus is False
        with pytest.raises(ProtocolRunRefusedError) as refused:
            s.save_layer_focus('BF', 1000.0)
        assert refused.value.reason == 'positions_unreachable'
    finally:
        s.shutdown()


def test_save_focus_and_apply_to_steps_write_through_the_one_path():
    for member in ('save_focus', 'apply_focus_to_layer_steps', 'save_layer_focus'):
        node = find_def('modules/scope_session.py', member, class_name='ScopeSession')
        assert '_store_layer_focus' in direct_call_names(node), member


def test_no_other_member_writes_a_layers_focus():
    # save_all_bookmarks stores the bookmark and every layer's focus under
    # one hold of the lock, from the live Z focus_z has already checked.
    allowed = {'ScopeSession._store_layer_focus', 'ScopeSession.save_all_bookmarks'}
    writers = set()
    for qualname, fn in walk_defs(parse_module('modules/scope_session.py').body):
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, 'attr', None) == '_store_setting'
                and node.args
                and isinstance(node.args[0], ast.JoinedStr)
                and isinstance(node.args[0].values[-1], ast.Constant)
                and str(node.args[0].values[-1].value).endswith('.focus')
            ):
                writers.add(qualname)
    assert writers == allowed, writers


def test_update_settings_on_a_focus_names_the_door(session):
    from modules.exceptions import SettingRefusedError

    with pytest.raises(SettingRefusedError) as refused:
        session.update_settings('BF.focus', 1000.0)
    assert refused.value.member == 'save_layer_focus'
    assert callable(ScopeSession.save_layer_focus)
