# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run whose steps lie outside the stage's travel is refused under its own reason.

The travel check was folded into ``validate_for_run``'s list of strings and
refused as ``validation_failed``, beside a misspelled labware name and a
duplicate filename, so a caller could not tell "this step is past the end
of Z" from "this file is malformed" without reading prose. It is now the
protocols API's, asked by ``prepare()`` after the connection and objective
gates, refused as ``positions_outside_travel`` naming the steps and the
axis.

The offset it judges with is the scope's (``runtime_state``), the one every
coordinate transform reads; a session's settings carrying no offset are
refused at creation instead of running at an invented one.
"""

from __future__ import annotations

import pytest

from modules.exceptions import ConfigError, ProtocolRunRefusedError
from modules.protocol import ProtocolFormatError
from modules.scope_session import ScopeSession
from tests.scope_fakes import bind_settings_like_a_session
from tests.settings_fixtures import complete_settings_without
from tests.test_a_protocol_needs_its_objectives_on_the_turret import (
    NOT_CARRIED,
    NOT_ON_TURRET,
    _prepare,
    _turret,
)
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_multi_step_protocol,
    executor,
    executors,
    scope,
)

OUTSIDE = 'positions_outside_travel'


def _past_z(scope) -> float:
    return scope.motion.get_axis_limits('Z')['max'] + 500.0


def _refusal(executor, protocol, tmp_path) -> ProtocolRunRefusedError:
    with pytest.raises(ProtocolRunRefusedError) as refusal:
        _prepare(executor, protocol, tmp_path)
    return refusal.value


def test_a_step_past_the_end_of_z_is_refused_naming_the_step_and_the_axis(
    executor, scope, tmp_path
):
    protocol = _make_multi_step_protocol(
        [{'name': 'in_range'}, {'name': 'too_high', 'z': _past_z(scope)}]
    )
    refusal = _refusal(executor, protocol, tmp_path)
    assert refusal.reason == OUTSIDE, refusal
    assert 'too_high' in str(refusal) and 'Z' in str(refusal), str(refusal)
    assert 'in_range' not in str(refusal), str(refusal)


def test_a_step_off_the_stage_in_x_is_refused_naming_the_axis(executor, scope, tmp_path):
    protocol = _make_multi_step_protocol([{'name': 'off_plate', 'x': -500.0}])
    refusal = _refusal(executor, protocol, tmp_path)
    assert refusal.reason == OUTSIDE, refusal
    assert 'off_plate' in str(refusal) and 'X' in str(refusal), str(refusal)


def test_a_whole_stack_past_z_is_one_refusal_that_counts_it(executor, scope, tmp_path):
    # The z-stack a hand-edited or older file carries past the end of Z
    # (the census's K73): every slice is outside, and the refusal says how
    # many rather than listing each in a validation summary.
    past = _past_z(scope)
    protocol = _make_multi_step_protocol([{'name': f'slice_{i}', 'z': past + i} for i in range(12)])
    refusal = _refusal(executor, protocol, tmp_path)
    assert refusal.reason == OUTSIDE, refusal
    assert '12' in str(refusal), str(refusal)


def test_the_offset_judged_is_the_scopes(executor, scope, tmp_path):
    # Inside the travel at a zero offset; moved past the X travel by the
    # offset the scope carries. A run judged (and driven) at an offset of
    # its own would admit it and image somewhere else.
    protocol = _make_multi_step_protocol([{'name': 'shifted'}])
    _prepare(executor, protocol, tmp_path)
    bind_settings_like_a_session(scope, stage_offset={'x': -50_000.0, 'y': 0.0})
    refusal = _refusal(executor, protocol, tmp_path)
    assert refusal.reason == OUTSIDE, refusal


def test_a_blank_x_never_reaches_the_travel_check():
    # A position that is not a number refuses the protocol where it is built,
    # so the travel check only ever reads numbers.
    with pytest.raises(ProtocolFormatError, match='X'):
        _make_multi_step_protocol([{'name': 'blank', 'x': ''}])


def test_a_disconnected_scope_is_told_it_is_disconnected_first(
    executor, scope, tmp_path, monkeypatch
):
    monkeypatch.setattr(scope, 'unconnected_parts', lambda: ('camera',))
    protocol = _make_multi_step_protocol([{'name': 'too_high', 'z': _past_z(scope)}])
    assert _refusal(executor, protocol, tmp_path).reason == 'hardware_disconnected'


def test_glass_the_turret_does_not_carry_is_told_first(executor, scope, tmp_path, monkeypatch):
    _turret(scope, monkeypatch)
    protocol = _make_multi_step_protocol(
        [{'name': 'too_high', 'z': _past_z(scope), 'objective': NOT_ON_TURRET}]
    )
    assert _refusal(executor, protocol, tmp_path).reason == NOT_CARRIED


def test_settings_without_a_stage_offset_cannot_make_a_session():
    with pytest.raises(ConfigError, match='stage_offset'):
        ScopeSession.create(complete_settings_without('stage_offset'), simulate=True)
