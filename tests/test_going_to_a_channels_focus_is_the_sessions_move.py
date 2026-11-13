# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Going to a channel's saved focus is one Session member, refused when none was saved.

The GUI's Go To Focus read the saved focus and moved Z there itself, so a
REST or SDK caller had no door that refuses a focus never saved: reading
``channel_settings`` answers None for it, leaving the caller to decide.
``ScopeSession.start_go_to_channel_focus`` (and its waiting form
``go_to_channel_focus``) reads the focus and moves Z, refusing before
anything moves.
"""

from __future__ import annotations

import pytest

from _lumascope.modules.axes import StageAxis
from _lumascope.modules.exceptions import ArgumentRefusedError, FocusNotSavedError
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0

_FOCUS = 7000.0


@pytest.fixture
def session(tmp_path):
    settings = complete_settings(live_folder=str(tmp_path), microscope='LS850')
    settings['BF']['focus'] = _FOCUS
    settings[LED0]['focus'] = None
    built = ScopeSession.create(settings, simulate=True)
    home_sim_scope(built.scope)
    yield built
    built.shutdown()


def _z(session) -> float:
    return session.scope.motion.get_current_position(StageAxis.Z)


def test_going_to_a_channels_focus_moves_z_to_its_saved_focus(session):
    session.start_go_to_channel_focus('BF').result(timeout=30.0)

    assert _z(session) == pytest.approx(_FOCUS, abs=0.5)


def test_the_waiting_form_returns_once_z_has_arrived(session):
    session.go_to_channel_focus('BF')

    assert _z(session) == pytest.approx(_FOCUS, abs=0.5)


def test_a_channel_whose_focus_was_never_saved_is_refused_and_nothing_moves(session):
    before = _z(session)

    with pytest.raises(FocusNotSavedError) as refused:
        session.start_go_to_channel_focus(LED0)

    assert refused.value.reason == 'focus_not_saved'
    assert refused.value.channel == LED0
    assert _z(session) == before


def test_a_name_that_is_no_channel_is_refused(session):
    with pytest.raises(ArgumentRefusedError) as refused:
        session.start_go_to_channel_focus('Ex999')

    assert refused.value.reason == 'channel_unknown'
