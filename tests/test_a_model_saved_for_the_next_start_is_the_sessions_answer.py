# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A model saved for the next start is the Session's answer, not the GUI's comparison.

Before, Advanced Settings compared the saved model with the running one
itself and then raised its notice at info, which the popup bridge never
shows, so choosing an LS560 on a running LS620 said nothing on screen. Now
the Session says which model waits for the next start, and the GUI shows
that answer.
"""

from __future__ import annotations

import pytest

from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def ls620_session(tmp_path):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope='LS620'), simulate=True
    )
    yield session
    session.shutdown()


def test_nothing_waits_while_the_saved_model_is_the_one_running(ls620_session):
    assert ls620_session.model_at_next_start is None


def test_a_different_model_waits_for_the_next_start_until_the_running_one_is_chosen_again(
    ls620_session,
):
    ls620_session.select_model('LS560')
    assert ls620_session.model_at_next_start == 'LS560'
    assert ls620_session.scope.layer_identity.model == 'LS620'

    ls620_session.select_model('LS620')
    assert ls620_session.model_at_next_start is None
