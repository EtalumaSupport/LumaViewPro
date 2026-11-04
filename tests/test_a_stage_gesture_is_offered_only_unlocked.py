# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A gesture on the stage map or the viewer is offered only on a stage, with the controls unlocked.

The stage map's touch and the viewer's right-click are bound input
observers: they fire whatever any widget's disabled state says, so they ask
``ui_helpers.xy_gestures_offered`` first. It answers from the scope's XY
capability and the App's copy of the Session's lock record; the whole
surface's lock, so a live recording withholds them as it greys the stage.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

import modules.app_context as _app_ctx
from modules.scope_session import ControlsLock, ScopeSession
from tests.settings_fixtures import complete_settings
from ui import ui_helpers

UNLOCKED = ControlsLock(run_named=None, control_named=None, recording=False)


@pytest.fixture
def offered(tmp_path, monkeypatch):
    """Ask the helper on a simulated scope of ``model`` under ``lock``."""
    sessions = []

    def ask(model: str, lock: ControlsLock) -> bool:
        session = ScopeSession.create(
            complete_settings(microscope=model, live_folder=str(tmp_path)),
            simulate=True,
            warn_pre_release=False,
        )
        sessions.append(session)
        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(scope=session.scope))
        # a stand-in by design: the App exists only when lumaviewpro.py runs
        # as the program; the helper reads one attribute of it, the record.
        app = SimpleNamespace(lock=lock)
        monkeypatch.setattr(
            sys.modules['kivy.app'].App, 'get_running_app', mock.Mock(return_value=app)
        )
        return ui_helpers.xy_gestures_offered()

    yield ask
    for session in sessions:
        session.shutdown()


def test_a_stage_with_the_controls_unlocked_is_offered(offered):
    assert offered('LS850', UNLOCKED) is True


def test_a_scope_with_no_xy_stage_is_not_offered(offered):
    assert offered('LS820', UNLOCKED) is False


def test_a_live_recording_withholds_them(offered):
    assert (
        offered('LS850', ControlsLock(run_named=None, control_named=None, recording=True)) is False
    )


def test_a_run_withholds_them(offered):
    locked = ControlsLock(
        run_named='A protocol run is in progress.', control_named=None, recording=False
    )
    assert offered('LS850', locked) is False
