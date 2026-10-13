# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run-state listener can be removed, so a client that leaves stops being told.

Before, ``add_run_state_listener`` had no remove: the REST stream, which
adds one listener per member each time the app opens, removes each through
``remove_<name>_listener`` when it closes, found none for run state, and
left its listener behind on every app lifespan. A removed listener is not
told again; removing one never added is a no-op, as for outcome listeners.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path / 'live')), simulate=True)
    yield s
    s.shutdown()


def test_a_removed_listener_is_not_told_again(session):
    told = []

    def listener():
        told.append(1)

    session.add_run_state_listener(listener)
    told.clear()

    session.remove_run_state_listener(listener)
    session.notify_run_state()

    assert told == []


def test_removing_a_listener_never_added_is_a_no_op(session):
    session.remove_run_state_listener(lambda: None)


def test_three_app_lifespans_leave_no_run_state_listener_behind(session):
    # The store has no public read; its length is the leak under test.
    before = len(session._run_state_listeners)

    for _ in range(3):
        with TestClient(build_app(session)):
            pass

    assert len(session._run_state_listeners) == before
