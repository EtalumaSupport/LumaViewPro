# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""While a home runs, the middle of the window says so.

A home greys every control, but the only word of it was a suffix in the
window title, so a person saw a greyed window and no reason. A banner over
the live view now carries the Session's own sentence for the hold while a
home holds the scope, and nothing at any other time: a run shows its own
progress, and a banner over the live view would hide what it captures.
"""

from __future__ import annotations

import ast
import threading

import pytest

from modules.scope_session import ScopeSession
from tests.ast_seams import find_def
from tests.settings_fixtures import complete_settings

HOME_S = 60.0


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(
        complete_settings(microscope='LS850', live_folder=str(tmp_path)),
        simulate=True,
        warn_pre_release=False,
    )
    try:
        yield s
    finally:
        s.shutdown()


def test_the_banner_names_a_home_while_it_holds_the_scope(session, monkeypatch):
    from ui.ui_helpers import homing_banner_text

    entered, release = threading.Event(), threading.Event()
    real = session.scope._motion_driver.home

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(HOME_S), 'the held home was never released'
        return real(*args, **kwargs)

    monkeypatch.setattr(session.scope._motion_driver, 'home', held)
    assert homing_banner_text(session) == ''

    home = session.scope.motion.start_home('ALL')
    try:
        assert entered.wait(HOME_S)
        assert homing_banner_text(session) == 'A home is in progress.'
    finally:
        release.set()
        home.exception(timeout=HOME_S)
    assert homing_banner_text(session) == ''


@pytest.mark.parametrize('kind', ['protocol', 'diagnostic', 'recording'])
def test_no_other_holder_shows_it(session, kind):
    from tests.protocol_drives import run_identity
    from ui.ui_helpers import homing_banner_text

    held = session.activity_claim.try_claim(
        kind, run=run_identity() if kind == 'protocol' else None
    )
    try:
        assert homing_banner_text(session) == ''
    finally:
        held.release()


def test_the_run_state_publisher_writes_the_banner():
    publish = find_def('lumaviewpro.py', 'publish_run_state', class_name='LumaViewProApp')
    assert publish is not None, 'LumaViewProApp.publish_run_state is gone'
    writes = [
        ast.unparse(node.value)
        for node in ast.walk(publish)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == 'homing_text' for t in node.targets)
    ]
    assert writes == ['homing_banner_text(session)'], writes
