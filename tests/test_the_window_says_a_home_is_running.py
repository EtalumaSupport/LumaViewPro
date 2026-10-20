# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""While a home runs, the middle of the window says so.

A home greys every control, but the only word of it was a suffix in the
window title, so a person saw a greyed window and no reason. A banner over
the middle of the window -- "System Homing..." beside a turning arc -- is
shown while a home holds the scope, and at no other time: a run shows its
own progress, and a banner over the live view would hide what it captures.

The banner binds the Session's live work as of the last edge
(``app.live_work``): shown while it holds a home item.
"""

from __future__ import annotations

import pathlib
import threading

import pytest

from modules import live_work
from modules.scope_session import ScopeSession
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


def _a_home_is_named(session) -> bool:
    return session.live_work.has(live_work.HOME)


def test_the_banner_names_a_home_while_it_holds_the_scope(session, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real = session.scope._motion_driver.home

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(HOME_S), 'the held home was never released'
        return real(*args, **kwargs)

    monkeypatch.setattr(session.scope._motion_driver, 'home', held)
    assert _a_home_is_named(session) is False

    home = session.scope.motion.start_home('ALL')
    try:
        assert entered.wait(HOME_S)
        assert _a_home_is_named(session) is True
    finally:
        release.set()
        home.exception(timeout=HOME_S)
    assert _a_home_is_named(session) is False


@pytest.mark.parametrize('kind', ['protocol', 'diagnostic', 'recording'])
def test_no_other_holder_shows_it(session, kind):
    from tests.protocol_drives import run_identity

    held = session.activity_claim.try_claim(
        kind, run=run_identity() if kind == 'protocol' else None
    )
    try:
        assert _a_home_is_named(session) is False
    finally:
        held.release()


def test_the_banner_binds_the_home_in_the_live_work():
    kv = (pathlib.Path(__file__).resolve().parent.parent / 'ui' / 'lumaviewpro.kv').read_text()
    banner = kv[kv.index('HomingBanner:') :]
    banner = banner[: banner.index('pos_hint')]
    assert 'active: app.live_work.has(live_work.HOME)' in banner
    assert '#:import live_work modules.live_work' in kv
