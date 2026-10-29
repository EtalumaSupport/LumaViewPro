# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The GUI's app context hands out the session's scope, runners and lanes, never copies.

``AppContext.over_session`` is the one builder: the GUI host builds its
context through it, and so does every test that puts a panel over a
simulated session.
"""

from types import SimpleNamespace

import pytest

from modules.app_context import AppContext


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    built = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield built
    built.shutdown()


def test_every_handle_the_context_holds_is_the_sessions_object(session):
    ctx = AppContext.over_session(session)

    bundle = session.executor_bundle
    held = {
        'scope': session.scope,
        'session': session,
        'sequenced_capture_runner': session.sequenced_capture_runner,
        'autofocus_runner': session.autofocus_runner,
        'io_executor': session.io_executor,
        'camera_executor': session.camera_executor,
        'protocol_thread': session.protocol_thread,
        'file_io_executor': session.file_io_executor,
        'post_processing_executor': session.post_processing.lane,
        'autofocus_thread': session.autofocus_thread,
        'scope_display_thread': bundle.scope_display_thread,
        'worker_pool': bundle.worker_pool,
    }
    wrong = [name for name, value in held.items() if getattr(ctx, name) is not value]
    assert not wrong, f"not the session's object: {wrong}"
    assert all(value is not None for value in held.values()), 'a simulated session holds them all'


def test_the_host_s_own_fields_reach_the_context(session):
    stage = SimpleNamespace()

    ctx = AppContext.over_session(session, stage=stage, version='4.0.0', live_view_fps=12)

    assert (ctx.stage, ctx.version, ctx.live_view_fps) == (stage, '4.0.0', 12)
    assert ctx.settings is session.settings
