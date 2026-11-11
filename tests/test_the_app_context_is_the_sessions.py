# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The GUI's app context hands out the session's scope and lanes, never copies.

``AppContext.over_session`` is the one builder: the GUI host builds its
context through it, and so does every test that puts a panel over a
simulated session.
"""

import dataclasses
from types import SimpleNamespace

import pytest

from _lumascope.modules.app_context import AppContext

# The session's handles no GUI code reads, which the context once copied.
_SESSION_HANDLES_NO_GUI_CODE_READS = {
    'sequenced_capture_runner',
    'autofocus_runner',
    'protocol_thread',
    'file_io_executor',
    'post_processing_executor',
    'autofocus_thread',
}


@pytest.fixture
def session(tmp_path):
    from _lumascope.modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    built = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield built
    built.shutdown()


def test_the_context_hands_out_the_sessions_handles_and_holds_none(session):
    """Each handle the GUI reads off the context is the session's object, read
    off the session as it is asked for; none is a field the context holds, so
    none can be replaced on the context or outlive what it was copied from."""
    ctx = AppContext.over_session(session)

    bundle = session.executor_bundle
    handed_out = {
        'scope': session.scope,
        'io_executor': session.io_executor,
        'camera_executor': session.camera_executor,
        'scope_display_thread': bundle.scope_display_thread,
        'worker_pool': bundle.worker_pool,
    }
    wrong = [name for name, value in handed_out.items() if getattr(ctx, name) is not value]
    assert not wrong, f"not the session's object: {wrong}"
    assert all(value is not None for value in handed_out.values()), (
        'a simulated session holds them all'
    )
    held = {f.name for f in dataclasses.fields(AppContext)}
    copies = sorted(held & (set(handed_out) | _SESSION_HANDLES_NO_GUI_CODE_READS))
    assert not copies, f'the context holds a copy of a session handle: {copies}'
    with pytest.raises(AttributeError):
        ctx.scope = session.scope


def test_a_context_with_no_session_reaches_no_handle():
    with pytest.raises(AttributeError, match='no session'):
        _ = AppContext().scope


def test_the_host_s_own_fields_reach_the_context(session):
    stage = SimpleNamespace()

    ctx = AppContext.over_session(session, stage=stage, version='4.0.0', live_view_fps=12)

    assert (ctx.stage, ctx.version, ctx.live_view_fps) == (stage, '4.0.0', 12)
    session.update_settings('video.max_fps', 14)
    assert ctx.settings['video']['max_fps'] == 14
