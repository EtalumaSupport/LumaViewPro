# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The GUI's STOP sends ``stop_motion`` at once, whatever the GUI is waiting on.

A press runs on a thread of its own: never the GUI thread, which a board
that has stopped answering would freeze for the STOP's whole exchange; never
the IO lane or the worker pool, which may be held by the very work the
person is stopping. Every press sends, and its outcome goes to the GUI's one
reporter. The button's place and colour are the kv's, walked in the
simulator.
"""

from __future__ import annotations

import threading
import types

import pytest

import modules.app_context as _app_ctx
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import IOTask
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from ui import ui_helpers

_WAIT_S = 5.0


@pytest.fixture
def session(tmp_path, monkeypatch):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        monkeypatch.setattr(
            _app_ctx,
            'ctx',
            types.SimpleNamespace(
                scope=s.scope,
                io_executor=s.scope._io_executor,
                worker_pool=s.executor_bundle.worker_pool,
            ),
        )
        yield s
    finally:
        s.shutdown()


def _hold(executor):
    """Keep *executor* busy until the returned event is set."""
    entered, release = threading.Event(), threading.Event()

    def busy():
        entered.set()
        release.wait(_WAIT_S)

    executor.put(IOTask(action=busy))
    assert entered.wait(_WAIT_S)
    return release


def test_a_press_sends_stop_off_the_gui_thread_with_the_lane_and_pool_held(session):
    pressed = []
    told = threading.Event()

    def heard():
        pressed.append(threading.current_thread() is threading.main_thread())
        told.set()

    session.scope.motion.add_stop_press_listener(heard)
    lane_held = _hold(session.scope._io_executor)
    pool_held = _hold(session.executor_bundle.worker_pool)
    try:
        ui_helpers.stop_the_stage()
        assert told.wait(_WAIT_S), 'the STOP waited behind the lane or the pool'
    finally:
        lane_held.set()
        pool_held.set()

    assert pressed == [False], 'the STOP was sent on the GUI thread'


def test_every_press_sends(session):
    presses = threading.Semaphore(0)
    session.scope.motion.add_stop_press_listener(presses.release)

    ui_helpers.stop_the_stage()
    ui_helpers.stop_the_stage()

    assert presses.acquire(timeout=_WAIT_S)
    assert presses.acquire(timeout=_WAIT_S), 'a second press was not sent'
