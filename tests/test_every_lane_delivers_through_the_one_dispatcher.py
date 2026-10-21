# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every lane of a session hands its callbacks to the process's one UI dispatcher.

A session's file, post-processing, worker-pool and diagnostics lanes were
once built with no dispatcher when its scope was built elsewhere, so on the
GUI host those lanes ran their callbacks on the worker while the scope's own
delivered on the UI thread. Each lane reads the one store at the moment it
dispatches.
"""

from __future__ import annotations

import threading

import pytest

from modules.kivy_utils import UiDispatcher
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import IOTask
from tests.scope_fakes import build_scope
from tests.settings_fixtures import complete_settings

LANES = (
    'io_executor',
    'camera_executor',
    'file_io_executor',
    'post_processing_executor',
    'worker_pool',
    'diagnostics_executor',
)


@pytest.fixture
def scheduled():
    seen = []

    def schedule(func, timeout):
        seen.append(func)
        func(timeout)

    ScopeSession.set_ui_dispatcher(UiDispatcher(schedule=schedule, thread=None))
    yield seen
    ScopeSession.set_ui_dispatcher(None)


def _delivered_through_the_store(lane, scheduled) -> bool:
    ran = threading.Event()
    before = len(scheduled)
    lane.put(IOTask(action=lambda: None, callback=ran.set))
    assert ran.wait(5.0), f'{lane.executor_name} never ran its callback'
    return len(scheduled) > before


@pytest.mark.parametrize('lane_name', LANES)
def test_a_session_dispatches_every_lane_through_the_store(lane_name, scheduled, tmp_path):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path)), simulate=True, warn_pre_release=False
    )
    try:
        lane = getattr(session.executor_bundle, lane_name)
        assert _delivered_through_the_store(lane, scheduled), (
            f'{lane.executor_name} ran its callback without the UI dispatcher'
        )
    finally:
        session.shutdown()


def test_with_no_dispatcher_a_lane_calls_its_callback_on_the_worker(tmp_path):
    scope = build_scope(simulate=True, warn_pre_release=False)
    try:
        seen = {}
        ran = threading.Event()

        def callback():
            seen['thread'] = threading.current_thread()
            ran.set()

        scope.io_lane().put(IOTask(action=lambda: None, callback=callback))
        assert ran.wait(5.0)
        assert seen['thread'] is not threading.main_thread()
    finally:
        scope.disconnect()
