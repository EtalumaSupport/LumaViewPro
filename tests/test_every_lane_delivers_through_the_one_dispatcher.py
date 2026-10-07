# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every lane of a session hands its callbacks to the process's one UI dispatcher.

A session built over a caller's scope used to build its file,
post-processing, worker-pool and diagnostics lanes with no dispatcher, so on
the GUI host those lanes ran their callbacks on the worker while the scope's
own delivered on the UI thread. Each lane now reads the one store at the
moment it dispatches, whoever built the scope.
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
def test_a_session_over_a_callers_scope_dispatches_every_lane_through_the_store(
    lane_name, scheduled, tmp_path
):
    scope = build_scope(simulate=True, warn_pre_release=False)
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), scope=scope)
    try:
        lane = getattr(session.executor_bundle, lane_name)
        assert _delivered_through_the_store(lane, scheduled), (
            f'{lane.executor_name} ran its callback without the UI dispatcher'
        )
    finally:
        session.shutdown()
        scope.disconnect()


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
