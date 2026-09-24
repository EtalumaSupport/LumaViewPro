# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A lane has one worker: starting a running lane again is refused.

`start()` spawned a new worker thread on every call. The session factory
already starts every lane it builds, and callers that started them again
got two workers on one lane, so two tasks on the camera lane ran at the
same moment -- the ordering every camera body relies on was gone, with
nothing to say so. A lane is now started once; a second start raises.
"""

import threading

import pytest

from modules.sequential_io_executor import IOTask, SequentialIOExecutor


def _lane(name):
    lane = SequentialIOExecutor(name=name)
    lane.start()
    return lane


def test_a_second_start_on_a_running_lane_raises():
    lane = _lane('ONCE')
    try:
        with pytest.raises(RuntimeError, match='already running'):
            lane.start()
    finally:
        lane.shutdown()


def test_a_refused_second_start_leaves_one_worker():
    lane = _lane('ONE_WORKER')
    held = threading.Event()
    release = threading.Event()
    second_ran = threading.Event()
    try:
        with pytest.raises(RuntimeError):
            lane.start()

        def hold():
            held.set()
            release.wait(5.0)

        lane.put(IOTask(action=hold))
        assert held.wait(5.0)
        lane.put(IOTask(action=second_ran.set))

        assert not second_ran.wait(0.3), 'a second worker ran a task beside the held one'
        release.set()
        assert second_ran.wait(5.0)
    finally:
        release.set()
        lane.shutdown()


def test_the_factory_starts_each_lane_once():
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(complete_settings(), simulate=True)
    try:
        for lane in (session.io_executor, session.camera_executor):
            assert lane.worker_alive, f'{lane.executor_name} was not started by the factory'
            with pytest.raises(RuntimeError, match='already running'):
                lane.start()
    finally:
        session.shutdown()


def test_the_session_has_no_second_start():
    from modules.scope_session import ScopeSession

    assert not hasattr(ScopeSession, 'start_executors'), (
        'the factory starts the lanes it builds; a second start door is how '
        'callers came to put two workers on one lane'
    )
