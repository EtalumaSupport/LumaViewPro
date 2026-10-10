# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run needs a camera that is delivering frames, and waits out a change in flight.

A camera that is open but not grabbing delivers no frame. A run admitted
on one used to start, capture nothing, and end ``incomplete`` /
``captures_failed`` as a fault -- a client's stopped feed reported as the
instrument failing. The run is refused ``camera_not_streaming`` before it
starts. A frame-size or binning change stops the grab and starts it again
on the camera lane, so the gate asks again behind the lane's queued
commands before it refuses, and a streaming camera is never asked twice.

Driven through the Session on the simulated scope.
"""

import threading
import time
from concurrent.futures import Future

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import _run, _settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


def test_a_run_on_a_stopped_feed_is_refused_and_runs_once_the_feed_starts(session, tmp_path):
    session.scope.imaging.stop_streaming()

    with pytest.raises(ProtocolRunRefusedError) as refused:
        _run(session, tmp_path)
    assert refused.value.reason == 'camera_not_streaming'

    session.scope.imaging.start_streaming()
    assert _run(session, tmp_path).status == 'completed'


def test_a_run_asked_for_while_the_grab_restarts_on_the_lane_is_admitted(session, tmp_path):
    driver = session.scope.imaging._driver
    stopped = threading.Event()

    def regrab():
        driver.stop_grabbing()
        stopped.set()
        time.sleep(0.5)
        driver.start_grabbing()

    waiter = Future()
    waiter.set_running_or_notify_cancel()
    session.scope.imaging._submit_camera(regrab, 'regrab', waiter=waiter)
    assert stopped.wait(5), 'the lane never ran the regrab'
    assert session.scope.imaging.is_streaming() is False

    assert _run(session, tmp_path).status == 'completed'
    waiter.result(timeout=5)


def test_a_streaming_camera_is_answered_without_the_lane(session, monkeypatch):
    def no_lane(*args, **kwargs):
        raise AssertionError('a streaming camera was asked on the lane')

    monkeypatch.setattr(session.scope.imaging, '_dispatch_camera', no_lane)

    assert session.scope.imaging.is_streaming_once_settled() is True
