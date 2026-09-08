# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""shutdown() tears down what the session owns, by the two ownership facts.

A headless host used to get a session whose shutdown() left the camera
streaming and the motion monitor alive: the hardware half of the
teardown (the LED drain through the io lane, motion stopped, the scope
disconnected) lived only in the GUI's close handler. Now shutdown() runs
that half for a scope a factory built, and leaves a caller's scope
alone, while the lane half follows the executor-ownership fact as
before. The rows below are the construction paths, each with the
teardown it must get.
"""

import atexit
import time
from unittest.mock import MagicMock

import pytest

import modules.scope_session as scope_session_module
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import SequentialIOExecutor
from tests.log_capture import capture_module_log, messages
from tests.scope_fakes import spec_scope
from tests.settings_fixtures import complete_settings


def _wait_dead(thread, seconds=2.0):
    end = time.monotonic() + seconds
    while thread is not None and thread.is_alive() and time.monotonic() < end:
        time.sleep(0.02)


def _record_unregister(monkeypatch):
    seen = []
    real = atexit.unregister

    def record(fn):
        seen.append(getattr(fn, '__qualname__', repr(fn)))
        return real(fn)

    monkeypatch.setattr(atexit, 'unregister', record)
    return seen


def _spy_lane(lane):
    """The order of LED-drain submissions and the lane's own shutdown."""
    events = []
    real_put, real_shutdown = lane.put, lane.shutdown

    def put(task, *args, **kwargs):
        events.append(('put', getattr(getattr(task, 'action', None), '__name__', '?')))
        return real_put(task, *args, **kwargs)

    def shutdown(*args, **kwargs):
        events.append(('shutdown',))
        return real_shutdown(*args, **kwargs)

    lane.put, lane.shutdown = put, shutdown
    return events


def _drain_before_lane_shutdown(events):
    puts = [i for i, e in enumerate(events) if e[0] == 'put' and e[1] == '_leds_off_impl']
    shuts = [i for i, e in enumerate(events) if e[0] == 'shutdown']
    return bool(puts) and bool(shuts) and puts[0] < shuts[0]


@pytest.fixture
def session_log(monkeypatch):
    return capture_module_log(monkeypatch, scope_session_module)


class TestAFactoryBuiltScopeIsTornDown:
    def test_over_the_factorys_own_bundle(self, tmp_path, monkeypatch):
        unregistered = _record_unregister(monkeypatch)
        session = ScopeSession.create_headless(
            settings=complete_settings(live_folder=str(tmp_path))
        )
        scope = session.scope
        stop_motion = MagicMock(wraps=scope.motion.stop_motion)
        monkeypatch.setattr(scope.motion, 'stop_motion', stop_motion)
        events = _spy_lane(session.executor_bundle.io_executor)
        try:
            session.shutdown()
            monitor = scope.motion._motion_monitor_thread
            _wait_dead(monitor)
            assert scope.imaging.is_streaming() is False, 'shutdown() left the camera streaming'
            assert monitor is None or not monitor.is_alive(), (
                'shutdown() left the motion monitor alive'
            )
            assert 'Lumascope._emergency_shutdown' in unregistered, (
                'the atexit hook is still registered'
            )
            assert scope.motor_connected is False
            stop_motion.assert_called()
            assert _drain_before_lane_shutdown(events), f'LED drain / lane order: {events}'
        finally:
            scope.disconnect()

    def test_over_caller_lanes_the_drain_runs_on_the_callers_io_lane(self, tmp_path, monkeypatch):
        unregistered = _record_unregister(monkeypatch)
        io, cam = (
            SequentialIOExecutor(name='IO_TEARDOWN'),
            SequentialIOExecutor(name='CAMERA_TEARDOWN'),
        )
        io.start()
        cam.start()
        session = ScopeSession.create(
            settings=complete_settings(live_folder=str(tmp_path)),
            io_executor=io,
            camera_executor=cam,
            simulate=True,
            warn_pre_release=False,
        )
        scope = session.scope
        stop_motion = MagicMock(wraps=scope.motion.stop_motion)
        monkeypatch.setattr(scope.motion, 'stop_motion', stop_motion)
        events = _spy_lane(io)
        try:
            assert session.executor_bundle is None and session._owns_executors is False
            session.shutdown()
            monitor = scope.motion._motion_monitor_thread
            _wait_dead(monitor)
            assert io.pending_shutdown and cam.pending_shutdown, "the caller's lanes are shut"
            assert io.worker_alive is False and cam.worker_alive is False, 'shut with wait'
            assert monitor is None or not monitor.is_alive()
            assert 'Lumascope._emergency_shutdown' in unregistered
            assert scope.motor_connected is False
            stop_motion.assert_called()
            assert _drain_before_lane_shutdown(events), (
                f"drain on the caller's io lane before it shuts: {events}"
            )
        finally:
            scope.disconnect()
            io.shutdown()
            cam.shutdown()

    def test_a_never_started_io_lane_skips_the_drain_without_waiting(self, tmp_path, monkeypatch):
        # The factory builds its own io lane beside a caller's camera lane
        # and never starts it: a submission there would sit unserviced
        # until the 2 s bound. The drain is skipped; disconnect() turns
        # the LEDs off inline.
        cam = SequentialIOExecutor(name='CAMERA_ONLY_TEARDOWN')
        cam.start()
        session = ScopeSession.create(
            settings=complete_settings(live_folder=str(tmp_path)),
            camera_executor=cam,
            simulate=True,
            warn_pre_release=False,
        )
        scope = session.scope
        assert session.io_executor.worker_alive is False
        events = _spy_lane(session.io_executor)
        try:
            started = time.monotonic()
            session.shutdown()
            elapsed = time.monotonic() - started
            assert not any(e[0] == 'put' for e in events), (
                f'the drain was submitted to a dead lane: {events}'
            )
            assert elapsed < 1.5, f'shutdown() waited on a lane with no worker ({elapsed:.2f}s)'
            assert scope.motor_connected is False, 'the hardware half still ran'
            assert scope.imaging.is_streaming() is False
        finally:
            scope.disconnect()
            cam.shutdown()

    def test_a_fenced_lane_is_logged_and_the_hardware_half_still_runs(
        self, tmp_path, monkeypatch, session_log
    ):
        io, cam = SequentialIOExecutor(name='IO_FENCED'), SequentialIOExecutor(name='CAMERA_FENCED')
        io.start()
        cam.start()
        session = ScopeSession.create(
            settings=complete_settings(live_folder=str(tmp_path)),
            io_executor=io,
            camera_executor=cam,
            simulate=True,
            warn_pre_release=False,
        )
        scope = session.scope
        monkeypatch.setattr(io, 'put', lambda *a, **k: None)
        try:
            session.shutdown()
            assert any('io lane refused the shutdown leds_off' in m for m in messages(session_log))
            assert scope.motor_connected is False
        finally:
            scope.disconnect()
            io.shutdown()
            cam.shutdown()

    def test_a_pass_that_raised_can_be_retried(self, tmp_path, monkeypatch, session_log):
        session = ScopeSession.create_headless(
            settings=complete_settings(live_folder=str(tmp_path))
        )
        scope = session.scope
        monkeypatch.setattr(
            scope.motion, 'stop_motion', MagicMock(side_effect=RuntimeError('bus gone'))
        )
        try:
            with pytest.raises(RuntimeError):
                session.shutdown()
            assert session._shut_down is False, 'a pass that raised is not a completed pass'
            monkeypatch.setattr(scope.motion, 'stop_motion', MagicMock())
            session.shutdown()
            assert session._shut_down is True
            assert scope.motor_connected is False
            assert not any('called again' in m for m in messages(session_log))
        finally:
            scope.disconnect()


class TestACallersScopeIsLeftAlone:
    def _caller_session(self, **kwargs):
        scope = spec_scope()
        session = ScopeSession.create(
            settings=complete_settings(),
            scope=scope,
            io_executor=MagicMock(),
            camera_executor=MagicMock(),
            **kwargs,
        )
        return session, scope

    def test_a_passed_scope_is_untouched(self):
        session, scope = self._caller_session()
        session.shutdown()
        scope.disconnect.assert_not_called()
        scope.motion.stop_motion.assert_not_called()
        scope.illumination._leds_off_impl.assert_not_called()
        session.io_executor.put.assert_not_called()
        session.io_executor.shutdown.assert_called_once()
        session.camera_executor.shutdown.assert_called_once()

    def test_a_directly_constructed_session_is_the_same_row(self):
        scope = spec_scope()
        session = ScopeSession(
            settings=complete_settings(),
            scope=scope,
            io_executor=MagicMock(),
            camera_executor=MagicMock(),
        )
        session.shutdown()
        scope.disconnect.assert_not_called()
        scope.motion.stop_motion.assert_not_called()
        session.io_executor.put.assert_not_called()

    def test_after_set_scope_neither_scope_is_disconnected(self, tmp_path):
        session = ScopeSession.create(
            settings=complete_settings(live_folder=str(tmp_path)),
            simulate=True,
            warn_pre_release=False,
        )
        built = session.scope
        swapped_in = spec_scope()
        try:
            session.set_scope(swapped_in)
            assert session._owns_scope is False
            session.shutdown()
            swapped_in.disconnect.assert_not_called()
            swapped_in.motion.stop_motion.assert_not_called()
            assert built.motor_connected is True, "the built scope is the caller's after the swap"
        finally:
            built.disconnect()

    def test_a_second_shutdown_logs_and_touches_nothing(self, session_log):
        session, _scope = self._caller_session()
        session.shutdown()
        io_calls = session.io_executor.shutdown.call_count
        session.shutdown()
        assert session.io_executor.shutdown.call_count == io_calls
        assert any('shutdown() called again' in m for m in messages(session_log)), (
            'the second shutdown() must log its no-op'
        )
