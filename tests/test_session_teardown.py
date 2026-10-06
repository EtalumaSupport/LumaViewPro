# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""shutdown() tears down what the session owns, by the two ownership facts.

A headless host used to get a session whose shutdown() left the camera
streaming and the motion monitor alive: the hardware half of the
teardown (the LED drain through the io lane, motion stopped, the scope
disconnected) lived only in the GUI's close handler. Now shutdown() runs
that half for a scope a factory built, and leaves a caller's scope
alone, lanes running. The bundle the session holds always stops. The
rows below are the construction paths, each with the teardown it must
get.
"""

import atexit
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import modules.scope_session as scope_session_module
from modules.exceptions import ScopeDisconnectError
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import IOTask
from tests.log_capture import capture_module_log, messages
from tests.scope_fakes import build_scope, spec_scope
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
        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
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

    def test_a_fenced_lane_is_logged_and_the_hardware_half_still_runs(
        self, tmp_path, monkeypatch, session_log
    ):
        session = ScopeSession.create(
            settings=complete_settings(live_folder=str(tmp_path)),
            simulate=True,
            warn_pre_release=False,
        )
        scope = session.scope
        monkeypatch.setattr(session.io_executor, 'put', lambda *a, **k: None)
        try:
            session.shutdown()
            assert any('io lane refused the shutdown leds_off' in m for m in messages(session_log))
            assert scope.motor_connected is False
        finally:
            scope.disconnect()

    def test_a_pass_that_raised_can_be_retried(self, tmp_path, monkeypatch, session_log):
        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        scope = session.scope
        monkeypatch.setattr(
            scope.motion, 'stop_motion', MagicMock(side_effect=RuntimeError('bus gone'))
        )
        try:
            with pytest.raises(ScopeDisconnectError) as excinfo:
                session.shutdown()
            assert isinstance(excinfo.value.__cause__, RuntimeError)
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
        session = ScopeSession.create(settings=complete_settings(), scope=scope, **kwargs)
        return session, scope

    def test_a_passed_scope_is_untouched(self):
        session, scope = self._caller_session()
        session.shutdown()
        scope.disconnect.assert_not_called()
        scope.motion.stop_motion.assert_not_called()
        scope.illumination._leds_off_impl.assert_not_called()
        # The lanes are the scope's, and the scope is the caller's: they run on.
        session.io_executor.put.assert_not_called()
        session.io_executor.shutdown.assert_not_called()
        session.camera_executor.shutdown.assert_not_called()

    def test_a_directly_constructed_session_is_the_same_row(self):
        scope = spec_scope()
        session = ScopeSession(
            settings=complete_settings(), scope=scope, executor_bundle=MagicMock()
        )
        session.shutdown()
        scope.disconnect.assert_not_called()
        scope.motion.stop_motion.assert_not_called()
        session.io_executor.put.assert_not_called()

    def test_a_second_shutdown_logs_and_touches_nothing(self, session_log):
        session, _scope = self._caller_session()
        session.shutdown()
        session.executor_bundle.shutdown = MagicMock()
        session.shutdown()
        session.executor_bundle.shutdown.assert_not_called()
        assert any('shutdown() called again' in m for m in messages(session_log)), (
            'the second shutdown() must log its no-op'
        )


class TestDisconnectShutsTheLanesFirst:
    def test_a_queued_led_on_does_not_run_after_the_off(self):
        # An LED on still waiting on the io lane behind work in flight must
        # not run after disconnect()'s off: the scope would read as
        # disconnected with a channel lit. Shutting the lanes first drops it.
        scope = build_scope(simulate=True, warn_pre_release=False)
        lane = scope.io_lane()
        in_flight, release = threading.Event(), threading.Event()

        def hold():
            # Released only by the teardown below (or the finally), never by
            # a clock: a hold that let go on its own would run the led_on
            # before the disconnect on a slow host.
            in_flight.set()
            release.wait()

        lane.put(IOTask(action=hold))
        assert in_flight.wait(2.0)
        outcome = {}

        def turn_on():
            try:
                outcome['result'] = scope.illumination.led_on(channel=0, illumination_ma=10.0)
            except BaseException as ex:
                outcome['error'] = ex

        caller = threading.Thread(target=turn_on)
        caller.start()
        deadline = time.monotonic() + 2.0
        while lane.queue.qsize() == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert lane.queue.qsize() == 1, 'the led_on never queued behind the task in flight'

        # The work in flight finishes during the teardown, just after the
        # off: with the lanes still open the worker would take the queued
        # led_on next and light the channel.
        real_off = scope.illumination._leds_off_emergency

        def off_then_finish():
            real_off()
            release.set()
            time.sleep(0.3)

        try:
            with patch.object(scope.illumination, '_leds_off_emergency', off_then_finish):
                scope.disconnect()
        finally:
            release.set()
        caller.join(3.0)

        assert 'error' in outcome, f'the queued led_on was not cancelled: {outcome}'
        lit = [c for c, st in scope.illumination.get_led_states().items() if st.get('enabled')]
        assert lit == [], f'channels lit after disconnect: {lit}'
