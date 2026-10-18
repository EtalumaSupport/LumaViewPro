# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for the LED state observer, ownership, and save/restore infrastructure.

Uses simulated hardware -- no real boards or Kivy needed.
"""

import threading

import pytest

from modules.lumascope_api.illumination import LedState

# Heavy deps are mocked by tests/conftest.py at module-import time.

from tests.protocol_drives import held_run_claim
from tests.scope_fakes import bind_settings_like_a_session, build_scope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scope():
    """Simulated Lumascope with fast timing, bound to the shipped settings."""
    s = build_scope(simulate=True)
    bind_settings_like_a_session(s)
    s._led_driver.set_timing_mode('fast')
    s._motion_driver.set_timing_mode('fast')
    s._camera_driver.set_timing_mode('fast')
    s._camera_driver.load_specimen()
    s.imaging.enable_streaming()
    yield s


# ---------------------------------------------------------------------------
# LED Listener Tests
# ---------------------------------------------------------------------------


class TestLEDListener:
    """Tests for add_led_listener / _fire_led_listeners."""

    def test_listener_fires_on_led_on(self, scope):
        events = []
        scope.illumination.add_led_listener(lambda a, n, st, m: events.append((a, n, st, m)))
        scope.illumination.led_on(channel=0, illumination_ma=100)
        assert len(events) == 1
        address, _name, state, illumination_ma = events[0]
        assert (address, state, illumination_ma) == (0, LedState.LIT, 100.0)

    def test_listener_fires_on_led_off(self, scope):
        events = []
        scope.illumination.led_on(channel=0, illumination_ma=100)
        scope.illumination.add_led_listener(lambda a, n, st, m: events.append((a, n, st, m)))
        scope.illumination.led_off(channel=0)
        assert len(events) == 1
        assert (events[0][0], events[0][2]) == (0, LedState.DARK)

    def test_listener_fires_on_leds_off(self, scope):
        events = []
        scope.illumination.led_on(channel=0, illumination_ma=100)
        scope.illumination.led_on(channel=1, illumination_ma=50)
        scope.illumination.add_led_listener(lambda a, n, st, m: events.append((a, n, st, m)))
        scope.illumination.leds_off()
        # One event per board address, every one dark.
        board = [r.address for r in scope.illumination.get_led_states()]
        assert [e[0] for e in events] == board
        assert all(e[2] is LedState.DARK for e in events)

    def test_listener_not_fired_on_skip(self, scope):
        """When led_on is called with same params (skip-check), no listener fires."""
        scope.illumination.led_on(channel=0, illumination_ma=100)
        events = []
        scope.illumination.add_led_listener(lambda a, n, st, m: events.append((a, n, st, m)))
        scope.illumination.led_on(channel=0, illumination_ma=100)  # redundant -- should skip
        assert len(events) == 0

    def test_remove_listener(self, scope):
        events = []
        listener = lambda a, n, st, m: events.append((a, n, st, m))
        scope.illumination.add_led_listener(listener)
        scope.illumination.remove_led_listener(listener)
        scope.illumination.led_on(channel=0, illumination_ma=100)
        assert len(events) == 0

    def test_listener_exception_does_not_propagate(self, scope):
        """A broken listener must not prevent the LED command from succeeding."""

        def bad_listener(a, n, st, m):
            raise RuntimeError('broken listener')

        scope.illumination.add_led_listener(bad_listener)
        # Should not raise
        scope.illumination.led_on(channel=0, illumination_ma=100)
        assert scope.illumination.get_led_state(0).state is LedState.LIT

    def test_listener_fires_from_multiple_threads(self, scope):
        """Listeners fire for every caller, whichever thread calls led_on.

        The command runs on the scope's io lane, so that is where they fire."""
        events = []
        lock = threading.Lock()

        def listener(a, n, st, m):
            with lock:
                events.append(threading.current_thread().name)

        scope.illumination.add_led_listener(listener)

        def turn_on(ch, illumination_ma):
            scope.illumination.led_on(channel=ch, illumination_ma=illumination_ma)

        t1 = threading.Thread(target=turn_on, args=(0, 100), name='thread-A')
        t2 = threading.Thread(target=turn_on, args=(1, 50), name='thread-B')
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert events == [scope.io_lane().executor_name] * 2


# ---------------------------------------------------------------------------
# Ownership Tests
# ---------------------------------------------------------------------------


class TestLEDOwnership:
    """Tests for LED ownership tracking."""

    def test_ownership_allows_own_off(self, scope):
        lease = scope.illumination.acquire_led_lease('autofocus', claim=held_run_claim())
        scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease=lease)
        scope.illumination._led_off_impl(channel=0, _lease=lease)
        assert scope.illumination.get_led_state(0).state is LedState.DARK

    def test_no_owner_off_is_unconditional(self, scope):
        """led_off without a lease works on a channel a (released) lease lit."""
        lease = scope.illumination.acquire_led_lease('autofocus', claim=held_run_claim())
        scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease=lease)
        lease.release(leave_on=True)
        scope.illumination.led_off(channel=0)  # no lease = unconditional
        assert scope.illumination.get_led_state(0).state is LedState.DARK

    def test_leds_off_nuclear_clears_all(self, scope):
        protocol = scope.illumination.acquire_led_lease('protocol', claim=held_run_claim())
        scope.illumination._led_on_impl(channel=1, illumination_ma=50, _lease=protocol)
        autofocus = protocol.acquire_child('autofocus')
        scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease=autofocus)
        assert scope.illumination.get_led_state(0).state is LedState.LIT
        assert scope.illumination.get_led_state(1).state is LedState.LIT
        scope.illumination.leds_off()  # nuclear
        assert scope.illumination.get_led_state(0).state is LedState.DARK
        assert scope.illumination.get_led_state(1).state is LedState.DARK


# ---------------------------------------------------------------------------
# Save/Restore Tests
# ---------------------------------------------------------------------------


class TestLEDSaveRestore:
    """Tests for save_led_state / restore_led_state."""

    def test_save_restore_roundtrip(self, scope):
        """Save state with LEDs on, turn all off, restore, verify original."""
        scope.illumination.led_on(channel=0, illumination_ma=100)
        scope.illumination.led_on(channel=1, illumination_ma=50)
        snapshot = scope.illumination.save_led_state('test')
        scope.illumination.leds_off()
        assert scope.illumination.get_led_state(0).state is LedState.DARK
        scope.illumination.restore_led_state(snapshot)
        assert scope.illumination.get_led_state(0).state is LedState.LIT
        assert scope.illumination.get_led_state(1).state is LedState.LIT

    def test_restoring_an_all_dark_snapshot_darkens_what_was_lit_since(self, scope):
        """A snapshot holds every address, so one taken with nothing lit
        restores every address dark."""
        snapshot = scope.illumination.save_led_state('dark')
        scope.illumination.led_on(channel=0, illumination_ma=100)
        scope.illumination.restore_led_state(snapshot)
        assert scope.illumination.get_led_state(0).state is LedState.DARK

    def test_af_pattern_save_restore(self, scope):
        """Simulate the AF pattern: save -> own LED on -> do work -> release -> restore."""
        # User has Blue LED on
        scope.illumination.led_on(channel=0, illumination_ma=100)
        # AF starts
        snapshot = scope.illumination.save_led_state('autofocus')
        lease = scope.illumination.acquire_led_lease('autofocus', claim=held_run_claim())
        scope.illumination._led_on_impl(channel=3, illumination_ma=200, _lease=lease)  # BF for AF
        # AF finishes
        lease.release()  # only kills the LED the AF lease lit
        assert scope.illumination.get_led_state(0).state is LedState.LIT  # user's Blue still on
        assert scope.illumination.get_led_state(3).state is LedState.DARK  # AF's BF off
        # Restore (should be a no-op since user's LED was never touched)
        scope.illumination.restore_led_state(snapshot)
        assert scope.illumination.get_led_state(0).state is LedState.LIT  # still on


# ---------------------------------------------------------------------------
# Camera Listener Tests
# ---------------------------------------------------------------------------


class TestCameraListener:
    """Tests for add_camera_listener / _fire_camera_listeners."""

    def test_listener_fires_on_set_gain(self, scope):
        events = []
        scope.camera.add_camera_listener(lambda p, v: events.append((p, v)))
        scope.camera.set_gain_db(5.0)
        assert len(events) == 1
        assert events[0] == ('gain', 5.0)

    def test_listener_fires_on_set_exposure(self, scope):
        events = []
        scope.camera.add_camera_listener(lambda p, v: events.append((p, v)))
        scope.camera.set_exposure_ms(25.0)  # Different from default 10ms
        assert len(events) == 1
        assert events[0] == ('exposure', 25.0)

    def test_listener_not_fired_on_redundant_gain(self, scope):
        """Skip-check: same gain value should not fire listener."""
        scope.camera.set_gain_db(5.0)
        events = []
        scope.camera.add_camera_listener(lambda p, v: events.append((p, v)))
        scope.camera.set_gain_db(5.0)  # redundant
        assert len(events) == 0

    def test_remove_camera_listener(self, scope):
        events = []
        listener = lambda p, v: events.append((p, v))
        scope.camera.add_camera_listener(listener)
        scope.camera.remove_camera_listener(listener)
        scope.camera.set_gain_db(5.0)
        assert len(events) == 0

    def test_camera_listener_exception_does_not_propagate(self, scope):
        def bad_listener(p, v):
            raise RuntimeError('broken')

        scope.camera.add_camera_listener(bad_listener)
        scope.camera.set_gain_db(5.0)  # should not raise


# ---------------------------------------------------------------------------
# Camera Save/Restore Tests
# ---------------------------------------------------------------------------


class TestCameraSaveRestore:
    """Tests for save_camera_state / restore_camera_state."""

    def test_save_restore_roundtrip(self, scope):
        scope.camera.set_gain_db(5.0)
        scope.camera.set_exposure_ms(25.0)
        snapshot = scope.camera.save_camera_state('test')
        # Change to different values
        scope.camera.set_gain_db(10.0)
        scope.camera.set_exposure_ms(50.0)
        assert scope.camera.get_gain_db() != 5.0
        # Restore
        scope.camera.restore_camera_state(snapshot)
        assert abs(scope.camera.get_gain_db() - 5.0) < 0.01
        assert abs(scope.camera.get_exposure_ms() - 25.0) < 0.01

    def test_restore_empty_snapshot(self, scope):
        scope.camera.set_gain_db(5.0)
        scope.camera.restore_camera_state(None)
        assert abs(scope.camera.get_gain_db() - 5.0) < 0.01  # unchanged
        scope.camera.restore_camera_state({})
        assert abs(scope.camera.get_gain_db() - 5.0) < 0.01  # unchanged

    def test_snapshot_contains_tag(self, scope):
        snapshot = scope.camera.save_camera_state('protocol')
        assert snapshot['tag'] == 'protocol'
        assert 'gain_db' in snapshot
        assert 'exposure_ms' in snapshot
