# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for the LED state observer, ownership, and save/restore infrastructure.

Uses simulated hardware — no real boards or Kivy needed.
"""

import threading
import time

import pytest

# Heavy deps are mocked by tests/conftest.py at module-import time.

from modules.lumascope_api import Lumascope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scope():
    """Simulated Lumascope with fast timing."""
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    s._motion_driver.set_timing_mode('fast')
    s._camera_driver.set_timing_mode('fast')
    s._camera_driver.load_cycle_images()
    s._camera_driver.start_grabbing()
    yield s


# ---------------------------------------------------------------------------
# LED Listener Tests
# ---------------------------------------------------------------------------

class TestLEDListener:
    """Tests for add_led_listener / _fire_led_listeners."""

    def test_listener_fires_on_led_on(self, scope):
        events = []
        scope.illumination.add_led_listener(lambda c, e, m, o: events.append((c, e, m, o)))
        scope.illumination.led_on(channel=0, mA=100)
        assert len(events) == 1
        color, enabled, mA, owner = events[0]
        assert enabled is True
        assert mA == 100.0
        assert owner == ''

    def test_listener_fires_on_led_off(self, scope):
        events = []
        scope.illumination.led_on(channel=0, mA=100)
        scope.illumination.add_led_listener(lambda c, e, m, o: events.append((c, e, m, o)))
        scope.illumination.led_off(channel=0)
        assert len(events) == 1
        assert events[0][1] is False  # enabled

    def test_listener_fires_on_leds_off(self, scope):
        events = []
        scope.illumination.led_on(channel=0, mA=100)
        scope.illumination.led_on(channel=1, mA=50)
        scope.illumination.add_led_listener(lambda c, e, m, o: events.append((c, e, m, o)))
        scope.illumination.leds_off()
        # Should fire once per channel in led_ma
        assert len(events) >= 2
        assert all(e[1] is False for e in events)

    def test_listener_fires_on_fast_methods(self, scope):
        events = []
        scope.illumination.add_led_listener(lambda c, e, m, o: events.append((c, e, m, o)))
        scope.illumination.led_on_fast(channel=0, mA=100)
        scope.illumination.led_off_fast(channel=0)
        assert len(events) == 2
        assert events[0][1] is True   # on
        assert events[1][1] is False  # off

    def test_listener_not_fired_on_skip(self, scope):
        """When led_on is called with same params (skip-check), no listener fires."""
        scope.illumination.led_on(channel=0, mA=100)
        events = []
        scope.illumination.add_led_listener(lambda c, e, m, o: events.append((c, e, m, o)))
        scope.illumination.led_on(channel=0, mA=100)  # redundant — should skip
        assert len(events) == 0

    def test_remove_listener(self, scope):
        events = []
        listener = lambda c, e, m, o: events.append((c, e, m, o))
        scope.illumination.add_led_listener(listener)
        scope.illumination.remove_led_listener(listener)
        scope.illumination.led_on(channel=0, mA=100)
        assert len(events) == 0

    def test_listener_exception_does_not_propagate(self, scope):
        """A broken listener must not prevent the LED command from succeeding."""
        def bad_listener(c, e, m, o):
            raise RuntimeError("broken listener")
        scope.illumination.add_led_listener(bad_listener)
        # Should not raise
        scope.illumination.led_on(channel=0, mA=100)
        assert scope.illumination.led_enabled(scope.illumination.ch2color(0))

    def test_listener_fires_from_multiple_threads(self, scope):
        """Listeners fire correctly regardless of which thread calls led_on."""
        events = []
        lock = threading.Lock()

        def listener(c, e, m, o):
            with lock:
                events.append(threading.current_thread().name)

        scope.illumination.add_led_listener(listener)

        def turn_on(ch, mA):
            scope.illumination.led_on(channel=ch, mA=mA)

        t1 = threading.Thread(target=turn_on, args=(0, 100), name='thread-A')
        t2 = threading.Thread(target=turn_on, args=(1, 50), name='thread-B')
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(events) == 2
        assert 'thread-A' in events
        assert 'thread-B' in events


# ---------------------------------------------------------------------------
# Ownership Tests
# ---------------------------------------------------------------------------

class TestLEDOwnership:
    """Tests for LED ownership tracking."""

    def test_ownership_blocks_foreign_off(self, scope):
        """led_off with wrong owner is a no-op."""
        scope.illumination.led_on(channel=0, mA=100, owner='autofocus')
        scope.illumination.led_off(channel=0, owner='protocol')  # wrong owner
        color = scope.illumination.ch2color(0)
        assert scope.illumination.led_enabled(color)  # still on

    def test_ownership_allows_own_off(self, scope):
        scope.illumination.led_on(channel=0, mA=100, owner='autofocus')
        scope.illumination.led_off(channel=0, owner='autofocus')
        color = scope.illumination.ch2color(0)
        assert not scope.illumination.led_enabled(color)

    def test_no_owner_off_is_unconditional(self, scope):
        """led_off without owner always works (backwards compatible)."""
        scope.illumination.led_on(channel=0, mA=100, owner='autofocus')
        scope.illumination.led_off(channel=0)  # no owner = unconditional
        color = scope.illumination.ch2color(0)
        assert not scope.illumination.led_enabled(color)

    def test_leds_off_nuclear_clears_all(self, scope):
        scope.illumination.led_on(channel=0, mA=100, owner='autofocus')
        scope.illumination.led_on(channel=1, mA=50, owner='protocol')
        scope.illumination.leds_off()  # nuclear
        assert not scope.illumination.led_enabled(scope.illumination.ch2color(0))
        assert not scope.illumination.led_enabled(scope.illumination.ch2color(1))

    def test_leds_off_owned(self, scope):
        """leds_off_owned only turns off channels owned by that owner."""
        scope.illumination.led_on(channel=0, mA=100, owner='autofocus')
        scope.illumination.led_on(channel=1, mA=50, owner='protocol')
        scope.illumination.leds_off_owned('autofocus')
        assert not scope.illumination.led_enabled(scope.illumination.ch2color(0))  # AF's LED off
        assert scope.illumination.led_enabled(scope.illumination.ch2color(1))       # protocol's LED still on

    def test_ownership_with_listener(self, scope):
        """Ownership info is passed through to listeners."""
        events = []
        scope.illumination.add_led_listener(lambda c, e, m, o: events.append(o))
        scope.illumination.led_on(channel=0, mA=100, owner='autofocus')
        assert events[-1] == 'autofocus'


# ---------------------------------------------------------------------------
# Save/Restore Tests
# ---------------------------------------------------------------------------

class TestLEDSaveRestore:
    """Tests for save_led_state / restore_led_state."""

    def test_save_restore_roundtrip(self, scope):
        """Save state with LEDs on, turn all off, restore, verify original."""
        scope.illumination.led_on(channel=0, mA=100)
        scope.illumination.led_on(channel=1, mA=50)
        snapshot = scope.illumination.save_led_state('test')
        scope.illumination.leds_off()
        assert not scope.illumination.led_enabled(scope.illumination.ch2color(0))
        scope.illumination.restore_led_state(snapshot)
        assert scope.illumination.led_enabled(scope.illumination.ch2color(0))
        assert scope.illumination.led_enabled(scope.illumination.ch2color(1))

    def test_restore_with_owner_only_clears_owned(self, scope):
        """Restore with owner only turns off that owner's channels first."""
        scope.illumination.led_on(channel=0, mA=100, owner='ui')
        scope.illumination.led_on(channel=1, mA=50, owner='autofocus')
        # Save state (both on)
        snapshot = scope.illumination.save_led_state('test')
        # AF turns off its channel
        scope.illumination.leds_off_owned('autofocus')
        # Restore with owner='autofocus' — should only affect AF's channels
        scope.illumination.restore_led_state(snapshot, owner='autofocus')
        # Both should be back on (ui's was never off, AF's is restored)
        assert scope.illumination.led_enabled(scope.illumination.ch2color(0))
        assert scope.illumination.led_enabled(scope.illumination.ch2color(1))

    def test_restore_empty_snapshot(self, scope):
        """Restoring None/empty snapshot is a no-op."""
        scope.illumination.led_on(channel=0, mA=100)
        scope.illumination.restore_led_state(None)
        assert scope.illumination.led_enabled(scope.illumination.ch2color(0))  # unchanged
        scope.illumination.restore_led_state({})
        assert scope.illumination.led_enabled(scope.illumination.ch2color(0))  # unchanged

    def test_af_pattern_save_restore(self, scope):
        """Simulate the AF pattern: save → own LED on → do work → off owned → restore."""
        # User has Blue LED on
        scope.illumination.led_on(channel=0, mA=100)
        # AF starts
        snapshot = scope.illumination.save_led_state('autofocus')
        scope.illumination.led_on(channel=3, mA=200, owner='autofocus')  # BF for AF
        # AF finishes
        scope.illumination.leds_off_owned('autofocus')  # only kills AF's LED
        assert scope.illumination.led_enabled(scope.illumination.ch2color(0))   # user's Blue still on
        assert not scope.illumination.led_enabled(scope.illumination.ch2color(3))  # AF's BF off
        # Restore (should be a no-op since user's LED was never touched)
        scope.illumination.restore_led_state(snapshot, owner='autofocus')
        assert scope.illumination.led_enabled(scope.illumination.ch2color(0))   # still on


# ---------------------------------------------------------------------------
# Camera Listener Tests
# ---------------------------------------------------------------------------

class TestCameraListener:
    """Tests for add_camera_listener / _fire_camera_listeners."""

    def test_listener_fires_on_set_gain(self, scope):
        events = []
        scope.imaging.add_camera_listener(lambda p, v: events.append((p, v)))
        scope.imaging.set_gain(5.0)
        assert len(events) == 1
        assert events[0] == ('gain', 5.0)

    def test_listener_fires_on_set_exposure(self, scope):
        events = []
        scope.imaging.add_camera_listener(lambda p, v: events.append((p, v)))
        scope.imaging.set_exposure_time(25.0)  # Different from default 10ms
        assert len(events) == 1
        assert events[0] == ('exposure', 25.0)

    def test_listener_not_fired_on_redundant_gain(self, scope):
        """Skip-check: same gain value should not fire listener."""
        scope.imaging.set_gain(5.0)
        events = []
        scope.imaging.add_camera_listener(lambda p, v: events.append((p, v)))
        scope.imaging.set_gain(5.0)  # redundant
        assert len(events) == 0

    def test_remove_camera_listener(self, scope):
        events = []
        listener = lambda p, v: events.append((p, v))
        scope.imaging.add_camera_listener(listener)
        scope.imaging.remove_camera_listener(listener)
        scope.imaging.set_gain(5.0)
        assert len(events) == 0

    def test_camera_listener_exception_does_not_propagate(self, scope):
        def bad_listener(p, v):
            raise RuntimeError("broken")
        scope.imaging.add_camera_listener(bad_listener)
        scope.imaging.set_gain(5.0)  # should not raise


# ---------------------------------------------------------------------------
# Camera Save/Restore Tests
# ---------------------------------------------------------------------------

class TestCameraSaveRestore:
    """Tests for save_camera_state / restore_camera_state."""

    def test_save_restore_roundtrip(self, scope):
        scope.imaging.set_gain(5.0)
        scope.imaging.set_exposure_time(25.0)
        snapshot = scope.imaging.save_camera_state('test')
        # Change to different values
        scope.imaging.set_gain(10.0)
        scope.imaging.set_exposure_time(50.0)
        assert scope.imaging.get_gain() != 5.0
        # Restore
        scope.imaging.restore_camera_state(snapshot)
        assert abs(scope.imaging.get_gain() - 5.0) < 0.01
        assert abs(scope.imaging.get_exposure_time() - 25.0) < 0.01

    def test_restore_empty_snapshot(self, scope):
        scope.imaging.set_gain(5.0)
        scope.imaging.restore_camera_state(None)
        assert abs(scope.imaging.get_gain() - 5.0) < 0.01  # unchanged
        scope.imaging.restore_camera_state({})
        assert abs(scope.imaging.get_gain() - 5.0) < 0.01  # unchanged

    def test_snapshot_contains_tag(self, scope):
        snapshot = scope.imaging.save_camera_state('protocol')
        assert snapshot['tag'] == 'protocol'
        assert 'gain_db' in snapshot
        assert 'exposure_ms' in snapshot
