# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for the LED state observer, ownership, and save/restore infrastructure.

Uses simulated hardware — no real boards or Kivy needed.
"""

import threading
import time

import pytest

from tests.conftest import install_mock_deps

install_mock_deps()

from modules.lumascope_api import Lumascope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scope():
    """Simulated Lumascope with fast timing."""
    s = Lumascope(simulate=True)
    s.led.set_timing_mode('fast')
    s.motion.set_timing_mode('fast')
    s.camera.set_timing_mode('fast')
    s.camera.load_cycle_images()
    s.camera.start_grabbing()
    yield s


# ---------------------------------------------------------------------------
# LED Listener Tests
# ---------------------------------------------------------------------------

class TestLEDListener:
    """Tests for add_led_listener / _fire_led_listeners."""

    def test_listener_fires_on_led_on(self, scope):
        events = []
        scope.add_led_listener(lambda c, e, m, o: events.append((c, e, m, o)))
        scope.led_on(channel=0, mA=100)
        assert len(events) == 1
        color, enabled, mA, owner = events[0]
        assert enabled is True
        assert mA == 100.0
        assert owner == ''

    def test_listener_fires_on_led_off(self, scope):
        events = []
        scope.led_on(channel=0, mA=100)
        scope.add_led_listener(lambda c, e, m, o: events.append((c, e, m, o)))
        scope.led_off(channel=0)
        assert len(events) == 1
        assert events[0][1] is False  # enabled

    def test_listener_fires_on_leds_off(self, scope):
        events = []
        scope.led_on(channel=0, mA=100)
        scope.led_on(channel=1, mA=50)
        scope.add_led_listener(lambda c, e, m, o: events.append((c, e, m, o)))
        scope.leds_off()
        # Should fire once per channel in led_ma
        assert len(events) >= 2
        assert all(e[1] is False for e in events)

    def test_listener_fires_on_fast_methods(self, scope):
        events = []
        scope.add_led_listener(lambda c, e, m, o: events.append((c, e, m, o)))
        scope.led_on_fast(channel=0, mA=100)
        scope.led_off_fast(channel=0)
        assert len(events) == 2
        assert events[0][1] is True   # on
        assert events[1][1] is False  # off

    def test_listener_not_fired_on_skip(self, scope):
        """When led_on is called with same params (skip-check), no listener fires."""
        scope.led_on(channel=0, mA=100)
        events = []
        scope.add_led_listener(lambda c, e, m, o: events.append((c, e, m, o)))
        scope.led_on(channel=0, mA=100)  # redundant — should skip
        assert len(events) == 0

    def test_remove_listener(self, scope):
        events = []
        listener = lambda c, e, m, o: events.append((c, e, m, o))
        scope.add_led_listener(listener)
        scope.remove_led_listener(listener)
        scope.led_on(channel=0, mA=100)
        assert len(events) == 0

    def test_listener_exception_does_not_propagate(self, scope):
        """A broken listener must not prevent the LED command from succeeding."""
        def bad_listener(c, e, m, o):
            raise RuntimeError("broken listener")
        scope.add_led_listener(bad_listener)
        # Should not raise
        scope.led_on(channel=0, mA=100)
        assert scope.led_enabled(scope.ch2color(0))

    def test_listener_fires_from_multiple_threads(self, scope):
        """Listeners fire correctly regardless of which thread calls led_on."""
        events = []
        lock = threading.Lock()

        def listener(c, e, m, o):
            with lock:
                events.append(threading.current_thread().name)

        scope.add_led_listener(listener)

        def turn_on(ch, mA):
            scope.led_on(channel=ch, mA=mA)

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
        scope.led_on(channel=0, mA=100, owner='autofocus')
        scope.led_off(channel=0, owner='protocol')  # wrong owner
        color = scope.ch2color(0)
        assert scope.led_enabled(color)  # still on

    def test_ownership_allows_own_off(self, scope):
        scope.led_on(channel=0, mA=100, owner='autofocus')
        scope.led_off(channel=0, owner='autofocus')
        color = scope.ch2color(0)
        assert not scope.led_enabled(color)

    def test_no_owner_off_is_unconditional(self, scope):
        """led_off without owner always works (backwards compatible)."""
        scope.led_on(channel=0, mA=100, owner='autofocus')
        scope.led_off(channel=0)  # no owner = unconditional
        color = scope.ch2color(0)
        assert not scope.led_enabled(color)

    def test_leds_off_nuclear_clears_all(self, scope):
        scope.led_on(channel=0, mA=100, owner='autofocus')
        scope.led_on(channel=1, mA=50, owner='protocol')
        scope.leds_off()  # nuclear
        assert not scope.led_enabled(scope.ch2color(0))
        assert not scope.led_enabled(scope.ch2color(1))

    def test_leds_off_owned(self, scope):
        """leds_off_owned only turns off channels owned by that owner."""
        scope.led_on(channel=0, mA=100, owner='autofocus')
        scope.led_on(channel=1, mA=50, owner='protocol')
        scope.leds_off_owned('autofocus')
        assert not scope.led_enabled(scope.ch2color(0))  # AF's LED off
        assert scope.led_enabled(scope.ch2color(1))       # protocol's LED still on

    def test_ownership_with_listener(self, scope):
        """Ownership info is passed through to listeners."""
        events = []
        scope.add_led_listener(lambda c, e, m, o: events.append(o))
        scope.led_on(channel=0, mA=100, owner='autofocus')
        assert events[-1] == 'autofocus'


# ---------------------------------------------------------------------------
# Save/Restore Tests
# ---------------------------------------------------------------------------

class TestLEDSaveRestore:
    """Tests for save_led_state / restore_led_state."""

    def test_save_restore_roundtrip(self, scope):
        """Save state with LEDs on, turn all off, restore, verify original."""
        scope.led_on(channel=0, mA=100)
        scope.led_on(channel=1, mA=50)
        snapshot = scope.save_led_state('test')
        scope.leds_off()
        assert not scope.led_enabled(scope.ch2color(0))
        scope.restore_led_state(snapshot)
        assert scope.led_enabled(scope.ch2color(0))
        assert scope.led_enabled(scope.ch2color(1))

    def test_restore_with_owner_only_clears_owned(self, scope):
        """Restore with owner only turns off that owner's channels first."""
        scope.led_on(channel=0, mA=100, owner='ui')
        scope.led_on(channel=1, mA=50, owner='autofocus')
        # Save state (both on)
        snapshot = scope.save_led_state('test')
        # AF turns off its channel
        scope.leds_off_owned('autofocus')
        # Restore with owner='autofocus' — should only affect AF's channels
        scope.restore_led_state(snapshot, owner='autofocus')
        # Both should be back on (ui's was never off, AF's is restored)
        assert scope.led_enabled(scope.ch2color(0))
        assert scope.led_enabled(scope.ch2color(1))

    def test_restore_empty_snapshot(self, scope):
        """Restoring None/empty snapshot is a no-op."""
        scope.led_on(channel=0, mA=100)
        scope.restore_led_state(None)
        assert scope.led_enabled(scope.ch2color(0))  # unchanged
        scope.restore_led_state({})
        assert scope.led_enabled(scope.ch2color(0))  # unchanged

    def test_af_pattern_save_restore(self, scope):
        """Simulate the AF pattern: save → own LED on → do work → off owned → restore."""
        # User has Blue LED on
        scope.led_on(channel=0, mA=100)
        # AF starts
        snapshot = scope.save_led_state('autofocus')
        scope.led_on(channel=3, mA=200, owner='autofocus')  # BF for AF
        # AF finishes
        scope.leds_off_owned('autofocus')  # only kills AF's LED
        assert scope.led_enabled(scope.ch2color(0))   # user's Blue still on
        assert not scope.led_enabled(scope.ch2color(3))  # AF's BF off
        # Restore (should be a no-op since user's LED was never touched)
        scope.restore_led_state(snapshot, owner='autofocus')
        assert scope.led_enabled(scope.ch2color(0))   # still on
