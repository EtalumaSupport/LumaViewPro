# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for `drivers/registry.py` (audit B2).

The registry replaces the hardcoded if/elif driver selection chains in
Lumascope.__init__. These tests cover:
  1. Unit tests on a fresh DriverRegistry — name/priority/auto/simulate
  2. End-to-end: Lumascope(simulate=True) produces sim drivers via the
     registry without any direct class references in the constructor
  3. Composite-hardware test double — proves the registry can hold the
     FX2 pattern (camera + LED sharing a singleton USB connection)
     before the real FX2 driver lands in Stage 3.
"""

import sys
import threading
from unittest.mock import MagicMock

# Mock heavy deps before importing Lumascope
sys.modules.setdefault('userpaths', MagicMock())
sys.modules.setdefault('requests', MagicMock())
sys.modules.setdefault('requests.structures', MagicMock())
_mock_lvp_logger = MagicMock()
_mock_lvp_logger.logger = MagicMock()
_mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
_mock_lvp_logger.unpause_thread = MagicMock()
_mock_lvp_logger.pause_thread = MagicMock()
sys.modules.setdefault('lvp_logger', _mock_lvp_logger)

import pytest

from drivers.registry import DriverRegistry, motor_registry, led_registry, camera_registry
from drivers.protocols import MotorBoardProtocol, LEDBoardProtocol


class TestDriverRegistryUnit:
    """Unit tests on a fresh DriverRegistry — no real drivers involved."""

    def test_register_and_create_by_name(self):
        reg = DriverRegistry('fake')

        @reg.register('alpha', priority=50)
        class Alpha:
            def __init__(self, **kw): self.kw = kw

        instance = reg.create('alpha', foo=1)
        assert isinstance(instance, Alpha)
        assert instance.kw == {'foo': 1}

    def test_unknown_name_raises_with_available_list(self):
        reg = DriverRegistry('fake')

        @reg.register('alpha', priority=50)
        class Alpha: pass

        with pytest.raises(ValueError, match=r"Available: \['alpha'\]"):
            reg.create('nonexistent')

    def test_duplicate_name_raises(self):
        reg = DriverRegistry('fake')

        @reg.register('alpha', priority=50)
        class Alpha: pass

        with pytest.raises(ValueError, match='already registered'):
            @reg.register('alpha', priority=60)
            class Alpha2: pass

    def test_auto_picks_highest_priority(self):
        reg = DriverRegistry('fake')
        order = []

        @reg.register('low', priority=10)
        class Low:
            def __init__(self, **kw): order.append('low')

        @reg.register('high', priority=100)
        class High:
            def __init__(self, **kw): order.append('high')

        @reg.register('mid', priority=50)
        class Mid:
            def __init__(self, **kw): order.append('mid')

        instance = reg.create('auto')
        assert isinstance(instance, High)
        assert order == ['high']  # Only the highest was tried

    def test_auto_falls_back_to_next_priority_on_exception(self):
        reg = DriverRegistry('fake')

        @reg.register('broken', priority=100)
        class Broken:
            def __init__(self, **kw):
                raise RuntimeError("no hardware")

        @reg.register('works', priority=50)
        class Works:
            def __init__(self, **kw): pass

        instance = reg.create('auto')
        assert isinstance(instance, Works)

    def test_auto_falls_back_to_null_when_all_real_fail(self):
        reg = DriverRegistry('fake')

        @reg.register('broken', priority=100)
        class Broken:
            def __init__(self, **kw):
                raise RuntimeError("no hardware")

        @reg.register('null', priority=0)
        class Null:
            def __init__(self): pass

        instance = reg.create('auto')
        assert isinstance(instance, Null)

    def test_auto_skips_found_false_drivers(self):
        """Drivers that signal failure via .found=False (SerialBoard
        pattern) are skipped in auto mode, same as if they raised."""
        reg = DriverRegistry('fake')

        @reg.register('unfound', priority=100)
        class Unfound:
            def __init__(self, **kw):
                self.found = False

        @reg.register('works', priority=50)
        class Works:
            def __init__(self, **kw):
                self.found = True

        instance = reg.create('auto')
        assert isinstance(instance, Works)

    def test_simulate_mode_only_considers_simulators(self):
        reg = DriverRegistry('fake')

        @reg.register('real', priority=100)
        class Real:
            def __init__(self, **kw): pass

        @reg.register('sim', priority=100, is_simulator=True)
        class Sim:
            def __init__(self, **kw): pass

        assert isinstance(reg.create('auto', simulate=True), Sim)
        assert isinstance(reg.create('auto', simulate=False), Real)

    def test_null_not_picked_as_primary_in_auto(self):
        """Null drivers (priority=0) are the fallback, not a candidate
        in the normal priority search."""
        reg = DriverRegistry('fake')

        @reg.register('null', priority=0)
        class Null:
            def __init__(self): pass

        @reg.register('real', priority=100)
        class Real:
            def __init__(self, **kw): pass

        instance = reg.create('auto')
        assert isinstance(instance, Real)

    def test_explicit_name_bypasses_auto_logic(self):
        """`create('name', ...)` with an explicit name goes straight to
        that class even if its priority would have put it last."""
        reg = DriverRegistry('fake')

        @reg.register('high', priority=100)
        class High:
            def __init__(self, **kw): pass

        @reg.register('low', priority=10)
        class Low:
            def __init__(self, **kw): pass

        assert isinstance(reg.create('low'), Low)


class TestLumascopeUsesRegistry:
    """End-to-end: Lumascope(simulate=True) produces sim drivers and
    Lumascope() with no real hardware produces Null drivers — entirely
    via the registry, not via hardcoded class references."""

    def test_simulate_true_yields_simulated_drivers(self):
        from modules.lumascope_api import Lumascope
        from drivers.simulated_motorboard import SimulatedMotorBoard
        from drivers.simulated_ledboard import SimulatedLEDBoard
        from drivers.simulated_camera import SimulatedCamera

        scope = Lumascope(simulate=True)
        assert isinstance(scope.motion, SimulatedMotorBoard)
        assert isinstance(scope.led, SimulatedLEDBoard)
        assert isinstance(scope.camera, SimulatedCamera)

    def test_simulated_scope_satisfies_protocols(self):
        """Cross-check with B1: whatever the registry returns in
        simulate mode must still satisfy the driver protocols."""
        from modules.lumascope_api import Lumascope
        scope = Lumascope(simulate=True)
        assert isinstance(scope.motion, MotorBoardProtocol)
        assert isinstance(scope.led, LEDBoardProtocol)

    def test_registries_know_about_current_drivers(self):
        """Sanity: the three global registries have all the expected
        entries after importing the driver modules.

        Motor and LED have full coverage (rp2040 / sim / null) on any
        platform. Camera 'ids' is optional — it only registers if the
        `ids_peak` SDK is installed (Windows/Linux with IDS drivers).
        On macOS dev machines without ids_peak, the ImportError guard
        in `lumascope_api.py` skips the `drivers.idscamera` import,
        so the decorator never runs and 'ids' is absent from the
        registry. That's correct graceful-degradation behavior."""
        assert 'rp2040' in motor_registry.registered_names()
        assert 'sim' in motor_registry.registered_names()
        assert 'null' in motor_registry.registered_names()

        assert 'rp2040' in led_registry.registered_names()
        assert 'sim' in led_registry.registered_names()
        assert 'null' in led_registry.registered_names()

        assert 'pylon' in camera_registry.registered_names()
        assert 'sim' in camera_registry.registered_names()
        # 'ids' only present if ids_peak SDK is installed.
        try:
            import ids_peak  # noqa: F401
            ids_available = True
        except ImportError:
            ids_available = False
        if ids_available:
            assert 'ids' in camera_registry.registered_names()


class TestRegistryAccommodatesCompositeHardware:
    """Proves B2's registry design can hold a multi-device driver that
    shares a singleton USB connection across camera and LED — the exact
    FX2 / Lumaview Classic pattern that Stage 3 will implement.

    This test uses a fake `_FakeLVCConnection` singleton and registers
    two test-only drivers (`FakeLVCCamera`, `FakeLVCLed`) that grab the
    singleton on construction. No production dead code — just a proof
    that the architecture supports the pattern before the real FX2
    driver lands.

    Note: this test covers camera + LED only, not motion. LS720 uses a
    standalone USB-to-serial motor controller that is NOT shared with
    the FX2; it will register as its own independent motor driver and
    doesn't need any special pattern.
    """

    def _make_fake_lvc_connection(self):
        class _FakeLVCConnection:
            _instance = None
            def __init__(self):
                self.commands_sent: list[tuple] = []
                self.lock = threading.Lock()
            @classmethod
            def get(cls):
                if cls._instance is None:
                    cls._instance = cls()
                return cls._instance
            @classmethod
            def reset(cls):
                cls._instance = None
        return _FakeLVCConnection

    def test_camera_and_led_can_share_a_singleton_connection(self):
        FakeConn = self._make_fake_lvc_connection()
        FakeConn.reset()

        local_led_reg = DriverRegistry('led')
        local_cam_reg = DriverRegistry('camera')

        @local_led_reg.register('fake_lvc', priority=50)
        class FakeLVCLed:
            def __init__(self, **kw):
                self._conn = FakeConn.get()
                self.found = True
            def led_on(self, channel, mA, **kw):
                with self._conn.lock:
                    self._conn.commands_sent.append(('led', channel, mA))

        @local_cam_reg.register('fake_lvc', priority=50)
        class FakeLVCCamera:
            def __init__(self, **kw):
                self._conn = FakeConn.get()
                self.found = True
            def grab(self):
                with self._conn.lock:
                    self._conn.commands_sent.append(('cam', 'grab'))
                return True

        led = local_led_reg.create('fake_lvc')
        cam = local_cam_reg.create('fake_lvc')

        led.led_on(0, 100)
        cam.grab()

        conn = FakeConn.get()
        assert ('led', 0, 100) in conn.commands_sent
        assert ('cam', 'grab') in conn.commands_sent

        # The key invariant — both drivers point at the *same* connection
        # object. This is what makes FX2's shared-USB pattern possible
        # without any special casing in Lumascope or the registry.
        assert led._conn is cam._conn

    def test_connection_initialized_lazily_on_first_driver_use(self):
        """The singleton is created when the first driver grabs it, not
        at import time. A scope that never constructs an FX2 driver
        never touches the FX2 USB connection."""
        FakeConn = self._make_fake_lvc_connection()
        FakeConn.reset()
        assert FakeConn._instance is None

        local_reg = DriverRegistry('thing')
        @local_reg.register('lazy', priority=50)
        class Lazy:
            def __init__(self, **kw):
                self._conn = FakeConn.get()

        _ = local_reg.create('lazy')
        assert FakeConn._instance is not None
