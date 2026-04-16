# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Hardware serial communication benchmarks and stress tests.

These tests REQUIRE real hardware and are skipped by default.
Run with: pytest tests/test_hardware_serial.py --run-hardware

Tests measure:
- Serial round-trip latency for LED and motor commands
- Rapid-fire command reliability (no dropped/garbled responses)
- LED on/off cycling speed
- Motor position query throughput
- Concurrent access safety (multiple threads hitting serial)
"""

import pytest
import sys
import time
import threading
import statistics
from unittest.mock import MagicMock

# Heavy deps are mocked by tests/conftest.py at module-import time.


hardware = pytest.mark.skipif(
    "--run-hardware" not in sys.argv,
    reason="Requires --run-hardware flag and real hardware"
)


@hardware
class TestLEDSerialBenchmark:
    """Benchmark LED board serial communication."""

    @pytest.fixture
    def led(self):
        from drivers.ledboard import LEDBoard
        board = LEDBoard()
        if not board.found:
            pytest.skip("LED board not found")
        yield board
        # Ensure LEDs are off after test
        try:
            board.leds_off()
        except Exception as e:
            print(f'WARNING: Failed to turn off LEDs in fixture teardown: {e}')

    def test_exchange_command_latency(self, led):
        """Measure round-trip time for exchange_command."""
        times = []
        for _ in range(100):
            start = time.perf_counter()
            resp = led.exchange_command('STATUS')
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms
            assert resp is not None, "Got None response"

        median = statistics.median(times)
        p95 = sorted(times)[94]
        print(f"\n  LED exchange_command latency (100 calls):")
        print(f"    Median: {median:.2f} ms")
        print(f"    P95:    {p95:.2f} ms")
        print(f"    Min:    {min(times):.2f} ms")
        print(f"    Max:    {max(times):.2f} ms")
        # Should be well under 50ms with optimized code
        assert median < 50, f"Median latency too high: {median:.2f} ms"

    def test_led_on_off_cycle_speed(self, led):
        """Measure how fast we can toggle an LED on and off."""
        times = []
        for _ in range(50):
            start = time.perf_counter()
            led.led_on(channel=3, mA=10)
            led.led_off(channel=3)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        median = statistics.median(times)
        print(f"\n  LED on+off cycle (50 cycles):")
        print(f"    Median: {median:.2f} ms per cycle")
        print(f"    Max:    {max(times):.2f} ms")

    def test_led_on_off_fast_cycle_speed(self, led):
        """Measure fast path LED toggling speed."""
        times = []
        for _ in range(50):
            start = time.perf_counter()
            led.led_on_fast(channel=3, mA=10)
            led.led_off_fast(channel=3)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        median = statistics.median(times)
        print(f"\n  LED fast on+off cycle (50 cycles):")
        print(f"    Median: {median:.2f} ms per cycle")
        print(f"    Max:    {max(times):.2f} ms")

    def test_rapid_fire_no_errors(self, led):
        """Send 200 rapid commands and verify no errors."""
        errors = []
        for i in range(200):
            resp = led.exchange_command('STATUS')
            if resp is None:
                errors.append(f"Command {i}: got None")
        assert not errors, f"Errors during rapid fire: {errors[:10]}"

    def test_response_content_valid(self, led):
        """Verify response content from LED board makes sense."""
        # With the optimized code, we should now get the actual result
        # (not the RE: echo)
        resp = led.exchange_command('STATUS')
        assert resp is not None
        print(f"\n  STATUS response: {resp!r}")

        led.led_on(channel=3, mA=100)
        resp_on = led.exchange_command('STATUS')
        print(f"  STATUS after LED3 on: {resp_on!r}")
        led.led_off(channel=3)


@hardware
class TestMotorSerialBenchmark:
    """Benchmark motor board serial communication."""

    @pytest.fixture
    def motor(self):
        from drivers.motorboard import MotorBoard
        board = MotorBoard()
        if not board.found:
            pytest.skip("Motor board not found")
        yield board

    def test_exchange_command_latency(self, motor):
        """Measure round-trip time for exchange_command."""
        times = []
        for _ in range(100):
            start = time.perf_counter()
            resp = motor.exchange_command('STATUS_RZ')
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
            assert resp is not None, "Got None response"

        median = statistics.median(times)
        p95 = sorted(times)[94]
        print(f"\n  Motor exchange_command latency (100 calls):")
        print(f"    Median: {median:.2f} ms")
        print(f"    P95:    {p95:.2f} ms")
        print(f"    Min:    {min(times):.2f} ms")
        print(f"    Max:    {max(times):.2f} ms")
        assert median < 50, f"Median latency too high: {median:.2f} ms"

    def test_position_query_throughput(self, motor):
        """Measure position query rate."""
        start = time.perf_counter()
        count = 200
        for _ in range(count):
            motor.current_pos('Z')
        elapsed = time.perf_counter() - start

        rate = count / elapsed
        print(f"\n  Position query throughput: {rate:.1f} queries/sec ({elapsed:.2f}s for {count})")

    def test_rapid_status_queries(self, motor):
        """Send 200 rapid STATUS queries and verify no errors."""
        errors = []
        for i in range(200):
            resp = motor.exchange_command('STATUS_RZ')
            if resp is None:
                errors.append(f"Query {i}: got None")
        assert not errors, f"Errors during rapid queries: {errors[:10]}"

    def test_all_axes_status(self, motor):
        """Query status of all axes and print results."""
        for axis in ('X', 'Y', 'Z'):
            pos = motor.current_pos(axis)
            target = motor.target_pos(axis)
            at_target = motor.target_status(axis)
            print(f"\n  {axis}: pos={pos:.1f}um target={target:.1f}um at_target={at_target}")

    def test_info_response(self, motor):
        """Verify INFO command returns valid firmware string."""
        resp = motor.exchange_command('INFO')
        assert resp is not None
        assert 'Etaluma' in resp or 'Motor' in resp or 'EL-' in resp, \
            f"Unexpected INFO response: {resp!r}"
        print(f"\n  INFO: {resp}")


@hardware
class TestConcurrentSerialAccess:
    """Test thread safety of optimized serial code on real hardware."""

    @pytest.fixture
    def led(self):
        from drivers.ledboard import LEDBoard
        board = LEDBoard()
        if not board.found:
            pytest.skip("LED board not found")
        yield board
        try:
            board.leds_off()
        except Exception as e:
            print(f'WARNING: Failed to turn off LEDs in fixture teardown: {e}')

    def test_concurrent_led_commands(self, led):
        """Multiple threads sending LED commands should not corrupt state."""
        errors = []
        call_count = [0]

        def worker(channel, iterations):
            for _ in range(iterations):
                try:
                    led.led_on(channel=channel, mA=10)
                    led.led_off(channel=channel)
                    call_count[0] += 1
                except Exception as e:
                    errors.append(f"Ch{channel}: {e}")

        threads = [
            threading.Thread(target=worker, args=(0, 30)),
            threading.Thread(target=worker, args=(1, 30)),
            threading.Thread(target=worker, args=(2, 30)),
        ]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        elapsed = time.perf_counter() - start

        print(f"\n  Concurrent LED test: {call_count[0]} on+off cycles in {elapsed:.2f}s")
        assert not errors, f"Errors: {errors[:10]}"
        # All LEDs should be off
        assert not led.is_led_on('Blue')
        assert not led.is_led_on('Green')
        assert not led.is_led_on('Red')
