# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for `FirmwareDiagnostics.measure_serial_latency` + report step.

The refactor routes both through `drivers.serial_latency.measure_callable_latencies`
so the tech-support report, the connect-time fingerprint, and the CLI
bench all share the same math. The report step prefers the board's
`connect_latency_summary` (captured at connect) over re-measuring.
"""
import json
from unittest.mock import MagicMock

import pytest

from modules.tech_support_report import FirmwareDiagnostics, TechSupportReport


def _make_diag_with_board(board):
    """Build a FirmwareDiagnostics carrying a single mocked board."""
    diag = FirmwareDiagnostics.__new__(FirmwareDiagnostics)
    diag.led_board = None
    diag.motor_board = board
    diag.scope = None
    diag._owns_boards = False
    diag._eng_mode_enabled = False
    return diag


class TestMeasureSerialLatency:
    def test_none_board_returns_error(self):
        diag = _make_diag_with_board(None)
        result = diag.measure_serial_latency(None)
        assert 'error' in result
        assert 'not connected' in result['error'].lower()

    def test_uses_board_bench_callables_by_default(self):
        board = MagicMock()
        fn = MagicMock()
        board._connect_bench_callables = MagicMock(return_value=[('fullinfo', fn)])
        diag = _make_diag_with_board(board)

        result = diag.measure_serial_latency(board, iterations=5, warmup=1)

        assert 'fullinfo' in result
        assert result['fullinfo']['count'] == 5
        # 5 measured + 1 warmup = 6 calls
        assert fn.call_count == 6

    def test_explicit_callables_override_default(self):
        board = MagicMock()
        board._connect_bench_callables = MagicMock(return_value=[('should_not_use', MagicMock())])
        custom_fn = MagicMock()
        diag = _make_diag_with_board(board)

        result = diag.measure_serial_latency(
            board, named_callables=[('custom', custom_fn)],
            iterations=3, warmup=0,
        )

        assert 'custom' in result
        assert 'should_not_use' not in result
        assert custom_fn.call_count == 3

    def test_returns_error_when_no_callables_available(self):
        board = MagicMock(spec=[])  # no _connect_bench_callables attr
        diag = _make_diag_with_board(board)
        result = diag.measure_serial_latency(board)
        assert 'error' in result


class TestStepSerialLatency:
    def _make_report(self, motor_board, led_board):
        report = TechSupportReport.__new__(TechSupportReport)
        report.diag = FirmwareDiagnostics.__new__(FirmwareDiagnostics)
        report.diag.led_board = led_board
        report.diag.motor_board = motor_board
        report.diag.scope = None
        report.diag._owns_boards = False
        report.diag._eng_mode_enabled = False
        return report

    def test_uses_connect_summary_when_available(self, tmp_path):
        # Motor board arrives with a pre-captured connect summary —
        # no fresh measurement should be triggered.
        fn = MagicMock()
        motor = MagicMock()
        motor.firmware_version = '4.0.0'
        motor.connect_latency_summary = {
            'fullinfo': {
                'count': 20, 'errors': 0,
                'mean_us': 2500.0, 'stddev_us': 300.0,
                'p50_us': 2400.0, 'p95_us': 3000.0, 'p99_us': 3500.0,
                'min_us': 2000.0, 'max_us': 4000.0,
            }
        }
        motor._connect_bench_callables = MagicMock(return_value=[('fullinfo', fn)])

        report = self._make_report(motor, None)
        (tmp_path / 'hardware_checks').mkdir()
        report._step_serial_latency(tmp_path)

        # fn must NOT have been called — the report used the cached summary.
        assert fn.call_count == 0
        # JSON file records source=connect for Motor, error for LED.
        with open(tmp_path / 'hardware_checks' / 'serial_latency.json') as f:
            payload = json.load(f)
        assert payload['Motor']['source'] == 'connect'
        assert payload['Motor']['firmware_version'] == '4.0.0'
        assert 'error' in payload['LED']

    def test_falls_back_to_fresh_measurement_when_summary_absent(self, tmp_path):
        fn = MagicMock()
        led = MagicMock()
        led.firmware_version = '3.0.6'
        led.connect_latency_summary = None  # bench was skipped
        led._connect_bench_callables = MagicMock(return_value=[('get_info', fn)])

        report = self._make_report(None, led)
        (tmp_path / 'hardware_checks').mkdir()
        report._step_serial_latency(tmp_path)

        # Fell through to fresh measurement — fn was invoked.
        assert fn.call_count >= 1
        with open(tmp_path / 'hardware_checks' / 'serial_latency.json') as f:
            payload = json.load(f)
        assert payload['LED']['source'] == 'report'
        assert 'per_method' in payload['LED']
        assert 'get_info' in payload['LED']['per_method']

    def test_text_output_includes_firmware_and_percentiles(self, tmp_path):
        motor = MagicMock()
        motor.firmware_version = '4.0.0'
        motor.connect_latency_summary = {
            'fullinfo': {
                'count': 20, 'errors': 0,
                'mean_us': 2500.0, 'stddev_us': 300.0,
                'p50_us': 2400.0, 'p95_us': 3000.0, 'p99_us': 3500.0,
                'min_us': 2000.0, 'max_us': 4000.0,
            }
        }

        report = self._make_report(motor, None)
        (tmp_path / 'hardware_checks').mkdir()
        report._step_serial_latency(tmp_path)

        text = (tmp_path / 'hardware_checks' / 'serial_latency.txt').read_text()
        assert 'fw=4.0.0' in text
        assert 'source=connect' in text
        assert 'fullinfo' in text
        assert 'p95=' in text
        assert 'p99=' in text
