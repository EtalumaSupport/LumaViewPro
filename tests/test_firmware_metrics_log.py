# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the [FIRMWARE METRICS] log line emitted by
``modules.config_helpers.log_system_metrics(scope=...)``.

The hourly performance metrics logger is enriched (Stage 3 Lane 2) with
motor + LED firmware versions, fan status, and per-board connect-time
latency summary. Scope-less callers (startup, tests) get the existing
CPU/RAM/disk path unchanged.
"""
from unittest.mock import MagicMock, patch

import pytest


class _CaptureLogs:
    """Capture log lines by intercepting calls to the mocked lvp_logger.

    Conftest replaces ``lvp_logger`` with a MagicMock, so ``logger.info``
    calls never hit a real logging.Logger. We patch the logger used by
    ``modules.config_helpers`` for the duration of the test.
    """

    def __init__(self):
        self.messages = []

    def info(self, msg, *args, **kwargs):
        self.messages.append(msg % args if args else str(msg))

    def warning(self, msg, *args, **kwargs):
        self.messages.append(msg % args if args else str(msg))

    def error(self, msg, *args, **kwargs):
        self.messages.append(msg % args if args else str(msg))

    def debug(self, *a, **k):
        pass

    def firmware_line(self):
        for m in self.messages:
            if '[FIRMWARE METRICS]' in m:
                return m
        return None


@pytest.fixture
def capture_logs():
    cap = _CaptureLogs()
    with patch('modules.config_helpers.logger', cap):
        yield cap


@pytest.fixture
def stub_common_utils():
    """Mock common_utils so log_system_metrics runs without real psutil.

    The test suite's conftest stubs ``psutil`` to a MagicMock, but
    ``log_system_metrics`` calls ``check_disk_space`` and compares the
    result against an int. Replace both helpers with deterministic
    returns so only the firmware-metrics branch varies across tests.
    """
    with patch('modules.common_utils.system_metrics') as metrics, \
         patch('modules.common_utils.check_disk_space') as disk, \
         patch('modules.common_utils.get_extra_disks_info') as extras:
        metrics.return_value = {
            'cpu_percent_total': 25.0,
            'ram_available_gb': 8.0,
            'ram_percent_total': 50.0,
            'disk_free_gb': 100.0,
            'disk_used_percent': 30.0,
            'cpu_percent_python': 5.0,
            'ram_used_python_mb': 200.0,
            'ram_used_python_percent': 2.5,
        }
        disk.return_value = 100000
        extras.return_value = None
        yield


def _make_scope(motor_connected=True, led_connected=True,
                motor_fw='4.0.0', led_fw='4.0.0',
                fan_status=None,
                motor_latency_summary=None,
                led_latency_summary=None):
    scope = MagicMock()
    scope.motion = MagicMock()
    scope.motion.is_connected = MagicMock(return_value=motor_connected)
    scope.motion.firmware_version = motor_fw
    scope.motion.connect_latency_summary = motor_latency_summary

    scope.led = MagicMock()
    scope.led.is_connected = MagicMock(return_value=led_connected)
    scope.led.firmware_version = led_fw
    scope.led.connect_latency_summary = led_latency_summary

    scope.get_fan_status = MagicMock(return_value=fan_status)
    return scope


def _stub_settings(tmp_path):
    return {'live_folder': str(tmp_path)}


class TestFirmwareMetricsLine:

    def test_no_line_when_scope_is_none(self, capture_logs, stub_common_utils, tmp_path):
        from modules import config_helpers
        config_helpers.log_system_metrics(_stub_settings(tmp_path), scope=None)
        assert capture_logs.firmware_line() is None

    def test_emits_line_with_fw_versions(self, capture_logs, stub_common_utils, tmp_path):
        from modules import config_helpers
        scope = _make_scope(motor_fw='4.0.0', led_fw='4.0.0')
        config_helpers.log_system_metrics(_stub_settings(tmp_path), scope=scope)
        line = capture_logs.firmware_line()
        assert line is not None
        assert 'motor=4.0.0' in line
        assert 'led=4.0.0' in line

    def test_skips_disconnected_boards(self, capture_logs, stub_common_utils, tmp_path):
        from modules import config_helpers
        scope = _make_scope(motor_connected=False, led_connected=True,
                            led_fw='3.0.7')
        config_helpers.log_system_metrics(_stub_settings(tmp_path), scope=scope)
        line = capture_logs.firmware_line()
        assert line is not None
        assert 'motor=' not in line
        assert 'led=3.0.7' in line

    def test_fan_block_formatted(self, capture_logs, stub_common_utils, tmp_path):
        from modules import config_helpers
        scope = _make_scope(fan_status={'mode': 'HILO', 'state': 'HI',
                                        'fan_pct': None, 'tach_rpm': None})
        config_helpers.log_system_metrics(_stub_settings(tmp_path), scope=scope)
        line = capture_logs.firmware_line()
        assert 'fan=[mode=HILO state=HI]' in line

    def test_fan_pwm_includes_pct_and_rpm(self, capture_logs, stub_common_utils, tmp_path):
        from modules import config_helpers
        scope = _make_scope(fan_status={'mode': 'PWM', 'state': None,
                                        'fan_pct': 40, 'tach_rpm': 2400})
        config_helpers.log_system_metrics(_stub_settings(tmp_path), scope=scope)
        line = capture_logs.firmware_line()
        assert 'mode=PWM' in line
        assert 'pct=40' in line
        assert 'rpm=2400' in line

    def test_latency_summary_emitted_per_method(self, capture_logs, stub_common_utils, tmp_path):
        from modules import config_helpers
        scope = _make_scope(
            motor_latency_summary={
                'fullinfo': {'mean_ms': 32.3, 'p95_ms': 34.0},
            },
            led_latency_summary={
                'get_info': {'mean_ms': 69.3, 'p95_ms': 69.7},
            },
        )
        config_helpers.log_system_metrics(_stub_settings(tmp_path), scope=scope)
        line = capture_logs.firmware_line()
        assert 'motor_fullinfo=mean32.3ms/p9534.0ms' in line
        assert 'led_get_info=mean69.3ms/p9569.7ms' in line

    def test_individual_failure_does_not_drop_whole_line(
        self, capture_logs, stub_common_utils, tmp_path,
    ):
        from modules import config_helpers
        # Motor raises on is_connected; LED is fine. Line should still
        # emit with the LED portion populated.
        scope = MagicMock()
        scope.motion = MagicMock()
        scope.motion.is_connected = MagicMock(side_effect=RuntimeError('boom'))
        scope.led = MagicMock()
        scope.led.is_connected = MagicMock(return_value=True)
        scope.led.firmware_version = '4.0.0'
        scope.led.connect_latency_summary = None
        scope.get_fan_status = MagicMock(return_value=None)

        config_helpers.log_system_metrics(_stub_settings(tmp_path), scope=scope)
        line = capture_logs.firmware_line()
        assert line is not None
        assert 'led=4.0.0' in line
        assert 'motor=' not in line
