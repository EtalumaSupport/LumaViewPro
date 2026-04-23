# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for SerialBoard connect-time latency fingerprint.

The bench fires at the end of `SerialBoard.connect()` and stores a
summary dict on `board.connect_latency_summary`. Verifies:
  - skip when LVP_SKIP_CONNECT_BENCH=1 (conftest default)
  - skip when firmware_version is None
  - skip when subclass returns an empty callable list
  - runs when env var unset + firmware known + subclass opts in
  - exceptions in bench never break connect
  - MotorBoard + LEDBoard override with the expected driver methods
"""
import os
from unittest.mock import MagicMock

import pytest

from drivers.serialboard import SerialBoard


@pytest.fixture
def board_factory(monkeypatch):
    """Build a bare SerialBoard with connect-time infra stubbed out.

    Skips port discovery, pyserial, and firmware reset so we can
    exercise _run_connect_latency_bench in isolation.
    """
    def _factory(firmware_version='4.0.0', bench_callables=None,
                 bench_enabled=True):
        board = SerialBoard.__new__(SerialBoard)
        board.__init__(vid=0, pid=0, label='[TEST]', port='/dev/fake')
        # Wire the stubbed connect path.
        monkeypatch.setattr(board, '_open_serial', lambda: setattr(board, 'driver', MagicMock()))
        monkeypatch.setattr(board, '_close_driver', lambda: setattr(board, 'driver', None))
        monkeypatch.setattr(board, '_reset_firmware',
                            lambda: setattr(board, 'firmware_version', firmware_version))
        if bench_callables is not None:
            monkeypatch.setattr(board, '_connect_bench_callables',
                                lambda: bench_callables)
        if bench_enabled:
            monkeypatch.delenv('LVP_SKIP_CONNECT_BENCH', raising=False)
        else:
            monkeypatch.setenv('LVP_SKIP_CONNECT_BENCH', '1')
        return board
    return _factory


class TestConnectLatencyBench:
    def test_skipped_by_env_var_default(self, board_factory):
        # conftest sets LVP_SKIP_CONNECT_BENCH=1 by default.
        fn = MagicMock()
        board = board_factory(
            firmware_version='4.0.0',
            bench_callables=[('fullinfo', fn)],
            bench_enabled=False,
        )
        board.connect()
        assert board.connect_latency_summary is None
        assert fn.call_count == 0

    def test_skipped_when_firmware_unknown(self, board_factory):
        fn = MagicMock()
        board = board_factory(
            firmware_version=None,
            bench_callables=[('fullinfo', fn)],
            bench_enabled=True,
        )
        board.connect()
        assert board.connect_latency_summary is None
        assert fn.call_count == 0

    def test_skipped_when_no_callables(self, board_factory):
        # Default base-class _connect_bench_callables returns [].
        board = board_factory(
            firmware_version='4.0.0',
            bench_callables=[],
            bench_enabled=True,
        )
        board.connect()
        assert board.connect_latency_summary is None

    def test_runs_when_enabled_and_callable_provided(self, board_factory):
        fn = MagicMock()
        board = board_factory(
            firmware_version='4.0.0',
            bench_callables=[('fullinfo', fn)],
            bench_enabled=True,
        )
        board.connect()
        # iterations (20) + warmup (3) = 23 calls
        assert fn.call_count == board._CONNECT_BENCH_ITERATIONS + board._CONNECT_BENCH_WARMUP
        assert board.connect_latency_summary is not None
        assert 'fullinfo' in board.connect_latency_summary
        assert board.connect_latency_summary['fullinfo']['count'] == 20

    def test_bench_exception_does_not_break_connect(self, board_factory):
        fn = MagicMock(side_effect=RuntimeError('always fails'))
        board = board_factory(
            firmware_version='4.0.0',
            bench_callables=[('fullinfo', fn)],
            bench_enabled=True,
        )
        # connect() must not raise — bench failures are swallowed.
        board.connect()
        # All iterations recorded as errors, count=0 but summary exists.
        summary = board.connect_latency_summary
        assert summary is not None
        assert summary['fullinfo']['count'] == 0
        assert summary['fullinfo']['errors'] == 20

    def test_inner_unexpected_exception_also_swallowed(self, board_factory, monkeypatch):
        # If the latency module itself errors (shouldn't, but
        # defensive), connect() still succeeds and the summary is
        # left at None. Force the failure by monkeypatching the
        # measurement entry point.
        board = board_factory(
            firmware_version='4.0.0',
            bench_callables=[('m', MagicMock())],
            bench_enabled=True,
        )
        from drivers import serial_latency
        monkeypatch.setattr(
            serial_latency, 'measure_callable_latencies',
            MagicMock(side_effect=RuntimeError('latency bug')),
        )
        board.connect()
        assert board.connect_latency_summary is None


class TestSubclassOverrides:
    def test_motorboard_returns_fullinfo(self):
        from drivers.motorboard import MotorBoard
        # Construct via __new__ to skip real hardware init — we only
        # need the bound method lookup.
        m = MotorBoard.__new__(MotorBoard)
        m.fullinfo = MagicMock()
        named = m._connect_bench_callables()
        assert len(named) == 1
        assert named[0][0] == 'fullinfo'
        assert named[0][1] is m.fullinfo

    def test_ledboard_returns_get_info(self):
        from drivers.ledboard import LEDBoard
        led = LEDBoard.__new__(LEDBoard)
        led.get_info = MagicMock()
        named = led._connect_bench_callables()
        assert len(named) == 1
        assert named[0][0] == 'get_info'
        assert named[0][1] is led.get_info

    def test_base_class_returns_empty(self):
        b = SerialBoard.__new__(SerialBoard)
        assert b._connect_bench_callables() == []


class TestMotorBoardConnectOverrideFiresHook:
    """MotorBoard.connect() re-implements the connection sequence instead
    of delegating to SerialBoard.connect(), so the connect-time latency
    bench hook has to fire from the override too — not just the base
    class. Caught on bench 2026-04-24: SN 115 LED summary populated,
    motor summary stayed None.
    """

    def test_motorboard_connect_fires_bench_hook(self, monkeypatch):
        import threading
        from drivers.motorboard import MotorBoard

        m = MotorBoard.__new__(MotorBoard)
        # Minimal SerialBoard state needed for the override's `with
        # self._lock:` + the sequence inside. Bypass __init__ entirely.
        m._lock = threading.RLock()
        m._state_lock = threading.Lock()
        m.driver = None
        m.port = '/dev/fake'
        m._connect_fails = 0
        m._connect_log_suppressed = False
        m._fullinfo = None
        m._CONNECT_BENCH_ITERATIONS = 5
        m._CONNECT_BENCH_WARMUP = 1
        m.connect_latency_summary = None

        # Stub every sub-call the override makes: open_serial / driver
        # close+reopen / _reset_firmware / fullinfo.
        fake_driver = MagicMock()
        fake_driver.is_open = True
        def _open_serial():
            m.driver = fake_driver
        monkeypatch.setattr(m, '_open_serial', _open_serial)
        monkeypatch.setattr(m, '_reset_firmware',
                            lambda: setattr(m, 'firmware_version', '4.0.0'))
        # fullinfo is the benched callable — must be a MagicMock that
        # responds so the bench records successes, not errors.
        m.fullinfo = MagicMock(return_value={'model': 'sim', 'serial_number': '0'})
        monkeypatch.delenv('LVP_SKIP_CONNECT_BENCH', raising=False)

        m.connect()

        # The whole point: connect_latency_summary must be populated.
        assert m.connect_latency_summary is not None, (
            'MotorBoard.connect override skipped the connect-bench hook'
        )
        assert 'fullinfo' in m.connect_latency_summary
        summary = m.connect_latency_summary['fullinfo']
        assert summary['count'] == m._CONNECT_BENCH_ITERATIONS
        assert summary['errors'] == 0

    def test_motorboard_connect_respects_env_var(self, monkeypatch):
        import threading
        from drivers.motorboard import MotorBoard

        m = MotorBoard.__new__(MotorBoard)
        m._lock = threading.RLock()
        m._state_lock = threading.Lock()
        m.driver = None
        m.port = '/dev/fake'
        m._connect_fails = 0
        m._connect_log_suppressed = False
        m._fullinfo = None
        m.connect_latency_summary = None

        fake_driver = MagicMock()
        fake_driver.is_open = True
        monkeypatch.setattr(m, '_open_serial',
                            lambda: setattr(m, 'driver', fake_driver))
        monkeypatch.setattr(m, '_reset_firmware',
                            lambda: setattr(m, 'firmware_version', '4.0.0'))
        m.fullinfo = MagicMock(return_value={})
        monkeypatch.setenv('LVP_SKIP_CONNECT_BENCH', '1')

        m.connect()

        # Env var opt-out honored by the override path too.
        assert m.connect_latency_summary is None
