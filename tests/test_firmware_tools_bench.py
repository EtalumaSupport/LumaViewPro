# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for `tools/firmware_tools.py` bench subcommand helpers.

Covers the pure functions: latency summarizing, CSV output, summary-
table formatting, and the measurement loop driven by a MagicMock
board (no hardware). The bench subcommand itself is hardware-gated
and not exercised here.
"""
import csv
import math
from unittest.mock import MagicMock

import pytest

from tools.firmware_tools import (
    _BENCH_DEFAULT_COMMANDS,
    _format_summary_table,
    _measure_latencies,
    _summarize_latencies,
    _write_bench_csv,
)


# ---------------------------------------------------------------------------
# _summarize_latencies
# ---------------------------------------------------------------------------

class TestSummarizeLatencies:
    def test_empty_list(self):
        s = _summarize_latencies([])
        assert s['count'] == 0
        assert s['errors'] == 0
        assert s['mean_us'] is None

    def test_all_errors(self):
        s = _summarize_latencies([None, None, None])
        assert s['count'] == 0
        assert s['errors'] == 3
        assert s['mean_us'] is None

    def test_mean_and_minmax(self):
        s = _summarize_latencies([100.0, 200.0, 300.0])
        assert s['count'] == 3
        assert s['errors'] == 0
        assert s['mean_us'] == pytest.approx(200.0)
        assert s['min_us'] == pytest.approx(100.0)
        assert s['max_us'] == pytest.approx(300.0)

    def test_stddev(self):
        # population stddev of [100,200,300] = sqrt(((100)²+0+(100)²)/3) ≈ 81.65
        s = _summarize_latencies([100.0, 200.0, 300.0])
        assert s['stddev_us'] == pytest.approx(math.sqrt(20000.0 / 3))

    def test_percentiles_nearest_rank(self):
        # 1..100, sorted — nearest-rank percentiles are exact
        durations = [float(i) for i in range(1, 101)]
        s = _summarize_latencies(durations)
        assert s['p50_us'] == 50.0
        assert s['p95_us'] == 95.0
        assert s['p99_us'] == 99.0

    def test_errors_mixed_with_valid(self):
        s = _summarize_latencies([50.0, None, 100.0, None, 150.0])
        assert s['count'] == 3
        assert s['errors'] == 2
        assert s['mean_us'] == pytest.approx(100.0)

    def test_single_value(self):
        s = _summarize_latencies([42.0])
        assert s['count'] == 1
        assert s['mean_us'] == 42.0
        assert s['stddev_us'] == 0.0
        assert s['p50_us'] == 42.0
        assert s['p95_us'] == 42.0
        assert s['p99_us'] == 42.0


# ---------------------------------------------------------------------------
# _measure_latencies
# ---------------------------------------------------------------------------

class TestMeasureLatencies:
    def test_invokes_warmup_then_measured(self):
        board = MagicMock()
        durations = _measure_latencies(board, 'INFO', iterations=10, warmup=3)
        assert board.exchange_command.call_count == 13
        assert len(durations) == 10
        # All valid — MagicMock doesn't raise.
        assert all(d is not None for d in durations)

    def test_captures_exception_as_none(self):
        board = MagicMock()
        # First 3 raise, next 2 succeed.
        board.exchange_command.side_effect = (
            [RuntimeError('boom')] * 3
            + [None] * 2  # no exception; returns None (ignored)
        )
        durations = _measure_latencies(board, 'INFO', iterations=5, warmup=0)
        assert durations[:3] == [None, None, None]
        assert durations[3] is not None
        assert durations[4] is not None

    def test_warmup_errors_ignored_but_dont_count(self):
        board = MagicMock()
        call_seq = [RuntimeError('warmup boom')] * 2 + [None] * 5
        board.exchange_command.side_effect = call_seq
        durations = _measure_latencies(board, 'INFO', iterations=5, warmup=2)
        # All 5 measured iterations succeeded; warmup errors didn't pollute.
        assert all(d is not None for d in durations)
        assert len(durations) == 5


# ---------------------------------------------------------------------------
# _write_bench_csv
# ---------------------------------------------------------------------------

class TestWriteBenchCsv:
    def test_writes_header_and_rows(self, tmp_path):
        out = tmp_path / 'bench.csv'
        rows = [
            ('motor', '4.0.0', 'INFO', 0, 1234.5),
            ('motor', '4.0.0', 'INFO', 1, None),        # error row
            ('motor', '4.0.0', 'POS_READ', 0, 567.8),
        ]
        _write_bench_csv(out, rows)

        with open(out) as f:
            reader = csv.reader(f)
            got = list(reader)

        assert got[0] == ['board', 'firmware_version', 'command',
                          'iteration', 'duration_us']
        assert got[1] == ['motor', '4.0.0', 'INFO', '0', '1234.5']
        assert got[2] == ['motor', '4.0.0', 'INFO', '1', '']  # None → empty
        assert got[3] == ['motor', '4.0.0', 'POS_READ', '0', '567.8']


# ---------------------------------------------------------------------------
# _format_summary_table
# ---------------------------------------------------------------------------

class TestFormatSummaryTable:
    def test_includes_all_commands_and_stats(self):
        summary = {
            'INFO': _summarize_latencies([100.0, 200.0, 300.0]),
            'STATUS': _summarize_latencies([50.0, None, 60.0]),
        }
        table = _format_summary_table(summary)
        assert 'INFO' in table
        assert 'STATUS' in table
        assert 'command' in table  # header row
        assert 'mean_us' in table

    def test_em_dash_for_none_stats(self):
        # All-errors row must not crash on None stats.
        summary = {'INFO': _summarize_latencies([None, None])}
        table = _format_summary_table(summary)
        assert 'INFO' in table
        # Each stat column renders as em dash for an all-None row.
        assert '—' in table


# ---------------------------------------------------------------------------
# _BENCH_DEFAULT_COMMANDS — guards against accidental drift between the
# CLI's default command set and what FW4.0 actually exposes.
# ---------------------------------------------------------------------------

class TestBenchDefaults:
    def test_motor_defaults_are_read_only_commands(self):
        # FW4.0 motor: INFO + POS_READ + STATUS + LIMIT_SW (verified by
        # grep of Motor Controller/Firmware/main.py). LED_SET would be
        # a write; LOAD_CONFIG would be a write. Any future addition
        # must stay on the read side to keep bench runs safe.
        assert _BENCH_DEFAULT_COMMANDS['motor'] == (
            'INFO', 'POS_READ', 'STATUS', 'LIMIT_SW'
        )

    def test_led_defaults_are_read_only_commands(self):
        # FW4.0 LED: INFO + LED_READ + STATUS. LED_SET is intentionally
        # excluded from defaults — it writes DAC state; caller can opt
        # in via --commands.
        assert _BENCH_DEFAULT_COMMANDS['led'] == (
            'INFO', 'LED_READ', 'STATUS'
        )

    def test_no_led_set_in_defaults(self):
        assert 'LED_SET' not in _BENCH_DEFAULT_COMMANDS['motor']
        assert 'LED_SET' not in _BENCH_DEFAULT_COMMANDS['led']
