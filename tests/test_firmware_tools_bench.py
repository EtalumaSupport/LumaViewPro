# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for the serial-latency primitive + bench-CLI helpers.

The primitive is shared across three consumers:
  - `SerialBoard._run_connect_latency_bench` (connect-time fingerprint)
  - `tools/firmware_tools.py bench` (release-gate campaign CLI)
  - `FirmwareDiagnostics.measure_serial_latency` (tech-support report)

Driver methods are the preferred callables so the v3.0.x / FW4.0
dispatch happens inside the driver — the measurement is then
cross-firmware-comparable by construction. Raw-command form remains
as an ad-hoc escape hatch.
"""
import csv
import math
from unittest.mock import MagicMock

import pytest

from drivers.serial_latency import (
    _summarize,
    format_one_line,
    measure_callable_latencies,
    measure_command_latencies,
    run_load_loop,
)
from tools.firmware_tools import (
    _format_summary_table,
    _write_bench_csv,
)


# ---------------------------------------------------------------------------
# _summarize — latency math (percentiles, error tracking, edge cases)
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_empty_list(self):
        s = _summarize([])
        assert s['count'] == 0
        assert s['errors'] == 0
        assert s['mean_us'] is None

    def test_all_errors(self):
        s = _summarize([None, None, None])
        assert s['count'] == 0
        assert s['errors'] == 3
        assert s['mean_us'] is None

    def test_mean_and_minmax(self):
        s = _summarize([100.0, 200.0, 300.0])
        assert s['count'] == 3
        assert s['errors'] == 0
        assert s['mean_us'] == pytest.approx(200.0)
        assert s['min_us'] == pytest.approx(100.0)
        assert s['max_us'] == pytest.approx(300.0)

    def test_stddev_population(self):
        # population stddev of [100,200,300] = sqrt(((100)²+0+(100)²)/3)
        s = _summarize([100.0, 200.0, 300.0])
        assert s['stddev_us'] == pytest.approx(math.sqrt(20000.0 / 3))

    def test_percentiles_nearest_rank(self):
        # 1..100, sorted — nearest-rank percentiles are exact
        durations = [float(i) for i in range(1, 101)]
        s = _summarize(durations)
        assert s['p50_us'] == 50.0
        assert s['p95_us'] == 95.0
        assert s['p99_us'] == 99.0

    def test_errors_mixed_with_valid(self):
        s = _summarize([50.0, None, 100.0, None, 150.0])
        assert s['count'] == 3
        assert s['errors'] == 2
        assert s['mean_us'] == pytest.approx(100.0)

    def test_single_value(self):
        s = _summarize([42.0])
        assert s['count'] == 1
        assert s['mean_us'] == 42.0
        assert s['stddev_us'] == 0.0
        assert s['p50_us'] == 42.0
        assert s['p95_us'] == 42.0
        assert s['p99_us'] == 42.0


# ---------------------------------------------------------------------------
# measure_callable_latencies — preferred driver-method form
# ---------------------------------------------------------------------------

class TestMeasureCallableLatencies:
    def test_warmup_plus_measured_calls(self):
        fn = MagicMock()
        summaries = measure_callable_latencies(
            [('m1', fn)], iterations=10, warmup=3
        )
        assert fn.call_count == 13
        assert summaries['m1']['count'] == 10
        assert summaries['m1']['errors'] == 0

    def test_multiple_methods_each_gets_own_summary(self):
        fn_a = MagicMock()
        fn_b = MagicMock()
        summaries = measure_callable_latencies(
            [('a', fn_a), ('b', fn_b)], iterations=5, warmup=0
        )
        assert set(summaries.keys()) == {'a', 'b'}
        assert fn_a.call_count == 5
        assert fn_b.call_count == 5

    def test_exception_recorded_as_error(self):
        fn = MagicMock(side_effect=[RuntimeError('boom')] * 3 + [None] * 2)
        summaries = measure_callable_latencies(
            [('m', fn)], iterations=5, warmup=0
        )
        assert summaries['m']['errors'] == 3
        assert summaries['m']['count'] == 2

    def test_warmup_errors_dont_count(self):
        fn = MagicMock(side_effect=[RuntimeError('warmup boom')] * 2 + [None] * 5)
        summaries = measure_callable_latencies(
            [('m', fn)], iterations=5, warmup=2
        )
        assert summaries['m']['errors'] == 0
        assert summaries['m']['count'] == 5

    def test_return_durations_yields_raw_timings(self):
        fn = MagicMock()
        summaries, raw = measure_callable_latencies(
            [('m', fn)], iterations=4, warmup=1,
            return_durations=True,
        )
        assert 'm' in summaries
        assert 'm' in raw
        assert len(raw['m']) == 4
        # All successful — every duration is a positive float, no Nones.
        assert all(d is not None and d >= 0 for d in raw['m'])

    def test_return_durations_errors_as_none(self):
        fn = MagicMock(side_effect=[RuntimeError('x'), None, RuntimeError('x'), None])
        summaries, raw = measure_callable_latencies(
            [('m', fn)], iterations=4, warmup=0,
            return_durations=True,
        )
        assert raw['m'][0] is None
        assert raw['m'][2] is None
        assert raw['m'][1] is not None
        assert raw['m'][3] is not None
        assert summaries['m']['errors'] == 2
        assert summaries['m']['count'] == 2


# ---------------------------------------------------------------------------
# measure_command_latencies — raw-command escape hatch
# ---------------------------------------------------------------------------

class TestMeasureCommandLatencies:
    def test_sends_each_command_through_exchange_command(self):
        board = MagicMock()
        summaries = measure_command_latencies(
            board, ['INFO', 'STATUS'], iterations=3, warmup=1
        )
        # 2 commands × (3 measured + 1 warmup) = 8 calls
        assert board.exchange_command.call_count == 8
        sent = [call_args.args[0] for call_args in board.exchange_command.call_args_list]
        assert sent.count('INFO') == 4
        assert sent.count('STATUS') == 4
        assert set(summaries.keys()) == {'INFO', 'STATUS'}

    def test_return_durations_for_raw_commands(self):
        board = MagicMock()
        summaries, raw = measure_command_latencies(
            board, ['INFO'], iterations=5, warmup=0,
            return_durations=True,
        )
        assert len(raw['INFO']) == 5


# ---------------------------------------------------------------------------
# run_load_loop — reliability loop (release gate §2.3)
# ---------------------------------------------------------------------------

class TestRunLoadLoop:
    def test_returns_load_specific_fields(self):
        fn = MagicMock()
        summary = run_load_loop(fn, duration_seconds=0.2, hz=10)
        assert 'duration_s' in summary
        assert 'actual_hz' in summary
        assert 'target_hz' in summary
        assert 'errors_per_hour' in summary
        assert summary['target_hz'] == 10
        assert summary['actual_hz'] <= summary['target_hz'] + 0.5

    def test_counts_errors(self):
        fn = MagicMock(side_effect=([RuntimeError('boom')] * 100 + [None] * 100))
        summary = run_load_loop(fn, duration_seconds=0.15, hz=50)
        assert summary['errors'] >= 1
        if summary['errors'] > 0:
            assert summary['errors_per_hour'] > 0

    def test_zero_duration_safe(self):
        fn = MagicMock()
        summary = run_load_loop(fn, duration_seconds=0.0, hz=10)
        assert summary['count'] == 0
        assert summary['errors'] == 0
        assert summary['errors_per_hour'] == 0.0

    def test_calls_the_provided_callable(self):
        fn = MagicMock()
        run_load_loop(fn, duration_seconds=0.15, hz=50)
        assert fn.call_count >= 1


# ---------------------------------------------------------------------------
# format_one_line — connect-time log line
# ---------------------------------------------------------------------------

class TestFormatOneLine:
    def test_single_method_summary_line(self):
        summary = {'fullinfo': _summarize([1000.0, 2000.0, 3000.0])}
        line = format_one_line('[XYZ Class ]', '4.0.0', summary)
        assert '[LATENCY]' in line
        assert '[XYZ Class ]' in line
        assert 'fw=4.0.0' in line
        assert 'fullinfo' in line
        assert 'mean=2.00ms' in line

    def test_unknown_firmware_version(self):
        summary = {'get_info': _summarize([500.0])}
        line = format_one_line('[LED Class ]', None, summary)
        assert 'fw=unknown' in line

    def test_all_failed_method_segment(self):
        summary = {'fullinfo': _summarize([None, None])}
        line = format_one_line('[XYZ Class ]', '3.0.9', summary)
        assert 'fullinfo ALL-FAILED' in line
        assert '2 err' in line


# ---------------------------------------------------------------------------
# CLI helpers — _write_bench_csv and _format_summary_table
# ---------------------------------------------------------------------------

class TestWriteBenchCsv:
    def test_writes_header_and_rows(self, tmp_path):
        out = tmp_path / 'bench.csv'
        rows = [
            ('motor', '4.0.0', 'fullinfo', 0, 1234.5),
            ('motor', '4.0.0', 'fullinfo', 1, None),      # error row
            ('motor', '4.0.0', 'fullinfo', 2, 567.8),
        ]
        _write_bench_csv(out, rows)

        with open(out) as f:
            got = list(csv.reader(f))

        assert got[0] == ['board', 'firmware_version', 'method',
                          'iteration', 'duration_us']
        assert got[1] == ['motor', '4.0.0', 'fullinfo', '0', '1234.5']
        assert got[2] == ['motor', '4.0.0', 'fullinfo', '1', '']
        assert got[3] == ['motor', '4.0.0', 'fullinfo', '2', '567.8']


class TestFormatSummaryTable:
    def test_includes_methods_and_stats(self):
        summary = {
            'fullinfo': _summarize([100.0, 200.0, 300.0]),
            'get_status': _summarize([50.0, None, 60.0]),
        }
        table = _format_summary_table(summary)
        assert 'fullinfo' in table
        assert 'get_status' in table
        assert 'method' in table
        assert 'mean_us' in table

    def test_em_dash_for_none_stats(self):
        summary = {'fullinfo': _summarize([None, None])}
        table = _format_summary_table(summary)
        assert 'fullinfo' in table
        assert '—' in table
