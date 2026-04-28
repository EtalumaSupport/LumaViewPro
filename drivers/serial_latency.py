# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Per-operation round-trip latency measurement.

Shared primitive used by SerialBoard (connect-time fingerprint),
`tools/firmware_tools.py bench` (release-gate campaign CLI), and
`FirmwareDiagnostics.measure_serial_latency` (tech-support report).
Pure functions — no Kivy, no Lumascope, no module-layer imports.
Sits in drivers/ so the three consumers above can all import down.

Single entry point: `measure_callable_latencies(named_callables,
iterations, warmup)`. Benchmarks driver-level methods (e.g.
`motor.fullinfo`). The driver dispatches v3.0.x vs FW4.0 internally,
so the measurement is cross-firmware-comparable by construction.
The release-gate §2.3 FW4.0-vs-v3.0.x comparison works because the
same driver call hits the drain-sleep path on v3.0.x and the
event-driven path on FW4.0.

Per Architecture Rule 22, no raw-command escape hatch exists — every
wire command must route through a driver method. To bench a specific
command, expose it as a driver method and pass the bound method here.

See docs/FW40_RELEASE_GATE.md §2.3 for the motivating thesis: FW4.0
drops the v3.0.x post-response drain sleep, so round-trip latency on
read commands should fall by ≥20 ms per call. Every connect captures
a lightweight fingerprint (via measure_callable_latencies) so
fleet-wide FW4.0-vs-v3.0.x comparison falls out of normal operation.
"""
import math
import time


def measure_callable_latencies(named_callables, iterations=20, warmup=3,
                               return_durations=False):
    """Measure per-method round-trip latency.

    Args:
        named_callables: Iterable of `(name, callable)` tuples. Each
            callable is invoked with zero arguments per iteration —
            typically a bound driver method like `board.fullinfo` or
            `board.get_info` that dispatches internally based on
            firmware version.
        iterations: Measured iterations per method.
        warmup: Warmup iterations per method, discarded from stats.
        return_durations: When True, returns `(summaries, durations)`
            where `durations` is `{name: [us | None, ...]}` — the
            raw per-iteration values. Use when the caller needs to
            emit raw data (e.g. CSV export) so only one measurement
            pass runs.

    Returns:
        `{name: summary_dict}` per method, or `(summaries, durations)`
        when `return_durations=True`. See `_summarize` for dict shape.
        On all-errors, latency fields are None and errors > 0.

    Errors are exceptions raised by the callable. Successful returns
    that yield empty / None values still count as successful round
    trips — the measurement is end-to-end driver-return wall time.
    """
    summaries = {}
    raw_by_name = {} if return_durations else None
    for name, fn in named_callables:
        durations = _measure_callable(fn, iterations, warmup)
        summaries[name] = _summarize(durations)
        if return_durations:
            raw_by_name[name] = durations
    if return_durations:
        return summaries, raw_by_name
    return summaries


def _measure_callable(fn, iterations, warmup):
    """Run warmup + measured iterations of one callable.

    Returns a list of per-iteration durations in microseconds. Failed
    iterations (exception) are recorded as None so the summarizer can
    separate error count from latency stats.
    """
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            pass  # warmup errors ignored

    durations = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        try:
            fn()
        except Exception:
            durations.append(None)
            continue
        t1 = time.perf_counter_ns()
        durations.append((t1 - t0) / 1000.0)
    return durations


def _summarize(durations):
    """Collapse a list of durations (None = error) to a summary dict.

    Keys: count, errors, mean_us, stddev_us, p50_us, p95_us, p99_us,
    min_us, max_us. On all-errors, latency fields are None.
    """
    valid = [d for d in durations if d is not None]
    errors = len(durations) - len(valid)
    if not valid:
        return {
            'count': 0, 'errors': errors,
            'mean_us': None, 'stddev_us': None,
            'p50_us': None, 'p95_us': None, 'p99_us': None,
            'min_us': None, 'max_us': None,
        }
    ordered = sorted(valid)
    n = len(ordered)
    mean = sum(ordered) / n
    variance = sum((d - mean) ** 2 for d in ordered) / n
    stddev = math.sqrt(variance)

    def pct(p):
        k = max(1, int(math.ceil(p * n / 100.0)))
        return ordered[min(k, n) - 1]

    return {
        'count': n, 'errors': errors,
        'mean_us': mean, 'stddev_us': stddev,
        'p50_us': pct(50), 'p95_us': pct(95), 'p99_us': pct(99),
        'min_us': ordered[0], 'max_us': ordered[-1],
    }


def run_load_loop(fn, duration_seconds, hz):
    """Invoke `fn` at `hz` Hz for `duration_seconds`.

    Release gate §2.3 reliability-under-load test. Shared between the
    CLI (`bench --load-minutes`) and any future integration.

    Scheduling: each iteration measures its own start, invokes `fn`,
    sleeps the remainder of `1.0 / hz` before the next iteration. If
    an iteration overruns the tick budget, sleep is skipped (no
    catch-up spin); `actual_hz` falls below `target_hz`.

    Returns the same summary as `_summarize` plus load-specific
    fields: `duration_s` (wall-clock elapsed), `target_hz`,
    `actual_hz` (successes / duration), `errors_per_hour`.
    """
    interval_s = 1.0 / hz
    durations = []
    t_start = time.perf_counter()
    deadline = t_start + duration_seconds
    while time.perf_counter() < deadline:
        t_tick = time.perf_counter()
        t0 = time.perf_counter_ns()
        try:
            fn()
        except Exception:
            durations.append(None)
        else:
            t1 = time.perf_counter_ns()
            durations.append((t1 - t0) / 1000.0)
        sleep_needed = interval_s - (time.perf_counter() - t_tick)
        if sleep_needed > 0:
            time.sleep(sleep_needed)

    elapsed_s = time.perf_counter() - t_start
    summary = _summarize(durations)
    summary['duration_s'] = elapsed_s
    summary['target_hz'] = hz
    summary['actual_hz'] = (summary['count'] / elapsed_s if elapsed_s > 0 else 0.0)
    summary['errors_per_hour'] = (
        summary['errors'] * 3600.0 / elapsed_s if elapsed_s > 0 else 0.0
    )
    return summary


def format_one_line(board_label, firmware_version, summary):
    """Single-line log summary, one segment per measurement.

    Intended for the INFO log line SerialBoard writes at connect. µs
    converts to ms so field reports are human-readable at a glance.
    """
    segments = []
    for name, s in summary.items():
        if s['count'] == 0:
            segments.append(f"{name} ALL-FAILED ({s['errors']} err)")
            continue
        mean_ms = s['mean_us'] / 1000.0
        p95_ms = s['p95_us'] / 1000.0
        segments.append(
            f"{name} mean={mean_ms:.2f}ms p95={p95_ms:.2f}ms err={s['errors']}"
        )
    return (
        f"[LATENCY] {board_label} fw={firmware_version or 'unknown'} | "
        + " | ".join(segments)
    )
