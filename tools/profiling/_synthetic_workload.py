# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A synthetic CPU workload with a KNOWN per-function split, for validating the
profiler against ground truth.

The profiler's load-bearing premise is ``absolute_cpu[fn] = share[fn] x
total_process_cpu``. On synthetic samples that is unit-tested arithmetic; on a
REAL running process it is a hypothesis until measured -- py-spy's self-sample
shares must actually reflect the true CPU-time split. This module is that known
truth: it burns a configured fraction of its time in two distinctly-named leaf
functions (default 70 / 30) plus a sleeping thread that must NOT be counted as
CPU. ``tools.profiling.ground_truth`` attaches the real profiler to it and
asserts the recovered split matches within the sampling-error band.

Run it standalone (it prints its PID and burns for a duration):

    python -m tools.profiling._synthetic_workload --split 0.7 --duration 60

The two hot functions share an IDENTICAL inner loop but are deliberately NOT
factored into one helper: py-spy attributes self-time to the LEAF frame, so the
split is only observable if each fraction executes under its own function name.
"""

from __future__ import annotations

import argparse
import os
import threading
import time

# Work per outer cycle, split between the two hot functions by the ratio. Large
# enough that each hot call dwarfs the outer-loop bookkeeping (whose samples land
# on run_workload, not the hot leaves), small enough to re-enter both functions
# thousands of times over a normal window so samples accumulate on each.
ITERS_PER_CYCLE = 200_000

# Names the ground-truth check looks for. Kept here so the two agree by import,
# not by a copy-pasted string literal.
HOT_A = '_hot_a'
HOT_B = '_hot_b'


def iter_counts(split: float, total: int = ITERS_PER_CYCLE) -> tuple[int, int]:
    """Partition ``total`` iterations into (a, b) at the given split.

    Same primitive op in each hot function means time is proportional to
    iteration count, so this ratio IS the CPU-time split the profiler must
    recover -- machine-independent, no calibration.

    Args:
        split: fraction of work for ``_hot_a`` (0..1); the rest goes to ``_hot_b``.
        total: total iterations per cycle to divide.

    Returns:
        ``(a_iters, b_iters)`` summing to ``total``.
    """
    if not 0.0 <= split <= 1.0:
        raise ValueError(f'split must be in [0, 1], got {split}')
    a_iters = round(total * split)
    return a_iters, total - a_iters


def _hot_a(iterations: int) -> int:
    # Pure inline arithmetic (an LCG step): no calls, so this frame is the leaf.
    # Body is intentionally identical to _hot_b -- see module docstring.
    acc = 1
    for _ in range(iterations):
        acc = (acc * 1103515245 + 12345) & 0x7FFFFFFF
    return acc


def _hot_b(iterations: int) -> int:
    acc = 1
    for _ in range(iterations):
        acc = (acc * 1103515245 + 12345) & 0x7FFFFFFF
    return acc


def _idle_sleeper(stop: threading.Event) -> None:
    # Sleeping releases the GIL and burns no CPU; py-spy excludes idle threads by
    # default, so this must contribute ~0 samples. It exists so the ground-truth
    # check can confirm idle time is not miscounted as CPU.
    while not stop.is_set():
        time.sleep(0.05)


def run_workload(split: float, duration_s: float) -> None:
    """Burn CPU at the configured split for ``duration_s`` seconds."""
    a_iters, b_iters = iter_counts(split)
    stop = threading.Event()
    sleeper = threading.Thread(target=_idle_sleeper, args=(stop,), daemon=True)
    sleeper.start()
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        _hot_a(a_iters)
        _hot_b(b_iters)
    stop.set()
    sleeper.join(timeout=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Known-split synthetic CPU workload.')
    parser.add_argument('--split', type=float, default=0.7, help='fraction of CPU in _hot_a')
    parser.add_argument('--duration', type=float, default=60.0, help='seconds to run')
    args = parser.parse_args(argv)
    # Flush so an attaching orchestrator sees the PID immediately.
    print(f'synthetic_workload pid={os.getpid()} split={args.split}', flush=True)
    run_workload(args.split, args.duration)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
