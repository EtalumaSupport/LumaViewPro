# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Turn py-spy folded-stack samples + a total-CPU number into absolute,
ranked per-function CPU.

The chain is deliberately simple and testable without py-spy:

  1. py-spy `-f raw` emits Brendan-Gregg folded stacks -- one line per distinct
     stack: ``frame;frame;...;leaf COUNT``. The COUNT is the number of samples
     that captured that exact stack.
  2. A function's SELF samples are the ones where it is the LEAF (the frame
     actually executing). Self-time is the right metric for "who is burning
     CPU," not cumulative-through-callers time.
  3. share = self_samples / total_samples is what py-spy really measures. It is
     a fraction of sampled on-CPU time, NOT an absolute -- a share is only a
     share until multiplied by a real total.
  4. absolute = share x total_process_cpu_cores. The total comes from psutil
     (100% = one core), sampled over the SAME window. This multiply -- not a
     fancier profiler -- is what makes the number absolute and comparable.
  5. error is the sampling (binomial) uncertainty: more samples -> tighter.
     Reported so a reader never treats a 3-sample function as a hard number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionCpu:
    """Absolute CPU attributed to one function (its self / leaf time)."""

    function: str
    self_samples: int
    share: float  # fraction of total on-CPU samples (0..1)
    cpu_cores: float  # absolute: share * total_process_cpu_cores
    cpu_ms_per_s: float  # cpu_cores * 1000 (ms of CPU per wall-second)
    err_cores_95: float  # +/- 95% sampling error, in cores


def parse_folded(text: str) -> tuple[dict[str, int], int, int]:
    """Parse py-spy folded-stack output into self-sample counts per leaf.

    Returns ``(self_counts, total_samples, skipped_lines)``. A folded line is
    ``frame;frame;...;leaf COUNT``; the leaf is the last frame and gets the
    samples. Frame text may contain spaces (``func (file.py:12)``), so the
    count is split off the RIGHT. Lines that do not end in an integer count are
    skipped and tallied (never silently dropped) so a format drift is visible.
    """
    self_counts: dict[str, int] = {}
    total = 0
    skipped = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        stack, _, count_str = line.rpartition(' ')
        if not stack:
            skipped += 1
            continue
        try:
            count = int(count_str)
        except ValueError:
            skipped += 1
            continue
        leaf = stack.rsplit(';', 1)[-1].strip()
        if not leaf:
            skipped += 1
            continue
        self_counts[leaf] = self_counts.get(leaf, 0) + count
        total += count
    return self_counts, total, skipped


def compute_absolute_cpu(
    self_counts: dict[str, int],
    total_samples: int,
    total_process_cpu_cores: float,
) -> list[FunctionCpu]:
    """Join per-function sample shares with the process total to get absolute,
    ranked (descending) per-function CPU.

    ``total_process_cpu_cores`` is the mean process CPU over the SAME window as
    the samples, in cores (psutil percent / 100; 164% -> 1.64). The share->
    absolute multiply lives here so a share is never mistaken for an absolute.
    """
    if total_samples <= 0:
        return []
    results: list[FunctionCpu] = []
    for function, count in self_counts.items():
        share = count / total_samples
        cpu_cores = share * total_process_cpu_cores
        # 95% sampling half-width on the share (binomial), scaled to cores. The
        # +-1.96*sqrt(p(1-p)/N) band shrinks as 1/sqrt(N): the honest signal
        # that a low-sample function is a hint, not a measurement.
        err_share = 1.96 * math.sqrt(share * (1.0 - share) / total_samples)
        results.append(
            FunctionCpu(
                function=function,
                self_samples=count,
                share=share,
                cpu_cores=cpu_cores,
                cpu_ms_per_s=cpu_cores * 1000.0,
                err_cores_95=err_share * total_process_cpu_cores,
            )
        )
    results.sort(key=lambda r: r.cpu_cores, reverse=True)
    return results
