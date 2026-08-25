# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: the [PROCESS METRICS] process-CPU figure must be a real reading.

system_metrics() built a fresh psutil.Process every call and read
proc.cpu_percent() off it. psutil.Process.cpu_percent() reports CPU since the
PRIOR call on the SAME object, so a fresh object always returns 0.0 -- the
process-CPU figure logged 0.0% on every snapshot, making per-process CPU (e.g.
an IDS-vs-Pylon comparison) unobservable. The fix caches + primes a single
Process handle so the reading is the delta over the inter-snapshot interval.
"""

from modules import common_utils


def test_self_process_is_cached():
    # The handle must be the SAME object across calls, or cpu_percent() can never
    # compute a delta and always reads 0.0.
    common_utils._SELF_PROC = None  # clean slate for the assertion
    p1 = common_utils._self_process()
    p2 = common_utils._self_process()
    assert p1 is p2


def test_system_metrics_uses_cached_process_not_a_fresh_one(monkeypatch):
    # system_metrics() must route through the cached handle, never construct a
    # fresh psutil.Process (the bug that pinned process CPU to 0.0%).
    common_utils._SELF_PROC = None
    constructed = []
    real_process = common_utils.psutil.Process

    def _counting_process(*args, **kwargs):
        constructed.append(args)
        return real_process(*args, **kwargs)

    monkeypatch.setattr(common_utils.psutil, 'Process', _counting_process)

    common_utils.system_metrics(collect_open_files=False)
    common_utils.system_metrics(collect_open_files=False)

    # The handle is built once (first call), then reused -- not per snapshot.
    assert len(constructed) == 1
