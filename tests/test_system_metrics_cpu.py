# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The [PROCESS METRICS] and [SYSTEM METRICS] CPU figures are real readings.

A fresh psutil.Process every tick read 0.0 process CPU on every tick, since
its baseline lives on the object; psutil's own system cpu_percent() keeps its
baseline per calling thread, and the tick runs on a fresh timer thread each
time, so a tick on a thread with none read 0.0. Each reader now holds its one
baseline, and answers None, never 0.0, until it has one.
"""

import threading
from collections import namedtuple

from _lumascope.modules import common_utils

_CpuTimes = namedtuple('_CpuTimes', 'user nice system idle')


def test_self_process_is_one_handle(monkeypatch):
    monkeypatch.setattr(common_utils, '_SELF_PROC', None)
    assert common_utils.self_process() is common_utils.self_process()


def test_the_system_cpu_baseline_is_kept_whatever_thread_reads_it(monkeypatch):
    # a stand-in by design: the host's CPU times, fixed so the percent is known.
    times = iter([_CpuTimes(100.0, 0.0, 50.0, 850.0), _CpuTimes(200.0, 0.0, 73.0, 1727.0)])
    monkeypatch.setattr(common_utils.psutil, 'cpu_times', lambda: next(times))
    monkeypatch.setattr(common_utils, '_system_cpu_last', {})
    readings = []
    for _ in range(2):
        reader = threading.Thread(target=lambda: readings.append(common_utils.system_cpu_percent()))
        reader.start()
        reader.join()
    assert readings == [None, 12.3]
