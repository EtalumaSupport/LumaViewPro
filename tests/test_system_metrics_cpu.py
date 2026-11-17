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


_LinuxCpuTimes = namedtuple('_LinuxCpuTimes', 'user nice system idle iowait guest guest_nice')


def test_the_busy_percent_of_two_readings_is_their_busy_share():
    before = _CpuTimes(100.0, 0.0, 50.0, 850.0)
    after = _CpuTimes(200.0, 0.0, 73.0, 1727.0)
    assert round(common_utils.cpu_busy_percent(before, after), 1) == 12.3


def test_a_field_that_ran_backwards_counts_as_no_time():
    # Each field's own delta is floored at zero, as psutil floors it: a
    # counter read lower than before adds nothing, busy or idle.
    before = _CpuTimes(100.0, 5.0, 50.0, 850.0)
    after = _CpuTimes(150.0, 0.0, 50.0, 900.0)
    assert common_utils.cpu_busy_percent(before, after) == 50.0


def test_guest_time_is_left_out_and_iowait_is_idle():
    # Linux counts guest time inside user already, and iowait as idle.
    before = _LinuxCpuTimes(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    after = _LinuxCpuTimes(30.0, 0.0, 10.0, 40.0, 20.0, 30.0, 5.0)
    assert common_utils.cpu_busy_percent(before, after) == 40.0


def test_two_readings_with_no_time_between_have_no_busy_percent():
    times = _CpuTimes(100.0, 0.0, 50.0, 850.0)
    assert common_utils.cpu_busy_percent(times, times) is None
