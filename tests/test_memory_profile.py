# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The memory profiler turns on only, and only when it is off.

A disabled call changes nothing: the GUI's second session in one process
(its fallback onto the template, where the setting is off) must neither stop
a running profiler's snapshots nor keep a later enabled call from starting
it. Disabled, every entry point is a no-op.
"""

import threading

import pytest

from _lumascope.lib import memory_profile


@pytest.fixture
def fresh_profiler(monkeypatch):
    """The profiler as a process that has not started it finds it.

    A stand-in by design: tracemalloc and the sampler thread are
    process-wide and never stop, so the test hands ``start`` a tracer that is
    already tracing (it starts none) and a thread that never runs.
    """
    import tracemalloc

    monkeypatch.setattr(memory_profile, '_started', False)
    monkeypatch.setattr(memory_profile, '_enabled', False)
    monkeypatch.setattr(memory_profile, '_prev_snapshot', None)
    monkeypatch.setattr(tracemalloc, 'is_tracing', lambda: True)
    monkeypatch.setattr(memory_profile.threading, 'Thread', _UnstartedThread)


class _UnstartedThread(threading.Thread):
    def start(self):
        pass


def test_a_disabled_call_then_an_enabled_one_starts_it(fresh_profiler):
    memory_profile.start(enabled=False, interval_s=5.0)
    memory_profile.start(enabled=True, interval_s=2.0)

    assert memory_profile.is_enabled() is True


def test_an_enabled_call_then_a_disabled_one_leaves_it_on(fresh_profiler):
    memory_profile.start(enabled=True, interval_s=2.0)
    memory_profile.start(enabled=False, interval_s=5.0)

    assert memory_profile.is_enabled() is True


def test_a_disabled_call_leaves_it_off_and_a_snapshot_does_nothing(fresh_profiler, monkeypatch):
    import tracemalloc

    taken = []
    monkeypatch.setattr(tracemalloc, 'take_snapshot', lambda: taken.append(1))

    memory_profile.start(enabled=False, interval_s=5.0)
    memory_profile.snapshot('any_state')

    assert memory_profile.is_enabled() is False
    assert taken == []
