# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A test's wait for a run fails on a stall, never on a slow host.

The run tests bounded the whole run by the wall clock, so the 50-step runs
(about 8 s alone against 15 s) failed whenever another suite loaded the
machine. ``wait_for_run_end`` fails only when the run stops starting steps.
"""

from __future__ import annotations

import threading
import time

import tests.protocol_drives as protocol_drives
from tests.protocol_drives import StepHeartbeat, wait_for_run_end


def test_a_run_that_stops_starting_steps_fails(monkeypatch):
    monkeypatch.setattr(protocol_drives, 'STEP_STALL_S', 0.3)
    started = time.monotonic()
    assert wait_for_run_end(threading.Event(), StepHeartbeat()) is False
    assert time.monotonic() - started < 2.0


def test_a_slow_run_that_keeps_starting_steps_ends(monkeypatch):
    # Steps 0.1 s apart for 1.2 s: four times the stall window in all, and
    # never a stall.
    monkeypatch.setattr(protocol_drives, 'STEP_STALL_S', 0.3)
    done = threading.Event()
    heartbeat = StepHeartbeat()

    def run():
        for _ in range(12):
            time.sleep(0.1)
            heartbeat(0)
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert wait_for_run_end(done, heartbeat) is True


def test_the_heartbeat_hands_each_step_to_the_tests_own_callback():
    seen = []
    StepHeartbeat(seen.append)(3)
    assert seen == [3]
