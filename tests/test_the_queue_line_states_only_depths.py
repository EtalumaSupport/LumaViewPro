# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every number on the watchdog's queue line is a lane's queue depth.

The protocol and display threads have no queue; the line printed them as
0 (running) or -1 (not), so an idle session read ``PROTOCOL:-1``, a depth
below zero. A lane whose depth could not be read printed -1 the same way.
The threads' states are words now, and a failed read fails the tick.
"""

from __future__ import annotations

import logging

import pytest

import modules.metrics_logger as metrics_logger_module
from tests.log_capture import records_of


def _queue_line_pairs(caplog) -> dict[str, str]:
    (line,) = [
        r.getMessage()
        for r in records_of(caplog, metrics_logger_module)
        if 'Queues --' in r.getMessage()
    ]
    return dict(pair.split(':') for pair in line.split('Queues -- ')[1].split())


def test_an_idle_sessions_line_states_depths_and_the_threads_in_words(sim_session, caplog):
    caplog.set_level(logging.DEBUG, logger='lvp_logger')
    sim_session.metrics_logger.tick_executor_watchdog()

    pairs = _queue_line_pairs(caplog)
    threads = {name: pairs.pop(name) for name in ('PROTOCOL', 'SCOPEDISPLAY')}
    # No run is in flight, and a host with no display never starts its thread.
    assert threads == {'PROTOCOL': 'idle', 'SCOPEDISPLAY': 'off'}
    assert set(pairs) == {'IO', 'CAMERA', 'FILE', 'POSTPROC', 'WORKER_POOL', 'DIAGNOSTICS'}
    assert all(depth.isdigit() for depth in pairs.values()), pairs


def test_a_lane_whose_depth_cannot_be_read_fails_the_tick(sim_session, monkeypatch):
    # a stand-in by design: no lane's depth read can be made to fail on the
    # simulator; the failure must reach the scheduler's error hook, never
    # print as a depth.
    def broken():
        raise RuntimeError('depth unreadable')

    monkeypatch.setattr(sim_session.executor_bundle.file_io_executor, 'queue_size', broken)
    with pytest.raises(RuntimeError, match='depth unreadable'):
        sim_session.metrics_logger.tick_executor_watchdog()
