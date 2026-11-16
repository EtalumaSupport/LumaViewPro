# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every executor the session's bundle builds is started, listed and stopped by it.

The bundle holds the scope's two lanes beside the executors it builds. What
it built it starts in ``create_default`` and stops in ``shutdown``; every
executor it holds is in ``queued_executors``, under its own name, which the
watchdog, the QUEUE and FUTURES metrics lines and the session's in-flight
read walk. The scope's lanes are the scope's to start and stop.

The built executors are found by identity (every executor field that is not
one of the lanes passed in), so one added later is covered with no edit here.
"""

from __future__ import annotations

from dataclasses import fields

from _lumascope.modules import executor_registry
from _lumascope.modules.lumascope_api._lumascope import scope_lane_closed
from _lumascope.modules.sequential_io_executor import SequentialIOExecutor


def test_the_bundle_starts_lists_and_stops_what_it_built_and_leaves_the_scopes_lanes_alone():
    io_lane = SequentialIOExecutor(name='IO', closed_answer=scope_lane_closed)
    camera_lane = SequentialIOExecutor(name='CAMERA', closed_answer=scope_lane_closed)
    bundle = executor_registry.create_default(io_lane, camera_lane)
    try:
        held = [getattr(bundle, f.name) for f in fields(bundle)]
        executors = [ex for ex in held if isinstance(ex, SequentialIOExecutor)]
        built = [ex for ex in executors if ex is not io_lane and ex is not camera_lane]
        assert built, 'the bundle built no executor of its own'

        assert [ex.executor_name for ex in built if not ex.worker_alive] == [], 'built, not started'
        listed = {id(ex): name for name, ex in bundle.queued_executors()}
        assert [ex.executor_name for ex in executors if id(ex) not in listed] == [], 'not listed'
        assert [ex.name for ex in executors if listed[id(ex)] != ex.name] == [], (
            'listed under another name'
        )
        assert not io_lane.worker_alive and not camera_lane.worker_alive, (
            "the bundle started the scope's lane"
        )
    finally:
        bundle.shutdown()

    assert [ex.executor_name for ex in built if not ex.pending_shutdown] == [], 'built, not stopped'
    assert not io_lane.pending_shutdown and not camera_lane.pending_shutdown, (
        "the bundle stopped the scope's lane"
    )
