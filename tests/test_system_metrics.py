# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The metrics snapshot never breaks another thread's work.

``system_metrics()`` runs on the metrics tick's own thread while a run moves,
grabs and saves on others. A tuple built from a generator is still being
grown while that generator waits on serial I/O with the GIL released, and
``tuple()`` refuses to finish one that any other thread holds a reference
to. The snapshot's heap count was ``len(gc.get_objects())``, which takes a
reference to every tracked object, and a field run's XY move raised
``SystemError: bad argument to internal function`` from it mid-protocol.
"""

import sys
import threading
import time

import pytest

from _lumascope.modules import common_utils


def _waiting_generator():
    # Each item waits with the GIL released, as a serial exchange does.
    for item in range(8):
        time.sleep(0.0005)
        yield item


@pytest.fixture
def a_field_sized_heap():
    # The field box held about 700k tracked objects (PERFORMANCE_BUDGETS.md
    # allocated_blocks_growth_per_hr, W10). A list of them lives long enough
    # for the GIL to change hands while it is held; a test process's small
    # heap builds and frees it inside one turn, and a short switch interval
    # stands in for the rest of the field's contention. Without both, the
    # gc.get_objects() count this test was written against refused no tuple
    # in about 70 builds with the heap alone, and seven or more in every
    # trial with both.
    heap = [[] for _ in range(700_000)]
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    yield heap
    sys.setswitchinterval(interval)


def test_a_tuple_built_while_the_snapshot_runs_is_finished(a_field_sized_heap):
    building = threading.Event()
    snapshots_done = threading.Event()

    def take_snapshots():
        building.wait()
        for _ in range(80):
            common_utils.system_metrics(collect_open_files=False)
            # Spaced, so the builds run between snapshots as they do between ticks.
            time.sleep(0.002)
        snapshots_done.set()

    snapshotter = threading.Thread(target=take_snapshots, daemon=True)
    snapshotter.start()
    refused = 0
    builds = 0
    while not snapshots_done.is_set():
        building.set()
        try:
            tuple(_waiting_generator())
        except SystemError:
            refused += 1
        builds += 1
    snapshotter.join()

    assert refused == 0, f'{refused} of {builds} tuples refused while the snapshot ran'
