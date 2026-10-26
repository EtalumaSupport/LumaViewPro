# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The scope's steady clock: the one clock every host stamp set against a frame is read from.

A frame's arrival, a change's write and its in-force instant, and the camera
clock latch's bracket are compared with one another and with the camera's
own clock, so they are read from this one function and never from a clock
chosen at the call. The scope publishes it as ``Lumascope.host_steady_ns()``
for a caller that sets its own "now" against those stamps.
"""

import time


def steady_ns() -> int:
    """The host's steady time, in nanoseconds.

    ``time.perf_counter_ns()``: the highest-resolution steady clock Python
    offers, system-wide, counting through sleep. ``time.monotonic_ns()`` is
    not it: on Windows under Python 3.12 that is ``GetTickCount64()``, a
    15.6 ms step, on which two events a frame apart read as one instant and a
    latch bracket reads 0 wide. From 3.13, and on macOS and Linux at any
    supported version, the two are one clock.
    """
    return time.perf_counter_ns()
