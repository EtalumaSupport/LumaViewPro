# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A delivered frame's arrival is on the clock the session publishes.

A caller sets its own "now" against a frame's ``arrived_ns`` -- the engineering
plugin waits for a phase of the frame period after the last arrival -- so the
two must be one clock. ``session.host_steady_ns()`` is that clock's one public
read; a frame's arrival, stamped by the driver, lies between two reads of it
taken around its delivery. On Windows under Python 3.12 a stamp on
``time.monotonic_ns`` against a read on a finer clock would not.
"""

from __future__ import annotations

import threading

from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


def test_a_frames_arrival_lies_between_two_reads_of_the_sessions_clock():
    session = ScopeSession.create(complete_settings(), simulate=True, warn_pre_release=False)
    seen = []
    second = threading.Event()

    def on_frame(frame):
        # Read on delivery: the frame's own stamp was taken before this call.
        seen.append((frame.arrived_ns, session.host_steady_ns()))
        if len(seen) == 2:
            second.set()

    try:
        session.scope.imaging.enable_streaming()
        session.scope.imaging.add_frame_listener(on_frame, name='arrivals')
        try:
            assert second.wait(5.0), 'no second frame arrived'
        finally:
            session.scope.imaging.remove_frame_listener(on_frame)
    finally:
        session.shutdown()

    # The first delivery's read precedes the second frame's arrival, which a
    # listener registered mid-stream cannot say of the first frame.
    (_, read_before), (arrived_ns, read_after) = seen[:2]
    assert read_before <= arrived_ns <= read_after
