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

from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


def test_a_frames_arrival_lies_between_two_reads_of_the_sessions_clock():
    session = ScopeSession.create(complete_settings(), simulate=True, warn_pre_release=False)
    seen = []
    arrived = threading.Event()

    def on_frame(frame):
        # Read on delivery: the frame's own stamp was taken before this call.
        seen.append((frame.arrived_ns, session.host_steady_ns()))
        arrived.set()

    try:
        # Read before the stream starts, so it precedes every frame's
        # arrival; a read in an earlier frame's listener would not, since the
        # camera stamps a frame before it is queued for delivery and the next
        # can be stamped while that one is still on its way.
        read_before = session.host_steady_ns()
        session.scope.imaging.enable_streaming()
        session.scope.imaging.add_frame_listener(on_frame, name='arrivals')
        try:
            assert arrived.wait(5.0), 'no frame arrived'
        finally:
            session.scope.imaging.remove_frame_listener(on_frame)
    finally:
        session.shutdown()

    arrived_ns, read_after = seen[0]
    assert read_before <= arrived_ns <= read_after
