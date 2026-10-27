# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated host backlog delivers frames made before a change, and each frame says what it was made under.

A frame can wait on its way to the host -- in the link, in the driver's
queue -- so it arrives after a change it was made before. The simulator had
no such wait, and no way to say what a frame was made under but its
pixels, so nothing could show a capture returning a frame older than the
change it waited for.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from drivers import host_clock
from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import ArgumentRefusedError
from modules.frame_validity import FrameValiditySource
from tests.camera_fakes import delivered_frame
from tests.scope_fakes import build_scope

BACKLOG = 3


def _frames_from(scope, change, wanted: int) -> list:
    """The frames delivered from just before ``change()`` on, ``wanted`` of them.

    Collecting starts before the change, since a setter can block while
    frames pass; a frame collected before the change lands is an old one.
    """
    frames = []
    collected = threading.Event()

    def on_frame(frame):
        if len(frames) < wanted:
            frames.append(frame)
            if len(frames) == wanted:
                collected.set()

    scope.imaging.add_frame_listener(on_frame, name='from the change')
    try:
        change()
        assert collected.wait(5.0), f'only {len(frames)} frames came from the change'
    finally:
        scope.imaging.remove_frame_listener(on_frame)
    return frames


def _wait_for(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_backlog_delivers_frames_made_before_the_change(sim_scope):
    camera = sim_scope.camera
    before = camera.read_gain_db()
    sim_scope.simulation.delay_delivery(BACKLOG)
    # The backlog stands once a frame arrives after it: none is delivered
    # until BACKLOG are held.
    _frames_from(sim_scope, lambda: None, 1)
    in_flight = SimulatedCamera.SENSOR_PIPELINE_FRAMES + BACKLOG
    after = before + 6.0

    frames = _frames_from(sim_scope, lambda: camera.set_gain_db(after), in_flight + 6)

    made_under = [sim_scope.simulation.exposed_under(f).gain_db for f in frames]
    old = made_under.index(after)
    assert set(made_under[:old]) == {before} and set(made_under[old:]) == {after}, made_under
    # Every frame started before the write and every one held behind them
    # arrived after it began: the frames collected before it are extra.
    assert old >= in_flight, made_under


def test_a_frame_held_before_the_store_is_placed_before_a_change_made_while_it_was_held(sim_scope):
    """The frames on their way to the store when a change is recorded are numbered before it.

    A frame is numbered where the driver first has it, so one still in a
    host queue when the write goes out sits at or below the change's
    ``at_seq`` however long the queue holds it. Numbered at the store, it
    would sit above it and be credited toward the change it predates, and
    a count measured on one host would then cover that host's queue.
    """
    camera = sim_scope.camera
    sim_scope.simulation.delay_delivery(BACKLOG)
    _frames_from(sim_scope, lambda: None, 1)  # the backlog stands
    received = []

    def on_frame(frame):
        received.append((frame, host_clock.steady_ns()))

    sim_scope.imaging.add_frame_listener(on_frame, name='after the change')
    try:
        camera.set_gain_db(camera.read_gain_db() + 6.0)
        change = sim_scope.imaging.recorded_changes()[FrameValiditySource.GAIN]

        def stored_after():
            return [f for f, at in list(received) if at > change.written_ns]

        assert _wait_for(lambda: len(stored_after()) > BACKLOG), 'too few frames came after'
    finally:
        sim_scope.imaging.remove_frame_listener(on_frame)

    held = [f.seq for f in stored_after() if f.seq <= change.at_seq]
    assert len(held) >= BACKLOG, (held, change.at_seq)


def test_a_capture_under_a_backlog_returns_a_frame_stored_after_it_asked(sim_scope):
    """The simulator's capture waits on the frames stored, never on those handed over.

    Under a backlog the camera has handed over frames it has not stored, so
    a wait on the hand-over count against a stored count is met at once and
    hands back the frame stored before the capture asked.
    """
    sim_scope.simulation.delay_delivery(BACKLOG)
    _frames_from(sim_scope, lambda: None, 1)  # the backlog stands
    asked = threading.Event()
    stored_since = []

    def on_frame(frame):
        if asked.is_set():
            stored_since.append(frame)

    sim_scope.imaging.add_frame_listener(on_frame, name='since the capture asked')
    try:
        asked.set()
        frame = sim_scope._camera_driver.grab_new_capture(5.0)
    finally:
        sim_scope.imaging.remove_frame_listener(on_frame)

    assert frame is not None
    assert any(f is frame for f in stored_since), (
        'the capture returned a frame stored before it asked'
    )


def test_clear_releases_the_held_frames(sim_scope):
    sim_scope.simulation.delay_delivery(BACKLOG)
    sim_scope.simulation.clear()

    frames = _frames_from(sim_scope, lambda: None, BACKLOG + 2)

    ids = [f.chunks['FrameID'] for f in frames]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert sim_scope.simulation.injected() == ()


def test_a_frame_the_camera_did_not_deliver_has_no_record(sim_scope):
    stranger = delivered_frame(np.zeros((2, 2), np.uint8), chunks={'FrameID': 10**9})

    with pytest.raises(ArgumentRefusedError) as refused:
        sim_scope.simulation.exposed_under(stranger)

    assert refused.value.reason == 'exposure_not_recorded'


def test_a_backlog_is_refused_below_one_frame(sim_scope):
    with pytest.raises(ArgumentRefusedError) as refused:
        sim_scope.simulation.delay_delivery(0)

    assert refused.value.reason == 'fault_bound'


def test_the_fx2_takes_neither():
    scope = build_scope(simulate=True, sim_model='LS620', register_atexit=False)
    try:
        frame = delivered_frame(np.zeros((2, 2), np.uint8))
        with pytest.raises(ArgumentRefusedError) as backlog:
            scope.simulation.delay_delivery(1)
        with pytest.raises(ArgumentRefusedError) as record:
            scope.simulation.exposed_under(frame)
    finally:
        scope.disconnect()

    assert backlog.value.reason == 'not_injectable_on_board'
    assert record.value.reason == 'not_injectable_on_board'
