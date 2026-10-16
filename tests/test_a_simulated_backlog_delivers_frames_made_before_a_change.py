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

import numpy as np
import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import ArgumentRefusedError
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


def test_a_backlog_delivers_frames_made_before_the_change(sim_scope):
    imaging = sim_scope.imaging
    before = imaging.get_gain_db()
    sim_scope.simulation.delay_delivery(BACKLOG)
    # The backlog stands once a frame arrives after it: none is delivered
    # until BACKLOG are held.
    _frames_from(sim_scope, lambda: None, 1)
    in_flight = SimulatedCamera.SENSOR_PIPELINE_FRAMES + BACKLOG
    after = before + 6.0

    frames = _frames_from(sim_scope, lambda: imaging.set_gain_db(after), in_flight + 6)

    made_under = [sim_scope.simulation.exposed_under(f).gain_db for f in frames]
    old = made_under.index(after)
    assert set(made_under[:old]) == {before} and set(made_under[old:]) == {after}, made_under
    # Every frame started before the write and every one held behind them
    # arrived after it began: the frames collected before it are extra.
    assert old >= in_flight, made_under


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
