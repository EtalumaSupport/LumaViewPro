# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A still waits out the changes the scope recorded, and a light LVP did not switch is not one of them.

``capture_still`` keeps only a frame that counts for every change frame
validity holds. A light outside the scope, such as another program's
shutter, is never recorded, so with frames still on their way to the host a
still taken just after it keeps a frame made before it, and the frame looks
right. Each frame is judged by what the simulated sensor made it under
(``scope.simulation.exposed_under``), never by its pixels.
"""

from __future__ import annotations

import threading

import pytest

from _lumascope.modules.exceptions import ArgumentRefusedError

# Frames held between the camera and the host, more than any source's count.
BACKLOG = 6
OUTSIDE_MA = 40.0


class _Delivered:
    """The frames the camera delivers, by their ``FrameID``, and a wait for the next."""

    def __init__(self):
        self._frames = {}
        self._arrived = threading.Condition()

    def keep(self, frame):
        with self._arrived:
            self._frames[frame.chunks['FrameID']] = frame
            self._arrived.notify_all()

    def __getitem__(self, frame_id):
        with self._arrived:
            return self._frames[frame_id]

    def next_one(self):
        """Wait for the next frame; raises TimeoutError, not an assertion, so a
        camera that stopped delivering fails as itself."""
        with self._arrived:
            seen = len(self._frames)
            if not self._arrived.wait_for(lambda: len(self._frames) > seen, timeout=5.0):
                raise TimeoutError('no frame arrived')


@pytest.fixture
def delivered(sim_scope):
    frames = _Delivered()
    sim_scope.imaging.add_frame_listener(frames.keep, name='frames delivered')
    yield frames
    sim_scope.imaging.remove_frame_listener(frames.keep)


def _still_after_outside_light(scope, delivered, **capture):
    """A light from outside the scope switched on behind a host backlog, a
    still just after it, and what the still's frame was made under."""
    scope.simulation.delay_delivery(BACKLOG)
    try:
        delivered.next_one()
        scope.simulation.light_outside(OUTSIDE_MA)
        record = scope.imaging.capture_still(**capture).record
        return scope.simulation.exposed_under(delivered[record.frame_id])
    finally:
        scope.simulation.clear()


def test_a_still_after_a_light_lvp_did_not_switch_keeps_a_frame_made_before_it(
    sim_scope, delivered
):
    made_under = _still_after_outside_light(sim_scope, delivered)

    assert made_under.outside_light_ma == 0.0, made_under


def _still_made_under(scope, delivered):
    """What a still taken once the frames in flight have passed was made under."""
    for _ in range(3):
        delivered.next_one()
    record = scope.imaging.capture_still().record
    return scope.simulation.exposed_under(delivered[record.frame_id])


def test_a_light_from_outside_lights_the_frames_after_it_until_cleared(sim_scope, delivered):
    sim = sim_scope.simulation
    sim.light_outside(OUTSIDE_MA)
    try:
        lit = _still_made_under(sim_scope, delivered)
    finally:
        sim.clear()
    out = _still_made_under(sim_scope, delivered)

    assert (lit.outside_light_ma, out.outside_light_ma) == (OUTSIDE_MA, 0.0)
    assert lit.light > out.light


def test_a_light_from_outside_of_no_current_is_refused(sim_scope):
    with pytest.raises(ArgumentRefusedError) as refused:
        sim_scope.simulation.light_outside(0.0)

    assert refused.value.reason == 'fault_bound'
