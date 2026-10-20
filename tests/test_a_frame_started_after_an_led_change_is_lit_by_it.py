# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A frame the simulated sensor starts after an LED change is recorded is lit by it.

A frame counts for an LED change by when it began against the change's
record, so every state the light is made from must hold the change by
then. The simulated sensor reads the light from the illumination API's
store; the API wrote the store after recording the change, and a frame
begun between the two was made under the old light while counting for
the new one. The store is written first.
"""

from __future__ import annotations

import threading
import time

# How long the LED write is held after its change is recorded: a loaded
# host's gap between the record and the rest of the write, several frames
# wide so frames begin inside it on every run.
HOLD_S = 0.1


def test_a_frame_started_after_an_led_change_is_recorded_is_lit_by_it(sim_scope, monkeypatch):
    scope = sim_scope
    scope.illumination.led_on('BF', 50.0)
    scope.imaging.capture_and_wait()
    mapping = scope.imaging.last_capture_info['frame_record'].clock_mapping
    fv = scope.frame_validity
    invalidate = fv.invalidate
    recorded = []

    def held(source):
        """A stand-in by design: the simulator cannot load the host, and a
        loaded host is what widens the gap. The record is the production one."""
        change = invalidate(source)
        if source == 'led':
            recorded.append(change.written_ns)
            time.sleep(HOLD_S)
        return change

    monkeypatch.setattr(fv, 'invalidate', held)
    frames = []
    past_the_hold = threading.Event()

    def on_frame(frame):
        frames.append(frame)
        if recorded and frame.chunks['Timestamp'] > mapping.camera_ticks_at(
            recorded[0] + 2 * int(HOLD_S * 1e9)
        ):
            past_the_hold.set()

    scope.imaging.add_frame_listener(on_frame, name='after the change')
    try:
        scope.illumination.led_on('BF', 100.0)
        assert past_the_hold.wait(5.0), 'no frame began after the held write'
    finally:
        scope.imaging.remove_frame_listener(on_frame)

    after = mapping.camera_ticks_at(recorded[0]) + mapping.uncertainty_ns * mapping.tick_hz // 10**9
    begun_after = [frame for frame in frames if frame.chunks['Timestamp'] > after]
    lit = [scope.simulation.exposed_under(frame).illumination_ma for frame in begun_after]
    assert begun_after and all(ma == 100.0 for ma in lit), lit
