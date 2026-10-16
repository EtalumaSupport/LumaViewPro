# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The frames a simulated camera had started when its exposure changed come out at the old exposure.

A real sensor finishes the frames it started: after an exposure change it
still delivers the dart's measured count of frames exposed at the old one,
then the new. The simulator applied a change to the very next frame, so what
the host does with those frames -- frame validity's drain, the stream check's
held bound after a drop -- could not be shown without a bench.
"""

from __future__ import annotations

import threading

from drivers.camera_profiles import simulated_exposure_delay_frames
from drivers.simulated_camera import SimulatedCamera
from tests.camera_fakes import grab_a_frame_made_after_now


def test_the_frames_in_flight_at_a_change_are_lit_by_the_old_exposure():
    in_flight = simulated_exposure_delay_frames()
    wanted = in_flight + 4
    cam = SimulatedCamera(width=48, height=24)
    # A test pattern's signal follows the exposure exactly; the specimen's is floored.
    cam.set_test_pattern(enabled=True, pattern='noise')
    cam.exposure_t(1.0)  # both exposures inside the delivery ceiling's period
    cam.open_and_start()
    try:
        dim = float(grab_a_frame_made_after_now(cam).image.mean())

        changed = threading.Event()
        means = []
        collected = threading.Event()

        def _collect(frame):
            if changed.is_set() and len(means) < wanted:
                means.append(float(frame.image.mean()))
                if len(means) == wanted:
                    collected.set()

        cam.register_frame_callback(_collect)
        # Collecting starts before the change, so the frame being made at
        # the change may be collected too: it predates the change and is old.
        changed.set()
        cam.exposure_t(20.0)
        assert collected.wait(5.0), f'only {len(means)} frames came after the change'
    finally:
        cam.stop_grabbing()

    bright = means[-1]
    assert bright > 5 * dim, f'precondition: 20 ms is brighter than 1 ms ({bright} vs {dim})'
    old = [m < (dim + bright) / 2 for m in means]
    first_new = old.index(False)
    assert all(old[:first_new]) and not any(old[first_new:]), means
    assert in_flight <= first_new <= in_flight + 1, (
        f'{first_new} frames came at the old exposure; the sensor had {in_flight} started'
    )
