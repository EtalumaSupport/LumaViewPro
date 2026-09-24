# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated camera free-runs at a real camera's rate: its ceiling, no faster, no slower.

A real camera's readout and link bandwidth bound its frame rate however
short the exposure. Without that bound the simulator free-runs at
1/exposure -- 1000 fps at 1 ms -- so a recording that keeps every
delivered frame would see a rate no bench ever produces. And a real
camera's frame period does not include host work: a pump that sleeps a
full interval AFTER generating each frame runs slower than its ceiling
by the generation time, so a full-size sim frame (~15 ms to build)
delivered 25 fps against a 40 fps ceiling.
"""

import time

from drivers.simulated_camera import SimulatedCamera

# Stand-in for the cost of building a full-size frame; large enough that a
# pump adding it to the interval misses the ceiling by a wide margin.
GENERATION_COST_S = 0.015


def _free_run_rate(cam, seconds=1.0):
    arrivals = []
    cam.register_frame_callback(lambda img, ts, chunks: arrivals.append(time.monotonic()))
    try:
        time.sleep(seconds)
    finally:
        cam.stop_grabbing()
    assert len(arrivals) >= 2, 'pump did not deliver; the rest of this test proves nothing'
    return (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])


def _slow_generation(cam):
    mint = cam._mint_frame

    def slow_mint(*args, **kwargs):
        time.sleep(GENERATION_COST_S)
        return mint(*args, **kwargs)

    cam._mint_frame = slow_mint


def test_short_exposure_free_run_stays_under_the_delivery_ceiling():
    cam = SimulatedCamera(width=32, height=24)
    cam.exposure_t(1.0)
    cam.start_grabbing()
    # The pump waits for each frame's due time, so it can only run at or
    # under the ceiling, never faster.
    assert _free_run_rate(cam) <= SimulatedCamera._MAX_DELIVERY_FPS * 1.05


def test_generation_time_does_not_lower_the_delivered_rate():
    cam = SimulatedCamera(width=32, height=24)
    cam.exposure_t(1.0)
    _slow_generation(cam)
    cam.start_grabbing()
    # Adding the generation cost to the 25 ms period would give 25 fps;
    # scheduled frames absorb it and hold the ceiling.
    assert _free_run_rate(cam) >= SimulatedCamera._MAX_DELIVERY_FPS * 0.85
