# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The frames a simulated camera had started when a setting changed come out under the old setting.

A real sensor finishes the frames it started: after a change of exposure,
gain or black level it still delivers the frames it had in its pipeline
under the old settings, then the new. Their chunks already carry the new
register value -- the chunk leads the content, as the bench measured on the
dart -- so only the frame's own exposure record tells a stale frame from a
new one. The simulator applied a change to the very next frame, and
stamped no chunks, so what the host does with those frames could not be
shown without a bench.
"""

from __future__ import annotations

import threading

import pytest

from drivers.simulated_camera import SimulatedCamera
from tests.camera_fakes import grab_a_frame_made_after_now


def _collect_after(cam, change, wanted: int):
    """The frames delivered from just before ``change()`` on, ``wanted`` of them."""
    changed = threading.Event()
    frames = []
    collected = threading.Event()

    def _collect(frame):
        if changed.is_set() and len(frames) < wanted:
            frames.append(frame)
            if len(frames) == wanted:
                collected.set()

    cam.register_frame_callback(_collect)
    # Collecting starts before the change, so the frame being made at the
    # change may be collected too: it predates the change and is old.
    changed.set()
    change()
    assert collected.wait(5.0), f'only {len(frames)} frames came after the change'
    return frames


# Three settings, three paths into the pixels: the integration time, the
# analogue scale and the offset.
@pytest.mark.parametrize(
    'change',
    [
        lambda cam: cam.exposure_t(20.0),
        lambda cam: cam.gain(20.0),
        lambda cam: cam.set_black_level(60.0),
    ],
    ids=['exposure', 'gain', 'black_level'],
)
def test_the_frames_in_flight_at_a_change_are_made_under_the_old_setting(change):
    in_flight = SimulatedCamera.SENSOR_PIPELINE_FRAMES
    cam = SimulatedCamera(width=48, height=24)
    # A test pattern's signal follows the settings exactly; the specimen's is floored.
    cam.set_test_pattern(enabled=True, pattern='noise')
    cam.exposure_t(1.0)  # both exposures inside the delivery ceiling's period
    cam.gain(0.0)
    cam.open_and_start()
    try:
        dim = float(grab_a_frame_made_after_now(cam).image.mean())
        frames = _collect_after(cam, lambda: change(cam), in_flight + 4)
    finally:
        cam.stop_grabbing()

    means = [float(frame.image.mean()) for frame in frames]
    bright = means[-1]
    assert bright > 2 * dim, f'precondition: the change brightens the frame ({bright} vs {dim})'
    old = [m < (dim + bright) / 2 for m in means]
    first_new = old.index(False)
    assert all(old[:first_new]) and not any(old[first_new:]), means
    assert in_flight <= first_new <= in_flight + 1, (
        f'{first_new} frames came under the old setting; the sensor had {in_flight} started'
    )


def test_a_stale_frames_chunk_already_carries_the_new_value():
    cam = SimulatedCamera(width=48, height=24)
    cam.set_test_pattern(enabled=True, pattern='noise')
    cam.exposure_t(1.0)
    cam.gain(0.0)
    cam.open_and_start()
    try:
        grab_a_frame_made_after_now(cam)
        wanted = SimulatedCamera.SENSOR_PIPELINE_FRAMES + 3
        frames = _collect_after(cam, lambda: cam.gain(20.0), wanted)
    finally:
        cam.stop_grabbing()

    means = [float(frame.image.mean()) for frame in frames]
    middle = (means[0] + means[-1]) / 2
    stale = [f for f, m in zip(frames, means, strict=True) if m < middle]
    assert any(f.chunks['Gain'] == 20.0 for f in stale), (
        'no frame made at the old gain was tagged after the write'
    )


def test_each_frame_carries_its_counter_and_its_exposure_start_on_the_cameras_clock():
    cam = SimulatedCamera(width=48, height=24)
    cam.exposure_t(1.0)
    cam.open_and_start()
    try:
        before = cam.camera_clock_ticks()
        frames = _collect_after(cam, lambda: None, 4)
    finally:
        cam.stop_grabbing()

    ids = [f.chunks['FrameID'] for f in frames]
    starts = [f.chunks['Timestamp'] for f in frames]
    assert ids == list(range(ids[0], ids[0] + 4))
    assert starts == sorted(starts) and len(set(starts)) == 4
    # The first frame delivered after ``before`` was started ahead of it:
    # the sensor had it in its pipeline.
    assert starts[0] < before
    assert cam.timestamp_tick_frequency_hz == SimulatedCamera.TIMESTAMP_TICK_HZ
