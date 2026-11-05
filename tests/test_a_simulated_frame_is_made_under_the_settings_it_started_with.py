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

import itertools
import threading
import time

import pytest

from drivers import host_clock
from drivers.simulated_camera import SimulatedCamera
from tests.camera_fakes import grab_a_frame_made_after_now


def _next_frames(cam, wanted: int):
    """The next ``wanted`` frames delivered."""
    frames = []
    collected = threading.Event()

    def _collect(frame):
        if len(frames) < wanted:
            frames.append(frame)
            if len(frames) == wanted:
                collected.set()

    cam.register_frame_callback(_collect)
    assert collected.wait(5.0), f'only {len(frames)} frames came'
    return frames


def _frames_through_a_change(cam, change):
    """The frames delivered after ``change``, through the first that must show it.

    ``change()`` is made from the next frame's callback, on the camera's
    thread, so it lands at one place in the stream; the camera's clock is
    read either side of it. Collection ends at the first frame whose
    exposure started after the change returned.

    Returns:
        The camera's clock before the change and after it, and the frames.
    """
    frames = []
    clock = {}
    done = threading.Event()

    def _collect(frame):
        if not clock:
            clock['before'] = cam.camera_clock_ticks()
            change()
            clock['after'] = cam.camera_clock_ticks()
        elif not done.is_set():
            frames.append(frame)
            if frame.chunks['Timestamp'] >= clock['after']:
                done.set()

    cam.register_frame_callback(_collect)
    assert done.wait(5.0), f'no frame started after the change took; {len(frames)} came'
    return clock['before'], clock['after'], frames


def _mean(frame) -> float:
    return float(frame.image.mean())


# Three settings, three paths into the pixels: the integration time, the
# analogue scale and the offset.
@pytest.mark.parametrize(
    ('change', 'setting', 'new'),
    [
        (lambda cam: cam.exposure_t(20.0), 'exposure_us', 20_000.0),
        (lambda cam: cam.gain(20.0), 'gain_db', 20.0),
        (lambda cam: cam.set_black_level(60.0), 'black_level', 60.0),
    ],
    ids=['exposure', 'gain', 'black_level'],
)
def test_a_frame_shows_a_setting_only_if_it_started_after_the_write(change, setting, new):
    """A frame started before a write is made under the old setting; one started after it, under the new.

    The frames the sensor had started at the write are among the old ones.
    How many frames that is depends on how far the host has fallen behind
    the camera; which frames, by when each started, does not.
    """
    cam = SimulatedCamera(width=48, height=24)
    # A test pattern's signal follows the settings exactly; the specimen's is floored.
    cam.set_test_pattern(enabled=True, pattern='noise')
    cam.exposure_t(1.0)  # both exposures inside the delivery ceiling's period
    cam.gain(0.0)
    cam.open_and_start()
    try:
        settled = grab_a_frame_made_after_now(cam)
        old = getattr(cam.exposed_under(settled.chunks['FrameID']), setting)
        before, after, frames = _frames_through_a_change(cam, lambda: change(cam))
        made_under = [getattr(cam.exposed_under(f.chunks['FrameID']), setting) for f in frames]
    finally:
        cam.stop_grabbing()

    assert old != new, f'precondition: the change changes {setting} ({old})'
    starts = [f.chunks['Timestamp'] for f in frames]
    too_soon = [
        (start - before, value)
        for start, value in zip(starts, made_under, strict=True)
        if start < before and value != old
    ]
    assert not too_soon, f'frames started (ticks after the write, {setting}) {too_soon}'
    too_late = [
        (start - after, value)
        for start, value in zip(starts, made_under, strict=True)
        if start >= after and value != new
    ]
    assert not too_late, f'frames started (ticks after the write, {setting}) {too_late}'
    assert any(start < before for start in starts), (
        'no frame started before the write came after it'
    )
    # The pixels follow the record: old frames dim, new ones bright.
    dim, bright = _mean(settled), _mean(frames[-1])
    assert bright > 2 * dim, f'precondition: the change brightens the frame ({bright} vs {dim})'
    looks_old = [_mean(f) < (dim + bright) / 2 for f in frames]
    assert looks_old == [value == old for value in made_under], (made_under, looks_old)


def test_a_stale_frames_chunk_already_carries_the_new_value():
    """A frame made at the old gain and tagged after the write carries the new gain in its chunk.

    The chunk is read from the register as the frame is tagged, so it leads
    the content, as on the bench: only the frame's exposure record tells a
    stale frame from a new one.
    """
    cam = SimulatedCamera(width=48, height=24)
    cam.set_test_pattern(enabled=True, pattern='noise')
    cam.exposure_t(1.0)
    cam.gain(0.0)
    cam.open_and_start()
    try:
        grab_a_frame_made_after_now(cam)
        _, _, frames = _frames_through_a_change(cam, lambda: cam.gain(20.0))
        stale = [f for f in frames if cam.exposed_under(f.chunks['FrameID']).gain_db == 0.0]
    finally:
        cam.stop_grabbing()

    assert stale, 'precondition: a frame made at the old gain came after the write'
    assert all(f.chunks['Gain'] == 20.0 for f in stale), [f.chunks['Gain'] for f in stale]


def test_a_frame_is_numbered_as_it_leaves_the_pipeline_before_its_transfer():
    """A frame is numbered and timed as it leaves the sensor's pipeline, before the host makes and carries it.

    The camera numbers a frame on its own time, whatever the host is doing.
    A write waits on the camera's lock while a frame is made and can land
    while it is carried, so a frame numbered after those would be placed
    after a change it predates, and the count a change needs would grow
    with the host's image time.
    """
    transfer_s = SimulatedCamera.TIMING_REALISTIC['grab_delay']
    cam = SimulatedCamera(width=48, height=24, timing='realistic')
    stored = []
    two = threading.Event()

    def on_frame(frame):
        stored.append((frame.arrived_ns, host_clock.steady_ns()))
        if len(stored) == 2:
            two.set()

    cam.register_frame_callback(on_frame)
    cam.open_and_start()
    try:
        assert two.wait(5.0), f'only {len(stored)} frames were stored'
    finally:
        cam.stop_grabbing()

    carried_s = [(at_store - arrived) / 1e9 for arrived, at_store in stored]
    assert all(s >= transfer_s for s in carried_s), carried_s


def test_a_frame_leaves_the_pipeline_its_own_period_after_it_starts():
    """A frame is handed over its own period after it starts, plus the rest of the pipeline at the delivery ceiling's period.

    At the pipeline's one frame there is no rest: a frame leaves as its
    exposure ends. A simulator that holds a frame for two of its own
    periods holds a long exposure for a second exposure, and every capture
    after an exposure change waits that much longer than on a camera.
    """
    exposure_ms = 134.0  # S1's long exposure on the LS850T
    period_s = exposure_ms / 1000.0
    ceiling_s = 1.0 / SimulatedCamera._MAX_DELIVERY_FPS
    expected_s = period_s + (SimulatedCamera.SENSOR_PIPELINE_FRAMES - 1) * ceiling_s
    cam = SimulatedCamera(width=48, height=24)
    cam.exposure_t(exposure_ms)
    cam.open_and_start()
    try:
        grab_a_frame_made_after_now(cam)
        frames = _next_frames(cam, 3)
        clock_offset_ns = cam.camera_clock_ticks() - host_clock.steady_ns()
        made_under = [cam.exposed_under(f.chunks['FrameID']).exposure_us for f in frames]
    finally:
        cam.stop_grabbing()

    assert made_under == [exposure_ms * 1000.0] * 3, f'precondition: {made_under}'
    lags_s = [(f.arrived_ns - (f.chunks['Timestamp'] - clock_offset_ns)) / 1e9 for f in frames]
    # Late by the loop's wake-up at most, and never a second exposure late.
    assert all(expected_s - 0.001 <= lag < expected_s + period_s / 2 for lag in lags_s), (
        f'start to hand-over {lags_s} s; the camera takes {expected_s} s'
    )


def test_a_late_sensor_thread_never_starts_two_exposures_less_than_a_period_apart():
    """However late the host reaches a frame's start, the exposures start a frame period apart.

    A sensor starts its exposures on its own clock; the host thread that
    simulates it can be held past a start, by a write holding the camera's
    lock or by the other threads of a loaded process. A frame started when the thread got
    there, rather than when it was due, would start the next one at once to
    re-anchor, two exposures a fraction of a millisecond apart, which no
    camera makes and which puts frames before a change that a camera's would
    not have started.
    """
    period_ns = round(1e9 / SimulatedCamera._MAX_DELIVERY_FPS)
    cam = SimulatedCamera(width=48, height=24)
    cam.exposure_t(1.0)  # the ceiling's period sets the frame period
    stamps = []
    enough = threading.Event()

    def on_frame(frame):
        stamps.append(frame.chunks['Timestamp'])
        if len(stamps) in (3, 8):
            # The sensor thread starts each frame under the camera's lock:
            # holding the lock here is a host that reaches the next start
            # more than a period late.
            with cam._lock:
                time.sleep(2 * period_ns / 1e9)
        if len(stamps) == 14:
            enough.set()

    cam.register_frame_callback(on_frame)
    cam.open_and_start()
    try:
        assert enough.wait(5.0), f'only {len(stamps)} frames were stored'
    finally:
        cam.stop_grabbing()

    gaps_ns = [later - earlier for earlier, later in itertools.pairwise(stamps)]
    assert min(gaps_ns) >= period_ns, (
        f'exposure starts {gaps_ns} ns apart; the period is {period_ns}'
    )


def test_each_frame_carries_its_counter_and_its_exposure_start_on_the_cameras_clock():
    cam = SimulatedCamera(width=48, height=24)
    cam.exposure_t(1.0)
    cam.open_and_start()
    try:
        before = cam.camera_clock_ticks()
        frames = _next_frames(cam, 4)
    finally:
        cam.stop_grabbing()

    ids = [f.chunks['FrameID'] for f in frames]
    starts = [f.chunks['Timestamp'] for f in frames]
    # Rising, not consecutive: a frame the host drops on its way to the
    # store has taken its number, as on a camera.
    assert ids == sorted(ids) and len(set(ids)) == 4
    assert starts == sorted(starts) and len(set(starts)) == 4
    # The first frame delivered after ``before`` was started ahead of it:
    # the sensor had it in its pipeline.
    assert starts[0] < before
    assert cam.timestamp_tick_frequency_hz == SimulatedCamera.TIMESTAMP_TICK_HZ
