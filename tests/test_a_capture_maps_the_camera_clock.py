# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture maps the camera's clock onto the host's, beside its grab.

A frame's timestamp is on the camera's own clock, whose epoch the host
does not know, so nothing could say on the host's clock when a frame was
exposed: whether it began before or after a change the host made. Each
capture now latches the camera's clock between two reads of the host's
steady clock and records the mapping, half the bracket being how far off
it can be; a camera without a latch records none.
"""

from __future__ import annotations

import logging
import threading

from modules.lumascope_api.frame_record import ClockMapping
from tests.scope_fakes import bind_settings_like_a_session, build_scope


def test_a_mapping_converts_a_host_time_at_the_cameras_tick_rate():
    # 25 MHz, not the simulator's 1 GHz, at which ticks and nanoseconds
    # are the same number and a unit slip would pass.
    mapping = ClockMapping(host_ns=1_000, camera_ticks=500, tick_hz=25_000_000, uncertainty_ns=0)

    assert mapping.camera_ticks_at(1_000 + 1_000_000_000) == 500 + 25_000_000
    assert mapping.camera_ticks_at(1_000 - 40) == 499


def _capture_lines(caplog):
    return [
        r.getMessage() for r in caplog.records if r.getMessage().startswith('capture_and_wait:')
    ]


def test_a_captured_frame_mapped_through_its_record_began_before_it_arrived(sim_scope, caplog):
    caplog.set_level(logging.INFO, logger='LVP.api')
    imaging = sim_scope.imaging
    arrivals = {}
    lock = threading.Lock()

    def arrived(frame):
        with lock:
            arrivals[frame.chunks['FrameID']] = frame.arrived_ns

    imaging.add_frame_listener(arrived, name='arrivals')
    try:
        imaging.capture_and_wait()
    finally:
        imaging.remove_frame_listener(arrived)
    record = imaging.last_capture_info['frame_record']
    mapping = record.clock_mapping

    assert sim_scope.capabilities.camera_has_clock_latch
    assert mapping.uncertainty_ns >= 0
    # The frame's exposure start, on the host's clock through the mapping:
    # one exposure at least before it reached the host, and well under a
    # second -- the camera's epoch, unmapped, is up to 1000 s away.
    began_ns = mapping.host_ns + (record.camera_timestamp_ticks - mapping.camera_ticks) * (
        1_000_000_000 / mapping.tick_hz
    )
    with lock:
        in_flight_ns = arrivals[record.frame_id] - began_ns
    exposure_ns = record.exposure_ms * 1_000_000
    assert exposure_ns - mapping.uncertainty_ns <= in_flight_ns < 1_000_000_000
    # The support bundle's one line names how well the clock was mapped.
    (line,) = _capture_lines(caplog)
    assert f' clock=+-{mapping.uncertainty_ns / 1000:.0f}us' in line, line


def test_a_camera_without_a_latch_records_no_mapping(caplog):
    caplog.set_level(logging.INFO, logger='LVP.api')
    scope = build_scope(simulate=True, sim_model='LS620', register_atexit=False)
    try:
        bind_settings_like_a_session(scope)
        scope.imaging.enable_streaming()
        scope.imaging.capture_and_wait()
        record = scope.imaging.last_capture_info['frame_record']
    finally:
        scope.disconnect()

    assert scope.capabilities.camera_has_clock_latch is False
    assert record.clock_mapping is None
    (line,) = _capture_lines(caplog)
    assert ' clock=none' in line, line
