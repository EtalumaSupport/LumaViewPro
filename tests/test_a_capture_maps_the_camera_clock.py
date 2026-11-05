# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture maps the camera's clock onto the host's, beside its grab.

A frame's timestamp is on the camera's own clock, whose epoch the host
does not know, so nothing could say on the host's clock when a frame was
exposed: whether it began before or after a change the host made. Each
capture now latches the camera's clock between two reads of the host's
steady clock and records the mapping, half the bracket being how far off
it can be; a camera without a latch records none.

A still's time is its first frame's start on that clock, carried to the
host's wall clock through the same mapping. A camera without a latch, or a frame
that came without its tick, states the frame's arrival instead.
"""

from __future__ import annotations

import datetime
import threading

import pytest

from _lumascope.drivers import host_clock
from _lumascope.lib.log_helpers import VERBOSE
from _lumascope.modules import wall_time
from _lumascope.modules.lumascope_api.frame_record import (
    ClockMapping,
    FrameTime,
    TimestampSource,
    arrival_time,
)
from tests.camera_fakes import (
    chunk_config_pylon_camera,
    grab_a_frame_made_after_now,
    init_configurable_pylon_camera,
)
from tests.scope_fakes import bind_settings_like_a_session, build_scope


def test_a_mapping_converts_a_host_time_at_the_cameras_tick_rate():
    # 25 MHz, not the simulator's 1 GHz, at which ticks and nanoseconds
    # are the same number and a unit slip would pass.
    mapping = ClockMapping(
        host_ns=1_000, camera_ticks=500, tick_hz=25_000_000, uncertainty_ns=0, wall_ns=0
    )

    assert mapping.camera_ticks_at(1_000 + 1_000_000_000) == 500 + 25_000_000
    assert mapping.camera_ticks_at(1_000 - 40) == 499


def test_a_mapping_converts_a_tick_to_the_wall_time_it_was_latched_beside():
    wall_ns = 1_755_000_000_123_456_000
    mapping = ClockMapping(
        host_ns=1_000, camera_ticks=500, tick_hz=25_000_000, uncertainty_ns=0, wall_ns=wall_ns
    )
    latched = datetime.datetime.fromtimestamp(
        1_755_000_000, datetime.UTC
    ).astimezone() + datetime.timedelta(microseconds=123_456)

    assert mapping.wall_time_at(500) == latched
    # A second of ticks later is a second later; 25 ticks are a microsecond,
    # and one tick (40 ns) rounds to the nearest one.
    assert mapping.wall_time_at(500 + 25_000_000) == latched + datetime.timedelta(seconds=1)
    assert mapping.wall_time_at(475) == latched - datetime.timedelta(microseconds=1)
    assert mapping.wall_time_at(499) == latched


def test_a_tick_in_the_hour_a_clock_change_repeats_maps_to_its_own_instant(pacific_time):
    # 2026-11-01 01:30:00.123456 happens twice in Los Angeles: at 08:30 UTC
    # (daylight time) and an hour later at 09:30 UTC (standard time). A
    # recording keeps a frame by comparing its time's instant with its own
    # start, so a time that names the wrong pass drops or keeps a whole hour;
    # and a saved time states the offset in force, so a reader can tell the
    # two passes apart.
    for utc_hour, offset_h in ((8, -7), (9, -8)):
        instant = datetime.datetime(2026, 11, 1, utc_hour, 30, tzinfo=datetime.UTC)
        wall_ns = int(instant.timestamp()) * 1_000_000_000 + 123_456_000
        mapping = ClockMapping(
            host_ns=0, camera_ticks=0, tick_hz=1_000_000_000, uncertainty_ns=0, wall_ns=wall_ns
        )

        at = mapping.wall_time_at(0)

        assert round(at.timestamp() * 1_000_000) == wall_ns // 1000, f'{utc_hour}:30 UTC'
        assert at.utcoffset() == datetime.timedelta(hours=offset_h), f'{utc_hour}:30 UTC'


def test_a_frame_time_without_an_offset_is_refused():
    with pytest.raises(ValueError, match='states its UTC offset'):
        FrameTime(at=datetime.datetime(2026, 11, 1, 1, 30), source=TimestampSource.HOST_ARRIVAL)


def test_a_frames_arrival_is_its_hand_overs_on_both_clocks(sim_camera):
    """The wall time a frame states it arrived at is read with its steady
    arrival, at the hand-over, so the two name one moment: their difference
    is the host's own wall-minus-steady, to the two reads' skew."""
    frame = grab_a_frame_made_after_now(sim_camera[1])
    wall_now, steady_now = host_clock.wall_ns(), host_clock.steady_ns()

    skew_ns = abs((frame.wall_ns - frame.arrived_ns) - (wall_now - steady_now))

    assert skew_ns < 2_000_000, f'{skew_ns} ns apart'
    assert arrival_time(frame).at == wall_time.local_time(frame.wall_ns)


def _capture_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith('capture_still:')]


def test_a_captured_frame_mapped_through_its_record_began_before_it_arrived(sim_scope, caplog):
    caplog.set_level(VERBOSE, logger='LVP.api')
    imaging = sim_scope.imaging
    arrivals = {}
    lock = threading.Lock()

    def arrived(frame):
        with lock:
            arrivals[frame.chunks['FrameID']] = frame.arrived_ns

    imaging.add_frame_listener(arrived, name='arrivals')
    try:
        record = imaging.capture_still().record
    finally:
        imaging.remove_frame_listener(arrived)
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
    caplog.set_level(VERBOSE, logger='LVP.api')
    scope = build_scope(simulate=True, sim_model='LS620', register_atexit=False)
    try:
        bind_settings_like_a_session(scope)
        scope.imaging.enable_streaming()
        record = scope.imaging.capture_still().record
    finally:
        scope.disconnect()

    assert scope.capabilities.camera_has_clock_latch is False
    assert record.clock_mapping is None
    assert record.captured_at.source is TimestampSource.HOST_ARRIVAL
    (line,) = _capture_lines(caplog)
    assert ' clock=none' in line, line


def _wall_time_of_steady(steady_ns: int) -> datetime.datetime:
    """The host's wall time at the steady instant ``steady_ns``, through a
    wall/steady pair this test reads itself, never the production mapping."""
    wall_now, steady_now = host_clock.wall_ns(), host_clock.steady_ns()
    return datetime.datetime.fromtimestamp(
        (wall_now + steady_ns - steady_now) / 1e9, datetime.UTC
    ).astimezone()


def test_a_summed_still_states_its_first_frames_start_and_its_facts(sim_scope, sim_camera):
    camera, sim = sim_camera
    camera.set_exposure_ms(20.0)

    still = sim_scope.imaging.capture_still(sum_count=3)
    returned = datetime.datetime.now().astimezone()
    record = still.record

    # The simulator's own account of the frame the record names: its
    # exposure's start, on the simulator's clock (the host's steady clock
    # plus its own epoch), and what it was exposed under.
    exposed = sim.exposed_under(record.frame_id)
    assert record.camera_timestamp_ticks == exposed.started_ticks
    assert record.exposure_ms == exposed.exposure_us / 1000.0
    assert record.gain_db == exposed.gain_db
    started = _wall_time_of_steady(exposed.started_ticks - sim._clock_offset_ns)
    # Within the latch's uncertainty, and 2 ms for the two wall clock reads'
    # own slew against the steady clock.
    tolerance = datetime.timedelta(microseconds=record.clock_mapping.uncertainty_ns / 1000 + 2000)
    assert record.captured_at.source is TimestampSource.CAMERA_FRAME_START
    assert abs(record.captured_at.at - started) <= tolerance
    # The first of three: each later frame starts at least one exposure
    # after the one before it, and the capture returns after the last one
    # has been exposed and has arrived.
    assert returned - record.captured_at.at >= datetime.timedelta(milliseconds=3 * 20.0) - tolerance


def test_a_still_whose_frame_came_without_its_tick_states_its_arrival(sim_scope):
    sim_scope.simulation.withhold_chunk('Timestamp', 1_000)
    try:
        before = datetime.datetime.now().astimezone()
        record = sim_scope.imaging.capture_still().record
        returned = datetime.datetime.now().astimezone()
    finally:
        sim_scope.simulation.clear()

    assert record.clock_mapping is not None, 'the camera latched; the frame had no tick'
    assert record.camera_timestamp_ticks is None
    assert record.captured_at.source is TimestampSource.HOST_ARRIVAL
    assert before <= record.captured_at.at <= returned


def test_two_latches_from_two_threads_each_map_their_own_instant(sim_scope, sim_camera):
    """A recording's latch and a still's are commands on one camera lane:
    a latch and the read of what it latched never interleave with the
    other's, so each mapping's tick is the camera's clock at its own
    bracket, by the simulator's clock (its steady clock plus its epoch)."""
    _camera, sim = sim_camera
    mappings = []
    lock = threading.Lock()

    def _recordings_latches():
        for _ in range(20):
            mapping = sim_scope.imaging.map_camera_clock()
            with lock:
                mappings.append(mapping)

    def _stills_latches():
        for _ in range(5):
            mapping = sim_scope.imaging.capture_still().record.clock_mapping
            with lock:
                mappings.append(mapping)

    threads = [
        threading.Thread(target=_recordings_latches),
        threading.Thread(target=_stills_latches),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30.0)
        assert not thread.is_alive(), 'a latch never returned'

    assert len(mappings) == 25
    for mapping in mappings:
        latched_steady_ns = mapping.camera_ticks - sim._clock_offset_ns
        assert abs(latched_steady_ns - mapping.host_ns) <= mapping.uncertainty_ns, mapping


def _configured_pylon_camera(advertised):
    """A Pylon camera reconnected through the real ``init_camera_config``,
    its chunks and its tick rate read as a connect reads them, advertising
    ``advertised`` chunks."""
    # a stand-in by design: the Pylon SDK never runs in the test process.
    cam = init_configurable_pylon_camera()
    del cam._enable_validity_chunks, cam._read_timestamp_tick_frequency
    # The rate an earlier connect of the same camera read.
    cam.timestamp_tick_frequency_hz = 1_000_000_000
    chunks = chunk_config_pylon_camera(advertised).active
    for node in ('GetNodeMap', 'ChunkModeActive', 'ChunkSelector', 'ChunkEnable'):
        setattr(cam.active, node, getattr(chunks, node))
    cam.init_camera_config()
    return cam


def test_a_camera_whose_frames_carry_no_start_states_no_tick_rate():
    """A latch maps the camera's clock onto the host's only for frames that
    carry their start on it: a camera whose ``Timestamp`` chunk did not
    turn on states no tick rate, so it has no clock latch and its moves are
    counted in frames from the stop, never judged by a start no frame has."""
    stamped = _configured_pylon_camera(['ExposureTime', 'Gain', 'Timestamp'])
    unstamped = _configured_pylon_camera(['ExposureTime', 'Gain'])

    assert 'Timestamp' in stamped.active.ChunkSelector.writes
    assert stamped.timestamp_tick_frequency_hz is not None
    assert 'Timestamp' not in unstamped.active.ChunkSelector.writes
    assert unstamped.timestamp_tick_frequency_hz is None
