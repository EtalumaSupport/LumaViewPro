# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A recorded frame's file says where and how the frame was taken, and what lit it.

The facts are read when the frame arrives and ride the engine's queue with
it, because the write runs later, behind the backlog; the frame's TIFF
carries them under the still capture's position keys, so the file says what
it is with or without a hyperstack. These tests pin the engine's carry,
the facts' and the record's construction from the frame and the scope's
tracked state, the frame's time and what it is, the metadata builder's
fields, and the plate transform a recording binds at start.
"""

import copy
import datetime
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from _lumascope.drivers import host_clock
from _lumascope.modules.coord_transformations import PlatePosition, StageOffset
from _lumascope.modules import wall_time
from _lumascope.modules.lumascope_api import AxisPosition, AxisState
from _lumascope.modules.lumascope_api.runtime_state import RuntimeState
from _lumascope.modules.common_utils import Light
from _lumascope.modules.lumascope_api.frame_record import (
    ClockMapping,
    FrameTime,
    LightTiming,
    SettingSource,
    TimestampSource,
    arrival_time,
    camera_frame_time,
)
from _lumascope.modules.image_utils import read_file_record
from _lumascope.modules.recording_frames import (
    FrameFact,
    RecordingCamera,
    arrival,
    frame_fact,
    write_frame,
)
from _lumascope.modules.video_recording import RecordingConfig, VideoRecordingEngine
from tests.camera_fakes import delivered_frame
from tests.frame_records import frame_record, plate
from tests.video_engine_harness import FakeClock, FrameFeed, WriterStub
from _lumascope.modules.channel_record import release_colors
from tests.stock_channels import LED0, LED1

A_FACT = FrameFact(plate_x_mm=1.5, plate_y_mm=2.5, z_um=3.0, moving=False)
A_LIGHT = Light(channel=LED1, illumination_by_channel_ma={LED1: 100.0})
A_TIME = FrameTime(
    at=wall_time.local_time(1_755_000_000_000_000_000), source=TimestampSource.HOST_ARRIVAL
)
A_CAMERA = RecordingCamera(binning_size=2, camera_model='simulated', camera_tick_hz=1_000_000_000)


def _arrival(scope, *, channel_tiebreak='BF', chunks=None, significant_bits=12):
    frame = delivered_frame(
        np.zeros((8, 8), dtype=np.uint16), chunks=chunks, significant_bits=significant_bits
    )
    return arrival(
        scope,
        frame,
        frame_time=A_TIME,
        channel_tiebreak=channel_tiebreak,
        to_plate=None,
        camera=A_CAMERA,
    )


def _fake_scope(positions):
    # a stand-in by design: frame_fact reads only the motion API, and a
    # position the simulator holds cannot be set unhomed or homing at will.
    return SimpleNamespace(
        motion=SimpleNamespace(axis_positions=lambda: positions, turret_status=lambda: None),
    )


def _to_plate(x_um, y_um):
    return PlatePosition(x_mm=x_um / 1000.0 + 0.5, y_mm=y_um / 1000.0 + 0.5)


class TestTheFactRidesWithItsFrame:
    def test_the_fact_given_at_ingest_reaches_the_write(self, tmp_path):
        clock = FakeClock(1000.0)
        writer = WriterStub(tmp_path, blocked=True)
        engine = _engine(writer, clock)
        engine.start(lambda: (_config(tmp_path), None))
        feed = FrameFeed()
        image, ts, chunk = feed.frame(clock(), with_camera_chunks=True)
        engine.ingest_frame(image, ts, chunk, fact=A_FACT)
        # Nothing has been written yet; whatever the world does now must
        # not change what the frame records.
        writer.unblock()
        engine.stop('user_stop')
        assert engine.wait_for_drain(timeout=5)
        assert writer.written_facts == [A_FACT]

    def test_a_caller_with_nothing_to_record_says_so(self, tmp_path):
        clock = FakeClock(1000.0)
        writer = WriterStub(tmp_path)
        engine = _engine(writer, clock)
        engine.start(lambda: (_config(tmp_path), None))
        feed = FrameFeed()
        image, ts, chunk = feed.frame(clock(), with_camera_chunks=True)
        with pytest.raises(TypeError):
            engine.ingest_frame(image, ts, chunk)
        engine.stop('user_stop')
        assert engine.wait_for_drain(timeout=5)


class TestTheFact:
    def test_a_known_stage_is_recorded_in_plate_millimetres(self):
        scope = _fake_scope(
            {
                'X': AxisPosition(AxisState.IDLE, 1000.0),
                'Y': AxisPosition(AxisState.IDLE, 2000.0),
                'Z': AxisPosition(AxisState.IDLE, 3.0),
            },
        )
        fact = frame_fact(scope, to_plate=_to_plate)
        assert fact == FrameFact(1.5, 2.5, 3.0, False)

    def test_an_unknown_axis_is_unknown_and_x_y_travel_as_a_pair(self):
        scope = _fake_scope(
            {
                'X': AxisPosition(AxisState.UNKNOWN, None),
                'Y': AxisPosition(AxisState.IDLE, 2000.0),
                'Z': AxisPosition(AxisState.IDLE, 3.0),
            }
        )
        fact = frame_fact(scope, to_plate=_to_plate)
        assert (fact.plate_x_mm, fact.plate_y_mm, fact.z_um) == (None, None, 3.0)

    def test_a_homing_axis_is_unknown_and_moving(self):
        scope = _fake_scope(
            {
                'X': AxisPosition(AxisState.IDLE, 1000.0),
                'Y': AxisPosition(AxisState.IDLE, 2000.0),
                'Z': AxisPosition(AxisState.HOMING, None),
            }
        )
        fact = frame_fact(scope, to_plate=_to_plate)
        assert fact.z_um is None
        assert fact.moving is True

    def test_a_turret_move_counts_as_moving(self):
        scope = _fake_scope(
            {
                'X': AxisPosition(AxisState.IDLE, 1000.0),
                'Y': AxisPosition(AxisState.IDLE, 2000.0),
                'Z': AxisPosition(AxisState.IDLE, 3.0),
                'T': AxisPosition(AxisState.MOVING, 2),
            }
        )
        assert frame_fact(scope, to_plate=_to_plate).moving is True

    def test_without_a_plate_transform_x_y_are_unknown_and_z_is_kept(self):
        scope = _fake_scope(
            {
                'X': AxisPosition(AxisState.IDLE, 1000.0),
                'Y': AxisPosition(AxisState.IDLE, 2000.0),
                'Z': AxisPosition(AxisState.IDLE, 3.0),
            }
        )
        fact = frame_fact(scope, to_plate=None)
        assert (fact.plate_x_mm, fact.plate_y_mm, fact.z_um) == (None, None, 3.0)

    def test_no_lit_led_records_the_channel_the_recording_started_on(self, sim_scope):
        arrived = _arrival(sim_scope, channel_tiebreak='Lumi')
        assert arrived.light == Light('Lumi', {})
        assert arrived.record.illumination_ma == {}

    def test_the_lit_led_names_the_frame_and_its_current_is_recorded(self, sim_scope):
        sim_scope.illumination.led_on(LED1, 100.0)
        arrived = _arrival(sim_scope)
        address = sim_scope.channel_identity.find(LED1).led_addresses[0]
        assert arrived.light == Light(LED1, {LED1: 100.0})
        assert arrived.record.illumination_ma == {address: 100.0}


class TestTheRecord:
    """A recorded frame's record states what the frame carries and what was
    true as it arrived, and nothing it was not told."""

    def test_a_frame_with_chunks_states_its_own_exposure_gain_and_clock(self, sim_scope):
        chunks = {'ExposureTime': 25000.0, 'Gain': 6.0, 'Timestamp': 123456789, 'FrameID': 41}
        record = _arrival(sim_scope, chunks=chunks, significant_bits=12).record

        assert (record.exposure_ms, record.exposure_source) == (25.0, SettingSource.FRAME)
        assert (record.gain_db, record.gain_source) == (6.0, SettingSource.FRAME)
        assert (record.camera_timestamp_ticks, record.frame_id) == (123456789, 41)
        assert record.captured_at == A_TIME
        assert record.frame_significant_bits == 12
        assert record.frames_summed == 1
        assert (record.binning_size, record.camera_model, record.camera_tick_hz) == A_CAMERA
        assert record.illumination_timing is LightTiming.ON_ARRIVAL
        assert record.held_channel == sim_scope.camera.get_held_layer()

    def test_a_frame_without_chunks_states_no_exposure_or_gain(self, sim_scope):
        """A setting can change while a recording runs, so a frame with no
        value of its own states none rather than a read from another moment."""
        record = _arrival(sim_scope, chunks=None).record

        assert (record.exposure_ms, record.exposure_source) == (None, None)
        assert (record.gain_db, record.gain_source) == (None, None)
        assert record.camera_timestamp_ticks is None

    def test_a_recorded_frame_is_unjudged_and_states_no_black_level(self, sim_scope):
        record = _arrival(sim_scope, chunks={'ExposureTime': 1000.0, 'Gain': 0.0}).record

        assert record.judgments is None, 'an empty tuple would say no change was recorded'
        assert record.black_level is None
        assert (record.clock_mapping, record.saturated_fraction, record.dark) == (None, None, None)
        assert record.auto_gain is None

    def test_a_setting_without_its_source_is_refused(self):
        with pytest.raises(ValueError, match='exposure and its source'):
            frame_record(exposure_source=None)
        with pytest.raises(ValueError, match='gain and its source'):
            frame_record(gain_db=None)


class TestTheFramesTime:
    """A recorded frame's time says what it is: its start on the camera's
    clock through the recording's mapping, or the host's arrival time."""

    _ARRIVED_NS = 1_755_000_009_000_000_000

    def _frame(self, chunks):
        return delivered_frame(np.zeros((2, 2), dtype=np.uint8), chunks, wall_ns=self._ARRIVED_NS)

    def test_a_frame_with_ticks_takes_its_start_through_the_mapping(self):
        # Latched at 5 s of ticks, at 1755000000 s on the wall; ticks a
        # quarter second later are a quarter second later, whenever the
        # frame arrived.
        mapping = ClockMapping(
            host_ns=0,
            camera_ticks=5_000_000_000,
            tick_hz=1_000_000_000,
            uncertainty_ns=0,
            wall_ns=1_755_000_000_000_000_000,
        )

        stated = camera_frame_time(self._frame({'Timestamp': 5_250_000_000}), mapping)

        assert stated == FrameTime(
            at=wall_time.local_time(1_755_000_000_250_000_000),
            source=TimestampSource.CAMERA_FRAME_START,
        )

    def test_a_frame_without_its_tick_has_no_camera_time(self):
        mapping = ClockMapping(
            host_ns=0, camera_ticks=0, tick_hz=1_000_000_000, uncertainty_ns=0, wall_ns=0
        )

        assert camera_frame_time(self._frame({'FrameID': 7}), mapping) is None
        assert camera_frame_time(self._frame(None), mapping) is None

    def test_a_frames_arrival_is_its_arrival(self):
        assert arrival_time(self._frame({'Timestamp': 5})) == FrameTime(
            at=wall_time.local_time(self._ARRIVED_NS), source=TimestampSource.HOST_ARRIVAL
        )


def _stated_fields(objective):
    """An objective as a frame's file states it: its own fields, a measurement
    the catalogue does not hold left out."""
    return {field: value for field, value in objective.to_dict().items() if value is not None}


def _written(scope, tmp_path, arrived, *, frame_number=0):
    """Write ``arrived``'s frame through the production edge and read its
    file back as the readers do."""
    path = tmp_path / f'frame_{frame_number}.tiff'
    write_frame(
        scope,
        np.zeros((8, 8), dtype=np.uint8),
        path,
        arrived,
        frame_number=frame_number,
        objective_id='10x Oly',
        labware=plate(),
        fallback_focus=None,
        false_color_on=False,
        channel_colors=release_colors(),
        save_encoding='8bit',
    )
    return read_file_record(path)


class TestTheLight:
    """What lit a recorded frame is one read of the scope's LEDs, named and measured from it."""

    def test_a_frame_two_leds_lit_states_both_and_is_named_for_the_first(self, sim_scope, tmp_path):
        sim_scope.illumination.led_on(LED0, 100.0)
        sim_scope.illumination.led_on(LED1, 50.0)

        written = _written(sim_scope, tmp_path, _arrival(sim_scope))

        assert written['channel'] == LED0
        assert written['illumination_by_channel_ma'] == {LED0: 100.0, LED1: 50.0}
        assert written['illumination_timing'] == 'on_arrival'

    def test_an_led_left_on_at_0_ma_neither_names_the_frame_nor_is_recorded(
        self, sim_scope, tmp_path
    ):
        sim_scope.illumination.led_on(LED0, 0.0)
        sim_scope.illumination.led_on(LED1, 100.0)

        written = _written(sim_scope, tmp_path, _arrival(sim_scope))

        assert written['channel'] == LED1
        assert written['illumination_by_channel_ma'] == {LED1: 100.0}


class TestTheFrameFile:
    """A recorded frame's file states what a still's does, from its own
    record, and the two facts only a recorded frame has."""

    def test_the_file_carries_the_fact_under_the_stills_keys(self, sim_scope, tmp_path):
        arrived = _arrival(sim_scope)._replace(fact=A_FACT)

        written = _written(sim_scope, tmp_path, arrived, frame_number=4)

        assert written['plate_pos_mm'] == {'x': 1.5, 'y': 2.5}
        assert written['z_pos_um'] == 3.0
        assert written['stage_moving'] is False
        assert written['recording_frame'] == 4

    def test_an_unknown_position_states_none(self, sim_scope, tmp_path):
        arrived = _arrival(sim_scope)._replace(fact=FrameFact(None, None, None, True))

        written = _written(sim_scope, tmp_path, arrived)

        assert 'plate_pos_mm' not in written
        assert 'z_pos_um' not in written
        assert written['stage_moving'] is True
        assert 'illumination_by_channel_ma' not in written, 'a frame no LED lit has no current'

    def test_the_file_states_the_frames_own_record(self, sim_scope, tmp_path):
        """The exposure and gain its chunks carried, its binning, the
        objective the recording started with, its plate, and its time with
        the clock it came from; no black level and no judgments."""
        chunks = {'ExposureTime': 25000.0, 'Gain': 6.0, 'Timestamp': 123456789, 'FrameID': 41}
        arrived = _arrival(sim_scope, chunks=chunks)

        written = _written(sim_scope, tmp_path, arrived)

        assert (written['exposure_time_ms'], written['gain_db']) == (25.0, 6.0)
        assert written['binning_size'] == A_CAMERA.binning_size
        assert written['objective'] == _stated_fields(sim_scope.objectives['10x Oly'])
        assert written['plate']['name'] == plate().name
        assert written['timestamp_iso'] == A_TIME.at.isoformat(timespec='microseconds')
        assert written['timestamp_source'] == 'host_arrival'
        assert (written['timestamp_camera_ticks'], written['frame_id']) == (123456789, 41)
        assert 'black_level' not in written
        assert 'frame_validity' not in written

    def test_a_frame_without_chunks_states_no_exposure_or_gain(self, sim_scope, tmp_path):
        written = _written(sim_scope, tmp_path, _arrival(sim_scope, chunks=None))

        assert 'exposure_time_ms' not in written
        assert 'gain_db' not in written


class TestTheBoundPlateTransform:
    def test_bound_to_the_labware_and_offset_selected_when_taken(self):
        plates = {
            'wide': SimpleNamespace(width_mm=100.0, height_mm=50.0),
            'narrow': SimpleNamespace(width_mm=10.0, height_mm=5.0),
        }
        settings = {'protocol.labware': 'wide', 'stage_offset': {'x': 0.0, 'y': 0.0}}
        state = RuntimeState(
            SimpleNamespace(
                read_setting=lambda path: copy.deepcopy(settings[path]),
                wellplate_loader=SimpleNamespace(get_plate=lambda labware_id: plates[labware_id]),
            )
        )
        to_plate = state.plate_transform()
        before = to_plate(1000.0, 2000.0)
        # A later selection, and an edit of the offset in place, move nothing
        # the transform already converts.
        settings['protocol.labware'] = 'narrow'
        settings['stage_offset']['x'] = 5000.0
        assert to_plate(1000.0, 2000.0) == before
        assert before == PlatePosition(x_mm=99.0, y_mm=48.0)

    def test_a_runs_transform_is_its_protocols_plate_and_the_offset_it_started_with(self):
        """A run states its frames in the frame its steps were driven in:
        the protocol's plate, whatever the scope has selected, and the
        offset the run started with, whatever is edited later."""
        from tests.test_run_refusal_contract import _make_single_step_protocol

        protocol = _make_single_step_protocol()
        plates = {
            protocol.labware_id(): SimpleNamespace(width_mm=100.0, height_mm=50.0),
            'narrow': SimpleNamespace(width_mm=10.0, height_mm=5.0),
        }
        settings = {'protocol.labware': 'narrow', 'stage_offset': {'x': 5000.0, 'y': 0.0}}
        state = RuntimeState(
            SimpleNamespace(
                read_setting=lambda path: copy.deepcopy(settings[path]),
                wellplate_loader=SimpleNamespace(get_plate=lambda labware_id: plates[labware_id]),
            )
        )
        # The selected plate and the live offset are never read for a
        # run's frames.
        to_plate = state.plate_transform(
            labware_id=protocol.labware_id(), stage_offset=StageOffset(x_um=0.0, y_um=0.0)
        )
        assert to_plate(1000.0, 2000.0) == PlatePosition(x_mm=99.0, y_mm=48.0)


def _engine(writer, clock):
    from _lumascope.modules.activity_claim import ActivityClaim

    return VideoRecordingEngine(
        write_frame=writer, claim=ActivityClaim(), clock=clock, steady_clock=host_clock.steady_ns
    )


def _config(out_dir):
    return RecordingConfig(
        fps=None,
        duration_s=60.0,
        width=8,
        height=8,
        bit_depth=8,
        output_dir=out_dir,
        filename_template='frame_{n:06d}.tiff',
        timestamp_overlay=False,
        timestamp_source=TimestampSource.HOST_ARRIVAL,
        manifest_extra={},
    )


class TestARecordingOnTheSimulator:
    """A frames recording on the simulated scope writes every frame through
    the still's builder: its own exposure and gain, the objective and plate
    it was taken under, its time with the clock it came from, and nothing
    the instrument did not report; its hyperstack states the frame's plate
    position as the frame's file does."""

    def test_each_frame_states_its_record_and_the_hyperstack_its_rounded_position(self, tmp_path):
        import threading

        from _lumascope.modules.recording_frames import MANUAL_HYPERSTACK_FILENAME
        from _lumascope.modules.scope_session import ScopeSession
        from tests.scope_fakes import home_sim_scope
        from tests.settings_fixtures import complete_settings

        settings = complete_settings(live_folder=str(tmp_path), video_as_frames=True)
        settings['image_output_format']['sequenced'] = 'OME-TIFF Hyperstack'
        session = ScopeSession.create(settings, simulate=True)
        delivered = threading.Event()
        count = {'n': 0}

        def _count(_frame):
            count['n'] += 1
            if count['n'] >= 3:
                delivered.set()

        try:
            # Homed, so every frame knows where it was taken.
            home_sim_scope(session.scope)
            session.show_channel(LED1)
            session.scope.illumination.led_on(LED1, 100.0)
            operation = session.manual_recording.start()
            session.scope.imaging.add_frame_listener(_count, name='test_count')
            assert delivered.wait(timeout=20), 'the simulated camera delivered no frames'
            operation.stop()
            operation.result(timeout=60)
            objective = _stated_fields(
                session.scope.objectives[session.scope.runtime_state.resolve_current_objective()[0]]
            )
            plate_name = session.scope.runtime_state.get_labware().name
        finally:
            session.scope.imaging.remove_frame_listener(_count)
            session.shutdown()

        folder = next((tmp_path / 'Manual').glob('Video_*'))
        frames = sorted(
            p for p in folder.glob('ManualVideo_Frame_*.tiff') if 'HyperStack' not in p.name
        )
        assert frames
        written = read_file_record(frames[0])
        assert written['channel'] == LED1
        assert written['exposure_time_ms'] > 0 and 'gain_db' in written
        assert written['objective'] == objective
        assert written['plate']['name'] == plate_name
        assert written['pixel_size_um'] is not None
        assert written['timestamp_source'] == 'camera_frame_start', 'the simulator latches'
        assert written['illumination_timing'] == 'on_arrival'
        assert 'frame_validity' not in written and 'black_level' not in written

        # The stage stood still, so every plane states the one position.
        stack = read_file_record(folder / MANUAL_HYPERSTACK_FILENAME)
        assert stack['plate_pos_mm'] == written['plate_pos_mm'], "the stack states the frame's"


class TestARecordingStatesEachFramesStart:
    """A recording's frame states the instant it started on the camera's
    clock, mapped through the one mapping the recording took as it started,
    on the frames leg's files and the MP4 leg's manifest alike; a frame that
    started before the recording did is not kept. The oracle is the
    simulator's own clock, never the production mapping."""

    _EXPOSURE_MS = 30.0
    # The latch's bracket on the simulator is microseconds wide; 3 ms covers
    # it and the two wall clock reads' own slew against the steady clock.
    _WITHIN = datetime.timedelta(milliseconds=3)

    def _record(self, tmp_path, *, video_as_frames):
        """A manual recording of a few frames on a simulated session: its
        folder, each delivered frame's true start by FrameID, and the wall
        time just before the recording was asked to start."""
        import threading

        from _lumascope.drivers import host_clock
        from _lumascope.modules.scope_session import ScopeSession
        from tests.settings_fixtures import complete_settings

        settings = complete_settings(live_folder=str(tmp_path), video_as_frames=video_as_frames)
        session = ScopeSession.create(settings, simulate=True)
        started = {}
        # The recording's start waits out the exposure change below, taking
        # frames the listener sees too, so six frames are counted from the
        # moment the recording is open.
        enough = {'n': None}
        six = threading.Event()
        try:
            scope = session.scope
            offset_ns = scope._camera_driver._clock_offset_ns
            scope.camera.set_exposure_ms(self._EXPOSURE_MS)

            def _truth(frame):
                exposure = scope.simulation.exposed_under(frame)
                steady_ns = exposure.started_ticks - offset_ns
                wall_ns = host_clock.wall_ns() + steady_ns - host_clock.steady_ns()
                started[frame.chunks['FrameID']] = wall_time.local_time(wall_ns)
                if enough['n'] is not None and len(started) >= enough['n']:
                    six.set()

            scope.imaging.add_frame_listener(_truth, name='the true starts')
            before = wall_time.now()
            operation = session.manual_recording.start()
            enough['n'] = len(started) + 6
            assert six.wait(timeout=20), 'the simulated camera delivered no frames'
            operation.stop()
            operation.result(timeout=60)
            scope.imaging.remove_frame_listener(_truth)
        finally:
            session.shutdown()
        return tmp_path / 'Manual', started, before

    def test_a_frames_recordings_first_frame_states_its_start(self, tmp_path):
        manual, started, before = self._record(tmp_path, video_as_frames=True)

        folder = next(manual.glob('Video_*'))
        first = read_file_record(sorted(folder.glob('ManualVideo_Frame_*.tiff'))[0])
        stated = datetime.datetime.fromisoformat(first['timestamp_iso'])
        assert first['timestamp_source'] == 'camera_frame_start'
        assert abs(stated - started[first['frame_id']]) <= self._WITHIN
        assert stated >= before, 'a frame started before the recording was kept'

    def test_an_mp4_recordings_manifest_states_its_first_frames_start(self, tmp_path):
        manual, started, before = self._record(tmp_path, video_as_frames=False)

        manifest = json.loads(next(manual.glob('*_manifest.json')).read_text())
        first = manifest['frame_index'][0]
        stated = wall_time.local_time(round(first['ts_s'] * 1e9))
        assert manifest['timestamp_grade'] == 'camera_frame_start'
        assert abs(stated - started[first['chunks']['FrameID']]) <= self._WITHIN
        assert stated >= before, 'a frame started before the recording was kept'


# --- A recording's frames take their capture's container -----------------


def _video_step(name):
    from tests.test_a_late_write_records_its_frame import _step

    return {
        **_step(name, 0, x=20.0, gain=1.0),
        'Acquire': 'video',
        'Video Config': {'duration': 0.5},
    }


def _record_a_video_step(tmp_path, *, sequenced_format, image_mode='8bit'):
    """Run one frames-leg video step headless on the simulator; its frames' paths, sorted."""
    from _lumascope.modules.recording_frames import frame_number
    from tests.test_a_late_write_records_its_frame import _protocol
    from tests.test_composite_run_e2e import headless_settings, open_composite_session

    settings = headless_settings(tmp_path, sequenced_format=sequenced_format)
    settings['video_as_frames'] = True
    settings['image_mode'] = image_mode
    with open_composite_session(settings) as (_session, runner):
        run = runner.start_single_scan(
            protocol=_protocol([_video_step('V1')]), parent_dir=str(tmp_path / 'runs')
        )
        assert run.result(timeout=60) is not None
        assert run.files.result(timeout=60) is not None
    return sorted((tmp_path / 'runs').rglob('*_video_Frame_*'), key=lambda p: frame_number(p.name))


def _record_manually(tmp_path, *, live_format, sequenced_format):
    """A manual "Frames" recording on the simulator; the frames' paths."""
    import threading

    from tests.test_composite_run_e2e import headless_settings, open_composite_session

    settings = headless_settings(
        tmp_path, sequenced_format=sequenced_format, live_format=live_format
    )
    settings['video_as_frames'] = True
    delivered = threading.Event()
    count = {'n': 0}

    def _count(_frame):
        count['n'] += 1
        if count['n'] >= 3:
            delivered.set()

    with open_composite_session(settings) as (session, _runner):
        session.show_channel(LED1)
        session.scope.illumination.led_on(LED1, 100.0)
        operation = session.manual_recording.start()
        session.scope.imaging.add_frame_listener(_count, name='test_count')
        try:
            assert delivered.wait(timeout=20), 'the simulated camera delivered no frames'
        finally:
            session.scope.imaging.remove_frame_listener(_count)
        operation.stop()
        operation.result(timeout=60)
    folder = next((tmp_path / 'Manual').glob('Video_*'))
    return [p for p in folder.iterdir() if p.name.startswith('ManualVideo_Frame_0')]


def _container(path):
    import tifffile

    with tifffile.TiffFile(path) as tif:
        return 'ome' if tif.is_ome else 'imagej' if tif.is_imagej else 'plain'


class TestARecordingTakesItsCapturesContainer:
    """A recording saved as frames is written in its capture's container: a
    protocol video step in the run's format, a manual recording in the Live
    format, as a manual still is. OME-TIFF writes OME frames named
    ``.ome.tiff``; every other format plain TIFF, so a JPG run's recording
    is never lossy and keeps its depth (a 12-bit frame in the ImageJ
    container, an 8-bit one plain). The frame states what it states in
    either container."""

    def test_an_ome_tiff_runs_frames_are_ome_and_state_what_an_imagej_frame_states(
        self, tmp_path, monkeypatch
    ):
        import _lumascope.modules.image_save as image_save
        import tifffile

        twins = tmp_path / 'twins'
        twins.mkdir()
        real_write = image_save.write_recorded_frame

        def _write_with_an_imagej_twin(scope, frame, file_loc, **kwargs):
            # The same frame and arguments through the same writer, named
            # .tiff: the ImageJ frame the parent wrote.
            real_write(scope, frame, file_loc, **kwargs)
            real_write(
                scope, frame, twins / pathlib.Path(file_loc).name.replace('.ome', ''), **kwargs
            )

        monkeypatch.setattr(image_save, 'write_recorded_frame', _write_with_an_imagej_twin)
        frames = _record_a_video_step(
            tmp_path, sequenced_format='OME-TIFF', image_mode='12bit_scientific'
        )

        assert frames, 'the video step wrote no frames'
        assert all(p.name.endswith('.ome.tiff') for p in frames), [p.name for p in frames]
        assert {_container(p) for p in frames} == {'ome'}
        twin = twins / frames[0].name.replace('.ome', '')
        assert _container(twin) == 'imagej'
        assert read_file_record(frames[0]) == read_file_record(twin)
        np.testing.assert_array_equal(tifffile.imread(frames[0]), tifffile.imread(twin))

    def test_a_jpg_runs_frames_are_twelve_bit_imagej_tiffs(self, tmp_path):
        import _lumascope.modules.image_utils as image_utils
        import tifffile

        frames = _record_a_video_step(
            tmp_path, sequenced_format='JPG', image_mode='12bit_scientific'
        )

        assert frames, 'the video step wrote no frames'
        assert all(p.suffixes == ['.tiff'] for p in frames), [p.name for p in frames]
        assert _container(frames[0]) == 'imagej'
        assert image_utils.read_tiff_significant_bits(frames[0]) == 12
        assert tifffile.imread(frames[0]).dtype == np.uint16

    @pytest.mark.parametrize(
        ('live_format', 'sequenced_format', 'suffixes', 'container'),
        [
            ('OME-TIFF', 'TIFF', ['.ome', '.tiff'], 'ome'),
            # The Sequenced format is a protocol's; a manual recording ignores it.
            ('TIFF', 'OME-TIFF', ['.tiff'], 'plain'),
        ],
    )
    def test_a_manual_recording_takes_the_live_format(
        self, tmp_path, live_format, sequenced_format, suffixes, container
    ):
        frames = _record_manually(
            tmp_path, live_format=live_format, sequenced_format=sequenced_format
        )

        assert frames, 'the manual recording wrote no frames'
        assert {tuple(p.suffixes[-len(suffixes) :]) for p in frames} == {tuple(suffixes)}
        assert {_container(p) for p in frames} == {container}
