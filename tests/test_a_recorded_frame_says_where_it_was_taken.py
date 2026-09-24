# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A recorded frame's file says where and how the frame was taken.

The fact is read when the frame arrives and rides the engine's queue with
it, because the write runs later, behind the backlog; the frame's TIFF
carries it under the still capture's position keys, so the file says what
it is with or without a hyperstack. These tests pin the engine's carry,
the fact's construction from the scope's tracked state, the metadata
builder's fields, and the plate transform a recording binds at start.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile as tf

from modules.image_save import write_video_frame
from modules.lumascope_api import AxisPosition, AxisState
from modules.lumascope_api.runtime_state import RuntimeState
from modules.recording_frames import FrameFact, frame_fact, tiff_frame_metadata
from modules.video_recording import RecordingConfig, VideoRecordingEngine
from tests.video_engine_harness import FakeClock, FrameFeed, WriterStub

A_FACT = FrameFact(plate_x_mm=1.5, plate_y_mm=2.5, z_um=3.0, moving=False, channel='Green')


def _fake_scope(positions, lit=None):
    return SimpleNamespace(
        motion=SimpleNamespace(axis_positions=lambda: positions),
        illumination=SimpleNamespace(
            get_led_states=lambda: {
                color: {'enabled': color == lit, 'illumination_ma': None}
                for color in ('Blue', 'Green', 'Red', 'BF')
            }
        ),
    )


def _to_plate(sx, sy):
    return sx / 1000.0 + 0.5, sy / 1000.0 + 0.5


class TestTheFactRidesWithItsFrame:
    def test_the_fact_given_at_ingest_reaches_the_write(self, tmp_path):
        clock = FakeClock(1000.0)
        writer = WriterStub(tmp_path, blocked=True)
        engine = _engine(writer, clock)
        engine.start(_config(tmp_path))
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
        engine.start(_config(tmp_path))
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
            lit='Green',
        )
        fact = frame_fact(scope, channel_tiebreak='BF', to_plate=_to_plate)
        assert fact == FrameFact(1.5, 2.5, 3.0, False, 'Green')

    def test_an_unknown_axis_is_unknown_and_x_y_travel_as_a_pair(self):
        scope = _fake_scope(
            {
                'X': AxisPosition(AxisState.UNKNOWN, None),
                'Y': AxisPosition(AxisState.IDLE, 2000.0),
                'Z': AxisPosition(AxisState.IDLE, 3.0),
            }
        )
        fact = frame_fact(scope, channel_tiebreak='BF', to_plate=_to_plate)
        assert (fact.plate_x_mm, fact.plate_y_mm, fact.z_um) == (None, None, 3.0)

    def test_a_homing_axis_is_unknown_and_moving(self):
        scope = _fake_scope(
            {
                'X': AxisPosition(AxisState.IDLE, 1000.0),
                'Y': AxisPosition(AxisState.IDLE, 2000.0),
                'Z': AxisPosition(AxisState.HOMING, None),
            }
        )
        fact = frame_fact(scope, channel_tiebreak='BF', to_plate=_to_plate)
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
        assert frame_fact(scope, channel_tiebreak='BF', to_plate=_to_plate).moving is True

    def test_without_a_plate_transform_x_y_are_unknown_and_z_is_kept(self):
        scope = _fake_scope(
            {
                'X': AxisPosition(AxisState.IDLE, 1000.0),
                'Y': AxisPosition(AxisState.IDLE, 2000.0),
                'Z': AxisPosition(AxisState.IDLE, 3.0),
            }
        )
        fact = frame_fact(scope, channel_tiebreak='BF', to_plate=None)
        assert (fact.plate_x_mm, fact.plate_y_mm, fact.z_um) == (None, None, 3.0)

    def test_no_lit_led_records_the_channel_the_recording_started_on(self):
        scope = _fake_scope({'Z': AxisPosition(AxisState.IDLE, 3.0)})
        assert frame_fact(scope, channel_tiebreak='Lumi', to_plate=None).channel == 'Lumi'


class TestTheFrameFile:
    def test_the_metadata_carries_the_fact_under_the_stills_keys(self):
        metadata, _ = tiff_frame_metadata(
            timestamp_s=1755000000.0,
            frame_number=4,
            chunks=None,
            tick_freq_hz=None,
            pixel_size_um=None,
            fact=A_FACT,
        )
        assert metadata['plate_pos_mm'] == {'x': 1.5, 'y': 2.5}
        assert metadata['x_pos'] == 1.5 and metadata['y_pos'] == 2.5
        assert metadata['z_pos_um'] == 3.0
        assert metadata['channel'] == 'Green'
        assert metadata['stage_moving'] is False

    def test_an_unknown_position_states_none(self):
        metadata, _ = tiff_frame_metadata(
            timestamp_s=1755000000.0,
            frame_number=0,
            chunks=None,
            tick_freq_hz=None,
            pixel_size_um=None,
            fact=FrameFact(None, None, None, True, 'BF'),
        )
        assert 'plate_pos_mm' not in metadata and 'x_pos' not in metadata
        assert 'z_pos_um' not in metadata
        assert metadata['stage_moving'] is True

    def test_a_frame_without_a_fact_is_refused(self):
        with pytest.raises(ValueError):
            tiff_frame_metadata(
                timestamp_s=1755000000.0,
                frame_number=0,
                chunks=None,
                tick_freq_hz=None,
                pixel_size_um=None,
                fact=None,
            )

    def test_the_written_file_reads_back_its_fact(self, tmp_path):
        metadata, _ = tiff_frame_metadata(
            timestamp_s=1755000000.0,
            frame_number=0,
            chunks=None,
            tick_freq_hz=None,
            pixel_size_um=None,
            fact=A_FACT,
        )
        path = tmp_path / 'frame.tiff'
        write_video_frame(
            frame=np.zeros((8, 8), dtype=np.uint8),
            file_loc=path,
            metadata=metadata,
            channel='Green',
            false_color_on=False,
            save_encoding='8bit',
            capture_depth=8,
        )
        with tf.TiffFile(path) as t:
            described = json.loads(t.pages[0].tags['ImageDescription'].value)
        assert described['plate_pos_mm'] == {'x': 1.5, 'y': 2.5}
        assert described['channel'] == 'Green'
        assert described['stage_moving'] is False


class TestTheBoundPlateTransform:
    def test_none_until_labware_and_offset_are_registered(self):
        state = RuntimeState(SimpleNamespace())
        assert state.plate_transform() is None
        state.set_labware(SimpleNamespace(get_dimensions=lambda: {'x': 100.0, 'y': 50.0}))
        assert state.plate_transform() is None
        state.set_stage_offset({'x': 0.0, 'y': 0.0})
        assert state.plate_transform() is not None

    def test_bound_to_the_labware_registered_when_taken(self):
        state = RuntimeState(SimpleNamespace())
        state.set_labware(SimpleNamespace(get_dimensions=lambda: {'x': 100.0, 'y': 50.0}))
        state.set_stage_offset({'x': 0.0, 'y': 0.0})
        to_plate = state.plate_transform()
        before = to_plate(1000.0, 2000.0)
        state.set_labware(SimpleNamespace(get_dimensions=lambda: {'x': 10.0, 'y': 5.0}))
        state.set_stage_offset(None)
        assert to_plate(1000.0, 2000.0) == before
        assert before == (99.0, 48.0)


def _engine(writer, clock):
    from modules.activity_claim import ActivityClaim

    return VideoRecordingEngine(write_frame=writer, claim=ActivityClaim(), clock=clock, notify=None)


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
        manifest_extra={},
    )
