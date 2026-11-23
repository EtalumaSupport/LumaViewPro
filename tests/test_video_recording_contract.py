# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Contract tests for the video recording engine (src/_lumascope/modules/video_recording.py).

These are the engine's specification, written against the signatures-only
skeleton: they run RED until the engine implementation lands, and the
implementation is done when they are green. Each class pins one contract;
none of them sleeps against the wall clock -- the engine accepts an
injected time source precisely so these stay deterministic.
"""

import dataclasses
import json
import typing
import logging
import pathlib
import sys
import threading

import pytest

from _lumascope.drivers import host_clock
from tests.settings_fixtures import complete_settings

from _lumascope.modules import path_utils
from _lumascope.modules.activity_claim import ActivityClaim, current_taking
from _lumascope.modules.caller import Caller
import _lumascope.modules.video_recording as video_recording_module
from _lumascope.modules.exceptions import ProtocolRunRefusedError, RecordingRefusedError
from _lumascope.modules.lumascope_api.frame_record import TimestampSource
from _lumascope.modules.video_recording import RecordingConfig, VideoRecordingEngine
from _lumascope.modules.video_recording import read_memory_headroom as _real_read_memory_headroom
from tests.video_engine_harness import (
    FakeClock,
    FrameFeed,
    WriterStub,
)
from tests.protocol_drives import run_identity
from tests.log_capture import records_of

# Every recording manifest must carry at least these keys; downstream
# consumers (support bundles, char tooling, the end-of-run report) key
# off them.
REQUIRED_MANIFEST_KEYS = {
    'end_reason',
    'frames_written',
    'write_failures',
    'frames_unstamped',
    'timestamp_grade',
    'measured_fps',
    'measured_duration_s',
    'frame_index',
}


def make_config(
    tmp_path,
    fps=5.0,
    duration_s=2.0,
    width=8,
    height=6,
    bit_depth=8,
    timestamp_overlay=True,
    timestamp_source=TimestampSource.CAMERA_FRAME_START,
    manifest_extra=None,
    manifest_filename=video_recording_module.MANIFEST_FILENAME,
):
    return RecordingConfig(
        fps=fps,
        duration_s=duration_s,
        width=width,
        height=height,
        bit_depth=bit_depth,
        output_dir=pathlib.Path(tmp_path),
        filename_template='frame_{n:06d}.tiff',
        timestamp_overlay=timestamp_overlay,
        timestamp_source=timestamp_source,
        manifest_extra=manifest_extra,
        manifest_filename=manifest_filename,
    )


class SingleFileWriterStub:
    """Writer edge for the flat-MP4 shape: every frame lands in ONE file.

    Like the real MP4 writer, its name is picked when it is made, by
    ``path_utils.free_stem``, so an existing file at the requested name moves
    it to a numbered one; the file appears at the first write, the first
    time the engine is told the name. So the name the caller asked for is not
    necessarily the name on disk.
    """

    def __init__(self, out_dir: pathlib.Path, requested_name: str):
        self.out_dir = pathlib.Path(out_dir)
        requested = self.out_dir / requested_name
        stem = path_utils.free_stem(self.out_dir, requested.stem, (requested.suffix,))
        self._name = requested.with_name(f'{stem}{requested.suffix}')
        self.output_path: pathlib.Path | None = None
        self.frames_written = 0

    def __call__(
        self, image, timestamp_s, frame_number, config, chunks=None, fact=None
    ) -> pathlib.Path:
        if self.output_path is None:
            self.output_path = self._name
            self.output_path.write_bytes(b'')
        self.frames_written += 1
        return self.output_path


def make_engine(tmp_path, *, clock=None, writer=None, claim=None, writers=1):
    clock = clock or FakeClock()
    writer = writer if writer is not None else WriterStub(tmp_path)
    claim = claim or ActivityClaim()
    engine = VideoRecordingEngine(
        write_frame=writer,
        writers=writers,
        claim=claim,
        clock=clock,
        steady_clock=host_clock.steady_ns,
    )
    return engine, writer, clock, claim


def feed_uniform(engine, clock, feed, *, delivery_fps, duration_s, chunks=True):
    """Deliver frames at delivery_fps for duration_s, clock in lockstep."""
    step = 1.0 / delivery_fps
    n = int(duration_s * delivery_fps)
    for _ in range(n):
        clock.advance(step)
        image, ts, chunk = feed.frame(clock(), with_camera_chunks=chunks)
        engine.ingest_frame(image, ts, chunk, fact=None)


class TestBudgetContract:
    def test_budget_is_exact_ceil(self, tmp_path):
        assert make_config(tmp_path, fps=5, duration_s=2).frame_budget == 10
        assert make_config(tmp_path, fps=0.5, duration_s=3).frame_budget == 2

    def test_no_truncation_at_any_duration(self, tmp_path):
        # An hour at 40 fps is 144,000 frames; no cap anywhere.
        assert make_config(tmp_path, fps=40, duration_s=3600).frame_budget == 144000

    def test_budget_full_closes_selection(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        config = make_config(tmp_path, fps=5, duration_s=1)  # budget 5
        engine.start(lambda: (config, None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=50, duration_s=1)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().frames_selected == 5

    def test_duration_elapsed_closes_selection(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        feed = FrameFeed()
        # Deliver well past the configured duration; selection must close
        # itself at the duration boundary without an explicit stop().
        feed_uniform(engine, clock, feed, delivery_fps=20, duration_s=3)
        assert not engine.is_recording
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().frames_selected == 5


class TestARateLimitedRecordingEndsOnItsFramesClock:
    """With a rate, a recording ends at its last slot's deadline on its frames' clock.

    The deadline is the start plus budget / fps, the instant the budget's last
    slot opens. A camera at the rate fills the budget there; one below it is
    closed by its first frame at or past that instant, once the frame has
    taken the slot it is owed. No caller's clock decides the content.
    """

    def _deliver(self, engine, start, offsets_s):
        feed = FrameFeed()
        for offset in offsets_s:
            image, ts, chunks = feed.frame(start + offset)
            engine.ingest_frame(image, ts, chunks, fact=None)

    def test_a_camera_below_the_rate_ends_at_the_duration_with_what_it_delivered(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=2), None))
        # 5 fps against 10 for 3 s: the budget of 20 never fills, and the
        # frame at 2.0 s is the first at the last slot's deadline.
        self._deliver(engine, clock(), [0.2 * k for k in range(1, 16)])
        assert not engine.is_recording, 'frames past the duration kept the recording open'
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert (result.end_reason, result.frames_selected) == ('duration_elapsed', 10)
        # The end reads as a complete recording; the budget beside the count
        # is what says it kept 10 of the 20 its rate and duration allow.
        manifest = json.loads(result.manifest_path.read_text())
        assert (result.frame_budget, manifest['frame_budget']) == (20, 20)
        assert manifest['frames_selected'] == 10
        assert result.summary().startswith('10 written of a 20-frame budget, 0 failed'), (
            result.summary()
        )

    def test_a_budget_whose_last_slot_is_past_the_duration_fills_it(self, tmp_path):
        # 3 fps for 1.5 s is ceil(4.5) = 5 slots; the last opens at 5/3 s,
        # after the duration, and its frame fills the budget before any
        # time end is read.
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=3, duration_s=1.5), None))
        self._deliver(engine, clock(), [k / 30 for k in range(1, 61)])
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert (result.end_reason, result.frames_selected) == ('frame_budget_filled', 5)


class TestRateContract:
    def test_configured_rate_honored_under_fast_delivery(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=2), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=40, duration_s=2)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().frames_selected == 10

    def test_over_configuration_is_safe(self, tmp_path):
        # Configured faster than the camera delivers: every delivered
        # frame is kept and the shortfall is reported, never an error.
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=50, duration_s=2), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=2)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.frames_selected == 20
        # 50 fps over 2 s promised 100 frames. The shortfall is visible in
        # the counts themselves, and it is not an error.
        assert result.frames_selected < result.configured_fps * 2
        assert result.aborted is False


class TestNoRateLimit:
    """A recording with no rate limit keeps every frame the camera delivers."""

    def test_no_budget_without_a_rate(self, tmp_path):
        assert make_config(tmp_path, fps=None, duration_s=2).frame_budget is None

    def test_every_delivered_frame_is_kept(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=None, duration_s=2), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=69, duration_s=1)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.frames_selected == 69
        assert result.configured_fps is None
        assert result.measured_fps == pytest.approx(69.0)

    def test_duration_elapsed_closes_selection(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        # 1.025 s sits between the 20th frame (1.00 s) and the 21st
        # (1.05 s), so float accumulation in the feed cannot move the edge.
        engine.start(lambda: (make_config(tmp_path, fps=None, duration_s=1.025), None))
        # Deliver well past the duration with no stop(): with no budget,
        # the duration itself must end selection.
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=3)
        assert not engine.is_recording
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.end_reason == 'duration_elapsed'
        assert result.frames_selected == 20

    def test_manifest_records_no_configured_rate(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=None, duration_s=2), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest['configured_fps'] is None
        assert manifest['frames_selected'] == 10
        # With no rate there is no count to fall short of.
        assert (result.frame_budget, manifest['frame_budget']) == (None, None)
        assert result.summary().startswith('10 written, 0 failed'), result.summary()


class TestABacklogWithMemoryToSpareIsAllKept:
    def test_a_lagging_writer_with_memory_to_spare_causes_no_capture_drop(self, tmp_path):
        # The PC's memory reads ample (the conftest's reading).
        writer = WriterStub(tmp_path, blocked=True)
        engine, _, clock, _ = make_engine(tmp_path, writer=writer)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=2), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=2)
        engine.stop('stopped')
        # Writer wrote nothing, yet every selected frame is queued.
        assert engine.pending_writes == 20
        writer.unblock()
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.frames_selected == 20
        assert result.frames_written == 20
        assert result.write_failures == 0


class TestDrainContinuesAfterCapture:
    def test_backlog_drains_after_stop(self, tmp_path):
        writer = WriterStub(tmp_path, blocked=True)
        engine, _, clock, _ = make_engine(tmp_path, writer=writer)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        assert not engine.is_recording
        assert engine.is_draining
        writer.unblock()
        assert engine.wait_for_drain(timeout=5)
        assert not engine.is_draining
        assert len(writer.written) == 10
        # Frame numbers are contiguous ordinals from enqueue order.
        assert [n for n, _, _ in writer.written] == list(range(10))


class TestStopPromptness:
    def test_stop_closes_selection_within_one_decision(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=10), None))
        feed = FrameFeed()
        feed_uniform(engine, clock, feed, delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        assert not engine.is_recording
        # The engine's own count: the writer lane runs on, and a sum of its
        # queue and its written list counts a frame twice mid-write.
        selected_at_stop = engine.frames_selected
        # Frames delivered after stop are never selected.
        feed_uniform(engine, clock, feed, delivery_fps=10, duration_s=1)
        assert engine.frames_selected == selected_at_stop
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().frames_selected == selected_at_stop

    def test_a_stops_reason_is_one_of_the_published_set_and_any_other_word_is_refused(
        self, tmp_path
    ):
        # The manifest and RecordingFinished carry the reason, and a caller
        # branches on it, so a word outside the set never closes selection.
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=10), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        with pytest.raises(ValueError, match="'user_stop'"):
            engine.stop('user_stop')
        assert engine.is_recording, 'a refused word closed selection'
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        from lumascope import RecordingEndReason

        assert engine.result().end_reason is RecordingEndReason.STOPPED


class TestLossIsNeverSilent:
    def test_per_frame_write_failure_costs_that_frame_only(self, tmp_path):
        writer = WriterStub(tmp_path, fail_frames={3})
        engine, _, clock, _ = make_engine(tmp_path, writer=writer)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.frames_selected == 10
        assert result.frames_written == 9
        assert result.write_failures == 1
        assert result.aborted is False

    def test_discard_pending_is_loud(self, tmp_path):
        writer = WriterStub(tmp_path, blocked=True)
        engine, _, clock, _ = make_engine(tmp_path, writer=writer)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        engine.discard_pending()
        result = engine.result()
        assert result.frames_written < result.frames_selected

    def test_zero_frame_recording_reports_honestly(self, tmp_path):
        # A recording that never received a frame (camera stalled or
        # disconnected) completes as an honest empty result -- it is
        # never treated as a clean normal recording, and computing the
        # measured statistics from zero timestamps must not crash.
        engine, _writer, _clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=1), None))
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.frames_selected == 0
        assert result.frames_written == 0
        assert result.aborted is False
        assert result.measured_fps == 0.0
        assert result.measured_duration_s == 0.0


class TestFatalityClassification:
    def test_writer_lane_death_aborts_the_recording(self, tmp_path):
        writer = WriterStub(tmp_path, die_on_frame=2)
        engine, _, clock, _ = make_engine(tmp_path, writer=writer)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.aborted is True
        assert isinstance(result.writer_failure, SystemExit)

    def test_under_delivery_is_not_fatal(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=50, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().aborted is False


class TestMeasuredTruth:
    def test_result_reports_measured_not_configured(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=2), None))
        # Camera actually delivers at 7 fps: measured truth must say so.
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=7, duration_s=2)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.configured_fps == 10
        assert 6.0 < result.measured_fps < 8.0
        assert 1.5 < result.measured_duration_s < 2.1

    def test_the_grade_is_the_source_the_recording_started_with(self, tmp_path):
        # The controller states its frames' source once; the engine never
        # infers it from the frames, which carry chunks either way.
        for source in (TimestampSource.CAMERA_FRAME_START, TimestampSource.HOST_ARRIVAL):
            folder = tmp_path / source
            folder.mkdir()
            engine, _writer, clock, _ = make_engine(folder)
            engine.start(
                lambda folder=folder, source=source: (
                    make_config(folder, fps=5, duration_s=1, timestamp_source=source),
                    None,
                )
            )
            feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1, chunks=True)
            engine.stop('stopped')
            assert engine.wait_for_drain(timeout=5)
            assert engine.result().timestamp_grade is source
            manifest = json.loads(engine.result().manifest_path.read_text())
            assert manifest['timestamp_grade'] == source.value

    def test_an_unstamped_frame_is_counted_and_never_selected(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.note_unstamped(None)  # before the recording: nothing to count it in
        engine.start(lambda: (make_config(tmp_path, fps=None, duration_s=10), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=0.3)
        engine.note_unstamped(None)
        engine.note_unstamped(None)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert (result.frames_selected, result.frames_unstamped) == (3, 2)
        assert json.loads(result.manifest_path.read_text())['frames_unstamped'] == 2


class TestManifestTruth:
    def test_manifest_written_with_required_schema(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=2), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=2)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        manifest_path = engine.result().manifest_path
        assert manifest_path is not None and manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        missing = REQUIRED_MANIFEST_KEYS - set(manifest)
        assert not missing, f'manifest missing keys: {sorted(missing)}'


class TestManifestNamesTheArtifactItDescribes:
    """A manifest belongs to the file that was written, not the one requested.

    The flat-MP4 leg shares one folder across recordings, so two recordings
    starting inside the same wall-clock second ask for the same video name.
    The writer renames the second one; a manifest named from the request
    would then overwrite the first recording's manifest AND describe a video
    it never measured.
    """

    def _record(self, tmp_path, clock_start, frames, requested='Video_same_second.mp4'):
        writer = SingleFileWriterStub(tmp_path, requested)
        engine, _w, clock, _c = make_engine(tmp_path, writer=writer)
        engine.start(
            lambda: (make_config(tmp_path, fps=5, duration_s=10, manifest_filename=None), None)
        )
        feed = FrameFeed()
        for _ in range(frames):
            clock.advance(1.0 / 5)
            image, ts, chunk = feed.frame(clock(), with_camera_chunks=True)
            engine.ingest_frame(image, ts, chunk, fact=None)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        return engine.result(), writer

    def test_same_second_recordings_each_keep_their_own_manifest(self, tmp_path):
        first, first_writer = self._record(tmp_path, 0.0, frames=3)
        second, second_writer = self._record(tmp_path, 0.0, frames=7)

        # The writer renamed the second video; the manifests must follow.
        assert first_writer.output_path != second_writer.output_path
        assert first.manifest_path != second.manifest_path
        assert first.manifest_path.exists() and second.manifest_path.exists()

        # Each manifest describes the video whose name it carries -- the
        # failure this pins is a manifest reporting the OTHER recording's
        # frame count.
        for result, writer in ((first, first_writer), (second, second_writer)):
            manifest = json.loads(result.manifest_path.read_text())
            assert result.manifest_path.stem == f'{writer.output_path.stem}_manifest'
            assert manifest['frames_written'] == writer.frames_written

    def test_manifest_takes_the_name_the_writer_resolved(self, tmp_path):
        (tmp_path / 'Video_taken.mp4').write_bytes(b'')  # force the rename
        result, writer = self._record(tmp_path, 0.0, frames=2, requested='Video_taken.mp4')
        assert writer.output_path.name == 'Video_taken_001.mp4'
        assert result.manifest_path.name == 'Video_taken_001_manifest.json'

    def test_a_pinned_name_still_wins(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.manifest_path.name == video_recording_module.MANIFEST_FILENAME

    def test_a_recording_that_wrote_nothing_leaves_no_manifest(self, tmp_path):
        writer = SingleFileWriterStub(tmp_path, 'Video_empty.mp4')
        engine, _w, _clock, _c = make_engine(tmp_path, writer=writer)
        engine.start(
            lambda: (make_config(tmp_path, fps=5, duration_s=10, manifest_filename=None), None)
        )
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        # No frames, so no container was ever opened and no file exists to
        # attach a manifest to. Naming one after the request would describe
        # a file that was never created.
        assert writer.output_path is None
        assert engine.result().manifest_path is None
        assert list(tmp_path.glob('*_manifest.json')) == []


class TestFrameIdentityTravelsWithTheFrame:
    """Chunk metadata rides the queue to the write edge and the manifest.

    The write edge stamps per-frame TIFF metadata (camera ticks, frame
    id) and the manifest is the downstream reader's index; if the engine
    dropped chunks at the queue boundary, both would silently lose the
    camera-grade identity the frames were captured with.
    """

    def test_chunks_reach_the_write_edge(self, tmp_path):
        engine, writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        assert writer.written, 'no frames written'
        assert len(writer.written_chunks) == len(writer.written)
        assert all(c is not None for c in writer.written_chunks)

    def test_manifest_frame_index_carries_chunks(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        manifest = json.loads(engine.result().manifest_path.read_text())
        assert manifest['frame_index'], 'empty frame index'
        for entry in manifest['frame_index']:
            assert entry['chunks'] is not None


class TestConfigSnapshotContract:
    def test_timestamp_overlay_is_required(self, tmp_path):
        # The overlay choice may never be silently defaulted: a config
        # without it must be unconstructible.
        with pytest.raises(TypeError):
            RecordingConfig(
                fps=5.0,
                duration_s=1.0,
                width=8,
                height=6,
                bit_depth=8,
                output_dir=pathlib.Path(tmp_path),
                filename_template='frame_{n:06d}.tiff',
                timestamp_source=TimestampSource.CAMERA_FRAME_START,
            )

    def test_manifest_extra_merges_and_engine_truth_wins(self, tmp_path):
        extra = {
            'provenance': {'hostname': 'bench-host'},
            'measured_fps': 'caller-lie',
        }
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(
            lambda: (make_config(tmp_path, fps=5, duration_s=1, manifest_extra=extra), None)
        )
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        manifest = json.loads(engine.result().manifest_path.read_text())
        assert manifest['provenance'] == {'hostname': 'bench-host'}
        # The colliding key is the engine's measurement, not the caller's.
        assert manifest['measured_fps'] != 'caller-lie'


class TestExclusivity:
    def test_engine_refuses_second_capture(self, tmp_path):
        engine, _writer, _clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=10), None))
        with pytest.raises(RecordingRefusedError) as excinfo:
            engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=10), None))
        # The reason codes are L2 vocabulary (SDK/REST callers dispatch
        # on them), so the strings are pinned, not just the raise.
        assert excinfo.value.reason == 'recording_active'

    def test_engine_refuses_when_claim_held_by_protocol(self, tmp_path):
        claim = ActivityClaim()
        assert claim.try_claim('protocol', run=run_identity(Caller.GUI))
        engine, _writer, _clock, _ = make_engine(tmp_path, claim=claim)
        with pytest.raises(RecordingRefusedError) as excinfo:
            engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        assert excinfo.value.reason == 'exclusive_activity_running'
        # Busy-with-what rides the payload: the holder's kind, and the
        # caller that started the holding run -- a run claim cannot be taken
        # without its run -- with the run named by its kind in the sentence.
        assert excinfo.value.holder == 'protocol'
        assert excinfo.value.holder_trigger == Caller.GUI
        assert excinfo.value.message.startswith('The scan run is using the microscope')

    @pytest.mark.parametrize(
        ('kind', 'run', 'named'),
        [
            ('protocol', run_identity(words='Z-stack'), 'The Z-stack run'),
            ('diagnostic', None, 'A diagnostic'),
            ('home', None, 'A home'),
        ],
        ids=['run', 'diagnostic', 'home'],
    )
    def test_the_refusal_names_what_holds_the_scope(self, tmp_path, kind, run, named):
        """One phrasing with every other holder refusal: a run by its kind,
        another activity by its kind -- never who started it."""
        claim = ActivityClaim()
        assert claim.try_claim(kind, run=run)
        engine, _writer, _clock, _ = make_engine(tmp_path, claim=claim)
        with pytest.raises(RecordingRefusedError) as excinfo:
            engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        assert excinfo.value.message.startswith(f'{named} is using the microscope')

    def test_a_recording_inside_a_run_leaves_the_runs_claim_held(self, tmp_path):
        """A video step records under the run's claim; its end must not
        free the claim the run holds until run end."""
        claim = ActivityClaim()
        run = claim.try_claim('protocol', run=run_identity())
        engine, _writer, _clock, _ = make_engine(tmp_path, claim=run.lend())

        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=10), None))
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)

        assert claim.owner == 'protocol', "the recording's end released the run's claim"
        assert claim.try_claim('recording') is None
        run.release()
        assert claim.owner is None

    def test_a_recording_lent_a_claim_its_run_no_longer_holds_is_refused(self, tmp_path):
        claim = ActivityClaim()
        run = claim.try_claim('protocol', run=run_identity())
        lent = run.lend()
        run.release()
        engine, _writer, _clock, _ = make_engine(tmp_path, claim=lent)

        with pytest.raises(RecordingRefusedError) as excinfo:
            engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        assert excinfo.value.reason == 'exclusive_activity_running'

    def test_claim_refusal_names_who_started_the_holding_run(self, tmp_path):
        claim = ActivityClaim()
        assert claim.try_claim('protocol', run=run_identity(Caller.REST))
        engine, _writer, _clock, _ = make_engine(tmp_path, claim=claim)
        with pytest.raises(RecordingRefusedError) as excinfo:
            engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        assert excinfo.value.holder == 'protocol'
        assert excinfo.value.holder_trigger == Caller.REST

    def test_concurrent_starts_exactly_one_wins(self, tmp_path):
        claim = ActivityClaim()
        outcomes = []
        barrier = threading.Barrier(2)

        def _try_start():
            engine, _, _, _ = make_engine(tmp_path, claim=claim)
            config = make_config(tmp_path, fps=5, duration_s=10)
            barrier.wait(timeout=5)
            try:
                engine.start(lambda: (config, None))
                outcomes.append('won')
            except RecordingRefusedError:
                outcomes.append('refused')

        threads = [threading.Thread(target=_try_start) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert sorted(outcomes) == ['refused', 'won']

    def test_claim_released_after_drain(self, tmp_path):
        claim = ActivityClaim()
        engine, _writer, clock, _ = make_engine(tmp_path, claim=claim)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        assert claim.owner == 'recording'
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        assert claim.owner is None


class TestSessionActivityClaim:
    """The session-tier half of the exclusivity contract.

    ScopeSession owns ONE compare-and-claim primitive (owner 'protocol'
    XOR 'recording') that both the sequenced-run gate and the engine
    acquire. These pin the session surface the engine's claim handle
    injects from.
    """

    @pytest.fixture
    def headless_session(self, tmp_path):
        from _lumascope.modules.scope_session import ScopeSession

        settings = {
            'stage_offset': {'x': 0.0, 'y': 0.0},
            'live_folder': str(tmp_path),
            'protocol': {
                'autogain': {
                    'target_brightness': 0.3,
                    'min_gain_db': 0.0,
                    'max_gain_db': 20.0,
                },
            },
        }
        session = ScopeSession.create(complete_settings(**settings), simulate=True)
        yield session
        session.shutdown()

    def test_session_owns_one_activity_claim(self, headless_session):
        claim = headless_session._activity_claim
        recording = claim.try_claim('recording')
        assert recording
        assert not claim.try_claim('protocol', run=run_identity())
        recording.release()
        protocol = claim.try_claim('protocol', run=run_identity())
        assert protocol
        protocol.release()

    def test_two_thread_claim_atomicity(self, headless_session):
        claim = headless_session._activity_claim
        wins = []
        barrier = threading.Barrier(2)

        def _race(owner, run):
            barrier.wait(timeout=5)
            held = claim.try_claim(owner, run=run)
            if held:
                wins.append(held)

        threads = [
            threading.Thread(target=_race, args=('protocol', run_identity())),
            threading.Thread(target=_race, args=('recording', None)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(wins) == 1
        wins[0].release()

    def test_protocol_start_refused_while_recording_holds_claim(self, headless_session, tmp_path):
        from tests.test_run_refusal_contract import (
            _make_single_step_protocol,
        )

        recording = headless_session._activity_claim.try_claim('recording')
        assert recording
        runner = headless_session.protocol_runner()
        with pytest.raises(ProtocolRunRefusedError):
            runner.start_single_scan(
                protocol=_make_single_step_protocol(),
                sequence_name='refused_while_recording',
                parent_dir=str(tmp_path),
            )
        recording.release()


class TestEndReason:
    """The manifest says WHY selection ended, not just that it was short.

    A support bundle must distinguish a user stop from a camera death; the
    frame counts run short for both.
    """

    def test_stop_reason_lands_in_manifest_and_result(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=2), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=1)
        engine.stop('camera_stalled')
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.end_reason == 'camera_stalled'
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest['end_reason'] == 'camera_stalled'

    def test_budget_fill_names_itself(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=2)
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().end_reason == 'frame_budget_filled'

    def test_first_close_wins(self, tmp_path):
        # A stop on an already-closed recording must not rewrite why it
        # ended -- the budget filled first, and that stays the record.
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=2)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().end_reason == 'frame_budget_filled'


class _QueueThatFailsToRead:
    """A lane queue whose read raises: an escape outside the per-frame try."""

    def put(self, item):
        pass

    def get(self):
        raise RuntimeError('scripted queue read failure')

    def get_nowait(self):
        raise RuntimeError('scripted queue read failure')


@dataclasses.dataclass(eq=False)
class _TakeWithAQueueThatFailsToRead(video_recording_module._Take):
    """A recording whose lane queue fails to read."""

    lane_queue: object = dataclasses.field(default_factory=_QueueThatFailsToRead)


class _NoThreadStarts(threading.Thread):
    """Thread exhaustion: every start raises, as the host's does.

    A stand-in by design: thread exhaustion is the host's, not the
    simulator's. A subclass, not a function: the patch is the process's one
    ``threading`` module, and a module first imported under it evaluates
    ``threading.Thread | None``.
    """

    def start(self):
        raise RuntimeError('scripted thread exhaustion')


class TestClaimLifetime:
    """The claim is freed on every end path, including the failing ones.

    A stranded claim outlives the engine that took it and blocks every
    later recording AND protocol run for the life of the process, with no
    user-visible cause -- so each of these asserts the claim is free, not
    merely that the recording looks finished.
    """

    def test_successful_recording_releases_and_reports(self, tmp_path):
        engine, _writer, clock, claim = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=2)
        assert engine.wait_for_drain(timeout=5)
        assert claim.owner is None
        assert engine.result().manifest_path is not None
        assert not engine.is_recording

    def test_lane_death_clears_recording_and_releases(self, tmp_path):
        writer = WriterStub(tmp_path, die_on_frame=2)
        engine, _writer, clock, claim = make_engine(tmp_path, writer=writer)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=2), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=1)
        assert engine.wait_for_drain(timeout=5)
        # The app-close gate reads is_recording; a lane death that leaves it
        # set means the application can never be closed again.
        assert not engine.is_recording
        assert claim.owner is None
        assert engine.result().aborted

    def test_finalize_escape_still_releases_and_drains(self, tmp_path, monkeypatch):
        engine, _writer, clock, claim = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))

        def _boom(*args, **kwargs):
            raise RuntimeError('scripted manifest failure')

        monkeypatch.setattr(engine, '_write_manifest', _boom)
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=2)
        # Without a release in a finally the lane leaves the claim held and
        # drained clear, and every later wait_for_drain parks forever.
        assert engine.wait_for_drain(timeout=5)
        assert claim.owner is None

    def test_double_finalize_does_not_double_release(self, tmp_path):
        # Behavior-preservation guard: the claim stub raises on a release by
        # a non-owner, so a second end path reaching the release would fail
        # loudly here rather than silently freeing someone else's claim. The
        # second path is the discard, after the lanes have finalized.
        engine, _writer, clock, claim = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=2)
        assert engine.wait_for_drain(timeout=5)
        written = engine.result().frames_written
        engine.discard_pending()
        assert claim.owner is None
        assert engine.result().frames_written == written

    def test_discard_with_wedged_writer_releases(self, tmp_path):
        # The app-close Discard hatch: the lane is stuck inside a write and
        # will never exit, so the release cannot be the lane's alone.
        writer = WriterStub(tmp_path, blocked=True)
        engine, _writer, clock, claim = make_engine(tmp_path, writer=writer)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop('stopped')
        engine.discard_pending()
        assert claim.owner is None
        assert engine.result().frames_written < engine.result().frames_selected

    def test_start_failure_releases_claim(self, tmp_path, monkeypatch):
        # Nothing outside start() ever holds a reference to this engine, so
        # if start() strands the claim no caller can free it.
        engine, _writer, _clock, claim = make_engine(tmp_path)
        real_thread = threading.Thread
        monkeypatch.setattr(threading, 'Thread', _NoThreadStarts)
        with pytest.raises(RuntimeError):
            engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        monkeypatch.setattr(threading, 'Thread', real_thread)
        assert claim.owner is None

    def test_start_failure_leaves_engine_startable_again(self, tmp_path, monkeypatch):
        engine, _writer, clock, claim = make_engine(tmp_path)
        real_thread = threading.Thread
        monkeypatch.setattr(threading, 'Thread', _NoThreadStarts)
        with pytest.raises(RuntimeError):
            engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        monkeypatch.setattr(threading, 'Thread', real_thread)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=2)
        assert engine.wait_for_drain(timeout=5)
        assert claim.owner is None

    def test_failed_start_publishes_no_manifest(self, tmp_path):
        # The literal, not the module constant: this test must import and
        # run against a tree where the constant does not exist yet.
        engine, _writer, _clock, claim = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        engine.stop('start_failed')
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().manifest_path is None
        assert not list(pathlib.Path(tmp_path).glob('*.json'))
        assert claim.owner is None

    def test_lane_boundary_logs_its_own_abnormal_exit(self, tmp_path, caplog, monkeypatch):
        # An exception escaping a thread target reaches
        # threading.excepthook, not this module's logger, so without the
        # lane's own boundary the failure it exists to report leaves no
        # trace in the app log -- a support bundle would show a recording
        # that simply stopped. A queue read that raises is outside the
        # per-frame try, so it carries an exception out of the lane body.
        # A stand-in by design: a queue that fails to read is a fault no
        # simulated part produces, and each start makes its own queue.
        monkeypatch.setattr(video_recording_module, '_Take', _TakeWithAQueueThatFailsToRead)
        engine, _writer, _clock, claim = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=2), None))
        assert engine.wait_for_drain(timeout=5)

        logged = [
            r.getMessage()
            for r in records_of(caplog, video_recording_module)
            if r.levelno == logging.CRITICAL
        ]
        assert any('writer lane exited abnormally' in message for message in logged)
        assert claim.owner is None

    def test_lane_death_frees_the_claim_for_the_next_recording(self, tmp_path):
        # The interaction, not just the flag: a lane death that strands the
        # claim is only visible when something later tries to take it.
        claim = ActivityClaim()
        writer = WriterStub(tmp_path, die_on_frame=2)
        engine, _writer, clock, _ = make_engine(tmp_path, writer=writer, claim=claim)
        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=2), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=1)
        assert engine.wait_for_drain(timeout=5)

        # A fresh engine, as production builds one per recording.
        next_engine, _w, next_clock, _c = make_engine(tmp_path, claim=claim)
        next_engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        feed_uniform(next_engine, next_clock, FrameFeed(), delivery_fps=20, duration_s=2)
        assert next_engine.wait_for_drain(timeout=5)
        assert next_engine.result().frames_written > 0
        assert claim.owner is None


class _LaneWriter:
    """A writer edge whose frames can be held, released or made to kill their lane.

    ``hold[n]`` keeps frame n's write waiting until its event is set;
    ``die_on`` raises SystemExit for that frame, the fatal escape. Each
    write records its number and the taking it ran under, in the order the
    writes finished.
    """

    def __init__(self, out_dir, *, hold=None, die_on=None):
        self.out_dir = pathlib.Path(out_dir)
        self.hold = dict(hold or {})
        self.die_on = die_on
        self.entered = {n: threading.Event() for n in self.hold}
        self.finished = []
        self.takings = []
        self._landed = {}
        self._lock = threading.Lock()

    def landed(self, frame_number) -> threading.Event:
        """Set once frame *frame_number*'s write has finished."""
        with self._lock:
            return self._landed.setdefault(frame_number, threading.Event())

    def __call__(self, image, timestamp_s, frame_number, config, chunks=None, fact=None):
        if frame_number in self.hold:
            self.entered[frame_number].set()
            assert self.hold[frame_number].wait(5), f'frame {frame_number} was held past 5 s'
        if frame_number == self.die_on:
            raise SystemExit('writer lane death (scripted)')
        path = self.out_dir / f'frame_{frame_number:06d}.tiff'
        path.write_bytes(image.tobytes())
        with self._lock:
            self.finished.append(frame_number)
            self.takings.append(current_taking())
        self.landed(frame_number).set()
        return path


class _ThirdLaneFailsToStart(threading.Thread):
    """Thread exhaustion at the third lane: the first two start, the third raises.

    A stand-in by design: thread exhaustion is the host's, not the simulator's.
    """

    started_lanes: typing.ClassVar[list] = []

    def start(self):
        if len(_ThirdLaneFailsToStart.started_lanes) == 2:
            raise RuntimeError('scripted thread exhaustion at the third lane')
        _ThirdLaneFailsToStart.started_lanes.append(self)
        super().start()


class TestFourWriterLanes:
    """A recording drains on the writer lanes its caller names, and stays one recording."""

    def test_every_frame_is_written_once_under_its_number_whatever_order_writes_finish(
        self, tmp_path
    ):
        # Frame 0's write waits until frame 3's has landed: only lanes
        # writing at once finish frame 3 first, so the order proves four.
        frame_3_landed = threading.Event()
        writer = _LaneWriter(tmp_path, hold={0: frame_3_landed})
        original = writer.__call__

        def _signal_frame_3(image, timestamp_s, frame_number, *rest):
            path = original(image, timestamp_s, frame_number, *rest)
            if frame_number == 3:
                frame_3_landed.set()
            return path

        engine, _writer, clock, claim = make_engine(tmp_path, writer=_signal_frame_3, writers=4)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)

        assert engine.wait_for_drain(timeout=10)
        result = engine.result()
        assert writer.finished.index(3) < writer.finished.index(0), writer.finished
        assert sorted(writer.finished) == list(range(result.frames_selected))
        assert result.frames_written == result.frames_selected == 10
        manifest = json.loads((tmp_path / video_recording_module.MANIFEST_FILENAME).read_text())
        assert [row['i'] for row in manifest['frame_index']] == list(range(10))
        assert claim.owner is None

    def test_the_drain_ends_only_after_the_last_lane(self, tmp_path):
        release = threading.Event()
        writer = _LaneWriter(tmp_path, hold={4: release})
        engine, _writer, clock, claim = make_engine(tmp_path, writer=writer, writers=4)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        assert writer.entered[4].wait(5)

        assert not engine.wait_for_drain(timeout=0.2), 'drained with a lane still writing'
        assert claim.owner is not None
        release.set()
        assert engine.wait_for_drain(timeout=10)
        assert engine.result().frames_written == 10
        assert claim.owner is None

    def test_each_lane_writes_under_the_recordings_taking(self, tmp_path):
        writer = _LaneWriter(tmp_path)
        engine, _writer, clock, claim = make_engine(tmp_path, writer=writer, writers=4)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=1), None))
        taking = claim.holder
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        assert engine.wait_for_drain(timeout=10)
        assert writer.takings and all(t is not None for t in writer.takings)
        assert {t.claim for t in writer.takings} == {claim}
        assert taking is not None

    def test_one_lanes_death_ends_the_recording_aborted_once_every_lane_has_exited(self, tmp_path):
        release = threading.Event()
        writer = _LaneWriter(tmp_path, hold={0: release}, die_on=1)
        engine, _writer, clock, claim = make_engine(tmp_path, writer=writer, writers=4)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=2), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        assert writer.entered[0].wait(5)

        # Frame 1's lane has died; frame 0's is still inside its write.
        assert not engine.wait_for_drain(timeout=0.5), 'finalized before every lane exited'
        release.set()
        assert engine.wait_for_drain(timeout=10)
        result = engine.result()
        assert result.aborted
        assert result.end_reason == 'aborted'
        assert 0 in writer.finished
        assert claim.owner is None

    def test_a_discard_with_every_lane_wedged_frees_the_claim_and_drops_the_queue(self, tmp_path):
        # Every lane is held inside a write; the frames behind them wait.
        release = threading.Event()
        writer = _LaneWriter(tmp_path, hold=dict.fromkeys(range(4), release))
        engine, _writer, clock, claim = make_engine(tmp_path, writer=writer, writers=4)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=2), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        assert all(writer.entered[n].wait(5) for n in range(4))

        engine.discard_pending()
        assert claim.owner is None
        assert engine.result().end_reason == 'discarded'
        release.set()
        assert all(writer.landed(n).wait(5) for n in range(4))
        # Each lane exits after its write in flight; no queued frame is written.
        assert sorted(writer.finished) == [0, 1, 2, 3]

    def test_a_late_write_after_a_discard_counts_in_its_own_recording(self, tmp_path):
        release = threading.Event()
        writer = _LaneWriter(tmp_path, hold={0: release})
        engine, _writer, clock, claim = make_engine(tmp_path, writer=writer, writers=4)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        assert writer.entered[0].wait(5)
        engine.discard_pending()

        # The next recording on the same engine, with the old lane still in
        # its write; the held write lands once the next one is under way.
        writer.hold.clear()
        next_dir = tmp_path / 'next'
        next_dir.mkdir()
        engine.start(lambda: (make_config(next_dir, fps=10, duration_s=1), None))
        release.set()
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        assert engine.wait_for_drain(timeout=10)

        result = engine.result()
        assert result.frames_selected == 10
        assert result.frames_written == 10, 'the late write was counted into the next recording'
        assert claim.owner is None

    def test_a_start_that_fails_at_the_third_lane_leaves_no_lane_and_restarts_clean(
        self, tmp_path, monkeypatch
    ):
        engine, _writer, clock, claim = make_engine(tmp_path, writers=4)
        _ThirdLaneFailsToStart.started_lanes = []
        monkeypatch.setattr(threading, 'Thread', _ThirdLaneFailsToStart)
        with pytest.raises(RuntimeError, match='third lane'):
            engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        monkeypatch.undo()

        assert len(_ThirdLaneFailsToStart.started_lanes) == 2
        assert not any(lane.is_alive() for lane in _ThirdLaneFailsToStart.started_lanes)
        assert claim.owner is None
        assert not engine.is_recording and not engine.is_draining

        engine.start(lambda: (make_config(tmp_path, fps=5, duration_s=1), None))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=2)
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.frames_written == result.frames_selected == 5
        assert claim.owner is None


_MEMORY_SHORT = video_recording_module.MemoryHeadroom(
    physical_available_bytes=256 << 20, commit_headroom_bytes=None
)
_COMMIT_SHORT = video_recording_module.MemoryHeadroom(
    physical_available_bytes=64 << 30, commit_headroom_bytes=256 << 20
)


def _read_as(monkeypatch, headroom):
    # A stand-in by design: the host's memory is not the simulator's.
    monkeypatch.setattr(video_recording_module, 'read_memory_headroom', lambda: headroom)


def _deliver(engine, clock, n, *, fps=10.0, first_id=None):
    """Deliver n frames at fps; each states a FrameID from first_id, or none."""
    for k in range(n):
        clock.advance(1.0 / fps)
        chunks = None if first_id is None else {'FrameID': first_id + k}
        engine.ingest_frame(FrameFeed().frame(clock())[0], clock(), chunks, fact=None)


class TestABacklogWithMemoryShortSkipsFrames:
    """Frames waiting beyond the writers with the PC's memory short are skipped, counted.

    The recording goes on: a skipped frame spends its slot, gets no number,
    and frames are kept again once the writers have shrunk the backlog.
    """

    @pytest.mark.parametrize(
        'short', [_MEMORY_SHORT, _COMMIT_SHORT], ids=['physical-short', 'commit-short']
    )
    def test_frames_are_skipped_until_the_backlog_shrinks_and_a_skipped_last_slot_ends_it(
        self, tmp_path, monkeypatch, short
    ):
        _read_as(monkeypatch, short)
        release_0, release_2 = threading.Event(), threading.Event()
        writer = _LaneWriter(tmp_path, hold={0: release_0, 2: release_2})
        engine, _writer, clock, claim = make_engine(tmp_path, writer=writer, writers=1)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=2), None))  # 20 slots

        # Frame 0 is in its write and frame 1 waits: the next eight wait
        # beyond the one writer with memory short, and are skipped.
        _deliver(engine, clock, 10)
        assert writer.entered[0].wait(5)
        release_0.set()
        assert writer.landed(1).wait(5)
        # The backlog has shrunk: frames are kept again. Once the lane is in
        # frame 2's write, frame 1 is wholly done, so frame 3 waits alone
        # and the rest are skipped; the twentieth slot, skipped, ends it.
        _deliver(engine, clock, 1)
        assert writer.entered[2].wait(5)
        _deliver(engine, clock, 9)
        assert not engine.is_recording
        release_2.set()
        assert engine.wait_for_drain(timeout=5)

        result = engine.result()
        assert (result.frames_selected, result.frames_skipped) == (4, 16)
        assert result.frames_written == 4
        assert sorted(writer.finished) == [0, 1, 2, 3], 'kept frames are numbered without gaps'
        assert result.end_reason == 'frame_budget_filled'
        assert not result.kept_every_frame(0)
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest['frames_skipped'] == 16
        assert claim.owner is None

    def test_a_writer_that_keeps_up_skips_none_with_memory_short(self, tmp_path, monkeypatch):
        _read_as(monkeypatch, _MEMORY_SHORT)
        writer = _LaneWriter(tmp_path)
        engine, _writer, clock, _ = make_engine(tmp_path, writer=writer, writers=4)
        engine.start(lambda: (make_config(tmp_path, fps=10, duration_s=1), None))
        for n in range(10):
            _deliver(engine, clock, 1)
            assert writer.landed(n).wait(5)
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert (result.frames_written, result.frames_skipped) == (10, 0)


class TestTheCamerasOwnLossesAreCounted:
    """A gap in the delivered frames' FrameID is frames the camera sent that never arrived."""

    def _record(self, tmp_path, feed, *, fps=None):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(lambda: (make_config(tmp_path, fps=fps, duration_s=60), None))
        feed(engine, clock)
        engine.stop('stopped')
        assert engine.wait_for_drain(timeout=5)
        return engine.result()

    def test_a_gap_of_n_counts_n_and_the_recording_did_not_keep_every_frame(self, tmp_path):
        def feed(engine, clock):
            _deliver(engine, clock, 3, first_id=100)
            _deliver(engine, clock, 2, first_id=110)  # 103-109 never arrived

        result = self._record(tmp_path, feed)
        assert result.frames_lost_at_camera == 7
        assert json.loads(result.manifest_path.read_text())['frames_lost_at_camera'] == 7
        assert '7 lost at the camera' in result.summary()
        assert not result.kept_every_frame(0)
        late = dataclasses.replace(result, save_wait_s=30.0)
        assert late.save_wait_to_tell(0) is None, (
            'a recording with gaps was told every frame was kept'
        )

    def test_frames_the_cadence_passes_over_are_no_loss(self, tmp_path):
        result = self._record(
            tmp_path, lambda engine, clock: _deliver(engine, clock, 20, first_id=0), fps=5
        )
        assert result.frames_selected < 20
        assert result.frames_lost_at_camera == 0
        assert dataclasses.replace(result, save_wait_s=30.0).save_wait_to_tell(0) == 30.0

    def test_an_unstamped_frame_reached_the_recording_and_is_no_gap(self, tmp_path):
        def feed(engine, clock):
            _deliver(engine, clock, 2, first_id=0)
            engine.note_unstamped({'FrameID': 2})
            _deliver(engine, clock, 2, first_id=3)

        result = self._record(tmp_path, feed)
        assert (result.frames_unstamped, result.frames_lost_at_camera) == (1, 0)

    def test_a_camera_that_numbers_no_frames_states_its_losses_not_known(self, tmp_path):
        result = self._record(tmp_path, lambda engine, clock: _deliver(engine, clock, 5))
        assert result.frames_lost_at_camera is None
        assert json.loads(result.manifest_path.read_text())['frames_lost_at_camera'] is None
        assert 'not known lost at the camera' in result.summary()

    def test_a_numbering_that_restarts_is_logged_and_counts_no_loss(self, tmp_path, caplog):
        def feed(engine, clock):
            _deliver(engine, clock, 3, first_id=500)
            _deliver(engine, clock, 3, first_id=0)

        with caplog.at_level(logging.WARNING):
            result = self._record(tmp_path, feed)
        assert result.frames_lost_at_camera == 0
        assert any(
            'numbering restarted' in r.getMessage()
            for r in records_of(caplog, video_recording_module)
        )


def test_the_memory_reading_states_what_this_platform_has():
    headroom = _real_read_memory_headroom()
    assert headroom.physical_available_bytes > 0
    if sys.platform == 'win32':
        assert headroom.commit_headroom_bytes > 0
    else:
        assert headroom.commit_headroom_bytes is None
