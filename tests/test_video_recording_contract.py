# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Contract tests for the video recording engine (modules/video_recording.py).

These are the engine's specification, written against the signatures-only
skeleton: they run RED until the engine implementation lands, and the
implementation is done when they are green. Each class pins one contract;
none of them sleeps against the wall clock -- the engine accepts an
injected time source precisely so these stay deterministic.
"""

import json
import pathlib
import threading

import pytest

from modules.exceptions import ProtocolRunRefusedError, RecordingRefusedError
from modules.video_recording import RecordingConfig, VideoRecordingEngine
from tests.video_engine_harness import (
    ClaimStub,
    FakeClock,
    FrameFeed,
    NotifyRecorder,
    WriterStub,
)

# Executable-spec marker: these tests define behavior the engine build
# delivers; they must FAIL (not error) until then, and strict=True turns
# each one that starts passing into an XPASS error so the marker cannot
# outlive the implementation.
ENGINE_XFAIL = pytest.mark.xfail(
    reason='executable spec: the video engine implementation is not built yet',
    strict=True,
)

# Every recording manifest must carry at least these keys; downstream
# consumers (support bundles, char tooling, the end-of-run report) key
# off them.
REQUIRED_MANIFEST_KEYS = {
    'frames_written',
    'write_failures',
    'short_delivery',
    'timestamp_grade',
    'measured_fps',
    'measured_duration_s',
    'frame_index',
}


def make_config(tmp_path, fps=5.0, duration_s=2.0, width=8, height=6, bit_depth=8):
    return RecordingConfig(
        fps=fps,
        duration_s=duration_s,
        width=width,
        height=height,
        bit_depth=bit_depth,
        output_dir=pathlib.Path(tmp_path),
        filename_template='frame_{n:06d}.tiff',
    )


def make_engine(tmp_path, *, clock=None, writer=None, claim=None, notify=None):
    clock = clock or FakeClock()
    writer = writer if writer is not None else WriterStub(tmp_path)
    claim = claim or ClaimStub()
    engine = VideoRecordingEngine(write_frame=writer, claim=claim, clock=clock, notify=notify)
    return engine, writer, clock, claim


def feed_uniform(engine, clock, feed, *, delivery_fps, duration_s, chunks=True):
    """Deliver frames at delivery_fps for duration_s, clock in lockstep."""
    step = 1.0 / delivery_fps
    n = int(duration_s * delivery_fps)
    for _ in range(n):
        clock.advance(step)
        image, ts, chunk = feed.frame(clock(), with_camera_chunks=chunks)
        engine.ingest_frame(image, ts, chunk)


class TestBudgetContract:
    def test_budget_is_exact_ceil(self, tmp_path):
        assert make_config(tmp_path, fps=5, duration_s=2).frame_budget == 10
        assert make_config(tmp_path, fps=0.5, duration_s=3).frame_budget == 2

    def test_no_truncation_at_any_duration(self, tmp_path):
        # An hour at 40 fps is 144,000 frames; no cap anywhere.
        assert make_config(tmp_path, fps=40, duration_s=3600).frame_budget == 144000

    @ENGINE_XFAIL
    def test_budget_full_closes_selection(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        config = make_config(tmp_path, fps=5, duration_s=1)  # budget 5
        engine.start(config)
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=50, duration_s=1)
        engine.stop()
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().frames_selected == 5

    @ENGINE_XFAIL
    def test_duration_elapsed_closes_selection(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(make_config(tmp_path, fps=5, duration_s=1))
        feed = FrameFeed()
        # Deliver well past the configured duration; selection must close
        # itself at the duration boundary without an explicit stop().
        feed_uniform(engine, clock, feed, delivery_fps=20, duration_s=3)
        assert not engine.is_recording
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().frames_selected == 5


@ENGINE_XFAIL
class TestRateContract:
    def test_configured_rate_honored_under_fast_delivery(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(make_config(tmp_path, fps=5, duration_s=2))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=40, duration_s=2)
        engine.stop()
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().frames_selected == 10

    def test_over_configuration_is_safe(self, tmp_path):
        # Configured faster than the camera delivers: every delivered
        # frame is kept and the shortfall is reported, never an error.
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(make_config(tmp_path, fps=50, duration_s=2))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=2)
        engine.stop()
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.frames_selected == 20
        assert result.short_delivery is True


@ENGINE_XFAIL
class TestEnqueueIsUnconditional:
    def test_lagging_writer_never_causes_capture_drop(self, tmp_path):
        writer = WriterStub(tmp_path, blocked=True)
        engine, _, clock, _ = make_engine(tmp_path, writer=writer)
        engine.start(make_config(tmp_path, fps=10, duration_s=2))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=2)
        engine.stop()
        # Writer wrote nothing, yet every selected frame is queued.
        assert engine.pending_writes == 20
        writer.unblock()
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.frames_selected == 20
        assert result.frames_written == 20
        assert result.write_failures == 0


@ENGINE_XFAIL
class TestDrainContinuesAfterCapture:
    def test_backlog_drains_after_stop(self, tmp_path):
        writer = WriterStub(tmp_path, blocked=True)
        engine, _, clock, _ = make_engine(tmp_path, writer=writer)
        engine.start(make_config(tmp_path, fps=10, duration_s=1))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop()
        assert not engine.is_recording
        assert engine.is_draining
        writer.unblock()
        assert engine.wait_for_drain(timeout=5)
        assert not engine.is_draining
        assert len(writer.written) == 10
        # Frame numbers are contiguous ordinals from enqueue order.
        assert [n for n, _, _ in writer.written] == list(range(10))


@ENGINE_XFAIL
class TestStopPromptness:
    def test_stop_closes_selection_within_one_decision(self, tmp_path):
        engine, writer, clock, _ = make_engine(tmp_path)
        engine.start(make_config(tmp_path, fps=10, duration_s=10))
        feed = FrameFeed()
        feed_uniform(engine, clock, feed, delivery_fps=10, duration_s=1)
        engine.stop()
        assert not engine.is_recording
        selected_at_stop = engine.pending_writes + len(writer.written)
        # Frames delivered after stop are never selected.
        feed_uniform(engine, clock, feed, delivery_fps=10, duration_s=1)
        assert engine.pending_writes + len(writer.written) == selected_at_stop
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().frames_selected == selected_at_stop


@ENGINE_XFAIL
class TestLossIsNeverSilent:
    def test_per_frame_write_failure_costs_that_frame_only(self, tmp_path):
        writer = WriterStub(tmp_path, fail_frames={3})
        notify = NotifyRecorder()
        engine, _, clock, _ = make_engine(tmp_path, writer=writer, notify=notify)
        engine.start(make_config(tmp_path, fps=10, duration_s=1))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop()
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.frames_selected == 10
        assert result.frames_written == 9
        assert result.write_failures == 1
        assert result.aborted is False
        # Non-fatal: no critical popup fired mid-run.
        assert 'critical' not in notify.severities()

    def test_discard_pending_is_loud(self, tmp_path):
        writer = WriterStub(tmp_path, blocked=True)
        engine, _, clock, _ = make_engine(tmp_path, writer=writer)
        engine.start(make_config(tmp_path, fps=10, duration_s=1))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop()
        engine.discard_pending()
        result = engine.result()
        assert result.short_delivery is True
        assert result.frames_written < result.frames_selected


@ENGINE_XFAIL
class TestFatalityClassification:
    def test_writer_lane_death_aborts_the_recording(self, tmp_path):
        writer = WriterStub(tmp_path, die_on_frame=2)
        notify = NotifyRecorder()
        engine, _, clock, _ = make_engine(tmp_path, writer=writer, notify=notify)
        engine.start(make_config(tmp_path, fps=10, duration_s=1))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop()
        engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.aborted is True
        assert result.abort_reason != ''
        assert 'critical' in notify.severities()

    def test_short_delivery_is_not_fatal(self, tmp_path):
        notify = NotifyRecorder()
        engine, _writer, clock, _ = make_engine(tmp_path, notify=notify)
        engine.start(make_config(tmp_path, fps=50, duration_s=1))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop()
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().aborted is False
        assert 'critical' not in notify.severities()


@ENGINE_XFAIL
class TestMeasuredTruth:
    def test_result_reports_measured_not_configured(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(make_config(tmp_path, fps=10, duration_s=2))
        # Camera actually delivers at 7 fps: measured truth must say so.
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=7, duration_s=2)
        engine.stop()
        assert engine.wait_for_drain(timeout=5)
        result = engine.result()
        assert result.configured_fps == 10
        assert 6.0 < result.measured_fps < 8.0
        assert 1.5 < result.measured_duration_s < 2.1

    def test_camera_chunks_grade_timestamps_camera(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(make_config(tmp_path, fps=5, duration_s=1))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1, chunks=True)
        engine.stop()
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().timestamp_grade == 'camera'

    def test_missing_chunks_grade_timestamps_host(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(make_config(tmp_path, fps=5, duration_s=1))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1, chunks=False)
        engine.stop()
        assert engine.wait_for_drain(timeout=5)
        assert engine.result().timestamp_grade == 'host'


@ENGINE_XFAIL
class TestManifestTruth:
    def test_manifest_written_with_required_schema(self, tmp_path):
        engine, _writer, clock, _ = make_engine(tmp_path)
        engine.start(make_config(tmp_path, fps=5, duration_s=2))
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=20, duration_s=2)
        engine.stop()
        assert engine.wait_for_drain(timeout=5)
        manifest_path = engine.result().manifest_path
        assert manifest_path is not None and manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        missing = REQUIRED_MANIFEST_KEYS - set(manifest)
        assert not missing, f'manifest missing keys: {sorted(missing)}'


@ENGINE_XFAIL
class TestExclusivity:
    def test_engine_refuses_second_capture(self, tmp_path):
        engine, _writer, _clock, _ = make_engine(tmp_path)
        engine.start(make_config(tmp_path, fps=5, duration_s=10))
        with pytest.raises(RecordingRefusedError):
            engine.start(make_config(tmp_path, fps=5, duration_s=10))

    def test_engine_refuses_when_claim_held_by_protocol(self, tmp_path):
        claim = ClaimStub()
        assert claim.try_claim('protocol')
        engine, _writer, _clock, _ = make_engine(tmp_path, claim=claim)
        with pytest.raises(RecordingRefusedError):
            engine.start(make_config(tmp_path, fps=5, duration_s=1))

    def test_concurrent_starts_exactly_one_wins(self, tmp_path):
        claim = ClaimStub()
        outcomes = []
        barrier = threading.Barrier(2)

        def _try_start():
            engine, _, _, _ = make_engine(tmp_path, claim=claim)
            config = make_config(tmp_path, fps=5, duration_s=10)
            barrier.wait(timeout=5)
            try:
                engine.start(config)
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
        claim = ClaimStub()
        engine, _writer, clock, _ = make_engine(tmp_path, claim=claim)
        engine.start(make_config(tmp_path, fps=5, duration_s=1))
        assert claim.owner == 'recording'
        feed_uniform(engine, clock, FrameFeed(), delivery_fps=10, duration_s=1)
        engine.stop()
        assert engine.wait_for_drain(timeout=5)
        assert claim.owner is None


@ENGINE_XFAIL
class TestSessionActivityClaim:
    """The session-tier half of the exclusivity contract.

    ScopeSession owns ONE compare-and-claim primitive (owner 'protocol'
    XOR 'recording') that both the sequenced-run gate and the engine
    acquire. These pin the session surface the engine's claim handle
    injects from.
    """

    @pytest.fixture
    def headless_session(self, tmp_path):
        from modules.scope_session import ScopeSession

        settings = {
            'BF': {'autofocus': False},
            'PC': {'autofocus': False},
            'DF': {'autofocus': False},
            'Red': {'autofocus': False},
            'Green': {'autofocus': False},
            'Blue': {'autofocus': False},
            'Lumi': {'autofocus': False},
            'stage_offset': {'x': 0.0, 'y': 0.0},
            'live_folder': str(tmp_path),
            'protocol': {
                'autogain': {
                    'target_brightness': 0.3,
                    'max_duration_seconds': 1.0,
                    'min_gain_db': 0.0,
                    'max_gain_db': 20.0,
                },
            },
        }
        session = ScopeSession.create_headless(settings=settings)
        yield session
        session.shutdown()

    def test_session_owns_one_activity_claim(self, headless_session):
        claim = headless_session.activity_claim
        assert claim.try_claim('recording')
        assert not claim.try_claim('protocol')
        claim.release('recording')
        assert claim.try_claim('protocol')
        claim.release('protocol')

    def test_two_thread_claim_atomicity(self, headless_session):
        claim = headless_session.activity_claim
        wins = []
        barrier = threading.Barrier(2)

        def _race(owner):
            barrier.wait(timeout=5)
            if claim.try_claim(owner):
                wins.append(owner)

        threads = [
            threading.Thread(target=_race, args=('protocol',)),
            threading.Thread(target=_race, args=('recording',)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(wins) == 1

    def test_protocol_start_refused_while_recording_holds_claim(self, headless_session, tmp_path):
        from tests.test_run_refusal_contract import (
            _make_single_step_protocol,
        )

        headless_session.start_executors()
        assert headless_session.activity_claim.try_claim('recording')
        runner = headless_session.create_protocol_runner()
        with pytest.raises(ProtocolRunRefusedError):
            runner.run_single_scan(
                protocol=_make_single_step_protocol(),
                sequence_name='refused_while_recording',
                parent_dir=str(tmp_path),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
            )
        headless_session.activity_claim.release('recording')
