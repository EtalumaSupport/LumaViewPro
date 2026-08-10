# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera that goes inactive mid-video-step must not report success.

The wait loop's camera-inactive break previously fell through to the
COMPLETED default, and the caller reset the consecutive-failure counter
on COMPLETED -- so a truncated step counted as a success. CAMERA_LOST
names the truncation; the caller strikes without resetting.
"""

import threading

from unittest.mock import MagicMock

import modules.protocol_recording as protocol_recording


def _make_recorder(tmp_path, clock, camera_active=True):
    scope = MagicMock()
    scope.imaging.frames_until_valid.return_value = 0
    scope.imaging.camera_active = camera_active
    scope.imaging.camera_identity = {
        'model': 'sim',
        'serial': '0',
        'timestamp_tick_frequency_hz': None,
    }
    scope.imaging.camera_frame_size = {'width': 8, 'height': 8}
    return protocol_recording.ProtocolVideoStep(
        scope=scope,
        step={
            'Video Config': {'fps': 5, 'duration': 10},
            'Color': 'BF',
            'False_Color': False,
            'Auto_Gain': False,
            'Exposure': 10.0,
        },
        save_folder=tmp_path,
        name='clip',
        video_as_frames=False,
        capture_config=MagicMock(capture_depth=8, save_encoding='8bit'),
        timestamp_overlay=True,
        global_max_fps=0,
        autogain_settings={},
        callbacks={},
        aborted_event=threading.Event(),
        is_run_in_progress=lambda: True,
        abort_run_fatal=MagicMock(),
        abort_run_on_writer_death=MagicMock(),
        record_step_row=MagicMock(),
        record_dropped_capture=MagicMock(),
        clock=lambda: clock['t'],
    )


class TestWaitForRecordingOutcome:
    def test_camera_inactive_returns_camera_lost(self, tmp_path):
        clock = {'t': 1000.0}
        recorder = _make_recorder(tmp_path, clock, camera_active=False)
        engine = MagicMock(is_recording=True)
        outcome, reason = recorder._wait_for_recording(engine, 10.0, stall_threshold=5.0)
        assert outcome == protocol_recording.CAMERA_LOST
        assert reason == 'camera_disconnected'

    def test_stop_request_beats_camera_loss(self, tmp_path):
        clock = {'t': 1000.0}
        recorder = _make_recorder(tmp_path, clock, camera_active=False)
        recorder._aborted.set()
        engine = MagicMock(is_recording=True)
        outcome, _ = recorder._wait_for_recording(engine, 10.0, stall_threshold=5.0)
        assert outcome == protocol_recording.CANCELLED

    def test_duration_elapsed_still_completes(self, tmp_path):
        # Each clock read advances 5.5 s: start at 1005.5, then 1011.0
        # (elapsed 5.5), then 1016.5 (elapsed 11 > duration) -> the wall
        # cap closes the loop with the camera alive the whole time.
        clock = {'t': 1000.0}

        def _advancing():
            clock['t'] += 5.5
            return clock['t']

        recorder = _make_recorder(tmp_path, clock, camera_active=True)
        recorder._clock = _advancing
        engine = MagicMock(is_recording=True)
        outcome, _ = recorder._wait_for_recording(engine, 10.0, stall_threshold=1000.0)
        assert outcome == protocol_recording.COMPLETED


class TestZeroFrameOutcomeMapping:
    """run_blocking's zero-frame branch must not swallow an early exit.

    A user Stop (or run abort) that lands before any frame arrived
    previously returned NO_FRAMES, so the caller recorded a capture
    failure and a strike toward the 3-strike run abort -- for a step
    the USER ended. Only a recording that ran its course with nothing
    delivered is the silent-camera failure.
    """

    def _run_with_outcome(self, tmp_path, outcome):
        from unittest.mock import patch

        clock = {'t': 1000.0}
        recorder = _make_recorder(tmp_path, clock, camera_active=True)
        recorder._video_as_frames = True
        with (
            patch.object(protocol_recording, 'VideoRecordingEngine', MagicMock()),
            patch.object(recorder, '_prologue', return_value=None),
            patch.object(recorder, '_wait_for_recording', return_value=(outcome, 'run_stop')),
        ):
            return recorder.run_blocking()

    def test_user_stop_with_zero_frames_stays_cancelled(self, tmp_path):
        result = self._run_with_outcome(tmp_path, protocol_recording.CANCELLED)
        assert result == protocol_recording.CANCELLED

    def test_run_abort_with_zero_frames_stays_aborted(self, tmp_path):
        result = self._run_with_outcome(tmp_path, protocol_recording.ABORTED)
        assert result == protocol_recording.ABORTED

    def test_full_course_with_zero_frames_is_no_frames(self, tmp_path):
        # Preservation guard: passes before and after the mapping fix.
        result = self._run_with_outcome(tmp_path, protocol_recording.COMPLETED)
        assert result == protocol_recording.NO_FRAMES

    def test_zero_frame_camera_loss_keeps_the_failure_row_shape(self, tmp_path):
        # A camera lost before delivering anything IS "delivered
        # nothing": NO_FRAMES keeps the caller's dropped-capture row and
        # the strike (the nonzero-frame camera-loss path records its row
        # via the finish thread instead).
        result = self._run_with_outcome(tmp_path, protocol_recording.CAMERA_LOST)
        assert result == protocol_recording.NO_FRAMES


class TestPrologueFeedDeath:
    """The pre-recording drain terminates on a dead feed -- and only then.

    Dead means zero frame ARRIVALS for the stall threshold. Stop is
    checked first each iteration, so a run stop always wins; and frames
    arriving with validity pinned (a stuck motion-settle) must never be
    blamed on the camera.
    """

    def _prologue_result(self, tmp_path, *, stop_at_s, frame_arrives):
        clock = {'t': 1000.0}
        recorder = _make_recorder(tmp_path, clock, camera_active=True)
        start = clock['t']
        recorder._is_run_in_progress = lambda: clock['t'] - start < stop_at_s

        def _grab(*args, **kwargs):
            clock['t'] += 1.0
            return object() if frame_arrives else None

        recorder._scope.imaging.frames_until_valid.return_value = 1
        recorder._scope.imaging.get_image.side_effect = _grab
        step = {'Exposure': 10.0, 'Auto_Gain': False}
        return recorder._prologue(step)

    def test_dead_feed_returns_no_frames_before_the_stop(self, tmp_path):
        # Threshold 5 s (10 ms exposure -> floor); stop would land at 8 s.
        # The death verdict must fire first.
        result = self._prologue_result(tmp_path, stop_at_s=8.0, frame_arrives=False)
        assert result == protocol_recording.NO_FRAMES

    def test_stop_before_the_threshold_wins(self, tmp_path):
        result = self._prologue_result(tmp_path, stop_at_s=2.0, frame_arrives=False)
        assert result == protocol_recording.CANCELLED

    def test_arriving_frames_suppress_the_death_verdict(self, tmp_path):
        # Validity pinned while frames arrive is a settle fault, not a
        # camera fault: the drain must never return NO_FRAMES for it.
        # (Bounded here by the stop; the pinned-validity hang itself is a
        # separately recorded finding.)
        result = self._prologue_result(tmp_path, stop_at_s=60.0, frame_arrives=True)
        assert result == protocol_recording.CANCELLED


class TestWaitLoopFeedStall:
    """A feed that dies WITHOUT a disconnect event must still end the step.

    camera_active stays True (the latch only flips on events); the stall
    watch on the ingest counter is what notices delivery stopped.
    """

    def _advancing_recorder(self, tmp_path, step_s=0.5):
        clock = {'t': 1000.0}

        def _advancing():
            clock['t'] += step_s
            return clock['t']

        recorder = _make_recorder(tmp_path, clock, camera_active=True)
        recorder._clock = _advancing
        return recorder

    def test_frozen_ingest_counter_ends_as_stalled(self, tmp_path):
        # duration 60 s >> threshold 5 s so the wall cap cannot mask the
        # stall verdict; _frames_seen never advances.
        recorder = self._advancing_recorder(tmp_path)
        engine = MagicMock(is_recording=True)
        outcome, reason = recorder._wait_for_recording(engine, 60.0, stall_threshold=5.0)
        assert outcome == protocol_recording.CAMERA_LOST
        assert reason == 'camera_stalled'

    def test_advancing_ingest_counter_never_stalls(self, tmp_path):
        # Preservation guard: frames arriving (counter advancing every
        # tick) run the step to the wall cap exactly as before.
        recorder = self._advancing_recorder(tmp_path)
        engine = MagicMock(is_recording=True)

        counter = {'n': 0}
        original_clock = recorder._clock

        def _clock_and_frame():
            counter['n'] += 1
            recorder._frames_seen = counter['n']
            return original_clock()

        recorder._clock = _clock_and_frame
        outcome, reason = recorder._wait_for_recording(engine, 20.0, stall_threshold=5.0)
        assert outcome == protocol_recording.COMPLETED
        assert reason == 'duration_elapsed'

    def test_stop_beats_the_stall_verdict(self, tmp_path):
        # Stop is checked first: even with the watch past threshold, a
        # stop requested in the same tick returns CANCELLED.
        recorder = self._advancing_recorder(tmp_path)
        recorder._aborted.set()
        engine = MagicMock(is_recording=True)
        outcome, _ = recorder._wait_for_recording(engine, 60.0, stall_threshold=5.0)
        assert outcome == protocol_recording.CANCELLED
