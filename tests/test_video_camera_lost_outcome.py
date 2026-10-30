# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera that goes inactive mid-video-step must not report success.

The wait loop's camera-inactive break previously fell through to the
COMPLETED default, and the caller reset the consecutive-failure counter
on COMPLETED -- so a truncated step counted as a success. CAMERA_LOST
names the truncation; the caller strikes without resetting.
"""

from unittest.mock import MagicMock, patch

import pytest

import modules.protocol_recording as protocol_recording
from tests.stock_channels import LED0
from tests.video_step_drives import video_step_on


def _make_recorder(scope, tmp_path, clock):
    return video_step_on(
        scope,
        tmp_path,
        step={'Video Config': {'fps': 5, 'duration': 10}, 'Color': 'BF'},
        video_as_frames=False,
        clock=lambda: clock['t'],
        elapsed_clock=lambda: clock['t'],
    )


class TestWaitForRecordingOutcome:
    def test_camera_inactive_returns_camera_lost(self, tmp_path, sim_scope):
        clock = {'t': 1000.0}
        recorder = _make_recorder(sim_scope, tmp_path, clock)
        sim_scope.simulation.remove_camera()
        engine = MagicMock(is_recording=True)
        outcome, reason = recorder._wait_for_recording(engine, 10.0, stall_threshold=5.0)
        assert outcome == protocol_recording.CAMERA_LOST
        assert reason == 'camera_disconnected'

    def test_stop_request_beats_camera_loss(self, tmp_path, sim_scope):
        clock = {'t': 1000.0}
        recorder = _make_recorder(sim_scope, tmp_path, clock)
        sim_scope.simulation.remove_camera()
        recorder._aborted.set()
        engine = MagicMock(is_recording=True)
        outcome, _ = recorder._wait_for_recording(engine, 10.0, stall_threshold=5.0)
        assert outcome == protocol_recording.CANCELLED

    def test_duration_elapsed_still_completes(self, tmp_path, sim_scope):
        # Each clock read advances 5.5 s: start at 1005.5, then 1011.0
        # (elapsed 5.5), then 1016.5 (elapsed 11 > duration) -> the
        # duration closes the loop with the camera alive the whole time.
        clock = {'t': 1000.0}

        def _advancing():
            clock['t'] += 5.5
            return clock['t']

        recorder = _make_recorder(sim_scope, tmp_path, clock)
        recorder._elapsed_clock = _advancing
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

    def _run_with_outcome(self, scope, tmp_path, outcome):
        clock = {'t': 1000.0}
        recorder = _make_recorder(scope, tmp_path, clock)
        recorder._video_as_frames = True
        with (
            patch.object(protocol_recording, 'VideoRecordingEngine', MagicMock()),
            patch.object(recorder, '_prologue', return_value=None),
            patch.object(recorder, '_wait_for_recording', return_value=(outcome, 'run_stop')),
        ):
            return recorder.run_blocking()

    def test_user_stop_with_zero_frames_stays_cancelled(self, tmp_path, sim_scope):
        result = self._run_with_outcome(sim_scope, tmp_path, protocol_recording.CANCELLED)
        assert result == protocol_recording.CANCELLED

    def test_run_abort_with_zero_frames_stays_aborted(self, tmp_path, sim_scope):
        result = self._run_with_outcome(sim_scope, tmp_path, protocol_recording.ABORTED)
        assert result == protocol_recording.ABORTED

    def test_full_course_with_zero_frames_is_no_frames(self, tmp_path, sim_scope):
        # Preservation guard: passes before and after the mapping fix.
        result = self._run_with_outcome(sim_scope, tmp_path, protocol_recording.COMPLETED)
        assert result == protocol_recording.NO_FRAMES

    def test_zero_frame_camera_loss_keeps_the_failure_row_shape(self, tmp_path, sim_scope):
        # A camera lost before delivering anything IS "delivered
        # nothing": NO_FRAMES keeps the caller's dropped-capture row and
        # the strike (the nonzero-frame camera-loss path records its row
        # via the finish thread instead).
        result = self._run_with_outcome(sim_scope, tmp_path, protocol_recording.CAMERA_LOST)
        assert result == protocol_recording.NO_FRAMES


class TestStartFailureEndsTheStep:
    """A step start that raises after the engine committed must end the
    recording it opened.

    The nested claim leaks nothing to the session, so the damage is not a
    stranded claim here -- it is a step left recording, which never
    satisfies the runner's end-of-run wait, so the whole protocol hangs
    on a step that never began.
    """

    def test_start_failure_ends_the_step_and_leaves_no_zero_frame_mp4(
        self, tmp_path, sim_scope, monkeypatch
    ):
        clock = {'t': 1000.0}
        recorder = _make_recorder(sim_scope, tmp_path, clock)
        # False color ON with a chromatic label is what makes the writer
        # open its container at construction; with it off the encoder init
        # defers and there is no artifact to leave behind, so the unlink
        # would go unexercised.
        recorder._step['Color'] = LED0
        recorder._step['False_Color'] = True

        # a stand-in by design: no simulated fault makes the registration
        # raise anything but its refusal, and the subject is the unwind of
        # an error the step does not expect.
        def _registration_fails(*_args, **_kwargs):
            raise RuntimeError('scripted registration failure')

        monkeypatch.setattr(sim_scope.imaging, 'add_frame_listener', _registration_fails)

        with (
            patch.object(recorder, '_prologue', return_value=None),
            pytest.raises(RuntimeError),
        ):
            recorder.run_blocking()

        assert recorder._engine is None
        assert recorder._writer is None
        assert not list(tmp_path.glob('*.mp4'))


class TestPrologueFeedDeath:
    """The pre-recording drain terminates on a dead feed -- and only then.

    Dead means zero frame ARRIVALS for the stall threshold. Stop is
    checked first each iteration, so a run stop always wins; and frames
    arriving with validity pinned (a stuck motion-settle) must never be
    blamed on the camera.

    A dead feed is the camera's stream stalled past the test. The step's
    clock moves 5 s at each read, the threshold for a 10 ms exposure, so
    one failed grab (its own 5 s wait for a new frame) is the whole dead
    feed.
    """

    def _prologue_result(self, scope, tmp_path, monkeypatch, *, stop_at_s, frame_arrives):
        clock = {'t': 0.0}

        def _advancing():
            clock['t'] += 5.0
            return clock['t']

        recorder = _make_recorder(scope, tmp_path, clock)
        recorder._elapsed_clock = _advancing
        recorder._is_run_in_progress = lambda: clock['t'] < stop_at_s
        # a stand-in by design: validity pinned is a settle check that never
        # ends, a production fault no simulated part makes.
        monkeypatch.setattr(scope.imaging, 'frames_until_valid', lambda *a, **k: 1)
        if not frame_arrives:
            scope.simulation.stall_stream(0.0, 600.0)
        return recorder._prologue({'Exposure': 10.0, 'Auto_Gain': False})

    @pytest.mark.slow
    def test_dead_feed_returns_no_frames_before_the_stop(self, tmp_path, sim_scope, monkeypatch):
        # Threshold 5 s (10 ms exposure -> floor); stop would land at 15 s.
        # The death verdict must fire first.
        result = self._prologue_result(
            sim_scope, tmp_path, monkeypatch, stop_at_s=15.0, frame_arrives=False
        )
        assert result == protocol_recording.NO_FRAMES

    def test_stop_before_the_threshold_wins(self, tmp_path, sim_scope, monkeypatch):
        # The stop is in before the first grab, on a dead feed.
        result = self._prologue_result(
            sim_scope, tmp_path, monkeypatch, stop_at_s=0.0, frame_arrives=False
        )
        assert result == protocol_recording.CANCELLED

    def test_arriving_frames_suppress_the_death_verdict(self, tmp_path, sim_scope, monkeypatch):
        # Validity pinned while frames arrive is a settle fault, not a
        # camera fault: the drain must never return NO_FRAMES for it.
        # (Bounded here by the stop; the pinned-validity hang itself is a
        # separately recorded finding.)
        result = self._prologue_result(
            sim_scope, tmp_path, monkeypatch, stop_at_s=60.0, frame_arrives=True
        )
        assert result == protocol_recording.CANCELLED


class TestWaitLoopFeedStall:
    """A feed that dies WITHOUT a disconnect event must still end the step.

    active stays True (the latch only flips on events); the stall
    watch on the ingest counter is what notices delivery stopped.
    """

    def _advancing_recorder(self, scope, tmp_path, step_s=0.5):
        clock = {'t': 1000.0}

        def _advancing():
            clock['t'] += step_s
            return clock['t']

        recorder = _make_recorder(scope, tmp_path, clock)
        recorder._elapsed_clock = _advancing
        return recorder

    def test_frozen_ingest_counter_ends_as_stalled(self, tmp_path, sim_scope):
        # duration 60 s >> threshold 5 s so the duration cannot mask the
        # stall verdict; _frames_seen never advances.
        recorder = self._advancing_recorder(sim_scope, tmp_path)
        engine = MagicMock(is_recording=True)
        outcome, reason = recorder._wait_for_recording(engine, 60.0, stall_threshold=5.0)
        assert outcome == protocol_recording.CAMERA_LOST
        assert reason == 'camera_stalled'

    def test_advancing_ingest_counter_never_stalls(self, tmp_path, sim_scope):
        # Preservation guard: frames arriving (counter advancing every
        # tick) run the step to its duration exactly as before.
        recorder = self._advancing_recorder(sim_scope, tmp_path)
        engine = MagicMock(is_recording=True)

        counter = {'n': 0}
        original_clock = recorder._elapsed_clock

        def _clock_and_frame():
            counter['n'] += 1
            recorder._frames_seen = counter['n']
            return original_clock()

        recorder._elapsed_clock = _clock_and_frame
        outcome, reason = recorder._wait_for_recording(engine, 20.0, stall_threshold=5.0)
        assert outcome == protocol_recording.COMPLETED
        assert reason == 'duration_elapsed'

    def test_stop_beats_the_stall_verdict(self, tmp_path, sim_scope):
        # Stop is checked first: even with the watch past threshold, a
        # stop requested in the same tick returns CANCELLED.
        recorder = self._advancing_recorder(sim_scope, tmp_path)
        recorder._aborted.set()
        engine = MagicMock(is_recording=True)
        outcome, _ = recorder._wait_for_recording(engine, 60.0, stall_threshold=5.0)
        assert outcome == protocol_recording.CANCELLED
