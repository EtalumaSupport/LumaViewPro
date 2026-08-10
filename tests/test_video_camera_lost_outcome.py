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
        assert recorder._wait_for_recording(engine, 10.0) == protocol_recording.CAMERA_LOST

    def test_stop_request_beats_camera_loss(self, tmp_path):
        clock = {'t': 1000.0}
        recorder = _make_recorder(tmp_path, clock, camera_active=False)
        recorder._aborted.set()
        engine = MagicMock(is_recording=True)
        assert recorder._wait_for_recording(engine, 10.0) == protocol_recording.CANCELLED

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
        assert recorder._wait_for_recording(engine, 10.0) == protocol_recording.COMPLETED


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
            patch.object(recorder, '_prologue', return_value=True),
            patch.object(recorder, '_wait_for_recording', return_value=outcome),
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
