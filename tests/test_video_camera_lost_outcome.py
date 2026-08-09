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
