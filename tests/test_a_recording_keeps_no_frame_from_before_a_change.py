# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A recording keeps no frame made before a change the scope recorded was in force.

A run's video step waits, before its first frame, on the condition a capture
drains on (``ImagingAPI.changes_in_force``), so the first frame it keeps
counts for every change, by the rule that change's source runs, with live
view off as a run records it. The step's own Auto_Gain arm records a gain
and an exposure change, and is waited out the same way.

Each test records on the simulated scope, reads the first frame the step
saved (its ``frame_id`` in the file's record), and judges the frame the
camera delivered with that id through ``judge_frame``. The step keeps every
frame (its rate above the camera's), so the first frame kept is the first
frame after its start, not a later cadence slot that would hide an early one.
"""

import threading

from modules.frame_validity import FrameValiditySource
from modules.image_utils import read_file_record
from tests.video_step_drives import video_step_on

AG_SETTINGS = {'target_brightness': 0.3, 'min_gain_db': 0.0, 'max_gain_db': 20.0}
EVERY_FRAME = {'Video Config': {'fps': 1000, 'duration': 1}}


class _Delivered:
    """Every frame the camera delivers, by its frame id, through the public listener."""

    def __init__(self, imaging):
        self._imaging = imaging
        self._lock = threading.Lock()
        self.by_id = {}
        imaging.add_frame_listener(self._on_frame, name='test_delivered')

    def _on_frame(self, frame):
        with self._lock:
            self.by_id[int(frame.chunks['FrameID'])] = frame

    def close(self):
        self._imaging.remove_frame_listener(self._on_frame)


def _first_kept_frame_is_judged(scope, save_folder, change, *, step=None, autogain=None):
    """Record a video step right after ``change``; the judgments of its first kept frame."""
    delivered = _Delivered(scope.imaging)
    try:
        change()
        kwargs = {'autogain_settings': autogain} if autogain is not None else {}
        recorder = video_step_on(scope, save_folder, step={**EVERY_FRAME, **(step or {})}, **kwargs)
        assert recorder.run_blocking() == 'completed'
    finally:
        delivered.close()
    first = sorted(save_folder.rglob('*_Frame_0000.tif*'))
    assert first, 'the step saved no frame'
    frame = delivered.by_id[read_file_record(first[0])['frame_id']]
    return {
        j.change.source: j
        for j in scope.imaging.judge_frame(frame, scope.imaging.map_camera_clock())
    }


def test_a_video_steps_first_frame_counts_for_a_black_level_change_just_before_it(
    sim_scope, sim_camera, tmp_path
):
    """Before the step waited on the clock rule, a fixed pause after a frame
    count nearly covered the black level's delay: its first frame started a
    few milliseconds early in some runs on the camera's realistic frame
    timing and not in others, so this pins the after-state; the Auto_Gain
    test below is the step's fail-before."""
    sim_camera[1].set_timing_mode('realistic')
    sim_scope.camera.set_black_level(2.0)
    sim_scope.imaging.capture_still()

    judged = _first_kept_frame_is_judged(
        sim_scope, tmp_path, lambda: sim_scope.camera.set_black_level(4.0)
    )

    assert judged[FrameValiditySource.BLACK_LEVEL].counts


def test_a_video_steps_first_frame_counts_for_the_gain_and_exposure_its_auto_gain_arm_set(
    sim_scope, tmp_path
):
    judged = _first_kept_frame_is_judged(
        sim_scope,
        tmp_path,
        lambda: None,
        step={'Auto_Gain': True},
        autogain=AG_SETTINGS,
    )

    assert judged[FrameValiditySource.GAIN].counts
    assert judged[FrameValiditySource.EXPOSURE].counts
