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

import pathlib
import threading
from typing import NamedTuple

import pytest

from modules.exceptions import RecordingRefusedError
from modules.frame_validity import FrameValiditySource
from modules.lumascope_api.frame_record import ClockMapping
from modules.image_utils import read_file_record
from tests.video_step_drives import video_step_on

AG_SETTINGS = {'target_brightness': 0.3, 'min_gain_db': 0.0, 'max_gain_db': 20.0}
EVERY_FRAME = {'Video Config': {'duration': 1}}


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
            if self._awaited is not None and len(self.by_id) >= self._awaited[0]:
                self._awaited[1].set()

    _awaited = None

    def await_more(self, n, timeout=10.0):
        """Wait until ``n`` more frames than now have been delivered."""
        event = threading.Event()
        with self._lock:
            self._awaited = (len(self.by_id) + n, event)
        assert event.wait(timeout), f'the camera delivered fewer than {n} frames'

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
    return _judged(scope, delivered.by_id[read_file_record(first[0])['frame_id']])


class _Judged(NamedTuple):
    """A first kept frame's judgments, with the clock mapping they were made through."""

    by_source: dict
    mapping: ClockMapping | None


def _judged(scope, frame):
    """``frame``'s judgments, through a clock mapping latched after the recording.

    Not the recording's own mapping: no public member reads it (a recorded
    frame's record carries none), and a latch after the recording carries the
    same camera clock, off by its own uncertainty, which ``_why_not`` states.
    """
    mapping = scope.imaging.map_camera_clock()
    return _Judged({j.change.source: j for j in scope.imaging.judge_frame(frame, mapping)}, mapping)


def _why_not(judged, source):
    """The whole judgment of ``source``, for a red: pytest's own rendering cuts it
    short, and its anchor, the frame's start and the mapping's uncertainty are
    what a red is read by."""
    judgment = judged.by_source[source]
    mapping = judged.mapping
    stated = [repr(judgment), f'mapping {mapping!r}']
    if mapping is not None and None not in (
        judgment.frame_start_ticks,
        judgment.qualifies_after_ticks,
    ):
        late_ms = (
            (judgment.frame_start_ticks - judgment.qualifies_after_ticks) * 1000 / mapping.tick_hz
        )
        stated.append(
            f'the frame started {late_ms:+.3f} ms after the instant it had to follow '
            '(negative: before it)'
        )
    return '\n'.join(stated)


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

    assert judged.by_source[FrameValiditySource.BLACK_LEVEL].counts, _why_not(
        judged, FrameValiditySource.BLACK_LEVEL
    )


def _first_manual_frame_is_judged(session, live_folder, change):
    """Start a manual recording right after ``change``; the judgments of its first kept frame."""
    scope = session.scope
    delivered = _Delivered(scope.imaging)
    try:
        change()
        operation = session.manual_recording.start()
        delivered.await_more(5)
        operation.stop()
        operation.result(timeout=30)
    finally:
        delivered.close()
    first = sorted(pathlib.Path(live_folder).rglob('*_0000_*.tif*'))
    assert first, 'the recording saved no frame'
    return _judged(scope, delivered.by_id[read_file_record(first[0])['frame_id']])


FRAMES_LEG = {'video_as_frames': True}


@pytest.mark.parametrize('sim_session', [FRAMES_LEG], indirect=True)
@pytest.mark.parametrize(
    ('source', 'before', 'after'),
    [
        (FrameValiditySource.GAIN, ('set_gain_db', 0.0), ('set_gain_db', 6.0)),
        (FrameValiditySource.EXPOSURE, ('set_exposure_ms', 30.0), ('set_exposure_ms', 10.0)),
        (FrameValiditySource.BLACK_LEVEL, ('set_black_level', 2.0), ('set_black_level', 4.0)),
    ],
)
def test_a_manual_recordings_first_frame_counts_for_a_change_just_before_record(
    sim_session, tmp_path, source, before, after
):
    """Live view off, as a headless start is: no preview counts a frame."""
    # parametrize-justified: three sources, each its own clock rule's delay.
    camera = sim_session.scope.camera
    getattr(camera, before[0])(before[1])
    sim_session.scope.imaging.capture_still()

    judged = _first_manual_frame_is_judged(
        sim_session, tmp_path, lambda: getattr(camera, after[0])(after[1])
    )

    assert judged.by_source[source].counts, _why_not(judged, source)


@pytest.mark.parametrize('sim_session', [FRAMES_LEG], indirect=True)
def test_a_manual_recording_whose_change_never_comes_into_force_is_refused(
    sim_session, monkeypatch
):
    # a stand-in by design: a change that never comes into force while frames
    # arrive is a settle check that never ends, a fault no simulated part makes.
    monkeypatch.setattr(sim_session.scope.imaging, 'changes_in_force', lambda *a, **k: False)

    with pytest.raises(RecordingRefusedError) as refused:
        sim_session.manual_recording.start()

    assert refused.value.reason == 'changes_not_in_force'
    assert not sim_session.manual_recording.is_recording
    assert not sim_session.manual_recording.is_busy


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

    for source in (FrameValiditySource.GAIN, FrameValiditySource.EXPOSURE):
        assert judged.by_source[source].counts, _why_not(judged, source)
