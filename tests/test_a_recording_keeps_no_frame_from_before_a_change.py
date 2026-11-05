# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A recording keeps no frame made before a change the scope recorded was in force.

A run's video step waits, before its first frame, on the condition a capture
drains on (``ImagingAPI.changes_in_force``), so the first frame it keeps
counts for every change, by that change's count of frames, with live
view off as a run records it. The step's own Auto_Gain arm records a gain
and an exposure change, and is waited out the same way.

Each test records on the simulated scope, reads the first frame the step
saved (its ``frame_id`` in the file's record), and judges the frame the
camera delivered with that id through ``judge_frame``. The step keeps every
frame (its rate above the camera's), so the first frame kept is the first
frame after its start, not a later cadence slot that would hide an early one.
"""

import json
import pathlib
import threading
import pytest

from drivers import host_clock
from modules.exceptions import RecordingRefusedError
from modules.frame_validity import FrameValiditySource
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

    def frames(self):
        """The frames delivered so far."""
        with self._lock:
            return list(self.by_id.values())

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


def _judged(scope, frame):
    """``frame``'s judgments, by source."""
    return {j.change.source: j for j in scope.imaging.judge_frame(frame)}


def _why_not(judged, source):
    """The whole judgment of ``source``, for a red: pytest's own rendering cuts
    it short, and the frames after the change and who counted the frames
    before it are what a red is read by."""
    return repr(judged[source])


def test_a_video_steps_first_frame_counts_for_a_black_level_change_just_before_it(
    sim_scope, sim_camera, tmp_path
):
    """The step's first kept frame counts for a black-level change made just
    before it, on the camera's realistic frame timing: this pins the
    after-state; the Auto_Gain test below is the step's fail-before."""
    sim_camera[1].set_timing_mode('realistic')
    sim_scope.camera.set_black_level(2.0)
    sim_scope.imaging.capture_still()

    judged = _first_kept_frame_is_judged(
        sim_scope, tmp_path, lambda: sim_scope.camera.set_black_level(4.0)
    )

    assert judged[FrameValiditySource.BLACK_LEVEL].counts, _why_not(
        judged, FrameValiditySource.BLACK_LEVEL
    )


def _first_manual_frame_is_judged(session, live_folder, change):
    """Start a manual recording right after ``change``; the judgments of its first kept frame."""
    scope = session.scope
    delivered = _Delivered(scope.imaging)
    try:
        change()
        operation = session.manual_recording.start()
        started_ns = host_clock.steady_ns()
        _await_frames_started_after(scope, delivered, started_ns, 5)
        stop_ns = host_clock.steady_ns()
        operation.stop()
        finished = operation.result(timeout=30)
    finally:
        delivered.close()
    first = sorted(pathlib.Path(live_folder).rglob('*_0000_*.tif*'))
    assert first, _why_none_kept(scope, live_folder, delivered, finished, started_ns, stop_ns)
    return _judged(scope, delivered.by_id[read_file_record(first[0])['frame_id']])


def _frame_start_ticks(frame):
    """When a delivered frame started, on the camera's clock (its ``Timestamp`` chunk); None without one."""
    return (frame.chunks or {}).get('Timestamp')


def _await_frames_started_after(scope, delivered, started_ns, n, timeout=10.0):
    """Wait until ``n`` delivered frames started after ``started_ns``.

    A count of delivered frames is no stand-in: frames already exposing when
    the recording started are delivered after it, a burst of them when the
    host paused (a 784 ms collection in CI run 41879806320 delivered six,
    every one started before ``start`` returned), and the recording rightly
    keeps none of them.
    """
    mapping = scope.imaging.map_camera_clock()
    assert mapping is not None, 'the camera states no clock, so no frame start can be placed'
    start_ticks = mapping.camera_ticks_at(started_ns)
    deadline_ns = host_clock.steady_ns() + int(timeout * 1e9)

    def started_after():
        ticks = (_frame_start_ticks(frame) for frame in delivered.frames())
        return sum(1 for t in ticks if t is not None and t > start_ticks)

    while started_after() < n:
        left_s = (deadline_ns - host_clock.steady_ns()) / 1e9
        assert left_s > 0, f'fewer than {n} frames started after the recording did, in {timeout} s'
        delivered.await_more(1, timeout=left_s)


def _why_none_kept(scope, live_folder, delivered, finished, started_ns, stop_ns):
    """What a recording that kept no frame was offered, for a red.

    Whether frames that started after ``start`` returned came before the
    stop, read through a clock mapping latched after the recording and off
    by its uncertainty: frames after the start and none kept is the
    recording's to answer, the stop before any such frame the test's.
    """
    stated = [f'the recording saved no frame: {finished!r}']
    for manifest in pathlib.Path(live_folder).rglob('recording_manifest.json'):
        facts = json.loads(manifest.read_text())
        counts = ('frames_selected', 'frames_written', 'write_failures', 'frames_unstamped')
        stated.append(f'manifest { ({key: facts.get(key) for key in counts})!r}')
    mapping = scope.imaging.map_camera_clock()
    if mapping is None:
        stated.append('no clock mapping, so no frame start is stated')
        return '\n'.join(stated)
    stated.append(f'mapping {mapping!r}')
    start_ticks = mapping.camera_ticks_at(started_ns)

    def after_start_ms(ticks):
        return (ticks - start_ticks) * 1000 / mapping.tick_hz

    stated.append(
        f'stop asked {after_start_ms(mapping.camera_ticks_at(stop_ns)):+.3f} ms after start returned'
    )
    for frame_id, frame in sorted(delivered.by_id.items())[-15:]:
        ticks = _frame_start_ticks(frame)
        when = f'{after_start_ms(ticks):+.3f} ms' if ticks is not None else 'at no stated tick'
        stated.append(f'frame {frame_id} started {when} after start returned')
    return '\n'.join(stated)


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
    # parametrize-justified: three sources, each its own setter and count of frames.
    camera = sim_session.scope.camera
    getattr(camera, before[0])(before[1])
    sim_session.scope.imaging.capture_still()

    judged = _first_manual_frame_is_judged(
        sim_session, tmp_path, lambda: getattr(camera, after[0])(after[1])
    )

    assert judged[source].counts, _why_not(judged, source)


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
        assert judged[source].counts, _why_not(judged, source)
