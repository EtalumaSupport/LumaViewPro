# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A recording's outcome is reported once, as its type says, where it stops.

Before, each site in the manual recording and the protocol video step wrote
its own title and words, posted them, and logged a line of its own beside
the post; the finalize and frames-dropped outcomes were written twice, once
per path. Now each is a typed fault handed to the one reporter, unsolicited,
under the category it had, and the words the person reads are the type's.
The "FPS budget exceeded" notice is gone: the setting is a limit, and a
camera slower than a limit is not a problem.
"""

import pytest

import _lumascope.modules.manual_recording as manual_recording_module
import _lumascope.modules.stack_builder as stack_builder_module
import _lumascope.modules.protocol_recording as protocol_recording
from _lumascope.modules.exceptions import (
    HyperstackRefusedError,
    RecordingFinalizeError,
    RecordingFramesUnstampedError,
    RecordingStoppedError,
    VideoFramesDroppedError,
)
from _lumascope.modules.notification_center import outcome_of
from tests.test_manual_recording_controller import feed_frames, finish, make_controller
from tests.video_step_drives import record_first_frame, video_step_on
from tests.video_engine_harness import NotifyRecorder


def _record_reports(monkeypatch, module):
    reported = []
    monkeypatch.setattr(
        module.notifications,
        'report_outcome',
        lambda exception, **kw: reported.append((exception, kw)),
    )
    return reported


@pytest.mark.parametrize(
    ('fault', 'title', 'words'),
    [
        (
            RecordingStoppedError('camera_stalled'),
            'Recording Stopped',
            'The camera stopped delivering frames, so the recording was stopped. '
            'Frames captured so far are saved; check the camera connection before '
            'recording again.',
        ),
        (
            RecordingStoppedError('disk_floor'),
            'Recording Stopped -- Disk Almost Full',
            'Free disk space fell below the safety floor, so the recording was '
            'stopped early. Frames captured so far are saved; free up space before '
            'recording again.',
        ),
        (
            RecordingFinalizeError(protocol_step=False),
            'Recording Finalize Failed',
            'The recording finished but its output could not be fully assembled. '
            'Frames already written are on disk; check the log.',
        ),
        (
            RecordingFinalizeError(protocol_step=True),
            'Video Finalize Failed',
            'A video step finished but its output could not be fully assembled. '
            'Frames already written are on disk; check the log.',
        ),
        (
            VideoFramesDroppedError(2, 9, unstamped=0, protocol_step=False),
            'Video Frames Dropped',
            '2 of 9 frame(s) could not be written, so the saved video is shorter '
            'than the recording. Check the log for the cause.',
        ),
        (
            VideoFramesDroppedError(2, 9, unstamped=0, protocol_step=True),
            'Video Frames Dropped',
            '2 of 9 frame(s) in a video step could not be written, so that video is '
            'shorter than its recording. Check the log for the cause.',
        ),
        (
            VideoFramesDroppedError(2, 9, unstamped=3, protocol_step=False),
            'Video Frames Dropped',
            '2 of 9 frame(s) could not be written and 3 frame(s) came from the camera '
            'without their timestamps and were left out, so the saved video is shorter '
            'than the recording. Check the log for the cause.',
        ),
        (
            VideoFramesDroppedError(0, 9, unstamped=3, protocol_step=True),
            'Video Frames Dropped',
            '3 frame(s) in a video step came from the camera without their timestamps '
            'and were left out, so that video is shorter than its recording. Check the '
            'log for the cause.',
        ),
        (
            RecordingFramesUnstampedError(12, protocol_step=False),
            'Recording Saved No Frames',
            'The camera sent 12 frame(s) without their timestamps, so the recording '
            'saved none. Reconnect the camera before recording again.',
        ),
        (
            RecordingFramesUnstampedError(12, protocol_step=True),
            'Recording Saved No Frames',
            'The camera sent 12 frame(s) without their timestamps, so the video step '
            'saved none. Reconnect the camera before recording again.',
        ),
        (HyperstackRefusedError('one channel only'), 'Hyperstack Not Built', 'one channel only'),
    ],
    ids=[
        'stalled',
        'disk',
        'finalize-manual',
        'finalize-step',
        'dropped-manual',
        'dropped-step',
        'dropped-and-unstamped-manual',
        'unstamped-step',
        'all-unstamped-manual',
        'all-unstamped-step',
        'hyperstack',
    ],
)
def test_each_outcome_speaks_in_its_own_words_under_its_own_title(fault, title, words):
    assert outcome_of(fault).for_person, 'shown in its words, not the generic body'
    assert fault.title == title
    assert str(fault) == words


def test_a_lost_camera_is_reported_once_unsolicited(tmp_path, monkeypatch, post_processing):
    controller, _scope, _clock = make_controller(tmp_path, post_processing=post_processing)
    reported = _record_reports(monkeypatch, manual_recording_module)

    controller._stop_for_camera_loss('camera_disconnected')

    ((fault, kw),) = reported
    assert isinstance(fault, RecordingStoppedError) and fault.reason == 'camera_disconnected'
    # The recording's end, so the repeat window does not hold it back.
    assert kw == {'solicited': False, 'category': 'Recording', 'ends_a_run': True}


def test_a_full_disk_is_reported_once_unsolicited(tmp_path, monkeypatch, post_processing):
    controller, scope, clock = make_controller(tmp_path, post_processing=post_processing)
    reported = _record_reports(monkeypatch, manual_recording_module)
    checks = {'n': 0}

    def _fake_check(path, required_mb):
        checks['n'] += 1
        return (checks['n'] == 1, 100.0)

    monkeypatch.setattr(manual_recording_module, 'check_disk_space_ok', _fake_check)
    controller.start()
    feed_frames(scope, clock, 5, fps=10.0)
    controller._engine.wait_for_drain(timeout=10)
    finish(controller)

    stopped = [(f, kw) for f, kw in reported if isinstance(f, RecordingStoppedError)]
    ((fault, kw),) = stopped
    assert fault.reason == 'disk_floor'
    # The recording's end, so the repeat window does not hold it back.
    assert kw == {'solicited': False, 'category': 'Recording', 'ends_a_run': True}


def test_a_failed_finish_is_one_fault_chained_from_what_failed(
    tmp_path, monkeypatch, post_processing
):
    controller, scope, clock = make_controller(
        tmp_path, hyperstack=True, post_processing=post_processing
    )
    reported = _record_reports(monkeypatch, manual_recording_module)
    failure = RuntimeError('scripted hyperstack failure')

    class _ExplodingBuilder:
        def __init__(self, **kwargs):
            raise failure

    monkeypatch.setattr(stack_builder_module, 'StackBuilder', _ExplodingBuilder)
    controller.start()
    feed_frames(scope, clock, 3, fps=10.0)
    controller.stop()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, RecordingFinalizeError) and fault.protocol_step is False
    assert fault.__cause__ is failure
    assert kw == {'solicited': False, 'category': 'Recording'}


def test_a_refused_hyperstack_is_reported_in_the_builders_words(
    tmp_path, monkeypatch, post_processing
):
    controller, scope, clock = make_controller(
        tmp_path, hyperstack=True, post_processing=post_processing
    )
    reported = _record_reports(monkeypatch, manual_recording_module)

    class _RefusingBuilder:
        def __init__(self, **kwargs):
            pass

        def create_single_recording_stack(self, **kwargs):
            return {'status': False, 'error': 'the builder says no'}

    monkeypatch.setattr(stack_builder_module, 'StackBuilder', _RefusingBuilder)
    controller.start()
    feed_frames(scope, clock, 3, fps=10.0)
    controller.stop()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, HyperstackRefusedError) and str(fault) == 'the builder says no'
    assert kw == {'solicited': False, 'category': 'Recording'}


def test_dropped_frames_are_reported_once_unsolicited(tmp_path, monkeypatch, post_processing):
    controller, scope, clock = make_controller(tmp_path, post_processing=post_processing)
    reported = _record_reports(monkeypatch, manual_recording_module)
    real_write = manual_recording_module.write_frame
    calls = {'n': 0}

    def _flaky_write(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 2:
            raise OSError('scripted write failure')
        return real_write(*args, **kwargs)

    monkeypatch.setattr(manual_recording_module, 'write_frame', _flaky_write)
    controller.start()
    feed_frames(scope, clock, 5, fps=10.0)
    controller.stop()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, VideoFramesDroppedError)
    assert (fault.dropped, fault.protocol_step) == (1, False)
    assert kw == {'solicited': False, 'category': 'Recording'}


def test_a_limit_above_the_cameras_rate_shows_nothing(tmp_path, centre_posts, post_processing):
    controller, _scope, _clock = make_controller(
        tmp_path, max_fps=1000, post_processing=post_processing
    )
    recorder = NotifyRecorder(centre_posts)

    controller.start()
    controller.stop()
    finish(controller)

    assert recorder.calls == []


def test_a_video_steps_dropped_frames_are_reported_under_protocol(sim_scope, tmp_path, monkeypatch):
    reported = _record_reports(monkeypatch, protocol_recording)

    record_first_frame(sim_scope, tmp_path, monkeypatch, write_fails=True)

    ((fault, kw),) = reported
    assert isinstance(fault, VideoFramesDroppedError) and fault.protocol_step is True
    assert kw == {'solicited': False, 'category': 'Protocol'}


def test_a_video_steps_failed_finish_is_chained_under_protocol(sim_scope, tmp_path, monkeypatch):
    reported = _record_reports(monkeypatch, protocol_recording)
    failure = RuntimeError('scripted result failure')

    def _result(self):
        raise failure

    monkeypatch.setattr(protocol_recording.VideoRecordingEngine, 'result', _result)

    record_first_frame(sim_scope, tmp_path, monkeypatch)

    ((fault, kw),) = reported
    assert isinstance(fault, RecordingFinalizeError) and fault.protocol_step is True
    assert fault.__cause__ is failure
    assert kw == {'solicited': False, 'category': 'Protocol'}


def test_a_video_step_sent_only_unstamped_frames_says_so(sim_scope, tmp_path, monkeypatch):
    reported = _record_reports(monkeypatch, protocol_recording)
    monkeypatch.setattr(protocol_recording, 'check_disk_space_ok', lambda *a, **k: (True, 999999))
    # The step's elapsed clock moves 0.4 s with each frame the camera
    # delivers, so its one-second recording ends on frames, never on wall
    # time, and a feed that delivers is never taken for a stalled one.
    elapsed = {'now': 0.0}

    def _delivered(_frame):
        elapsed['now'] += 0.4

    sim_scope.simulation.withhold_chunk('Timestamp', 100_000)
    sim_scope.imaging.add_frame_listener(_delivered, name='the step clock')
    try:
        step = video_step_on(sim_scope, tmp_path, elapsed_clock=lambda: elapsed['now'])
        step.run_blocking()
        assert step.wait_until_finished(timeout=10.0), 'the finish never ended'
    finally:
        sim_scope.imaging.remove_frame_listener(_delivered)
        sim_scope.simulation.clear()

    ((fault, kw),) = reported
    assert isinstance(fault, RecordingFramesUnstampedError) and fault.protocol_step is True
    assert fault.unstamped > 0
    assert kw == {'solicited': False, 'category': 'Protocol'}
