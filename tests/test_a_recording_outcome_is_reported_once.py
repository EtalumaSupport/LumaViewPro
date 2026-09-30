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

import modules.manual_recording as manual_recording_module
import modules.protocol_recording as protocol_recording
from modules.exceptions import (
    HyperstackRefusedError,
    RecordingFinalizeError,
    RecordingStoppedError,
    VideoFramesDroppedError,
)
from modules.notification_center import _TYPED_FAULTS
from tests.test_manual_recording_controller import feed_frames, finish, make_controller
import tests.test_video_writer as video_writer_tests
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
            VideoFramesDroppedError(2, 9, protocol_step=False),
            'Video Frames Dropped',
            '2 of 9 frame(s) could not be written, so the saved video is shorter '
            'than the recording. Check the log for the cause.',
        ),
        (
            VideoFramesDroppedError(2, 9, protocol_step=True),
            'Video Frames Dropped',
            '2 of 9 frame(s) in a video step could not be written, so that video is '
            'shorter than its recording. Check the log for the cause.',
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
        'hyperstack',
    ],
)
def test_each_outcome_speaks_in_its_own_words_under_its_own_title(fault, title, words):
    assert isinstance(fault, _TYPED_FAULTS), 'shown in its words, not the generic body'
    assert fault.title == title
    assert str(fault) == words


def test_a_lost_camera_is_reported_once_unsolicited(tmp_path, monkeypatch):
    controller, _scope, _clock = make_controller(tmp_path)
    reported = _record_reports(monkeypatch, manual_recording_module)

    controller._stop_for_camera_loss('camera_disconnected')

    ((fault, kw),) = reported
    assert isinstance(fault, RecordingStoppedError) and fault.reason == 'camera_disconnected'
    assert kw == {'solicited': False, 'category': 'Recording'}


def test_a_full_disk_is_reported_once_unsolicited(tmp_path, monkeypatch):
    controller, scope, clock = make_controller(tmp_path)
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
    assert kw == {'solicited': False, 'category': 'Recording'}


def test_a_failed_finish_is_one_fault_chained_from_what_failed(tmp_path, monkeypatch):
    controller, scope, clock = make_controller(tmp_path, hyperstack=True)
    reported = _record_reports(monkeypatch, manual_recording_module)
    failure = RuntimeError('scripted hyperstack failure')

    class _ExplodingBuilder:
        def __init__(self, **kwargs):
            raise failure

    monkeypatch.setattr(manual_recording_module, 'StackBuilder', _ExplodingBuilder)
    controller.start()
    feed_frames(scope, clock, 3, fps=10.0)
    controller.stop()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, RecordingFinalizeError) and fault.protocol_step is False
    assert fault.__cause__ is failure
    assert kw == {'solicited': False, 'category': 'Recording'}


def test_a_refused_hyperstack_is_reported_in_the_builders_words(tmp_path, monkeypatch):
    controller, scope, clock = make_controller(tmp_path, hyperstack=True)
    reported = _record_reports(monkeypatch, manual_recording_module)

    class _RefusingBuilder:
        def __init__(self, **kwargs):
            pass

        def create_single_recording_stack(self, **kwargs):
            return {'status': False, 'error': 'the builder says no'}

    monkeypatch.setattr(manual_recording_module, 'StackBuilder', _RefusingBuilder)
    controller.start()
    feed_frames(scope, clock, 3, fps=10.0)
    controller.stop()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, HyperstackRefusedError) and str(fault) == 'the builder says no'
    assert kw == {'solicited': False, 'category': 'Recording'}


def test_dropped_frames_are_reported_once_unsolicited(tmp_path, monkeypatch):
    controller, scope, clock = make_controller(tmp_path)
    reported = _record_reports(monkeypatch, manual_recording_module)
    real_write = manual_recording_module.image_save.write_video_frame
    calls = {'n': 0}

    def _flaky_write(**kwargs):
        calls['n'] += 1
        if calls['n'] == 2:
            raise OSError('scripted write failure')
        return real_write(**kwargs)

    monkeypatch.setattr(manual_recording_module.image_save, 'write_video_frame', _flaky_write)
    controller.start()
    feed_frames(scope, clock, 5, fps=10.0)
    controller.stop()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, VideoFramesDroppedError)
    assert (fault.dropped, fault.protocol_step) == (1, False)
    assert kw == {'solicited': False, 'category': 'Recording'}


def test_a_limit_above_the_cameras_rate_shows_nothing(tmp_path, monkeypatch):
    controller, _scope, _clock = make_controller(tmp_path, max_fps=1000)
    recorder = NotifyRecorder()
    monkeypatch.setattr(manual_recording_module, 'notifications', recorder)

    controller.start()
    controller.stop()
    finish(controller)

    assert recorder.calls == []


def test_a_video_steps_dropped_frames_are_reported_under_protocol(tmp_path, monkeypatch):
    reported = _record_reports(monkeypatch, protocol_recording)

    video_writer_tests.TestProtocolVideoDropNotification()._run_one_frame_step(
        tmp_path, monkeypatch, write_fails=True
    )

    ((fault, kw),) = reported
    assert isinstance(fault, VideoFramesDroppedError) and fault.protocol_step is True
    assert kw == {'solicited': False, 'category': 'Protocol'}


def test_a_video_steps_failed_finish_is_chained_under_protocol(tmp_path, monkeypatch):
    reported = _record_reports(monkeypatch, protocol_recording)
    failure = RuntimeError('scripted result failure')

    def _result(self):
        raise failure

    monkeypatch.setattr(protocol_recording.VideoRecordingEngine, 'result', _result)

    video_writer_tests.TestProtocolVideoDropNotification()._run_one_frame_step(
        tmp_path, monkeypatch, write_fails=False
    )

    ((fault, kw),) = reported
    assert isinstance(fault, RecordingFinalizeError) and fault.protocol_step is True
    assert fault.__cause__ is failure
    assert kw == {'solicited': False, 'category': 'Protocol'}
