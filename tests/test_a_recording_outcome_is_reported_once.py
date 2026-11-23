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

import pathlib
import threading

import pytest

import _lumascope.modules.manual_recording as manual_recording_module
import _lumascope.modules.stack_builder as stack_builder_module
import _lumascope.modules.protocol_recording as protocol_recording
import _lumascope.modules.video_recording as video_recording
from _lumascope.modules.exceptions import (
    FramesLostIn,
    HyperstackRefusedError,
    RecordingFinalizeError,
    RecordingFramesUnstampedError,
    RecordingStoppedError,
    VideoFramesDroppedError,
    VideoNotFinishedError,
    VideoWriterFailedError,
)
from _lumascope.modules.notification_center import outcome_of
from _lumascope.modules.run_outcome import RunEnding, RunStatus
from _lumascope.modules.image_mode import RecordingFormat
from tests.disk_fakes import PLENTY_MB, free_space_is
from tests.test_manual_recording_controller import feed_frames, finish, make_controller
from tests.video_close_faults import fail_the_close
from tests.video_step_drives import record_first_frame, record_first_mp4_frame, video_step_on
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
            VideoNotFinishedError('/data/clip.mp4'),
            'Video Not Finished',
            'The video /data/clip.mp4 could not be finished and may not open. '
            'Check disk space and the log.',
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
            VideoFramesDroppedError(
                2,
                9,
                unstamped=0,
                skipped=0,
                lost_at_camera=0,
                lost_in=FramesLostIn.MANUAL_RECORDING,
            ),
            'Video Frames Dropped',
            '2 of 9 frame(s) could not be written, so the saved video has gaps. Check '
            'the log for why frames could not be written.',
        ),
        (
            VideoFramesDroppedError(
                2, 9, unstamped=0, skipped=0, lost_at_camera=0, lost_in=FramesLostIn.VIDEO_STEP
            ),
            'Video Frames Dropped',
            '2 of 9 frame(s) in a video step could not be written, so that video has '
            'gaps. Check the log for why frames could not be written.',
        ),
        (
            VideoFramesDroppedError(
                2,
                9,
                unstamped=3,
                skipped=0,
                lost_at_camera=0,
                lost_in=FramesLostIn.MANUAL_RECORDING,
            ),
            'Video Frames Dropped',
            '2 of 9 frame(s) could not be written; 3 frame(s) came from the camera '
            'without their timestamps and were left out, so the saved video has gaps. '
            'Check the log for why frames could not be written.',
        ),
        (
            VideoFramesDroppedError(
                0, 9, unstamped=3, skipped=0, lost_at_camera=0, lost_in=FramesLostIn.VIDEO_STEP
            ),
            'Video Frames Dropped',
            '3 frame(s) in a video step came from the camera without their timestamps '
            'and were left out, so that video has gaps.',
        ),
        (
            VideoFramesDroppedError(
                0,
                9,
                unstamped=0,
                skipped=4,
                lost_at_camera=None,
                lost_in=FramesLostIn.MANUAL_RECORDING,
            ),
            'Video Frames Dropped',
            '4 frame(s) were skipped because this PC could not save frames as fast as '
            'the camera sent them, so the saved video has gaps. A lower Recording max '
            'FPS in Advanced Settings avoids the skips.',
        ),
        (
            VideoFramesDroppedError(
                0, 9, unstamped=0, skipped=0, lost_at_camera=5, lost_in=FramesLostIn.VIDEO_STEP
            ),
            'Video Frames Dropped',
            '5 frame(s) the camera sent during a video step never reached the '
            'recording, so that video has gaps.',
        ),
        (
            VideoFramesDroppedError(
                0, 9, unstamped=0, skipped=4, lost_at_camera=5, lost_in=FramesLostIn.RUN
            ),
            'Video Frames Dropped',
            "4 frame(s) in the run's video steps were skipped because this PC could not "
            'save frames as fast as the camera sent them; 5 frame(s) the camera sent during '
            "the run's video steps never reached their recordings, so those videos have "
            'gaps. A lower Recording max FPS in Advanced Settings avoids the skips.',
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
        'video-not-finished',
        'finalize-manual',
        'finalize-step',
        'dropped-manual',
        'dropped-step',
        'dropped-and-unstamped-manual',
        'unstamped-step',
        'skipped-manual',
        'lost-at-camera-step',
        'skipped-and-lost-at-camera-run',
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
    # The start finds room; every rolling probe finds the disk full.
    free_space_is(monkeypatch, PLENTY_MB)
    monkeypatch.setattr(manual_recording_module, 'check_disk_space_ok', lambda *_: (False, 100.0))
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
        tmp_path, recording_format=RecordingFormat.HYPERSTACK, post_processing=post_processing
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
        tmp_path, recording_format=RecordingFormat.HYPERSTACK, post_processing=post_processing
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
    assert (fault.dropped, fault.lost_in) == (1, FramesLostIn.MANUAL_RECORDING)
    assert kw == {'solicited': False, 'category': 'Recording'}


def test_frames_skipped_for_memory_are_reported_once_in_their_own_words(
    tmp_path, monkeypatch, post_processing
):
    controller, scope, clock = make_controller(tmp_path, post_processing=post_processing)
    reported = _record_reports(monkeypatch, manual_recording_module)
    # A stand-in by design: the host's memory is not the simulator's.
    monkeypatch.setattr(
        video_recording,
        'read_memory_headroom',
        lambda: video_recording.MemoryHeadroom(
            physical_available_bytes=256 << 20, commit_headroom_bytes=None
        ),
    )
    real_write = manual_recording_module.write_frame
    release = threading.Event()

    def _held_write(*args, **kwargs):
        assert release.wait(5), 'the write was held past 5 s'
        return real_write(*args, **kwargs)

    monkeypatch.setattr(manual_recording_module, 'write_frame', _held_write)
    controller.start()
    # Four frames in the four writers' hands and one waiting; the next three
    # wait beyond the writers with memory short.
    feed_frames(scope, clock, 8, fps=10.0)
    controller.stop()
    release.set()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, VideoFramesDroppedError)
    assert (fault.skipped, fault.dropped, fault.selected) == (3, 0, 5)
    assert str(fault).startswith(
        '3 frame(s) were skipped because this PC could not save frames as fast as the '
        'camera sent them'
    )
    assert kw == {'solicited': False, 'category': 'Recording'}


def test_frames_the_camera_lost_are_reported_once_and_never_told_as_kept(
    tmp_path, monkeypatch, post_processing
):
    controller, scope, clock = make_controller(tmp_path, post_processing=post_processing)
    reported = _record_reports(monkeypatch, manual_recording_module)
    monkeypatch.setattr(video_recording, 'RECORDING_SAVE_WAIT_NOTICE_S', -1.0)
    controller.start()
    feed_frames(scope, clock, 3, fps=10.0, start_id=0)
    feed_frames(scope, clock, 3, fps=10.0, start_id=10)  # 3-9 never arrived
    controller.stop()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, VideoFramesDroppedError), 'a recording with gaps was told as kept'
    assert fault.lost_at_camera == 7
    assert '7 frame(s) the camera sent never reached the recording' in str(fault)
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
    assert isinstance(fault, VideoFramesDroppedError)
    assert fault.lost_in is FramesLostIn.VIDEO_STEP
    assert kw == {'solicited': False, 'category': 'Protocol', 'log_only': True}


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


def test_a_manual_video_that_cannot_be_finished_is_reported_as_itself(
    tmp_path, monkeypatch, post_processing
):
    # The disk fills mid-stream: frames fail to encode, then the close fails.
    # Their counts describe a file that may not open, so the one report is
    # the video not finished, and the file is kept.
    controller, scope, clock = make_controller(
        tmp_path, recording_format=RecordingFormat.MP4, post_processing=post_processing
    )
    reported = _record_reports(monkeypatch, manual_recording_module)
    failing = fail_the_close(monkeypatch, after_packets=2)

    controller.start()
    feed_frames(scope, clock, 40, fps=10.0)
    controller.stop()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, VideoNotFinishedError)
    assert str(fault.output_path) == failing[0]
    assert kw == {'solicited': False, 'category': 'Recording'}
    assert failing.packets_refused, 'no frame failed to encode; the case is not mid-stream'
    assert pathlib.Path(failing[0]).exists(), 'the only copy of the recording was removed'


def test_a_manual_video_that_cannot_be_finished_is_never_said_saved_slowly(
    tmp_path, monkeypatch, post_processing
):
    # The disk fills only at the trailer: every frame encoded, the close
    # failed. A slow save is told as every frame kept, untrue of this file.
    controller, scope, clock = make_controller(
        tmp_path, recording_format=RecordingFormat.MP4, post_processing=post_processing
    )
    reported = _record_reports(monkeypatch, manual_recording_module)
    monkeypatch.setattr(video_recording, 'RECORDING_SAVE_WAIT_NOTICE_S', -1.0)
    fail_the_close(monkeypatch)

    controller.start()
    feed_frames(scope, clock, 3, fps=10.0)
    controller.stop()
    finish(controller)

    assert [type(fault) for fault, _kw in reported] == [VideoNotFinishedError]


def test_a_video_step_that_cannot_be_finished_is_one_file_not_written(
    sim_scope, tmp_path, monkeypatch
):
    reported = _record_reports(monkeypatch, protocol_recording)
    failing = fail_the_close(monkeypatch)

    step = record_first_mp4_frame(sim_scope, tmp_path, monkeypatch)

    ((fault, kw),) = reported
    assert isinstance(fault, VideoNotFinishedError)
    assert str(fault.output_path) == failing[0]
    assert kw == {'solicited': False, 'category': 'Protocol'}
    step._record_dropped_capture.assert_called_once_with(
        reason='video_finalize_failed', capture_time=step._start_dt
    )
    step._record_step_row.assert_not_called()
    # Its loss is the file: the run's end neither counts its frames nor
    # says every frame was kept.
    assert step.frame_losses is None
    assert step.kept_every_frame is False


def test_a_video_steps_writer_death_still_ends_the_run_when_its_close_fails(
    sim_scope, tmp_path, monkeypatch
):
    reported = _record_reports(monkeypatch, protocol_recording)
    fail_the_close(monkeypatch)

    class _LaneKiller(BaseException):
        """Not an Exception, so the writer thread treats it as its own death."""

    def _dying_write(self, *args, **kwargs):
        raise _LaneKiller('scripted writer death')

    monkeypatch.setattr(protocol_recording.ProtocolVideoStep, '_write_frame', _dying_write)
    # False color colors the stream, so the writer opens its file at once
    # and has a close to fail though no frame reaches it.
    step = record_first_mp4_frame(sim_scope, tmp_path, monkeypatch, step={'False_Color': True})

    assert {type(fault) for fault, _kw in reported} == {
        VideoNotFinishedError,
        VideoWriterFailedError,
    }
    (died,) = [fault for fault, _kw in reported if isinstance(fault, VideoWriterFailedError)]
    step._abort_run_fatal.assert_called_once_with(
        RunEnding(RunStatus.FAILED, 'video_writer_died', str(died)), 'Recording'
    )


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
    assert kw == {'solicited': False, 'category': 'Protocol', 'log_only': True}


def test_a_dropped_frames_fault_carries_its_counts_on_the_wire():
    from _lumascope.rest import problems

    fault = VideoFramesDroppedError(
        1, 9, unstamped=2, skipped=3, lost_at_camera=None, lost_in=FramesLostIn.MANUAL_RECORDING
    )
    body = problems.answered_by_member(fault, 'req').body
    assert body['type'].endswith('video_frames_dropped')
    assert {
        k: body[k] for k in ('dropped', 'selected', 'unstamped', 'skipped', 'lost_at_camera')
    } == {
        'dropped': 1,
        'selected': 9,
        'unstamped': 2,
        'skipped': 3,
        'lost_at_camera': None,
    }
