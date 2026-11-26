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

import errno
import json
import os
import pathlib
import re
import threading

import pytest

import _lumascope.modules.manual_recording as manual_recording_module
import _lumascope.modules.stack_builder as stack_builder_module
import _lumascope.modules.protocol_recording as protocol_recording
import _lumascope.modules.video_recording as video_recording
from _lumascope.modules.exceptions import (
    INTERNAL_FAULT_SENTENCE,
    FramesLostIn,
    HyperstackNotBuiltError,
    InputFramesUnusableError,
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
import _lumascope.modules.common_utils as common_utils
from tests.disk_fakes import PLENTY_MB, free_space_is, the_start_finds_room
from tests.test_manual_recording_controller import feed_frames, finish, make_controller
from tests.video_close_faults import fail_the_close, refuse_packets
from tests.video_step_drives import record_first_frame, record_first_mp4_frame, video_step_on
from tests.video_engine_harness import NotifyRecorder


# Two frames a full disk refused, as the recording keeps them.
_FULL_DISK_TWICE = (('frame 3', 'No space left on device'), ('frame 4', 'No space left on device'))


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
            RecordingStoppedError('camera_stalled', recording_format=RecordingFormat.FRAMES),
            'Recording Stopped',
            'The camera stopped delivering frames, so the recording was stopped. '
            'What was saved before it is kept. Check the camera connection before '
            'recording again.',
        ),
        (
            # Headed as a run that lost its camera is.
            RecordingStoppedError('camera_disconnected', recording_format=RecordingFormat.FRAMES),
            'Camera Disconnected',
            'The camera stopped delivering frames, so the recording was stopped. '
            'What was saved before it is kept. Check the camera connection before '
            'recording again.',
        ),
        (
            RecordingStoppedError('disk_space_critical', recording_format=RecordingFormat.FRAMES),
            'Disk Space Critical',
            'Free disk space fell below the safety floor, so the recording was '
            'stopped early. What was saved before it is kept. Free up space before '
            'recording again.',
        ),
        (
            VideoNotFinishedError(
                '/data/clip.mp4', OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), '/data/clip.mp4')
            ),
            'Video Not Finished',
            'The video /data/clip.mp4 could not be finished (No space left on device) '
            'and may not open.',
        ),
        (
            RecordingFinalizeError(protocol_step=False, recording_format=RecordingFormat.FRAMES),
            'Recording Finalize Failed',
            'The recording finished but its output could not be fully assembled. '
            f'What was saved before it is kept. {INTERNAL_FAULT_SENTENCE}',
        ),
        (
            RecordingFinalizeError(protocol_step=True, recording_format=RecordingFormat.FRAMES),
            'Recording Finalize Failed',
            'A video step finished but its output could not be fully assembled. '
            f'What was saved before it is kept. {INTERNAL_FAULT_SENTENCE}',
        ),
        (
            VideoFramesDroppedError(
                _FULL_DISK_TWICE,
                9,
                frames_unstamped=0,
                frames_skipped=0,
                frames_lost_at_camera=0,
                lost_in=FramesLostIn.MANUAL_RECORDING,
            ),
            'Video Frames Dropped',
            '2 of 9 frame(s) could not be written, so the saved video has gaps. Not '
            'written: No space left on device (2: frame 3, frame 4).',
        ),
        (
            VideoFramesDroppedError(
                (('frame 3', 'No space left on device'), ('frame 7', 'Permission denied')),
                9,
                frames_unstamped=0,
                frames_skipped=0,
                frames_lost_at_camera=0,
                lost_in=FramesLostIn.VIDEO_STEP,
            ),
            'Video Frames Dropped',
            '2 of 9 frame(s) in a video step could not be written, so that video has '
            'gaps. Not written: No space left on device (frame 3); Permission denied '
            '(frame 7).',
        ),
        (
            VideoFramesDroppedError(
                _FULL_DISK_TWICE,
                9,
                frames_unstamped=3,
                frames_skipped=0,
                frames_lost_at_camera=0,
                lost_in=FramesLostIn.MANUAL_RECORDING,
            ),
            'Video Frames Dropped',
            '2 of 9 frame(s) could not be written; 3 frame(s) came from the camera '
            'without their timestamps and were left out, so the saved video has gaps. '
            'Not written: No space left on device (2: frame 3, frame 4).',
        ),
        (
            VideoFramesDroppedError(
                (),
                9,
                frames_unstamped=3,
                frames_skipped=0,
                frames_lost_at_camera=0,
                lost_in=FramesLostIn.VIDEO_STEP,
            ),
            'Video Frames Dropped',
            '3 frame(s) in a video step came from the camera without their timestamps '
            'and were left out, so that video has gaps.',
        ),
        (
            VideoFramesDroppedError(
                (),
                9,
                frames_unstamped=0,
                frames_skipped=4,
                frames_lost_at_camera=None,
                lost_in=FramesLostIn.MANUAL_RECORDING,
            ),
            'Video Frames Dropped',
            '4 frame(s) were skipped because this PC could not save frames as fast as '
            'the camera sent them, so the saved video has gaps. A lower Recording max '
            'FPS in Advanced Settings avoids the skips.',
        ),
        (
            VideoFramesDroppedError(
                (),
                9,
                frames_unstamped=0,
                frames_skipped=0,
                frames_lost_at_camera=5,
                lost_in=FramesLostIn.VIDEO_STEP,
            ),
            'Video Frames Dropped',
            '5 frame(s) the camera sent during a video step never reached the '
            'recording, so that video has gaps.',
        ),
        (
            VideoFramesDroppedError(
                (),
                9,
                frames_unstamped=0,
                frames_skipped=4,
                frames_lost_at_camera=5,
                lost_in=FramesLostIn.RUN,
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
            'saved none. Restart LumaViewPro before recording again.',
        ),
        (
            RecordingFramesUnstampedError(12, protocol_step=True),
            'Recording Saved No Frames',
            'The camera sent 12 frame(s) without their timestamps, so the video step '
            'saved none. Restart LumaViewPro before recording again.',
        ),
        (
            HyperstackNotBuiltError('one channel only', 'builder_declined'),
            'Hyperstack Not Built',
            'one channel only',
        ),
    ],
    ids=[
        'stalled',
        'camera-disconnected',
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

    # The camera's watch runs only while a recording does.
    controller.start()
    controller._stop_for_camera_loss('camera_disconnected')
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, RecordingStoppedError) and fault.reason == 'camera_disconnected'
    # The recording's end, so the repeat window does not hold it back.
    assert kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}


def test_a_full_disk_is_reported_once_unsolicited(tmp_path, monkeypatch, post_processing):
    controller, scope, clock = make_controller(tmp_path, post_processing=post_processing)
    reported = _record_reports(monkeypatch, manual_recording_module)
    # The start finds room; every rolling probe finds the disk full.
    monkeypatch.setattr(
        common_utils, 'check_disk_space_ok', the_start_finds_room(lambda *_: (False, 100.0))
    )
    controller.start()
    feed_frames(scope, clock, 5, fps=10.0)
    controller._engine.wait_for_drain(timeout=10)
    finish(controller)

    stopped = [(f, kw) for f, kw in reported if isinstance(f, RecordingStoppedError)]
    ((fault, kw),) = stopped
    assert fault.reason == 'disk_space_critical'
    # The recording's end, so the repeat window does not hold it back.
    assert kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}


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
    # Told once per recording, so the repeat window does not hold it back.
    assert kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}


def test_a_hyperstack_the_disk_refused_is_not_built_in_the_systems_words(
    tmp_path, monkeypatch, post_processing
):
    controller, scope, clock = make_controller(
        tmp_path, recording_format=RecordingFormat.HYPERSTACK, post_processing=post_processing
    )
    reported = _record_reports(monkeypatch, manual_recording_module)
    full = OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    # A stand-in by design: a disk that fills at the stack's write, which the
    # simulator has no member to produce.
    class _DiskFullBuilder:
        def __init__(self, **kwargs):
            pass

        def create_single_recording_stack(self, **kwargs):
            raise full

    monkeypatch.setattr(stack_builder_module, 'StackBuilder', _DiskFullBuilder)
    controller.start()
    feed_frames(scope, clock, 3, fps=10.0)
    controller.stop()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, HyperstackNotBuiltError)
    assert fault.reason == 'hyperstack_not_written'
    assert '(No space left on device)' in str(fault) and 'log' not in str(fault)
    assert fault.__cause__ is full
    assert kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}


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
    assert isinstance(fault, HyperstackNotBuiltError) and str(fault) == 'the builder says no'
    assert fault.reason == 'builder_declined'
    # Told once per recording, so the repeat window does not hold it back.
    assert kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}


def test_frames_not_one_size_are_a_hyperstack_not_built_for_that_reason(
    tmp_path, monkeypatch, post_processing
):
    """Frames the stack cannot share are the recording's own fault, named for
    their size, not folded into the builder declining."""
    controller, scope, clock = make_controller(
        tmp_path, recording_format=RecordingFormat.HYPERSTACK, post_processing=post_processing
    )
    reported = _record_reports(monkeypatch, manual_recording_module)
    unusable = InputFramesUnusableError('Frame 2 is 640x480, not 1900x1900.', 'geometry')

    class _MixedSizesBuilder:
        # a stand-in by design: the simulated camera writes every frame of a
        # recording at one size, so frames of two sizes need the builder's
        # own refusal
        def __init__(self, **kwargs):
            pass

        def create_single_recording_stack(self, **kwargs):
            raise unusable

    monkeypatch.setattr(stack_builder_module, 'StackBuilder', _MixedSizesBuilder)
    controller.start()
    feed_frames(scope, clock, 3, fps=10.0)
    controller.stop()
    finish(controller)

    ((fault, _),) = reported
    assert isinstance(fault, HyperstackNotBuiltError)
    assert (fault.reason, str(fault)) == (
        'input_frames_not_one_size',
        'Frame 2 is 640x480, not 1900x1900.',
    )
    assert fault.__cause__ is unusable


def test_a_recording_fault_claims_what_was_kept_only_for_a_format_that_writes_frames():
    # An MP4 is one file whose index is written at its close, so no fault
    # raised before that close may say what was saved is kept.
    refusal = OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
    faults = {
        **{
            f'stopped {reason}': lambda fmt, reason=reason: RecordingStoppedError(
                reason, recording_format=fmt
            )
            for reason in ('camera_disconnected', 'camera_stalled', 'disk_space_critical')
        },
        'stopped write_failed': lambda fmt: RecordingStoppedError(
            'write_failed', recording_format=fmt, refusal=refusal
        ),
        **{
            f'{cls.__name__} step={step}': lambda fmt, cls=cls, step=step: cls(
                protocol_step=step, recording_format=fmt
            )
            for cls in (RecordingFinalizeError, VideoWriterFailedError)
            for step in (False, True)
        },
    }
    told = {
        (name, fmt): str(build(fmt)) for name, build in faults.items() for fmt in RecordingFormat
    }
    wrong = [
        (name, fmt.value, words)
        for (name, fmt), words in told.items()
        if ('What was saved before it is kept.' in words, 'kept' in words)
        != (fmt.writes_frames, fmt.writes_frames)
    ]
    assert wrong == []


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
    assert (fault.write_failures, fault.lost_in) == (1, FramesLostIn.MANUAL_RECORDING)
    # Told once per recording, so the repeat window does not hold it back.
    assert kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}


def test_a_frame_a_full_disk_refused_between_written_ones_is_told_in_its_words_and_counted(
    tmp_path, monkeypatch, post_processing
):
    # One refused frame between written ones is the recording's failed
    # write, and the recording goes on: told in the system's words, and
    # counted so in the details file, never as a frame written.
    controller, scope, clock = make_controller(
        tmp_path, recording_format=RecordingFormat.MP4, post_processing=post_processing
    )
    reported = _record_reports(monkeypatch, manual_recording_module)
    failing = refuse_packets(monkeypatch, frozenset({2, 4}))

    controller.start()
    feed_frames(scope, clock, 40, fps=10.0)
    controller.stop()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, VideoFramesDroppedError)
    assert fault.write_failures == failing.packets_refused == 2
    assert (
        f'{fault.write_failures} of {fault.frames_selected} frame(s) could not be written'
        in str(fault)
    )
    assert 'Not written: No space left on device (2: frame ' in str(fault)
    assert 'log' not in str(fault)
    (manifest_path,) = tmp_path.rglob('*_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['write_failures'] == 2
    assert manifest['frames_written'] == manifest['frames_selected'] - 2
    assert manifest['end_reason'] == 'stopped'
    assert kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}


def test_a_second_frame_in_a_row_its_save_location_refuses_stops_the_recording_told_once(
    tmp_path, monkeypatch, post_processing, caplog
):
    # A drive that went away refuses every frame after it: the second in a
    # row stops the recording, told once in the system's words, and the
    # details file is not written where the frames were just refused. An
    # MP4's file is finished only at its close, so the stop claims nothing
    # kept.
    controller, scope, clock = make_controller(
        tmp_path, recording_format=RecordingFormat.MP4, post_processing=post_processing
    )
    reported = _record_reports(monkeypatch, manual_recording_module)
    refuse_packets(monkeypatch, frozenset({2, 3}))

    controller.start()
    feed_frames(scope, clock, 40, fps=10.0)
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, RecordingStoppedError)
    assert fault.reason == 'write_failed'
    assert str(fault) == (
        'A frame could not be saved (No space left on device), so the recording was stopped.'
    )
    assert kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}
    assert controller.end_reason == 'write_failed'
    assert not list(tmp_path.rglob('*_manifest.json'))
    assert not [r for r in caplog.records if r.levelname == 'ERROR']


def test_a_frames_recording_whose_save_location_refuses_two_frames_in_a_row_says_what_was_kept(
    tmp_path, monkeypatch, post_processing
):
    # Each written frame is a finished file, so the stop says what was saved
    # is kept, and it is the one report: there is no close to fail.
    controller, scope, clock = make_controller(
        tmp_path, recording_format=RecordingFormat.FRAMES, post_processing=post_processing
    )
    reported = _record_reports(monkeypatch, manual_recording_module)
    real_write = manual_recording_module.write_frame
    calls = {'n': 0}
    counting = threading.Lock()

    def _drive_gone_after_two(*args, **kwargs):
        # The frames' writers run on several lanes at once.
        with counting:
            calls['n'] += 1
            gone = calls['n'] > 2
        if gone:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return real_write(*args, **kwargs)

    monkeypatch.setattr(manual_recording_module, 'write_frame', _drive_gone_after_two)
    controller.start()
    feed_frames(scope, clock, 40, fps=10.0)
    finish(controller)

    ((fault, kw),) = reported
    assert (type(fault), fault.reason) == (RecordingStoppedError, 'write_failed')
    assert str(fault) == (
        'A frame could not be saved (No space left on device), so the recording was '
        'stopped. What was saved before it is kept.'
    )
    assert kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}
    assert len(list(tmp_path.rglob('*.tif*'))) == 2


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
    assert (fault.frames_skipped, fault.write_failures, fault.frames_selected) == (3, 0, 5)
    assert str(fault).startswith(
        '3 frame(s) were skipped because this PC could not save frames as fast as the '
        'camera sent them'
    )
    # Told once per recording, so the repeat window does not hold it back.
    assert kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}


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
    assert fault.frames_lost_at_camera == 7
    assert '7 frame(s) the camera sent never reached the recording' in str(fault)
    # Told once per recording, so the repeat window does not hold it back.
    assert kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}


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

    step, _ = record_first_frame(sim_scope, tmp_path, monkeypatch, refused_writes=1)

    ((fault, kw),) = reported
    assert isinstance(fault, VideoFramesDroppedError)
    assert fault.lost_in is FramesLostIn.VIDEO_STEP
    assert fault.write_failures == 1
    assert 'Not written: disk said no (frame 0).' in str(fault)
    # The run's losses name each frame with its video, so a run of several
    # videos names each lost frame once.
    lost = step.frame_losses.failed_writes
    assert len(lost) == 1
    assert all(
        re.fullmatch(r'frame \d+ of \S+', frame) and cause == 'disk said no'
        for frame, cause in lost
    ), lost
    assert kw == {'solicited': False, 'category': 'Protocol', 'log_only': True}


def test_a_video_step_whose_save_location_refuses_two_frames_in_a_row_stops_the_run(
    sim_scope, tmp_path, monkeypatch
):
    # The run is stopped at the refusal, through its fatal funnel, in the
    # system's words; the step answers the run ABORTED, so it is never
    # counted captured; the frames lost are that one event, not told again.
    reported = _record_reports(monkeypatch, protocol_recording)
    answered = []
    run_blocking = protocol_recording.ProtocolVideoStep.run_blocking

    def _answered(self):
        answered.append(run_blocking(self))
        return answered[-1]

    monkeypatch.setattr(protocol_recording.ProtocolVideoStep, 'run_blocking', _answered)

    step, _ = record_first_frame(sim_scope, tmp_path, monkeypatch, refused_writes=2)

    assert answered == [protocol_recording.ABORTED]
    ((ending, category),) = [c.args for c in step._abort_run_fatal.call_args_list]
    assert (ending.status, ending.reason, category) == (RunStatus.FAILED, 'write_failed', 'FileIO')
    # The run's record cannot list the frames the refusal cut short, so the
    # words claim nothing about what was kept, whatever the step's format.
    assert ending.message == (
        'A frame of the video step could not be saved (disk said no), so the run was stopped.'
    )
    assert reported == []
    assert step.frame_losses is None


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
    # The disk fills mid-stream: frames fail to encode, which stops the
    # recording, then the close fails. The stop claims nothing about the
    # MP4, whose file is finished only at its close, so the close's failure
    # is shown after it, and the file is left where it is.
    controller, scope, clock = make_controller(
        tmp_path, recording_format=RecordingFormat.MP4, post_processing=post_processing
    )
    reported = _record_reports(monkeypatch, manual_recording_module)
    failing = fail_the_close(monkeypatch, after_packets=2)

    controller.start()
    feed_frames(scope, clock, 40, fps=10.0)
    finish(controller)

    ((stopped, stopped_kw), (unfinished, unfinished_kw)) = reported
    assert (type(stopped), stopped.reason) == (RecordingStoppedError, 'write_failed')
    assert 'kept' not in str(stopped)
    # Told once per recording, so the repeat window does not hold it back.
    assert stopped_kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}
    assert isinstance(unfinished, VideoNotFinishedError)
    assert str(unfinished.output_path) == failing[0]
    assert unfinished_kw == {'solicited': False, 'category': 'Recording', 'once_per_run': True}
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
    assert kw == {'solicited': False, 'category': 'Protocol', 'log_only': False}
    step._record_dropped_capture.assert_called_once_with(
        reason='video_finalize_failed',
        capture_time=step._start_dt,
        cause=outcome_of(fault).shown_words,
    )
    assert 'No space left on device' in outcome_of(fault).shown_words
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
    free_space_is(monkeypatch, PLENTY_MB)
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
    assert fault.frames_unstamped > 0
    assert kw == {'solicited': False, 'category': 'Protocol', 'log_only': True}


def test_a_dropped_frames_fault_carries_its_counts_on_the_wire():
    from _lumascope.rest import problems

    fault = VideoFramesDroppedError(
        (('frame 3', 'No space left on device'),),
        9,
        frames_unstamped=2,
        frames_skipped=3,
        frames_lost_at_camera=None,
        lost_in=FramesLostIn.MANUAL_RECORDING,
    )
    body = problems.answered_by_member(fault, 'req').body
    assert body['type'].endswith('video_frames_dropped')
    assert {
        k: body[k]
        for k in (
            'write_failures',
            'frames_selected',
            'frames_unstamped',
            'frames_skipped',
            'frames_lost_at_camera',
        )
    } == {
        'write_failures': 1,
        'frames_selected': 9,
        'frames_unstamped': 2,
        'frames_skipped': 3,
        'frames_lost_at_camera': None,
    }
