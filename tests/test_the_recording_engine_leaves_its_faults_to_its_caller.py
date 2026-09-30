# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The recording engine reports nothing; whoever finishes the recording does.

Before, the engine posted two popups through a sink it was handed: "Recording
Failed" when its writer died, at critical, from the writer thread, and
"Recording details not saved" when the manifest write failed. The manual
recording reported neither to anyone else, and a protocol run shown the first
also recorded a different, never-shown ending ("Video Writer Failed"), and
showed its popup before the run was stopped and the light was dark.

Now the engine records each failure on its result, chained from what failed,
and the finish reports it through the one reporter: the manual recording
unsolicited, under the category it had; a protocol run's writer death through
the run's fatal funnel, so the popup and the run's ending say the same thing.
"""

import inspect
import pathlib

import modules.manual_recording as manual_recording_module
import modules.protocol_recording as protocol_recording
import tests.test_video_writer as video_writer_tests
from modules.activity_claim import ActivityClaim
from modules.exceptions import RecordingDetailsNotSavedError, VideoWriterFailedError
from modules.notification_center import _TYPED_FAULTS
from modules.video_recording import VideoRecordingEngine
from tests.test_manual_recording_controller import feed_frames, finish, make_controller
from tests.test_video_recording_contract import make_config
from tests.video_engine_harness import FakeClock, FrameFeed, WriterStub


def _engine(tmp_path, writer=None):
    clock = FakeClock()
    engine = VideoRecordingEngine(
        write_frame=writer if writer is not None else WriterStub(tmp_path),
        claim=ActivityClaim(),
        clock=clock,
    )
    return engine, clock


def _record(engine, clock, tmp_path, *, output_dir=None, frames=10):
    engine.start(lambda: make_config(output_dir or tmp_path, fps=10, duration_s=1))
    feed = FrameFeed()
    for _ in range(frames):
        clock.advance(0.1)
        image, ts, chunk = feed.frame(clock(), with_camera_chunks=True)
        engine.ingest_frame(image, ts, chunk, fact=None)
    engine.stop('user_stop')
    assert engine.wait_for_drain(timeout=5)
    return engine.result()


def _record_reports(monkeypatch, module):
    reported = []
    monkeypatch.setattr(
        module.notifications,
        'report_outcome',
        lambda exception, **kw: reported.append((exception, kw)),
    )
    return reported


def _fail_manifest_writes(monkeypatch):
    """Make every manifest write raise, leaving every other write alone."""
    real_write_text = pathlib.Path.write_text
    refused = OSError('scripted: disk said no to the details file')

    def _write_text(self, *args, **kwargs):
        if self.name.endswith('manifest.json'):
            raise refused
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, 'write_text', _write_text)
    return refused


class _LaneKiller(BaseException):
    """Not an Exception, so the writer lane treats it as its own death."""


# ---------------------------------------------------------------------------
# The types
# ---------------------------------------------------------------------------


def test_the_writers_death_says_the_run_stopped_only_on_a_run():
    manual = VideoWriterFailedError(protocol_step=False)
    step = VideoWriterFailedError(protocol_step=True)

    assert manual.title == step.title == 'Recording Failed'
    assert str(manual) == (
        'The video writer stopped working and the recording was aborted. '
        'Frames already written are on disk; check the log for the cause.'
    )
    assert str(step) == (
        'The video writer stopped working, so the recording and the run were stopped. '
        'Frames already written are on disk; check the log for the cause.'
    )
    assert manual.reason == step.reason == 'video_writer_died'
    assert isinstance(manual, _TYPED_FAULTS)


def test_the_lost_details_file_keeps_its_words():
    fault = RecordingDetailsNotSavedError()

    assert fault.title == 'Recording details not saved'
    assert str(fault) == (
        'The video frames are safe on disk, but the recording details file '
        'could not be written. Videos built from this recording may be '
        'grayscale and use a default frame rate; check disk space and the log.'
    )
    assert isinstance(fault, _TYPED_FAULTS)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def test_the_engine_takes_no_sink_to_post_through():
    assert 'notify' not in inspect.signature(VideoRecordingEngine).parameters


def test_a_writers_death_rides_the_result(tmp_path):
    engine, clock = _engine(tmp_path, WriterStub(tmp_path, die_on_frame=2))

    result = _record(engine, clock, tmp_path)

    assert isinstance(result.writer_failure, SystemExit)
    assert result.aborted is True
    assert result.manifest_failure is None


def test_a_lost_details_file_rides_the_result_and_the_recording_stands(tmp_path):
    engine, clock = _engine(tmp_path)

    # A directory squatting on the manifest path makes the write raise
    # without touching the frames.
    (tmp_path / 'recording_manifest.json').mkdir()
    result = _record(engine, clock, tmp_path)

    assert isinstance(result.manifest_failure, OSError)
    assert result.manifest_path is None
    assert result.aborted is False
    assert result.frames_written > 0


def test_a_clean_recording_carries_no_fault(tmp_path):
    engine, clock = _engine(tmp_path)

    result = _record(engine, clock, tmp_path)

    assert result.writer_failure is None and result.manifest_failure is None
    assert result.manifest_path is not None


def test_a_recording_with_nothing_to_describe_is_not_a_lost_details_file(tmp_path):
    engine, _clock = _engine(tmp_path)

    # A start that failed recorded nothing, so no manifest is written: its
    # absence is the honest answer, not a failure to report.
    engine.start(lambda: make_config(tmp_path, fps=10, duration_s=1))
    engine.stop('start_failed')
    assert engine.wait_for_drain(timeout=5)
    result = engine.result()

    assert result.manifest_path is None
    assert result.manifest_failure is None


# ---------------------------------------------------------------------------
# The manual recording
# ---------------------------------------------------------------------------


def test_a_manual_writers_death_is_reported_once_chained(tmp_path, monkeypatch):
    controller, scope, clock = make_controller(tmp_path)
    reported = _record_reports(monkeypatch, manual_recording_module)
    killer = _LaneKiller('scripted writer death')

    def _dying_write(**kwargs):
        raise killer

    monkeypatch.setattr(manual_recording_module.image_save, 'write_video_frame', _dying_write)
    controller.start()
    feed_frames(scope, clock, 5, fps=10.0)
    controller.stop()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, VideoWriterFailedError) and fault.protocol_step is False
    assert fault.__cause__ is killer
    assert kw == {'solicited': False, 'category': 'Recording'}


def test_a_manual_recordings_lost_details_file_is_reported_once_chained(tmp_path, monkeypatch):
    controller, scope, clock = make_controller(tmp_path)
    reported = _record_reports(monkeypatch, manual_recording_module)
    refused = _fail_manifest_writes(monkeypatch)

    controller.start()
    feed_frames(scope, clock, 5, fps=10.0)
    controller.stop()
    finish(controller)

    ((fault, kw),) = reported
    assert isinstance(fault, RecordingDetailsNotSavedError)
    assert fault.__cause__ is refused
    assert kw == {'solicited': False, 'category': 'Video Recording'}


# ---------------------------------------------------------------------------
# The protocol video step
# ---------------------------------------------------------------------------


def test_a_video_steps_writers_death_ends_the_run_in_the_faults_words(tmp_path, monkeypatch):
    reported = _record_reports(monkeypatch, protocol_recording)
    killer = _LaneKiller('scripted writer death')

    def _dying_write(self, *args, **kwargs):
        raise killer

    monkeypatch.setattr(protocol_recording.ProtocolVideoStep, '_write_frame', _dying_write)

    step = video_writer_tests.TestProtocolVideoDropNotification()._run_one_frame_step(
        tmp_path, monkeypatch, write_fails=False
    )

    # Logged once with its cause, never shown by the reporter: the run's
    # fatal funnel shows the one popup, after the light is dark.
    ((fault, kw),) = reported
    assert isinstance(fault, VideoWriterFailedError) and fault.protocol_step is True
    assert fault.__cause__ is killer
    assert kw == {'solicited': False, 'category': 'Recording', 'log_only': True}
    step._abort_run_fatal.assert_called_once_with(
        'video_writer_died', 'Recording', fault.title, str(fault)
    )


def test_a_video_steps_lost_details_file_is_reported_once_chained(tmp_path, monkeypatch):
    reported = _record_reports(monkeypatch, protocol_recording)
    refused = _fail_manifest_writes(monkeypatch)

    step = video_writer_tests.TestProtocolVideoDropNotification()._run_one_frame_step(
        tmp_path, monkeypatch, write_fails=False
    )

    ((fault, kw),) = reported
    assert isinstance(fault, RecordingDetailsNotSavedError)
    assert fault.__cause__ is refused
    assert kw == {'solicited': False, 'category': 'Video Recording'}
    step._abort_run_fatal.assert_not_called()
