"""A recording that produced no file must not announce one.

Bench evidence (2026-08-17): 19 rapid start/stop cycles at a 250 ms
exposure produced 17 recordings that captured zero frames. The log
announced ``Video written to <path>`` for every one of them and none of
those paths existed -- FFmpeg's mp4 muxer writes neither header nor
trailer for an empty stream, so no file is created at any point.

The cost is real: counting recordings by grepping ``Video written to``
returned 19 where the true answer was 2, and log-based triage is where
every bench diagnosis on this project starts.

These drive the real controller through the existing recording harness
and assert on the calls the code actually made. ``caplog`` cannot see
them: production modules do ``from lvp_logger import logger`` and
conftest replaces that module with a MagicMock, so the records never
reach the logging framework. Reading the mock's call list is what makes
these assertions real rather than vacuous -- a first draft used caplog
and passed nothing but empty lists.
"""

from __future__ import annotations

import pytest

import modules.manual_recording as manual_recording_module
import modules.video_writer as video_writer_module
from tests.test_manual_recording_controller import (
    feed_frames,
    finish,
    make_controller,
)


@pytest.fixture(autouse=True)
def _healthy_disk(monkeypatch):
    # conftest mocks psutil, so the real probe returns MagicMocks and the
    # start refusal compares one against an int. The sibling controller
    # tests carry the same fixture; importing their helpers does not
    # import it.
    monkeypatch.setattr(
        manual_recording_module, 'check_disk_space_ok', lambda *_: (True, 1_000_000.0)
    )


@pytest.fixture(autouse=True)
def _fresh_log_calls():
    """The mocked logger is module-level and accumulates across tests."""
    manual_recording_module.logger.reset_mock()
    video_writer_module.logger.reset_mock()
    yield


def _logged(module, *needles):
    """Messages the module passed to logger.info / logger.warning."""
    calls = list(module.logger.info.call_args_list) + list(module.logger.warning.call_args_list)
    out = []
    for call in calls:
        if not call.args:
            continue
        message = str(call.args[0])
        if any(n in message for n in needles):
            out.append(message)
    return out


def _announcements():
    return _logged(manual_recording_module, 'Video written to', 'No video written')


class TestEmptyRecordingAnnouncesNoFile:
    def test_zero_frames_leaves_no_mp4(self, tmp_path):
        """The premise: an empty recording produces nothing on disk.

        If this fails the muxer changed, and the announcement logic needs
        revisiting -- an empty-but-present file would want cleanup, not
        merely a different log line.
        """
        controller, _scope, _clock = make_controller(tmp_path, video_as_frames=False)
        controller.start()
        controller.stop()
        finish(controller)

        assert list((tmp_path / 'Manual').glob('Video_*.mp4')) == []

    def test_zero_frames_does_not_announce_a_path(self, tmp_path):
        """The defect: no line may name a file that was never created."""
        controller, _scope, _clock = make_controller(tmp_path, video_as_frames=False)
        controller.start()
        controller.stop()
        finish(controller)

        said = _announcements()
        assert said, 'the controller logged no outcome at all; assertions below would be vacuous'
        assert not any('Video written to' in m for m in said), (
            f'an empty recording announced an output path: {said!r} -- that path '
            f'does not exist, and grepping this line to count recordings is how '
            f'the bench run reported 19 videos for 2 real ones'
        )
        assert any('No video written' in m for m in said), (
            f'an empty recording never said it produced no file: {said!r}'
        )

    def test_frames_still_announce_their_path(self, tmp_path):
        """Behaviour preservation: a real recording still names its file."""
        controller, scope, clock = make_controller(tmp_path, video_as_frames=False)
        controller.start()
        feed_frames(scope, clock, 5, fps=10.0)
        controller.stop()
        finish(controller)

        mp4s = list((tmp_path / 'Manual').glob('Video_*.mp4'))
        assert len(mp4s) == 1

        announced = [m for m in _announcements() if 'Video written to' in m]
        assert len(announced) == 1, f'expected one announcement, got {_announcements()!r}'
        assert mp4s[0].name in announced[0], (
            f'the announcement names a different file than the one written: '
            f'{announced[0]!r} vs {mp4s[0].name}'
        )
        assert not any('No video written' in m for m in _announcements())

    @pytest.mark.parametrize('n_frames', [0, 3])
    def test_announcement_count_matches_files_on_disk(self, tmp_path, n_frames):
        """The invariant, across both outcomes.

        Parameterised so a future change cannot satisfy one direction by
        breaking the other.
        """
        controller, scope, clock = make_controller(tmp_path, video_as_frames=False)
        controller.start()
        if n_frames:
            feed_frames(scope, clock, n_frames, fps=10.0)
        controller.stop()
        finish(controller)

        on_disk = list((tmp_path / 'Manual').glob('Video_*.mp4'))
        announced = [m for m in _announcements() if 'Video written to' in m]
        assert len(announced) == len(on_disk), (
            f'{len(announced)} announcement(s) for {len(on_disk)} file(s) on disk'
        )


class TestWriterCloseMessageNamesItsRealCondition:
    """close() before the encoder opened is not the same as 'no frames'."""

    def test_unopened_encoder_message_does_not_claim_a_frame_count(self):
        """_init_pyav opens eagerly, so this branch means never-opened.

        A recording that ran and captured nothing takes the other branch
        and reports "closed (0 frames)"; describing this one in terms of
        frames pointed readers at the wrong condition.
        """
        import threading

        writer = video_writer_module.VideoWriter.__new__(video_writer_module.VideoWriter)
        writer._container = None
        writer._finished = False
        writer._frame_lock = threading.Lock()

        writer.close()

        said = _logged(video_writer_module, 'close()', 'frames')
        assert said, 'close() on an unopened writer logged nothing'
        assert any('encoder was opened' in m for m in said), (
            f'the unopened-encoder branch does not name its real condition: {said!r}'
        )
        assert not any('without adding any frames' in m for m in said), (
            f'the message still describes a frame count: {said!r}'
        )
