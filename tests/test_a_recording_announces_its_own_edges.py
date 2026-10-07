# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A recording announces each of its run-state edges through the claim.

The Session hears run state through the claim's listener, and a recording
changes that state three times the claim's own grant and release do not
cover: it goes live after the grant, its selection closes (live to
draining) with the claim still held, and its finish ends after the release.
A listener told only of the grant and the release reads a recording that is
not yet live, and a recording still finishing after it has finished; before
these edges the GUI polled to cover them, which a headless caller cannot.
"""

import threading

import pytest

from modules.activity_claim import ActivityClaim
import modules.manual_recording as manual_recording_module
from modules.manual_recording import ManualRecordingController
from modules.video_recording import VideoRecordingEngine
from tests.test_manual_recording_controller import (
    _capture_engine_and_writer,
    _FakeScope,
    feed_frames,
    finish,
    make_controller,
    make_settings,
)
from tests.test_video_recording_contract import make_config
from tests.video_engine_harness import FakeClock, FrameFeed, ManualFireScheduler, WriterStub
from tests.protocol_drives import run_identity


def _engine_heard_by(tmp_path, claim_for, *, blocked=False):
    """An engine whose claim's listener records (is_recording, is_draining)."""
    heard = []
    engine = None

    def _listener():
        # A lent claim's run is taken before the engine exists, and that
        # taking is heard too; only the engine's own edges are recorded.
        if engine is not None:
            heard.append((engine.is_recording, engine.is_draining))

    claim = claim_for(_listener)
    writer = WriterStub(tmp_path, blocked=blocked)
    clock = FakeClock()
    engine = VideoRecordingEngine(write_frame=writer, claim=claim, clock=clock)
    return engine, writer, clock, heard


def _own_claim(listener):
    return ActivityClaim(on_transition=listener)


def _lent_run_claim(listener):
    run = ActivityClaim(on_transition=listener).try_claim('protocol', run=run_identity('scan'))
    return run.lend()


@pytest.mark.parametrize('claim_for', [_own_claim, _lent_run_claim], ids=['own', 'lent'])
class TestTheEngine:
    def test_a_listener_hears_it_go_live(self, tmp_path, claim_for):
        engine, _writer, _clock, heard = _engine_heard_by(tmp_path, claim_for)

        engine.start(lambda: make_config(tmp_path))
        try:
            assert (True, True) in heard, (
                f'no listener was told the recording is live; heard {heard}'
            )
        finally:
            engine.stop('user_stop')
            assert engine.wait_for_drain(5.0)

    def test_a_listener_hears_it_go_to_draining(self, tmp_path, claim_for):
        engine, writer, clock, heard = _engine_heard_by(tmp_path, claim_for, blocked=True)
        engine.start(lambda: make_config(tmp_path, fps=None))
        clock.advance(0.1)
        image, ts, chunks = FrameFeed().frame(clock())
        engine.ingest_frame(image, ts, chunks, fact=None)

        engine.stop('user_stop')
        try:
            assert (False, True) in heard, (
                f'no listener was told the recording stopped selecting and is draining; '
                f'heard {heard}'
            )
        finally:
            writer.unblock()
            assert engine.wait_for_drain(5.0)


class TestTheController:
    def test_the_last_thing_a_listener_hears_is_the_finish_ended(self, tmp_path):
        # The claim is released when the drain ends, while the finish (the
        # encoder's close, the hyperstack) still runs; the finish's own end
        # is the edge after which nothing about the recording is busy.
        heard = []
        controller = None

        def _listener():
            heard.append(controller.is_busy)

        scope = _FakeScope()
        clock = FakeClock()
        controller = ManualRecordingController(
            scope=scope,
            settings=make_settings(tmp_path),
            activity_claim=ActivityClaim(on_transition=_listener),
            scheduler=ManualFireScheduler(),
            clock=clock,
        )
        controller.start()
        feed_frames(scope, clock, 3)
        controller.stop()
        finish(controller)

        assert heard, 'no listener heard the recording at all'
        assert heard[-1] is False, (
            f'the last edge a listener heard still read the recording busy; heard {heard}'
        )

    def test_a_finish_that_could_not_start_leaves_nothing_busy(self, tmp_path, monkeypatch):
        # Thread exhaustion raises at the finish thread's start(), after the
        # controller has marked its finish pending. Nothing will ever run to
        # end that finish, so the failed start must: a finish left pending
        # refuses every later recording and holds the application's close.
        made = _capture_engine_and_writer(monkeypatch)
        real_thread = threading.Thread

        class _Unstartable(real_thread):
            def start(self):
                raise RuntimeError("can't start new thread")

        def _thread(*args, **kwargs):
            if kwargs.get('name') == 'ManualRecordingFinish':
                return _Unstartable(*args, **kwargs)
            return real_thread(*args, **kwargs)

        monkeypatch.setattr(manual_recording_module.threading, 'Thread', _thread)
        controller, _scope, _clock = make_controller(tmp_path)

        with pytest.raises(RuntimeError):
            controller.start()

        assert made['engine'].wait_for_drain(timeout=5)
        assert not controller.is_busy, 'a finish that never started is still pending'
