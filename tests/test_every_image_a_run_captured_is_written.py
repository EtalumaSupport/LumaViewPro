# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every image a run captured is written, and the run's files are told once.

The file lane used to hold a run's writes in a lane-wide run mode, inferring
from its own queues when "the run's writes" were done. The inference was
wrong in ways that lost images while the run reported ``completed``: a
worker waiting on its default queue when a short run started and ended read
the empty wait as drained and discarded the run's image; an error abort
cleared every write still queued; the end-of-files notice was one lane slot,
fired for the wrong run or not at all.

Each run now owns its writes. These tests drive real headless runs on the
simulator, hold writes where timing would otherwise decide, and read the
disk and the callbacks.
"""

import queue
import threading
import time

import pytest

import modules.protocol_image_writer as protocol_image_writer
import modules.protocol_run_loop as protocol_run_loop
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 30.0


def _images(run_parent):
    return sorted(run_parent.rglob('*.tiff'))


def _record_rows(run_parent):
    return [
        line
        for record in run_parent.rglob('protocol_record.tsv')
        for line in record.read_text().splitlines()
        if '.tiff\t' in line
    ]


class _Callbacks:
    """run_complete and files_complete, recorded in the order they arrive."""

    def __init__(self):
        self.events = []
        self.files_done = threading.Event()

    def as_dict(self):
        return {
            'run_complete': lambda **kw: self.events.append(('run_complete', kw.get('run_dir'))),
            'files_complete': self._files_complete,
        }

    def _files_complete(self, **kw):
        self.events.append(('files_complete', kw.get('run_dir'), kw.get('files')))
        self.files_done.set()


def _hold_saves(monkeypatch):
    """Hold every image save until released; returns (release, started)."""
    release = threading.Event()
    started = threading.Semaphore(0)
    real_save = protocol_image_writer.save_image

    def _held(scope, **kwargs):
        started.release()
        assert release.wait(WAIT_S), 'a held save was never released'
        return real_save(scope, **kwargs)

    monkeypatch.setattr(protocol_image_writer, 'save_image', _held)
    return release, started


class TestAShortRunWhileTheFileWorkerWaits:
    def test_the_one_image_is_written(self, tmp_path):
        """The file worker parked in its default-queue wait for the whole of
        a one-image run: the run's image and its record row reach the disk.
        (Before, the worker woke to a run already ended, took the empty wait
        as the run drained, and discarded the image; the run said
        ``completed``.)"""
        run_parent = tmp_path / 'runs'
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            lane_queue = session.file_io_executor.queue
            real_get = lane_queue.get
            parked = threading.Event()
            release = threading.Event()
            armed = [True]

            def _parked_get(*args, **kwargs):
                if armed[0]:
                    armed[0] = False
                    parked.set()
                    release.wait(WAIT_S)
                    raise queue.Empty
                return real_get(*args, **kwargs)

            lane_queue.get = _parked_get
            try:
                assert parked.wait(WAIT_S), 'the file worker never came back to its wait'
                outcome = runner.run_single_scan(
                    protocol=_protocol([_step('C2', 0, x=20.0, gain=1.0)]),
                    parent_dir=str(run_parent),
                    image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                )
                result = outcome.wait(timeout_s=WAIT_S)
                assert result is not None and result.status == 'completed', result
                assert runner.sequenced_capture_runner.wait_for_run_idle(WAIT_S)
            finally:
                release.set()

            # The image and then its row: wait for both, bounded, since the
            # loss this pins is silent -- nothing ever arrives.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                images = _images(run_parent)
                rows = _record_rows(run_parent)
                if images and rows:
                    break
                time.sleep(0.05)

        assert len(images) == 1, f'the run reported completed and wrote {len(images)} image(s)'
        assert len(rows) == 1, f'the record holds {len(rows)} row(s)'


class TestAnErrorAbort:
    def test_every_captured_image_is_written(self, tmp_path, monkeypatch):
        """The scope reported disconnected after both steps captured, with
        their writes held: the run ends failed, and both images are on
        disk. (Before, an error abort cleared the writes still queued.)"""
        run_parent = tmp_path / 'runs'
        release, _started = _hold_saves(monkeypatch)
        captured = []
        disconnected = threading.Event()
        real_capture = protocol_image_writer.ProtocolImageWriter.capture

        def _counted(writer, *args, **kwargs):
            result = real_capture(writer, *args, **kwargs)
            captured.append(kwargs['step']['Name'])
            if len(captured) == 2:
                disconnected.set()
            return result

        monkeypatch.setattr(protocol_image_writer.ProtocolImageWriter, 'capture', _counted)
        monkeypatch.setattr(protocol_run_loop, 'HW_CHECK_INTERVAL_S', -1)
        callbacks = _Callbacks()
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            real_connected = session.scope.are_all_connected
            monkeypatch.setattr(
                session.scope,
                'are_all_connected',
                lambda: False if disconnected.is_set() else real_connected(),
            )
            outcome = runner.run_single_scan(
                protocol=_protocol(
                    [_step('B3', 0, x=20.0, gain=1.0), _step('B1', 1, x=60.0, gain=1.0)]
                ),
                parent_dir=str(run_parent),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks=callbacks.as_dict(),
            )
            result = outcome.wait(timeout_s=WAIT_S)
            assert result is not None and result.status == 'failed', result
            assert result.reason == 'hardware_disconnected', result
            assert runner.sequenced_capture_runner.wait_for_run_idle(WAIT_S)
            release.set()
            assert callbacks.files_done.wait(WAIT_S), 'files_complete never came'

        names = sorted(p.name for p in _images(run_parent))
        assert len(names) == 2, f'an error abort lost captured images: on disk {names}'
        assert callbacks.events[-1][2] == 'written'

    def test_the_runs_record_is_reconciled_once_its_images_land(self, tmp_path, monkeypatch):
        """A run that ends failed still has every image it captured written,
        so its record is reconciled against them -- once, after the last
        write lands and before the files are reported done -- and a
        shortfall is never left unreported for the ending it had."""
        from modules.protocol_execution_record import ProtocolExecutionRecord

        run_parent = tmp_path / 'runs'
        release, _started = _hold_saves(monkeypatch)
        disconnected = threading.Event()
        real_capture = protocol_image_writer.ProtocolImageWriter.capture

        def _disconnect_after(writer, *args, **kwargs):
            result = real_capture(writer, *args, **kwargs)
            disconnected.set()
            return result

        monkeypatch.setattr(protocol_image_writer.ProtocolImageWriter, 'capture', _disconnect_after)
        monkeypatch.setattr(protocol_run_loop, 'HW_CHECK_INTERVAL_S', -1)
        callbacks = _Callbacks()
        real_complete = ProtocolExecutionRecord.complete

        def _recorded_complete(record):
            callbacks.events.append(('record_complete', len(_images(run_parent))))
            return real_complete(record)

        monkeypatch.setattr(ProtocolExecutionRecord, 'complete', _recorded_complete)
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            real_connected = session.scope.are_all_connected
            monkeypatch.setattr(
                session.scope,
                'are_all_connected',
                lambda: False if disconnected.is_set() else real_connected(),
            )
            outcome = runner.run_single_scan(
                protocol=_protocol(
                    [_step('B3', 0, x=20.0, gain=1.0), _step('B1', 1, x=60.0, gain=1.0)]
                ),
                parent_dir=str(run_parent),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks=callbacks.as_dict(),
            )
            result = outcome.wait(timeout_s=WAIT_S)
            assert result is not None and result.status == 'failed', result
            assert runner.sequenced_capture_runner.wait_for_run_idle(WAIT_S)
            assert not any(e[0] == 'record_complete' for e in callbacks.events), (
                'the record was reconciled with a write still held'
            )
            release.set()
            assert callbacks.files_done.wait(WAIT_S), 'files_complete never came'

        ends = [e for e in callbacks.events if e[0] in ('record_complete', 'files_complete')]
        assert [e[0] for e in ends] == ['record_complete', 'files_complete'], callbacks.events
        assert ends[0][1] >= 1, 'the record was reconciled before its image landed'


class TestBackToBackRuns:
    def test_each_run_hears_its_own_files_once_after_its_run_complete(self, tmp_path):
        run_parent = tmp_path / 'runs'
        heard = []
        with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
            engine = runner.sequenced_capture_runner
            for index in range(3):
                callbacks = _Callbacks()
                outcome = runner.run_single_scan(
                    protocol=_protocol([_step(f'C{index}', 0, x=20.0, gain=1.0)]),
                    parent_dir=str(run_parent),
                    image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                    callbacks=callbacks.as_dict(),
                )
                assert outcome.wait(timeout_s=WAIT_S) is not None
                assert callbacks.files_done.wait(WAIT_S), f'run {index} never heard files_complete'
                assert engine.wait_for_run_idle(WAIT_S)
                heard.append(callbacks)

            time.sleep(0.3)

        run_dirs = []
        for index, callbacks in enumerate(heard):
            kinds = [event[0] for event in callbacks.events]
            assert kinds == ['run_complete', 'files_complete'], (
                f'run {index} heard {kinds}; each run hears run_complete, then its files, once'
            )
            run_dir = callbacks.events[0][1]
            assert callbacks.events[1][1] == run_dir, (
                f'run {index} heard its files for another run directory'
            )
            assert run_dir is not None and any(run_dir.rglob(f'C{index}_*.tiff')), (
                f'run {index} was handed a directory without its own image'
            )
            run_dirs.append(run_dir)
        assert len(set(run_dirs)) == 3


class TestANewRunWhileTheLastRunsFilesWrite:
    def test_it_is_refused_until_they_land_and_admitted_the_moment_they_do(
        self, tmp_path, monkeypatch
    ):
        from modules.exceptions import ProtocolRunRefusedError

        run_parent = tmp_path / 'runs'
        release, started = _hold_saves(monkeypatch)
        callbacks = _Callbacks()
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            engine = runner.sequenced_capture_runner
            outcome = runner.run_single_scan(
                protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
                parent_dir=str(run_parent),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks=callbacks.as_dict(),
            )
            assert outcome.wait(timeout_s=WAIT_S) is not None
            # Read the moment the outcome releases its caller: an L2 script
            # waits on the outcome and then asks whether the disk has
            # settled, and must not be told it has.
            assert session.protocol_files_draining, (
                'the outcome released its caller before the run closed its writes'
            )
            assert engine.wait_for_run_idle(WAIT_S)
            assert started.acquire(timeout=WAIT_S)

            assert session.protocol_files_draining
            assert session.protocol_files_pending == 1
            with pytest.raises(ProtocolRunRefusedError) as refused:
                runner.run_single_scan(
                    protocol=_protocol([_step('C2', 0, x=20.0, gain=1.0)]),
                    parent_dir=str(run_parent),
                    image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                )
            assert refused.value.reason == 'files_writing'

            release.set()
            assert callbacks.files_done.wait(WAIT_S)
            assert not session.protocol_files_draining
            second = runner.run_single_scan(
                protocol=_protocol([_step('C3', 0, x=20.0, gain=1.0)]),
                parent_dir=str(run_parent),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
            )
            assert second.wait(timeout_s=WAIT_S).status == 'completed'


class TestTheRunsWritesCloseBeforeItEnds:
    def test_the_batch_is_closed_when_the_outcome_settles_and_when_the_run_goes_idle(
        self, tmp_path, monkeypatch
    ):
        """A caller released by the outcome, and a next run's prepare()
        admitted at IDLE, read the finished run's batch; each must find it
        closed -- draining or done -- never open and not yet asked. Read
        at the instant of each edge, since the window is too short to catch
        from outside."""
        from modules.sequenced_capture_runner import ProtocolState

        release, started = _hold_saves(monkeypatch)
        seen = {}
        with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
            engine = runner.sequenced_capture_runner

            def _closed():
                batch = engine.write_batch()
                return batch.draining or batch.outcome is not None

            real_settle = engine._settle_run_outcome
            real_set_state = engine._set_state

            def _settle(ending):
                seen['settled'] = _closed()
                return real_settle(ending)

            def _set_state(new_state):
                if new_state is ProtocolState.IDLE:
                    seen['idle'] = _closed()
                return real_set_state(new_state)

            monkeypatch.setattr(engine, '_settle_run_outcome', _settle)
            monkeypatch.setattr(engine, '_set_state', _set_state)
            outcome = runner.run_single_scan(
                protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
                parent_dir=str(tmp_path / 'runs'),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
            )
            assert outcome.wait(timeout_s=WAIT_S) is not None
            assert engine.wait_for_run_idle(WAIT_S)
            assert started.acquire(timeout=WAIT_S)
            release.set()

        assert seen == {'settled': True, 'idle': True}, (
            f'the run ended with its writes still open: {seen}'
        )


class TestACleanupThatRaises:
    def test_run_complete_then_files_complete_each_once_and_the_next_run_is_admitted(
        self, tmp_path, monkeypatch
    ):
        """A cleanup that raised before sending run_complete used to leave it
        unsent -- and the run's auto-run with it."""
        import modules.sequenced_capture_runner as scr

        def _raises(**kwargs):
            raise RuntimeError('cleanup fell over')

        monkeypatch.setattr(scr, 'run_cleanup', _raises)
        run_parent = tmp_path / 'runs'
        callbacks = _Callbacks()
        with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
            engine = runner.sequenced_capture_runner
            runner.run_single_scan(
                protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
                parent_dir=str(run_parent),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks=callbacks.as_dict(),
            )
            assert callbacks.files_done.wait(WAIT_S), 'files_complete never came'
            assert engine.wait_for_run_idle(WAIT_S)
            time.sleep(0.3)
            monkeypatch.undo()
            second = runner.run_single_scan(
                protocol=_protocol([_step('C2', 0, x=20.0, gain=1.0)]),
                parent_dir=str(run_parent),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
            )
            assert second.wait(timeout_s=WAIT_S).status == 'completed'

        assert [event[0] for event in callbacks.events] == ['run_complete', 'files_complete']


class TestAStartThatFails:
    @pytest.mark.parametrize('where', ['the run folder', 'the dispatch'])
    def test_its_files_are_reported_once(self, tmp_path, monkeypatch, where):
        """A start that fails before the image writer exists wrote nothing,
        and its files are still reported done exactly once: whoever waits on
        files_complete for a run is never left waiting because the run
        never reached its first capture."""
        from concurrent.futures import Future

        callbacks = _Callbacks()
        with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
            engine = runner.sequenced_capture_runner
            if where == 'the run folder':

                def _no_folder():
                    raise OSError('save folder vanished')

                monkeypatch.setattr(engine, '_setup_run_dir', _no_folder)
            else:

                def _refused(*args, **kwargs):
                    future = Future()
                    future.set_exception(RuntimeError('Protocol already in progress'))
                    return future

                monkeypatch.setattr(engine.protocol_thread, 'run_protocol', _refused)
            outcome = runner.run_single_scan(
                protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
                parent_dir=str(tmp_path / 'runs'),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks=callbacks.as_dict(),
            )
            result = outcome.wait(timeout_s=WAIT_S)
            assert result is not None and result.status == 'failed_at_start', result
            assert engine.wait_for_run_idle(WAIT_S)
            assert callbacks.files_done.wait(WAIT_S), 'files_complete never came'
            time.sleep(0.3)

        files = [event for event in callbacks.events if event[0] == 'files_complete']
        assert len(files) == 1, callbacks.events
        assert files[0][2] == 'written'


class TestARunAfterAnErrorAbort:
    def test_every_write_of_the_next_run_runs(self, tmp_path, monkeypatch):
        """An error abort ends only its own run's writes: the next run's
        writes, held on the lane behind its first, all reach the disk."""
        run_parent = tmp_path / 'runs'
        disconnected = threading.Event()
        real_capture = protocol_image_writer.ProtocolImageWriter.capture

        def _disconnect_after(writer, *args, **kwargs):
            result = real_capture(writer, *args, **kwargs)
            if kwargs['step']['Name'].startswith('A'):
                disconnected.set()
            return result

        monkeypatch.setattr(protocol_image_writer.ProtocolImageWriter, 'capture', _disconnect_after)
        monkeypatch.setattr(protocol_run_loop, 'HW_CHECK_INTERVAL_S', -1)
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            engine = runner.sequenced_capture_runner
            real_connected = session.scope.are_all_connected
            monkeypatch.setattr(
                session.scope,
                'are_all_connected',
                lambda: False if disconnected.is_set() else real_connected(),
            )
            first_files = _Callbacks()
            first = runner.run_single_scan(
                protocol=_protocol(
                    [_step('A1', 0, x=20.0, gain=1.0), _step('A2', 1, x=60.0, gain=1.0)]
                ),
                parent_dir=str(run_parent / 'first'),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks=first_files.as_dict(),
            )
            assert first.wait(timeout_s=WAIT_S).status == 'failed'
            assert engine.wait_for_run_idle(WAIT_S)
            assert first_files.files_done.wait(WAIT_S)
            disconnected.clear()

            release, started = _hold_saves(monkeypatch)
            second_files = _Callbacks()
            second = runner.run_single_scan(
                protocol=_protocol(
                    [
                        _step('B1', 0, x=20.0, gain=1.0),
                        _step('B2', 1, x=60.0, gain=1.0),
                        _step('B3', 2, x=20.0, gain=2.0),
                    ]
                ),
                parent_dir=str(run_parent / 'second'),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks=second_files.as_dict(),
            )
            assert started.acquire(timeout=WAIT_S), "the second run's first write never started"
            assert second.wait(timeout_s=WAIT_S).status == 'completed'
            release.set()
            assert second_files.files_done.wait(WAIT_S), 'files_complete never came'

        names = sorted(p.name.split('_')[0] for p in _images(run_parent / 'second'))
        assert names == ['B1', 'B2', 'B3'], f'the second run wrote {names}'
        assert second_files.events[-1][2] == 'written'


class TestAWriteWhoseFailureReportRaises:
    def test_the_runs_files_are_still_reported_done(self, tmp_path, monkeypatch):
        """A save that fails, and whose failure report then fails too, still
        brings the run's count down: a count left up would hold every later
        run and build open."""
        run_parent = tmp_path / 'runs'
        callbacks = _Callbacks()

        def _save_fails(scope, **kwargs):
            raise OSError('the drive went away')

        monkeypatch.setattr(protocol_image_writer, 'save_image', _save_fails)
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            lane = session.file_io_executor
            reports = []

            def _report_fails(*args, **kwargs):
                reports.append(args)
                raise RuntimeError('reporting the failure failed')

            monkeypatch.setattr(lane, '_report_task_failure', _report_fails)
            outcome = runner.run_single_scan(
                protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
                parent_dir=str(run_parent),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks=callbacks.as_dict(),
            )
            assert outcome.wait(timeout_s=WAIT_S) is not None
            assert runner.sequenced_capture_runner.wait_for_run_idle(WAIT_S)
            assert callbacks.files_done.wait(WAIT_S), 'the failed write left the batch owing it'
            assert not session.protocol_files_draining
        assert reports, "the save's failure was never reported"
