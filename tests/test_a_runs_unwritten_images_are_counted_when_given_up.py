# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's images that never reach the disk are counted, never cleared silently.

Two things may give up on a finished run's images: the stalled-writer
recovery, and a shutdown that cannot wait. Before, recovery cleared the
lane's queue -- whatever was on it, whoever's it was -- and shutdown's
executor teardown cleared it with one INFO line. Now each gives up on the
run's own outstanding writes, counts them, and nothing that reads the run's
folder as whole -- the composite merge, the hyperstack build, the
post-processing auto-run -- runs on it.

A write handed to a run whose writes have ended is refused, and its caller
says what was not saved: an autofocus data save arriving after its run's
cleanup no longer lands in the next run.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

import modules.protocol_image_writer as protocol_image_writer
import modules.scope_session as scope_session
from modules.exceptions import FileWriterNotStuckError, RunFilesNotWrittenError
from modules.protocol_image_writer import RunWriteBatch
from tests.af_drives import af_runner_and_scope, drive_af
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 30.0


def _hold_saves(monkeypatch):
    release = threading.Event()
    started = threading.Semaphore(0)
    real_save = protocol_image_writer.save_image

    def _held(scope, **kwargs):
        started.release()
        release.wait(WAIT_S)
        return real_save(scope, **kwargs)

    monkeypatch.setattr(protocol_image_writer, 'save_image', _held)
    return release, started, real_save


def _finish_one_run(runner, run_parent, name, callbacks=None):
    outcome = runner.run_single_scan(
        protocol=_protocol([_step(name, 0, x=20.0, gain=1.0)]),
        parent_dir=str(run_parent),
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
        callbacks=callbacks,
    )
    result = outcome.wait(timeout_s=WAIT_S)
    assert result is not None and result.status == 'completed', result
    assert runner.sequenced_capture_runner.wait_for_run_idle(WAIT_S)


class TestRecoveringAStuckWriter:
    def test_it_gives_up_on_the_stuck_runs_images_and_the_next_run_writes_all_of_its(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(protocol_image_writer, 'WRITE_STALL_FATAL_S', 0.3)
        release, started, real_save = _hold_saves(monkeypatch)
        run_parent = tmp_path / 'runs'
        files = []
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            _finish_one_run(
                runner,
                run_parent,
                'C1',
                {'files_complete': lambda **kw: files.append(kw['files'])},
            )
            assert started.acquire(timeout=WAIT_S)
            time.sleep(0.5)
            assert session.protocol_files_stalled

            abandoned = session.recover_file_writer()

            assert abandoned == 1
            assert not session.protocol_files_draining
            deadline = time.monotonic() + WAIT_S
            while not files and time.monotonic() < deadline:
                time.sleep(0.02)
            assert files == ['abandoned']

            monkeypatch.setattr(protocol_image_writer, 'save_image', real_save)
            _finish_one_run(runner, run_parent, 'C2')
            deadline = time.monotonic() + WAIT_S
            while not any(run_parent.rglob('C2_*.tiff')) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert any(run_parent.rglob('C2_*.tiff')), "the next run's image was not written"

            release.set()
            time.sleep(0.3)
            assert files == ['abandoned'], 'the stuck write returning counted again'

    def test_it_is_refused_while_the_writer_is_making_progress(self, tmp_path, monkeypatch):
        release, started, _real_save = _hold_saves(monkeypatch)
        run_parent = tmp_path / 'runs'
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            _finish_one_run(runner, run_parent, 'C1')
            assert started.acquire(timeout=WAIT_S)

            with pytest.raises(FileWriterNotStuckError) as refused:
                session.recover_file_writer()

            assert refused.value.reason == 'file_writer_not_stuck'
            assert refused.value.pending == 1
            release.set()


class TestShuttingDownWithImagesStillWriting:
    def test_the_wait_is_bounded_and_what_is_left_is_counted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(scope_session, '_SHUTDOWN_RUN_FILES_WAIT_S', 0.3)
        release, started, _real_save = _hold_saves(monkeypatch)
        run_parent = tmp_path / 'runs'
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            _finish_one_run(runner, run_parent, 'C1')
            assert started.acquire(timeout=WAIT_S)
            batch = runner.sequenced_capture_runner.write_batch()
            warned = []
            monkeypatch.setattr(
                protocol_image_writer.logger, 'warning', lambda msg, *a, **k: warned.append(msg)
            )

            began = time.monotonic()
            session.shutdown()
            took = time.monotonic() - began

            assert batch.outcome == 'abandoned'
            assert any("Session shutdown: 1 of the run's write(s) abandoned" in m for m in warned)
            assert took < 10.0, f'shutdown waited {took:.1f} s on a stuck write'
            release.set()

    def test_a_write_that_lands_inside_the_wait_is_written(self, tmp_path, monkeypatch):
        """A finished run's write still in flight gets its chance: shutdown
        waits for it rather than giving it up at once."""
        release, started, _real_save = _hold_saves(monkeypatch)
        run_parent = tmp_path / 'runs'
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            _finish_one_run(runner, run_parent, 'C1')
            assert started.acquire(timeout=WAIT_S)
            batch = runner.sequenced_capture_runner.write_batch()
            assert batch.draining
            threading.Timer(0.3, release.set).start()

            session.shutdown()

            assert batch.outcome == 'written'
            assert any(run_parent.rglob('C1_*.tiff'))

    def test_a_run_still_live_is_given_up_on_at_once(self, tmp_path, monkeypatch):
        """A live run has not closed its writes, so nothing can complete them
        while shutdown waits; the wait would buy nothing but delay."""
        monkeypatch.setattr(protocol_image_writer, 'WRITE_BACKLOG_BOUND', 1)
        release, started, _real_save = _hold_saves(monkeypatch)
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            runner.run_single_scan(
                protocol=_protocol([_step(f'C{i}', i, x=20.0, gain=1.0) for i in range(4)]),
                parent_dir=str(tmp_path / 'runs'),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
            )
            assert started.acquire(timeout=WAIT_S)
            batch = runner.sequenced_capture_runner.write_batch()
            assert batch.outcome is None and not batch.draining, 'the run was not live'

            began = time.monotonic()
            session.shutdown()
            took = time.monotonic() - began

            release.set()
            assert batch.outcome == 'abandoned'
            assert took < 3.0, f'shutdown waited {took:.1f} s on a run that could not finish'


class TestALateAutofocusSave:
    def test_it_is_refused_reported_and_never_blocks_the_restore(self, tmp_path, monkeypatch):
        import modules.autofocus_runner as autofocus_runner

        warned = []
        monkeypatch.setattr(
            autofocus_runner.logger, 'warning', lambda msg, *a, **k: warned.append(msg)
        )
        runner, _scope = af_runner_and_scope()
        batch = RunWriteBatch(MagicMock())
        batch.close(lambda outcome: None)

        drive_af(runner, save_results_to_file=True, results_dir=tmp_path, write_batch=batch)

        assert any(
            'Autofocus data not saved: The autofocus data was not saved' in m for m in warned
        )
        assert runner.saved_data_path() is None
        assert batch.pending == 0

    def test_a_save_without_its_runs_batch_cannot_be_asked_for(self, tmp_path):
        runner, _scope = af_runner_and_scope()

        with pytest.raises(ValueError, match="run's write batch"):
            drive_af(runner, save_results_to_file=True, results_dir=tmp_path)


class TestNothingBuildsFromAFolderMissingImages:
    def test_the_post_processing_auto_run_does_not_run(self):
        from modules.plugins import run_protocol_complete_processors

        processor = MagicMock()
        spec = MagicMock(auto_run_on_protocol_complete=True)
        ctx = MagicMock()
        ctx.plugins.post_processing.handlers.return_value = [(spec, processor)]

        run_protocol_complete_processors(
            ctx, input_dir='run', manifest={}, output_dir='run', files='abandoned'
        )

        processor.assert_not_called()

    def test_the_hyperstack_build_says_why_and_builds_nothing(self, monkeypatch, tmp_path):
        import modules.stack_builder as stack_builder
        from modules.notification_center import notifications

        reported = []
        monkeypatch.setattr(
            notifications, 'report_outcome', lambda ex, **kw: reported.append((ex, kw))
        )
        loaded = MagicMock()
        monkeypatch.setattr(stack_builder.StackBuilder, 'load_folder', loaded)

        def _abandoned():
            raise RunFilesNotWrittenError('write_batch_abandoned')

        stack_builder.build_hyperstacks_for_run(
            run_dir=tmp_path,
            has_turret=False,
            tiling_configs_file_loc=tmp_path / 'tiling.json',
            wait_for_images=_abandoned,
        )

        loaded.assert_not_called()
        [(ex, kw)] = reported
        assert ex.reason == 'write_batch_abandoned'
        assert kw['fault_title'] == 'Hyperstacks Not Saved'
