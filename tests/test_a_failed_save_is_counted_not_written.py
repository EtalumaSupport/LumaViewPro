# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's image whose save fails on disk is counted not written.

The run's write batch settled every write that ran, whether it landed or
raised, so a run whose save failed completed ``written``: its
``files_written`` said so, and the composite merge, the hyperstack build
and the post-processing auto-run all read the folder as whole while an
image was missing from it. A write that raises is now counted not written,
the run's files end ``incomplete``, and a build refused on it says the save
failed.
"""

import pathlib
import threading
import time

import pytest

import modules.protocol_image_writer as protocol_image_writer
from modules.exceptions import CaptureError, RunFilesNotWrittenError
from modules.protocol_image_writer import RunWriteBatch
from modules.run_events import RunEvents
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 30.0


@pytest.fixture
def lane():
    from modules.sequential_io_executor import SequentialIOExecutor

    executor = SequentialIOExecutor(name='FILE_TEST')
    executor.start()
    yield executor
    executor.shutdown()


def _closed(batch):
    seen = []
    batch.close()
    batch.when_complete(seen.append)
    return seen


def _fail_saves(monkeypatch, failing=lambda kwargs: True):
    real_save = protocol_image_writer.save_image

    def _save(scope, **kwargs):
        if failing(kwargs):
            raise OSError('the save drive went away')
        return real_save(scope, **kwargs)

    monkeypatch.setattr(protocol_image_writer, 'save_image', _save)


def _wait_for(seen):
    deadline = time.monotonic() + WAIT_S
    while not seen and time.monotonic() < deadline:
        time.sleep(0.02)


class TestTheBatch:
    def test_a_write_that_raises_leaves_the_runs_files_incomplete(self, lane):
        def _write():
            raise OSError('the save drive went away')

        batch = RunWriteBatch(lane)
        batch.submit(_write, {}, what='The image', pace_until=None)
        seen = _closed(batch)

        assert batch.wait_complete(WAIT_S)
        assert seen == ['incomplete']
        assert batch.outcome == 'incomplete'
        with pytest.raises(RunFilesNotWrittenError) as not_written:
            batch.wait_until_written(0.1)
        assert not_written.value.reason == 'write_batch_save_failed'

    def test_a_writer_given_up_on_is_named_ahead_of_a_failed_save(self, lane):
        """A dying disk can fail a save and then stall the writer. A run with
        more than one cause names one, in a fixed order: given up on, never
        taken, failed to save."""
        release = threading.Event()

        def _fails():
            raise OSError('the save drive went away')

        def _stuck():
            release.wait(WAIT_S)

        batch = RunWriteBatch(lane)
        batch.submit(_fails, {}, what='The first image', pace_until=None)
        batch.submit(_stuck, {}, what='The second image', pace_until=None)
        seen = _closed(batch)
        batch.abandon('Writer recovery')
        release.set()

        assert batch.wait_complete(WAIT_S)
        assert seen == ['incomplete']
        with pytest.raises(RunFilesNotWrittenError) as not_written:
            batch.wait_until_written(0.1)
        assert not_written.value.reason == 'write_batch_abandoned'


class TestARunWhoseSaveFails:
    def test_its_files_are_told_incomplete(self, tmp_path, monkeypatch):
        _fail_saves(monkeypatch)
        files = []
        with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
            outcome = runner.run_single_scan(
                protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
                parent_dir=str(tmp_path / 'runs'),
                events=RunEvents(files_written=lambda _run_dir, files_: files.append(files_)),
            )
            assert outcome.wait(timeout_s=WAIT_S) is not None
            _wait_for(files)

            assert files == ['incomplete'], (
                'a run with a failed save was told its files were written'
            )

    def test_its_composite_is_not_built_and_says_the_save_failed(self, tmp_path, monkeypatch):
        _fail_saves(monkeypatch, failing=lambda kwargs: kwargs['channel'] == 'Blue')
        with (
            open_composite_session(headless_settings(tmp_path)) as (_session, runner),
            pytest.raises(CaptureError) as failed,
        ):
            runner.run_composite(sequence_name='e2e', parent_dir=str(tmp_path))

        assert failed.value.reason == 'write_batch_save_failed'
        # The run's frames are named for the composite too, so the merged
        # artifact shows as an image beyond the one frame that saved.
        images = sorted(p.name for p in pathlib.Path(tmp_path).rglob('*.tif*'))
        assert len(images) == 1 and '_BF_' in images[0], (
            f'expected only the BF frame; a composite was built from a folder '
            f'missing a channel: {images}'
        )
