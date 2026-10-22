# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's operation says when its files are done, and what became of its images.

The run's operation is done once the run no longer holds the scope, but its
files can still be writing, and a next run is refused 'files_writing' until
they land. Nothing let a caller wait for that: it polled the Session's
drain, which answers for whichever run ran last. ``files`` is done for THIS
run's images, its hyperstack build when it has one, and the run's end, and
its result is what became of the images.
"""

import threading

import pytest

from modules import live_work
import modules.image_mode as image_mode
import modules.protocol_image_writer as protocol_image_writer
import modules.stack_builder as stack_builder
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 60.0
# Long enough that a held write or build is still held when it runs out.
SHORT_S = 0.3


def _scan(runner, run_parent, name, sequenced_format='TIFF'):
    runner.session.update_settings('image_output_format.sequenced', sequenced_format)
    return runner.start_single_scan(
        protocol=_protocol([_step(name, 0, x=20.0, gain=1.0)]),
        parent_dir=str(run_parent),
    )


@pytest.fixture
def held_saves(monkeypatch):
    """Every image save waits until the returned event is set."""
    release = threading.Event()
    real_save = protocol_image_writer.save_image

    def _save(scope, **kwargs):
        release.wait(WAIT_S)
        return real_save(scope, **kwargs)

    monkeypatch.setattr(protocol_image_writer, 'save_image', _save)
    yield release
    release.set()


def test_the_files_wait_returns_once_the_images_land_and_the_next_run_is_admitted(
    tmp_path, held_saves
):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        first = _scan(runner, tmp_path / 'runs', 'C1')
        assert first.result(timeout=WAIT_S) is not None
        # The positive control: the Session's drain poll sees the held write.
        assert session.live_work.has(live_work.RUN_FILES), 'the held save did not hold the drain'

        assert not first.files.done(), (
            "the run's files were done while an image was still unwritten"
        )

        held_saves.set()
        files = first.files.result(timeout=WAIT_S)
        assert files is not None
        assert files.outcome == 'written'
        assert files.written >= 1
        assert files.not_written == 0
        assert files.not_written_reason is None

        second = _scan(runner, tmp_path / 'runs', 'C2')
        assert second.result(timeout=WAIT_S) is not None


def test_an_image_that_failed_to_save_is_counted(tmp_path, monkeypatch):
    def _fail(scope, **kwargs):
        raise OSError('the save drive went away')

    monkeypatch.setattr(protocol_image_writer, 'save_image', _fail)
    with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
        run = _scan(runner, tmp_path / 'runs', 'C1')
        files = run.files.result(timeout=WAIT_S)

    assert files is not None
    assert files.outcome == 'incomplete'
    assert files.not_written >= 1
    assert files.not_written_reason == 'write_batch_save_failed'


def test_a_hyperstack_run_waits_for_its_build(tmp_path, monkeypatch):
    built = threading.Event()
    release = threading.Event()
    real_load = stack_builder.StackBuilder.load_folder

    def _held_load(self, *args, **kwargs):
        # The build ends however its folder answers; a scan this short may
        # have nothing to stack.
        release.wait(WAIT_S)
        try:
            return real_load(self, *args, **kwargs)
        finally:
            built.set()

    monkeypatch.setattr(stack_builder.StackBuilder, 'load_folder', _held_load)
    try:
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            run = _scan(
                runner,
                tmp_path / 'runs',
                'C1',
                sequenced_format=image_mode.OUTPUT_FORMAT_HYPERSTACK,
            )
            assert run.result(timeout=WAIT_S) is not None
            assert session.sequenced_capture_runner.write_batch().wait_complete(WAIT_S)
            # The positive control: the images are down and the build is held.
            assert not built.is_set()

            assert not run.files.done(), "the run's files were done before its build finished"

            release.set()
            files = run.files.result(timeout=WAIT_S)
            assert files is not None
            assert built.is_set()
    finally:
        release.set()
