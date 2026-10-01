# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run counts its own still-image writes to disk.

A post-run step that reads the run's captured frames back off disk -- the
composite merge -- must wait for those frames to actually land. The only
drain signal that existed was the file executor's global protocol-queue
predicate, which answers for whatever is in the queue rather than for a
particular run. Waiting on it couples one run's post-step to the NEXT
run's writes: whichever run is filling the queue holds the previous run's
merge open.

The count is therefore per run: every write the writer hands over is
counted in the run's own write batch, built with the run and handed to the
run's writer. "This run's files are written" is the batch's completion --
closed by the run's cleanup, with nothing outstanding -- never a count that
reaches zero between two writes while the run can still submit.
"""

import threading
from unittest.mock import MagicMock

import pytest

from tests.protocol_drives import lent_run_claim
from tests.frame_records import plate
import modules.protocol_image_writer as piw
from modules.exceptions import RunWriteRefusedError
from modules.image_mode import ImageCaptureConfig
from modules.protocol_image_writer import ProtocolImageWriter, RunWriteBatch
from tests.scope_fakes import spec_scope


from modules.run_outcome import EndingLatch


def _writer(file_io_executor=None):
    return ProtocolImageWriter(
        scope=spec_scope(),
        callbacks=MagicMock(),
        aborted=threading.Event(),
        write_batch=RunWriteBatch(file_io_executor or MagicMock()),
        abort_fn=MagicMock(),
        fatal_abort_event=threading.Event(),
        ending=EndingLatch(),
        execution_record=MagicMock(),
        leds_off_fn=MagicMock(),
        is_run_in_progress_fn=lambda: True,
        image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
        timestamp_overlay=False,
        video_max_fps=0,
        engineering_mode=False,
        run_claim=lent_run_claim(),
        labware=plate(),
        captures_asked=1,
    )


def _submit(writer, **overrides):
    """Push one write through the single enqueue owner."""
    kwargs = {
        'kwargs': {},
        'step': {'Color': 'BF'},
        'step_index': 0,
        'scan_count': 0,
        'capture_time': None,
        'name': 'A1_BF',
    }
    kwargs.update(overrides)
    return writer._submit_write(**kwargs)


def _owed(writer):
    """What the writer's run still owes the disk."""
    return writer._write_batch.pending


def _close(writer):
    """The run's cleanup ending its writes."""
    writer._write_batch.close(lambda outcome: None)


class TestStillPendingWrites:
    def test_a_fresh_writer_owes_nothing(self):
        assert _owed(_writer()) == 0

    def test_a_submitted_write_is_owed_until_it_runs(self):
        executor = MagicMock()
        # The executor accepts the task but never runs it, which is exactly
        # the state the counter has to make visible.
        executor.put.return_value = object()
        writer = _writer(executor)

        _submit(writer)

        assert _owed(writer) == 1, (
            'a write handed to the executor but not yet on disk must be owed; '
            'a post-run step reading the run directory would otherwise find a '
            'frame missing'
        )

    def test_running_the_write_settles_the_debt(self):
        executor = MagicMock()
        captured = {}

        def _accept(task, **kwargs):
            captured['task'] = task
            return object()

        executor.put.side_effect = _accept
        writer = _writer(executor)
        writer.write_capture = MagicMock(name='write_capture')

        _submit(writer)
        assert _owed(writer) == 1

        # Run the task the way the executor's worker would.
        captured['task'].action(**captured['task'].kwargs)

        assert _owed(writer) == 0

    def test_a_write_that_raises_still_settles_the_debt(self):
        executor = MagicMock()
        captured = {}
        executor.put.side_effect = lambda task, **kw: captured.setdefault('task', task)
        writer = _writer(executor)
        writer.write_capture = MagicMock(side_effect=OSError('save drive vanished'))

        _submit(writer)
        with pytest.raises(OSError):
            captured['task'].action(**captured['task'].kwargs)

        assert _owed(writer) == 0, (
            'a failed write must not leave a permanent debt; the merge would '
            'wait out its whole bound for a frame that will never arrive'
        )

    @pytest.mark.parametrize(
        'refusal',
        [None, 'wedged'],
        ids=['declined_submit', 'wedged_queue'],
    )
    def test_a_write_the_executor_never_took_is_not_owed(self, refusal, monkeypatch):
        # Neither refusal ever runs the task, so neither can settle its own
        # debt. A declined submit is a write handed over after the run's
        # cleanup ended its writes: it is refused, never taken. A wedged
        # writer is a full backlog with the write in flight stuck past the
        # stall budget: the frame is not taken, and it is counted abandoned
        # -- the run's files then end abandoned, never reported written.
        executor = MagicMock()
        executor.put.return_value = object()
        writer = _writer(executor)
        writer._abort_run_fatal = MagicMock()
        if refusal == 'wedged':
            # A backlog with no room at all, and a write in flight past a
            # stall budget already spent, so the paced submit declares the
            # wedge at its first look.
            monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', 0)
            monkeypatch.setattr(piw, 'WRITE_STALL_FATAL_S', 0.0)
            executor.in_flight_task_stalled.return_value = True
            _submit(writer)
        else:
            _close(writer)
            with pytest.raises(RunWriteRefusedError) as refused:
                _submit(writer)
            assert refused.value.reason == 'run_ended'

        assert _owed(writer) == 0, (
            'a write the executor never took cannot be owed, or the run ends '
            'holding a debt nothing will ever settle'
        )
        if refusal == 'wedged':
            _close(writer)
        assert writer._write_batch.wait_complete(timeout_s=0.1) is True
        assert writer._write_batch.outcome == ('incomplete' if refusal == 'wedged' else 'written')

    def test_each_run_counts_only_its_own_writes(self):
        executor = MagicMock()
        executor.put.return_value = object()
        first = _writer(executor)
        _submit(first)

        second = _writer(executor)

        assert _owed(second) == 0, (
            "a new run's writer must start clear; sharing the count is what "
            "couples one run's post-step to the next run's writes"
        )
        assert _owed(first) == 1


class TestWaitForStillWrites:
    """The wait is for the batch's completion: the run's cleanup has closed it
    and nothing is outstanding. A count reaching zero between two writes
    while the run can still submit is not "written"."""

    def test_returns_true_when_nothing_is_owed(self):
        writer = _writer()
        _close(writer)
        assert writer._write_batch.wait_complete(timeout_s=0.1) is True

    def test_returns_false_when_the_debt_outlives_the_bound(self):
        executor = MagicMock()
        executor.put.return_value = object()
        writer = _writer(executor)
        _submit(writer)
        _close(writer)

        assert writer._write_batch.wait_complete(timeout_s=0.05) is False, (
            'the wait is bounded: a wedged writer must not hold a post-run step open forever'
        )

    def test_returns_true_once_the_write_lands(self):
        executor = MagicMock()
        captured = {}
        executor.put.side_effect = lambda task, **kw: captured.setdefault('task', task)
        writer = _writer(executor)
        writer.write_capture = MagicMock()
        _submit(writer)
        _close(writer)

        def _drain():
            captured['task'].action(**captured['task'].kwargs)

        threading.Timer(0.05, _drain).start()

        assert writer._write_batch.wait_complete(timeout_s=5.0) is True
