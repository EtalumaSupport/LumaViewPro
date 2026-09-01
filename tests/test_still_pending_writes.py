# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run counts its own still-image writes to disk.

A post-run step that reads the run's captured frames back off disk -- the
composite merge -- must wait for those frames to actually land. The only
drain signal that existed was the file executor's global protocol-queue
predicate, which answers for whatever is in the queue rather than for a
particular run. Waiting on it couples one run's post-step to the NEXT
run's writes: whichever run is filling the queue holds the previous run's
merge open.

The counter is therefore per writer, and a writer is built per run. Video
steps already carry their own pending-write counters for the same reason;
still images had none, so this mirrors that shipped shape rather than
inventing a second one.
"""

import threading
from unittest.mock import MagicMock

import pytest

from modules.image_mode import ImageCaptureConfig
from modules.protocol_image_writer import ProtocolImageWriter


def _writer(file_io_executor=None):
    return ProtocolImageWriter(
        scope=MagicMock(),
        callbacks=MagicMock(),
        aborted=threading.Event(),
        file_io_executor=file_io_executor or MagicMock(),
        abort_fn=MagicMock(),
        fatal_abort_event=threading.Event(),
        execution_record=MagicMock(),
        leds_off_fn=MagicMock(),
        is_run_in_progress_fn=lambda: True,
        image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
        timestamp_overlay=False,
        video_max_fps=0,
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


class TestStillPendingWrites:
    def test_a_fresh_writer_owes_nothing(self):
        assert _writer().still_pending_writes == 0

    def test_a_submitted_write_is_owed_until_it_runs(self):
        executor = MagicMock()
        # The executor accepts the task but never runs it, which is exactly
        # the state the counter has to make visible.
        executor.protocol_put_wait.return_value = object()
        writer = _writer(executor)

        _submit(writer)

        assert writer.still_pending_writes == 1, (
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

        executor.protocol_put_wait.side_effect = _accept
        writer = _writer(executor)
        writer.write_capture = MagicMock(name='write_capture')

        _submit(writer)
        assert writer.still_pending_writes == 1

        # Run the task the way the executor's worker would.
        captured['task'].action(**captured['task'].kwargs)

        assert writer.still_pending_writes == 0

    def test_a_write_that_raises_still_settles_the_debt(self):
        executor = MagicMock()
        captured = {}
        executor.protocol_put_wait.side_effect = lambda task, **kw: captured.setdefault(
            'task', task
        )
        writer = _writer(executor)
        writer.write_capture = MagicMock(side_effect=OSError('save drive vanished'))

        _submit(writer)
        with pytest.raises(OSError):
            captured['task'].action(**captured['task'].kwargs)

        assert writer.still_pending_writes == 0, (
            'a failed write must not leave a permanent debt; the merge would '
            'wait out its whole bound for a frame that will never arrive'
        )

    @pytest.mark.parametrize(
        'refusal',
        [None, 'wedged'],
        ids=['declined_submit', 'wedged_queue'],
    )
    def test_a_write_the_executor_never_took_is_not_owed(self, refusal):
        # A refused submit (executor disabled, or no run in session) returns
        # a bare None, the same shape as a wait cancelled by abort. Neither
        # ever runs the task, so neither can settle its own debt.
        from modules.sequential_io_executor import PROTOCOL_QUEUE_WEDGED

        executor = MagicMock()
        executor.protocol_put_wait.return_value = (
            PROTOCOL_QUEUE_WEDGED if refusal == 'wedged' else None
        )
        writer = _writer(executor)
        writer._abort_run_fatal = MagicMock()

        _submit(writer)

        assert writer.still_pending_writes == 0, (
            'a write the executor never took cannot be owed, or the run ends '
            'holding a debt nothing will ever settle'
        )
        assert writer.wait_for_still_writes(timeout_s=0.1) is True

    def test_each_run_counts_only_its_own_writes(self):
        executor = MagicMock()
        executor.protocol_put_wait.return_value = object()
        first = _writer(executor)
        _submit(first)

        second = _writer(executor)

        assert second.still_pending_writes == 0, (
            "a new run's writer must start clear; sharing the count is what "
            "couples one run's post-step to the next run's writes"
        )
        assert first.still_pending_writes == 1


class TestWaitForStillWrites:
    def test_returns_true_when_nothing_is_owed(self):
        assert _writer().wait_for_still_writes(timeout_s=0.1) is True

    def test_returns_false_when_the_debt_outlives_the_bound(self):
        executor = MagicMock()
        executor.protocol_put_wait.return_value = object()
        writer = _writer(executor)
        _submit(writer)

        assert writer.wait_for_still_writes(timeout_s=0.05) is False, (
            'the wait is bounded: a wedged writer must not hold a post-run step open forever'
        )

    def test_returns_true_once_the_write_lands(self):
        executor = MagicMock()
        captured = {}
        executor.protocol_put_wait.side_effect = lambda task, **kw: captured.setdefault(
            'task', task
        )
        writer = _writer(executor)
        writer.write_capture = MagicMock()
        _submit(writer)

        def _drain():
            captured['task'].action(**captured['task'].kwargs)

        threading.Timer(0.05, _drain).start()

        assert writer.wait_for_still_writes(timeout_s=5.0) is True
