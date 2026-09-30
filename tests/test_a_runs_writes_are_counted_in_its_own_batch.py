# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's writes are counted in the run's own batch, from hand-over to landing.

The file lane serves everyone, so it cannot say which writes are a run's or
when THAT run's writes are done; asked, it answered for whatever was queued,
and the answer drifted across runs: a one-image run's image discarded while
the run reported ``completed``, a completion notice delivered to the wrong
run or not at all, a count that never came back to zero after a raise.

The batch is the run's own account. These tests hold writes on events where
timing would otherwise decide, and read the account: every write handed over
either runs or is counted as abandoned, and completion happens exactly once,
after the run closed its writes and the last one landed.
"""

import threading
import time

import pytest

import modules.protocol_image_writer as piw
from modules.exceptions import RunFilesNotWrittenError, RunWriteRefusedError
from modules.protocol_image_writer import WRITER_WEDGED, RunWriteBatch
from modules.sequential_io_executor import SequentialIOExecutor

HELD_S = 5.0


@pytest.fixture
def lane():
    executor = SequentialIOExecutor(name='TEST_FILE')
    executor.start()
    try:
        yield executor
    finally:
        executor.shutdown(wait=False)


class _Held:
    """A write that runs only when released, and says when it started."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.ran = False

    def __call__(self):
        self.started.set()
        assert self.release.wait(HELD_S), 'a held write was never released'
        self.ran = True


def _close(batch):
    """Close the batch; returns the outcomes its completion saw, and when."""
    seen = []
    done = threading.Event()

    def _on_complete(outcome):
        seen.append(outcome)
        done.set()

    batch.close(_on_complete)
    return seen, done


class TestCompletion:
    def test_it_waits_for_the_close_and_the_last_write(self, lane):
        batch = RunWriteBatch(lane)
        held = _Held()
        batch.submit(held, {}, what='The image', pace_until=None)
        assert held.started.wait(HELD_S)

        seen, done = _close(batch)
        assert not done.wait(0.2), 'completed with a write still in flight'
        assert batch.draining

        held.release.set()
        assert done.wait(HELD_S)
        assert seen == ['written']
        assert not batch.draining

    def test_a_run_that_wrote_nothing_completes_at_its_close(self, lane):
        seen, done = _close(RunWriteBatch(lane))

        assert done.is_set()
        assert seen == ['written']

    @pytest.mark.parametrize('raising', ['the write', 'its failure report'])
    def test_a_write_that_raises_still_leaves_the_count(self, lane, raising):
        """A raise anywhere in a write's task still brings the count down:
        a count that stayed up held every later build and run open."""

        def _write():
            if raising == 'the write':
                raise OSError('the drive went away')
            try:
                raise OSError('the drive went away')
            except OSError:
                raise RuntimeError('reporting the failure failed') from None

        batch = RunWriteBatch(lane)
        batch.submit(_write, {}, what='The image', pace_until=None)
        seen, done = _close(batch)

        assert done.wait(HELD_S), 'a raising write left the batch owing it'
        assert seen == ['written']

    def test_completion_happens_once(self, lane):
        batch = RunWriteBatch(lane)
        writes = [_Held() for _ in range(3)]
        for write in writes:
            batch.submit(write, {}, what='The image', pace_until=None)
        seen, done = _close(batch)
        for write in writes:
            write.release.set()

        assert done.wait(HELD_S)
        time.sleep(0.1)
        assert seen == ['written']

    def test_a_write_after_the_close_is_refused(self, lane):
        batch = RunWriteBatch(lane)
        _close(batch)

        with pytest.raises(RunWriteRefusedError) as refused:
            batch.submit(lambda: None, {}, what='The autofocus data', pace_until=None)

        assert refused.value.reason == 'run_ended'
        assert str(refused.value).startswith('The autofocus data was not saved')


class TestPacing:
    def test_a_full_backlog_waits_for_room(self, lane, monkeypatch):
        monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', 1)
        batch = RunWriteBatch(lane)
        first = _Held()
        batch.submit(first, {}, what='The image', pace_until=lambda: False)
        assert first.started.wait(HELD_S)

        taken = threading.Event()
        threading.Thread(
            target=lambda: (
                batch.submit(lambda: None, {}, what='The image', pace_until=lambda: False),
                taken.set(),
            ),
            daemon=True,
        ).start()

        assert not taken.wait(0.6), 'the second write was taken over a full backlog'
        first.release.set()
        assert taken.wait(HELD_S)

    def test_an_abort_hands_a_waiting_frame_over_instead_of_dropping_it(self, lane, monkeypatch):
        """A frame the run captured is written however the run ends: the
        abort ends the wait, not the frame."""
        monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', 1)
        batch = RunWriteBatch(lane)
        first = _Held()
        batch.submit(first, {}, what='The image', pace_until=lambda: False)
        assert first.started.wait(HELD_S)
        aborted = threading.Event()
        second = _Held()
        second.release.set()
        taken = threading.Event()
        threading.Thread(
            target=lambda: (
                batch.submit(second, {}, what='The image', pace_until=aborted.is_set),
                taken.set(),
            ),
            daemon=True,
        ).start()
        assert not taken.wait(0.4)

        aborted.set()
        assert taken.wait(HELD_S), 'the abort did not end the wait'
        first.release.set()
        seen, done = _close(batch)

        assert done.wait(HELD_S)
        assert second.ran, 'the frame waiting at the abort was dropped'
        assert seen == ['written']

    def test_an_unpaced_write_never_waits(self, lane, monkeypatch):
        """The autofocus data save must not delay autofocus's restore of the
        LED, camera and Z, so it goes over the bound rather than wait."""
        monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', 1)
        batch = RunWriteBatch(lane)
        first = _Held()
        batch.submit(first, {}, what='The image', pace_until=lambda: False)
        assert first.started.wait(HELD_S)

        started = time.monotonic()
        batch.submit(lambda: None, {}, what='The autofocus data', pace_until=None)

        assert time.monotonic() - started < 0.2
        assert batch.pending == 2
        first.release.set()

    def test_a_stuck_writer_refuses_the_waiting_frame_and_counts_it_lost(self, lane, monkeypatch):
        monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', 1)
        monkeypatch.setattr(piw, 'WRITE_STALL_FATAL_S', 0.3)
        batch = RunWriteBatch(lane)
        stuck = _Held()
        batch.submit(stuck, {}, what='The image', pace_until=lambda: False)
        assert stuck.started.wait(HELD_S)

        result = batch.submit(lambda: None, {}, what='The image', pace_until=lambda: False)

        assert result is WRITER_WEDGED
        stuck.release.set()
        seen, done = _close(batch)
        assert done.wait(HELD_S)
        assert seen == ['abandoned'], 'a frame the stuck writer refused was not counted'


class TestAbandon:
    def test_every_outstanding_write_is_counted_once(self, lane):
        batch = RunWriteBatch(lane)
        stuck = _Held()
        queued = _Held()
        queued.release.set()
        batch.submit(stuck, {}, what='The image', pace_until=None)
        batch.submit(queued, {}, what='The image', pace_until=None)
        assert stuck.started.wait(HELD_S)

        abandoned = batch.abandon('test recovery')
        seen, done = _close(batch)

        assert abandoned == 2
        assert done.is_set(), 'an abandoned, closed batch did not complete'
        assert seen == ['abandoned']
        stuck.release.set()
        time.sleep(0.2)
        assert not queued.ran, 'a write given up on still ran when its turn came'
        assert seen == ['abandoned'], 'the stuck write returning counted again'

    def test_a_complete_batch_abandons_nothing(self, lane):
        batch = RunWriteBatch(lane)
        _close(batch)

        assert batch.abandon('test recovery') == 0
        assert batch.outcome == 'written'

    def test_a_write_after_the_abandon_is_refused(self, lane):
        batch = RunWriteBatch(lane)
        batch.abandon('test shutdown')

        with pytest.raises(RunWriteRefusedError) as refused:
            batch.submit(lambda: None, {}, what='The image', pace_until=None)

        assert refused.value.reason == 'writes_abandoned'


class TestABuildWaitingOnTheWrites:
    def test_it_is_told_when_the_bound_expires(self, lane):
        batch = RunWriteBatch(lane)
        held = _Held()
        batch.submit(held, {}, what='The image', pace_until=None)
        _close(batch)

        with pytest.raises(RunFilesNotWrittenError) as not_written:
            batch.wait_until_written(0.1)

        assert not_written.value.reason == 'write_batch_timeout'
        held.release.set()

    def test_it_is_told_when_writes_were_abandoned(self, lane):
        batch = RunWriteBatch(lane)
        held = _Held()
        batch.submit(held, {}, what='The image', pace_until=None)
        batch.abandon('test recovery')
        _close(batch)

        with pytest.raises(RunFilesNotWrittenError) as not_written:
            batch.wait_until_written(HELD_S)

        assert not_written.value.reason == 'write_batch_abandoned'
        held.release.set()

    def test_it_returns_once_every_write_landed(self, lane):
        batch = RunWriteBatch(lane)
        held = _Held()
        batch.submit(held, {}, what='The image', pace_until=None)
        _close(batch)
        held.release.set()

        batch.wait_until_written(HELD_S)

        assert held.ran

    def test_it_is_told_a_frame_the_stuck_writer_refused_was_never_taken(self, lane, monkeypatch):
        """A frame a paced submit found the writer stuck on never reached
        the writer; no recovery ran and nothing shut down, so the build is
        not told either happened."""
        monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', 1)
        monkeypatch.setattr(piw, 'WRITE_STALL_FATAL_S', 0.3)
        batch = RunWriteBatch(lane)
        stuck = _Held()
        batch.submit(stuck, {}, what='The image', pace_until=lambda: False)
        assert stuck.started.wait(HELD_S)
        assert (
            batch.submit(lambda: None, {}, what='The image', pace_until=lambda: False)
            is WRITER_WEDGED
        )
        stuck.release.set()
        _close(batch)

        with pytest.raises(RunFilesNotWrittenError) as not_written:
            batch.wait_until_written(HELD_S)

        assert not_written.value.reason == 'write_batch_not_taken'
        message = str(not_written.value)
        assert 'recovered' not in message
        assert 'shut down' not in message

    def test_it_is_told_a_write_the_shut_lane_refused_was_never_taken(self, lane):
        lane.shutdown(wait=False)
        batch = RunWriteBatch(lane)

        with pytest.raises(RunWriteRefusedError) as refused:
            batch.submit(lambda: None, {}, what='The image', pace_until=None)
        assert refused.value.reason == 'writer_shut_down'
        assert batch.pending == 0
        _close(batch)

        with pytest.raises(RunFilesNotWrittenError) as not_written:
            batch.wait_until_written(HELD_S)

        assert not_written.value.reason == 'write_batch_not_taken'

    def test_a_write_given_up_on_before_the_lane_refused_it_is_not_said_to_have_run(
        self, lane, monkeypatch
    ):
        """A shutdown can give up on a write between its hand-over and the
        lane refusing it. That write never ran, so the log must not say it
        finished late and may be on disk."""
        batch = RunWriteBatch(lane)

        def _abandon_then_refuse(task, return_future=False):
            batch.abandon('test shutdown')
            return None

        monkeypatch.setattr(lane, 'put', _abandon_then_refuse)
        warnings = []
        monkeypatch.setattr(
            piw.logger, 'warning', lambda message, *a, **k: warnings.append(message)
        )

        with pytest.raises(RunWriteRefusedError):
            batch.submit(lambda: None, {}, what='The image', pace_until=None)

        assert batch.pending == 0
        assert not [w for w in warnings if 'finished after its run gave up' in w], warnings

    def test_an_abandon_after_a_refused_frame_is_still_told_as_abandoned(self, lane, monkeypatch):
        """When a recovery or shutdown also gave up on writes, that is what
        the build is told, even if the writer had refused a frame first."""
        monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', 1)
        monkeypatch.setattr(piw, 'WRITE_STALL_FATAL_S', 0.3)
        batch = RunWriteBatch(lane)
        stuck = _Held()
        batch.submit(stuck, {}, what='The image', pace_until=lambda: False)
        assert stuck.started.wait(HELD_S)
        assert (
            batch.submit(lambda: None, {}, what='The image', pace_until=lambda: False)
            is WRITER_WEDGED
        )
        batch.abandon('test recovery')
        _close(batch)

        with pytest.raises(RunFilesNotWrittenError) as not_written:
            batch.wait_until_written(HELD_S)

        assert not_written.value.reason == 'write_batch_abandoned'
        stuck.release.set()
