# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Session run-state facts and the records derived from them.

Each run-state FACT has exactly one owner -- the activity claim
(arbitration), the recording engine (live-vs-drain), the file writer
(pending work) -- and the Session's records are synchronous derivations
over them: ``controls_lock`` and ``live_work`` (each state's answer:
``test_each_run_state_is_read_in_the_records.py``).

The drain terms encode the documented asymmetry: a draining recording
HOLDS its claim while the controls free; a finished protocol FREES its
claim while the controls stay locked until its write batch completes.
Transitions notify level-read listeners (they re-read the records when
they fire, so out-of-order delivery degrades to bounded staleness, never
a permanently wrong publish).
"""

from unittest.mock import MagicMock

import pytest

from modules import live_work
from tests.scope_fakes import session_on_a_simulated_scope
from tests.protocol_drives import run_identity


def _held_file_lane():
    """A file lane that runs nothing it is given until the test runs it.

    A stand-in by design: the subject is what the session's records say
    while a run's writes are still to land, and a real lane would land them
    before the test could read the records.
    """
    return MagicMock()


def _draining_run(session, writes=1):
    """The runner's last run, ended with ``writes`` of its files still to land."""
    from modules.protocol_image_writer import RunWriteBatch

    batch = RunWriteBatch(session.file_io_executor)
    for _ in range(writes):
        batch.submit(lambda: None, {}, what='an image', pace_until=None)
    session.sequenced_capture_runner._write_batch = batch
    batch.close()
    return batch


class TestDerivations:
    def test_draining_recording_frees_controls_while_claim_refuses(self, monkeypatch):
        # The recording drain window: claim held (new runs refuse), but
        # capturing is over so the control surface frees.
        session = session_on_a_simulated_scope()
        held = session.activity_claim.try_claim('recording')
        assert held
        monkeypatch.setattr(
            session.manual_recording, '_engine', MagicMock(is_recording=False, is_draining=True)
        )
        assert session.live_work.has(live_work.WorkKind.RECORDING), (
            'the draining recording let go of its claim'
        )
        assert session.controls_lock.controls_locked is False
        held.release()

    @pytest.mark.slow
    def test_the_pending_count_is_the_write_batchs_own(self):
        session = session_on_a_simulated_scope(file_io_executor=_held_file_lane())
        batch = _draining_run(session, writes=7)
        assert [item.left for item in session.live_work.work] == [7]
        batch.abandon('the test is over')

    def test_a_live_runs_writes_are_not_counted_as_draining(self):
        """While the run is live its own state answers; its writes held on
        the lane are not a finished run's files still to land."""
        from modules.protocol_image_writer import RunWriteBatch

        session = session_on_a_simulated_scope(file_io_executor=_held_file_lane())
        batch = RunWriteBatch(session.file_io_executor)
        batch.submit(lambda: None, {}, what='an image', pace_until=None)
        session.sequenced_capture_runner._write_batch = batch
        assert batch.pending == 1

        assert session.live_work.work == ()
        assert session.controls_lock.run_lockout is False
        batch.abandon('the test is over')
        # Planted as a live run's: end it as the run's cleanup would.
        batch.close()

    @pytest.mark.slow
    def test_a_stalled_drain_is_judged_by_the_run_refusals_threshold(self):
        # One threshold for "stuck": the display of a stalled writer and
        # the refusal of a new run over it must not disagree.
        from modules.protocol_image_writer import WRITE_STALL_FATAL_S

        session = session_on_a_simulated_scope(file_io_executor=_held_file_lane())
        batch = _draining_run(session)
        executor = session.file_io_executor
        executor.in_flight_task_stalled.return_value = True
        assert session.protocol_files_stalled is True
        executor.in_flight_task_stalled.assert_called_once_with(WRITE_STALL_FATAL_S)
        batch.abandon('the test is over')

    def test_the_close_discard_drops_both_video_drain_sources(self, monkeypatch):
        session = session_on_a_simulated_scope()
        recording_engine = MagicMock()
        monkeypatch.setattr(session.manual_recording, '_engine', recording_engine)
        monkeypatch.setattr(session, 'sequenced_capture_runner', MagicMock())

        session.discard_close_drain()

        recording_engine.discard_pending.assert_called_once_with()
        session.sequenced_capture_runner.discard_video_pending.assert_called_once_with()


class TestTransitionNotification:
    def test_claim_grant_and_release_notify(self):
        session = session_on_a_simulated_scope()
        fired = []
        session._run_state_listeners.append(lambda: fired.append(True))
        held = session.activity_claim.try_claim('protocol', run=run_identity())
        assert held
        assert len(fired) == 1
        held.release()
        assert len(fired) == 2

    def test_registration_level_syncs_immediately(self):
        # Transitions are edges; a listener registered after a grant
        # must still see current truth, so registration republishes.
        session = session_on_a_simulated_scope()
        fired = []
        session.add_run_state_listener(lambda: fired.append(True))
        assert fired, 'registration must invoke the listener once (level sync)'

    def test_the_write_batchs_completion_is_one_run_state_edge(self):
        # The drain's end is a run-state change: the run lockout drops with it,
        # so the Session must hear it -- once, when the run's last write
        # lands, not when the run closes its writes.
        from modules.protocol_image_writer import RunWriteBatch
        from modules.run_events import RunEvents

        session = session_on_a_simulated_scope(file_io_executor=_held_file_lane())
        runner = session.sequenced_capture_runner
        # The run's fields start() sets that the completion captures.
        runner._events = RunEvents()
        runner._disable_saving_artifacts = True
        batch = RunWriteBatch(session.file_io_executor)
        runner._write_batch = batch
        batch.submit(lambda: None, {}, what='an image', pace_until=None)
        (task,) = [c.args[0] for c in session.file_io_executor.put.call_args_list]
        fired = []
        session._run_state_listeners.append(lambda: fired.append(session.controls_lock.run_lockout))

        files_written = runner._close_run_writes(batch, MagicMock(), MagicMock())
        # The run's end hands the batch its completion, after the release.
        batch.when_complete(files_written)
        assert fired == [], 'closing with a write outstanding is not the drain ending'
        assert session.controls_lock.run_lockout is True

        task.action(**task.kwargs)

        assert batch.outcome == 'written'
        assert fired == [False], (
            f'the last write landing must reach the run-state edge once, unlocked; got {fired}'
        )

    def test_failed_claim_does_not_notify(self):
        session = session_on_a_simulated_scope()
        held = session.activity_claim.try_claim('protocol', run=run_identity())
        assert held
        fired = []
        session._run_state_listeners.append(lambda: fired.append(True))
        assert not session.activity_claim.try_claim('recording')
        assert not fired, 'a refused claim is not a transition'
        held.release()

    def test_listener_exception_does_not_break_others(self):
        session = session_on_a_simulated_scope()
        fired = []
        session._run_state_listeners.append(lambda: (_ for _ in ()).throw(RuntimeError('x')))
        session._run_state_listeners.append(lambda: fired.append(True))
        session.notify_run_state()
        assert fired
