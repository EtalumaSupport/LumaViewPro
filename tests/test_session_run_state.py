# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Session run-state facts and derivations.

Each run-state FACT has exactly one owner -- the activity claim
(arbitration), the recording engine (live-vs-drain), the file writer
(pending work), the motion driver (XY stage) -- and every consumer
truth is a synchronous derivation over them:

    is_protocol_running = owner == 'protocol'
    run_lockout         = owner == 'protocol' or protocol_files_draining
    controls_locked     = run_lockout or (owner == 'recording' and manual_recording.is_recording)
    motion_enabled      = capabilities.has_xy_stage and not run_lockout

The drain terms encode today's documented asymmetry: a draining
recording HOLDS its claim while the controls free; a finished protocol
FREES its claim while the controls stay locked until its write batch
completes. Transitions notify level-read listeners (they re-read the
derivations when they fire, so out-of-order delivery degrades to
bounded staleness, never a permanently wrong publish).
"""

from unittest.mock import MagicMock

import pytest

from tests.scope_fakes import spec_scope
from tests.protocol_drives import run_identity


def _make_session(has_xy_stage=True):
    from modules.scope_session import ScopeSession

    # motion_enabled reads the XY fact off the live scope, so the double
    # has to carry it explicitly rather than leaving it to autospec truthiness.
    scope = spec_scope()
    scope.capabilities.has_xy_stage = has_xy_stage
    return ScopeSession(
        settings={},
        scope=scope,
        executor_bundle=MagicMock(file_io_executor=MagicMock()),
    )


def _draining_run(session, writes=1):
    """The runner's last run, ended with ``writes`` of its files still to land."""
    from modules.protocol_image_writer import RunWriteBatch

    batch = RunWriteBatch(session.file_io_executor)
    for _ in range(writes):
        batch.submit(lambda: None, {}, what='an image', pace_until=None)
    session.sequenced_capture_runner._write_batch = batch
    batch.close(lambda outcome: None)


class TestDerivations:
    def test_idle_session_is_fully_unlocked(self):
        session = _make_session()
        assert session.exclusive_activity is None
        assert session.is_protocol_running is False
        assert session.run_lockout is False
        assert session.controls_locked is False
        assert session.motion_enabled is True

    def test_protocol_claim_locks_everything(self):
        session = _make_session()
        assert session.activity_claim.try_claim('protocol', run=run_identity())
        assert session.run_lockout is True
        assert session.controls_locked is True
        assert session.motion_enabled is False

    @pytest.mark.slow
    def test_protocol_drain_holds_lockout_after_claim_release(self):
        # A finished protocol frees its claim while files drain; the
        # control surface stays locked until its files are written.
        session = _make_session()
        _draining_run(session)
        assert session.exclusive_activity is None
        assert session.run_lockout is True
        assert session.controls_locked is True
        assert session.motion_enabled is False

    def test_live_recording_locks_controls_but_not_run_lockout(self):
        session = _make_session()
        assert session.activity_claim.try_claim('recording')
        session.manual_recording._engine = MagicMock(is_recording=True)
        assert session.manual_recording.is_recording is True
        assert session.run_lockout is False, (
            'a recording is not a run: run_lockout carries only runs and the protocol file drain'
        )
        assert session.controls_locked is True

    def test_draining_recording_frees_controls_while_claim_refuses(self):
        # The recording drain window: claim held (new runs refuse), but
        # capturing is over so the control surface frees.
        session = _make_session()
        assert session.activity_claim.try_claim('recording')
        session.manual_recording._engine = MagicMock(is_recording=False, is_draining=True)
        assert session.exclusive_activity == 'recording'
        assert session.controls_locked is False

    def test_no_xystage_disables_motion_even_unlocked(self):
        session = _make_session(has_xy_stage=False)
        assert session.run_lockout is False
        assert session.motion_enabled is False

    @pytest.mark.slow
    def test_the_pending_count_is_the_write_batchs_own(self):
        session = _make_session()
        _draining_run(session, writes=7)
        assert session.protocol_files_pending == 7

    def test_a_live_runs_writes_are_not_counted_as_draining(self):
        """While the run is live its own state answers; its writes held on
        the lane are not a finished run's files still to land."""
        from modules.protocol_image_writer import RunWriteBatch

        session = _make_session()
        batch = RunWriteBatch(session.file_io_executor)
        batch.submit(lambda: None, {}, what='an image', pace_until=None)
        session.sequenced_capture_runner._write_batch = batch
        assert batch.pending == 1

        assert session.protocol_files_draining is False
        assert session.protocol_files_pending == 0

    @pytest.mark.slow
    def test_a_stalled_drain_is_judged_by_the_run_refusals_threshold(self):
        # One threshold for "stuck": the display of a stalled writer and
        # the refusal of a new run over it must not disagree.
        from modules.protocol_image_writer import WRITE_STALL_FATAL_S

        session = _make_session()
        _draining_run(session)
        executor = session.file_io_executor
        executor.in_flight_task_stalled.return_value = True
        assert session.protocol_files_stalled is True
        executor.in_flight_task_stalled.assert_called_once_with(WRITE_STALL_FATAL_S)

    def test_close_drain_pending_covers_both_video_drain_sources(self):
        """What a close would interrupt on the video side, in one read.

        Two independent drains can hold queued frames at close: a manual
        recording's own, and a finished run's video-step tail. The close
        handler used to OR them together itself; the fact belongs here,
        where every consumer -- GUI, headless, REST -- reads the same one.
        """
        session = _make_session()
        session.manual_recording._engine = None
        session.sequenced_capture_runner = MagicMock(video_drain_busy=False)
        assert session.close_drain_pending is False

        session.manual_recording._engine = MagicMock(is_recording=False, is_draining=True)
        assert session.close_drain_pending is True, 'a recording drain is pending work'

        session.manual_recording._engine = None
        session.sequenced_capture_runner.video_drain_busy = True
        assert session.close_drain_pending is True, "a run's video tail is pending work"

    def test_the_close_count_adds_both_video_drain_sources(self):
        """How many frames a close would wait for, from the same two sources
        close_drain_pending reads, as an attribute like it."""
        session = _make_session()
        session.manual_recording._engine = None
        session.sequenced_capture_runner = MagicMock(video_pending_writes=0)
        assert session.close_drain_frames == 0

        session.manual_recording._engine = MagicMock(pending_writes=5)
        session.sequenced_capture_runner.video_pending_writes = 7
        assert session.close_drain_frames == 12

    def test_the_close_discard_drops_both_video_drain_sources(self):
        session = _make_session()
        recording_engine = MagicMock()
        session.manual_recording._engine = recording_engine
        session.sequenced_capture_runner = MagicMock()

        session.discard_close_drain()

        recording_engine.discard_pending.assert_called_once_with()
        session.sequenced_capture_runner.discard_video_pending.assert_called_once_with()

    def test_a_live_recording_is_both_capturing_and_close_pending(self):
        """The close gate needs the two apart, and they overlap.

        manual_recording.is_recording is the narrower fact: it alone means the rest
        of the take is still to come, which is what the close confirms
        about. close_drain_pending stays true across the whole window.
        """
        session = _make_session()
        session.sequenced_capture_runner = MagicMock(video_drain_busy=False)
        session.manual_recording._engine = MagicMock(is_recording=True, is_draining=False)
        assert session.manual_recording.is_recording is True
        assert session.close_drain_pending is True


class TestTransitionNotification:
    def test_claim_grant_and_release_notify(self):
        session = _make_session()
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
        session = _make_session()
        fired = []
        session.add_run_state_listener(lambda: fired.append(True))
        assert fired, 'registration must invoke the listener once (level sync)'

    def test_the_write_batchs_completion_is_one_run_state_edge(self):
        # The drain's end is a run-state change: run_lockout drops with it,
        # so the Session must hear it -- once, when the run's last write
        # lands, not when the run closes its writes.
        from modules.protocol_callbacks import ProtocolCallbacks
        from modules.protocol_cleanup import RunCompleteNotice
        from modules.protocol_image_writer import RunWriteBatch

        session = _make_session()
        runner = session.sequenced_capture_runner
        # The run's fields start() sets that the completion captures.
        runner._callbacks = ProtocolCallbacks()
        runner._disable_saving_artifacts = True
        batch = RunWriteBatch(session.file_io_executor)
        runner._write_batch = batch
        batch.submit(lambda: None, {}, what='an image', pace_until=None)
        (task,) = [c.args[0] for c in session.file_io_executor.put.call_args_list]
        fired = []
        session._run_state_listeners.append(lambda: fired.append(session.run_lockout))

        runner._close_run_writes(
            batch,
            RunCompleteNotice(runner._callbacks, protocol=None, ending=MagicMock(), run_dir=None),
        )
        assert fired == [], 'closing with a write outstanding is not the drain ending'
        assert session.run_lockout is True

        task.action(**task.kwargs)

        assert batch.outcome == 'written'
        assert fired == [False], (
            f'the last write landing must reach the run-state edge once, unlocked; got {fired}'
        )

    def test_failed_claim_does_not_notify(self):
        session = _make_session()
        assert session.activity_claim.try_claim('protocol', run=run_identity())
        fired = []
        session._run_state_listeners.append(lambda: fired.append(True))
        assert not session.activity_claim.try_claim('recording')
        assert not fired, 'a refused claim is not a transition'

    def test_listener_exception_does_not_break_others(self):
        session = _make_session()
        fired = []
        session._run_state_listeners.append(lambda: (_ for _ in ()).throw(RuntimeError('x')))
        session._run_state_listeners.append(lambda: fired.append(True))
        session.notify_run_state()
        assert fired
