# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Session run-state facts and derivations.

Each run-state FACT has exactly one owner -- the activity claim
(arbitration), the recording engine (live-vs-drain), the file writer
(pending work), the motion driver (XY stage) -- and every consumer
truth is a synchronous derivation over them:

    is_protocol_running = owner == 'protocol'
    run_lockout         = owner == 'protocol' or protocol_files_draining
    controls_locked     = run_lockout or (owner == 'recording' and recording_capturing)
    motion_enabled      = capabilities.has_xy_stage and not run_lockout

The drain terms encode today's documented asymmetry: a draining
recording HOLDS its claim while the controls free; a finished protocol
FREES its claim while the controls stay locked until the file queue
empties. Transitions notify level-read listeners (they re-read the
derivations when they fire, so out-of-order delivery degrades to
bounded staleness, never a permanently wrong publish).
"""

from unittest.mock import MagicMock

from tests.scope_fakes import spec_scope


def _make_session(file_io_executor=None, has_xy_stage=True):
    from modules.scope_session import ScopeSession

    # motion_enabled reads the XY fact off the live scope, so the double
    # has to carry it explicitly rather than leaving it to autospec truthiness.
    scope = spec_scope()
    scope.capabilities.has_xy_stage = has_xy_stage
    return ScopeSession(
        settings={},
        scope=scope,
        io_executor=MagicMock(),
        camera_executor=MagicMock(),
        file_io_executor=file_io_executor,
    )


def _file_executor(active=False):
    executor = MagicMock()
    executor.is_protocol_queue_active.return_value = active
    return executor


class TestDerivations:
    def test_idle_session_is_fully_unlocked(self):
        session = _make_session(_file_executor(active=False))
        assert session.exclusive_activity is None
        assert session.is_protocol_running is False
        assert session.run_lockout is False
        assert session.controls_locked is False
        assert session.motion_enabled is True

    def test_protocol_claim_locks_everything(self):
        session = _make_session(_file_executor(active=False))
        assert session.activity_claim.try_claim('protocol')
        assert session.run_lockout is True
        assert session.controls_locked is True
        assert session.motion_enabled is False

    def test_protocol_drain_holds_lockout_after_claim_release(self):
        # A finished protocol frees its claim while files drain; the
        # control surface stays locked until the queue empties.
        session = _make_session(_file_executor(active=True))
        assert session.exclusive_activity is None
        assert session.run_lockout is True
        assert session.controls_locked is True
        assert session.motion_enabled is False

    def test_live_recording_locks_controls_but_not_run_lockout(self):
        session = _make_session(_file_executor(active=False))
        assert session.activity_claim.try_claim('recording')
        session.manual_recording._engine = MagicMock(is_recording=True)
        assert session.recording_capturing is True
        assert session.run_lockout is False, (
            'a recording is not a run: run_lockout carries only runs and the protocol file drain'
        )
        assert session.controls_locked is True

    def test_draining_recording_frees_controls_while_claim_refuses(self):
        # The recording drain window: claim held (new runs refuse), but
        # capturing is over so the control surface frees.
        session = _make_session(_file_executor(active=False))
        assert session.activity_claim.try_claim('recording')
        session.manual_recording._engine = MagicMock(is_recording=False, is_draining=True)
        assert session.exclusive_activity == 'recording'
        assert session.controls_locked is False

    def test_no_xystage_disables_motion_even_unlocked(self):
        session = _make_session(_file_executor(active=False), has_xy_stage=False)
        assert session.run_lockout is False
        assert session.motion_enabled is False

    def test_no_file_executor_reads_drain_as_false(self):
        session = _make_session(file_io_executor=None)
        assert session.protocol_files_draining is False


class TestTransitionNotification:
    def test_claim_grant_and_release_notify(self):
        session = _make_session(_file_executor(active=False))
        fired = []
        session._run_state_listeners.append(lambda: fired.append(True))
        assert session.activity_claim.try_claim('protocol')
        assert len(fired) == 1
        session.activity_claim.release('protocol')
        assert len(fired) == 2

    def test_registration_level_syncs_immediately(self):
        # Transitions are edges; a listener registered after a grant
        # must still see current truth, so registration republishes.
        session = _make_session(_file_executor(active=False))
        fired = []
        session.add_run_state_listener(lambda: fired.append(True))
        assert fired, 'registration must invoke the listener once (level sync)'

    def test_session_registers_on_the_file_drain_signal(self):
        executor = _file_executor(active=False)
        session = _make_session(executor)
        executor.add_protocol_idle_listener.assert_called_once_with(session.notify_run_state)

    def test_set_scope_republishes(self):
        session = _make_session(_file_executor(active=False))
        fired = []
        session._run_state_listeners.append(lambda: fired.append(True))
        session.set_scope(spec_scope())
        assert fired, 'a scope rebind must level-republish run state'

    def test_failed_claim_does_not_notify(self):
        session = _make_session(_file_executor(active=False))
        assert session.activity_claim.try_claim('protocol')
        fired = []
        session._run_state_listeners.append(lambda: fired.append(True))
        assert not session.activity_claim.try_claim('recording')
        assert not fired, 'a refused claim is not a transition'

    def test_listener_exception_does_not_break_others(self):
        session = _make_session(_file_executor(active=False))
        fired = []
        session._run_state_listeners.append(lambda: (_ for _ in ()).throw(RuntimeError('x')))
        session._run_state_listeners.append(lambda: fired.append(True))
        session.notify_run_state()
        assert fired
