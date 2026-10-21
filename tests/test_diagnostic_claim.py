# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A diagnostic holds the scope the way a run does, for as long as it runs.

A characterization or the support report's hardware steps drive every
axis, the LEDs and the camera directly. Before this, nothing arbitrated
them: a run or a recording could start in the middle of one, and the
support report pressed mid-recording homed the stage under it. The
diagnostic now takes the session's one activity claim, and everything the
claim already refuses and locks for a run applies to it.
"""

from unittest.mock import MagicMock

import pytest

from modules.caller import Caller
from modules.exceptions import (
    ClaimStillLentError,
    DiagnosticRefusedError,
    HardwareCommandRefusedError,
)
from tests.scope_fakes import real_executor_bundle, spec_scope
from tests.protocol_drives import run_identity


def _make_session():
    from modules.scope_session import ScopeSession

    file_io_executor = MagicMock()
    return ScopeSession(
        settings={},
        scope=spec_scope(),
        executor_bundle=real_executor_bundle(file_io_executor=file_io_executor),
    )


@pytest.fixture
def sim_session(tmp_path):
    """A simulated turreted scope with a known objective in the live slot."""
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            microscope='LS850T',
            objective_confirmed=True,
            turret_objectives={1: '10x Oly', 2: '4x Oly', 3: None, 4: None},
        ),
        simulate=True,
    )
    try:
        home_sim_scope(s.scope)
        s.scope.motion.move_turret(1)
        yield s
    finally:
        s.shutdown()


class TestTheDiagnosticHoldsTheScope:
    def test_it_holds_the_claim_for_the_block_and_releases_after(self):
        session = _make_session()
        with session.diagnostic_claim() as held:
            assert held.holds
            assert session.activity_claim.owner == 'diagnostic'
        assert not held.holds
        assert session.activity_claim.owner is None

    def test_a_raise_inside_the_block_still_releases(self):
        session = _make_session()
        with (
            pytest.raises(RuntimeError, match='the diagnostic failed'),
            session.diagnostic_claim(),
        ):
            raise RuntimeError('the diagnostic failed')
        assert session.activity_claim.owner is None, (
            'a diagnostic that raised left the claim held, which refuses every '
            'run and recording for the life of the process'
        )

    def test_the_controls_lock_while_it_holds_and_listeners_see_both_edges(self):
        session = _make_session()
        seen = []
        session.add_run_state_listener(lambda: seen.append(session.controls_lock.controls_locked))
        seen.clear()
        with session.diagnostic_claim():
            assert session.controls_lock.run_lockout is True
            assert session.controls_lock.controls_locked is True
        assert session.controls_lock.controls_locked is False
        assert seen == [True, False], f'a listener must see the lock set and cleared; saw {seen}'

    @pytest.mark.parametrize(
        'write',
        [
            lambda s: s.select_objective('20x Oly'),
            lambda s: s.assign_turret_objective(1, '20x Oly'),
            lambda s: s.clear_turret_objective(2),
        ],
        ids=['select', 'assign', 'clear'],
    )
    def test_an_objective_change_is_refused_while_it_holds(self, sim_session, write):
        before = dict(sim_session.settings['turret_objectives'])
        with (
            sim_session.diagnostic_claim(),
            pytest.raises(HardwareCommandRefusedError) as excinfo,
        ):
            write(sim_session)
        assert excinfo.value.reason == 'exclusive_activity_running'
        assert sim_session.settings['turret_objectives'] == before
        assert sim_session.scope.runtime_state.get_current_objective_id() == '10x Oly'

    def test_a_recording_start_is_refused_naming_it(self):
        from modules.exceptions import RecordingRefusedError
        from modules.video_recording import VideoRecordingEngine

        session = _make_session()
        engine = VideoRecordingEngine(
            write_frame=MagicMock(), claim=session.activity_claim, clock=lambda: 0.0
        )
        with session.diagnostic_claim(), pytest.raises(RecordingRefusedError) as excinfo:
            engine.start(lambda: MagicMock())
        assert excinfo.value.reason == 'exclusive_activity_running'
        assert excinfo.value.holder == 'diagnostic'


class TestTheDiagnosticIsRefusedWhenTheScopeIsHeld:
    @pytest.mark.parametrize('holder_kind', ['protocol', 'recording', 'diagnostic'])
    def test_refused_naming_the_holder_and_nothing_taken(self, holder_kind):
        session = _make_session()
        run = run_identity() if holder_kind == 'protocol' else None
        other = session.activity_claim.try_claim(holder_kind, run=run)
        try:
            with pytest.raises(DiagnosticRefusedError) as excinfo, session.diagnostic_claim():
                pytest.fail('the block ran while another activity held the scope')
            assert excinfo.value.reason == 'exclusive_activity_running'
            assert excinfo.value.holder == holder_kind
            # A run is named by its kind in words; any other activity by its kind.
            named = 'The scan run' if holder_kind == 'protocol' else holder_kind
            assert named in excinfo.value.message
            assert other.holds, "a refused diagnostic must not touch the holder's claim"
        finally:
            other.release()

    def test_who_started_the_holding_run_is_named(self):
        session = _make_session()
        run = session.activity_claim.try_claim('protocol', run=run_identity(Caller.GUI))
        try:
            with pytest.raises(DiagnosticRefusedError) as excinfo, session.diagnostic_claim():
                pass
            assert excinfo.value.holder_trigger == Caller.GUI
        finally:
            run.release()


class TestALentClaim:
    """Work inside a diagnostic acts under its claim and cannot end it."""

    def test_a_borrowing_holds_while_its_lender_holds(self):
        from modules.activity_claim import ActivityClaim

        claim = ActivityClaim()
        held = claim.try_claim('diagnostic')
        borrowing = held.lend().try_claim('protocol', run=run_identity())
        assert borrowing.holds
        borrowing.release()
        assert held.holds, "a borrowing's release must leave the lender's claim held"
        held.release()
        assert not borrowing.holds, 'a borrowing outlived the claim it borrowed'

    def test_a_borrowing_lends_onward_under_the_same_lender(self):
        """A run inside a diagnostic lends its claim to its own recordings."""
        from modules.activity_claim import ActivityClaim

        claim = ActivityClaim()
        held = claim.try_claim('diagnostic')
        run_taking = held.lend().try_claim('protocol', run=run_identity())
        recording_taking = run_taking.lend().try_claim('recording')
        assert recording_taking is not None and recording_taking.holds
        recording_taking.release()
        run_taking.release()
        assert claim.owner == 'diagnostic'
        held.release()

    def test_the_lender_is_not_in_its_own_borrowers_way(self):
        from modules.activity_claim import ActivityClaim

        claim = ActivityClaim()
        held = claim.try_claim('diagnostic')
        borrow = held.lend()
        assert claim.blocking_holder.kind == 'diagnostic'
        assert borrow.blocking_holder is None
        held.release()
        later = claim.try_claim('recording')
        assert borrow.blocking_holder.kind == 'recording', (
            "once the lender released, whoever took the claim since is in the borrow's way"
        )
        assert borrow.try_claim('protocol', run=run_identity()) is None
        later.release()


def _end_the_run(runner, monkeypatch):
    """The run's cleanup: the runner back to IDLE, the session told, the kept claim released."""
    from modules.protocol_state_machine import ProtocolState

    monkeypatch.setattr(runner, '_state', ProtocolState.IDLE)
    runner._on_run_idle()


class TestTheDiagnosticEndsOnlyWhenItsRunDoes:
    def test_a_run_still_live_past_the_wait_keeps_the_claim_until_it_ends(self, monkeypatch):
        from modules.protocol_state_machine import ProtocolState

        session = _make_session()
        runner = session.sequenced_capture_runner
        monkeypatch.setattr(runner, '_state', ProtocolState.RUNNING)
        monkeypatch.setattr(runner, 'wait_for_run_idle', lambda timeout_s: False)
        with pytest.raises(ClaimStillLentError) as kept, session.diagnostic_claim():
            pass
        assert kept.value.reason == 'claim_still_lent'
        assert isinstance(kept.value, RuntimeError), 'still what it was documented as'
        assert session.activity_claim.owner == 'diagnostic', (
            'the claim was released underneath a run still acting under it'
        )

        # The run ends: its cleanup puts the runner back to IDLE and tells
        # the session, which releases the claim nobody else holds.
        monkeypatch.setattr(runner, '_state', ProtocolState.IDLE)
        runner._on_run_idle()

        assert session.activity_claim.owner is None, 'the claim was left with nobody to release it'

    def test_a_block_that_raised_keeps_its_own_exception_and_the_kept_claim_is_told_once(
        self, monkeypatch, centre_posts
    ):
        """A raise at the block's end would replace the block's own exception,
        the one its caller is owed: in exactly the case the wait is for, a
        caller that raised between starting a run and waiting on it."""
        from modules.protocol_state_machine import ProtocolState

        session = _make_session()
        runner = session.sequenced_capture_runner
        monkeypatch.setattr(runner, '_state', ProtocolState.RUNNING)
        monkeypatch.setattr(runner, 'wait_for_run_idle', lambda timeout_s: False)

        with pytest.raises(ValueError, match='the sweep failed'), session.diagnostic_claim():
            raise ValueError('the sweep failed')

        told = [n for n in centre_posts if n.reason == 'claim_still_lent']
        assert [n.title for n in told] == [ClaimStillLentError.title]
        assert session.activity_claim.owner == 'diagnostic'
        _end_the_run(runner, monkeypatch)

    def test_a_clean_block_inside_a_callers_except_still_raises_the_kept_claim(self, monkeypatch):
        """Whether the block raised is the block's own fact, not the thread's
        exception state: a caller handling an exception is not a block that raised."""
        from modules.protocol_state_machine import ProtocolState

        session = _make_session()
        runner = session.sequenced_capture_runner
        monkeypatch.setattr(runner, '_state', ProtocolState.RUNNING)
        monkeypatch.setattr(runner, 'wait_for_run_idle', lambda timeout_s: False)

        try:
            raise KeyError('handled by the caller')
        except KeyError:
            with pytest.raises(ClaimStillLentError), session.diagnostic_claim():
                pass
        _end_the_run(runner, monkeypatch)

    def test_it_waits_for_the_run_before_it_releases(self, monkeypatch):
        session = _make_session()
        order = []

        def _wait(timeout_s):
            order.append(('wait', session.activity_claim.owner))
            return True

        monkeypatch.setattr(session.sequenced_capture_runner, 'wait_for_run_idle', _wait)
        with session.diagnostic_claim():
            pass
        assert order == [('wait', 'diagnostic')], order
        assert session.activity_claim.owner is None
