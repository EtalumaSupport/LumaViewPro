# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Session answers what holds the scope, and announces each change of it.

A run inside a diagnostic acts under the diagnostic's claim: the holder stays
the diagnostic, so a refusal names what a person can wait for, but the run is
still a run -- it is running, it has a trigger, and its start and end change
what a listener would draw. Before this, a lent run read as no run at all and
neither its lend nor its return told anyone.

A run control needs one more answer: is the scope held by something other
than the run this control started? Its own run leaves it live as that run's
Stop; anything else holding the scope greys it.
"""

import pytest

from modules import live_work
from modules.protocol_image_writer import RunWriteBatch
from modules.run_outcome import PendingRunOutcome
from modules.sequenced_capture_runner import EngineRun
from modules.caller import Caller
from tests.protocol_drives import run_identity
from tests.scope_fakes import session_on_a_simulated_scope


class TestALentRun:
    def test_a_run_lent_a_diagnostics_claim_is_the_run_in_progress(self):
        session = session_on_a_simulated_scope()
        with session.diagnostic_claim() as held:
            run = held.lend().try_claim('protocol', run=run_identity(Caller.REST))
            try:
                assert session.live_work.has(live_work.WorkKind.PROTOCOL)
                assert session.sequenced_capture_runner.run_started_by() == Caller.REST
                assert session.activity_claim.owner == 'diagnostic', (
                    'the holder stays the diagnostic, which a refusal names'
                )
                assert session.activity_claim.blocking_holder.kind == 'diagnostic'
            finally:
                run.release()
            assert not session.live_work.has(live_work.WorkKind.PROTOCOL)
            assert session.sequenced_capture_runner.run_started_by() is None

    def test_the_lend_and_the_return_are_each_announced(self):
        session = session_on_a_simulated_scope()
        heard = []
        session.add_run_state_listener(
            lambda: heard.append(session.live_work.has(live_work.WorkKind.PROTOCOL))
        )
        heard.clear()
        with session.diagnostic_claim() as held:
            heard.clear()
            run = held.lend().try_claim('protocol', run=run_identity())
            assert heard == [True], f'the lend was not announced as a run; heard {heard}'
            run.release()
            assert heard == [True, False], f'the return was not announced; heard {heard}'

    def test_a_recording_lent_from_a_run_leaves_the_run_holding(self):
        session = session_on_a_simulated_scope()
        run = session.activity_claim.try_claim('protocol', run=run_identity(Caller.GUI))
        try:
            recording = run.lend().try_claim('recording')
            assert session.live_work.has(live_work.WorkKind.PROTOCOL)
            assert session.sequenced_capture_runner.run_started_by() == Caller.GUI
            recording.release()
            assert session.live_work.has(live_work.WorkKind.PROTOCOL)
        finally:
            run.release()

    def test_a_lent_run_does_not_outlive_its_lender(self):
        session = session_on_a_simulated_scope()
        with session.diagnostic_claim() as held:
            held.lend().try_claim('protocol', run=run_identity())
        assert not session.live_work.has(live_work.WorkKind.PROTOCOL), (
            'the diagnostic released its claim, and the run it lent still reads running'
        )


class TestHeldByOther:
    def test_nothing_holds(self):
        session = session_on_a_simulated_scope()
        assert session.held_by_other(None) is False

    @pytest.mark.parametrize('kind', ['diagnostic', 'recording', 'protocol'])
    def test_any_holder_but_the_given_run_is_other(self, kind):
        session = session_on_a_simulated_scope()
        run = run_identity() if kind == 'protocol' else None
        held = session.activity_claim.try_claim(kind, run=run)
        try:
            assert session.held_by_other(None) is True
            # Left out is None: a query string says None by omission.
            assert session.held_by_other() is True
        finally:
            held.release()

    def test_the_session_answers_the_engines_answer(self, monkeypatch):
        session = session_on_a_simulated_scope()
        asked = []

        def _held_by_other(run):
            asked.append(run)
            return True

        monkeypatch.setattr(session.sequenced_capture_runner, 'held_by_other', _held_by_other)
        engine = session.sequenced_capture_runner
        run = EngineRun(
            engine, PendingRunOutcome(), RunWriteBatch(session.file_io_executor)
        ).operation
        assert session.held_by_other(run) is True
        assert asked == [run]
