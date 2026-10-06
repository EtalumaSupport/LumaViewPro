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

from types import SimpleNamespace

import pytest

from tests.test_diagnostic_claim import _make_session
from tests.protocol_drives import run_identity


class TestALentRun:
    def test_a_run_lent_a_diagnostics_claim_is_the_run_in_progress(self):
        session = _make_session()
        with session.diagnostic_claim() as held:
            run = held.lend().try_claim('protocol', run=run_identity('api_autofocus'))
            try:
                assert session.is_protocol_running is True
                assert session.sequenced_capture_runner.run_trigger_source() == 'api_autofocus'
                assert session.exclusive_activity == 'diagnostic', (
                    'the holder stays the diagnostic, which a refusal names'
                )
                assert session.activity_claim.blocking_holder.kind == 'diagnostic'
            finally:
                run.release()
            assert session.is_protocol_running is False
            assert session.sequenced_capture_runner.run_trigger_source() is None

    def test_the_lend_and_the_return_are_each_announced(self):
        session = _make_session()
        heard = []
        session.add_run_state_listener(lambda: heard.append(session.is_protocol_running))
        heard.clear()
        with session.diagnostic_claim() as held:
            heard.clear()
            run = held.lend().try_claim('protocol', run=run_identity('api_autofocus'))
            assert heard == [True], f'the lend was not announced as a run; heard {heard}'
            run.release()
            assert heard == [True, False], f'the return was not announced; heard {heard}'

    def test_a_recording_lent_from_a_run_leaves_the_run_holding(self):
        session = _make_session()
        run = session.activity_claim.try_claim('protocol', run=run_identity('scan'))
        try:
            recording = run.lend().try_claim('recording')
            assert session.is_protocol_running is True
            assert session.sequenced_capture_runner.run_trigger_source() == 'scan'
            recording.release()
            assert session.is_protocol_running is True
        finally:
            run.release()

    def test_a_lent_run_does_not_outlive_its_lender(self):
        session = _make_session()
        with session.diagnostic_claim() as held:
            held.lend().try_claim('protocol', run=run_identity('api_autofocus'))
        assert session.is_protocol_running is False, (
            'the diagnostic released its claim, and the run it lent still reads running'
        )


class TestHeldByOther:
    def test_nothing_holds(self):
        session = _make_session()
        assert session.held_by_other(None) is False

    @pytest.mark.parametrize('kind', ['diagnostic', 'recording', 'protocol'])
    def test_any_holder_but_the_given_run_is_other(self, kind):
        session = _make_session()
        run = run_identity() if kind == 'protocol' else None
        held = session.activity_claim.try_claim(kind, run=run)
        try:
            assert session.held_by_other(None) is True
        finally:
            held.release()

    def test_the_session_answers_the_engines_answer(self, monkeypatch):
        session = _make_session()
        asked = []

        def _held_by_other(run):
            asked.append(run)
            return True

        monkeypatch.setattr(session.sequenced_capture_runner, 'held_by_other', _held_by_other)
        handle = object()
        assert session.held_by_other(handle) is True
        assert asked == [handle]


class TestWhatTheGuiDerivesToday:
    def test_a_live_recording_is_recording_active_and_its_drain_is_not(self, monkeypatch):
        session = _make_session()
        held = session.activity_claim.try_claim('recording')
        try:
            monkeypatch.setattr(session, 'manual_recording', SimpleNamespace(is_recording=True))
            assert session.recording_active is True
            monkeypatch.setattr(session, 'manual_recording', SimpleNamespace(is_recording=False))
            assert session.recording_active is False
        finally:
            held.release()

    def test_a_run_in_progress_is_the_engines(self, monkeypatch):
        session = _make_session()
        assert session.run_in_progress is False
        monkeypatch.setattr(session.sequenced_capture_runner, 'run_in_progress', lambda: True)
        assert session.run_in_progress is True
