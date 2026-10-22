# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run records who started it and a stop says who stopped it, from the host's stamp.

No caller passes provenance: each host stamps the threads its calls arrive
on (``modules.caller``) -- the GUI where it hands a request to a worker or
a lane, the REST server on each call's own thread -- and a thread nobody
stamped is an in-process caller. A run records the stamp on the thread
that starts it; a refusal it causes publishes it as ``holder_trigger``, on
the wire too; a Stop's ending names the stopper.
"""

from __future__ import annotations

import asyncio
import inspect
import threading

import pytest

from modules.caller import Caller, calling, current_caller
from modules.exceptions import ProtocolRunRefusedError
from modules.protocol_runner import ProtocolRunner
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import SequentialIOExecutor
from rest import problems
from rest.jobs import JobRegistry
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT, _settings
from tests.test_a_run_that_ends_mid_move_halts_the_stage import _HOLD_WAIT_S
from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


def _start_held(session, tmp_path):
    """Start a one-step scan held part of the way along its X move: live until stopped."""
    held = session.scope.simulation.hold_axis('X')
    step = {**_make_single_step_protocol().step_row(step_index=0), 'X': 60.0, 'Y': 20.0}
    run = session.create_protocol_runner().start_single_scan(
        protocol=_build_real_protocol([step]), parent_dir=str(tmp_path)
    )
    assert held.wait(_HOLD_WAIT_S), "the run's move never got part of the way"
    return run


class TestTheRunRecordsTheStampOnItsStartersThread:
    @pytest.mark.parametrize('caller', [Caller.GUI, Caller.REST, Caller.API])
    def test_a_second_start_is_refused_naming_who_started_the_first(
        self, session, tmp_path, caller
    ):
        with calling(caller):
            run = _start_held(session, tmp_path)
        try:
            with pytest.raises(ProtocolRunRefusedError) as refused:
                session.create_protocol_runner().start_single_scan(
                    protocol=_build_real_protocol(
                        [_make_single_step_protocol().step_row(step_index=0)]
                    ),
                    parent_dir=str(tmp_path),
                )
            assert refused.value.reason == 'already_running'
            assert refused.value.holder_trigger is caller
        finally:
            run.stop()
            run.result(timeout=COMPLETION_TIMEOUT)

    def test_no_starter_takes_a_provenance_argument(self):
        starters = [
            name
            for name, member in vars(ProtocolRunner).items()
            if name.startswith(('start_', 'run_')) and callable(member)
        ]
        assert len(starters) >= 12, starters
        taking = [
            name
            for name in starters
            if 'run_trigger_source' in inspect.signature(getattr(ProtocolRunner, name)).parameters
        ]
        assert taking == []


class TestAStopSaysWhoStoppedTheRun:
    def test_the_ending_names_the_gui(self, session, tmp_path):
        run = _start_held(session, tmp_path)

        with calling(Caller.GUI):
            run.stop()
        outcome = run.result(timeout=COMPLETION_TIMEOUT)

        assert (outcome.status, outcome.reason) == ('aborted', 'stopped')
        assert outcome.message == 'Stopped by LumaViewPro.'


class TestTheRefusalCarriesItOnTheWire:
    def test_a_problem_body_carries_the_holder_and_who_started_its_run(self):
        refusal = ProtocolRunRefusedError(
            'already_running',
            'The Z-stack run is using the microscope.',
            holder='protocol',
            holder_trigger=Caller.REST,
        )

        body = problems.answered_by_member(refusal, 'request-1').body

        assert (body['holder'], body['holder_trigger']) == ('protocol', 'rest')


class TestEachHostStampsItsThreads:
    def test_a_rest_call_runs_as_a_rest_client(self):
        async def _call():
            return await JobRegistry(ended=lambda _job: None).run(
                current_caller, 'probe', 'request-1'
            )

        assert asyncio.run(_call()) is Caller.REST

    def test_a_gui_request_runs_as_the_gui_on_its_lane(self, monkeypatch):
        from ui import ui_helpers

        lane = SequentialIOExecutor(name='probe_lane')
        lane.start()
        seen = []
        ran = threading.Event()

        def _call():
            seen.append(current_caller())
            ran.set()

        try:
            ui_helpers.submit_reported(_call, None, 'probe', lane=lane)
            assert ran.wait(COMPLETION_TIMEOUT)
        finally:
            lane.shutdown()
        assert seen == [Caller.GUI]

    def test_an_unstamped_thread_is_an_in_process_caller(self):
        seen = []
        thread = threading.Thread(target=lambda: seen.append(current_caller()))
        thread.start()
        thread.join()
        assert seen == [Caller.API]
