# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run is refused for the scope's state first, by one answer the status can read too.

`SequencedCaptureRunner.state_refusal` holds the gates that need no protocol
and make no hardware exchange: what holds the scope, a run already live, a
run's files still writing, an orphaned autofocus sweep, a part not
connected. `prepare` raises its answer before any protocol gate, and a
reader with no protocol (the session's status) asks it, so the two cannot
disagree. It is read without the run lock, from inside a run's own
transition. An axis whose position is unknown is not among them: a scope
with no motor for an axis a protocol needs is told that before it is told
to home.

Driven through the Session on the simulated scope.
"""

import pytest

from drivers.null_ledboard import NullLEDBoard
from modules.exceptions import ProtocolRunRefusedError
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session
from tests.test_a_step_may_have_no_position import _rewrite_steps, _saved_steps

RUN_WAIT_S = 60


@pytest.fixture
def make_session(tmp_path):
    sessions = []

    def _make(model, **kwargs):
        session = _session(tmp_path, model, **kwargs)
        sessions.append(session)
        return session

    yield _make
    for session in sessions:
        session.shutdown()


def _one_step_protocol(session):
    protocol = session.create_empty_protocol()
    session.set_channel_acquire('BF', 'image')
    session.add_step(protocol)
    return protocol


def test_a_held_scope_is_refused_by_the_gates_as_by_a_start(make_session, tmp_path):
    session = make_session('LS850')
    runner = session.sequenced_capture_runner
    protocol = _one_step_protocol(session)

    with session.diagnostic_claim():
        answered = runner.state_refusal(session.activity_claim)
        with pytest.raises(ProtocolRunRefusedError) as refused:
            session.create_protocol_runner().start_single_scan(protocol, parent_dir=str(tmp_path))

    assert answered is not None
    assert answered.reason == 'exclusive_activity_running'
    assert refused.value.reason == answered.reason
    assert runner.state_refusal(session.activity_claim) is None


def test_a_run_between_taking_the_scope_and_starting_is_already_running(make_session, tmp_path):
    session = make_session('LS850')
    runner = session.sequenced_capture_runner
    heard = []

    def on_edge():
        # Asked from the transitioning thread, which may hold the run lock:
        # an answer that waited on it would hang the start below.
        if session.activity_claim.owner == 'protocol':
            refusal = runner.state_refusal(session.activity_claim)
            heard.append(refusal.reason if refusal is not None else None)

    session.add_run_state_listener(on_edge)
    try:
        run = session.create_protocol_runner().start_single_scan(
            _one_step_protocol(session), parent_dir=str(tmp_path)
        )
        run.result(timeout=RUN_WAIT_S)
    finally:
        session.remove_run_state_listener(on_edge)

    assert heard, 'no run-state edge was heard while the run held the scope'
    assert set(heard) == {'already_running'}, heard


def test_a_part_not_connected_is_told_before_an_empty_protocol(make_session, tmp_path):
    session = make_session('LS850')
    session.scope._led_driver = NullLEDBoard()

    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.create_protocol_runner().start_single_scan(
            session.create_empty_protocol(), parent_dir=str(tmp_path)
        )

    assert refused.value.reason == 'not_connected'
    assert session.sequenced_capture_runner.state_refusal(session.activity_claim).reason == (
        'not_connected'
    )


def test_a_scope_that_cannot_reach_the_wells_is_told_so_before_it_is_told_to_home(
    make_session, tmp_path
):
    stage = make_session('LS850')
    path = _rewrite_steps(
        _saved_steps(stage, tmp_path / 'recorded.tsv', 4),
        X='',
        Y='',
        Well=['A1', 'A2', 'A3', 'A4'],
    )
    session = make_session('LS820', homed=False)
    assert session.scope.motion.axes_without_position()
    protocol = session.load_protocol(path)

    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.create_protocol_runner().start_single_scan(protocol, parent_dir=str(tmp_path))

    assert refused.value.reason == 'positions_unreachable'
    assert session.sequenced_capture_runner.state_refusal(session.activity_claim) is None
