# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Once a session's close has begun, nothing new takes the scope.

The close finishes the work already under way and then releases the
hardware. A run started on another thread while it waited was left live
for ever, holding the scope, 30 of 30 (the close probes' F3): the close
had no state a start could be refused by. The refusal is the claim's, for
every activity that takes the scope; work lent by an activity already
under way -- a run's video step under its run -- is that activity's and is
not refused (refusing it failed the run), and a post-processing build,
which a protocol's processors start through the same door, is waited for,
not refused.
"""

import pytest

from modules.exceptions import SessionClosingError
from tests.protocol_drives import run_identity


@pytest.fixture
def sim_session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings
    from tests.test_a_run_needs_every_axis_position import _settings

    session = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(session.scope)
        yield session
    finally:
        session.shutdown()


def _refused(call, activity):
    with pytest.raises(SessionClosingError) as refused:
        call()
    assert refused.value.reason == 'session_closing'
    assert refused.value.activity == activity


def test_a_run_a_recording_a_home_and_a_diagnostic_are_each_refused(sim_session, tmp_path):
    from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

    sim_session.activity_claim.begin_closing()
    assert sim_session.live_work.closing is True

    runner = sim_session.create_protocol_runner()
    protocol = _build_real_protocol([_make_single_step_protocol().step(idx=0)])
    _refused(
        lambda: runner.run_single_scan(
            protocol=protocol, sequence_name='closing', parent_dir=str(tmp_path)
        ),
        'protocol',
    )
    _refused(lambda: sim_session.manual_recording.start(layer='BF'), 'recording')
    _refused(lambda: sim_session.scope.motion.home('Z'), 'home')

    def a_diagnostic():
        with sim_session.diagnostic_claim():
            pass

    _refused(a_diagnostic, 'diagnostic')

    assert sim_session.activity_claim.holder is None, 'nothing was taken'
    assert sim_session.live_work.work == ()


def test_the_refusal_says_what_was_not_started_and_why():
    refused = SessionClosingError('home')
    assert str(refused) == (
        'A home was not started: LumaViewPro is closing, and finishes what is already '
        'under way before it lets go of the microscope.'
    )
    assert refused.title == 'Closing'


def test_work_lent_by_an_activity_already_under_way_is_still_admitted(sim_session):
    held = sim_session.activity_claim.try_claim('diagnostic')
    try:
        lent = held.lend()
        sim_session.activity_claim.begin_closing()

        borrowed = lent.try_claim('protocol', run=run_identity())

        assert borrowed is not None, "a run inside the diagnostic is the diagnostic's own work"
        borrowed.release()
    finally:
        held.release()


def test_a_post_processing_build_is_not_refused(sim_session, tmp_path):
    sim_session.activity_claim.begin_closing()

    def build(folder, *, on_progress):
        return {'message': 'built'}

    assert sim_session.post_processing._run(build, 'stitch', tmp_path) == {'message': 'built'}
