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

from modules.exceptions import DiagnosticRefusedError, HardwareCommandRefusedError
from tests.scope_fakes import spec_scope


def _make_session():
    from modules.scope_session import ScopeSession

    scope = spec_scope()
    scope.capabilities.has_xy_stage = True
    file_io_executor = MagicMock()
    file_io_executor.is_protocol_queue_active.return_value = False
    return ScopeSession(
        settings={},
        scope=scope,
        io_executor=MagicMock(),
        camera_executor=MagicMock(),
        file_io_executor=file_io_executor,
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
            assert session.exclusive_activity == 'diagnostic'
        assert not held.holds
        assert session.exclusive_activity is None

    def test_a_raise_inside_the_block_still_releases(self):
        session = _make_session()
        with (
            pytest.raises(RuntimeError, match='the diagnostic failed'),
            session.diagnostic_claim(),
        ):
            raise RuntimeError('the diagnostic failed')
        assert session.exclusive_activity is None, (
            'a diagnostic that raised left the claim held, which refuses every '
            'run and recording for the life of the process'
        )

    def test_the_controls_lock_while_it_holds_and_listeners_see_both_edges(self):
        session = _make_session()
        seen = []
        session.add_run_state_listener(lambda: seen.append(session.controls_locked))
        seen.clear()
        with session.diagnostic_claim():
            assert session.run_lockout is True
            assert session.controls_locked is True
            assert session.motion_enabled is False
        assert session.controls_locked is False
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
            engine.start(MagicMock())
        assert excinfo.value.reason == 'exclusive_activity_running'
        assert excinfo.value.holder == 'diagnostic'


class TestTheDiagnosticIsRefusedWhenTheScopeIsHeld:
    @pytest.mark.parametrize('holder_kind', ['protocol', 'recording', 'diagnostic'])
    def test_refused_naming_the_holder_and_nothing_taken(self, holder_kind):
        session = _make_session()
        other = session.activity_claim.try_claim(holder_kind, run_trigger_source=None)
        try:
            with pytest.raises(DiagnosticRefusedError) as excinfo, session.diagnostic_claim():
                pytest.fail('the block ran while another activity held the scope')
            assert excinfo.value.reason == 'exclusive_activity_running'
            assert excinfo.value.holder == holder_kind
            assert holder_kind in excinfo.value.message
            assert other.holds, "a refused diagnostic must not touch the holder's claim"
        finally:
            other.release()

    def test_a_run_holders_trigger_is_named(self):
        session = _make_session()
        run = session.activity_claim.try_claim('protocol', run_trigger_source='scan')
        try:
            with pytest.raises(DiagnosticRefusedError) as excinfo, session.diagnostic_claim():
                pass
            assert excinfo.value.holder_trigger == 'scan'
        finally:
            run.release()
