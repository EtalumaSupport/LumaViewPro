# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A waiting form called on the UI thread is refused before it starts anything.

A waiting form is its start followed by the started operation's
``result()``, and a ``result()`` on the UI thread for an operation not yet
done is refused ``wait_on_ui_thread`` (``modules/operation.py``). Asked
only at the wait, that refusal reached the caller after the start: the
caller was told "refused" while the stage drove on or the run went on. So
each waiting form refuses first, by the same reason, and the scope is left
as it was: nothing holds it, no lane has work, no axis moves.
"""

import pytest

from modules import kivy_utils
from modules.exceptions import BlockingCallRefusedError
from modules.scope_session import ScopeSession
from tests.test_a_wait_returns_after_its_own_callbacks import _DeferringDispatcher
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.test_going_to_a_step_is_the_sessions_move import _PLATE, _protocol, _step


@pytest.fixture
def ui():
    stand_in = _DeferringDispatcher()
    ScopeSession.set_ui_dispatcher(
        kivy_utils.UiDispatcher(schedule=stand_in.schedule, thread=stand_in.thread)
    )
    yield stand_in
    ScopeSession.set_ui_dispatcher(None)
    stand_in.stop()


def _far_x(motion) -> float:
    limits = motion.get_axis_limits('X')
    here = motion.get_current_position('X')
    return limits['max'] * 0.9 if here < (limits['min'] + limits['max']) / 2 else limits['min']


def _far_step(session):
    objective = session.scope.runtime_state.get_current_objective_id()
    return _protocol(_PLATE, _step(20.0, 20.0, 5000.0, objective))


# Each form's call, given the session, its runner and a folder no run may write.
_WAITING_FORMS = {
    'move_absolute': lambda s, r, d: s.scope.motion.move_absolute('X', _far_x(s.scope.motion)),
    'move_relative': lambda s, r, d: s.scope.motion.move_relative(
        'X', _far_x(s.scope.motion) - s.scope.motion.get_current_position('X')
    ),
    'go_to_step': lambda s, r, d: s.go_to_step(_far_step(s), 0),
    'start_go_to_step': lambda s, r, d: s.start_go_to_step(_far_step(s), 0),
    'home': lambda s, r, d: s.scope.motion.home('Z'),
    'move_turret': lambda s, r, d: s.scope.motion.move_turret(
        2 if s.scope.motion.get_turret_slot() == 1 else 1
    ),
    'run_application_session': lambda s, r, d: s.run_application_session(),
    'run_composite': lambda s, r, d: r.run_composite(parent_dir=str(d)),
    'run_single_scan': lambda s, r, d: r.run_single_scan(_far_step(s), parent_dir=str(d)),
    'run_protocol': lambda s, r, d: r.run_protocol(_far_step(s), parent_dir=str(d)),
    'run_autofocus': lambda s, r, d: r.run_autofocus('BF', parent_dir=str(d)),
    'run_autofocus_all_steps': lambda s, r, d: r.run_autofocus_all_steps(_far_step(s)),
    'run_zstack': lambda s, r, d: r.run_zstack('BF', parent_dir=str(d)),
}


# parametrize-justified: each form needs a scope nothing else has started on,
# since a form that did start would hold the scope the next one asks for.
@pytest.mark.parametrize('member', sorted(_WAITING_FORMS))
def test_a_waiting_form_on_the_ui_thread_is_refused_before_it_starts(member, ui, tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        # Seconds of travel, so a move that did start is still under way
        # when the refusal comes back.
        session.scope._motion_driver.set_timing_mode('realistic')
        never_written = tmp_path / 'refused'

        how, answer = ui.call(lambda: _WAITING_FORMS[member](session, runner, never_written))
        started = session.in_flight_named()

        assert how == 'raised' and isinstance(answer, BlockingCallRefusedError), (how, answer)
        assert started == [], f'{member} started {started} before it was refused'
        assert (answer.reason, answer.member) == ('wait_on_ui_thread', member)
        assert not never_written.exists()
