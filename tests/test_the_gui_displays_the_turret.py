# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The GUI shows where the API says the turret is, and decides nothing.

Every turret action -- a press, a home, a run's step, an objective answer
-- ends in one display, VerticalControl.show_turret_state, which shows the
slot the API reports, each slot's assignment and the active objective. A
failed move therefore shows the turret in no known slot, never in the one
that was asked for. A person's pick of an objective reaches the Session
once, as a pick; a write to the spinner's text is display and reaches
nothing. The widget methods below are the real ones, run against a real
simulated LS850T session; only the widget tree is stood in for.
"""

from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import ui.notification_popup as notification_popup
import ui.vertical_control as vc
from modules.exceptions import AxisStateUnknownError, ObjectiveUnknownError
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from ui.pick_spinner import PickSpinner
from ui.vertical_control import VerticalControl

ASSIGNED = {'1': '10x Oly', '2': '4x Oly', '3': None, '4': 'no such glass'}


class _Stand:
    show_turret_state = VerticalControl.show_turret_state
    pick_objective = VerticalControl.pick_objective
    turret_select = VerticalControl.turret_select
    reset_turret_objective = VerticalControl.reset_turret_objective

    def __init__(self):
        self.ids = {'objective_spinner2': SimpleNamespace(text='Unknown')}
        for position in range(1, 5):
            self.ids[f'turret_pos_{position}_btn'] = SimpleNamespace(
                text=f'< {position} >', state='normal'
            )
        self.prompts = 0

    def prompt_if_objective_unknown(self, on_resolved=None):
        self.prompts += 1


class _RunNowExecutor:
    """The io lane as the display sees it: the task runs, then its
    callback, whether the task raised or not. What it raised is kept, as
    the real lane reports it rather than dropping it."""

    def __init__(self):
        self.raised = []

    def put(self, task, **_):
        try:
            task.action(*task.args, **task.kwargs)
        except Exception as e:
            self.raised.append(e)
        task.callback(*task.cb_args, **task.cb_kwargs)


@pytest.fixture
def session():
    built = ScopeSession.create(
        complete_settings(
            microscope='LS850T',
            objective_confirmed=True,
            turret_position=1,
            turret_objectives=dict(ASSIGNED),
        ),
        simulate=True,
    )
    yield built
    built.shutdown()


class _HeldClock:
    """Kivy's Clock before the event loop runs: scheduled work waits."""

    def __init__(self):
        self.pending = []

    def schedule_once(self, callback, timeout=0):
        self.pending.append(callback)

    def run(self):
        pending, self.pending = self.pending, []
        for callback in pending:
            callback(0)


@pytest.fixture
def stand(monkeypatch, session):
    fov = []
    popups = []
    clock = _HeldClock()
    monkeypatch.setattr(vc, 'Clock', clock)
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            scope=session.scope,
            session=session,
            objective_helper=session.objective_helper,
            io_executor=_RunNowExecutor(),
            lumaview=SimpleNamespace(scope=session.scope),
            motion_settings=SimpleNamespace(
                ids={
                    'microscope_settings_id': SimpleNamespace(
                        refresh_fov_labels=lambda: fov.append(True)
                    )
                }
            ),
        ),
    )
    monkeypatch.setattr(
        notification_popup, 'show_notification_popup', lambda **kw: popups.append(kw)
    )
    built = _Stand()
    built.fov = fov
    built.popups = popups
    built.clock = clock
    return built


def _down(stand):
    return [p for p in range(1, 5) if stand.ids[f'turret_pos_{p}_btn'].state == 'down']


class TestTheDisplayIsTheApisAnswer:
    def test_an_unknown_slot_shows_no_slot_and_an_unknown_objective(self, stand):
        stand.ids['turret_pos_1_btn'].state = 'down'
        stand.show_turret_state()
        assert _down(stand) == []
        assert stand.ids['objective_spinner2'].text == 'Unknown'
        assert stand.fov == [True]
        # Outside a run, an unknown objective asks -- once the event loop
        # runs, never from inside the display: the startup home's display
        # runs before the loop, where a popup opens invisible.
        assert stand.prompts == 0
        stand.clock.run()
        assert stand.prompts == 1

    def test_a_known_slot_shows_its_button_and_its_objective(self, stand, session):
        home_sim_scope(session.scope)
        session.scope.motion.move_turret(2)
        stand.show_turret_state(prompt=False)
        assert _down(stand) == [2]
        assert stand.ids['objective_spinner2'].text == '4x Oly'
        assert stand.prompts == 0

    def test_each_slot_shows_its_assignment(self, stand):
        stand.show_turret_state(prompt=False)
        labels = [stand.ids[f'turret_pos_{p}_btn'].text for p in range(1, 5)]
        # An assignment outside the catalogue is shown as assigned, because
        # it is; the objective there is unknown, and the spinner says so.
        assert labels == ['10x', '4x', '< 3 >', 'no such glass']


class TestAFailedMoveShowsNoSlot:
    def test_a_refused_move_leaves_no_button_down(self, stand, session):
        # Never homed: the API refuses the move. The display must not show
        # the slot that was asked for.
        stand.turret_select(3)
        raised = _app_ctx.ctx.io_executor.raised
        assert [type(e) for e in raised] == [AxisStateUnknownError]
        assert _down(stand) == []
        assert stand.ids['objective_spinner2'].text == 'Unknown'


class TestAPickReachesTheSessionOnce:
    def test_a_pick_assigns_the_slot_in_the_light_path(self, stand, session):
        home_sim_scope(session.scope)
        session.scope.motion.move_turret(3)
        stand.pick_objective('20x Oly')
        assert session.settings['turret_objectives'][3] == '20x Oly'
        # No other slot was written.
        assert session.settings['turret_objectives'][1] == '10x Oly'
        assert stand.ids['objective_spinner2'].text == '20x Oly'
        assert _down(stand) == [3]

    def test_a_pick_at_an_unknown_slot_is_refused_and_shown(self, stand, session):
        stand.pick_objective('20x Oly')
        assert len(stand.popups) == 1
        assert 'home the turret' in stand.popups[0]['message']
        assert stand.ids['objective_spinner2'].text == 'Unknown'
        assert session.settings['turret_objectives'][3] is None

    def test_the_spinner_reports_a_pick_without_taking_it(self):
        spinner = SimpleNamespace(text='10x Oly', is_open=True, dispatched=[])
        spinner.dispatch = lambda event, value: spinner.dispatched.append((event, value))
        PickSpinner._on_dropdown_select(spinner, None, '20x Oly')
        assert spinner.dispatched == [('on_pick', '20x Oly')]
        assert spinner.text == '10x Oly'
        assert spinner.is_open is False


class TestResetClearsTheApisSlot:
    def test_reset_clears_the_slot_in_the_light_path(self, stand, session):
        home_sim_scope(session.scope)
        session.scope.motion.move_turret(2)
        stand.reset_turret_objective()
        assert session.settings['turret_objectives'][2] is None
        assert session.settings['turret_objectives'][1] == '10x Oly'
        assert stand.ids['turret_pos_2_btn'].text == '< 2 >'

    def test_reset_at_an_unknown_slot_is_refused(self, session):
        with pytest.raises(ObjectiveUnknownError) as excinfo:
            session.clear_current_turret_objective()
        assert excinfo.value.reason == 'slot_unknown'
        assert session.settings['turret_objectives'] == {int(k): v for k, v in ASSIGNED.items()}
