# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The GUI shows where the API says the turret is, and decides nothing.

Every turret action -- a press, a home, a run's step, an objective answer
-- ends in one display, VerticalControl.show_turret_state, which shows the
slot the API reports, each slot's assignment and the active objective. A
failed move therefore shows the turret in no known slot, never in the one
that was asked for. Each slot button is labelled with the objective
assigned to it. Pressing a slot nothing is assigned to asks which objective
it holds; on a scope without a turret, one Assign Objective button asks
(#861). The widget methods below are the real ones, run against a real
simulated LS850T session; only the widget tree is stood in for.
"""

from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import ui.notification_popup as notification_popup
import ui.vertical_control as vc
from modules.exceptions import ObjectiveUnknownError
from modules.sequential_io_executor import ENQUEUED
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from ui.vertical_control import VerticalControl

ASSIGNED = {'1': '10x Oly', '2': '4x Oly', '3': None, '4': 'no such glass'}


class _Stand:
    show_turret_state = VerticalControl.show_turret_state
    turret_select = VerticalControl.turret_select
    reset_turret_objective = VerticalControl.reset_turret_objective
    assign_objective = VerticalControl.assign_objective
    _render_objective_question = VerticalControl._render_objective_question
    _apply_objective_answer = VerticalControl._apply_objective_answer
    _resolve_objective = VerticalControl._resolve_objective

    def __init__(self):
        self.ids = {'assign_objective_btn': SimpleNamespace(text='Assign Objective')}
        for position in range(1, 5):
            self.ids[f'turret_pos_{position}_btn'] = SimpleNamespace(
                text=f'< {position} >', state='normal'
            )
        self.prompts = 0

    def prompt_if_objective_unknown(self, on_resolved=None):
        self.prompts += 1


class _AskingStand(_Stand):
    """The stand with the production prompt: an unknown objective asks the
    Session's question, rendered by the one popup."""

    prompt_if_objective_unknown = VerticalControl.prompt_if_objective_unknown


class _RunNowExecutor:
    """The io lane as the display sees it: the task runs, then its
    callback when it has one, whether the task raised or not. What it
    raised is kept, as the real lane reports it rather than dropping it."""

    def __init__(self):
        self.raised = []

    def put(self, task, **_):
        try:
            task.action(*task.args, **task.kwargs)
        except Exception as e:
            self.raised.append(e)
        if task.callback is not None:
            task.callback(*task.cb_args, **task.cb_kwargs)
        return ENQUEUED


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


def _built(monkeypatch, session, cls):
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
    built = cls()
    built.fov = fov
    built.popups = popups
    built.clock = clock
    return built


@pytest.fixture
def stand(monkeypatch, session):
    return _built(monkeypatch, session, _Stand)


@pytest.fixture
def asking_stand(monkeypatch, session):
    return _built(monkeypatch, session, _AskingStand)


@pytest.fixture
def asked(monkeypatch):
    # A stand-in by design: the popup's widget tree. The question it is
    # handed and the answer it returns are the production ones.
    shown = []
    monkeypatch.setattr(
        notification_popup, 'show_objective_selection_popup', lambda **kw: shown.append(kw)
    )
    return shown


def _press(stand, position):
    """A slot button's press, and the event loop's turn after it."""
    stand.turret_select(position)
    stand.clock.run()


def _down(stand):
    return [p for p in range(1, 5) if stand.ids[f'turret_pos_{p}_btn'].state == 'down']


class TestTheDisplayIsTheApisAnswer:
    def test_an_unknown_slot_shows_no_slot_and_an_unknown_objective(self, stand):
        stand.ids['turret_pos_1_btn'].state = 'down'
        stand.show_turret_state()
        assert _down(stand) == []
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
        assert stand.ids['turret_pos_2_btn'].text == '4x Oly'
        assert stand.prompts == 0

    def test_each_slot_shows_its_assignment(self, stand):
        stand.show_turret_state(prompt=False)
        labels = [stand.ids[f'turret_pos_{p}_btn'].text for p in range(1, 5)]
        # Each by its catalogue name, which the button wraps to two lines. An
        # assignment outside the catalogue is shown as assigned, because it
        # is; the objective there is unknown.
        assert labels == ['10x Oly', '4x Oly', '< 3 >', 'no such glass']


class TestAFailedMoveShowsNoSlot:
    def test_a_refused_move_leaves_no_button_down(self, stand, session, monkeypatch, centre_posts):
        # Never homed: the API refuses the move, and the boundary shows the
        # refusal. The display must not show the slot that was asked for.
        stand.turret_select(3)
        assert [n.title for n in centre_posts] == ['Scope Not Homed']
        assert _down(stand) == []


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


class TestAnEmptySlotAsksWhatItHolds:
    """Pressing a slot nothing is assigned to turns the turret there and asks
    the Session's question in the one popup: Confirm assigns it, Cancel
    leaves it empty, and the next press asks again. Reset, then a press,
    replaces a slot's glass."""

    def test_confirm_assigns_the_slot_and_names_its_button(self, asking_stand, session, asked):
        home_sim_scope(session.scope)
        _press(asking_stand, 3)
        (popup,) = asked
        assert popup['message'].startswith(
            'Please confirm the objective installed at turret position 3.'
        )
        popup['on_confirm']('20x Phase')
        assert session.settings['turret_objectives'][3] == '20x Phase'
        assert asking_stand.ids['turret_pos_3_btn'].text == '20x Phase'
        assert _down(asking_stand) == [3]
        assert session.scope.runtime_state.get_current_objective_id() == '20x Phase'

    def test_cancel_leaves_the_slot_empty_and_the_next_press_asks_again(
        self, asking_stand, session, asked
    ):
        home_sim_scope(session.scope)
        _press(asking_stand, 3)
        asked[0]['on_cancel']()
        assert session.settings['turret_objectives'] == {int(k): v for k, v in ASSIGNED.items()}
        # Every capture is refused naming the slot until it is assigned.
        with pytest.raises(ObjectiveUnknownError) as refused:
            session.scope.runtime_state.resolve_current_objective()
        assert refused.value.reason == 'slot_unassigned'

        _press(asking_stand, 3)
        assert len(asked) == 2

    def test_reset_then_a_press_replaces_the_slot_s_glass(self, asking_stand, session, asked):
        home_sim_scope(session.scope)
        _press(asking_stand, 1)
        asking_stand.reset_turret_objective()
        asking_stand.clock.run()
        assert asked == [], 'Reset asked straight back what the slot holds'

        _press(asking_stand, 1)
        (popup,) = asked
        popup['on_confirm']('40x w/collar')
        assert session.settings['turret_objectives'][1] == '40x w/collar'
        assert asking_stand.ids['turret_pos_1_btn'].text == '40x w/collar'

    def test_pressing_an_assigned_slot_asks_nothing(self, asking_stand, session, asked):
        home_sim_scope(session.scope)
        _press(asking_stand, 2)
        assert asked == []
        assert session.scope.runtime_state.get_current_objective_id() == '4x Oly'


class TestATurretlessScopeAssignsWithOneButton:
    @pytest.fixture
    def turretless(self):
        built = ScopeSession.create(
            complete_settings(microscope='LS850', objective_confirmed=True, objective_id='4x Oly'),
            simulate=True,
        )
        yield built
        built.shutdown()

    def test_the_button_names_the_objective_and_its_answer_selects_one(
        self, monkeypatch, turretless, asked
    ):
        stand = _built(monkeypatch, turretless, _AskingStand)
        stand.show_turret_state(prompt=False)
        assert stand.ids['assign_objective_btn'].text == '4x Oly'

        stand.assign_objective()
        (popup,) = asked
        assert popup['message'].startswith('Please confirm the installed objective.')
        popup['on_confirm']('20x Phase')
        assert turretless.scope.runtime_state.get_current_objective_id() == '20x Phase'
        assert stand.ids['assign_objective_btn'].text == '20x Phase'

    def test_the_assign_row_shows_exactly_when_the_turret_rows_do_not(self):
        from ui.motion_settings import MotionSettings

        rows = ('turret_selection_label', 'turret_btn_box', 'turret_objective_btn_box')
        ids = {row: SimpleNamespace(visible=None) for row in (*rows, 'assign_objective_box')}
        panel = SimpleNamespace(ids={'verticalcontrol_id': SimpleNamespace(ids=ids)})
        shown = {}
        for has_turret in (True, False):
            MotionSettings.set_turret_control_visibility(panel, has_turret)
            shown[has_turret] = [ids[row].visible for row in (*rows, 'assign_objective_box')]
        assert shown == {True: [True, True, True, False], False: [False, False, False, True]}


class TestTheAskedQuestion:
    """``current_slot_objective_question``: the owed question's proposal
    rule, for the slot in the light path, at any time."""

    def test_it_proposes_the_slot_s_assignment(self, session):
        home_sim_scope(session.scope)
        session.scope.motion.move_turret(2)
        question = session.current_slot_objective_question()
        assert (question.turret_slot, question.proposed) == (2, '4x Oly')

    def test_a_slot_holding_glass_the_catalogue_lacks_proposes_the_default(self, session):
        from modules.objectives_loader import DEFAULT_PROPOSED_OBJECTIVE_ID

        home_sim_scope(session.scope)
        session.scope.motion.move_turret(4)
        question = session.current_slot_objective_question()
        assert question.proposed == DEFAULT_PROPOSED_OBJECTIVE_ID
        session.confirm_objective(question.proposed, slot=question.turret_slot)
        assert session.settings['turret_objectives'][4] == DEFAULT_PROPOSED_OBJECTIVE_ID

    def test_an_unknown_slot_is_refused(self, session):
        with pytest.raises(ObjectiveUnknownError) as excinfo:
            session.current_slot_objective_question()
        assert excinfo.value.reason == 'slot_unknown'

    def test_a_scope_with_no_turret_is_asked_with_no_slot(self):
        """Without a turret the question names no slot, and its answer goes
        back through ``confirm_objective`` with none."""
        turretless = ScopeSession.create(
            complete_settings(microscope='LS850', objective_confirmed=True, objective_id='4x Oly'),
            simulate=True,
        )
        try:
            question = turretless.current_slot_objective_question()
            turretless.confirm_objective('20x Phase', slot=question.turret_slot)
            objective = turretless.scope.runtime_state.get_current_objective_id()
        finally:
            turretless.shutdown()
        assert (question.turret_slot, question.proposed) == (None, '4x Oly')
        assert objective == '20x Phase'


def test_a_full_home_shows_the_turret(monkeypatch):
    """The XY Home homes every axis, the turret included, and its display
    shows the turret: the firmware's home leaves it on slot 1."""
    from ui import ui_helpers

    shown = []
    vertical = SimpleNamespace(
        update_gui=lambda vertical_control=False: None,
        show_turret_state=lambda prompt=True: shown.append(prompt),
    )
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            scope=SimpleNamespace(capabilities=SimpleNamespace(has_turret=True)),
            motion_settings=SimpleNamespace(
                ids={'verticalcontrol_id': vertical},
                update_xy_stage_control_gui=lambda: None,
            ),
        ),
    )
    ui_helpers._handle_ui_update_for_axis('ALL')
    assert shown == [True]
