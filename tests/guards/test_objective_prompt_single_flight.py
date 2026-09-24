# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""One objective question at a time, and only when an answer can matter.

Three triggers can fire the objective prompt on the same startup (the
unconfirmed-install gate, the unassigned-slot gate, and the turret move
that lands on the slot), so the prompt itself has to be single-flight:
the modal is cancel-less and every trigger asks the same question, so a
second popup would stack a duplicate whose answer silently overwrites
the first.

Two states make the question pointless rather than duplicated -- no
hardware this session (nothing in the light path, no capture to stamp)
and provisional settings (every settings write is refused, so the answer
would be silently lost). Both suppress it, and both leave the install
unconfirmed so the next usable session asks.
"""

import ast
import json
import shutil
from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import modules.settings_init as settings_init
import ui.notification_popup as notification_popup
from modules.exceptions import ConfigError
from modules.scope_session import ObjectiveQuestion, ScopeSession
from tests.ast_seams import (
    REPO_ROOT,
    direct_call_names,
    find_def,
    iter_package_modules,
    parse_module,
    production_modules,
    walk_defs,
)
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from ui.vertical_control import VerticalControl

SHIPPED_TEMPLATE = REPO_ROOT / 'data' / 'settings.json'

# Read at IMPORT time, before the reset fixture below can create or
# clear the flag -- test_the_single_flight_flag_starts_clear pins the
# module's own starting state, not the fixture's.
_FLAG_AT_IMPORT = getattr(notification_popup, '_objective_popup_open', 'MISSING')

CATALOGUE = ['4x Oly', '10x Oly', '20x Oly', '40x Oly']


# ---------------------------------------------------------------------------
# Kivy stand-ins
# ---------------------------------------------------------------------------
# The single-flight flag lives INSIDE show_objective_selection_popup, so
# these tests must run the real function -- monkeypatching it away would
# test the stand instead of the guard. The conftest kivy.uix stubs are
# bare subclassable shells (no bind / add_widget / open), so the widget
# classes the function reaches for are replaced with these instead.


class _FakeWidget:
    """Builds from kwargs, holds children, records bound handlers."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._handlers = {}
        self._children = []

    def bind(self, **handlers):
        for event, callback in handlers.items():
            self._handlers.setdefault(event, []).append(callback)

    def add_widget(self, widget):
        self._children.append(widget)

    def fire(self, event):
        for callback in list(self._handlers.get(event, [])):
            callback(self)


class _FakePopup(_FakeWidget):
    """Popup whose dismiss() actually fires its bound on_dismiss handlers.

    That is the whole point of the fake: the production code clears the
    single-flight flag from on_dismiss, so a stub that swallowed dismiss
    would make every one of these tests pass vacuously.
    """

    opened = None  # per-test list, installed by the fixture

    def open(self):
        self.is_open = True
        _FakePopup.opened.append(self)

    def dismiss(self):
        if not getattr(self, 'is_open', False):
            return
        self.is_open = False
        self.fire('on_dismiss')


@pytest.fixture
def popups(monkeypatch):
    """Install the widget fakes; return the list of opened popups."""
    opened = []
    monkeypatch.setattr(_FakePopup, 'opened', opened)
    for name in ('BoxLayout', 'Label', 'Button', 'Spinner'):
        monkeypatch.setattr(notification_popup, name, _FakeWidget)
    monkeypatch.setattr(notification_popup, 'Popup', _FakePopup)
    return opened


def confirm(popup, choice=None):
    """Click Confirm on an open popup -- the real production callback."""
    widgets = popup.content._children
    spinner = next(w for w in widgets if hasattr(w, 'values'))
    button = next(w for w in widgets if getattr(w, 'text', None) == 'Confirm')
    if choice is not None:
        spinner.text = choice
    button.fire('on_release')


@pytest.fixture(autouse=True)
def clear_single_flight(monkeypatch):
    """No test inherits another's outstanding-question flag.

    raising=False so a build without the flag yields honest assertion
    failures rather than a collection error.
    """
    monkeypatch.setattr(notification_popup, '_objective_popup_open', False, raising=False)


@pytest.fixture(autouse=True)
def clear_folded_continuations(monkeypatch):
    """No test inherits another's folded continuations."""
    monkeypatch.setattr(notification_popup, '_objective_popup_folded', [], raising=False)


@pytest.fixture(autouse=True)
def settings_are_keepable(monkeypatch):
    """Default every test to non-provisional settings; D4 overrides it."""
    monkeypatch.setattr(settings_init, 'rejected_current_json', None)


# ---------------------------------------------------------------------------
# The VerticalControl stand, on a REAL sim Session
# ---------------------------------------------------------------------------
# Borrows the REAL renderer methods off the class so the production body
# runs; only the Kivy widget tree and the FOV refresh are stood in for.
# The decision (ask / withhold / defer) is the Session's, so every case
# below runs a real simulated Session rather than scripting the answer.


class _Stand:
    prompt_if_objective_unknown = VerticalControl.prompt_if_objective_unknown
    _render_objective_question = VerticalControl._render_objective_question
    _apply_objective_answer = VerticalControl._apply_objective_answer
    # Borrowed, not stubbed: it decides whether a startup step waiting
    # on the objective runs at all.
    _resolve_objective = VerticalControl._resolve_objective

    def __init__(self):
        self.ids = {'objective_spinner2': _FakeWidget(text='')}
        for position in range(1, 5):
            self.ids[f'turret_pos_{position}_btn'] = _FakeWidget(text=str(position))
        self.shown = []

    def show_turret_state(self, prompt=True):
        # The display after an answer; its own tests run it for real.
        self.shown.append(prompt)


@pytest.fixture
def session():
    """A fresh install on a turret model: unconfirmed, slot 1 unassigned."""
    built = ScopeSession.create(
        complete_settings(
            microscope='LS850T',
            objective_confirmed=False,
            turret_position=1,
            turret_objectives={'1': None, '2': None, '3': None, '4': None},
            objective_id='20x Oly',
        ),
        simulate=True,
    )
    # The question names the live slot, so the turret is put in slot 1 --
    # where a real startup's home leaves it.
    home_sim_scope(built.scope)
    built.scope.motion.move_turret(1)
    yield built
    built.shutdown()


def _install_ctx(monkeypatch, session, *, no_hardware=False):
    """App context with the pieces the renderer reads."""
    session.scope._no_hardware = no_hardware
    ctx = SimpleNamespace(settings=session.settings, session=session)
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    return ctx


# ---------------------------------------------------------------------------
# D1 -- one question, however many triggers fire
# ---------------------------------------------------------------------------


class TestSingleFlight:
    def test_the_fresh_install_double_fire_opens_one_prompt(self, monkeypatch, popups, session):
        """Startup's gates can fire twice on a fresh install with an
        unassigned slot 1; the user must see ONE question, not two
        stacked modals whose answers race."""
        _install_ctx(monkeypatch, session)
        stand = _Stand()

        stand.prompt_if_objective_unknown()
        stand.prompt_if_objective_unknown()

        assert len(popups) == 1

    def test_answering_re_arms_the_prompt(self, monkeypatch, popups, session):
        """Single-flight, not once-per-process: the next unknowable
        objective still gets asked about."""
        _install_ctx(monkeypatch, session)
        stand = _Stand()

        stand.prompt_if_objective_unknown()
        confirm(popups[0], '10x Oly')
        assert session.settings['objective_confirmed'] is True
        assert session.settings['turret_objectives'][1] == '10x Oly'

        session.clear_turret_objective(1)
        stand.prompt_if_objective_unknown()
        assert len(popups) == 2

    def test_an_empty_catalogue_does_not_latch(self, monkeypatch, popups, session):
        """An empty catalogue is refused by the Session BEFORE the popup
        exists -- one notification, no modal -- and must not leave the
        flag set with no popup to clear it."""
        _install_ctx(monkeypatch, session)
        stand = _Stand()
        errors = []
        monkeypatch.setattr(
            notification_popup, 'show_notification_popup', lambda **kw: errors.append(kw['title'])
        )
        catalogue = session.objective_helper.get_objectives_list
        monkeypatch.setattr(session.objective_helper, 'get_objectives_list', lambda: [])

        stand.prompt_if_objective_unknown()
        assert popups == []
        assert len(errors) == 1

        monkeypatch.setattr(session.objective_helper, 'get_objectives_list', catalogue)
        stand.prompt_if_objective_unknown()
        assert len(popups) == 1


# ---------------------------------------------------------------------------
# D2 -- no hardware, no question
# ---------------------------------------------------------------------------


class TestNoHardwareSuppression:
    def test_no_hardware_suppresses_the_objective_prompt(self, monkeypatch, popups, session):
        _install_ctx(monkeypatch, session, no_hardware=True)
        stand = _Stand()

        stand.prompt_if_objective_unknown()

        assert popups == []
        # Unanswered, not answered-by-default: the next session with
        # hardware attached has to ask.
        assert session.settings['objective_confirmed'] is False

    def test_hardware_present_still_prompts(self, monkeypatch, popups, session):
        _install_ctx(monkeypatch, session, no_hardware=False)
        stand = _Stand()

        stand.prompt_if_objective_unknown()

        assert len(popups) == 1

    def test_suppressed_then_recovered(self, monkeypatch, popups, session):
        """Suppression must not latch either -- reconnecting hardware
        mid-session leaves the question still owed."""
        _install_ctx(monkeypatch, session, no_hardware=True)
        stand = _Stand()

        stand.prompt_if_objective_unknown()
        assert popups == []

        session.scope._no_hardware = False
        stand.prompt_if_objective_unknown()
        assert len(popups) == 1


# ---------------------------------------------------------------------------
# Lens-mandated: the flag cannot be stranded
# ---------------------------------------------------------------------------


class TestTheFlagCannotStrand:
    def test_raise_inside_apply_does_not_wedge_the_prompt(self, monkeypatch, popups, session):
        """The answer is applied AFTER dismiss and a failure applying it
        is shown, not raised -- so it cannot leave the app unable to ever
        ask again, nor exit it from the popup's callback."""
        from modules.notification_center import Severity
        from tests.shown_outcomes import capture_shown

        _install_ctx(monkeypatch, session)
        stand = _Stand()
        shown = capture_shown(monkeypatch)

        def _boom(objective_id, turret_position=None):
            raise ConfigError('turret write failed')

        monkeypatch.setattr(session, 'confirm_objective', _boom)

        stand.prompt_if_objective_unknown()
        confirm(popups[0], '10x Oly')
        assert [(n.severity, n.message) for n in shown] == [(Severity.ERROR, 'turret write failed')]

        stand.prompt_if_objective_unknown()
        assert len(popups) == 2

    def test_the_single_flight_flag_starts_clear(self):
        """A module-level flag imported already set would suppress the
        first prompt of every session."""
        assert _FLAG_AT_IMPORT is False


# ---------------------------------------------------------------------------
# D4 -- provisional settings, and their resolution
# ---------------------------------------------------------------------------


class TestProvisionalSettings:
    def test_provisional_settings_suppress_the_objective_prompt(self, monkeypatch, popups, session):
        """An answer given now would be refused by the writer, and the
        cancel-less modal would cover the question whose resolution is
        what makes answers saveable again."""
        monkeypatch.setattr(
            settings_init,
            'rejected_current_json',
            ('/data/current.json', 'invalid JSON'),
        )
        _install_ctx(monkeypatch, session)
        stand = _Stand()

        stand.prompt_if_objective_unknown()

        assert popups == []
        assert session.settings['objective_confirmed'] is False

    def test_resolution_releases_the_prompt(self, monkeypatch, popups, tmp_path, session):
        rejected = tmp_path / 'current.json'
        rejected.write_text('{ not json')
        monkeypatch.setattr(settings_init, 'rejected_current_json', (str(rejected), 'invalid JSON'))
        _install_ctx(monkeypatch, session)
        stand = _Stand()

        stand.prompt_if_objective_unknown()
        assert popups == []

        settings_init.retire_rejected_current_json()

        stand.prompt_if_objective_unknown()
        assert len(popups) == 1

    def test_an_answer_after_resolution_reaches_disk(self, tmp_path, monkeypatch):
        """Characterization pin: passes before and after this change.

        The suppression is only worth having if the answer becomes
        durable once settings stop being provisional -- this guards the
        durability the deferral is trading against.
        """
        data = tmp_path / 'data'
        data.mkdir()
        shutil.copy(SHIPPED_TEMPLATE, data / 'settings.json')
        shutil.copy(SHIPPED_TEMPLATE, data / 'current.json')
        # The factory builds the session's helpers from this root and refuses
        # to configure the scope without them.
        for name in ('objectives.json', 'labware.json'):
            shutil.copy(SHIPPED_TEMPLATE.parent / name, data / name)
        monkeypatch.setattr(settings_init, 'settings', None)
        monkeypatch.setattr(settings_init, 'rejected_current_json', None)
        session = ScopeSession.create(
            ScopeSession.load_user_settings(str(tmp_path)), source_path=str(tmp_path), simulate=True
        )
        # The real scope with its connection flags up, so the save still
        # reads the live runtime state it records the turret slot from.
        for flag in ('camera_connected', 'motor_connected', 'led_connected'):
            monkeypatch.setattr(type(session.scope), flag, property(lambda self: True))

        session.update_settings('objective_confirmed', True)
        session.save_settings()

        with open(data / 'current.json') as f:
            assert json.load(f)['objective_confirmed'] is True


# ---------------------------------------------------------------------------
# AST pins -- the seams a refactor could quietly reopen
# ---------------------------------------------------------------------------


class TestOneOpenerOneMessage:
    def test_the_objective_popup_has_one_production_opener(self):
        """Single-flight is enforced inside the popup helper, so a second
        opener would not duplicate the modal -- but a second CALLER that
        built its own popup would. Pin the funnel."""
        openers = set()
        for rel_path, tree in production_modules():
            for qualname, fn in walk_defs(tree.body):
                if 'show_objective_selection_popup' in direct_call_names(fn):
                    openers.add((rel_path, qualname))

        assert openers == {('ui/vertical_control.py', 'VerticalControl._render_objective_question')}

    def test_the_objective_prompt_has_one_message_builder(self):
        """Every trigger asks the same question. A per-trigger message
        variant is how the old first_run flag grew, and it made the same
        question read as two different ones."""
        fn = find_def(
            'ui/vertical_control.py',
            '_render_objective_question',
            class_name='VerticalControl',
        )
        assert fn is not None

        assert direct_call_names(fn).count('show_objective_selection_popup') == 1

    def test_the_question_carries_no_per_trigger_variant(self):
        """The decision moved into the Session with the flag it reads; the
        QUESTION it hands the renderer is one shape -- a position, a
        proposal and the choices -- so the message cannot vary by trigger."""
        import dataclasses

        assert [f.name for f in dataclasses.fields(ObjectiveQuestion)] == [
            'turret_position',
            'proposed',
            'choices',
        ]

    def test_no_hardcoded_objective_default_in_the_ui(self):
        """The prompt's pre-selection chain is slot -> stored objective ->
        first catalogue entry. A literal objective name in ui/ code is a
        fourth, invisible source of the same fact, so no string constant
        may carry one.

        Scoped to ui/ plus the Session that now owns the chain: the string
        legitimately appears in modules/lumascope_api/runtime_state.py
        docstring examples.
        """
        modules = list(iter_package_modules(['ui']))
        modules.append(('modules/scope_session.py', parse_module('modules/scope_session.py')))
        offenders = [
            rel_path
            for rel_path, module in modules
            for node in ast.walk(module)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and '20x Oly' in node.value
        ]
        assert offenders == []


class TestProvisionalResolutionReAsks:
    def test_the_provisional_resolution_re_asks(self):
        """Startup suppresses the question while settings are
        provisional; if the reset path did not re-ask, that install would
        never be asked at all."""
        on_start = find_def('lumaviewpro.py', 'on_start', class_name='LumaViewProApp')
        assert on_start is not None
        assert '_prompt_objective_if_needed' in direct_call_names(on_start)

        revert = find_def('lumaviewpro.py', '_revert', class_name='LumaViewProApp')
        assert revert is not None, 'the _revert closure inside _ask_about_rejected_settings is gone'
        assert '_prompt_objective_if_needed' in direct_call_names(revert)


# ---------------------------------------------------------------------------
# A request folded into the prompt already on screen is still answered
# ---------------------------------------------------------------------------


class TestTheFoldedRequestIsStillAnswered:
    """Showing one dialog instead of two is correct; losing the second
    caller's work is not.

    On a turreted scope whose current slot is unassigned this was not a
    race but the deterministic startup order: the blocking turret move
    schedules its question first, so Kivy's FIFO Clock renders that one
    and startup's own asker -- the one carrying the persisted-protocol
    load -- reached a single-flight guard that returned, taking the
    continuation with it. The saved protocol then silently never loaded,
    on every startup, and no test could see it because the prompt and the
    settings both looked exactly right afterwards.
    """

    def test_the_folded_caller_s_continuation_runs_on_the_one_answer(
        self, monkeypatch, popups, session
    ):
        _install_ctx(monkeypatch, session)
        stand = _Stand()
        ran = []

        stand.prompt_if_objective_unknown()
        stand.prompt_if_objective_unknown(on_resolved=lambda: ran.append('startup'))

        assert len(popups) == 1, 'still one dialog'
        assert ran == [], 'and nothing runs before it is answered'

        confirm(popups[0], '10x Oly')

        assert ran == ['startup']

    def test_the_folded_caller_gets_the_answer_the_user_gave(self, monkeypatch, popups, session):
        """The continuation is a startup step that reads the objective;
        running it before the answer landed would read the stale one."""
        _install_ctx(monkeypatch, session)
        stand = _Stand()
        seen = []

        stand.prompt_if_objective_unknown()
        stand.prompt_if_objective_unknown(
            on_resolved=lambda: seen.append(session.scope.runtime_state.get_current_objective_id())
        )
        confirm(popups[0], '4x Oly')

        assert seen == ['4x Oly']

    def test_one_answer_writes_one_slot_when_the_turret_moved_between_requests(
        self, monkeypatch, popups, session
    ):
        """The question on screen names slot 1. The turret then lands on
        slot 2, also unassigned, and that asks too -- folded into the prompt
        on screen. The answer is about slot 1 and is applied once; applying
        it again for the folded request wrote it into slot 2 as well, glass
        nobody had said was there."""
        _install_ctx(monkeypatch, session)
        stand = _Stand()

        stand.prompt_if_objective_unknown()
        session.scope.motion.move_turret(2)
        stand.prompt_if_objective_unknown()
        assert len(popups) == 1
        confirm(popups[0], '10x Oly')

        assert session.settings['turret_objectives'][1] == '10x Oly'
        assert session.settings['turret_objectives'][2] is None
        # And the display follows, free to ask about slot 2 now.
        assert stand.shown == [True]

    def test_it_runs_once(self, monkeypatch, popups, session):
        """A continuation that fires twice would load the saved protocol
        twice, and the second load happens after the first has applied."""
        _install_ctx(monkeypatch, session)
        stand = _Stand()
        ran = []

        stand.prompt_if_objective_unknown()
        stand.prompt_if_objective_unknown(on_resolved=lambda: ran.append('startup'))
        confirm(popups[0], '10x Oly')

        assert ran == ['startup']
        assert session.settings['turret_objectives'][1] == '10x Oly'
        assert session.settings['objective_confirmed'] is True

    def test_folding_re_arms_like_any_other_answer(self, monkeypatch, popups, session):
        """The fold must not leave the flag set: the next unknowable
        objective still gets asked about."""
        _install_ctx(monkeypatch, session)
        stand = _Stand()

        stand.prompt_if_objective_unknown()
        stand.prompt_if_objective_unknown(on_resolved=lambda: None)
        confirm(popups[0], '10x Oly')

        session.clear_turret_objective(1)
        stand.prompt_if_objective_unknown()

        assert len(popups) == 2

    def test_a_prompt_that_goes_away_unanswered_takes_its_folded_work_with_it(
        self, monkeypatch, popups, session
    ):
        """A continuation left behind would fire on the NEXT prompt's
        answer -- a startup step run against a question nobody asked it
        about. Unreachable while the modal is cancel-less; pinned so that
        adding a cancel path cannot quietly create it."""
        _install_ctx(monkeypatch, session)
        stand = _Stand()
        ran = []

        stand.prompt_if_objective_unknown()
        stand.prompt_if_objective_unknown(on_resolved=lambda: ran.append('startup'))
        popups[0].dismiss()

        stand.prompt_if_objective_unknown()
        confirm(popups[1], '10x Oly')

        assert ran == []
