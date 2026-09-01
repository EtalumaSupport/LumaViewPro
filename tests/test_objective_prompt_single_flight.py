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
from pathlib import Path
from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import modules.settings_init as settings_init
import ui.notification_popup as notification_popup
from modules.scope_session import ScopeSession
from tests.ast_seams import find_def, iter_package_modules, parse_module
from ui.vertical_control import VerticalControl

REPO_ROOT = Path(__file__).resolve().parents[1]
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
def settings_are_keepable(monkeypatch):
    """Default every test to non-provisional settings; D4 overrides it."""
    monkeypatch.setattr(settings_init, 'rejected_current_json', None)


# ---------------------------------------------------------------------------
# The VerticalControl stand
# ---------------------------------------------------------------------------
# Borrows the REAL prompt methods off the class (as the sibling prompt
# test does) so the production body runs; only the Kivy widget tree and
# the two hardware writes are stood in for.


class _Stand:
    prompt_objective_selection = VerticalControl.prompt_objective_selection
    maybe_prompt_objective_selection = VerticalControl.maybe_prompt_objective_selection

    def __init__(self, objectives=None):
        catalogue = CATALOGUE if objectives is None else objectives
        self.spinner = _FakeWidget(text='', values=list(catalogue))
        self.ids = {'objective_spinner2': self.spinner}
        self.turret_states = []
        self.set_calls = 0

    def load_objectives(self):
        pass

    def update_all_turret_btn_states(self, position):
        self.turret_states.append(position)

    def set_turret_objective(self):
        self.set_calls += 1


def _install_ctx(monkeypatch, settings, *, no_hardware=False):
    """App context with the pieces the prompt body reads."""
    ctx = SimpleNamespace(
        settings=settings,
        lumaview=SimpleNamespace(scope=SimpleNamespace(no_hardware=no_hardware)),
        session=SimpleNamespace(
            update_settings=lambda key, value: settings.__setitem__(key, value),
            # The real Session delegate, not a constant: the provisional
            # gate now asks the session, and the D4 tests below set the
            # module state the real delegate reads.
            settings_are_provisional=settings_init.settings_are_provisional,
        ),
    )
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    return ctx


def _fresh_install_settings():
    return {
        'objective_confirmed': False,
        'turret_position': 1,
        'turret_objectives': {1: None},
        'objective_id': '20x Oly',
    }


# ---------------------------------------------------------------------------
# D1 -- one question, however many triggers fire
# ---------------------------------------------------------------------------


class TestSingleFlight:
    def test_the_fresh_install_double_fire_opens_one_prompt(self, monkeypatch, popups):
        """Startup's two gates both fire on a fresh install with an
        unassigned slot 1; the user must see ONE question, not two
        stacked modals whose answers race."""
        _install_ctx(monkeypatch, _fresh_install_settings())
        stand = _Stand()

        stand.prompt_objective_selection(turret_position=1)
        stand.maybe_prompt_objective_selection(model_has_turret=True)

        assert len(popups) == 1

    def test_answering_re_arms_the_prompt(self, monkeypatch, popups):
        """Single-flight, not once-per-process: the next unknowable
        objective still gets asked about."""
        settings = _fresh_install_settings()
        _install_ctx(monkeypatch, settings)
        stand = _Stand()

        stand.prompt_objective_selection(turret_position=1)
        confirm(popups[0], '10x Oly')
        assert settings['objective_confirmed'] is True

        stand.prompt_objective_selection(turret_position=2)
        assert len(popups) == 2

    def test_an_empty_catalogue_does_not_latch(self, monkeypatch, popups):
        """The catalogue-empty return happens BEFORE the popup exists, so
        it must not leave the flag set with no popup to clear it."""
        _install_ctx(monkeypatch, _fresh_install_settings())
        stand = _Stand(objectives=[])

        stand.prompt_objective_selection(turret_position=1)
        assert popups == []

        stand.spinner.values = list(CATALOGUE)
        stand.prompt_objective_selection(turret_position=1)
        assert len(popups) == 1


# ---------------------------------------------------------------------------
# D2 -- no hardware, no question
# ---------------------------------------------------------------------------


class TestNoHardwareSuppression:
    def test_no_hardware_suppresses_the_objective_prompt(self, monkeypatch, popups):
        settings = _fresh_install_settings()
        _install_ctx(monkeypatch, settings, no_hardware=True)
        stand = _Stand()

        stand.maybe_prompt_objective_selection(model_has_turret=True)

        assert popups == []
        # Unanswered, not answered-by-default: the next session with
        # hardware attached has to ask.
        assert settings['objective_confirmed'] is False

    def test_hardware_present_still_prompts(self, monkeypatch, popups):
        _install_ctx(monkeypatch, _fresh_install_settings(), no_hardware=False)
        stand = _Stand()

        stand.maybe_prompt_objective_selection(model_has_turret=True)

        assert len(popups) == 1

    def test_suppressed_then_recovered(self, monkeypatch, popups):
        """Suppression must not latch either -- reconnecting hardware
        mid-session leaves the question still owed."""
        ctx = _install_ctx(monkeypatch, _fresh_install_settings(), no_hardware=True)
        stand = _Stand()

        stand.prompt_objective_selection(turret_position=1)
        assert popups == []

        ctx.lumaview.scope.no_hardware = False
        stand.prompt_objective_selection(turret_position=1)
        assert len(popups) == 1


# ---------------------------------------------------------------------------
# Lens-mandated: the flag cannot be stranded
# ---------------------------------------------------------------------------


class TestTheFlagCannotStrand:
    def test_raise_inside_apply_does_not_wedge_the_prompt(self, monkeypatch, popups):
        """The answer is applied AFTER dismiss, so a failure applying it
        cannot leave the app unable to ever ask again."""
        _install_ctx(monkeypatch, _fresh_install_settings())
        stand = _Stand()

        def _boom():
            raise RuntimeError('turret write failed')

        monkeypatch.setattr(stand, 'set_turret_objective', _boom)

        stand.prompt_objective_selection(turret_position=1)
        with pytest.raises(RuntimeError):
            confirm(popups[0], '10x Oly')

        stand.prompt_objective_selection(turret_position=1)
        assert len(popups) == 2

    def test_the_single_flight_flag_starts_clear(self):
        """A module-level flag imported already set would suppress the
        first prompt of every session."""
        assert _FLAG_AT_IMPORT is False


# ---------------------------------------------------------------------------
# D4 -- provisional settings, and their resolution
# ---------------------------------------------------------------------------


class TestProvisionalSettings:
    def test_provisional_settings_suppress_the_objective_prompt(self, monkeypatch, popups):
        """An answer given now would be refused by the writer, and the
        cancel-less modal would cover the question whose resolution is
        what makes answers saveable again."""
        monkeypatch.setattr(
            settings_init,
            'rejected_current_json',
            ('/data/current.json', 'invalid JSON'),
        )
        settings = _fresh_install_settings()
        _install_ctx(monkeypatch, settings)
        stand = _Stand()

        stand.maybe_prompt_objective_selection(model_has_turret=True)

        assert popups == []
        assert settings['objective_confirmed'] is False

    def test_resolution_releases_the_prompt(self, monkeypatch, popups, tmp_path):
        rejected = tmp_path / 'current.json'
        rejected.write_text('{ not json')
        monkeypatch.setattr(settings_init, 'rejected_current_json', (str(rejected), 'invalid JSON'))
        _install_ctx(monkeypatch, _fresh_install_settings())
        stand = _Stand()

        stand.maybe_prompt_objective_selection(model_has_turret=True)
        assert popups == []

        settings_init.retire_rejected_current_json()

        stand.maybe_prompt_objective_selection(model_has_turret=True)
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
        monkeypatch.setattr(settings_init, 'settings', None)
        monkeypatch.setattr(settings_init, 'rejected_current_json', None)
        session = ScopeSession.create_headless(source_path=str(tmp_path))
        monkeypatch.setattr(
            session,
            'scope',
            SimpleNamespace(camera_connected=True, motor_connected=True, led_connected=True),
        )

        session.update_settings('objective_confirmed', True)
        session.save_settings()

        with open(data / 'current.json') as f:
            assert json.load(f)['objective_confirmed'] is True


# ---------------------------------------------------------------------------
# AST pins -- the seams a refactor could quietly reopen
# ---------------------------------------------------------------------------


def _direct_call_names(fn) -> list[str]:
    """Names called in this function's OWN body; nested defs excluded.

    Without the exclusion an outer function would be credited with every
    call its closures make, and the "exactly one opener" count below
    would silently drift.
    """
    names = []
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
        stack.extend(ast.iter_child_nodes(node))
    return names


def _walk_defs(body, prefix=''):
    """Yield (qualname, node) for every def, methods and closures included."""
    for node in body:
        if isinstance(node, ast.ClassDef):
            yield from _walk_defs(node.body, f'{prefix}{node.name}.')
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = f'{prefix}{node.name}'
            yield qualname, node
            yield from _walk_defs(node.body, f'{qualname}.')


def _production_modules():
    """Every production module the prompt could be opened from.

    iter_package_modules walks packages only, and the startup path lives
    in the top-level lumaviewpro.py -- the one module a package-only
    sweep would miss.
    """
    yield from iter_package_modules(['ui', 'modules'])
    yield 'lumaviewpro.py', parse_module('lumaviewpro.py')


class TestOneOpenerOneMessage:
    def test_the_objective_popup_has_one_production_opener(self):
        """Single-flight is enforced inside the popup helper, so a second
        opener would not duplicate the modal -- but a second CALLER that
        built its own popup would. Pin the funnel."""
        openers = set()
        for rel_path, tree in _production_modules():
            for qualname, fn in _walk_defs(tree.body):
                if 'show_objective_selection_popup' in _direct_call_names(fn):
                    openers.add((rel_path, qualname))

        assert openers == {('ui/vertical_control.py', 'VerticalControl.prompt_objective_selection')}

    def test_the_objective_prompt_has_one_message_builder(self):
        """Every trigger asks the same question. A per-trigger message
        variant is how the old first_run flag grew, and it made the same
        question read as two different ones."""
        fn = find_def(
            'ui/vertical_control.py',
            'prompt_objective_selection',
            class_name='VerticalControl',
        )
        assert fn is not None

        assert _direct_call_names(fn).count('show_objective_selection_popup') == 1

        identifiers = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.arg):
                identifiers.add(node.arg)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                identifiers.add(node.arg)
        assert 'first_run' not in identifiers

    def test_no_hardcoded_objective_default_in_the_ui(self):
        """The prompt's pre-selection chain is slot -> stored objective ->
        first catalogue entry. A literal objective name in ui/ code is a
        fourth, invisible source of the same fact, so no string constant
        may carry one.

        Scoped to ui/ deliberately: the string legitimately appears in
        modules/lumascope_api/runtime_state.py docstring examples.
        """
        offenders = [
            rel_path
            for rel_path, module in iter_package_modules(['ui'])
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
        assert '_prompt_objective_if_needed' in _direct_call_names(on_start)

        revert = find_def('lumaviewpro.py', '_revert', class_name='LumaViewProApp')
        assert revert is not None, 'the _revert closure inside _ask_about_rejected_settings is gone'
        assert '_prompt_objective_if_needed' in _direct_call_names(revert)
