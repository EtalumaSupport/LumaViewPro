# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Autofocus button runs through ``ProtocolRunner.run_autofocus`` and draws its run.

There is one implementation of "autofocus once, here". The button once
built the whole run itself -- the objective check, the position, the layer
config, the protocol, ``prepare`` and ``start`` -- beside the member built
for scripts and REST, so the two could drift. Now the button states only
what a running GUI knows (the open drawer, its own attended trigger, the
live engineering flag) and the member decides everything else, every
refusal included.

A press is a Stop only when the engine says the run this button started is
still live. One redraw styles the button, from what the engine reports
about that run and no other: a protocol's autofocus steps are not this
button's to show.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in ('kivy.clock', 'kivy.uix'):
    sys.modules.setdefault(_name, MagicMock())

_boxlayout = types.ModuleType('kivy.uix.boxlayout')
_boxlayout.BoxLayout = _StubWidget
sys.modules.setdefault('kivy.uix.boxlayout', _boxlayout)

import modules.app_context as _app_ctx
import ui.ui_helpers as ui_helpers
import ui.vertical_control as vc
from modules.exceptions import ProtocolRunRefusedError, RunAlreadyEndedError
from modules.run_outcome import PendingRunOutcome
from tests.pool_fakes import run_task_now


class _Button(vc.VerticalControl):
    """The real class with the widget tree stubbed, the toggle as a first
    press leaves it."""

    def __init__(self):
        self.button = SimpleNamespace(state='down', text='Autofocus')
        self.ids = {'autofocus_id': self.button}
        self.autofocus_pending = False
        self._autofocus_run = None
        self._af_safety_run = None
        self._af_safety_event = None
        self.armed = []

    def _schedule_af_safety_timer(self, run):
        self.armed.append(run)


@pytest.fixture
def held():
    """Pool tasks, when a test holds them instead of running them at once."""
    return []


@pytest.fixture
def pressed(monkeypatch, held):
    """A GUI with nothing running; the pool runs each request at once."""
    from modules.sequential_io_executor import ENQUEUED
    from tests.shown_outcomes import capture_shown

    shown = capture_shown(monkeypatch)
    pool = MagicMock()

    def _put(task):
        if held == ['hold']:
            held.append(task)
        else:
            run_task_now(task)
        return ENQUEUED

    pool.put.side_effect = _put
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn, timeout=0: fn(0))
    member = MagicMock()
    handle = PendingRunOutcome()
    member.run_autofocus.return_value = handle
    engine = MagicMock()
    engine.is_live_run.return_value = False
    engine.is_stopping.return_value = False
    session = MagicMock()
    session.create_protocol_runner.return_value = member
    # Nothing holds the scope; a MagicMock's own answer would be truthy.
    session.held_by_other.return_value = False
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            session=session,
            sequenced_capture_runner=engine,
            image_settings=MagicMock(),
            engineering_mode=True,
            worker_pool=pool,
        ),
    )
    monkeypatch.setattr(vc, 'require_file_writes_idle', lambda operation: True)
    monkeypatch.setattr(vc, 'live_display_callbacks', dict)
    monkeypatch.setattr(vc.gui_logger, 'button', lambda *a, **kw: None)
    monkeypatch.setattr(vc.common_utils, 'get_opened_layer', lambda _settings: 'Green')
    return SimpleNamespace(
        member=member, engine=engine, handle=handle, shown=shown, pool=pool, button=_Button()
    )


def _live(engine, *runs):
    engine.is_live_run.side_effect = lambda run: run is not None and any(run is r for r in runs)


def test_the_button_states_only_what_the_gui_knows(pressed):
    pressed.button.run_autofocus_from_ui()

    pressed.member.run_autofocus.assert_called_once()
    kwargs = pressed.member.run_autofocus.call_args.kwargs
    assert kwargs['layer'] == 'Green'
    assert kwargs['run_trigger_source'] == 'autofocus', 'the attended trigger'
    assert kwargs['engineering_mode'] is True
    assert kwargs['save_characterization_data'] is True, (
        'in engineering mode the sweep saves its characterization data'
    )
    # Nothing else about the run is the button's to say.
    assert set(kwargs) == {
        'layer',
        'save_characterization_data',
        'callbacks',
        'run_trigger_source',
        'engineering_mode',
    }


def test_the_button_starts_no_run_of_its_own(pressed):
    pressed.button.run_autofocus_from_ui()

    assert not pressed.engine.prepare.called
    assert not pressed.engine.start.called
    assert pressed.button._autofocus_run is pressed.handle, 'the handle is what its Stop names'


def test_a_refused_start_is_shown_once_and_the_button_draws_idle(pressed):
    pressed.member.run_autofocus.side_effect = ProtocolRunRefusedError(
        reason='objective_unknown', title='Objective Unknown', message='m'
    )

    pressed.button.run_autofocus_from_ui()

    assert [n.title for n in pressed.shown] == ['Objective Unknown']
    assert (pressed.button.button.state, pressed.button.button.text) == ('normal', 'Autofocus')
    assert pressed.button.autofocus_pending is False, 'the button must come back for the next press'


def test_an_unexpected_failure_is_one_fault_and_the_button_draws_idle(pressed):
    pressed.member.run_autofocus.side_effect = TypeError('bad call')

    pressed.button.run_autofocus_from_ui()

    assert len(pressed.shown) == 1
    assert pressed.button.button.state == 'normal'
    assert pressed.button.autofocus_pending is False


def test_a_started_run_draws_focusing(pressed):
    _live(pressed.engine, pressed.handle)

    pressed.button.run_autofocus_from_ui()

    assert (pressed.button.button.state, pressed.button.button.text) == ('down', 'Focusing...')


def test_a_second_press_stops_its_own_run_ahead_of_queued_work(pressed):
    from modules.sequential_io_executor import PRIORITY_HIGH

    pressed.button._autofocus_run = pressed.handle
    _live(pressed.engine, pressed.handle)
    pressed.engine.is_stopping.side_effect = lambda run: pressed.engine.reset.called

    pressed.button.run_autofocus_from_ui()

    task = pressed.pool.put.call_args.args[0]
    assert task.priority == PRIORITY_HIGH, 'a Stop must not wait behind queued work'
    pressed.engine.reset.assert_called_once_with(pressed.handle)
    assert not pressed.member.run_autofocus.called, 'a Stop is not a start'
    assert pressed.button.button.text == 'Stopping...'


def test_a_stop_that_finds_its_run_already_ended_shows_nothing(pressed):
    pressed.button._autofocus_run = pressed.handle
    _live(pressed.engine, pressed.handle)

    def _ended(run):
        _live(pressed.engine)  # the run ended between the press and the pool
        raise RunAlreadyEndedError('the run already ended')

    pressed.engine.reset.side_effect = _ended

    pressed.button.run_autofocus_from_ui()

    assert pressed.shown == []
    assert (pressed.button.button.state, pressed.button.button.text) == ('normal', 'Autofocus')


def test_a_refused_stop_leaves_the_button_showing_its_own_run(pressed):
    pressed.button._autofocus_run = pressed.handle
    _live(pressed.engine, pressed.handle)
    pressed.engine.reset.side_effect = ProtocolRunRefusedError(
        reason='run_not_live', title='Not Running', message='That run is not the one running.'
    )

    pressed.button.run_autofocus_from_ui()

    assert [n.title for n in pressed.shown] == ['Not Running']
    assert (pressed.button.button.state, pressed.button.button.text) == ('down', 'Focusing...')


def test_the_button_is_disabled_while_its_own_request_is_in_flight(pressed, held):
    held.append('hold')

    pressed.button.run_autofocus_from_ui()

    assert pressed.button.autofocus_pending is True, (
        'a second press must not race the first to the pool'
    )
    task = held[1]
    run_task_now(task)
    assert pressed.button.autofocus_pending is False, (
        "the request's own redraw brings the button back"
    )
    assert pressed.member.run_autofocus.call_count == 1


def test_a_protocols_autofocus_is_not_this_buttons_to_show(pressed):
    """Another run is live -- a protocol with autofocus steps -- and this
    button started nothing: it draws idle."""
    _live(pressed.engine, PendingRunOutcome())
    pressed.button.button.state = 'normal'

    pressed.button.draw_autofocus_button()

    assert (pressed.button.button.state, pressed.button.button.text) == ('normal', 'Autofocus')
    assert pressed.button.armed == []


def test_the_panels_runs_register_no_autofocus_indicator():
    """The protocol panel's runs no longer light this button: no AF-step
    callback is handed to the engine from the panel."""
    import ast

    from tests.ast_seams import parse_module

    tree = parse_module('ui/protocol_settings.py')
    keys = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not keys & {'autofocus_in_progress', 'autofocus_complete'}


class _HeldClock:
    """Kivy's clock, holding each scheduled callback so a test can fire it."""

    def __init__(self):
        self.scheduled = []

    def schedule_once(self, fn, timeout=0):
        self.scheduled.append(fn)
        return fn

    def unschedule(self, event):
        if event in self.scheduled:
            self.scheduled.remove(event)

    def create_trigger(self, fn, timeout=0):
        return MagicMock()


@pytest.fixture
def real(pressed, monkeypatch):
    """The real VerticalControl, constructed, with the stuck-AF bound's clock held."""
    clock = _HeldClock()
    monkeypatch.setattr(vc, 'Clock', clock)
    button = vc.VerticalControl()
    button.ids = {'autofocus_id': SimpleNamespace(state='down', text='Autofocus')}
    # A Kivy property; the stubbed kivy has no descriptor to give it a default.
    button.autofocus_pending = False
    return SimpleNamespace(button=button, clock=clock)


def test_a_new_button_holds_no_run_and_no_bound(real):
    assert real.button._autofocus_run is None
    assert real.button._af_safety_run is None


def test_a_stuck_autofocus_is_stopped_by_its_bound(pressed, real):
    _live(pressed.engine, pressed.handle)
    real.button.run_autofocus_from_ui()
    [bound] = real.clock.scheduled

    bound(0)

    pressed.engine.reset.assert_called_once_with(pressed.handle)


def test_a_bound_that_outlived_its_run_leaves_the_next_autofocus_alone(pressed, real):
    from modules.run_outcome import PendingRunOutcome

    first, second = pressed.handle, PendingRunOutcome()
    _live(pressed.engine, first)
    real.button.run_autofocus_from_ui()
    [stale] = real.clock.scheduled

    _live(pressed.engine)  # the first run ends
    pressed.member.run_autofocus.return_value = second
    _live(pressed.engine, second)
    real.button.run_autofocus_from_ui()

    stale(0)

    assert not pressed.engine.reset.called, (
        'a timer armed for the first run fired while the second was live, and stopped it'
    )


def test_the_button_is_disabled_while_its_request_is_in_flight_in_the_kv():
    from tests.ast_seams import REPO_ROOT

    kv = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text()
    idx = kv.find('id: autofocus_id')
    assert idx > 0
    window = kv[idx : idx + 500].splitlines()
    disabled = [line for line in window if line.strip().startswith('disabled:')]
    assert disabled, 'the Autofocus button has no disabled binding'
    assert 'root.autofocus_held or root.autofocus_pending' in disabled[0]


def test_the_button_greys_while_anything_else_holds_the_scope(pressed):
    """The Session answers for this button's own run: another holder greys it,
    its own run leaves it live as that run's Stop."""
    import modules.app_context as app_context

    own = object()
    pressed.button._autofocus_run = own
    asked = []

    for held in (True, False):
        app_context.ctx.session.held_by_other = lambda run, held=held: asked.append(run) or held
        pressed.button.draw_autofocus_button()
        assert pressed.button.autofocus_held is held

    assert asked == [own, own], 'the button must ask about the run it started'
