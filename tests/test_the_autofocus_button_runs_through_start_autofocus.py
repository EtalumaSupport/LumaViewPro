# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Autofocus button runs through ``ProtocolRunner.start_autofocus`` and draws its run.

There is one implementation of "autofocus once, here". The button once
built the whole run itself -- the objective check, the position, the layer
config, the protocol, ``prepare`` and ``start`` -- beside the member built
for scripts and REST, so the two could drift. Now the button states only
what a running GUI knows (the open drawer, the live engineering flag),
the GUI's stamp on the worker says who started the run, and the member
decides everything else, every refusal included.

A press is a Stop only when the engine says the run this button started is
still live. One redraw styles the button, from what the engine reports
about that run and no other: a protocol's autofocus steps are not this
button's to show.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


import ui.ui_helpers as ui_helpers
import ui.vertical_control as vc
from modules.caller import Caller, current_caller
from modules.protocol_runner import ProtocolRunner
from tests.pool_fakes import run_task_now
from tests.scope_fakes import home_sim_scope

_HOLD_WAIT_S = 10.0
_END_WAIT_S = 60.0


class _Button(vc.VerticalControl):
    """The real class with the widget tree stubbed, the toggle as a first
    press leaves it."""

    def __init__(self):
        self.button = SimpleNamespace(state='down', text='Autofocus')
        self.ids = {'autofocus_id': self.button}
        self.autofocus_pending = False
        self._autofocus_run = None


@pytest.fixture
def held():
    """Pool tasks, when a test holds them instead of running them at once."""
    return []


@pytest.fixture
def pressed(gui_ctx, monkeypatch, held, centre_posts):
    """The GUI over an unhomed simulated session in engineering mode, nothing
    running; the pool runs each request at once, and each start the member
    is asked for is recorded with who asked, then made."""
    from modules.sequential_io_executor import ENQUEUED

    pool = MagicMock()

    def _put(task):
        if held == ['hold']:
            held.append(task)
        else:
            run_task_now(task)
        return ENQUEUED

    pool.put.side_effect = _put
    gui_ctx.worker_pool = pool
    gui_ctx.image_settings = MagicMock()
    gui_ctx.session.engineering_mode = True
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn: fn(0))
    monkeypatch.setattr(vc.gui_logger, 'button', lambda *a, **kw: None)
    asked = []
    started = []
    real_start = ProtocolRunner.start_autofocus

    def _start_autofocus(self, **kwargs):
        asked.append((kwargs, current_caller()))
        run = real_start(self, **kwargs)
        started.append(run)
        return run

    monkeypatch.setattr(ProtocolRunner, 'start_autofocus', _start_autofocus)
    return SimpleNamespace(
        session=gui_ctx.session,
        asked=asked,
        started=started,
        posts=centre_posts,
        pool=pool,
        button=_Button(),
    )


@pytest.fixture
def homed(pressed):
    """``pressed`` homed, Z mid-travel; any run left live is stopped at teardown."""
    scope = pressed.session.scope
    home_sim_scope(scope)
    scope.motion.move_z(5000.0)
    yield pressed
    scope.simulation.clear()
    scope.motion.stop_motion()
    for run in pressed.started:
        run.result(timeout=_END_WAIT_S)


def _press_held(homed, button):
    """Press ``button`` with Z's next move held: the run it starts stays live at
    its go-to-step until the test or the teardown ends it."""
    reached = homed.session.scope.simulation.hold_axis('Z')
    button.run_autofocus_from_ui()
    assert reached.wait(_HOLD_WAIT_S), "the autofocus run's first move never started"
    return button._autofocus_run


def _drawn(button):
    return (button.button.state, button.button.text)


def test_the_button_states_only_what_the_gui_knows(homed):
    _press_held(homed, homed.button)

    ((kwargs, starter),) = homed.asked
    assert kwargs['channel'] == homed.session.shown_channel.channel
    assert kwargs['save_characterization_data'] is True, (
        'in engineering mode the sweep saves its characterization data'
    )
    # Nothing else about the run is the button's to say; who started it is
    # the GUI's stamp on the thread that starts it.
    assert set(kwargs) == {
        'channel',
        'save_characterization_data',
        'events',
    }
    assert starter is Caller.GUI


def test_the_button_holds_the_run_the_member_started(homed):
    run = _press_held(homed, homed.button)

    assert homed.started == [run], 'the operation is what its Stop names'


def test_a_refused_start_is_shown_once_and_the_button_draws_idle(pressed):
    # Unhomed, the member refuses an autofocus: Z does not know where it is.
    pressed.button.run_autofocus_from_ui()

    assert [n.title for n in pressed.posts] == ['Scope Not Homed']
    assert pressed.started == []
    assert _drawn(pressed.button) == ('normal', 'Autofocus')
    assert pressed.button.autofocus_pending is False, 'the button must come back for the next press'


def test_an_unexpected_failure_is_one_fault_and_the_button_draws_idle(pressed, monkeypatch):
    def _fails(self, **kwargs):
        raise TypeError('bad call')

    # a stand-in by design: a member raising a type no session raises, a fault the simulator cannot produce
    monkeypatch.setattr(ProtocolRunner, 'start_autofocus', _fails)

    pressed.button.run_autofocus_from_ui()

    assert len(pressed.posts) == 1
    assert pressed.button.button.state == 'normal'
    assert pressed.button.autofocus_pending is False


def test_a_started_run_draws_focusing(homed):
    _press_held(homed, homed.button)

    assert _drawn(homed.button) == ('down', 'Focusing...')


def test_a_second_press_stops_its_own_run_ahead_of_queued_work(homed):
    from modules.sequential_io_executor import PRIORITY_HIGH

    run = _press_held(homed, homed.button)

    homed.button.run_autofocus_from_ui()

    task = homed.pool.put.call_args.args[0]
    assert task.priority == PRIORITY_HIGH, 'a Stop must not wait behind queued work'
    assert len(homed.asked) == 1, 'a Stop is not a start'
    outcome = run.result(timeout=_END_WAIT_S)
    assert (outcome.status, outcome.reason) == ('aborted', 'stopped')


def test_a_stop_that_finds_its_run_already_ended_shows_nothing(homed, held):
    # The press reads its run live and hands a Stop to the pool; the run
    # ends before the pool runs it. A Stop of an ended run does nothing and
    # raises nothing, so there is nothing to show.
    run = _press_held(homed, homed.button)
    held.append('hold')

    homed.button.run_autofocus_from_ui()
    homed.session.scope.simulation.clear()
    homed.session.scope.motion.stop_motion()
    ended = run.result(timeout=_END_WAIT_S)
    homed.posts.clear()
    run_task_now(held[1])

    assert homed.posts == []
    assert ended.reason == 'stage_stopped', 'the STOP ended it, not the button'
    assert _drawn(homed.button) == ('normal', 'Autofocus')


def test_the_button_is_disabled_while_its_own_request_is_in_flight(pressed, held):
    held.append('hold')

    pressed.button.run_autofocus_from_ui()

    assert pressed.button.autofocus_pending is True, (
        'a second press must not race the first to the pool'
    )
    run_task_now(held[1])
    assert pressed.button.autofocus_pending is False, (
        "the request's own redraw brings the button back"
    )
    assert len(pressed.asked) == 1


def test_a_protocols_autofocus_is_not_this_buttons_to_show(homed):
    """Another run is live -- an autofocus this button did not start -- and
    this button started nothing: it draws idle."""
    reached = homed.session.scope.simulation.hold_axis('Z')
    homed.session.create_protocol_runner().start_autofocus(channel='BF')
    assert reached.wait(_HOLD_WAIT_S)
    homed.button.button.state = 'normal'

    homed.button.draw_autofocus_button()

    assert _drawn(homed.button) == ('normal', 'Autofocus')


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
    """The real VerticalControl, constructed, with Kivy's clock held."""
    clock = _HeldClock()
    monkeypatch.setattr(vc, 'Clock', clock)
    button = vc.VerticalControl()
    button.ids = {'autofocus_id': SimpleNamespace(state='down', text='Autofocus')}
    # A Kivy property; the stubbed kivy has no descriptor to give it a default.
    button.autofocus_pending = False
    return SimpleNamespace(button=button, clock=clock)


def test_a_new_button_holds_no_run(real):
    assert real.button._autofocus_run is None


def test_a_live_autofocus_is_never_stopped_by_the_button_on_its_own(homed, real):
    """The button puts no time limit on the sweep it started: every wait
    inside a sweep is a move the motion monitor judges or a bounded capture.
    Before, a 15 s timer stopped any sweep still live, and a far-from-focus
    sweep was cut short and its Z put back (#703)."""
    run = _press_held(homed, real.button)

    for scheduled in list(real.clock.scheduled):
        scheduled(0)

    assert run.progress().is_live


def test_the_button_is_disabled_while_its_request_is_in_flight_in_the_kv():
    from tests.ast_seams import REPO_ROOT

    kv = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text()
    idx = kv.find('id: autofocus_id')
    assert idx > 0
    window = kv[idx : idx + 500].splitlines()
    disabled = [line for line in window if line.strip().startswith('disabled:')]
    assert disabled, 'the Autofocus button has no disabled binding'
    assert 'root.autofocus_held or root.autofocus_pending' in disabled[0]


def test_the_button_greys_while_anything_else_holds_the_scope(homed):
    """The Session answers for this button's own run: another holder greys it,
    its own run leaves it live as that run's Stop."""
    homed.button.draw_autofocus_button()
    assert homed.button.autofocus_held is False

    diagnostic = homed.session.activity_claim.try_claim('diagnostic')
    try:
        homed.button.draw_autofocus_button()
        assert homed.button.autofocus_held is True
    finally:
        diagnostic.release()

    _press_held(homed, homed.button)
    homed.button.draw_autofocus_button()
    assert homed.button.autofocus_held is False, 'its own run is not another holder'


def test_a_press_while_the_controls_are_locked_reaches_the_api(homed):
    """A lock on the controls is the Session's editing policy; whether the
    press may run is the API's refusal, not the button's."""
    diagnostic = homed.session.activity_claim.try_claim('diagnostic')
    try:
        assert homed.session.controls_lock.run_lockout
        homed.button.run_autofocus_from_ui()
    finally:
        diagnostic.release()

    assert len(homed.asked) == 1
    assert len(homed.posts) == 1, [n.title for n in homed.posts]
    assert homed.started == []
