# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Autofocus button runs through ``ProtocolRunner.start_autofocus`` and draws its run.

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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


import modules.app_context as _app_ctx
import ui.ui_helpers as ui_helpers
import ui.vertical_control as vc
from modules.exceptions import ProtocolRunRefusedError
from modules.run_outcome import PendingRunOutcome
from modules.protocol_image_writer import RunWriteBatch
from modules.sequenced_capture_runner import EngineRun, RunProgress
from tests.pool_fakes import run_task_now
from modules.scope_session import ControlsLock, ShownChannel


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
def pressed(monkeypatch, held, centre_posts):
    """A GUI with nothing running; the pool runs each request at once."""
    from modules.sequential_io_executor import ENQUEUED

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
    engine = MagicMock()
    run = EngineRun(engine, PendingRunOutcome(), RunWriteBatch(MagicMock()))
    member.start_autofocus.return_value = run.operation
    _live(engine)
    session = MagicMock()
    session.create_protocol_runner.return_value = member
    # Nothing holds the scope; a MagicMock's own answer would be truthy.
    session.held_by_other.return_value = False
    session.engineering_mode = True
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            session=session,
            sequenced_capture_runner=engine,
            image_settings=MagicMock(),
            worker_pool=pool,
        ),
    )
    monkeypatch.setattr(vc.gui_logger, 'button', lambda *a, **kw: None)
    session.shown_channel = ShownChannel('Green', lit=False)
    return SimpleNamespace(
        member=member, engine=engine, run=run, posts=centre_posts, pool=pool, button=_Button()
    )


def _progress(*, live: bool, stopping: bool = False) -> RunProgress:
    """The engine's record of one autofocus run: live or ended, stopping or not."""
    return RunProgress(
        is_live=live,
        is_stopping=live and stopping,
        is_last_run=True,
        run_dir=None,
        step_number=1 if live else None,
        num_steps=1 if live else None,
        remaining_scans=1 if live else None,
        interval=None,
    )


def _live(engine, *runs):
    """The engine reports each of *runs* live, and every other run ended."""
    engine._progress_of.side_effect = lambda run: _progress(live=any(run is r for r in runs))


def test_the_button_states_only_what_the_gui_knows(pressed):
    pressed.button.run_autofocus_from_ui()

    pressed.member.start_autofocus.assert_called_once()
    kwargs = pressed.member.start_autofocus.call_args.kwargs
    assert kwargs['layer'] == 'Green'
    assert kwargs['run_trigger_source'] == 'autofocus', 'the attended trigger'
    assert kwargs['save_characterization_data'] is True, (
        'in engineering mode the sweep saves its characterization data'
    )
    # Nothing else about the run is the button's to say.
    assert set(kwargs) == {
        'layer',
        'save_characterization_data',
        'events',
        'run_trigger_source',
    }


def test_the_button_starts_no_run_of_its_own(pressed):
    pressed.button.run_autofocus_from_ui()

    assert not pressed.engine.prepare.called
    assert not pressed.engine.start.called
    assert pressed.button._autofocus_run is pressed.run.operation, (
        'the operation is what its Stop names'
    )


def test_a_refused_start_is_shown_once_and_the_button_draws_idle(pressed):
    pressed.member.start_autofocus.side_effect = ProtocolRunRefusedError(
        reason='autofocus_running', message='m'
    )

    pressed.button.run_autofocus_from_ui()

    assert [n.title for n in pressed.posts] == ['Autofocus Running']
    assert (pressed.button.button.state, pressed.button.button.text) == ('normal', 'Autofocus')
    assert pressed.button.autofocus_pending is False, 'the button must come back for the next press'


def test_an_unexpected_failure_is_one_fault_and_the_button_draws_idle(pressed):
    pressed.member.start_autofocus.side_effect = TypeError('bad call')

    pressed.button.run_autofocus_from_ui()

    assert len(pressed.posts) == 1
    assert pressed.button.button.state == 'normal'
    assert pressed.button.autofocus_pending is False


def test_a_started_run_draws_focusing(pressed):
    _live(pressed.engine, pressed.run)

    pressed.button.run_autofocus_from_ui()

    assert (pressed.button.button.state, pressed.button.button.text) == ('down', 'Focusing...')


def test_a_second_press_stops_its_own_run_ahead_of_queued_work(pressed):
    from modules.sequential_io_executor import PRIORITY_HIGH

    pressed.button._autofocus_run = pressed.run.operation
    pressed.engine._progress_of.side_effect = lambda run: _progress(
        live=run is pressed.run, stopping=pressed.engine._stop_run.called
    )

    pressed.button.run_autofocus_from_ui()

    task = pressed.pool.put.call_args.args[0]
    assert task.priority == PRIORITY_HIGH, 'a Stop must not wait behind queued work'
    pressed.engine._stop_run.assert_called_once_with(pressed.run)
    assert not pressed.member.start_autofocus.called, 'a Stop is not a start'
    assert pressed.button.button.text == 'Stopping...'


def test_a_stop_that_finds_its_run_already_ended_shows_nothing(pressed, held):
    # The press reads its run live and hands a Stop to the pool; the run
    # ends before the pool runs it. A Stop of an ended run does nothing and
    # raises nothing, so there is nothing to show and nothing is stopped.
    held.append('hold')
    pressed.button._autofocus_run = pressed.run.operation
    _live(pressed.engine, pressed.run)

    pressed.button.run_autofocus_from_ui()
    _live(pressed.engine)
    pressed.run.operation.set_result(None)
    run_task_now(held[1])

    assert pressed.posts == []
    assert not pressed.engine._stop_run.called, 'a Stop of an ended run reached the engine'
    assert (pressed.button.button.state, pressed.button.button.text) == ('normal', 'Autofocus')


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
    assert pressed.member.start_autofocus.call_count == 1


def test_a_protocols_autofocus_is_not_this_buttons_to_show(pressed):
    """Another run is live -- a protocol with autofocus steps -- and this
    button started nothing: it draws idle."""
    _live(
        pressed.engine, EngineRun(pressed.engine, PendingRunOutcome(), RunWriteBatch(MagicMock()))
    )
    pressed.button.button.state = 'normal'

    pressed.button.draw_autofocus_button()

    assert (pressed.button.button.state, pressed.button.button.text) == ('normal', 'Autofocus')


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


def test_a_live_autofocus_is_never_stopped_by_the_button_on_its_own(pressed, real):
    """The button puts no time limit on the sweep it started: every wait
    inside a sweep is a move the motion monitor judges or a bounded capture.
    Before, a 15 s timer stopped any sweep still live, and a far-from-focus
    sweep was cut short and its Z put back (#703)."""
    _live(pressed.engine, pressed.run)
    real.button.run_autofocus_from_ui()

    for scheduled in list(real.clock.scheduled):
        scheduled(0)

    assert not pressed.engine._stop_run.called


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

    own = EngineRun(
        app_context.ctx.sequenced_capture_runner, PendingRunOutcome(), RunWriteBatch(MagicMock())
    ).operation
    pressed.button._autofocus_run = own
    asked = []

    for held in (True, False):
        app_context.ctx.session.held_by_other = lambda run, held=held: asked.append(run) or held
        pressed.button.draw_autofocus_button()
        assert pressed.button.autofocus_held is held

    assert asked == [own, own], 'the button must ask about the run it started'


def test_a_press_during_a_finished_runs_drain_reaches_the_api(pressed):
    """A finished run's files draining is the API's to refuse, not the button's."""
    import modules.app_context as app_context

    app_context.ctx.session.controls_lock = ControlsLock(
        run_named="A protocol's files are still being written.", recording=False
    )

    pressed.button.run_autofocus_from_ui()

    pressed.member.start_autofocus.assert_called_once()
