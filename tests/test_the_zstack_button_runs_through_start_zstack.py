# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Z-stack Acquire button runs through ``ProtocolRunner.start_zstack``.

There is one implementation of "assemble and start a z-stack". The button
once built the whole run itself -- objective, stack, capture config, the
protocol, ``prepare`` and ``start`` -- beside the member built for scripts
and REST, so the two could drift and only tests ever ran the member. Now the
button states only what a running GUI knows (the open drawer, the live
engineering flag, the engineering panel's saving switch), the GUI's stamp
on the worker says who started the run, and the member decides everything
else, every refusal included.

The button's "Z n/total" readout asks the run's progress for its step count:
the member built the protocol, so the widget never holds it, and a second
count computed from settings could disagree with the run.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


import modules.app_context as _app_ctx
import ui.zstack as zs
from modules.caller import Caller, current_caller
from modules.exceptions import ProtocolRunRefusedError
from modules.run_outcome import PendingRunOutcome
from modules.protocol_image_writer import RunWriteBatch
from modules.sequenced_capture_runner import EngineRun, RunProgress
from tests.pool_fakes import run_task_now
from modules.scope_session import ShownChannel


def _progress(
    run: EngineRun,
    *,
    live: bool,
    stopping: bool = False,
    step: int | None = None,
    steps: int | None = None,
) -> RunProgress:
    """The engine's record of *run*: its folder always, its step and count only while live."""
    return RunProgress(
        is_live=live,
        is_stopping=stopping,
        is_last_run=True,
        run_dir=run._run_dir,
        step_number=step if live else None,
        num_steps=steps if live else None,
        remaining_scans=1 if live else None,
        interval=None,
    )


class _Starter(zs.ZStack):
    """The real class with the widget tree stubbed, the button as a first
    click leaves it."""

    def __init__(self):
        self.button = SimpleNamespace(state='down', text='Acquire')
        self.ids = {'zstack_aqr_btn': self.button}
        self.zstack_pending = False


@pytest.fixture
def clicked(monkeypatch, centre_posts):
    """Click Acquire with nothing running; return what the member was asked."""
    from modules.sequential_io_executor import ENQUEUED
    import ui.ui_helpers as ui_helpers

    pool = MagicMock()

    def _run_now(task):
        run_task_now(task)
        return ENQUEUED

    pool.put.side_effect = _run_now
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn, timeout=0: fn(0))
    member_runner = MagicMock()
    engine = MagicMock()
    run = EngineRun(engine, PendingRunOutcome(), RunWriteBatch(MagicMock()))
    run._run_dir = '/runs/zstack_1'
    starters = []

    def _start_zstack(**_kwargs):
        starters.append(current_caller())
        return run.operation

    member_runner.start_zstack.side_effect = _start_zstack
    # The run has ended by the time anything reads it, unless a test says otherwise.
    engine._progress_of.side_effect = lambda asked: _progress(asked, live=False)
    session = MagicMock()
    session.create_protocol_runner.return_value = member_runner
    # Nothing holds the scope; a MagicMock's own answer would be truthy.
    session.held_by_other.return_value = False
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            session=session,
            sequenced_capture_runner=engine,
            image_settings=MagicMock(),
            scope_display=SimpleNamespace(start=lambda: None, stop=lambda: None),
            motion_settings=MagicMock(),
            worker_pool=pool,
        ),
    )
    monkeypatch.setattr(zs.gui_logger, 'button', lambda *a, **kw: None)
    session.shown_channel = ShownChannel('Green', lit=False)
    monkeypatch.setattr(zs, 'is_image_saving_enabled', lambda: False)
    linked = []
    monkeypatch.setattr(zs, 'set_last_save_folder', lambda **kw: linked.append(kw['dir']))
    return SimpleNamespace(
        runner=member_runner,
        engine=engine,
        run=run,
        linked=linked,
        starters=starters,
        posts=centre_posts,
        starter=_Starter(),
    )


def test_the_button_states_only_what_the_gui_knows(clicked):
    clicked.starter.run_zstack_acquire_from_ui()

    clicked.runner.start_zstack.assert_called_once()
    kwargs = clicked.runner.start_zstack.call_args.kwargs
    assert kwargs['layer'] == 'Green'
    assert kwargs['enable_image_saving'] is False
    # Nothing else about the run is the button's to say; who started it is
    # the GUI's stamp on the thread that starts it.
    assert set(kwargs) == {
        'layer',
        'events',
        'enable_image_saving',
    }
    assert clicked.starters == [Caller.GUI]


def test_the_button_starts_no_run_of_its_own(clicked):
    clicked.starter.run_zstack_acquire_from_ui()

    assert not clicked.engine.prepare.called
    assert not clicked.engine.start.called


def test_the_stop_names_the_handle_the_member_returned(clicked):
    clicked.starter.run_zstack_acquire_from_ui()

    assert clicked.starter._zstack_run is clicked.run.operation
    assert clicked.linked == ['/runs/zstack_1']


def test_a_refusal_is_shown_once_draws_idle_and_links_no_folder(clicked):
    clicked.runner.start_zstack.side_effect = ProtocolRunRefusedError(
        reason='zstack_not_configured', message='m'
    )

    clicked.starter.run_zstack_acquire_from_ui()

    assert [n.title for n in clicked.posts] == ['Z-Stack Not Configured']
    assert clicked.starter.button.state == 'normal'
    assert clicked.starter.button.text == 'Acquire'
    assert clicked.starter.zstack_pending is False, 'the button must come back for the next press'
    assert clicked.linked == []


def test_progress_reads_the_runs_own_step_and_count(clicked):
    clicked.engine._progress_of.side_effect = lambda run: _progress(
        run, live=run is clicked.run, step=3, steps=7
    )

    clicked.starter.run_zstack_acquire_from_ui()
    step_started = clicked.runner.start_zstack.call_args.kwargs['events'].step_started
    step_started(2)

    assert clicked.starter.button.text == 'Z 3/7'
    # Any other edge mid-run draws the same label: there is one writer.
    clicked.starter.draw_zstack_button()
    assert clicked.starter.button.text == 'Z 3/7'


def test_progress_after_the_run_ended_draws_idle(clicked):
    clicked.starter.run_zstack_acquire_from_ui()
    step_started = clicked.runner.start_zstack.call_args.kwargs['events'].step_started
    step_started(2)

    assert (clicked.starter.button.state, clicked.starter.button.text) == ('normal', 'Acquire')


def test_a_second_press_stops_its_own_stack_ahead_of_queued_work(clicked):
    from modules.sequential_io_executor import PRIORITY_HIGH

    clicked.starter._zstack_run = clicked.run.operation
    clicked.engine._progress_of.side_effect = lambda run: _progress(
        run, live=run is clicked.run, stopping=clicked.engine._stop_run.called
    )

    clicked.starter.run_zstack_acquire_from_ui()

    task = _app_ctx.ctx.worker_pool.put.call_args.args[0]
    assert task.priority == PRIORITY_HIGH, 'a Stop must not wait behind queued work'
    clicked.engine._stop_run.assert_called_once_with(clicked.run)
    assert not clicked.runner.start_zstack.called, 'a Stop is not a start'
    assert clicked.starter.button.text == 'Stopping...'


def test_a_press_is_a_start_whatever_the_toggle_reads(clicked):
    # Kivy flips the toggle at touch-down, and a redraw can have left it
    # either way; whether a press means Stop is only the engine's answer.
    clicked.starter.button.state = 'normal'

    clicked.starter.run_zstack_acquire_from_ui()

    assert not clicked.engine._stop_run.called, "a toggle reading 'normal' is not a Stop"
    clicked.runner.start_zstack.assert_called_once()


def test_the_button_is_disabled_while_its_own_request_is_in_flight(clicked):
    from modules.sequential_io_executor import ENQUEUED

    held = []
    _app_ctx.ctx.worker_pool.put.side_effect = lambda task: held.append(task) or ENQUEUED

    clicked.starter.run_zstack_acquire_from_ui()

    assert clicked.starter.zstack_pending is True, (
        'a second press must not race the first to the pool'
    )
    run_task_now(held[0])
    assert clicked.starter.zstack_pending is False, "the request's own redraw brings it back"


class TestTheMemberCarriesWhatTheGuiStates:
    """What only a running GUI knows reaches the engine as stated; a caller
    that states none of it gets today's headless run. Engineering mode is
    not among them: it is the session's."""

    def test_they_reach_prepare(self):
        from tests.test_start_zstack_entry_point import _prepared, _runner

        runner = _runner()
        runner.session.engineering_mode = True
        runner.start_zstack(
            layer='BF',
            enable_image_saving=False,
        )

        prepared = _prepared(runner)
        assert prepared['engineering_mode'] is True
        assert prepared['enable_image_saving'] is False

    def test_a_headless_caller_keeps_its_defaults(self):
        from tests.test_start_zstack_entry_point import _prepared, _runner

        runner = _runner()
        runner.session.engineering_mode = False
        runner.start_zstack(layer='BF')

        prepared = _prepared(runner)
        assert prepared['engineering_mode'] is False
        assert prepared['enable_image_saving'] is True


def test_the_button_greys_while_anything_else_holds_the_scope(clicked):
    """The Session answers for this button's own run: another holder greys it,
    its own run leaves it live as that run's Stop."""
    import modules.app_context as app_context

    own = EngineRun(
        app_context.ctx.sequenced_capture_runner, PendingRunOutcome(), RunWriteBatch(MagicMock())
    ).operation
    clicked.starter._zstack_run = own
    asked = []

    for held in (True, False):
        app_context.ctx.session.held_by_other = lambda run, held=held: asked.append(run) or held
        clicked.starter.draw_zstack_button()
        assert clicked.starter.zstack_held is held

    assert asked == [own, own], 'the button must ask about the run it started'
