# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Z-stack Acquire button runs through ``ProtocolRunner.start_zstack``.

There is one implementation of "assemble and start a z-stack". The button
once built the whole run itself -- objective, stack, capture config, the
protocol, ``prepare`` and ``start`` -- beside the member built for scripts
and REST, so the two could drift and only tests ever ran the member. Now the
button states only what a running GUI knows (the open drawer and the
engineering panel's saving switch), the GUI's stamp
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


import ui.zstack as zs
from ui.protocol_settings import ProtocolSettings
from modules.caller import Caller, current_caller
from modules.protocol_runner import ProtocolRunner
from tests.pool_fakes import run_task_now
from tests.scope_fakes import home_sim_scope

_HOLD_WAIT_S = 10.0
_END_WAIT_S = 60.0


class _SavingSwitch(ProtocolSettings):
    """The real protocol panel, with only its saving checkbox stubbed."""

    def __init__(self):
        self.box = SimpleNamespace(active=False)
        self.ids = {'protocol_disable_image_saving_id': self.box}


class _Starter(zs.ZStack):
    """The real class with the widget tree stubbed, the button as a first
    click leaves it."""

    def __init__(self):
        self.button = SimpleNamespace(state='down', text='Acquire')
        self.ids = {'zstack_aqr_btn': self.button}
        self.zstack_pending = False
        self._zstack_run = None


@pytest.fixture
def held():
    """Pool tasks, when a test holds them instead of running them at once."""
    return []


@pytest.fixture
def clicked(gui_ctx, monkeypatch, held, centre_posts):
    """The GUI over an unhomed simulated session on the shipped template, whose
    stack has no range; the pool runs each request at once, and each start the
    member is asked for is recorded with who asked, then made."""
    from modules.sequential_io_executor import ENQUEUED

    pool = MagicMock()

    def _put(task):
        if held == ['hold']:
            held.append(task)
        else:
            run_task_now(task)
        return ENQUEUED

    pool.put.side_effect = _put
    switch = _SavingSwitch()
    gui_ctx.worker_pool = pool
    gui_ctx.image_settings = MagicMock()
    gui_ctx.scope_display = SimpleNamespace(start=lambda: None, stop=lambda: None)
    gui_ctx.motion_settings = SimpleNamespace(ids={'protocol_settings_id': switch})
    monkeypatch.setattr('ui.ui_helpers._schedule_ui', lambda fn: fn(0))
    monkeypatch.setattr(zs.gui_logger, 'button', lambda *a, **kw: None)
    linked = []
    monkeypatch.setattr(zs, 'set_last_save_folder', lambda **kw: linked.append(kw['dir']))
    asked = []
    started = []
    real_start = ProtocolRunner.start_zstack

    def _start_zstack(self, **kwargs):
        asked.append((kwargs, current_caller()))
        run = real_start(self, **kwargs)
        started.append(run)
        return run

    monkeypatch.setattr(ProtocolRunner, 'start_zstack', _start_zstack)
    return SimpleNamespace(
        session=gui_ctx.session,
        pool=pool,
        asked=asked,
        started=started,
        linked=linked,
        posts=centre_posts,
        starter=_Starter(),
        saving_box=switch.box,
    )


@pytest.fixture
def stacked(clicked):
    """``clicked`` homed, Z mid-travel, a seven-slice stack stored (60 um in
    10 um steps); any run left live is stopped at teardown."""
    scope = clicked.session.scope
    home_sim_scope(scope)
    scope.motion.move_z(5000.0)
    clicked.session.update_settings('zstack.step_size', 10.0)
    clicked.session.update_settings('zstack.range', 60.0)
    yield clicked
    scope.simulation.clear()
    scope.motion.stop_motion()
    for run in clicked.started:
        run.result(timeout=_END_WAIT_S)


def _click_held(stacked):
    """Click Acquire with Z's next move held: the stack it starts stays live at
    its first slice until the test or the teardown ends it."""
    reached = stacked.session.scope.simulation.hold_axis('Z')
    stacked.starter.run_zstack_acquire_from_ui()
    assert reached.wait(_HOLD_WAIT_S), "the stack's first move never started"
    return stacked.starter._zstack_run


def test_the_button_states_only_what_the_gui_knows(stacked):
    _click_held(stacked)

    ((kwargs, starter),) = stacked.asked
    assert kwargs['channel'] == stacked.session.shown_channel.channel
    assert kwargs['enable_image_saving'] is True, 'an unticked box saves images'
    # Nothing else about the run is the button's to say; who started it is
    # the GUI's stamp on the thread that starts it.
    assert set(kwargs) == {
        'channel',
        'events',
        'enable_image_saving',
    }
    assert starter is Caller.GUI


def test_a_ticked_disable_image_saving_box_starts_a_stack_that_saves_none(stacked):
    stacked.saving_box.active = True

    _click_held(stacked)

    ((kwargs, _starter),) = stacked.asked
    assert kwargs['enable_image_saving'] is False


def test_the_stop_names_the_handle_the_member_returned(stacked):
    run = _click_held(stacked)

    assert stacked.started == [run]
    assert stacked.linked == [run.progress().run_dir]
    assert stacked.linked[0] is not None


def test_a_refusal_is_shown_once_draws_idle_and_links_no_folder(clicked):
    """Homed, on the template, whose stored stack has no range: the member refuses."""
    home_sim_scope(clicked.session.scope)

    clicked.starter.run_zstack_acquire_from_ui()

    assert [n.title for n in clicked.posts] == ['Z-Stack Not Configured']
    assert clicked.started == []
    assert clicked.starter.button.state == 'normal'
    assert clicked.starter.button.text == 'Acquire'
    assert clicked.starter.zstack_pending is False, 'the button must come back for the next press'
    assert clicked.linked == []


def test_progress_reads_the_runs_own_step_and_count(stacked):
    _click_held(stacked)

    # The run counts its steps from 0; a person reads the first as 1.
    assert stacked.starter.button.text == 'Z 1/7'
    # Any other edge mid-run draws the same label: there is one writer.
    stacked.starter.draw_zstack_button()
    assert stacked.starter.button.text == 'Z 1/7'


def test_progress_after_the_run_ended_draws_idle(stacked):
    run = _click_held(stacked)
    ((kwargs, _starter),) = stacked.asked
    stacked.session.scope.simulation.clear()
    stacked.session.scope.motion.stop_motion()
    run.result(timeout=_END_WAIT_S)

    kwargs['events'].step_started(2)

    assert (stacked.starter.button.state, stacked.starter.button.text) == ('normal', 'Acquire')


def test_a_second_press_stops_its_own_stack_ahead_of_queued_work(stacked):
    from modules.sequential_io_executor import PRIORITY_HIGH

    run = _click_held(stacked)

    stacked.starter.run_zstack_acquire_from_ui()

    task = stacked.pool.put.call_args.args[0]
    assert task.priority == PRIORITY_HIGH, 'a Stop must not wait behind queued work'
    assert len(stacked.asked) == 1, 'a Stop is not a start'
    # A Stop only asks: the run ends once the move it is in arrives, so the
    # held move keeps it stopping until it is let go.
    assert stacked.starter.button.text == 'Stopping...'
    stacked.session.scope.simulation.clear()
    outcome = run.result(timeout=_END_WAIT_S)
    assert (outcome.status, outcome.reason) == ('aborted', 'stopped')


def test_a_press_is_a_start_whatever_the_toggle_reads(clicked):
    # Kivy flips the toggle at touch-down, and a redraw can have left it
    # either way; whether a press means Stop is only the engine's answer.
    clicked.starter.button.state = 'normal'

    clicked.starter.run_zstack_acquire_from_ui()

    assert len(clicked.asked) == 1, "a toggle reading 'normal' is not a Stop"


def test_the_button_is_disabled_while_its_own_request_is_in_flight(clicked, held):
    held.append('hold')

    clicked.starter.run_zstack_acquire_from_ui()

    assert clicked.starter.zstack_pending is True, (
        'a second press must not race the first to the pool'
    )
    run_task_now(held[1])
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
            channel='BF',
            enable_image_saving=False,
        )

        prepared = _prepared(runner)
        assert prepared['engineering_mode'] is True
        assert prepared['enable_image_saving'] is False

    def test_a_headless_caller_keeps_its_defaults(self):
        from tests.test_start_zstack_entry_point import _prepared, _runner

        runner = _runner()
        runner.session.engineering_mode = False
        runner.start_zstack(channel='BF')

        prepared = _prepared(runner)
        assert prepared['engineering_mode'] is False
        assert prepared['enable_image_saving'] is True


def test_the_button_greys_while_anything_else_holds_the_scope(stacked):
    """The Session answers for this button's own run: another holder greys it,
    its own run leaves it live as that run's Stop."""
    stacked.starter.draw_zstack_button()
    assert stacked.starter.zstack_held is False

    diagnostic = stacked.session.activity_claim.try_claim('diagnostic')
    try:
        stacked.starter.draw_zstack_button()
        assert stacked.starter.zstack_held is True
    finally:
        diagnostic.release()

    _click_held(stacked)
    stacked.starter.draw_zstack_button()
    assert stacked.starter.zstack_held is False, 'its own run is not another holder'
