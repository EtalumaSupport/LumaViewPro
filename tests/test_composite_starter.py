# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The composite button is a run starter, and it decides nothing.

Its whole job is to hand the press to the engine through the boundary and
draw what the engine then says of the run it started. Every refusal -- a
rival run, files draining, too few channels, a camera that is absent -- is
the engine's, shown once by the one reporter; a still mid-capture is not a
refusal at all (the run waits for it). Whether a press is a Stop is the
engine's answer about this button's own run, never the toggle's state.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


from modules.acquire_mode import AcquireMode
from modules.caller import Caller, current_caller
from modules.protocol_runner import ProtocolRunner
from tests.pool_fakes import run_task_now
from tests.protocol_drives import run_identity
from tests.scope_fakes import home_sim_scope
from tests.stock_channels import LED1
import ui.composite_capture as cc
import ui.ui_helpers as ui_helpers

_HOLD_WAIT_S = 10.0
_END_WAIT_S = 60.0


class _Starter(cc.CompositeCapture):
    """The real class, with only the widget tree stubbed.

    It subclasses rather than mimics: the starter calls its own completion
    handler, so a stand-in that merely looked like the widget would have to
    reimplement the very method under test.
    """

    def __init__(self):
        # 'down' is what a FIRST click leaves behind: the abort branch keys
        # off 'normal', so a mock defaulting the other way would route every
        # test through the stop path instead of the start path.
        self.button = SimpleNamespace(state='down')
        self.ids = {'composite_btn': self.button}


@pytest.fixture
def held():
    """Pool tasks, when a test holds them instead of running them at once."""
    return []


@pytest.fixture
def app_ctx(gui_ctx, held, monkeypatch):
    """The GUI over an unhomed simulated session with Brightfield and one
    fluorescence channel acquiring; the pool runs each request at once, and
    each start the member is asked for is recorded with who asked, then made."""
    from modules.sequential_io_executor import ENQUEUED

    pool = MagicMock()

    def _put(task):
        # One worker, run as it is handed over: the order a person's
        # presses reach the engine is the order they were made.
        if held == ['hold']:
            held.append(task)
        else:
            run_task_now(task)
        return ENQUEUED

    pool.put.side_effect = _put
    gui_ctx.worker_pool = pool
    gui_ctx.ui_listener_bridge = MagicMock()
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn: fn(0))
    for channel in ('BF', LED1):
        gui_ctx.session.set_channel_acquire(channel, AcquireMode.IMAGE)
    asked = []
    started = []
    real_start = ProtocolRunner.start_composite

    def _start_composite(self, **kwargs):
        asked.append((kwargs, current_caller()))
        run = real_start(self, **kwargs)
        started.append(run)
        return run

    monkeypatch.setattr(ProtocolRunner, 'start_composite', _start_composite)
    return SimpleNamespace(
        session=gui_ctx.session,
        scope=gui_ctx.scope,
        worker_pool=pool,
        ui_listener_bridge=gui_ctx.ui_listener_bridge,
        asked=asked,
        started=started,
    )


@pytest.fixture
def homed(app_ctx):
    """``app_ctx`` homed, Z mid-travel; any run left live is stopped at teardown."""
    scope = app_ctx.scope
    home_sim_scope(scope)
    scope.motion.move_z(5000.0)
    yield app_ctx
    scope.simulation.clear()
    scope.motion.stop_motion()
    for run in app_ctx.started:
        run.result(timeout=_END_WAIT_S)


@pytest.fixture
def rival(app_ctx):
    """Another run holds the scope: a REST scan's claim."""
    claim = app_ctx.session.activity_claim.try_claim('protocol', run=run_identity(Caller.REST))
    yield claim
    claim.release()


@pytest.fixture(autouse=True)
def _quiet_ui(monkeypatch):
    """Neutralise the cosmetics so a test reads the toggle, not the theme."""
    monkeypatch.setattr(cc, 'set_title_event_text', MagicMock())
    monkeypatch.setattr(cc, 'set_last_save_folder', MagicMock())


def _click(starter):
    cc.CompositeCapture.composite_capture(starter)


def _click_held(homed, starter):
    """Click with Z's next move held: the composite it starts stays live until
    the test or the teardown ends it."""
    reached = homed.scope.simulation.hold_axis('Z')
    _click(starter)
    assert reached.wait(_HOLD_WAIT_S), "the composite's first move never started"
    return starter._composite_run


def test_the_button_hands_the_press_to_the_engine_and_decides_nothing(homed):
    # No folder: the API owns where a composite goes, so a script's and a
    # press's land in the same place. No camera pre-check: a camera that
    # is not streaming is the engine's refusal, and the button cannot know
    # that better than prepare() does.
    _click_held(homed, _Starter())

    ((kwargs, starter),) = homed.asked
    assert 'parent_dir' not in kwargs, 'the button composed a folder the API already owns'
    assert starter is Caller.GUI, 'the press reaches the API stamped as the GUI'
    assert 'engineering_mode' not in kwargs, "the mode is the session's; the button states none"


def test_a_refused_start_is_shown_once_and_the_button_draws_idle(app_ctx, rival, centre_posts):
    # The button changed nothing ahead of the engine's answer, so there is
    # nothing to undo: the redraw asks the engine, which has no live run.
    starter = _Starter()

    _click(starter)

    assert [n.title for n in centre_posts] == ['Already Running']
    assert starter.button.state == 'normal'
    assert starter.composite_pending is False, 'the button must come back for the next press'


def test_an_unexpected_failure_is_one_fault_and_the_button_draws_idle(
    app_ctx, centre_posts, monkeypatch
):
    def _fails(self, **kwargs):
        raise TypeError('bad call')

    # a stand-in by design: a member raising a type no session raises, a fault the simulator cannot produce
    monkeypatch.setattr(ProtocolRunner, 'start_composite', _fails)
    starter = _Starter()

    _click(starter)

    assert len(centre_posts) == 1
    assert starter.button.state == 'normal'
    assert starter.composite_pending is False


def test_a_started_run_keeps_its_handle_and_draws_running(homed):
    starter = _Starter()

    run = _click_held(homed, starter)

    assert homed.started == [run], (
        'the button must keep the handle its start returned: it is what its Stop names'
    )
    assert starter.button.state == 'down', 'the button draws its own live run'


def test_a_redraw_draws_its_button_and_nothing_every_run_shares(app_ctx, monkeypatch):
    # The redraw fires on every run-state edge, including the one where
    # another run takes the scope: were it to hand equalization, the title
    # or the LED toggles back as it drew itself idle, it would undo that
    # run's display. Those are draw_shared_run_displays', drawn once.
    shared = MagicMock()
    monkeypatch.setattr(ui_helpers, 'live_histo_reverse', shared.live_histo_reverse)
    monkeypatch.setattr(ui_helpers, 'reset_title', shared.reset_title)
    starter = _Starter()

    cc.CompositeCapture.draw_composite_button(starter)

    assert starter.button.state == 'normal'
    assert shared.mock_calls == []
    app_ctx.ui_listener_bridge.reconcile_led_buttons.assert_not_called()


def test_a_second_press_on_its_own_live_composite_stops_it_ahead_of_queued_work(homed):
    from modules.sequential_io_executor import PRIORITY_HIGH

    starter = _Starter()
    run = _click_held(homed, starter)

    _click(starter)

    task = homed.worker_pool.put.call_args.args[0]
    assert task.priority == PRIORITY_HIGH, 'a Stop must not wait behind queued work'
    assert len(homed.asked) == 1, 'a Stop is not a start'
    # A Stop only asks: the run ends once the move it is in arrives.
    homed.scope.simulation.clear()
    outcome = run.result(timeout=_END_WAIT_S)
    assert (outcome.status, outcome.reason) == ('aborted', 'stopped')


def test_a_press_during_someone_elses_run_is_not_a_stop(app_ctx, rival):
    # A rival run is the engine's to refuse. Treating this as a second press
    # would let the composite button stop a scan it never started.
    _click(_Starter())

    assert len(app_ctx.asked) == 1, 'the press is a start, refused by the engine'
    assert app_ctx.session.held_by_other(None), 'the rival still holds the scope'


def test_a_press_is_a_start_whatever_the_toggle_reads(app_ctx):
    # Kivy flips the toggle at touch-down, and a redraw can have left it
    # either way; whether a press means Stop is only the engine's answer.
    starter = _Starter()
    starter.button.state = 'normal'

    _click(starter)

    assert len(app_ctx.asked) == 1, "a toggle reading 'normal' is not a Stop"


def test_the_button_is_disabled_while_its_own_request_is_in_flight(app_ctx, held):
    held.append('hold')
    starter = _Starter()

    _click(starter)

    assert starter.composite_pending is True, 'a second press must not race the first to the pool'
    run_task_now(held[1])
    assert starter.composite_pending is False, "the request's own redraw brings the button back"


def test_the_button_greys_while_anything_else_holds_the_scope(homed):
    """The Session answers for this button's own run: another holder greys it,
    its own run leaves it live as that run's Stop."""
    starter = _Starter()
    cc.CompositeCapture.draw_composite_button(starter)
    assert starter.composite_held is False

    diagnostic = homed.session.activity_claim.try_claim('diagnostic')
    try:
        cc.CompositeCapture.draw_composite_button(starter)
        assert starter.composite_held is True
    finally:
        diagnostic.release()

    _click_held(homed, starter)
    cc.CompositeCapture.draw_composite_button(starter)
    assert starter.composite_held is False, 'its own run is not another holder'
