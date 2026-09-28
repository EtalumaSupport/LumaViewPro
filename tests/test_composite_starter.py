# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The composite button is a run starter, and it decides nothing.

Its whole job is to hand the press to the engine through the boundary and
draw what the engine then says of the run it started. Every refusal -- a
rival run, files draining, too few channels, a camera that is absent -- is
the engine's, shown once by the one reporter; a still mid-capture is not a
refusal at all (the run waits for it). Whether a press is a Stop is the
engine's answer about this button's own run, never the toggle's state.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ui.composite_capture is a Kivy widget module; conftest mocks `kivy` but not
# the uix submodules, and CompositeCapture subclasses FloatLayout (a bare
# MagicMock cannot be subclassed).
class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in ('kivy.clock', 'kivy.uix'):
    sys.modules.setdefault(_name, MagicMock())

_floatlayout = types.ModuleType('kivy.uix.floatlayout')
_floatlayout.FloatLayout = _StubWidget
sys.modules.setdefault('kivy.uix.floatlayout', _floatlayout)

import modules.app_context as _app_ctx
from modules.exceptions import ProtocolRunRefusedError
from tests.scope_fakes import spec_scope
import ui.composite_capture as cc
import ui.ui_helpers as ui_helpers


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
def runner():
    return MagicMock()


@pytest.fixture
def engine():
    """The session's sequenced-capture engine: what a Stop is handed to,
    and what the button asks whether its run is live."""
    e = MagicMock()
    e.is_live_run.return_value = False
    return e


@pytest.fixture
def app_ctx(runner, engine, tmp_path, monkeypatch):
    from modules.sequential_io_executor import ENQUEUED

    saved = getattr(_app_ctx, 'ctx', None)
    session = MagicMock()
    session.create_protocol_runner.return_value = runner
    pool = MagicMock()

    def _run_now(task):
        # One worker, run as it is handed over: the order a person's
        # presses reach the engine is the order they were made.
        task.action(*task.args, **task.kwargs)
        return ENQUEUED

    pool.put.side_effect = _run_now
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn, timeout=0: fn(0))
    _app_ctx.ctx = SimpleNamespace(
        scope=spec_scope(camera_connected=True),
        session=session,
        settings={'live_folder': str(tmp_path)},
        worker_pool=pool,
        sequenced_capture_runner=engine,
        ui_listener_bridge=MagicMock(),
        # The starter hands the context's live flag to the run; production's
        # context always carries it.
        engineering_mode=False,
    )
    yield _app_ctx.ctx
    _app_ctx.ctx = saved


@pytest.fixture(autouse=True)
def _quiet_ui(monkeypatch):
    """Neutralise the cosmetics so a test reads the toggle, not the theme."""
    monkeypatch.setattr(cc, 'set_title_event_text', MagicMock())
    monkeypatch.setattr(cc, 'set_last_save_folder', MagicMock())


@pytest.fixture
def shown(monkeypatch):
    from tests.shown_outcomes import capture_shown

    return capture_shown(monkeypatch)


def _click(starter):
    cc.CompositeCapture.composite_capture(starter)


def test_the_button_hands_the_press_to_the_engine_and_decides_nothing(app_ctx, runner, shown):
    # No folder: the API owns where a composite goes, so a script's and a
    # press's land in the same place. No camera pre-check: a camera that
    # is not streaming is the engine's refusal, and the button cannot know
    # that better than prepare() does.
    app_ctx.scope.imaging.active_cached = False
    starter = _Starter()

    _click(starter)

    runner.start_composite.assert_called_once()
    kwargs = runner.start_composite.call_args.kwargs
    assert 'parent_dir' not in kwargs, 'the button composed a folder the API already owns'
    assert kwargs['run_trigger_source'] == 'composite'
    assert kwargs['engineering_mode'] is False
    assert not app_ctx.sequenced_capture_runner.reset.called, 'a start is not a stop'


def test_a_refused_start_is_shown_once_and_the_button_draws_idle(app_ctx, runner, shown):
    # The button changed nothing ahead of the engine's answer, so there is
    # nothing to undo: the redraw asks the engine, which has no live run.
    runner.start_composite.side_effect = ProtocolRunRefusedError(
        'already_running', 'Already Running', 'Another run is already in progress.'
    )
    starter = _Starter()

    _click(starter)

    assert [n.title for n in shown] == ['Already Running']
    assert starter.button.state == 'normal'
    assert starter.composite_pending is False, 'the button must come back for the next press'


def test_an_unexpected_failure_is_one_fault_and_the_button_draws_idle(app_ctx, runner, shown):
    runner.start_composite.side_effect = TypeError('bad call')
    starter = _Starter()

    _click(starter)

    assert len(shown) == 1
    assert starter.button.state == 'normal'
    assert starter.composite_pending is False


def test_a_started_run_keeps_its_handle_and_draws_running(app_ctx, runner, engine):
    engine.is_live_run.side_effect = lambda run: run is runner.start_composite.return_value
    starter = _Starter()

    _click(starter)

    assert starter._composite_run is runner.start_composite.return_value, (
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


def test_a_second_press_on_its_own_live_composite_stops_it_ahead_of_queued_work(
    app_ctx, runner, engine
):
    from modules.run_outcome import PendingRunOutcome
    from modules.sequential_io_executor import PRIORITY_HIGH

    starter = _Starter()
    starter._composite_run = PendingRunOutcome()
    engine.is_live_run.side_effect = lambda run: run is starter._composite_run

    _click(starter)

    task = app_ctx.worker_pool.put.call_args.args[0]
    assert task.priority == PRIORITY_HIGH, 'a Stop must not wait behind queued work'
    # The Stop names the run this button's own start returned, so the
    # engine can tell it from a rival's.
    engine.reset.assert_called_once_with(starter._composite_run)
    runner.start_composite.assert_not_called()


def test_a_press_during_someone_elses_run_is_not_a_stop(app_ctx, runner, engine):
    # A rival run is the engine's to refuse. Treating this as a second press
    # would let the composite button stop a scan it never started.
    starter = _Starter()
    engine.is_live_run.side_effect = lambda run: False

    _click(starter)

    engine.is_live_run.assert_any_call(None)
    assert not engine.reset.called, 'a rival run must not be stopped from here'
    runner.start_composite.assert_called_once()
