# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The composite button is a run starter, and it decides nothing.

Its whole job is to hand the click to the engine, display what the engine
answers, and never leave the app in a state a second click cannot recover
from. Every refusal -- a rival run, files draining, too few channels, a
camera that is absent -- is the engine's; a still mid-capture is not a
refusal at all (the run waits for it). What the button owns is the toggle:
a refused or failed start puts it back, a started run keeps it down until
the run ends, and a second click on its own live run is a Stop.
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
    r = MagicMock()
    r.is_running.return_value = False
    r.run_trigger_source.return_value = None
    # Nothing is live, so no handle is the live run -- what the real
    # runner answers before any run starts.
    r.is_live_run.return_value = False
    return r


@pytest.fixture
def engine():
    """The session's sequenced-capture engine, which a Stop is handed to."""
    return MagicMock()


@pytest.fixture
def app_ctx(runner, engine, tmp_path):
    saved = getattr(_app_ctx, 'ctx', None)
    session = MagicMock()
    session.create_protocol_runner.return_value = runner
    _app_ctx.ctx = SimpleNamespace(
        scope=spec_scope(camera_connected=True),
        session=session,
        settings={'live_folder': str(tmp_path)},
        worker_pool=MagicMock(),
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
    monkeypatch.setattr(cc, 'live_histo_off', MagicMock())
    monkeypatch.setattr(cc, 'live_histo_reverse', MagicMock())
    monkeypatch.setattr(cc, 'set_title_event_text', MagicMock())
    monkeypatch.setattr(cc, 'reset_title', MagicMock())
    monkeypatch.setattr(cc, 'set_last_save_folder', MagicMock())
    # The error popup builds a real widget tree, which does not exist here.
    monkeypatch.setattr('ui.notification_popup.show_notification_popup', MagicMock())


def _click(starter):
    cc.CompositeCapture.composite_capture(starter)


def test_the_button_hands_the_click_to_the_engine_and_decides_nothing(app_ctx, runner):
    # No folder: the API owns where a composite goes, so a script's and a
    # click's land in the same place. No camera pre-check: a camera that
    # is not streaming is the engine's hardware_disconnected refusal, and
    # the button cannot know that better than prepare() does.
    app_ctx.scope.imaging.active_cached = False
    starter = _Starter()

    _click(starter)

    runner.start_composite.assert_called_once()
    kwargs = runner.start_composite.call_args.kwargs
    assert 'parent_dir' not in kwargs, 'the button composed a folder the API already owns'
    assert kwargs['run_trigger_source'] == 'composite'
    assert kwargs['engineering_mode'] is False
    assert not app_ctx.worker_pool.put.called, 'a start is not a stop'


def test_the_engines_refusal_resets_the_toggle(app_ctx, runner):
    # The engine owns rival runs, draining files, too few channels and a
    # missing camera; its typed refusal is the only thing telling the
    # button the run never started. Left 'down', the next click would be
    # swallowed as an abort of a run that was never started.
    runner.start_composite.side_effect = ProtocolRunRefusedError(
        'already_running', 'Run refused', 'Another run is already in progress.'
    )
    starter = _Starter()

    _click(starter)

    assert starter.button.state == 'normal'
    assert not app_ctx.worker_pool.put.called, 'nothing to dispatch on a refusal'


def test_an_unexpected_failure_resets_the_toggle(app_ctx, runner):
    # The refusal boundary catches only the typed refusal. A programming
    # error at the call site raises straight past it, which is exactly the
    # path a per-exit reset would miss.
    runner.start_composite.side_effect = TypeError('bad call')
    starter = _Starter()

    _click(starter)

    assert starter.button.state == 'normal'


def test_a_started_run_keeps_its_handle_and_the_toggle_down(app_ctx, runner):
    starter = _Starter()

    _click(starter)

    runner.start_composite.assert_called_once()
    assert starter.button.state == 'down', 'the button stays actionable during its own run'
    assert starter._composite_run is runner.start_composite.return_value, (
        'the button must keep the handle its start returned: it is what its Stop names'
    )


def test_completion_hands_the_led_buttons_back_to_the_hardware(app_ctx):
    # The run's LED restore can end without emitting the events the enable
    # toggles listen for, so completion reconciles them against the driver
    # rather than trusting the events to have arrived.
    starter = _Starter()

    cc.CompositeCapture._composite_finished(starter)

    app_ctx.ui_listener_bridge.reconcile_led_buttons.assert_called_once()
    assert starter.button.state == 'normal'


def test_a_second_click_on_a_live_composite_stops_it(app_ctx, runner, engine):
    # The stop must not queue behind ordinary pool work: the pool runs one
    # worker, so a stop that waited its turn would not arrive until the
    # thing the user is interrupting had already finished.
    from modules.run_outcome import PendingRunOutcome
    from modules.sequential_io_executor import PRIORITY_HIGH

    starter = _Starter()
    starter._composite_run = PendingRunOutcome()
    runner.is_running.return_value = True
    runner.run_trigger_source.return_value = 'composite'
    runner.is_live_run.side_effect = lambda run: run is starter._composite_run

    _click(starter)

    assert app_ctx.worker_pool.put.called, 'the stop must be dispatched'
    task = app_ctx.worker_pool.put.call_args.args[0]
    assert task.priority == PRIORITY_HIGH
    # The stop names the run this starter's own start returned, so the
    # engine can tell it from a rival's; and it goes through the refusal
    # boundary like every run control's Stop, not the bare method.
    assert task.action.func is ui_helpers.reset_with_refusal_boundary
    assert task.action.args == (engine, starter._composite_run)
    assert task.action.keywords == {}
    runner.start_composite.assert_not_called()


def test_a_click_during_someone_elses_run_is_not_an_abort(app_ctx, runner):
    # A rival run is the engine's to refuse. Treating this as a second click
    # would let the composite button stop a scan it never started.
    starter = _Starter()
    runner.is_running.return_value = True
    runner.run_trigger_source.return_value = 'protocol'
    # The live run is the rival's, so the handle this button holds (none:
    # it started nothing) is not the live run.
    runner.is_live_run.side_effect = lambda run: False

    _click(starter)

    runner.is_live_run.assert_called_with(None)
    assert not app_ctx.worker_pool.put.called, 'a rival run must not be aborted from here'
    runner.start_composite.assert_called_once()
