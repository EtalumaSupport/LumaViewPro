# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The composite button is a run starter, and its guard always has a clearer.

The starter's whole job is to refuse what it alone can see, hand everything
else to the engine, and never leave the app in a state a second click
cannot recover from. The dangerous half is the guard Event: it is shared
with live capture, and both entry points return at their ``is_set()`` check
BEFORE enqueuing the work whose completion would clear it. So a guard left
set is not a transient -- it disables both capture buttons for the life of
the process, and only a restart brings them back.

That makes "every path that can set the guard has exactly one clearer" the
invariant worth pinning, rather than any single refusal message. The tests
below walk the paths that can set it -- a typed refusal, an unexpected
raise, and a run that genuinely starts -- and check the guard each time.
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
    return r


@pytest.fixture
def app_ctx(runner, tmp_path):
    saved = getattr(_app_ctx, 'ctx', None)
    scope = spec_scope(camera_connected=True)
    scope.imaging.active_cached = True
    session = MagicMock()
    session.create_protocol_runner.return_value = runner
    _app_ctx.ctx = SimpleNamespace(
        scope=scope,
        session=session,
        settings={'live_folder': str(tmp_path)},
        worker_pool=MagicMock(),
        ui_listener_bridge=MagicMock(),
    )
    yield _app_ctx.ctx
    _app_ctx.ctx = saved


@pytest.fixture(autouse=True)
def _clean_guard():
    cc.CompositeCapture._capturing.clear()
    yield
    cc.CompositeCapture._capturing.clear()


@pytest.fixture(autouse=True)
def _quiet_ui(monkeypatch):
    """Neutralise the cosmetics so a test reads the guard, not the theme."""
    monkeypatch.setattr(cc, 'live_histo_off', MagicMock())
    monkeypatch.setattr(cc, 'live_histo_reverse', MagicMock())
    monkeypatch.setattr(cc, 'set_title_event_text', MagicMock())
    monkeypatch.setattr(cc, 'reset_title', MagicMock())
    monkeypatch.setattr(cc, 'set_last_save_folder', MagicMock())
    # The error popup builds a real widget tree, which does not exist here.
    monkeypatch.setattr('ui.notification_popup.show_notification_popup', MagicMock())


def _click(starter):
    cc.CompositeCapture.composite_capture(starter)


def test_an_inactive_camera_refusal_leaves_both_capture_buttons_usable(app_ctx, monkeypatch):
    # The camera can be connected and still not be streaming, and the engine
    # has no refusal for that -- its connectivity check never asks. So this
    # gate stays in the GUI, and it has to refuse without leaving a mark.
    notifications = MagicMock()
    monkeypatch.setattr('modules.notification_center.notifications', notifications)
    app_ctx.scope.imaging.active_cached = False
    starter = _Starter()

    _click(starter)

    assert not cc.CompositeCapture._capturing.is_set(), (
        'a refusal must leave the capture guard clear; a set guard with '
        'nothing left to clear it wedges BOTH capture buttons for the '
        'lifetime of the process'
    )
    assert not cc.live_histo_off.called, (
        'the refusal must precede display-state changes; a refused click '
        'must not leave live-histogram equalization off'
    )
    assert not app_ctx.worker_pool.put.called, 'nothing to dispatch on a refusal'
    assert notifications.warning.called or notifications.error.called, (
        'the operator must be told why nothing happened'
    )
    assert starter.button.state == 'normal', (
        'a gate that leaves the toggle down makes the NEXT click read as '
        'the second click of a pair, and be swallowed as an abort'
    )


def test_the_engines_refusal_clears_the_guard(app_ctx, runner):
    # The engine owns rival runs, draining files and too-few-channels. The
    # starter does not pre-check any of them, so the typed refusal is the
    # only thing telling it the run never started.
    runner.start_composite.side_effect = ProtocolRunRefusedError(
        'already_running', 'Run refused', 'Another run is already in progress.'
    )
    starter = _Starter()

    _click(starter)

    assert not cc.CompositeCapture._capturing.is_set()
    assert starter.button.state == 'normal'


def test_an_unexpected_failure_still_clears_the_guard(app_ctx, runner):
    # The refusal boundary catches only the typed refusal. A programming
    # error at the call site raises straight past it, which is exactly the
    # path a per-exit clearer would miss.
    runner.start_composite.side_effect = TypeError('bad call')
    starter = _Starter()

    _click(starter)

    assert not cc.CompositeCapture._capturing.is_set(), (
        'the guard must survive an exception the refusal boundary does not catch'
    )
    assert starter.button.state == 'normal'


def test_a_started_run_holds_the_guard_until_it_completes(app_ctx, runner):
    starter = _Starter()

    _click(starter)

    runner.start_composite.assert_called_once()
    assert cc.CompositeCapture._capturing.is_set(), (
        'a live run must keep the guard set, or a second click starts a '
        'rival composite instead of stopping this one'
    )
    assert starter.button.state == 'down', 'the button stays actionable during its own run'


def test_completion_hands_the_led_buttons_back_to_the_hardware(app_ctx):
    # The run's LED restore can end without emitting the events the enable
    # toggles listen for, so completion reconciles them against the driver
    # rather than trusting the events to have arrived.
    starter = _Starter()
    cc.CompositeCapture._capturing.set()

    cc.CompositeCapture._composite_finished(starter)

    app_ctx.ui_listener_bridge.reconcile_led_buttons.assert_called_once()
    assert not cc.CompositeCapture._capturing.is_set()
    assert starter.button.state == 'normal'


def test_a_second_click_on_a_live_composite_stops_it(app_ctx, runner):
    # The stop must not queue behind ordinary pool work: the pool runs one
    # worker, so a stop that waited its turn would not arrive until the
    # thing the user is interrupting had already finished.
    from modules.sequential_io_executor import PRIORITY_HIGH

    runner.is_running.return_value = True
    runner.run_trigger_source.return_value = 'composite'
    starter = _Starter()

    _click(starter)

    assert app_ctx.worker_pool.put.called, 'the stop must be dispatched'
    task = app_ctx.worker_pool.put.call_args.args[0]
    assert task.priority == PRIORITY_HIGH
    assert task.action == runner.reset
    runner.start_composite.assert_not_called()


def test_a_click_during_someone_elses_run_is_not_an_abort(app_ctx, runner):
    # A rival run is the engine's to refuse. Treating this as a second click
    # would let the composite button stop a scan it never started.
    runner.is_running.return_value = True
    runner.run_trigger_source.return_value = 'protocol'
    starter = _Starter()

    _click(starter)

    assert not app_ctx.worker_pool.put.called, 'a rival run must not be aborted from here'
    runner.start_composite.assert_called_once()
