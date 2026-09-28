# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""After a failed start, the next Record press must record on the FIRST press.

Record and Stop are one ToggleButton, so the button's state decides which
branch the next press takes. A start that failed after the refusal check
used to leave the toggle 'down' with nothing recording: the next press
flipped it to 'normal', which reads as "the user is stopping a live
recording", and the stop branch returns silently when there is nothing to
stop. The user pressed Record and nothing happened -- twice.

The toggle no longer decides: whether a press means Stop is the API's
answer (is a recording open), a start and its refusal go through the
boundary, and one redraw draws the toggle from the controller. A refusal
is a warning in the API's words, shown once.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock


class _StubWidget:
    def __init__(self, **kwargs):
        pass


def _real_base_module(name, **attrs):
    if name in sys.modules and not isinstance(sys.modules[name], MagicMock):
        return
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


# MainDisplay descends from a real Kivy layout; conftest mocks `kivy` but
# not the uix submodules, and a bare MagicMock cannot be subclassed.
_real_base_module('kivy.uix.floatlayout', FloatLayout=_StubWidget)

import modules.app_context as _app_ctx
import ui.main_display as main_display
import ui.ui_helpers as ui_helpers
from modules.exceptions import RecordingRefusedError


class _FakeToggle:
    """The Record/Stop ToggleButton.

    Kivy flips a toggle's state on press and THEN calls the handler, so a
    test that never flips is not testing the branch the user reaches.
    """

    def __init__(self):
        self.state = 'normal'

    def press(self):
        self.state = 'down' if self.state == 'normal' else 'normal'


class _Controller:
    def __init__(self, error=None):
        self.error = error
        self.is_recording = False
        self.save_folder = None
        self.stop_calls = 0
        self.start_calls = 0

    def start(self, layer=None, false_color_on=False, on_complete=None):
        self.start_calls += 1
        if self.error is not None:
            raise self.error
        self.is_recording = True

    def stop(self):
        self.stop_calls += 1
        self.is_recording = False


class _ImmediateClock:
    """Run scheduled callbacks inline; the UI cleanup IS the behavior here."""

    @staticmethod
    def schedule_once(callback, timeout=0):
        callback(0)

    @staticmethod
    def schedule_interval(callback, timeout):
        return object()

    @staticmethod
    def unschedule(event):
        pass


def _make_display(monkeypatch, controller):
    from modules.sequential_io_executor import ENQUEUED
    from tests.shown_outcomes import capture_shown

    display = main_display.MainDisplay.__new__(main_display.MainDisplay)
    toggle = _FakeToggle()
    display.ids = {'record_btn': toggle}
    display._recording_poll = None

    monkeypatch.setattr(main_display, 'Clock', _ImmediateClock)
    monkeypatch.setattr(main_display.gui_logger, 'button', lambda *a, **kw: None)
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn, timeout=0: fn(0))
    ctx = MagicMock()
    ctx.session.manual_recording = controller

    def _run_now(task):
        task.action(*task.args, **task.kwargs)
        return ENQUEUED

    ctx.worker_pool.put.side_effect = _run_now
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    return display, toggle, ctx, capture_shown(monkeypatch)


class TestRecordButtonRecovery:
    def test_first_press_records_after_failed_start(self, monkeypatch):
        controller = _Controller(error=RuntimeError('scripted post-commit failure'))
        display, toggle, _ctx, shown = _make_display(monkeypatch, controller)

        toggle.press()
        display.record_button()

        assert len(shown) == 1, 'the failure is reported once, by the boundary'
        assert toggle.state == 'normal'

        # The next press. A 'down' toggle here would flip to 'normal', and
        # a toggle-decided button would take the stop branch instead.
        controller.error = None
        toggle.press()
        display.record_button()

        assert controller.start_calls == 2
        assert controller.stop_calls == 0
        assert toggle.state == 'down'

    def test_a_refusal_is_one_warning_and_the_button_draws_idle(self, monkeypatch):
        from modules.notification_center import Severity

        refusal = RecordingRefusedError(
            reason='recording_active',
            title='Recording Active',
            message='A recording is still finishing.',
        )
        controller = _Controller(error=refusal)
        display, toggle, _ctx, shown = _make_display(monkeypatch, controller)

        toggle.press()
        display.record_button()

        assert [(n.title, n.severity) for n in shown] == [('Recording Active', Severity.WARNING)]
        assert toggle.state == 'normal'


class TestStartOrStopIsTheApisAnswer:
    def test_a_press_while_recording_stops_whatever_the_toggle_reads(self, monkeypatch):
        controller = _Controller()
        controller.is_recording = True
        display, toggle, _ctx, _shown = _make_display(monkeypatch, controller)
        toggle.state = 'normal'
        toggle.press()  # a redraw had not yet caught up: the press reads 'down'

        display.record_button()

        assert controller.stop_calls == 1
        assert controller.start_calls == 0
        assert toggle.state == 'normal'

    def test_a_press_with_nothing_recording_starts_whatever_the_toggle_reads(self, monkeypatch):
        controller = _Controller()
        display, toggle, _ctx, _shown = _make_display(monkeypatch, controller)
        toggle.state = 'down'
        toggle.press()

        display.record_button()

        assert controller.start_calls == 1
        assert controller.stop_calls == 0
        assert toggle.state == 'down'

    def test_the_start_runs_on_the_pool_not_the_camera_lane(self, monkeypatch):
        controller = _Controller()
        display, toggle, ctx, _shown = _make_display(monkeypatch, controller)

        toggle.press()
        display.record_button()

        assert ctx.worker_pool.put.called
        assert not ctx.camera_executor.put.called

    def test_a_started_recording_starts_the_status_poll_once(self, monkeypatch):
        controller = _Controller()
        display, toggle, _ctx, _shown = _make_display(monkeypatch, controller)

        toggle.press()
        display.record_button()
        poll = display._recording_poll
        display.draw_record_button()

        assert poll is not None
        assert display._recording_poll is poll
