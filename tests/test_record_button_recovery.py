# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""After a failed start, the next Record press must record on the FIRST press.

Record and Stop are one ToggleButton, so the button's state decides which
branch the next press takes. A start that failed after the refusal check
used to leave the toggle 'down' with nothing recording: the next press
flipped it to 'normal', which reads as "the user is stopping a live
recording", and the stop branch returns silently when there is nothing to
stop. The user pressed Record and nothing happened -- twice.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


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

    def stop(self):
        self.stop_calls += 1


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
    display = main_display.MainDisplay.__new__(main_display.MainDisplay)
    toggle = _FakeToggle()
    display.ids = {'record_btn': toggle}
    display._recording_poll = None

    monkeypatch.setattr(main_display, 'Clock', _ImmediateClock)
    ctx = MagicMock()
    ctx.session.manual_recording = controller
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    return display, toggle, ctx


class TestRecordButtonRecovery:
    def test_first_press_records_after_failed_start(self, monkeypatch):
        controller = _Controller(error=RuntimeError('scripted post-commit failure'))
        display, toggle, ctx = _make_display(monkeypatch, controller)

        toggle.press()
        display.record_button()
        assert ctx.camera_executor.put.called

        # The executor runs this; the failure must reach it, not be
        # swallowed, because the executor is what reports it to the user.
        with pytest.raises(RuntimeError):
            display._start_recording_task()
        assert toggle.state == 'normal'

        # The next press. A 'down' toggle here would flip to 'normal' and
        # take the stop branch instead.
        controller.error = None
        ctx.camera_executor.put.reset_mock()
        toggle.press()
        display.record_button()

        assert ctx.camera_executor.put.called
        assert controller.stop_calls == 0

    def test_refusal_still_resets_without_escaping(self, monkeypatch):
        # Preservation: a refusal is reported and handled, not re-raised --
        # the executor's generic popup would name it worse than its own
        # title and message do.
        refusal = RecordingRefusedError(
            reason='recording_active',
            title='Recording Active',
            message='A recording is still finishing.',
        )
        controller = _Controller(error=refusal)
        display, toggle, _ctx = _make_display(monkeypatch, controller)

        toggle.press()
        display._start_recording_task()
        assert toggle.state == 'normal'
