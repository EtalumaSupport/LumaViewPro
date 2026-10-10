# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The display's frame rate is what the window drew in the last second, read when asked.

The display rate was computed only when a frame was drawn and published as a
stored value, so when drawing stopped -- the live view paused, the display
thread stalled -- every reader (the title, ``[BUFFER METRICS]``) kept seeing
the last healthy rate as current: three samples a minute apart through a
paused display all read ``display_fps=38.6``. The rate is now counted from
the draws inside the last second at the moment it is read, so a display that
draws nothing reads 0. A saved protocol image put on the screen is not a live
frame and is not counted.

The real ScopeDisplay is a Kivy widget; it is built here on stub base
classes, so the real methods run without a GL context.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


class _StubWidget:
    def __init__(self, **kwargs):
        pass


def _real_base_module(name, **attrs):
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


for _name in (
    'kivy.uix',
    'kivy.graphics',
    'kivy.graphics.texture',
    'kivy.metrics',
    'kivy.properties',
    'kivy.input',
    'kivy.clock',
):
    sys.modules.setdefault(_name, MagicMock())

_real_base_module('kivy.uix.image', Image=_StubWidget)
_real_base_module('kivy.uix.widget', Widget=_StubWidget)

import ui.scope_display as scope_display_module
from modules import app_context
from ui.scope_display import ScopeDisplay


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    # The display module's own name for the time module, so no other thread
    # sees the fake clock.
    monkeypatch.setattr(scope_display_module, 'time', SimpleNamespace(monotonic=fake))
    return fake


@pytest.fixture
def display(monkeypatch, clock):
    monkeypatch.setattr(app_context, 'ctx', None)
    # The constructor draws its overlays on the canvas and binds Kivy
    # properties; the stub base has neither, so the instance gets them first.
    widget = ScopeDisplay.__new__(ScopeDisplay)
    widget.canvas = MagicMock()
    widget.bind = MagicMock()
    ScopeDisplay.__init__(widget)
    return widget


def _draw_live(display, n, clock, spacing_s):
    frame = np.zeros((4, 4), dtype=np.uint8)
    for _ in range(n):
        clock.now += spacing_s
        display.create_and_set_texture(frame.tobytes(), frame.shape, generation=0, live=True)


class TestTheRateIsTheLastSecondsDraws:
    def test_draws_in_the_last_second_are_the_rate(self, display, clock):
        _draw_live(display, 20, clock, 0.05)
        assert display.display_fps() == pytest.approx(20, abs=1)

    def test_a_display_that_stopped_drawing_reads_zero(self, display, clock):
        _draw_live(display, 30, clock, 1 / 30)
        assert display.display_fps() > 0
        # The live view is paused: nothing is drawn for two seconds.
        clock.now += 2.0
        assert display.display_fps() == 0

    def test_a_saved_image_put_on_the_screen_is_not_a_live_frame(self, display, clock, monkeypatch):
        # The hold runs its blit through Kivy's Clock; run it at once.
        monkeypatch.setattr(
            sys.modules['kivy.clock'].Clock, 'schedule_once', lambda fn, _t=0: fn(0)
        )
        image = np.zeros((4, 4), dtype=np.uint8)
        display.hold_protocol_saved_image(image, 8)
        assert display.frames_shown == 1, 'the hold did not put the image on the screen'
        assert display.display_fps() == 0
