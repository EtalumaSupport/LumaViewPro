# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Freeze button freezes the live picture and nothing else.

Pressing it freezes the picture; pressing it again unfreezes it. The LEDs,
captures, recordings and runs carry on as they were, and no camera is
needed. The display thread's pause is the one store of whether the picture
is frozen, and the button is drawn from it after every press, so the icon
never disagrees with the picture.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tests.scope_fakes import channel_reading


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
from modules.lumascope_api.illumination import LedState
from modules.scope_display_thread import ScopeDisplayThread
import ui.listener_bridge as listener_bridge
from ui.scope_display import ScopeDisplay
from tests.stock_channels import LED0


class _Toggle:
    """The Freeze ToggleButton: Kivy flips its state on press, then calls the handler."""

    def __init__(self):
        self.state = 'normal'

    def press(self):
        self.state = 'down' if self.state == 'normal' else 'normal'


class _Renderer:
    """What the display thread renders through; the subject is the pause, not the frame."""

    def _render_one_frame(self, **_kwargs):
        return None


def _main_display(scope):
    display = SimpleNamespace(scope=scope, ids={'live_btn': _Toggle()})
    display.draw_live_button = lambda: main_display.MainDisplay.draw_live_button(display)
    return display


def _press(display):
    display.ids['live_btn'].press()
    main_display.MainDisplay.cam_toggle(display)


@pytest.fixture
def thread(monkeypatch):
    t = ScopeDisplayThread()
    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(scope_display_thread=t))
    yield t
    t.stop()


def test_a_press_freezes_the_picture_and_leaves_the_lit_led_lit(thread, sim_scope):
    sim_scope.illumination.led_on(LED0, 50.0)
    display = _main_display(sim_scope)

    _press(display)
    assert thread.is_paused
    assert display.ids['live_btn'].state == 'down'
    blue = channel_reading(sim_scope.illumination, LED0)
    assert (blue.state, blue.illumination_ma) == (LedState.LIT, 50.0)

    _press(display)
    assert not thread.is_paused
    assert display.ids['live_btn'].state == 'normal'
    assert channel_reading(sim_scope.illumination, LED0).state is LedState.LIT


def test_a_press_with_the_camera_gone_freezes_the_picture(thread):
    # The simulator's gap: its camera's disconnect does not reach
    # ImagingAPI.active until a reconnect poll.
    # a stand-in by design: a fault the simulator cannot yet produce
    scope = SimpleNamespace(imaging=SimpleNamespace(active=False))
    display = _main_display(scope)

    _press(display)

    assert thread.is_paused
    assert display.ids['live_btn'].state == 'down'


def test_changing_the_live_rate_keeps_a_frozen_picture_frozen(thread, monkeypatch):
    renderer = _Renderer()
    thread.start(renderer, fps=30)
    generation = thread.generation
    # a stand-in by design: the settings write is not the subject; the
    # window's apply of a changed rate is
    ctx = SimpleNamespace(
        scope_display_thread=thread,
        settings={'live_view_fps': 30},
        live_view_fps=30,
        scope_display=SimpleNamespace(
            start=lambda fps=None: ScopeDisplay.start(renderer, fps),
            stop=lambda: ScopeDisplay.stop(renderer),
        ),
    )
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    display = _main_display(scope=None)
    _press(display)

    ctx.settings['live_view_fps'] = 10
    # The window's apply of a stored rate, as the settings listener runs it.
    listener_bridge._apply_live_view_fps(ctx, 'live_view_fps')

    assert thread.is_paused
    assert thread.fps == 10
    assert thread.generation == generation
