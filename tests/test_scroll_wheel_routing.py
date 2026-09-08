"""Regression: mouse-wheel routing across the sidebar panels.

Sidebar scroll was routed by whichever widget the pointer was over intercepting
the touch in its own on_touch_down, and they conflicted:

- A TextInput or RangeSlider under the pointer swallowed the wheel, so the menu
  would not scroll (the range slider even jumped a handle to the cursor).
- A ModSlider adjusts on scroll only when the user clicked it to arm it.

Fix (single-owner routing):
- ModSlider arms on click, shows a highlight, disarms when the cursor leaves
  its bounds, and marks the touch (`modslider_scroll_consumed`) when it adjusts.
- ModSliderAwareScrollView (the owner) consumes the wheel ONLY when that marker
  is set; any other widget claiming the touch is ignored so the menu scrolls.
- RangeSlider ignores wheel touches entirely (it used to grab the wheel and
  jump a handle to the cursor position).

(The live-image ctrl/shift modifier fix lives in test_shader_scroll_modifiers.)

The routing invariants are structural (code-shape), so they are asserted by
reading the source rather than instantiating the GL widgets (which need a real
window). The wheel DIRECTION contract at the bottom of this file is not: both
handlers run unbound against a stand-in, so nothing there depends on how the
source happens to be formatted.
"""

from __future__ import annotations

import re
from pathlib import Path

_UI = Path(__file__).resolve().parents[1] / 'ui'


def _read(name: str) -> str:
    return (_UI / name).read_text()


def _method_body(src: str, name: str) -> str:
    # Grab from `def name(` to the next def/decorator/class at the same or
    # outer indentation (or EOF). Return-annotation-safe.
    m = re.search(
        rf'def {re.escape(name)}\b.*?(?=\n    def |\n    @|\nclass |\Z)',
        src,
        re.DOTALL,
    )
    assert m is not None, f'{name}() not found'
    return m.group(0)


# --------------------------------------------------------------------------
# ModSlider: arm/disarm/highlight + marks the touch on adjust.
# --------------------------------------------------------------------------


def test_modslider_has_armed_property_and_highlight():
    src = _read('mod_slider.py')
    assert re.search(r'armed\s*=\s*BooleanProperty\(', src), (
        'ModSlider must expose an `armed` BooleanProperty driving the highlight.'
    )
    # The armed state must drive a visible change (the highlight colour).
    assert '_refresh_armed_visual' in src and 'self.armed' in src, (
        'The armed state must drive a visible highlight so it is obvious which '
        'slider the wheel will move.'
    )


def test_modslider_disarms_when_cursor_leaves():
    src = _read('mod_slider.py')
    assert 'mouse_pos=self._disarm_if_cursor_left' in src, (
        'An armed ModSlider must bind Window.mouse_pos to disarm when the '
        'cursor leaves its bounds (so the wheel returns to the menu).'
    )
    body = _method_body(src, '_disarm_if_cursor_left')
    assert 'self._disarm' in body, '_disarm_if_cursor_left must disarm on leave.'


def test_modslider_scroll_gated_on_armed_and_marks_touch():
    body = _method_body(_read('mod_slider.py'), 'on_touch_down')
    assert re.search(r'not\s+self\.armed', body), (
        'ModSlider.on_touch_down must fall through (return False) when not '
        'armed so the menu scrolls; only an armed slider adjusts on scroll.'
    )
    assert 'modslider_scroll_consumed' in body, (
        'An armed ModSlider that adjusts must mark the touch '
        '(modslider_scroll_consumed) so the scroll-view owner knows a slider '
        'consumed the wheel.'
    )
    # The old sticky-focus predicate must be gone.
    assert '_is_focused' not in _read('mod_slider.py'), (
        'The sticky _is_focused model was replaced by the hover-scoped `armed` '
        'state; no reference should remain.'
    )


# --------------------------------------------------------------------------
# ModSliderAwareScrollView (owner): consume only on the slider marker.
# --------------------------------------------------------------------------


def test_scrollview_consumes_only_on_slider_marker():
    # The owner's on_touch_down is the SECOND in the file; extract from the
    # ModSliderAwareScrollView class specifically.
    src = _read('mod_slider.py')
    owner = src[src.index('class ModSliderAwareScrollView') :]
    owner_body = _method_body(owner, 'on_touch_down')
    # It must only return True (consume, blocking the menu scroll) when the
    # armed-slider marker is present.
    ret_true = re.search(r'if\s+touch\.ud\.get\([\'"]modslider_scroll_consumed[\'"]\)', owner_body)
    assert ret_true is not None, (
        'ModSliderAwareScrollView must consume the wheel only when '
        'modslider_scroll_consumed is set; otherwise it must fall through to '
        'content scroll so a text box / range slider cannot block the menu.'
    )
    assert 'super().on_touch_down(touch)' in owner_body, (
        'The owner must fall through to ScrollView content scroll when no armed '
        'slider consumed the wheel.'
    )


# --------------------------------------------------------------------------
# RangeSlider: never grab / move a handle on a wheel touch.
# --------------------------------------------------------------------------


def test_range_slider_ignores_scroll_wheel():
    body = _method_body(_read('range_slider.py'), 'on_touch_down')
    guard = re.search(
        r'scrollup.*scrolldown.*\n\s*return False|scrolldown.*scrollup.*\n\s*return False',
        body,
        re.DOTALL,
    )
    assert guard is not None, (
        'RangeSlider.on_touch_down must return False for scrollup/scrolldown '
        'touches (let the wheel fall through) BEFORE it grabs the touch -- it '
        'used to grab the wheel and jump a handle to the cursor position.'
    )
    # The guard must precede the grab so no handle moves on a wheel tick.
    assert body.index('return False') < body.index('touch.grab'), (
        'The scroll-ignore guard must come before touch.grab(self).'
    )


# --------------------------------------------------------------------------
# Wheel DIRECTION: every wheel-driven control must move the same way.
# --------------------------------------------------------------------------


class TestWheelDirectionIsConsistent:
    """Rolling the wheel up must raise every wheel-driven control.

    Kivy names its wheel tokens for the document rather than the finger:
    physical wheel-up arrives as `touch.button == 'scrolldown'`, the roll
    that drags a document's content toward its top. Kivy's own ScrollView
    reads them that way, refusing 'scrolldown' once scroll_y >= 1.

    Three sites read wheel direction -- the slider adjust in mod_slider,
    and scroll-to-focus plus scroll-to-zoom in shader. For months the
    slider disagreed with the other two, so one roll raised the objective
    over the live image and lowered it on the Z slider. Nothing tested the
    agreement, and two separate analyses took the token names at face
    value and concluded the shader pair were the inverted ones.

    All three are asserted by RUNNING the real handler against a stand-in,
    not by reading source text: a pin on the source passes or fails on
    formatting, and would keep passing if the branch were gutted.
    """

    TOKEN_PHYSICAL_UP = 'scrolldown'
    TOKEN_PHYSICAL_DOWN = 'scrollup'

    @staticmethod
    def _armed_slider(value=50.0):
        """A stand-in exposing only what on_touch_down touches.

        ModSlider itself cannot be instantiated here -- the stubbed Widget
        base has no register_event_type -- so the handler runs unbound
        against this.
        """
        import types

        return types.SimpleNamespace(
            armed=True,
            step=5,
            value=value,
            min=0.0,
            max=100.0,
            collide_point=lambda *a: True,
            dispatch=lambda *a: None,
        )

    @staticmethod
    def _wheel(token):
        import types

        return types.SimpleNamespace(profile=['button'], button=token, pos=(0, 0), ud={})

    def _roll(self, monkeypatch, token, start=50.0):
        import sys

        from ui.mod_slider import ModSlider

        # Window.modifiers is a MagicMock attribute under the stubs and the
        # handler does set(Window.modifiers); give it a real empty sequence
        # so no shift multiplier applies.
        monkeypatch.setattr(sys.modules['kivy.core.window'].Window, 'modifiers', [], raising=False)
        slider = self._armed_slider(start)
        ModSlider.on_touch_down(slider, self._wheel(token))
        return slider.value

    def test_wheel_up_increases_slider_value(self, monkeypatch):
        assert self._roll(monkeypatch, self.TOKEN_PHYSICAL_UP) == 55.0, (
            'Rolling the wheel up over an armed slider must raise its value. '
            'This is the whole of the reported complaint: it used to lower it.'
        )

    def test_wheel_down_decreases_slider_value(self, monkeypatch):
        assert self._roll(monkeypatch, self.TOKEN_PHYSICAL_DOWN) == 45.0, (
            'Rolling the wheel down over an armed slider must lower its value.'
        )

    def test_slider_clamps_at_its_bounds(self, monkeypatch):
        assert self._roll(monkeypatch, self.TOKEN_PHYSICAL_UP, start=98.0) == 100.0
        assert self._roll(monkeypatch, self.TOKEN_PHYSICAL_DOWN, start=2.0) == 0.0

    @staticmethod
    def _viewer_stub(scale=1.0):
        """A stand-in for ShaderViewer holding only the scroll state.

        _scroll_inertia_window of 0 forces speed_factor to 1.0, so the
        queued delta is exactly one step and the assertion is on the SIGN
        without depending on wall-clock timing.
        """
        import types

        return types.SimpleNamespace(
            scale=scale,
            _scroll_last_time=0.0,
            _scroll_inertia_window=0.0,
            _scroll_z_pending=0.0,
            _scroll_z_trigger=lambda *a: None,
        )

    def _scroll_live_image(self, monkeypatch, token, ctrl_held, scale=1.0):
        import types

        from ui import shader

        # Neither side panel is under the pointer, so the wheel reaches the
        # image; the objective supplies the step size the handler scales.
        panel = types.SimpleNamespace(to_widget=lambda x, y: (x, y), collide_point=lambda *a: False)
        monkeypatch.setattr(
            shader._app_ctx,
            'ctx',
            types.SimpleNamespace(
                image_settings=panel,
                motion_settings=panel,
                session=types.SimpleNamespace(
                    controls_locked=False,
                    get_current_objective_info=lambda: (
                        None,
                        {'z_fine': 10.0, 'z_coarse': 100.0},
                    ),
                ),
            ),
        )
        monkeypatch.setattr(
            shader.Window, 'modifiers', ['ctrl'] if ctrl_held else [], raising=False
        )
        viewer = self._viewer_stub(scale)
        touch = types.SimpleNamespace(
            is_mouse_scrolling=True, pos=(0, 0), button=token, profile=['button'], ud={}
        )
        shader.ShaderViewer.on_touch_down(viewer, touch)
        return viewer

    def test_wheel_up_raises_the_objective(self, monkeypatch):
        up = self._scroll_live_image(monkeypatch, self.TOKEN_PHYSICAL_UP, ctrl_held=True)
        down = self._scroll_live_image(monkeypatch, self.TOKEN_PHYSICAL_DOWN, ctrl_held=True)
        assert up._scroll_z_pending > 0, (
            'ctrl + wheel up must queue a POSITIVE Z delta -- move_relative '
            'consumes this directly, so the sign IS the direction the '
            'objective travels. It must match the Z slider.'
        )
        assert down._scroll_z_pending < 0, 'ctrl + wheel down must lower the objective.'

    def test_wheel_up_zooms_in(self, monkeypatch):
        up = self._scroll_live_image(monkeypatch, self.TOKEN_PHYSICAL_UP, ctrl_held=False)
        down = self._scroll_live_image(
            monkeypatch, self.TOKEN_PHYSICAL_DOWN, ctrl_held=False, scale=50.0
        )
        assert up.scale > 1.0, (
            'Wheel up over the live image must zoom IN, matching the slider and the focus gesture.'
        )
        assert down.scale < 50.0, 'Wheel down over the live image must zoom out.'
