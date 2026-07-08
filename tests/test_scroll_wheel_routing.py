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

These are structural (code-shape) invariants, so they are asserted by reading
the source rather than instantiating the GL widgets (which need a real window).
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
