# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for issue #617 -- LED toggle unreliability on slider move.

Original report: "When you change a slider, the toggle turns off then back on
very quickly. Sometimes it leaves the LED off."

Two root causes were identified and fixed in the same commit:

Fix A -- `disable_leds_for_other_layers` overreach:
  disable_leds_for_other_layers() once fired a nuclear leds_off_async()
  + led_on_async() on every apply_settings to ensure only one LED is
  physically on when switching channels. That blanked every LED and
  re-lit the current one -- correct for layer switches but wrong for
  slider-only moves on the active layer, which fire apply_settings many
  times per drag. The nuclear leds_off also cleared the LED-state cache,
  so the following led_on could not self-skip: visible LED flicker
  (off->on) on every slider move.

  Fix: switch off only the OTHER layers' LEDs, one channel at a time.
  led_off self-skips a channel already off, so a plain slider move with
  nothing else lit touches the bus zero times. This layer's own LED is
  owned by update_led_state and is never disturbed here, so it can no
  longer blink. One LED on at a time (the #614 guarantee) is preserved.

Fix B -- Widget handler recursion via programmatic widget writes:
  ill_text() -> sets slider.value -> on_value fires -> ill_slider() -> which
  writes settings and schedules apply_settings. The handler's `_initializing`
  check only guarded logging and the apply call, not the settings write or
  the recursive chain. Similarly, the camera listener's _update_camera_ui
  sets slider values and can trigger the handler.

  Fix: ill_slider/gain_slider/exp_slider now early-return on _initializing.
  ill_text/_validate_and_apply_text_input/_update_camera_ui wrap their
  programmatic widget writes in self._initializing=True so the on_value
  handler no-ops. set_step_state is a pure widget setter -- manual step
  navigation writes the settings itself in step_navigation.go_to_step --
  so the early return is safe.

These tests use AST-based source checks to pin the fix structure. Behavioral
tests on the Kivy widget tree would require a full Kivy app context and are
out of scope for unit tests (see tests/test_headless_protocol.py for the
headless-executor-only pattern).
"""

import ast
import pathlib

import pytest


REPO = pathlib.Path(__file__).parent.parent
LAYER_CONTROL = REPO / 'ui' / 'layer_control.py'
LUMAVIEWPRO = REPO / 'lumaviewpro.py'
# LVP-A-6 (2026-05-04): the camera/LED/position listener closures moved
# from lumaviewpro.py:on_start into modules/ui_listener_bridge.py.
# The #617 safeguard tests below scan this file instead.
UI_LISTENER_BRIDGE = REPO / 'modules' / 'ui_listener_bridge.py'


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'))


def _find_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in {tree}')


def _source_of(node: ast.AST) -> str:
    return ast.unparse(node)


class TestFixA_DisableLedsForOtherLayersGuard:
    """#617 Fix A: disable_leds_for_other_layers must switch off only the
    OTHER layers' LEDs (one channel at a time), never blank every LED and
    re-light this one. The nuclear leds_off cleared the LED-state cache, so
    the following led_on could not self-skip and blinked the channel off->on
    on every slider move."""

    def _body(self):
        source = LAYER_CONTROL.read_text()
        idx = source.find('def disable_leds_for_other_layers')
        assert idx != -1
        return source[idx : idx + 2500]

    def test_disable_leds_offs_other_layers_individually(self):
        """Other layers are switched off one channel at a time
        (led_off_async), preserving the #614 one-LED-at-a-time guarantee."""
        body = self._body()
        assert 'led_off_async(' in body, (
            'disable_leds_for_other_layers must turn off other layers '
            'individually with led_off_async (#614 one LED at a time)'
        )

    def test_disable_leds_does_not_nuke_all_leds(self):
        """The cache-clearing nuclear leds_off must NOT be used here -- it was
        the off->on blink source on every slider move (#617)."""
        body = self._body()
        assert 'leds_off_async()' not in body, (
            'nuclear leds_off_async() clears the LED-state cache and blinks '
            'an already-correct channel off->on; off the other layers '
            'individually instead (#617)'
        )

    def test_disable_leds_does_not_relight_self(self):
        """This layer's LED current is owned by update_led_state; disable_leds
        must NOT re-light it. Re-lighting was only needed to undo the nuclear
        leds_off, which also turned this layer off."""
        body = self._body()
        assert 'led_on_async(' not in body, (
            'disable_leds_for_other_layers must not re-light this layer '
            '(update_led_state owns this layer current); re-lighting here '
            'was only there to undo the nuclear leds_off (#617)'
        )


class TestFixB1_SliderHandlerEarlyReturn:
    """#617 Fix B.1: ill_slider, gain_slider, exp_slider must early-return
    when _initializing=True, so programmatic widget writes don't re-enter."""

    @pytest.mark.parametrize('handler', ['ill_slider', 'gain_slider', 'exp_slider'])
    def test_handler_has_initializing_early_return(self, handler):
        tree = _parse(LAYER_CONTROL)
        func = _find_method(tree, 'LayerControl', handler)
        body = _source_of(func)
        # Must have an early return on self._initializing
        assert 'if self._initializing:' in body, (
            f'{handler} must have `if self._initializing: return` (#617 Fix B.1)'
        )
        # The return must come before the settings write / logger / apply call.
        # We check this by finding the first occurrence of each and asserting
        # the early return is first in the function body order.
        init_check_pos = body.find('if self._initializing:')
        settings_write_pos = body.find('settings[self.layer]')
        assert init_check_pos != -1
        assert settings_write_pos != -1
        assert init_check_pos < settings_write_pos, (
            f'{handler}: _initializing check must come before settings write'
        )


class TestFixB2_ProgrammaticWidgetWriteWrapping:
    """#617 Fix B.2: any code path that programmatically writes to a slider
    or text widget must wrap the write in _initializing=True so the on_value
    handler doesn't re-enter."""

    def test_ill_text_wraps_slider_write(self):
        tree = _parse(LAYER_CONTROL)
        func = _find_method(tree, 'LayerControl', 'ill_text')
        body = _source_of(func)
        # The slider.value assignment must be inside a self._initializing=True block
        assert 'self._initializing = True' in body, (
            'ill_text must wrap programmatic widget writes in _initializing=True (#617)'
        )
        assert 'self._initializing = False' in body, (
            'ill_text must reset _initializing = False after the write'
        )

    def test_validate_and_apply_text_input_wraps_widget_writes(self):
        tree = _parse(LAYER_CONTROL)
        func = _find_method(tree, 'LayerControl', '_validate_and_apply_text_input')
        body = _source_of(func)
        assert 'self._initializing = True' in body, (
            '_validate_and_apply_text_input must wrap widget writes (#617)'
        )
        # slider.value set must be after _initializing = True
        init_pos = body.find('self._initializing = True')
        slider_set_pos = body.find('slider.value')
        assert init_pos != -1 and slider_set_pos != -1
        assert init_pos < slider_set_pos, '_initializing=True must be set before the slider write'

    def test_update_camera_ui_is_text_only(self):
        """The camera listener handler must only update text widgets,
        never slider.value.

        Structural fix (4.1 session 13 follow-up to #617): the slider is
        the user-input source of truth. The listener exists to display
        the actual camera value in the readout text, not to push values
        back into the slider -- doing so was the root cause of the
        handler-recursion feedback loop the `_initializing` flag was
        papering over.

        LVP-A-6 (2026-05-04): the closure moved from
        ``lumaviewpro.py:on_start`` into
        ``modules/ui_listener_bridge.py:UIListenerBridge._on_camera_setting_changed``
        (with the inner ``_update_camera_ui`` closure). Scanning the new
        location.
        """
        source = UI_LISTENER_BRIDGE.read_text()
        idx = source.find('def _update_camera_ui')
        assert idx != -1, '_update_camera_ui not found in ui_listener_bridge.py'
        # Function is ~3000 chars; slice large enough to catch the body
        body = source[idx : idx + 3500]

        # Text writes must still be present -- this is the whole point of
        # the listener.
        assert 'gain_text' in body, '_update_camera_ui must still update gain_text for #617 display'
        assert 'exp_text' in body, '_update_camera_ui must still update exp_text for #617 display'

        # Must NOT write to slider.value -- that path is the recursion root.
        for forbidden in (
            "gain_slider'].value =",
            "exp_slider'].value =",
            '"gain_slider"].value =',
            '"exp_slider"].value =',
        ):
            assert forbidden not in body, (
                f'_update_camera_ui must not assign to slider.value; found '
                f'{forbidden!r} -- this is the recursion root from #617'
            )

        # Must still respect _initializing set by other code paths
        # (set_step_state, layer switches). We don't write our own, but
        # we do early-return when someone else set it.
        assert 'if layer_obj._initializing:' in body, (
            '_update_camera_ui must still early-return when another code '
            'path has set layer_obj._initializing'
        )

        # Must NOT set _initializing itself anymore -- no slider writes
        # means no recursion to protect against.
        assert 'layer_obj._initializing = True' not in body, (
            '_update_camera_ui should not set _initializing; text-only '
            'updates do not trigger handler recursion'
        )
