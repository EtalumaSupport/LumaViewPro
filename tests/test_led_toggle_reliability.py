# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for issue #617 -- LED toggle unreliability on slider move.

Original report: "When you change a slider, the toggle turns off then back on
very quickly. Sometimes it leaves the LED off."

Two root causes were identified. The first, a nuclear leds_off on every
apply, is gone with the apply's LED half: a settings change shows the layer
through the Session's show_channel, whose diff against the LED store sends
nothing to a channel already right (tests/test_led_authority_skeleton.py).
The second:

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
# The camera/LED/position listener closures moved out of
# lumaviewpro.py:on_start and now live in ui/listener_bridge.py.
# The feedback-loop safeguard tests below scan that file instead.


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
        settings_write_pos = body.find('update_settings(')
        assert init_check_pos != -1
        assert settings_write_pos != -1
        assert init_check_pos < settings_write_pos, (
            f'{handler}: _initializing check must come before settings write'
        )


class TestFixB2_ProgrammaticWidgetWriteWrapping:
    """#617 Fix B.2: any code path that programmatically writes to a slider
    or text widget must wrap the write in _initializing=True so the on_value
    handler doesn't re-enter."""

    def test_the_typed_paths_write_widgets_only_through_the_wrapped_writer(self):
        """ill_text and the generic text handler must not write a widget
        themselves. Both hand the value to the one writer, which is where the
        wrapping is asserted below -- an inline write here would be a slider
        write with no suppression, which the handler reads back as a drag."""
        tree = _parse(LAYER_CONTROL)
        body = _source_of(_find_method(tree, 'LayerControl', '_validate_and_apply_text_input'))
        assert '_show_value_on_widgets' in body, (
            '_validate_and_apply_text_input must render through _show_value_on_widgets'
        )
        assert 'slider.value' not in body, (
            '_validate_and_apply_text_input must not assign a slider value itself'
        )
        for method in ('ill_text', 'exp_text'):
            body = _source_of(_find_method(tree, 'LayerControl', method))
            assert '_validate_and_apply_text_input' in body, (
                f'{method} must commit through the shared text handler'
            )
            assert 'slider.value' not in body, f'{method} must not assign a slider value itself'

    def test_the_widget_writer_wraps_its_writes(self):
        """The wrapping the typed paths used to carry inline now lives in the
        one writer they share."""
        tree = _parse(LAYER_CONTROL)
        body = _source_of(_find_method(tree, 'LayerControl', '_show_value_on_widgets'))
        assert 'self._initializing = True' in body, (
            '_show_value_on_widgets must wrap programmatic widget writes'
        )
        # The slider write must be after the flag is raised.
        init_pos = body.find('self._initializing = True')
        slider_set_pos = body.find('slider.value')
        assert init_pos != -1 and slider_set_pos != -1
        assert init_pos < slider_set_pos, '_initializing=True must be set before the slider write'
        # And the flag must come back down: restored to what it was, so a
        # caller that is already suppressing keeps its own guard.
        assert 'self._initializing = was_initializing' in body, (
            '_show_value_on_widgets must restore the previous flag state, not clear it'
        )
