# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for P2 bug fix -- stim slider writes must match text-path type.

Audit source: /Users/ericweiner/Documents/Firmware/docs/NUMBER_INPUT_AUDIT_2026-04-21.md
              sec.P2 Pulse Count + Pulse Width type-mismatch.

Bug
---
`stim_pulse_count_slider` / `stim_pulse_width_slider` previously wrote Kivy's
raw `float` slider value to settings[layer]['stim_config'][...]. The
corresponding text handlers (`stim_pulse_count_text`, `stim_pulse_width_text`)
cast via `_validate_and_apply_text_input(..., cast=int)`.

Dragging slider stored `float`; clicking out of text stored `int`. Firmware
STIM expects integers for both fields (pulse_count is a discrete count,
pulse_width is integer milliseconds per V31_COMMAND_REFERENCE + FW4.0
stim_pulse_train), so the paths diverged silently.

Test approach
-------------
Kivy's BoxLayout is MagicMock'd in the test env, which means
`class LayerControl(BoxLayout)` produces a MagicMock at class-body time and
its real method bodies are unreachable via attribute lookup. To test the
actual method bodies we parse `ui/layer_control.py` with `ast`, extract the
two handler FunctionDef nodes, and exec them into a clean namespace with the
globals the methods need (numpy, logger, gui_logger, _app_ctx stub). The
resulting plain functions can then be called against a SimpleNamespace "self".

Covers both slider handlers and the cross-path invariant:
    for any mix of slider + text writes, settings stays `int`.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


REPO = pathlib.Path(__file__).parent.parent
LAYER_CONTROL_SRC = REPO / 'ui' / 'layer_control.py'


# ---------------------------------------------------------------------------
# Extract bare functions from the LayerControl class source
# ---------------------------------------------------------------------------


def _extract_method_source(source: str, class_name: str, method_name: str) -> str:
    """Return the source text of a method body as a top-level function."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.unparse(child)
    raise AssertionError(f'{class_name}.{method_name} not found in source')


def _compile_handler(method_name: str, extra_globals: dict):
    """Compile a LayerControl method into a standalone callable.

    The method's body is unparsed (no decorator, no class indentation) and
    exec'd into a namespace holding the module-scope names it uses: numpy,
    logger, gui_logger, and _app_ctx.
    """
    src = LAYER_CONTROL_SRC.read_text()
    fn_src = _extract_method_source(src, 'LayerControl', method_name)
    ns = {
        'np': np,
        **extra_globals,
    }
    exec(compile(fn_src, f'<layer_control::{method_name}>', 'exec'), ns)
    return ns[method_name]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _make_slider_mock(value: float, minv: float = 0.0, maxv: float = 1000.0) -> MagicMock:
    m = MagicMock()
    m.value = value
    m.min = minv
    m.max = maxv
    m.disabled = False
    return m


def _make_text_mock(text: str) -> MagicMock:
    m = MagicMock()
    m.text = text
    return m


def _make_fake_self(layer: str) -> SimpleNamespace:
    fake = SimpleNamespace()
    fake.layer = layer
    fake._initializing = False
    fake.ids = {}
    fake.apply_settings = MagicMock()
    fake.apply_gain_slider = MagicMock()
    fake.apply_exp_slider = MagicMock()
    fake.apply_ill_slider = MagicMock()
    return fake


# ---------------------------------------------------------------------------
# Compiled handlers (fixture-scoped so parse happens once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def handler_globals():
    """Globals the extracted handlers need at call time."""
    app_ctx_stub = SimpleNamespace(ctx=SimpleNamespace(settings={}))
    return {
        'logger': MagicMock(),
        'gui_logger': MagicMock(),
        '_app_ctx': app_ctx_stub,
    }


@pytest.fixture
def settings(handler_globals):
    """Reset the shared settings dict each test, via the stub app_ctx."""
    s = {
        'Blue': {
            'stim_config': {
                'pulse_count': 5,
                'pulse_width': 10,
                'frequency': 1.0,
            },
        },
    }
    handler_globals['_app_ctx'].ctx.settings = s
    return s


@pytest.fixture(scope='module')
def pulse_count_handler(handler_globals):
    return _compile_handler('stim_pulse_count_slider', handler_globals)


@pytest.fixture(scope='module')
def pulse_width_handler(handler_globals):
    return _compile_handler('stim_pulse_width_slider', handler_globals)


# ---------------------------------------------------------------------------
# pulse_count -- slider path
# ---------------------------------------------------------------------------


class TestPulseCountSliderWritesInt:
    """Slider path must cast to int, matching the text path's cast=int."""

    @pytest.mark.parametrize('slider_val', [7.5, 10.3, 0.9, 42.0, 99.999])
    def test_slider_stores_int_type(self, settings, pulse_count_handler, slider_val):
        fake = _make_fake_self('Blue')
        fake.ids['stim_pulse_count_slider'] = _make_slider_mock(slider_val)

        pulse_count_handler(fake)

        stored = settings['Blue']['stim_config']['pulse_count']
        assert type(stored) is int, (
            f'slider wrote {type(stored).__name__} ({stored!r}); '
            f'expected int. Float would desync from text-path cast=int.'
        )
        assert stored == int(slider_val)


class TestPulseWidthSliderWritesInt:
    @pytest.mark.parametrize('slider_val', [7.5, 10.3, 0.9, 42.0, 99.999])
    def test_slider_stores_int_type(self, settings, pulse_width_handler, slider_val):
        fake = _make_fake_self('Blue')
        fake.ids['stim_pulse_width_slider'] = _make_slider_mock(slider_val)

        pulse_width_handler(fake)

        stored = settings['Blue']['stim_config']['pulse_width']
        assert type(stored) is int
        assert stored == int(slider_val)


# ---------------------------------------------------------------------------
# Cross-path consistency: slider then text (and vice versa) must leave int
# ---------------------------------------------------------------------------


class TestCrossPathConsistency:
    """After any mix of slider and text updates, settings value stays int.

    The text path goes through `_validate_and_apply_text_input` which already
    has `cast=int` wiring (see layer_control.py lines 119+127). Stubbing it
    here would only re-test our stub, so we model the text path's final
    settings write in-line -- the thing under test is that the slider path
    does not later clobber the int with a float.
    """

    def test_slider_then_simulated_text(self, settings, pulse_count_handler):
        fake = _make_fake_self('Blue')
        fake.ids['stim_pulse_count_slider'] = _make_slider_mock(7.9)

        pulse_count_handler(fake)
        assert type(settings['Blue']['stim_config']['pulse_count']) is int

        # Simulate the text path's final write (cast=int, see
        # _validate_and_apply_text_input lines 119+127 in layer_control.py)
        settings['Blue']['stim_config']['pulse_count'] = int(float('12'))
        assert type(settings['Blue']['stim_config']['pulse_count']) is int

    def test_simulated_text_then_slider(self, settings, pulse_width_handler):
        # Text path first (simulated cast=int write)
        settings['Blue']['stim_config']['pulse_width'] = int(float('15'))
        assert type(settings['Blue']['stim_config']['pulse_width']) is int

        fake = _make_fake_self('Blue')
        fake.ids['stim_pulse_width_slider'] = _make_slider_mock(22.7)
        pulse_width_handler(fake)
        assert type(settings['Blue']['stim_config']['pulse_width']) is int
        assert settings['Blue']['stim_config']['pulse_width'] == 22


# ---------------------------------------------------------------------------
# Sanity: apply_settings is still called (handler wiring unchanged)
# ---------------------------------------------------------------------------


class TestHandlerStillAppliesSettings:
    def test_pulse_count_calls_apply(self, settings, pulse_count_handler):
        fake = _make_fake_self('Blue')
        fake.ids['stim_pulse_count_slider'] = _make_slider_mock(3.0)
        pulse_count_handler(fake)
        fake.apply_settings.assert_called_once()

    def test_pulse_width_calls_apply(self, settings, pulse_width_handler):
        fake = _make_fake_self('Blue')
        fake.ids['stim_pulse_width_slider'] = _make_slider_mock(3.0)
        pulse_width_handler(fake)
        fake.apply_settings.assert_called_once()
