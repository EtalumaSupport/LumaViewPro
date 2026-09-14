# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Toggling auto-gain off writes back the value the API's lock result
carries, and nothing else.

Bug shape: the toggle-off read gain and exposure straight off the camera
while its auto loop was still running, clipped the exposure to a per-class
floor it owned, and wrote the floored number to settings and slider with
nothing recording the raw value anywhere. The callback now consumes the
API's lock result: the API decided the value to store (the achieved
exposure floored to the class's usable floor) and has already told the
user about a limit state; the GUI writes what it is handed and clips only
to its own slider range.
"""

from __future__ import annotations

import ast
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import modules.common_utils as real_common_utils
from modules.lumascope_api.imaging import (
    AutoGainConvergence,
    AutoGainLock,
    stored_exposure_after_lock,
)
from tests import ast_seams
from ui.layer_control import _LAYER_VALUE_WIDGETS


def _method_source(name: str) -> str:
    node = ast_seams.find_def('ui/layer_control.py', name, class_name='LayerControl')
    assert node is not None, f'LayerControl.{name} not found'
    return ast.unparse(node)


def _compile_callback():
    """update_auto_gain_cb as a standalone callable, together with the render
    path it hands the stored values to -- all three compiled from the same
    production source, so what puts the value on the slider here is what does
    it in the app."""
    ns = {
        'np': np,
        'logger': MagicMock(),
        'common_utils': real_common_utils,
        '_app_ctx': SimpleNamespace(
            ctx=SimpleNamespace(settings={}, settings_lock=threading.Lock())
        ),
        '_LAYER_VALUE_WIDGETS': _LAYER_VALUE_WIDGETS,
    }
    for name in (
        '_show_value_on_widgets',
        'render_layer_values_from_settings',
        'update_auto_gain_cb',
    ):
        exec(compile(_method_source(name), f'<{name}>', 'exec'), ns)
    return ns['update_auto_gain_cb'], ns['_app_ctx']


def _lock(state, exposure_ms, gain_db, floor_ms, ceiling_ms) -> AutoGainLock:
    """A lock result as the API builds it: the value to store is the API's
    decision, computed by the same rule the lock uses."""
    stored = stored_exposure_after_lock(exposure_ms, floor_ms) if exposure_ms is not None else None
    return AutoGainLock(
        state, exposure_ms, gain_db, floor_ms, ceiling_ms, stored_exposure_ms=stored
    )


def _fake_layer(layer: str, slider_min: float, slider_max: float = 1000.0):
    fake = SimpleNamespace()
    fake.layer = layer
    # The render path suppresses the layer's handlers around its writes.
    fake._initializing = False
    fake.ids = {
        'auto_gain': MagicMock(state='normal'),
        'exp_slider': MagicMock(min=slider_min, max=slider_max),
        'gain_slider': MagicMock(min=0, max=48, value=0),
        'gain_text': MagicMock(text='0'),
        'exp_text': MagicMock(text='0'),
    }
    fake.apply_settings = MagicMock()
    return fake


def _bind(fake, cb):
    # The callback delegates the widget writes; the fake needs that path for
    # the same reason the real layer does. cb.__globals__ IS the namespace it
    # was compiled into, so these are the same functions it will call.
    ns = cb.__globals__
    for name in ('_show_value_on_widgets', 'render_layer_values_from_settings'):
        setattr(fake, name, ns[name].__get__(fake))
    return lambda result: cb(fake, result=result)


def test_at_minimum_stores_the_floor_the_api_decided():
    cb, app_ctx = _compile_callback()
    app_ctx.ctx.settings = {'Blue': {'exposure_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}
    fake = _fake_layer('Blue', slider_min=1.0)
    run = _bind(fake, cb)
    run((False, _lock(AutoGainConvergence.AT_MINIMUM, 0.4, 3.0, 1.0, 200.0)))
    assert app_ctx.ctx.settings['Blue']['exposure_ms'] == 1.0
    assert fake.ids['exp_slider'].value == 1.0
    assert app_ctx.ctx.settings['Blue']['gain_db'] == 3.0
    fake.apply_settings.assert_called_once()


def test_transmitted_at_camera_floor_keeps_the_class_floor():
    cb, app_ctx = _compile_callback()
    app_ctx.ctx.settings = {'BF': {'exposure_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}
    fake = _fake_layer('BF', slider_min=0.01)
    run = _bind(fake, cb)
    run((False, _lock(AutoGainConvergence.AT_MINIMUM, 0.03, 0.0, 0.1, 50.0)))
    assert app_ctx.ctx.settings['BF']['exposure_ms'] == 0.1


def test_maxed_writes_the_ceiling():
    cb, app_ctx = _compile_callback()
    app_ctx.ctx.settings = {'Red': {'exposure_ms': 50.0, 'gain_db': 0.0, 'auto_gain': True}}
    fake = _fake_layer('Red', slider_min=1.0)
    run = _bind(fake, cb)
    run((False, _lock(AutoGainConvergence.MAXED, 200.0, 20.0, 1.0, 200.0)))
    assert app_ctx.ctx.settings['Red']['exposure_ms'] == 200.0


def test_converged_writes_the_achieved_value():
    cb, app_ctx = _compile_callback()
    app_ctx.ctx.settings = {'BF': {'exposure_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}
    fake = _fake_layer('BF', slider_min=0.01)
    run = _bind(fake, cb)
    run((False, _lock(AutoGainConvergence.CONVERGED, 5.0, 2.0, 0.1, 50.0)))
    assert app_ctx.ctx.settings['BF']['exposure_ms'] == 5.0


def test_failed_keeps_previous_settings():
    cb, app_ctx = _compile_callback()
    app_ctx.ctx.settings = {'BF': {'exposure_ms': 42.0, 'gain_db': 7.0, 'auto_gain': True}}
    fake = _fake_layer('BF', slider_min=0.01)
    run = _bind(fake, cb)
    run((False, _lock(AutoGainConvergence.FAILED, None, None, 0.1, 50.0)))
    assert app_ctx.ctx.settings['BF']['exposure_ms'] == 42.0
    assert app_ctx.ctx.settings['BF']['gain_db'] == 7.0


def test_program_start_and_toggle_on_leave_settings_alone():
    cb, app_ctx = _compile_callback()
    app_ctx.ctx.settings = {'BF': {'exposure_ms': 42.0, 'gain_db': 7.0, 'auto_gain': True}}
    fake = _fake_layer('BF', slider_min=0.01)
    run = _bind(fake, cb)
    run((True, _lock(AutoGainConvergence.CONVERGED, 5.0, 2.0, 0.1, 50.0)))
    assert app_ctx.ctx.settings['BF']['exposure_ms'] == 42.0
    fake.ids['auto_gain'].state = 'down'
    run((False, None))
    assert app_ctx.ctx.settings['BF']['exposure_ms'] == 42.0
    assert app_ctx.ctx.settings['BF']['auto_gain'] is True
