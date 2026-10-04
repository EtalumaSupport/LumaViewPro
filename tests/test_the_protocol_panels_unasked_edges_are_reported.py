# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The protocol panel's unasked edges hand a fault to the reporter, not the main loop.

Two of the panel's code paths run with no person waiting on them: the step
editor's debounced redraw, a clock trigger, and the shutdown override that
stops a live run when the application closes. A raise out of either used
to be caught at the site and turned into a blank label or an ERROR line
the panel wrote itself; uncaught, it would reach Kivy's main loop and
close the application. Both now run through the GUI's unasked boundary.

These tests drive the real handlers with the widget tree stubbed.
"""

import pathlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest


class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in (
    'kivy.app',
    'kivy.properties',
    'kivy.uix',
    'kivy.uix.label',
    'kivy.uix.popup',
    'kivy.lang',
    'kivy.metrics',
    'kivy.graphics',
):
    sys.modules.setdefault(_name, MagicMock())
for _name, _attr in (
    ('kivy.uix.floatlayout', 'FloatLayout'),
    ('kivy.uix.boxlayout', 'BoxLayout'),
    ('kivy.uix.scrollview', 'ScrollView'),
    ('kivy.uix.widget', 'Widget'),
):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        setattr(_mod, _attr, _StubWidget)
        sys.modules[_name] = _mod

import modules.app_context as _app_ctx
import ui.protocol_settings as ps
from modules.notification_center import notifications
from modules.protocol import Protocol

REPO = pathlib.Path(__file__).resolve().parent.parent

LAYER_CONFIG = {
    'autofocus': False,
    'false_color': False,
    'illumination_ma': 100.0,
    'gain_db': 0.0,
    'auto_gain': False,
    'exposure_ms': 10.0,
    'sum': 1,
    'acquire': 'image',
    'video_config': {},
    'focus': None,
}


def _protocol(num_steps):
    protocol = Protocol(
        tiling_configs_file_loc=REPO / 'data' / 'tiling.json',
        config={'steps': pd.DataFrame(), 'custom_step_count': 0},
    )
    for i in range(num_steps):
        protocol.insert_step(
            step_name=None,
            layer='BF',
            layer_config=LAYER_CONFIG,
            plate_position={'x': float(i), 'y': 0.0, 'z': 0.0},
            objective_id='10x Oly',
            stim_configs={},
            after_step=protocol.num_steps() - 1,
        )
    return protocol


class _Panel(ps.ProtocolSettings):
    """The real class, with only the widget tree stubbed."""

    def __init__(self, protocol, curr_step):
        self.ids = {
            'step_number_input': SimpleNamespace(text=''),
            'step_total_input': SimpleNamespace(text=''),
            'step_name_input': SimpleNamespace(text='', hint_text=''),
            'step_focus_z_label': SimpleNamespace(text='stale'),
        }
        self._protocol = protocol
        self.curr_step = curr_step


@pytest.fixture
def reported(monkeypatch):
    reports = []
    monkeypatch.setattr(
        notifications,
        'report_outcome',
        lambda exc, **kw: reports.append((type(exc).__name__, kw['category'], kw['solicited'])),
    )
    return reports


def test_a_step_editor_frame_on_a_stale_pointer_is_reported_not_raised(reported, monkeypatch):
    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace())
    panel = _Panel(_protocol(3), curr_step=7)

    panel._do_update_step_ui(0.05)

    assert reported == [('StepNotFoundError', 'UI:STEP_UI', False)]


def test_a_step_editor_frame_on_a_real_step_shows_it(reported, monkeypatch):
    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace())
    panel = _Panel(_protocol(3), curr_step=1)

    panel._do_update_step_ui(0.05)

    assert reported == []
    assert panel.ids['step_number_input'].text == '2'
    assert panel.ids['step_focus_z_label'].text == '0 um'


def test_the_shutdown_overrides_failure_is_reported_once_as_unasked(reported, monkeypatch):
    def force_reset(*, reason):
        raise RuntimeError(f'the engine would not reset for {reason}')

    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(sequenced_capture_runner=SimpleNamespace(force_reset=force_reset)),
    )
    panel = _Panel(_protocol(0), curr_step=-1)

    panel.cancel_all_protocols()

    assert reported == [('RuntimeError', 'UI:APP_SHUTDOWN', False)]
