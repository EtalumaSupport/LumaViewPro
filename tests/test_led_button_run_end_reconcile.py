# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests: LED enable-toggles reconcile to driver truth at run end.

Bug shape: during a run, step navigation writes the upcoming step's
enable-toggle 'down' as a run indicator. A Stop landing between that
write and the step's LED-on leaves the widget claiming ON while the LED
is off: run-end cleanup restores hardware only, an all-dark restore
fires no LED events, and the edge-triggered listener bridge never
corrects the widget (bench evidence: 2026-07-29 bundle, BF toggle stuck
exactly at the aborted step's channel).

Contract under test, two halves:
1. Run completion reconciles ALL enable-toggles from driver truth
   (level-based, not edge-based) -- healing any stale indicator a
   Stop left behind.
2. The indicator write itself gates on the RUNNER's truth, not the
   ctx.protocol_running lockout flag: that flag deliberately outlives
   the run through the "Writing Files..." window, where a manual step
   would re-create the stale write with no later healer.
"""

import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

# ui.protocol_settings is a Kivy widget module; conftest mocks `kivy` but
# not the uix submodules, and ProtocolSettings/LayerControl subclass
# FloatLayout/BoxLayout (a bare MagicMock can't be subclassed). Real
# minimal bases for those; permissive MagicMocks for the rest.


class _StubWidget:
    def __init__(self, **kwargs):
        pass


def _real_base_module(name, **attrs):
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


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

_real_base_module('kivy.uix.floatlayout', FloatLayout=_StubWidget)
_real_base_module('kivy.uix.boxlayout', BoxLayout=_StubWidget)
_real_base_module('kivy.uix.scrollview', ScrollView=_StubWidget)
_real_base_module('kivy.uix.widget', Widget=_StubWidget)

import modules.app_context as _app_ctx
import ui.protocol_settings as ps_module
from modules.ui_listener_bridge import UIListenerBridge
from ui.protocol_settings import ProtocolSettings
from ui.step_navigation import go_to_step_update_ui


class _Button:
    def __init__(self, state='normal', **kwargs):
        self.state = state
        self.text = ''
        self.disabled = False


class _Stand:
    """Carries the run/files-complete handlers with hand-built widget state."""

    _scan_run_complete = ProtocolSettings._scan_run_complete
    _scan_files_complete = ProtocolSettings._scan_files_complete
    _update_file_write_status = ProtocolSettings._update_file_write_status

    def __init__(self):
        self._scan_files_completed_event = threading.Event()
        self._file_write_status_event = None
        self._wedge_recovery_offered = False
        self.ids = {
            'run_scan_btn': _Button(),
            'run_protocol_btn': _Button(),
            'run_autofocus_btn': _Button(),
        }

    def _reset_run_scan_button(self):
        pass

    def reset_autofocus_ui(self):
        pass

    def _update_write_lockout_button(self, name):
        pass


def _build_env(monkeypatch, queue_active=False, run_in_progress=False):
    """Fake ctx + real bridge; returns (stand, layers) with BF stale 'down'."""
    layers = {}

    def layer_lookup(layer):
        if layer not in layers:
            layers[layer] = SimpleNamespace(
                ids={'enable_led_btn': _Button()},
                set_step_state=lambda step: None,
            )
        return layers[layer]

    scope = MagicMock()
    scope.illumination.get_led_state.return_value = {'enabled': False}

    ctx = SimpleNamespace(
        ready=True,
        protocol_running=threading.Event(),
        file_io_executor=MagicMock(),
        stage=MagicMock(),
        image_settings=SimpleNamespace(
            layer_lookup=layer_lookup,
            set_expanded_layer=lambda layer: None,
            ids={'toggle_imagesettings': _Button(state='down')},
            toggle_settings=lambda: None,
        ),
        sequenced_capture_runner=SimpleNamespace(run_in_progress=lambda: run_in_progress),
        ui_listener_bridge=None,
        scope=scope,
    )
    ctx.protocol_running.set()
    ctx.file_io_executor.is_protocol_queue_active.return_value = queue_active
    ctx.file_io_executor.protocol_queue_size.return_value = 1 if queue_active else 0

    ctx.ui_listener_bridge = UIListenerBridge(
        scope=scope,
        ctx=ctx,
        stage=MagicMock(),
        ui_dispatcher=lambda callback, dt: callback(dt),
    )

    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    for name in (
        'live_histo_reverse',
        'reset_acquire_ui',
        'set_title_event_text',
        'reset_title',
    ):
        monkeypatch.setattr(ps_module, name, lambda *a, **k: None)

    stand = _Stand()
    # The stale state under test: BF's toggle claims ON, driver says dark.
    layer_lookup('BF').ids['enable_led_btn'].state = 'down'
    return stand, layers


def test_run_complete_reconciles_stale_toggle_to_driver_truth(monkeypatch):
    stand, layers = _build_env(monkeypatch, queue_active=False)

    stand._scan_run_complete()

    assert layers['BF'].ids['enable_led_btn'].state == 'normal', (
        'run-complete must reconcile enable-toggles from driver truth; '
        'the stale ON indicator survived the run end'
    )


def test_drain_window_step_does_not_write_the_run_indicator(monkeypatch):
    """Writing-files window: lockout flag still set, runner idle. A manual
    step must NOT force the indicator 'down' -- nothing later heals it."""
    _stand, layers = _build_env(monkeypatch, queue_active=True, run_in_progress=False)
    layers['BF'].ids['enable_led_btn'].state = 'normal'

    go_to_step_update_ui({'Color': 'BF'}, called_from_protocol=False)

    assert layers['BF'].ids['enable_led_btn'].state == 'normal', (
        'the run indicator must gate on runner truth; the lockout flag '
        'outlives the run through the writing-files window'
    )


def test_live_run_step_still_writes_the_run_indicator(monkeypatch):
    """Behavior-preservation guard (passes pre- and post-fix): during a
    live run the indicator still shows the upcoming step's channel."""
    _stand, layers = _build_env(monkeypatch, run_in_progress=True)
    layers['BF'].ids['enable_led_btn'].state = 'normal'

    go_to_step_update_ui({'Color': 'BF'}, called_from_protocol=True)

    assert layers['BF'].ids['enable_led_btn'].state == 'down'
