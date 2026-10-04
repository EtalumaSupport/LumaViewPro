# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``TOGGLE LED_<layer>`` is recorded when a person presses the button, and only then.

``update_led_state`` is both what the Enable LED button runs and the LED leg of
every ``apply_settings`` -- a slider drag, a typed commit, a drawer opening,
start-up. While the record sat in it, each of those wrote a toggle nobody made.
The record belongs to the press; the leg records nothing.

The last test pins the fact that lets the app write the button's ``state``
without a guard: a ``state`` write is not a press, so it never reaches the
button's ``on_release``. It runs real Kivy in a child process, because the
test environment stands Kivy's widgets in.
"""

from __future__ import annotations

import logging
import pathlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.app_context as _app_ctx

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def layer(monkeypatch):
    import ui.layer_control as layer_control

    context = SimpleNamespace(
        scope=SimpleNamespace(illumination=MagicMock()),
        io_executor=object(),
        settings={'Green': {'illumination_ma': 120.0}},
        ui_listener_bridge=SimpleNamespace(reconcile_led_buttons=MagicMock()),
    )
    monkeypatch.setattr(_app_ctx, 'ctx', context)
    monkeypatch.setattr(layer_control, 'submit_reported', lambda call, *a, **kw: call())
    widget = SimpleNamespace(
        layer='Green',
        _initializing=False,
        ids={'enable_led_btn': SimpleNamespace(state='down')},
        apply_settings=MagicMock(),
    )
    widget.update_led_state = layer_control.LayerControl.update_led_state.__get__(widget)
    return SimpleNamespace(
        widget=widget, illumination=context.scope.illumination, cls=layer_control.LayerControl
    )


def _toggles(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith('TOGGLE LED_')]


@pytest.mark.parametrize('state', ['down', 'normal'])
def test_the_apply_leg_records_no_toggle(layer, caplog, state):
    layer.widget.ids['enable_led_btn'].state = state
    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        layer.widget.update_led_state(apply_settings=False)

    assert _toggles(caplog) == []
    member = layer.illumination.led_on if state == 'down' else layer.illumination.led_off
    member.assert_called_once()


@pytest.mark.parametrize('state, word', [('down', 'ON'), ('normal', 'OFF')])
def test_a_press_records_one_toggle_and_drives_the_led(layer, caplog, state, word):
    layer.widget.ids['enable_led_btn'].state = state
    press = layer.cls.led_toggle.__get__(layer.widget)
    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        press()

    assert _toggles(caplog) == [f'TOGGLE LED_Green {word}']
    member = layer.illumination.led_on if state == 'down' else layer.illumination.led_off
    member.assert_called_once()
    layer.widget.apply_settings.assert_called_once_with(update_led=False)


_STATE_WRITE_IS_NOT_A_PRESS = r"""
import os
os.environ['KIVY_NO_ARGS'] = '1'
os.environ['KIVY_NO_CONSOLELOG'] = '1'
from kivy.uix.togglebutton import ToggleButton
released = []
button = ToggleButton()
button.bind(on_release=lambda *_: released.append('state write'))
button.state = 'down'
button.state = 'normal'
print(len(released))
released.clear()
button.bind(on_release=lambda *_: released.append('press'))
button.trigger_action(duration=0)
print(len(released))
"""


def test_a_state_write_is_not_a_press():
    done = subprocess.run(
        [sys.executable, '-c', _STATE_WRITE_IS_NOT_A_PRESS],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    after_writes, after_press = done.stdout.split()[-2:]
    assert after_writes == '0', 'a state write dispatched on_release'
    # The known positive: a press does reach on_release (both bound handlers).
    assert after_press == '2'
