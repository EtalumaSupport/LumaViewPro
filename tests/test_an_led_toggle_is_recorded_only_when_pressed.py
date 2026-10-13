# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``TOGGLE LED_<layer>`` is recorded when a person presses the button, and only then.

The LED leg once ran both from the Enable LED button and from every
``apply_settings`` -- a slider drag, a typed commit, a drawer opening,
start-up -- and while the record sat in it, each of those wrote a toggle
nobody made. The record belongs to the press; an apply records nothing.

The last test pins the fact that lets the app write the button's ``state``
without a guard: a ``state`` write is not a press, so it never reaches the
button's ``on_release``. It runs real Kivy in a child process, because the
test environment stands Kivy's widgets in.
"""

from __future__ import annotations

import logging
import os
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
        session=SimpleNamespace(light_shown_channel=MagicMock(), show_channel=MagicMock()),
        ui_listener_bridge=SimpleNamespace(reconcile_led_buttons=MagicMock()),
    )
    monkeypatch.setattr(_app_ctx, 'ctx', context)
    monkeypatch.setattr(layer_control, 'submit_reported', lambda call, *a, **kw: call())
    widget = SimpleNamespace(
        layer='Green',
        _initializing=False,
        ids={'enable_led_btn': SimpleNamespace(state='down')},
        show_as_shown=MagicMock(),
    )
    return SimpleNamespace(widget=widget, session=context.session, cls=layer_control.LayerControl)


def _toggles(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith('TOGGLE LED_')]


def test_an_apply_records_no_toggle(layer, caplog):
    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        layer.cls.apply_settings(layer.widget)

    assert _toggles(caplog) == []
    layer.session.show_channel.assert_called_once_with('Green')


@pytest.mark.parametrize('state, word', [('down', 'ON'), ('normal', 'OFF')])
def test_a_press_records_one_toggle_and_drives_the_led(layer, caplog, state, word):
    layer.widget.ids['enable_led_btn'].state = state
    press = layer.cls.led_toggle.__get__(layer.widget)
    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        press()

    assert _toggles(caplog) == [f'TOGGLE LED_Green {word}']
    layer.session.light_shown_channel.assert_called_once_with(state == 'down')


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


@pytest.mark.needs_display
def test_a_state_write_is_not_a_press(tmp_path):
    # Kivy reads its config from and writes its logs and icons into KIVY_HOME,
    # so the child gets one of its own rather than the shared ~/.kivy.
    done = subprocess.run(
        [sys.executable, '-c', _STATE_WRITE_IS_NOT_A_PRESS],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, 'KIVY_HOME': str(tmp_path)},
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    after_writes, after_press = done.stdout.split()[-2:]
    assert after_writes == '0', 'a state write dispatched on_release'
    # The known positive: a press does reach on_release (both bound handlers).
    assert after_press == '2'
