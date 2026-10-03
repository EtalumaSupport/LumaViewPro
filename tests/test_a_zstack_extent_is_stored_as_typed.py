# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A z-stack step size or range is stored as typed; the build refuses a bad one.

The two boxes clamped a negative entry to 0 and substituted 0 for one that
was not a number, then stored that 0 -- a range decided in a widget. A typed
-5 was later refused as "step size (0)", and an emptied box stored an extent
nobody asked for. A number is now stored as typed, whatever its sign: the
protocol build refuses a stack whose step or range is not above zero, naming
what was typed, and the Steps field reads 0 meanwhile. An entry that is not
a number is not a request: the box goes back to what settings hold.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import ui.zstack as zstack_module
from ui.zstack import ZStack


@pytest.fixture
def panel(monkeypatch):
    settings = {'zstack': {'step_size': 5.0, 'range': 50.0}}
    monkeypatch.setattr(
        _app_ctx, 'ctx', SimpleNamespace(settings=settings, settings_lock=threading.Lock())
    )
    monkeypatch.setattr(zstack_module.gui_logger, 'text_input', lambda *a: None)
    monkeypatch.setattr(
        zstack_module.common_utils,
        'convert_zstack_reference_position_setting_to_config',
        lambda text_label: 'center',
    )
    zstack = ZStack.__new__(ZStack)
    zstack.ids = {
        'zstack_stepsize_id': SimpleNamespace(text='5'),
        'zstack_range_id': SimpleNamespace(text='50'),
        'zstack_spinner': SimpleNamespace(text='Current Position at Center'),
        'zstack_steps_id': SimpleNamespace(text=''),
    }
    return zstack, settings


def test_a_negative_step_is_stored_as_typed_and_the_stack_has_no_steps(panel):
    zstack, settings = panel
    zstack.ids['zstack_stepsize_id'].text = '-5'
    zstack.set_steps()
    assert settings['zstack']['step_size'] == -5.0
    assert zstack.ids['zstack_stepsize_id'].text == '-5'
    assert zstack.ids['zstack_steps_id'].text == '0'


def test_an_emptied_range_stores_nothing_and_the_box_goes_back(panel):
    zstack, settings = panel
    zstack.ids['zstack_range_id'].text = ''
    zstack.set_steps()
    assert settings['zstack']['range'] == 50.0
    assert zstack.ids['zstack_range_id'].text == '50.0'


def test_a_number_is_stored_and_counted(panel):
    zstack, settings = panel
    zstack.ids['zstack_stepsize_id'].text = '2.5'
    zstack.set_steps()
    assert settings['zstack']['step_size'] == 2.5
    assert zstack.ids['zstack_steps_id'].text == '21'
