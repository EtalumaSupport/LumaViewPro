# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A z-stack step size or range is written as typed, and the writer holds its range.

The two boxes clamped a negative entry to 0 and substituted 0 for one that
was not a number, then stored that 0 -- a range decided in a widget. A typed
-5 was later refused as "step size (0)", and an emptied box stored an extent
nobody asked for. A number is now handed to the writer as typed: a negative
one is refused there, in the writer's words naming it, and the box shows what
is stored; a 0 is stored, and the build refuses a stack whose step or range
is not above zero. An entry that is not a number is not a request: the box
goes back to what settings hold.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import ui.zstack as zstack_module
from modules.exceptions import SettingRefusedError
from modules.notification_center import notifications
from ui.zstack import ZStack
from tests.settings_fixtures import settings_writer


@pytest.fixture
def panel(monkeypatch):
    settings = {'zstack': {'step_size': 5.0, 'range': 50.0}}
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            settings=settings,
            settings_lock=threading.Lock(),
            update_settings=settings_writer(settings),
        ),
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


def test_a_negative_step_is_refused_by_the_writer_and_the_box_shows_the_stored_one(
    panel, monkeypatch
):
    zstack, settings = panel
    reported = []
    monkeypatch.setattr(
        notifications, 'report_outcome', lambda outcome, **kw: reported.append(outcome)
    )
    zstack.ids['zstack_stepsize_id'].text = '-5'
    zstack.set_steps()
    assert [(type(e), e.reason, e.path) for e in reported] == [
        (SettingRefusedError, 'out_of_range', 'zstack.step_size')
    ]
    assert settings['zstack']['step_size'] == 5.0
    assert zstack.ids['zstack_stepsize_id'].text == '5.0'
    assert zstack.ids['zstack_steps_id'].text == '11'


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


def test_a_number_taken_stays_as_typed_and_records_no_put_back(panel, monkeypatch):
    # The box is redrawn from the store only when the store did not take the
    # entry: a taken '5' redrawn as '5.0' was recorded as a put-back, and so
    # was the range box nobody touched.
    zstack, _settings = panel
    recorded = []
    monkeypatch.setattr(zstack_module.gui_logger, 'text_input', lambda *a: recorded.append(a))
    zstack.set_steps()
    assert zstack.ids['zstack_stepsize_id'].text == '5'
    assert zstack.ids['zstack_range_id'].text == '50'
    assert recorded == []
