# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol schedule the store cannot use is the store's refusal, shown once.

The period and duration fields hand the protocol the schedule the settings
store holds. A hand-edited settings file can hold a value that will not
parse, and a typed ``nan`` or ``inf`` is a float the field accepts and the
schedule cannot take. Either way the store's answer is an exception, and
the panel hands it to the one reporter under the field's label instead of
picking a level and a title for it -- or, for the values the old handler
never caught, letting it out of a Kivy focus handler and closing the app.

These tests drive the real handlers with the widget tree stubbed.
"""

import datetime
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
PERIOD = datetime.timedelta(minutes=5)
DURATION = datetime.timedelta(hours=2)


class _Panel(ps.ProtocolSettings):
    """The real class, with only the widget tree stubbed."""

    def __init__(self, protocol, period_text, duration_text):
        self.ids = {
            'capture_period': SimpleNamespace(text=period_text),
            'capture_dur': SimpleNamespace(text=duration_text),
        }
        self._protocol = protocol


@pytest.fixture
def env(monkeypatch):
    reported = []
    monkeypatch.setattr(
        notifications,
        'report_outcome',
        lambda exc, **kw: reported.append((type(exc).__name__, kw['category'])),
    )
    # The field's own clamp and not-a-number warnings are the ranges owner's
    # and not under test; they are kept from reaching the bus.
    warned = []
    monkeypatch.setattr(notifications, 'warning', lambda *a, **kw: warned.append(a))
    monkeypatch.setattr(ps.gui_logger, 'text_input', lambda *a, **kw: None)

    protocol = Protocol(
        tiling_configs_file_loc=REPO / 'data' / 'tiling.json',
        config={
            'steps': pd.DataFrame(),
            'custom_step_count': 0,
            'period': PERIOD,
            'duration': DURATION,
        },
    )

    def panel(*, store, period_text='', duration_text=''):
        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(settings={'protocol': store}))
        return _Panel(protocol, period_text, duration_text)

    return SimpleNamespace(panel=panel, reported=reported, warned=warned, protocol=protocol)


def test_a_stored_value_that_will_not_parse_is_reported_once_under_the_field(env):
    panel = env.panel(store={'period': 1.0, 'duration': 'two'}, period_text='3')

    panel.update_period()

    assert env.reported == [('ConfigError', 'UI:PROTOCOL_PERIOD')]
    assert (env.protocol.period(), env.protocol.duration()) == (PERIOD, DURATION)


def test_a_typed_nan_is_reported_not_raised_out_of_the_handler(env):
    panel = env.panel(store={'period': 1.0, 'duration': 1.0}, duration_text='nan')

    panel.update_duration()

    assert env.reported == [('ValueError', 'UI:PROTOCOL_DURATION')]
    assert (env.protocol.period(), env.protocol.duration()) == (PERIOD, DURATION)


def test_a_usable_schedule_reaches_the_protocol_with_nothing_reported(env):
    panel = env.panel(store={'period': 1.0, 'duration': 1.0}, period_text='10')

    panel.update_period()

    assert env.reported == []
    assert env.protocol.period() == datetime.timedelta(minutes=10)
    assert env.protocol.duration() == datetime.timedelta(hours=1)
