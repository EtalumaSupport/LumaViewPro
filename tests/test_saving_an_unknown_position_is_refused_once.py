# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Saving a position the scope does not know is one warning and no store write.

Six gestures save the live position: the X, Y and Z bookmarks, Set All
Bookmarks, Save Focus and Apply Focus to a channel's steps. An axis that
lost its reference keeps answering the last number it reported, so each
asks the motion API first, and the API refuses in its own words. The
refusal raises out of the gesture's call and the boundary shows it once;
nothing is read and nothing is written.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in ('kivy.clock', 'kivy.uix', 'kivy.metrics', 'kivy.properties'):
    sys.modules.setdefault(_name, MagicMock())

_boxlayout = types.ModuleType('kivy.uix.boxlayout')
_boxlayout.BoxLayout = _StubWidget
sys.modules.setdefault('kivy.uix.boxlayout', _boxlayout)

import modules.app_context as _app_ctx
import ui.layer_control as layer_control
import ui.motion_settings as motion_settings
import ui.vertical_control as vertical_control
from modules.exceptions import AxisStateUnknownError
from modules.notification_center import Severity


def _stand(cls, *names, **attrs):
    """The real handlers of *cls*, bound to a stand with no widget tree."""
    stand = SimpleNamespace(**attrs)
    for name in names:
        setattr(stand, name, getattr(cls, name).__get__(stand))
    return stand


GESTURES = {
    'x bookmark': (
        lambda: _stand(motion_settings.XYStageControl, 'set_xbookmark', 'ex_set_xbookmark'),
        'set_xbookmark',
    ),
    'y bookmark': (
        lambda: _stand(motion_settings.XYStageControl, 'set_ybookmark', 'ex_set_ybookmark'),
        'set_ybookmark',
    ),
    'z bookmark': (
        lambda: _stand(vertical_control.VerticalControl, 'set_bookmark', 'ex_set_bookmark'),
        'set_bookmark',
    ),
    'all bookmarks': (
        lambda: _stand(
            vertical_control.VerticalControl, 'set_all_bookmarks', 'ex_set_all_bookmarks'
        ),
        'set_all_bookmarks',
    ),
    'save focus': (
        lambda: _stand(layer_control.LayerControl, 'save_focus', 'execute_save_focus', layer='BF'),
        'save_focus',
    ),
    'apply focus': (
        lambda: _stand(
            layer_control.LayerControl,
            'apply_focus_to_channel_steps',
            'execute_apply_focus_to_channel_steps',
            layer='BF',
        ),
        'apply_focus_to_channel_steps',
    ),
}


@pytest.fixture
def unknown(monkeypatch):
    """A scope whose axes do not know their positions, and what the person is shown."""
    from tests.scope_fakes import spec_scope
    from tests.shown_outcomes import capture_shown

    def _refuse(axes, *, recording, then):
        raise AxisStateUnknownError(dict.fromkeys(axes, 'unknown'), then=then)

    scope = spec_scope()
    scope.motion.refuse_unknown_positions.side_effect = _refuse
    settings = {
        'bookmark': {'x': 1.0, 'y': 2.0, 'z': 3.0},
        **{
            layer: {'focus': 7000.0} for layer in ('BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi')
        },
    }
    ctx = SimpleNamespace(
        scope=scope,
        lumaview=SimpleNamespace(scope=scope),
        settings=settings,
        settings_lock=MagicMock(),
        motion_settings=MagicMock(),
        protocol=None,
    )
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    for module in (motion_settings, vertical_control, layer_control):
        monkeypatch.setattr(module.gui_logger, 'button', lambda *a, **kw: None)
    return SimpleNamespace(scope=scope, settings=settings, shown=capture_shown(monkeypatch))


@pytest.mark.parametrize('gesture', list(GESTURES))
def test_the_refusal_is_one_warning_and_nothing_is_saved(unknown, gesture):
    import copy

    build, press = GESTURES[gesture]
    before = copy.deepcopy(unknown.settings)

    getattr(build(), press)()

    assert [(n.title, n.severity) for n in unknown.shown] == [('Scope Not Homed', Severity.WARNING)]
    assert unknown.settings == before, 'a refused save writes nothing'
    assert not unknown.scope.motion.get_current_position.called, (
        'the position is not even read once the API refused'
    )
    kwargs = unknown.scope.motion.refuse_unknown_positions.call_args.kwargs
    assert kwargs['recording'] is True
