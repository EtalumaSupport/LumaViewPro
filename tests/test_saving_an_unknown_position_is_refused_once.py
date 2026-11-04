# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Saving a position the scope does not know is one warning and no store write.

Six gestures save the live position: the X, Y and Z bookmarks, Set All
Bookmarks, Save Focus and Apply Focus to a channel's steps. An axis that
lost its reference keeps answering the last number it reported, so each
reads the position through the motion API's ``positions_to_keep``, which
refuses it in its own words. The refusal raises out of the gesture's call
and the boundary shows it once; nothing is written.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


import ui.layer_control as layer_control
import ui.motion_settings as motion_settings
import ui.vertical_control as vertical_control
from modules.notification_center import Severity
from tests.test_run_refusal_contract import _make_single_step_protocol


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
        lambda: _stand(
            layer_control.LayerControl,
            'save_focus',
            '_refresh_step_views',
            layer='BF',
        ),
        'save_focus',
    ),
    'apply focus': (
        lambda: _stand(
            layer_control.LayerControl,
            'apply_focus_to_channel_steps',
            '_refresh_step_views',
            layer='BF',
        ),
        'apply_focus_to_channel_steps',
    ),
}


@pytest.fixture
def unknown(gui_ctx, monkeypatch, centre_posts):
    """An unhomed simulated session, whose axes do not know their positions,
    and what the person is shown."""
    # The protocol panel, holding a protocol with no step selected.
    panel = SimpleNamespace(
        _protocol=_make_single_step_protocol(), curr_step=-1, update_step_ui=lambda: None
    )
    gui_ctx.motion_settings = SimpleNamespace(ids={'protocol_settings_id': panel})
    gui_ctx.stage = MagicMock()
    for module in (motion_settings, vertical_control, layer_control):
        monkeypatch.setattr(module.gui_logger, 'button', lambda *a, **kw: None)
    return SimpleNamespace(session=gui_ctx.session, posts=centre_posts)


# parametrize-justified: each gesture builds its own GUI fixture and counts its own popups
@pytest.mark.parametrize('gesture', list(GESTURES))
def test_the_refusal_is_one_warning_and_nothing_is_saved(unknown, gesture):
    import copy

    build, press = GESTURES[gesture]
    before = copy.deepcopy(unknown.session.settings)

    getattr(build(), press)()

    assert [(n.title, n.severity) for n in unknown.posts] == [('Scope Not Homed', Severity.WARNING)]
    # An unhomed axis still answers the last number it reported: a gesture
    # that read it past the refusing read would have saved it.
    assert unknown.session.settings == before, 'a refused save writes nothing'
