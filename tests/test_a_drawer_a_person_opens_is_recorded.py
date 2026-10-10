# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A drawer a person opens is recorded once; one the app opens is not.

A clicked layer drawer wrote nothing to gui_interactions.log, although it
turns LEDs off and applies a layer's settings to the camera, while the app's
own expand (start-up, a step's layer) was recorded as a person's pick. Kivy
expands a collapsed item from its touch handler alone, so the record is
written there, by LoggedAccordionItem.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ui.ui_helpers import LoggedAccordionItem


@pytest.fixture
def kivy_expand(monkeypatch):
    # The suite's AccordionItem stand-in has no touch handler; this is Kivy's.
    calls = []
    base = LoggedAccordionItem.__mro__[1]
    monkeypatch.setattr(
        base, 'on_touch_down', lambda self, touch: calls.append(touch) or True, raising=False
    )
    return calls


def _item(*, collapse=True, disabled=False, inside=True):
    item = LoggedAccordionItem()
    item.log_group = 'IMAGE_LAYER'
    item.log_item = 'PC'
    item.collapse = collapse
    item.disabled = disabled
    item.collide_point = lambda *pos: inside
    return item


def _touch():
    return SimpleNamespace(pos=(1, 1))


def test_opening_a_collapsed_drawer_is_recorded_once(kivy_expand):
    with patch('ui.ui_helpers.gui_logger.select') as select:
        _item().on_touch_down(_touch())
    select.assert_called_once_with('IMAGE_LAYER', 'PC')
    assert len(kivy_expand) == 1


@pytest.mark.parametrize(
    'state',
    [{'collapse': False}, {'disabled': True}, {'inside': False}],
    ids=['already open', 'disabled', 'touch elsewhere'],
)
def test_a_touch_that_opens_nothing_records_nothing(kivy_expand, state):
    with patch('ui.ui_helpers.gui_logger.select') as select:
        _item(**state).on_touch_down(_touch())
    select.assert_not_called()
    assert len(kivy_expand) == 1


def test_the_apps_own_expand_is_not_recorded_as_a_pick(monkeypatch, caplog):
    """The app opening a layer's drawer (start-up, a step's layer) writes no
    record: no person picked it. The drawers are the real LoggedAccordionItem,
    whose record is written only from a touch."""
    import modules.app_context as _app_ctx
    import modules.common_utils as common_utils
    from ui.image_settings import ImageSettings

    layers = common_utils.get_layers()
    drawers = {}
    for layer in layers:
        drawers[layer] = _item()
        drawers[layer].log_item = layer
    opened, target = layers[0], layers[1]
    drawers[opened].collapse = False
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
    panel = SimpleNamespace(
        accordion_item_lookup=lambda layer: drawers[layer],
        layer_lookup=lambda layer: SimpleNamespace(walk=lambda: []),
    )
    monkeypatch.setattr(
        _app_ctx, 'ctx', SimpleNamespace(session=SimpleNamespace(run_lockout=False))
    )

    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        ImageSettings.set_expanded_layer(panel, target)

    assert [layer for layer in layers if not drawers[layer].collapse] == [target], (
        'the expand did not run, so it recorded nothing for the wrong reason'
    )
    recorded = [r.getMessage() for r in caplog.records if r.name == 'LVP.gui_interactions']
    assert recorded == [], f'the app expanding a drawer was recorded as a pick: {recorded}'
