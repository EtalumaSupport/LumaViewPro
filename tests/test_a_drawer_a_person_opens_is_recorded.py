# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A drawer a person opens is recorded once and asks the Session to show it;
one the app opens is neither.

A clicked layer drawer wrote nothing to gui_interactions.log, while the
app's own expand (start-up, a step's layer) was recorded as a person's pick.
Kivy expands a collapsed item from its touch handler alone, so the record is
written there, by LoggedAccordionItem, and a layer drawer's
``person_opened`` asks the Session to show its layer from the same place.
The app's own expand draws the channel the Session shows, and asks nothing.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modules.scope_session import ShownChannel
from ui.ui_helpers import LoggedAccordionItem
from tests.stock_channels import LED0


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


def test_opening_a_collapsed_drawer_is_recorded_once_and_asks_once(kivy_expand):
    asked = []
    item = _item()
    item.person_opened = lambda: asked.append('PC')
    with patch('ui.ui_helpers.gui_logger.select') as select:
        item.on_touch_down(_touch())
    select.assert_called_once_with('IMAGE_LAYER', 'PC')
    assert asked == ['PC']
    assert len(kivy_expand) == 1


@pytest.mark.parametrize(
    'state',
    [{'collapse': False}, {'disabled': True}, {'inside': False}],
    ids=['already open', 'disabled', 'touch elsewhere'],
)
def test_a_touch_that_opens_nothing_records_nothing_and_asks_nothing(kivy_expand, state):
    asked = []
    item = _item(**state)
    item.person_opened = lambda: asked.append('PC')
    with patch('ui.ui_helpers.gui_logger.select') as select:
        item.on_touch_down(_touch())
    select.assert_not_called()
    assert asked == []
    assert len(kivy_expand) == 1


def test_the_apps_own_draw_is_not_recorded_as_a_pick_and_asks_nothing(monkeypatch, caplog):
    """The app drawing the shown channel's drawer (start-up, a step's layer,
    a run's step) writes no pick and asks the Session nothing: no person
    picked it. The drawers are the real LoggedAccordionItem, whose record and
    ask are made only from a touch."""
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
    asked = []
    for layer in layers:
        drawers[layer].person_opened = lambda layer=layer: asked.append(layer)
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
    panel = SimpleNamespace(
        accordion_item_lookup=lambda layer: drawers[layer],
        layer_lookup=lambda layer: SimpleNamespace(walk=lambda: [], show_as_shown=lambda: None),
    )
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(session=SimpleNamespace(shown_channel=ShownChannel(target, lit=False))),
    )

    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        ImageSettings.draw_shown_channel(panel)

    assert [layer for layer in layers if not drawers[layer].collapse] == [target], (
        'the expand did not run, so it recorded nothing for the wrong reason'
    )
    recorded = [r.getMessage() for r in caplog.records if r.name == 'LVP.gui_interactions']
    picks = [message for message in recorded if 'OPEN_LAYER' not in message]
    assert picks == [], f'the app drawing a drawer was recorded as a pick: {picks}'
    assert asked == []


def test_a_persons_open_asks_the_session_and_redraws_from_its_answer(monkeypatch):
    """A person's open of a layer's drawer asks the Session to show that
    layer, on the worker pool since show_channel spans the IO and camera
    lanes; whatever the answer, the drawers are redrawn from the store, so a
    refused show puts the drawer back."""
    import modules.app_context as _app_ctx
    import ui.ui_helpers as ui_helpers
    from ui.image_settings import ImageSettings

    asked = []
    submits = []
    session = SimpleNamespace(show_channel=asked.append)
    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(session=session))
    monkeypatch.setattr(
        ui_helpers,
        'submit_reported',
        lambda call, redraw, label, **kwargs: submits.append((redraw, label, kwargs)) or call(),
    )
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the ask
    panel = SimpleNamespace(draw_shown_channel=lambda: None)

    ImageSettings._show_channel_for_person(panel, LED0)

    assert asked == [LED0]
    assert submits == [(panel.draw_shown_channel, f'SHOW_CHANNEL_{LED0}', {})]
