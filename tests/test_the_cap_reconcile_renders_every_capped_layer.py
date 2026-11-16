# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The capped-layer reconcile re-renders every capped layer and applies none.

The camera holds one layer. The reconcile once applied each layer whose
stored gain or exposure the attached camera cannot reach, one after another,
so the camera ended on the last of them in catalogue order -- not the layer
on screen, and not the BF that bring-up had just put on it; then it applied
the shown one, a second call for what bring-up had done. The Session caps
each value as it puts a layer on the camera, so the reconcile only redraws;
another layer reaches the camera, capped, when it is next shown.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import _lumascope.modules.app_context as _app_ctx
from _lumascope.modules.channel_record import release_catalogue
from _lumascope.modules.scope_session import ShownChannel
from tests.stock_channels import LED0, LED1, LED2


def _reconcile(monkeypatch, *, shown, capped):
    import ui.image_settings as image_settings

    # a stand-in by design: the subject is the GUI's reconcile, which Kivy
    # keeps out of the test process; the Session's answer of which channels
    # are capped (capped_channels) is pinned at the API, in
    # test_a_capped_layer_setting_is_told_once_at_bring_up.
    context = MagicMock()
    context.session.shown_channel = ShownChannel(shown, lit=False)
    context.session.capped_channels.return_value = tuple(
        layer for layer in release_catalogue() if layer in capped
    )
    monkeypatch.setattr(_app_ctx, 'ctx', context)

    channel_controls = {layer: MagicMock(name=layer) for layer in release_catalogue()}
    stand_in = SimpleNamespace(channel_control_lookup=lambda channel: channel_controls[channel])
    image_settings.ImageSettings.reconcile_channels_to_camera_caps(stand_in)
    return channel_controls


def test_every_capped_layer_is_rendered_and_none_applied(monkeypatch):
    channel_controls = _reconcile(monkeypatch, shown=LED0, capped={LED0, LED2})

    for layer in (LED0, LED2):
        channel_controls[layer].render_channel_values_from_settings.assert_called_once_with()
    channel_controls[LED1].render_channel_values_from_settings.assert_not_called()
    for channel_control in channel_controls.values():
        channel_control.apply_settings.assert_not_called()


def test_with_no_capped_layer_shown_nothing_is_applied(monkeypatch):
    channel_controls = _reconcile(monkeypatch, shown='BF', capped={LED1, LED2})

    for channel_control in channel_controls.values():
        channel_control.apply_settings.assert_not_called()
    channel_controls[LED1].render_channel_values_from_settings.assert_called_once_with()
    channel_controls[LED2].render_channel_values_from_settings.assert_called_once_with()
