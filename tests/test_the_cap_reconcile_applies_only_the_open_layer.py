# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The capped-layer reconcile re-renders every capped layer and applies only the open one.

The camera holds one layer. The reconcile applied each layer whose stored
gain or exposure the attached camera cannot reach, one after another, so the
camera ended on the last of them in catalogue order -- not the layer on
screen, and not the BF that bring-up had just put on it. A closed layer
reaches the camera, capped, when it is next applied.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import modules.app_context as _app_ctx
from modules.lumascope_api.imaging import AppliedCameraSetting

_CAP_DB = 24.0


def _applied_gain(stored):
    return AppliedCameraSetting(
        stored=stored, applied=min(stored, _CAP_DB), capped=stored > _CAP_DB
    )


def _applied_exposure(stored):
    return AppliedCameraSetting(stored=stored, applied=stored, capped=False)


def _reconcile(monkeypatch, *, opened, capped):
    import ui.image_settings as image_settings

    settings = {
        layer: {'gain_db': 48.0 if layer in capped else 1.0, 'exposure_ms': 10.0}
        for layer in ('BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi')
    }
    context = MagicMock()
    context.settings = settings
    imaging = context.lumaview.scope.imaging
    imaging.applied_gain_db_for.side_effect = _applied_gain
    imaging.applied_exposure_ms_for.side_effect = _applied_exposure
    monkeypatch.setattr(_app_ctx, 'ctx', context)

    layer_objs = {layer: MagicMock(name=layer) for layer in settings}
    stand_in = SimpleNamespace(
        layer_lookup=lambda layer: layer_objs[layer],
        accordion_item_lookup=lambda layer: SimpleNamespace(collapse=layer != opened),
    )
    image_settings.ImageSettings.reconcile_layers_to_camera_caps(stand_in)
    return layer_objs


def test_every_capped_layer_is_rendered_and_only_the_open_one_applied(monkeypatch):
    layer_objs = _reconcile(monkeypatch, opened='Blue', capped={'Blue', 'Red'})

    for layer in ('Blue', 'Red'):
        layer_objs[layer].render_layer_values_from_settings.assert_called_once_with()
    layer_objs['Blue'].apply_settings.assert_called_once_with()
    layer_objs['Red'].apply_settings.assert_not_called()


def test_with_no_capped_layer_open_nothing_is_applied(monkeypatch):
    layer_objs = _reconcile(monkeypatch, opened='BF', capped={'Green', 'Red'})

    for layer_obj in layer_objs.values():
        layer_obj.apply_settings.assert_not_called()
    layer_objs['Green'].render_layer_values_from_settings.assert_called_once_with()
    layer_objs['Red'].render_layer_values_from_settings.assert_called_once_with()
