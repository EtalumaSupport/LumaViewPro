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

import modules.app_context as _app_ctx
from modules.layer_record import release_catalogue
from modules.lumascope_api.camera import AppliedExposure, AppliedGain, cap_stored_value
from modules.scope_session import ShownChannel

_CAP_DB = 24.0


def _applied_gain(stored):
    # The API's own cap rule over a camera whose gain ceiling is _CAP_DB.
    stored_db, applied_db = cap_stored_value(stored, _CAP_DB)
    return AppliedGain(stored_db=stored_db, applied_db=applied_db, capped=applied_db != stored_db)


def _applied_exposure(stored):
    # No exposure ceiling: nothing is capped.
    stored_ms, applied_ms = cap_stored_value(stored, None)
    return AppliedExposure(stored_ms=stored_ms, applied_ms=applied_ms, capped=False)


def _reconcile(monkeypatch, *, shown, capped):
    import ui.image_settings as image_settings

    settings = {
        layer: {'gain_db': 48.0 if layer in capped else 1.0, 'exposure_ms': 10.0}
        for layer in release_catalogue()
    }
    context = MagicMock()
    context.settings = settings
    context.session.shown_channel = ShownChannel(shown, lit=False)
    imaging = context.lumaview.scope.imaging
    imaging.applied_gain_db_for.side_effect = _applied_gain
    imaging.applied_exposure_ms_for.side_effect = _applied_exposure
    monkeypatch.setattr(_app_ctx, 'ctx', context)

    layer_objs = {layer: MagicMock(name=layer) for layer in settings}
    stand_in = SimpleNamespace(layer_lookup=lambda layer: layer_objs[layer])
    image_settings.ImageSettings.reconcile_layers_to_camera_caps(stand_in)
    return layer_objs


def test_every_capped_layer_is_rendered_and_none_applied(monkeypatch):
    layer_objs = _reconcile(monkeypatch, shown='Blue', capped={'Blue', 'Red'})

    for layer in ('Blue', 'Red'):
        layer_objs[layer].render_layer_values_from_settings.assert_called_once_with()
    for layer_obj in layer_objs.values():
        layer_obj.apply_settings.assert_not_called()


def test_with_no_capped_layer_shown_nothing_is_applied(monkeypatch):
    layer_objs = _reconcile(monkeypatch, shown='BF', capped={'Green', 'Red'})

    for layer_obj in layer_objs.values():
        layer_obj.apply_settings.assert_not_called()
    layer_objs['Green'].render_layer_values_from_settings.assert_called_once_with()
    layer_objs['Red'].render_layer_values_from_settings.assert_called_once_with()
