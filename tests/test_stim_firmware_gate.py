# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Stimulation controls are shown only where the scope's LED firmware can stimulate.

The GUI asks the scope: ``capabilities.has_firmware_stim``. The channel
drawers' stimulation support follows that answer and the user's
``stimulation_enabled`` setting; firmware that cannot stimulate never shows
the controls, whatever the setting says.

No simulated scope model announces stimulation yet (the bring-up knob waits
on the firmware stimulation work), so the capable case builds the
simulated LED board with its own ``supports_firmware_stim`` option and the
scope's capabilities from it.
"""

from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from drivers.simulated_ledboard import SimulatedLEDBoard
from modules.scope_capabilities import ScopeCapabilities
from tests.scope_fakes import build_scope
from ui.microscope_settings import MicroscopeSettings


@pytest.fixture
def scope():
    scope = build_scope(simulate=True, register_atexit=False)
    yield scope
    scope.disconnect()


def _able_to_stimulate(scope):
    """The scope's capabilities, read from a simulated LED board whose firmware can stimulate."""
    scope.capabilities = ScopeCapabilities.from_drivers(
        motion=scope._motion_driver,
        led=SimulatedLEDBoard(supports_firmware_stim=True),
        camera=scope._camera_driver,
        channel_identity=scope.channel_identity,
        scope_models=scope._scope_models,
        timing_record=None,
    )


def _drawers_after_apply(scope, monkeypatch):
    """Each fluorescence drawer's stimulation support after the panel applies it."""
    drawers = {
        layer: SimpleNamespace(
            stimulation_support=None, show_stim_controls=None, show_camera_controls=None, ids={}
        )
        for layer in common_utils.get_fluorescence_layers()
    }
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            scope=scope,
            settings={'stimulation_enabled': True},
            image_settings=SimpleNamespace(layer_lookup=lambda layer: drawers.get(layer)),
        ),
    )
    MicroscopeSettings.__new__(MicroscopeSettings).apply_stimulation_support()
    return {layer: d.stimulation_support for layer, d in drawers.items()}


def test_the_controls_are_supported_where_the_firmware_can_stimulate(scope, monkeypatch):
    _able_to_stimulate(scope)

    supported = _drawers_after_apply(scope, monkeypatch)

    assert supported and set(supported.values()) == {True}


def test_the_controls_are_hidden_where_the_firmware_cannot(scope, monkeypatch):
    assert scope.capabilities.has_firmware_stim is False

    supported = _drawers_after_apply(scope, monkeypatch)

    assert supported and set(supported.values()) == {False}
