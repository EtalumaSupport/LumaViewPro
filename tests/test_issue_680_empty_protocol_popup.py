# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#680: a config whose every layer is disabled builds an empty protocol.

Protocol.from_config skips any layer that does not acquire, so with every
layer skipped the Protocol is legally constructed and empty. The click
that meant steps is refused before the build, by the protocols API's
no_acquiring_layer refusal through ScopeSession.new_protocol
(tests/test_creating_a_protocol_is_an_api_capability.py); this file pins
the builder's half, the precondition that refusal exists for.
"""

from __future__ import annotations

import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'


def _from_config_input(acquire_by_layer: dict[str, str]) -> dict:
    """Minimal valid input_config for Protocol.from_config with one layer
    per entry, each with the given acquire mode."""
    layer_configs = {
        layer: {
            'acquire': acquire,
            'autofocus': False,
            'false_color': False,
            'illumination_ma': 50.0,
            'gain_db': 1.0,
            'auto_gain': False,
            'exposure_ms': 10.0,
            'sum': 1,
            'focus': 100.0,
            'video_config': {'duration': 30},
            'stim_config': None,
        }
        for layer, acquire in acquire_by_layer.items()
    }
    return {
        'labware_id': '96 well microplate',
        'objective_id': '4x Oly',
        'zstack_params': {'range': 0, 'step_size': 0, 'z_reference': 'center'},
        'use_zstacking': False,
        'tiling': '1x1',
        'layer_configs': layer_configs,
        'period': None,
        'duration': None,
        'frame_dimensions': {'width': 2048, 'height': 2048},
        'binning_size': 1,
        'stim_config': {},
    }


def test_protocol_from_config_filters_non_acquire_layers(scale_capabilities):
    """The upstream filter that produces 0 steps: layers whose acquire is
    neither 'image' nor 'video' contribute no steps, so a config where
    every layer is disabled yields an empty (0-step) Protocol -- the
    precondition the #680 UI guard catches. A layer set to 'image' still
    produces steps."""
    from modules.labware_loader import WellPlateLoader
    from modules.objectives_loader import ObjectiveLoader
    from modules.protocol import Protocol

    tiling_configs = REPO / 'data' / 'tiling.json'

    all_disabled = Protocol.from_config(
        input_config=_from_config_input({'BF': 'none', 'Blue': 'none'}),
        tiling_configs_file_loc=tiling_configs,
        capabilities=scale_capabilities,
        objective_helper=ObjectiveLoader(),
        wellplate_loader=WellPlateLoader(),
    )
    assert all_disabled.num_steps() == 0, (
        'every-layer-disabled must construct an EMPTY protocol (the #680 '
        f'guard precondition); got {all_disabled.num_steps()} steps'
    )

    one_enabled = Protocol.from_config(
        input_config=_from_config_input({'BF': 'image', 'Blue': 'none'}),
        tiling_configs_file_loc=tiling_configs,
        capabilities=scale_capabilities,
        objective_helper=ObjectiveLoader(),
        wellplate_loader=WellPlateLoader(),
    )
    assert one_enabled.num_steps() > 0, 'an image layer must still produce steps'
    step_colors = set(one_enabled.steps()['Color'].unique())
    assert step_colors == {'BF'}, (
        f'only the acquiring layer may contribute steps; got {step_colors}'
    )
