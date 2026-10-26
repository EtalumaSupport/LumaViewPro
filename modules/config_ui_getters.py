# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Slider bounds the GUI asks for, handed the live state it holds.

The GUI asks the scope its capability and labware questions directly
(``ctx.scope.capabilities``, ``ctx.scope.runtime_state``); what remains
here takes its inputs as arguments and reaches for nothing.

For GUI-independent equivalents, see config_helpers.py.
"""

import logging

import modules.config_helpers as config_helpers

logger = logging.getLogger('LVP.modules.config_ui_getters')


def get_layer_exposure_slider_max(camera_max_ms: float, layer: str) -> float:
    """The exposure-slider upper bound for ``layer``: the connected camera's
    cap, narrowed to the layer's manual ceiling where it has one.

    Takes the cap rather than reading it off the app context -- the caller
    already holds the value the settings load resolved through
    camera_max_exposure_for_ui, and a module reaching up for it is the
    direction this layer is not allowed to depend in.
    """
    return config_helpers.layer_max_exposure_ms_for_ui(camera_max_ms, layer)
