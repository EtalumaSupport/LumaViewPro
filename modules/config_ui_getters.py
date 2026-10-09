# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
UI-dependent configuration getter functions.

These functions read Kivy widget state and return configuration
dicts / tuples. They require a running GUI and cannot be used in
headless or REST API mode.

For GUI-independent equivalents, see config_helpers.py.
"""

import logging

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
import modules.labware as labware

logger = logging.getLogger('LVP.modules.config_ui_getters')


# ---------------------------------------------------------------------------
# Capability gates
# ---------------------------------------------------------------------------


def _live_scope():
    """The LIVE scope object, or None if one has not been built yet.

    ``ctx.lumaview.scope`` is the reference a scope swap rebuilds first; the
    ``ctx.scope`` registry field is a copy, refreshed after it. Every gate that
    resolves off the attached scope goes through here, so a swap is reflected
    everywhere at once and the gates cannot drift apart -- and so this module
    reaches for the app context in exactly one place.
    """
    lumaview = getattr(_app_ctx.ctx, 'lumaview', None)
    return getattr(lumaview, 'scope', None)


def _live_capabilities():
    """The capability surface of the LIVE scope, or None if not built yet."""
    return getattr(_live_scope(), 'capabilities', None)


# ---------------------------------------------------------------------------
# Image scale, off the live scope
# ---------------------------------------------------------------------------


def get_pixel_size(focal_length: float, binning_size: int) -> float | None:
    """Effective um/pixel on the live scope, for the GUI's readouts.

    None when no scope has been built yet or the scope cannot report its
    optics; the readouts then show no scale rather than an invented one. The
    resolver itself takes the capabilities as an argument -- the engine's
    producers hand it the scope they image with -- and this is the one place
    the GUI resolves them off the live scope.
    """
    capabilities = _live_capabilities()
    if capabilities is None:
        return None
    return common_utils.get_pixel_size(
        focal_length=focal_length, binning_size=binning_size, capabilities=capabilities
    )


def get_field_of_view(focal_length: float, frame_size: dict, binning_size: int) -> dict | None:
    """Field of view on the live scope, for the GUI's readouts; None when the
    scope cannot report its scale."""
    capabilities = _live_capabilities()
    if capabilities is None:
        return None
    return common_utils.get_field_of_view(
        focal_length=focal_length,
        frame_size=frame_size,
        binning_size=binning_size,
        capabilities=capabilities,
    )


def firmware_stim_supported() -> bool:
    """True only when the connected LED firmware supports stimulation.

    The single gate for all stimulation UI: when this is False, stim controls
    stay hidden no matter what the user's stimulation_enabled setting says.
    Fails safe to False (hide) when the scope or its capability surface is not
    yet available, so stim never appears on firmware that cannot drive it.
    """
    caps = _live_capabilities()
    return bool(caps.has_firmware_stim) if caps is not None else False


def get_layer_illumination_slider_max(layer: str) -> int | None:
    """The illumination-slider upper bound for ``layer`` from the live scope's
    LED driver, narrowed to the transmitted-layer policy; None before the
    scope is built (the .kv placeholder stands until then).
    """
    caps = _live_capabilities()
    if caps is None:
        return None
    return config_helpers.layer_max_illumination_ma_for_ui(caps, layer)


def get_layer_exposure_slider_max(camera_max_ms: float, layer: str) -> float:
    """The exposure-slider upper bound for ``layer``: the connected camera's
    cap, narrowed to the manual transmitted policy.

    Takes the cap rather than reading it off the app context -- the caller
    already holds the value the settings load resolved through
    camera_max_exposure_for_ui, and a module reaching up for it is the
    direction this layer is not allowed to depend in.
    """
    return config_helpers.layer_max_exposure_ms_for_ui(camera_max_ms, layer)


def camera_autogain_supported() -> bool:
    """True when the connected camera has hardware auto-gain or auto-exposure.

    The single gate for the "Auto Gain/Exp" control, which drives BOTH
    auto-gain and auto-exposure -- so it stays visible if the hardware offers
    either, and hides only when the camera offers neither (IDS U3-34Lx, FX2
    LS620). Fails safe to True (show) when no capability surface exists yet, so
    a not-yet-built scope keeps the prior always-shown behavior rather than
    hiding on unknown.
    """
    caps = _live_capabilities()
    if caps is None:
        return True
    return bool(caps.camera_supports_auto_gain or caps.camera_supports_auto_exposure)


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------


def is_image_saving_enabled() -> bool:
    return not (
        _app_ctx.ctx.session.engineering_mode
        and _app_ctx.ctx.motion_settings.ids['protocol_settings_id']
        .ids['protocol_disable_image_saving_id']
        .active
    )


# ---------------------------------------------------------------------------
# Binning / Z-stack
# ---------------------------------------------------------------------------


def get_zstack_params() -> dict:
    """The z-stack range, step size and reference for the running GUI.

    Reads the settings store, not the three widgets. Each widget commits to the
    store as it changes -- the two TextInputs through the z-stack step handler,
    the spinner through its position handler -- so the store already holds the
    stack; parsing the widget text a second time here only created a way for
    the two lanes to answer differently.

    Raises:
        ConfigError: the stored stack will not parse -- an unmapped position
            label, or a range / step size that is not a number.
    """
    return config_helpers.get_zstack_params_from_settings(_app_ctx.ctx.settings)


# ---------------------------------------------------------------------------
# Layer / channel configuration
# ---------------------------------------------------------------------------


def get_layer_configs(
    specific_layers: list | None = None,
) -> dict[dict]:
    return config_helpers.get_layer_configs(_app_ctx.ctx.settings, specific_layers)


def get_active_layer_config(layer: str | None) -> tuple[str, dict]:
    """The capture config for one named layer.

    Takes the layer rather than reading which accordion drawer is open:
    an open drawer is a fact about the running GUI and means nothing to a
    caller that has none, so the GUI names its layer and every other
    caller names its own.

    The refusal stays here rather than moving into the three GUI callers:
    "nothing is selected" is one answer to one question, and answering it
    per-caller is how three of them come to disagree.
    """
    if layer is None:
        raise Exception('No layer currently selected')

    layer_configs = get_layer_configs(specific_layers=[layer])

    return layer, layer_configs[layer]


# ---------------------------------------------------------------------------
# Position / labware
# ---------------------------------------------------------------------------


def get_selected_labware() -> tuple[str, labware.WellPlate]:
    """The currently-selected labware, read from SETTINGS.

    Settings is the single labware store: the spinner writes through on
    every selection (select_labware persists the choice), so a
    spinner-first read here would only re-open the divergence -- a
    settings write that bypassed the spinner (protocol load) used to
    make the GUI and headless paths answer differently.

    Returns (labware_id, wellplate_obj). A stored plate the catalogue
    does not have raises ConfigError; no other plate is substituted.
    """
    return config_helpers.get_selected_labware_from_settings(
        _app_ctx.ctx.settings,
        _app_ctx.ctx.wellplate_loader,
    )
