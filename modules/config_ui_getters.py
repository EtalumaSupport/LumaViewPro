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
from modules.image_mode import ImageCaptureConfig
from modules.zstack_config import ZStackConfig

logger = logging.getLogger('LVP.modules.config_ui_getters')


# ---------------------------------------------------------------------------
# Capability gates
# ---------------------------------------------------------------------------


def _live_capabilities():
    """The capability surface of the LIVE scope, or None if not built yet.

    Reads ``ctx.lumaview.scope`` -- the reference a scope swap rebuilds
    first (the ``ctx.scope`` registry field is a copy, refreshed after
    it). Every capability gate must resolve through here so a swap is
    reflected and the gates can't drift apart.
    """
    lumaview = getattr(_app_ctx.ctx, 'lumaview', None)
    scope = getattr(lumaview, 'scope', None)
    return getattr(scope, 'capabilities', None)


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
    return bool(caps.supports('firmware_stim')) if caps is not None else False


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
        _app_ctx.ctx.engineering_mode
        and _app_ctx.ctx.motion_settings.ids['protocol_settings_id']
        .ids['protocol_disable_image_saving_id']
        .active
    )


# ---------------------------------------------------------------------------
# Binning / Z-stack
# ---------------------------------------------------------------------------


def get_binning_from_ui() -> int:
    try:
        text = (
            _app_ctx.ctx.motion_settings.ids['microscope_settings_id'].ids['binning_spinner'].text
        )
        # Spinner text may be formatted as "1x1", "2x2", etc. -- extract the first number.
        if 'x' in text:
            text = text.split('x')[0]
        return int(text)
    except Exception:
        logger.warning('Failed to read binning from UI, defaulting to 1', exc_info=True)
        from modules.notification_center import notifications

        notifications.warning(
            'Camera',
            'Binning',
            'Could not read the binning setting; using 1x1. Check the binning '
            'selector in microscope settings.',
        )
        return 1


def get_zstack_params() -> dict:
    zstack_settings = _app_ctx.ctx.motion_settings.ids['verticalcontrol_id'].ids['zstack_id']
    range = float(zstack_settings.ids['zstack_range_id'].text)
    step_size = float(zstack_settings.ids['zstack_stepsize_id'].text)
    z_reference = common_utils.convert_zstack_reference_position_setting_to_config(
        text_label=zstack_settings.ids['zstack_spinner'].text
    )

    return {
        'range': range,
        'step_size': step_size,
        'z_reference': z_reference,
    }


def get_zstack_positions() -> tuple[bool, dict]:
    config = get_zstack_params()

    ctx = _app_ctx.ctx
    current_pos = ctx.scope.motion.get_current_position('Z')

    zstack_config = ZStackConfig(
        range=config['range'],
        step_size=config['step_size'],
        current_z_reference=config['z_reference'],
        current_z_value=current_pos,
    )

    if zstack_config.number_of_steps() <= 0:
        return False, {None: None}

    return True, zstack_config.step_positions()


# ---------------------------------------------------------------------------
# Layer / channel configuration
# ---------------------------------------------------------------------------


def get_layer_configs(
    specific_layers: list | None = None,
) -> dict[dict]:
    return config_helpers.get_layer_configs(_app_ctx.ctx.settings, specific_layers)


def get_active_layer_config() -> tuple[str, dict]:
    c_layer = common_utils.get_opened_layer(_app_ctx.ctx.image_settings)

    if c_layer is None:
        raise Exception('No layer currently selected')

    layer_configs = get_layer_configs(specific_layers=[c_layer])

    return c_layer, layer_configs[c_layer]


def get_stim_configs() -> dict:
    return config_helpers.get_stim_configs(_app_ctx.ctx.settings)


def get_enabled_stim_configs() -> dict:
    return config_helpers.get_enabled_stim_configs(_app_ctx.ctx.settings)


# ---------------------------------------------------------------------------
# Position / labware
# ---------------------------------------------------------------------------


def get_current_frame_dimensions() -> dict:
    microscope_settings = _app_ctx.ctx.motion_settings.ids['microscope_settings_id']
    try:
        frame_width = int(microscope_settings.ids['frame_width_id'].text)
        frame_height = int(microscope_settings.ids['frame_height_id'].text)
    except Exception as e:
        raise ValueError('Invalid value for frame width/height') from e

    frame = {'width': frame_width, 'height': frame_height}
    return frame


def get_selected_labware() -> tuple[str | None, labware.WellPlate | None]:
    """The currently-selected labware, read from SETTINGS.

    Settings is the single labware store: the spinner writes through on
    every selection (select_labware persists the choice), so a
    spinner-first read here would only re-open the divergence -- a
    settings write that bypassed the spinner (protocol load) used to
    make the GUI and headless paths answer differently.

    Returns (labware_id, wellplate_obj); the lookup falls back to the
    shipped default or first available plate and only raises
    ConfigError if the wellplate loader is completely empty.
    """
    return config_helpers.get_selected_labware_from_settings(
        _app_ctx.ctx.settings,
        _app_ctx.ctx.wellplate_loader,
    )


# ---------------------------------------------------------------------------
# Image capture / sequenced capture
# ---------------------------------------------------------------------------


def get_image_capture_config_from_ui() -> ImageCaptureConfig:
    """The image capture config for the running GUI.

    Reads the settings store, not the widgets. Every value here is
    committed to settings the moment the user picks it -- each output-format
    spinner handler writes its key, and the image-mode selector writes its
    key alongside the display mirror it drives -- so the store is already
    the current answer and the widgets are a rendering of it. Assembling
    the config from the widgets instead gave a headless caller, which can
    only see the store, a different answer than the screen; the mode also
    reaches saved output through capture_depth, so the drift was reachable
    in the files.
    """
    return config_helpers.get_image_capture_config_from_settings(_app_ctx.ctx.settings)


def get_sequenced_capture_config_from_ui() -> dict:
    objective_id, _ = _app_ctx.ctx.session.get_current_objective_info()
    time_params = get_protocol_time_params()
    labware_id, _ = get_selected_labware()
    protocol_settings = _app_ctx.ctx.motion_settings.ids['protocol_settings_id']
    tiling = protocol_settings.ids['tiling_size_spinner'].text
    tiling_overlap_percent = protocol_settings.get_tiling_overlap_percent()
    use_zstacking = protocol_settings.ids['acquire_zstack_id'].active
    frame_dimensions = get_current_frame_dimensions()
    zstack_params = get_zstack_params()

    layer_configs = get_layer_configs()

    return config_helpers.build_sequenced_capture_config(
        {
            'labware_id': labware_id,
            'objective_id': objective_id,
            'zstack_params': zstack_params,
            'use_zstacking': use_zstacking,
            'tiling': tiling,
            'tiling_overlap_percent': tiling_overlap_percent,
            'layer_configs': layer_configs,
            'period': time_params['period'],
            'duration': time_params['duration'],
            'frame_dimensions': frame_dimensions,
            'binning_size': get_binning_from_ui(),
            'stim_config': get_stim_configs(),
        }
    )


# ---------------------------------------------------------------------------
# Auto gain / objective / protocol time
# ---------------------------------------------------------------------------


def get_auto_gain_settings() -> dict:
    return config_helpers.get_auto_gain_settings(_app_ctx.ctx.settings)


def get_ag_ae_max_exposure_ms(layer: str) -> float:
    return config_helpers.get_ag_ae_max_exposure_ms(
        layer, _app_ctx.ctx.settings.get('ag_ae_max_exposure_ms', {})
    )


def get_ag_ae_min_exposure_ms(layer: str) -> float:
    return config_helpers.get_ag_ae_min_exposure_ms(layer)


def get_protocol_time_params() -> dict:
    """The protocol period and duration for the running GUI.

    Reads the settings store, not the two text fields. Each field commits its
    parsed value to the store when the user leaves it or presses enter, so the
    store already holds the schedule; parsing the widget text a second time
    here only created a way for the two lanes to answer differently.

    It also gave the failure two different shapes. This lane used to swallow
    an unparseable value, substitute one minute or one hour, and say so in a
    popup that no headless caller can see, while the settings lane let a raw
    conversion error escape. Both now surface the one refusal the store lane
    raises, so a REST caller gets the same failure the screen shows.

    The 1-second floor still applies and is still silent, because save and
    run-start both call this and a clamp warning here would repeat; that
    warning fires once, at the field edit.

    Raises:
        ConfigError: a stored period or duration will not parse as a number.
    """
    return config_helpers.get_protocol_time_params_from_settings(_app_ctx.ctx.settings)
