# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
UI-dependent configuration getter functions.

These functions read Kivy widget state and return configuration
dicts / tuples. They require a running GUI and cannot be used in
headless or REST API mode.

For GUI-independent equivalents, see config_helpers.py.
"""

import datetime
import logging
import pathlib

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
import modules.image_mode as image_mode
import modules.labware as labware
from modules.stack_builder import StackBuilder
from modules.tiling_config import TilingConfig
from modules.zstack_config import ZStackConfig

logger = logging.getLogger('LVP.modules.config_ui_getters')


# ---------------------------------------------------------------------------
# Capability gates
# ---------------------------------------------------------------------------


def firmware_stim_supported() -> bool:
    """True only when the connected LED firmware supports stimulation.

    The single gate for all stimulation UI: when this is False, stim controls
    stay hidden no matter what the user's stimulation_enabled setting says.
    Fails safe to False (hide) when the scope or its capability surface is not
    yet available, so stim never appears on firmware that cannot drive it.
    """
    scope = getattr(_app_ctx.ctx, 'scope', None)
    caps = getattr(scope, 'capabilities', None)
    return bool(caps.supports('firmware_stim')) if caps is not None else False


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
    c_layer = None
    for layer in common_utils.get_layers():
        accordion_item_obj = _app_ctx.ctx.image_settings.accordion_item_lookup(layer=layer)
        if not accordion_item_obj.collapse:
            c_layer = layer
            break

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


def get_current_plate_position():
    ctx = _app_ctx.ctx
    return config_helpers.get_current_plate_position(
        scope=ctx.scope,
        settings=ctx.settings,
        coordinate_transformer=ctx.coordinate_transformer,
        wellplate_loader=ctx.wellplate_loader,
    )


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
    """Read the currently-selected labware from the spinner UI.

    Falls back to settings['protocol']['labware'] if the spinner text is empty
    (e.g. before the spinner has been populated from settings on startup).

    Returns (labware_id, wellplate_obj). On UI/spinner read failure
    returns (None, None); the labware lookup itself never returns None
    (the headless helper falls back to the shipped default or first
    available plate, and only raises ConfigError if the wellplate
    loader is completely empty).
    """
    try:
        protocol_settings = _app_ctx.ctx.motion_settings.ids['protocol_settings_id']
        labware_id = protocol_settings.ids['labware_spinner'].text
        if not labware_id:
            labware_id = _app_ctx.ctx.settings.get('protocol', {}).get('labware', '')
    except Exception:
        logger.exception('LVP Main: Failed to read labware id from UI/settings')
        return None, None

    return config_helpers.get_selected_labware_from_settings(
        {'protocol': {'labware': labware_id}},
        _app_ctx.ctx.wellplate_loader,
    )


# ---------------------------------------------------------------------------
# Image capture / sequenced capture
# ---------------------------------------------------------------------------


def get_image_capture_config_from_ui() -> dict:
    microscope_settings = _app_ctx.ctx.motion_settings.ids['microscope_settings_id']
    output_format = {
        'live': microscope_settings.ids['live_image_output_format_spinner'].text,
        'sequenced': microscope_settings.ids['sequenced_image_output_format_spinner'].text,
    }
    mode = _app_ctx.ctx.scope_display.image_mode
    derived = image_mode.resolve_image_mode(mode)
    return {
        'output_format': output_format,
        'image_mode': mode,
        'capture_depth': derived['capture_depth'],
        'save_encoding': derived['save_encoding'],
        # Legacy keys derived from the resolved mode for any straggler reader
        # during the transition; dropped once every caller consumes image_mode.
        'use_full_pixel_depth': derived['capture_depth'] == 12,
        'false_color_16bit': derived['save_encoding'] == image_mode.SAVE_ENCODING_RGB,
        'jpg_quality': int(_app_ctx.ctx.settings.get('jpg_quality', 90)),
    }


def get_sequenced_capture_config_from_ui() -> dict:
    objective_id, _ = get_current_objective_info()
    time_params = get_protocol_time_params()
    labware_id, _ = get_selected_labware()
    protocol_settings = _app_ctx.ctx.motion_settings.ids['protocol_settings_id']
    tiling = protocol_settings.ids['tiling_size_spinner'].text
    tiling_overlap_percent = TilingConfig.validate_overlap_percent(
        protocol_settings.ids['tiling_overlap_spinner'].text.strip().rstrip('%')
    )
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
    return config_helpers.get_ag_ae_max_exposure_ms(layer, _app_ctx.ctx.settings)


def get_current_objective_info() -> tuple[str, dict]:
    return config_helpers.get_current_objective_info(
        _app_ctx.ctx.settings, _app_ctx.ctx.objective_helper
    )


def get_protocol_time_params() -> dict:
    protocol_settings = _app_ctx.ctx.motion_settings.ids['protocol_settings_id']
    try:
        period = float(protocol_settings.ids['capture_period'].text)
    except Exception:
        logger.warning('Failed to read capture period from UI, defaulting to 1', exc_info=True)
        period = 1
        from modules.notification_center import notifications

        notifications.warning(
            'Protocol',
            'Capture Timing',
            'Could not read the capture period; using 1 minute. Check the period '
            'field and restart the protocol if the timing is wrong.',
        )

    period = datetime.timedelta(minutes=period)
    try:
        duration = float(protocol_settings.ids['capture_dur'].text)
    except Exception:
        logger.warning('Failed to read capture duration from UI, defaulting to 1', exc_info=True)
        duration = 1
        from modules.notification_center import notifications

        notifications.warning(
            'Protocol',
            'Capture Timing',
            'Could not read the capture duration; using 1 hour. Check the duration '
            'field and restart the protocol if the timing is wrong.',
        )

    duration = datetime.timedelta(hours=duration)

    # 1-second floor (preserves the 0 single-scan marker) so a short
    # interval/duration stays representable and doesn't round to 0 on display.
    # The clamp is silent here -- this getter runs on every save and run-start,
    # so notifying here re-warns repeatedly. The clamp warning fires once, at
    # the field edit, in ProtocolSettings.update_period / update_duration.
    return {
        'period': config_helpers.floor_protocol_time(period),
        'duration': config_helpers.floor_protocol_time(duration),
    }


# ---------------------------------------------------------------------------
# Hyperstack creation
# ---------------------------------------------------------------------------


def create_hyperstacks_if_needed():
    ctx = _app_ctx.ctx
    image_capture_config = get_image_capture_config_from_ui()
    if image_capture_config['output_format']['sequenced'] == 'OME-TIFF Hyperstack':
        import threading

        from modules.notification_center import notifications

        notifications.info(
            'FileIO',
            'Saving Hyperstacks',
            'Building OME-TIFF Hyperstacks from captured data.\n'
            'This may take several minutes for large datasets.',
        )

        logger.info('Building OME-TIFF Hyperstacks from captured data')
        _, objective = get_current_objective_info()
        run_dir = ctx.sequenced_capture_runner.run_dir()
        tiling_loc = pathlib.Path(ctx.source_path) / 'data' / 'tiling.json'
        binning = get_binning_from_ui()
        has_turret = ctx.scope.motion.has_turret()
        focal_length = objective['focal_length']

        def _build():
            try:
                stack_builder = StackBuilder(has_turret=has_turret)
                stack_builder.load_folder(
                    path=run_dir,
                    tiling_configs_file_loc=tiling_loc,
                    binning_size=binning,
                    focal_length=focal_length,
                )
                logger.info('Hyperstack creation complete')
            except Exception as ex:
                logger.exception('Error building hyperstacks')
                # Background thread: the user already saw the "Saving
                # Hyperstacks" info popup; without this they never see
                # a result.
                notifications.error(
                    'Post-processing',
                    'Hyperstack build failed',
                    f'Could not create hyperstacks: {type(ex).__name__}: {ex}. '
                    f'See logs for details; source files are untouched.',
                )

        threading.Thread(target=_build, daemon=True).start()
