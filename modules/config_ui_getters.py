# Copyright Etaluma, Inc.
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

logger = logging.getLogger('LVP.modules.config_ui_getters')

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
from modules.stack_builder import StackBuilder
from modules.zstack_config import ZStackConfig
import modules.labware as labware


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def is_image_saving_enabled() -> bool:
    if _app_ctx.ctx.engineering_mode:
        if _app_ctx.ctx.motion_settings.ids['protocol_settings_id'].ids['protocol_disable_image_saving_id'].active:
            return False

    return True


# ---------------------------------------------------------------------------
# Binning / Z-stack
# ---------------------------------------------------------------------------

def get_binning_from_ui() -> int:
    try:
        return int(_app_ctx.ctx.motion_settings.ids['microscope_settings_id'].ids['binning_spinner'].text)
    except Exception:
        logger.warning("Failed to read binning from UI, defaulting to 1", exc_info=True)
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
    current_pos = ctx.scope.get_current_position('Z')

    zstack_config = ZStackConfig(
        range=config['range'],
        step_size=config['step_size'],
        current_z_reference=config['z_reference'],
        current_z_value=current_pos
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
        raise Exception("No layer currently selected")

    layer_configs = get_layer_configs(
        specific_layers=[c_layer]
    )

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
    if not ctx.scope.motion.driver:
        logger.error(f"Cannot retrieve current plate position")
        return {
            'x': 0,
            'y': 0,
            'z': 0
        }

    pos = ctx.scope.get_current_position(axis=None)
    _, labware_obj = get_selected_labware()
    px, py = ctx.coordinate_transformer.stage_to_plate(
        labware=labware_obj,
        stage_offset=ctx.settings['stage_offset'],
        sx=pos['X'],
        sy=pos['Y'],
    )

    return {
        'x': round(px, common_utils.max_decimal_precision('x')),
        'y': round(py, common_utils.max_decimal_precision('y')),
        'z': round(pos['Z'], common_utils.max_decimal_precision('z'))
    }


def get_current_frame_dimensions() -> dict:
    microscope_settings = _app_ctx.ctx.motion_settings.ids['microscope_settings_id']
    try:
        frame_width = int(microscope_settings.ids['frame_width_id'].text)
        frame_height = int(microscope_settings.ids['frame_height_id'].text)
    except Exception:
        raise ValueError(f"Invalid value for frame width/height")

    frame = {
        'width': frame_width,
        'height': frame_height
    }
    return frame


def get_selected_labware() -> tuple[str, labware.WellPlate]:
    protocol_settings = _app_ctx.ctx.motion_settings.ids['protocol_settings_id']
    labware_id = protocol_settings.ids['labware_spinner'].text
    try:
        if len(labware_id) < 1:
            labware_id = _app_ctx.ctx.settings['protocol']['labware']
        try:
            labware_obj = _app_ctx.ctx.wellplate_loader.get_plate(plate_key=labware_id)
            return labware_id, labware_obj
        except Exception as e:
            logger.exception("LVP Main: Settings file issue. Replace file with a known working version")
            logger.exception(e)
    except Exception as e:
        logger.exception(f"LVP Main: Labware could not be loaded: {e}")
        logger.warning(f"Check to ensure labware {labware_id} is in the labware file")
        return None, None


# ---------------------------------------------------------------------------
# Image capture / sequenced capture
# ---------------------------------------------------------------------------

def get_image_capture_config_from_ui() -> dict:
    microscope_settings = _app_ctx.ctx.motion_settings.ids['microscope_settings_id']
    output_format = {
        'live': microscope_settings.ids['live_image_output_format_spinner'].text,
        'sequenced': microscope_settings.ids['sequenced_image_output_format_spinner'].text,
    }
    use_full_pixel_depth = _app_ctx.ctx.scope_display.use_full_pixel_depth
    return {
        'output_format': output_format,
        'use_full_pixel_depth': use_full_pixel_depth,
    }

def get_sequenced_capture_config_from_ui() -> dict:
    objective_id, _ = get_current_objective_info()
    time_params = get_protocol_time_params()
    labware_id, _ = get_selected_labware()
    protocol_settings = _app_ctx.ctx.motion_settings.ids['protocol_settings_id']
    tiling = protocol_settings.ids['tiling_size_spinner'].text
    use_zstacking = protocol_settings.ids['acquire_zstack_id'].active
    frame_dimensions = get_current_frame_dimensions()
    zstack_params = get_zstack_params()

    layer_configs = get_layer_configs()

    config = {
        'labware_id': labware_id,
        'objective_id': objective_id,
        'zstack_params': zstack_params,
        'use_zstacking': use_zstacking,
        'tiling': tiling,
        'layer_configs': layer_configs,
        'period': time_params['period'],
        'duration': time_params['duration'],
        'frame_dimensions': frame_dimensions,
        'binning_size': get_binning_from_ui(),
        'stim_config': get_stim_configs(),
    }

    return config


# ---------------------------------------------------------------------------
# Auto gain / objective / protocol time
# ---------------------------------------------------------------------------

def get_auto_gain_settings() -> dict:
    return config_helpers.get_auto_gain_settings(_app_ctx.ctx.settings)


def get_current_objective_info() -> tuple[str, dict]:
    return config_helpers.get_current_objective_info(_app_ctx.ctx.settings, _app_ctx.ctx.objective_helper)


def get_protocol_time_params() -> dict:
    protocol_settings = _app_ctx.ctx.motion_settings.ids['protocol_settings_id']
    try:
        period = float(protocol_settings.ids['capture_period'].text)
    except Exception:
        logger.warning("Failed to read capture period from UI, defaulting to 1", exc_info=True)
        period = 1

    period = datetime.timedelta(minutes=period)
    try:
        duration = float(protocol_settings.ids['capture_dur'].text)
    except Exception:
        logger.warning("Failed to read capture duration from UI, defaulting to 1", exc_info=True)
        duration = 1

    duration = datetime.timedelta(hours=duration)

    return {
        'period': period,
        'duration': duration
    }


# ---------------------------------------------------------------------------
# Hyperstack creation
# ---------------------------------------------------------------------------

def create_hyperstacks_if_needed():
    ctx = _app_ctx.ctx
    image_capture_config = get_image_capture_config_from_ui()
    if image_capture_config['output_format']['sequenced'] == 'ImageJ Hyperstack':
        from kivy.clock import Clock
        try:
            from ui.notification_popup import show_notification_popup
            Clock.schedule_once(lambda dt: show_notification_popup(
                title='Saving Hyperstacks',
                message='Building ImageJ Hyperstacks from captured data.\nThis may take several minutes for large datasets.'
            ), 0)
        except ImportError:
            logger.info("Building ImageJ Hyperstacks from captured data")
        _, objective = get_current_objective_info()
        stack_builder = StackBuilder(
            has_turret=ctx.scope.has_turret(),
        )
        stack_builder.load_folder(
            path=ctx.sequenced_capture_executor.run_dir(),
            tiling_configs_file_loc=pathlib.Path(ctx.source_path) / "data" / "tiling.json",
            binning_size=get_binning_from_ui(),
            focal_length=objective['focal_length'],
        )
