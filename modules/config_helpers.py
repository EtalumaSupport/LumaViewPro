"""
GUI-independent configuration and state helper functions.

These functions extract configuration data from the settings dict
and scope objects without any Kivy/GUI dependencies. They can be
used by LumaViewPro, the REST API, or standalone scripts.
"""

import datetime
import os
import pathlib

from lvp_logger import logger
import modules.common_utils as common_utils


# ---------------------------------------------------------------------------
# Protocol / Step helpers
# ---------------------------------------------------------------------------

def find_nearest_step(x: float, y: float, protocol) -> int:
    """Given a position, find the nearest step index in the protocol."""
    if protocol is None or protocol.num_steps() <= 0:
        return -1

    steps_df = protocol.steps()
    idx = (steps_df[['X', 'Y']].sub([x, y]).pow(2).sum(1)).idxmin()
    return idx


# ---------------------------------------------------------------------------
# Layer / channel configuration
# ---------------------------------------------------------------------------

def get_layer_configs(settings: dict, specific_layers: list | None = None) -> dict:
    """Build config dicts for each layer from settings.

    Returns:
        dict of layer_name -> config dict with keys:
        acquire, video_config, stim_config, autofocus, false_color,
        illumination, gain, auto_gain, exposure, sum, focus
    """
    layer_configs = {}
    for layer in common_utils.get_layers():

        if (specific_layers is not None) and (layer not in specific_layers):
            continue

        layer_settings = settings[layer]

        acquire = layer_settings['acquire']
        video_config = layer_settings['video_config']

        if 'stim_config' in layer_settings:
            # Force an update to keep stim_config.illumination in sync with layer illumination
            settings[layer]['stim_config']['illumination'] = layer_settings['ill']
            stim_config = layer_settings['stim_config']
        else:
            stim_config = None

        autofocus = layer_settings['autofocus']
        false_color = layer_settings['false_color']
        illumination = round(layer_settings['ill'], common_utils.max_decimal_precision('illumination'))
        sum_count = layer_settings['sum']
        gain = round(layer_settings['gain'], common_utils.max_decimal_precision('gain'))
        auto_gain = common_utils.to_bool(layer_settings['auto_gain'])
        exposure = round(layer_settings['exp'], common_utils.max_decimal_precision('exposure'))
        focus = layer_settings['focus']

        # Final check to ensure consistent stim_config.illumination
        if stim_config is not None:
            stim_config['illumination'] = illumination

        layer_configs[layer] = {
            'acquire': acquire,
            'video_config': video_config,
            'stim_config': stim_config,
            'autofocus': autofocus,
            'false_color': false_color,
            'illumination': illumination,
            'gain': gain,
            'auto_gain': auto_gain,
            'exposure': exposure,
            'sum': sum_count,
            'focus': focus,
        }

    return layer_configs


def get_stim_configs(settings: dict) -> dict:
    """Build per-layer stim configs from settings."""
    stim_configs = {}
    layer_configs = get_layer_configs(settings)
    for layer in layer_configs:
        if layer_configs[layer]['stim_config'] is not None:
            stim_configs[layer] = layer_configs[layer]['stim_config']
    return stim_configs


def get_enabled_stim_configs(settings: dict) -> dict:
    """Return only stim configs where enabled is True."""
    stim_configs = get_stim_configs(settings)
    return {layer: cfg for layer, cfg in stim_configs.items() if cfg['enabled']}


def get_auto_gain_settings(settings: dict) -> dict:
    """Extract auto gain settings, converting max_duration_seconds to timedelta."""
    autogain_settings = settings['protocol']['autogain'].copy()
    autogain_settings['max_duration'] = datetime.timedelta(
        seconds=autogain_settings['max_duration_seconds']
    )
    del autogain_settings['max_duration_seconds']
    return autogain_settings


def get_current_objective_info(settings: dict, objective_helper) -> tuple[str, dict]:
    """Return (objective_id, objective_info_dict) from current settings."""
    objective_id = settings['objective_id']
    objective = objective_helper.get_objective_info(objective_id=objective_id)
    return objective_id, objective


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def get_current_plate_position(
    scope,
    settings: dict,
    coordinate_transformer,
    wellplate_loader,
) -> dict:
    """Get current plate position in plate coordinates.

    Returns:
        dict with keys 'x', 'y', 'z' in plate coordinates (um).
    """
    if not scope.motion or not scope.motion.driver:
        logger.error("Cannot retrieve current plate position")
        return {'x': 0, 'y': 0, 'z': 0}

    pos = scope.get_current_position(axis=None)

    labware_id = settings.get('protocol', {}).get('labware', '')
    try:
        labware = wellplate_loader.get_plate(plate_key=labware_id)
    except Exception:
        logger.warning(f"Could not load labware '{labware_id}' for position conversion")
        return {
            'x': round(pos.get('X', 0), common_utils.max_decimal_precision('x')),
            'y': round(pos.get('Y', 0), common_utils.max_decimal_precision('y')),
            'z': round(pos.get('Z', 0), common_utils.max_decimal_precision('z')),
        }

    px, py = coordinate_transformer.stage_to_plate(
        labware=labware,
        stage_offset=settings['stage_offset'],
        sx=pos['X'],
        sy=pos['Y'],
    )

    return {
        'x': round(px, common_utils.max_decimal_precision('x')),
        'y': round(py, common_utils.max_decimal_precision('y')),
        'z': round(pos['Z'], common_utils.max_decimal_precision('z')),
    }


# ---------------------------------------------------------------------------
# System / logging helpers
# ---------------------------------------------------------------------------

def log_system_metrics(settings: dict):
    """Log CPU, RAM, and disk metrics."""
    path = settings.get('live_folder', '.')
    metrics = common_utils.system_metrics(path=path)
    free_space = common_utils.check_disk_space(path=path)

    if free_space < 1024:  # Less than 1 GB
        logger.error(
            f"Low disk space: {free_space:.1f} MB remaining",
            extra={'force_error': True},
        )

    logger.info(
        f"[SYSTEM METRICS] CPU Usage: {metrics['cpu_percent_total']:.1f}% | "
        f"RAM Available: {metrics['ram_available_gb']:.1f} GB | "
        f"RAM Usage: {metrics['ram_percent_total']:.1f}%",
        extra={'force_error': True},
    )
    logger.info(
        f"[DISK METRICS] Disk Free: {metrics['disk_free_gb']:.1f} GB | "
        f"Disk Usage: {metrics['disk_used_percent']:.1f}%",
        extra={'force_error': True},
    )
    logger.info(
        f"[PROCESS METRICS] Process CPU Usage: {metrics['cpu_percent_python']:.1f}% | "
        f"Process RAM Usage: {metrics['ram_used_python_mb']:.1f} MB, "
        f"{metrics['ram_used_python_percent']:.1f}%",
        extra={'force_error': True},
    )

    extra_disks = common_utils.get_extra_disks_info(exclude_path=path)
    if extra_disks:
        logger.info(f"[EXTRA DISKS] {extra_disks}", extra={'force_error': True})


def focus_log(positions, values, focus_round: int, source_path: str) -> int:
    """Log autofocus positions and scores to file. Returns incremented focus_round."""
    if False:  # disabled — kept for future use
        os.chdir(source_path)
        try:
            file = open('./logs/focus_log.txt', 'a')
        except Exception:
            if not os.path.isdir('./logs'):
                raise FileNotFoundError("Couldn't find 'logs' directory.")
            else:
                raise
        for i, p in enumerate(positions):
            mssg = str(focus_round) + '\t' + str(p) + '\t' + str(values[i]) + '\n'
            file.write(mssg)
        file.close()
    return focus_round + 1


def block_wait_for_threads(futures: list, log_loc: str = "LVP") -> None:
    """Block until all futures complete, logging any errors."""
    for future in futures:
        try:
            future.result()
        except Exception as e:
            logger.error(f"{log_loc} ] Thread Error: {e}")
