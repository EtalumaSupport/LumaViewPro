# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import os
import json


settings = None

debug_setting = None

# Which file load_debug_setting() actually read debug_mode from
# (current.json or settings.json basename), so the startup banner can
# state the source. Editing the wrong file is a common confusion -- the
# live value comes from current.json once it exists, not settings.json.
debug_setting_source = None

# Required keys are validated in two passes because the load order demands it:
# a settings file is read first and only afterwards has missing keys merged in
# from the shipped defaults. Keys that have always existed are checked at read
# time; keys a newer build introduced are checked after the merge, since a file
# carried forward from an older build legitimately lacks them until then.
#
# Adding a new key to data/settings.json that callers index directly? Register
# it in _MERGE_SUPPLIED_KEYS, never here -- listing it here rejects every
# settings file written by a previous release.
_REQUIRED_SETTINGS_KEYS = frozenset(
    {
        'microscope',
        'live_folder',
        'frame',
    }
)

# Keys the defaults merge is responsible for supplying.
_MERGE_SUPPLIED_KEYS = frozenset(
    {
        'preview_host_downscale',
    }
)


def _validate_settings(settings: dict, filepath: str, logger) -> None:
    """Check that loaded settings contain all required keys and types.

    Raises on missing critical keys. Warns on missing optional keys or
    type mismatches -- allows the app to start with partial config.
    """
    missing = _REQUIRED_SETTINGS_KEYS - settings.keys()
    if missing:
        raise ValueError(
            f'[Settings ] {filepath} missing required keys: {sorted(missing)}. '
            'App cannot start without these keys.'
        )

    # Type checks for critical nested structures
    if 'frame' in settings:
        frame = settings['frame']
        if not isinstance(frame, dict):
            logger.warning(
                f'[Settings ] {filepath}: "frame" should be a dict, got {type(frame).__name__}'
            )
        else:
            for field in ('width', 'height'):
                if field not in frame:
                    logger.warning(f'[Settings ] {filepath}: "frame" missing "{field}"')
                elif not isinstance(frame[field], int):
                    logger.warning(
                        f'[Settings ] {filepath}: "frame.{field}" should be int, got {type(frame[field]).__name__}'
                    )

    # Validate layer settings have expected structure
    from modules.common_utils import get_layers

    _REQUIRED_LAYER_FIELDS = {
        'ill_ma': (int, float),
        'gain_db': (int, float),
        'exp_ms': (int, float),
        'acquire': (str, type(None)),
        'autofocus': bool,
        'false_color': (bool, list),
        'focus': (int, float),
    }
    for layer in get_layers():
        if layer not in settings:
            logger.warning(f'[Settings ] {filepath}: missing layer "{layer}"')
            continue
        layer_settings = settings[layer]
        if not isinstance(layer_settings, dict):
            logger.warning(f'[Settings ] {filepath}: "{layer}" should be dict')
            continue
        for field, _expected_type in _REQUIRED_LAYER_FIELDS.items():
            if field not in layer_settings:
                logger.warning(f'[Settings ] {filepath}: "{layer}" missing "{field}"')

    # Validate motion settings
    if 'motion' in settings:
        if not isinstance(settings['motion'], dict):
            logger.warning(f'[Settings ] {filepath}: "motion" should be dict')
        elif 'acceleration_max_pct' not in settings['motion']:
            logger.warning(f'[Settings ] {filepath}: "motion" missing "acceleration_max_pct"')


def load_settings(logger, filename, lvp_appdata):

    global settings

    # load settings JSON file
    filepath = os.path.join(lvp_appdata, filename) if not os.path.isabs(filename) else filename
    try:
        with open(filepath) as read_file:
            settings = json.load(read_file)
        _validate_settings(settings, filepath, logger)
    except json.JSONDecodeError:
        logger.exception(f'[LVP Main  ] Incompatible JSON file for Microscope Settings: {filepath}')
        settings = None
        raise
    except Exception:
        logger.exception(f'[LVP Main  ] Unable to open file {filepath}')
        raise


def _validate_merged_settings(settings: dict, logger) -> None:
    """Check the settings dict that callers will actually read.

    Runs after the defaults merge, which is the only step that supplies keys a
    newer build added. Callers index these keys directly, so an absent one
    surfaces as a KeyError deep in a render or capture path -- failing here
    names the missing key while the cause is still obvious.

    Raises:
        ValueError: a required key is still absent after the merge.
    """
    missing = _MERGE_SUPPLIED_KEYS - settings.keys()
    if missing:
        raise ValueError(
            f'[Settings ] Settings are missing required keys after merging defaults: '
            f'{sorted(missing)}. Restore data/settings.json from the installation '
            'and restart.'
        )
    logger.debug(f'[Settings ] Validated {len(_MERGE_SUPPLIED_KEYS)} merge-supplied keys')


def _deep_merge_defaults(current: dict, defaults: dict, path: str = '', logger=None) -> list[str]:
    """Recursively merge missing keys from defaults into current.

    Only adds keys that are absent in current -- never overwrites existing
    values. Returns list of keys that were added (for logging).
    """
    added = []
    for key, default_value in defaults.items():
        full_key = f'{path}.{key}' if path else key
        if key not in current:
            current[key] = default_value
            added.append(full_key)
        elif isinstance(default_value, dict) and isinstance(current[key], dict):
            added.extend(_deep_merge_defaults(current[key], default_value, full_key, logger))
    return added


def _migrate_image_mode_setting(logger) -> None:
    """Fold the retiring capture/save toggles into image_mode on load.

    Runs on the loaded dict before the settings.json default-merge so an
    install carrying the old keys keeps its choice rather than picking up
    the merged-in image_mode default.
    """
    global settings
    if settings is None:
        return
    # Deferred to break an import cycle: lvp_logger imports load_debug_setting
    # from this module at its module top (before lvp_logger.logger is defined),
    # and image_mode imports lvp_logger.logger at its own top. load_debug_setting
    # never touches image_mode, so importing it here -- only when a settings load
    # actually migrates -- keeps the logger import safe.
    from modules import image_mode

    if image_mode.migrate_settings_dict(settings):
        logger.info('[Settings ] Consolidated capture/save toggles into image_mode')


def load_lvp_settings(logger, lvp_appdata):
    global settings

    current_path = os.path.join(lvp_appdata, 'data', 'current.json')
    settings_path = os.path.join(lvp_appdata, 'data', 'settings.json')
    data_dir = os.path.join(lvp_appdata, 'data')

    if os.path.exists(current_path):
        try:
            load_settings(logger, current_path, lvp_appdata)
        except (json.JSONDecodeError, ValueError) as e:
            # current.json is corrupt -- fall back to settings.json
            logger.warning(f'[Settings ] {current_path} is corrupt, falling back to settings.json')
            settings = None
            if os.path.exists(settings_path):
                load_settings(logger, settings_path, lvp_appdata)
            else:
                raise FileNotFoundError(
                    f'current.json corrupt and no settings.json fallback in {data_dir}'
                ) from e

        _migrate_image_mode_setting(logger)

        # Merge missing keys from settings.json defaults into current.json.
        # current.json drifts from settings.json as new features add keys.
        # This ensures new keys are available without losing user values.
        if settings is not None and os.path.exists(settings_path):
            try:
                with open(settings_path) as f:
                    defaults = json.load(f)
                added = _deep_merge_defaults(settings, defaults, logger=logger)
                if added:
                    logger.info(
                        f'[Settings ] Merged {len(added)} missing keys from settings.json: {added}'
                    )
            except Exception:
                # Non-fatal on its own: the validation below decides whether the
                # keys this merge would have supplied are actually absent.
                logger.warning(
                    f'[Settings ] Could not merge defaults from {settings_path}', exc_info=True
                )

        _validate_merged_settings(settings, logger)

    elif os.path.exists(settings_path):
        load_settings(logger, settings_path, lvp_appdata)
        _migrate_image_mode_setting(logger)
        _validate_merged_settings(settings, logger)
    else:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Couldn't find 'data' directory at {data_dir}")
        else:
            raise FileNotFoundError(f'No settings files found in {data_dir}')


def _resolve_settings_path(directory):
    current_path = os.path.join(directory, 'data', 'current.json')
    settings_path = os.path.join(directory, 'data', 'settings.json')
    data_dir = os.path.join(directory, 'data')

    if os.path.exists(current_path):
        return current_path
    if os.path.exists(settings_path):
        return settings_path
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Couldn't find 'data' directory at {data_dir}")
    raise FileNotFoundError(f'No settings files found in {data_dir}')


def load_debug_setting(directory):
    global debug_setting, debug_setting_source

    try:
        filename = _resolve_settings_path(directory)
        debug_setting_source = os.path.basename(filename)

        with open(filename) as read_file:
            temp_settings = json.load(read_file)

        debug_setting = temp_settings.get('debug_mode', False)
        return debug_setting

    except Exception as e:
        raise e


def load_profile_trace_setting(directory):
    """Read profile_trace.enabled + profile_trace.output_dir from settings.

    Returns a dict {"enabled": bool, "output_dir": str | None}. Missing
    or unreadable settings file resolves to {"enabled": False,
    "output_dir": None} so the caller never has to guard for absence;
    profile_trace defaults OFF in that case.

    Called from lib/profile_trace.py at module-import time, mirroring
    the timing of load_debug_setting() above. Replaces the prior
    LVP_PROFILE_TRACE environment-variable gate.
    """
    try:
        filename = _resolve_settings_path(directory)
        with open(filename) as read_file:
            temp_settings = json.load(read_file)
    except Exception:
        return {'enabled': False, 'output_dir': None}

    return {
        'enabled': bool(temp_settings.get('profile_trace_enabled', False)),
        'output_dir': temp_settings.get('profile_trace_output_dir') or None,
    }


def load_tracemalloc_setting(directory):
    """Read tracemalloc_enabled from settings.

    Returns bool. Missing or unreadable settings file resolves to False
    so the caller never has to guard for absence; tracemalloc defaults
    OFF in that case (10-30% process-memory overhead is the cost).

    Called from modules/common_utils.py at module-import time, mirroring
    the timing of load_profile_trace_setting() above. Replaces the prior
    LVP_TRACEMALLOC environment-variable gate.
    """
    try:
        filename = _resolve_settings_path(directory)
        with open(filename) as read_file:
            temp_settings = json.load(read_file)
    except Exception:
        return False

    return bool(temp_settings.get('tracemalloc_enabled', False))


def load_memory_profile_setting(directory):
    """Read memory_profile settings (gate + cadence).

    Returns ``{"enabled": bool, "interval_s": float}``. Missing or unreadable
    settings file resolves to ``{"enabled": False, "interval_s": 5.0}`` so the
    caller never has to guard for absence; the memory profiler defaults OFF
    (tracemalloc carries 10-30% process-memory overhead, the same cost as the
    tracemalloc gate). Enable via ``memory_profile_enabled: true`` in the live
    settings (current.json once it exists, settings.json default) -- the same
    merged-settings path as profile_trace / tracemalloc.

    Called from lib/memory_profile.py, mirroring load_profile_trace_setting /
    load_tracemalloc_setting above.
    """
    try:
        filename = _resolve_settings_path(directory)
        with open(filename) as read_file:
            temp_settings = json.load(read_file)
    except Exception:
        return {'enabled': False, 'interval_s': 5.0}

    return {
        'enabled': bool(temp_settings.get('memory_profile_enabled', False)),
        'interval_s': float(temp_settings.get('memory_profile_interval_s', 5.0)),
    }


def load_fx2_debug_wire_setting(directory):
    """Read fx2_debug_wire_enabled from settings.

    Returns bool. Missing or unreadable settings file resolves to False
    so the caller never has to guard for absence; the FX2 wire-protocol
    debug trace defaults OFF (it is an L4 diagnostic surface).

    Called from drivers/fx2driver.py, ui/layer_control.py, and
    modules/lumascope_api/illumination.py at module-import time.
    Replaces the prior LVP_FX2_DEBUG_WIRE environment-variable gate.
    """
    try:
        filename = _resolve_settings_path(directory)
        with open(filename) as read_file:
            temp_settings = json.load(read_file)
    except Exception:
        return False

    return bool(temp_settings.get('fx2_debug_wire_enabled', False))
