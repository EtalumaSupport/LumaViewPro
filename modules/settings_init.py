# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import os
import json
import time


settings = None

debug_setting = None

# Which file load_debug_setting() actually read debug_mode from
# (current.json or settings.json basename), so the startup banner can
# state the source. Editing the wrong file is a common confusion -- the
# live value comes from current.json once it exists, not settings.json.
debug_setting_source = None

# Required top-level keys that must exist in a valid settings file.
# Missing keys cause hard-to-debug runtime errors downstream.
_REQUIRED_SETTINGS_KEYS = frozenset(
    {
        'microscope',
        'live_folder',
        'frame',
    }
)


class SettingsFileError(ValueError):
    """A settings file exists but could not be turned into a settings dict.

    Subclasses ValueError deliberately: `load_lvp_settings` routes a bad
    `current.json` to the shipped template by catching
    `(JSONDecodeError, ValueError)`, and a sibling of that family would
    escape the catch and turn every recoverable failure into a startup
    crash.
    """


def read_settings_json(path, logger=None):
    """Open and parse one settings file. THE one place these files are read.

    Every reader of `current.json` / `settings.json` goes through here so
    that "what counts as an unusable settings file" is decided once. What
    each caller DOES about it still belongs to the caller -- they disagree
    for good reasons (the bootstrap readers try the next file, the report
    generator falls back to other directories, the GUI asks the user), so
    this classifies and they choose.

    No `encoding=` argument, deliberately. Every reader and the writer in
    `microscope_settings.save_settings` use the platform default, which is
    cp1252 on Windows. Reading as UTF-8 here would make a config that
    Windows wrote with any non-ASCII byte -- a live_folder under an
    accented user directory is the everyday case -- suddenly unparseable,
    and the caller would then offer to reset it. Changing this means
    migrating the read and the write together.

    `logger` is optional because `load_debug_setting` runs during logger
    bootstrap, before there is a logger to pass.

    Raises:
        FileNotFoundError: passed through untouched. Callers distinguish
            "no file" from "bad file" -- a missing file is normal on a
            fresh install, and at least one caller catches exactly this
            type to substitute an empty config.
        SettingsFileError: the file exists but did not yield a settings
            dict -- unparseable, undecodable, unreadable, or valid JSON
            that isn't an object (a top-level list used to raise
            AttributeError deep in validation and kill startup).
    """
    try:
        with open(path) as read_file:
            parsed = json.load(read_file)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        raise SettingsFileError(f'{path}: not valid JSON ({e})') from e
    except (OSError, UnicodeDecodeError) as e:
        raise SettingsFileError(f'{path}: could not be read ({e})') from e

    if not isinstance(parsed, dict):
        raise SettingsFileError(f'{path}: expected a JSON object, got {type(parsed).__name__}')
    if logger is not None:
        logger.debug(f'[Settings ] read {path}')
    return parsed


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
        settings = read_settings_json(filepath, logger)
        _validate_settings(settings, filepath, logger)
    except SettingsFileError:
        logger.exception(f'[LVP Main  ] Incompatible JSON file for Microscope Settings: {filepath}')
        settings = None
        raise
    except Exception:
        logger.exception(f'[LVP Main  ] Unable to open file {filepath}')
        raise


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


def migrate_video_settings_dict(settings_dict: dict) -> bool:
    """Carry a configured manual_video section to its new name, video.

    The rate/duration authority applies to every recording path, not just
    manual record, so the section renamed. Must run on a loaded dict
    before the settings.json default-merge: the merge only ADDS missing
    keys, so without this fold an install carrying manual_video.max_fps
    = 10 would get the shipped video.max_fps = 0 merged in and silently
    lose its configured cap.

    Returns:
        True when a manual_video section was found and folded.
    """
    old = settings_dict.pop('manual_video', None)
    if old is None:
        return False
    video = settings_dict.setdefault('video', {})
    for key, value in old.items():
        video.setdefault(key, value)
    return True


def _migrate_video_settings(logger) -> None:
    """Apply migrate_video_settings_dict to the loaded global settings."""
    global settings
    if settings is None:
        return
    if migrate_video_settings_dict(settings):
        logger.info('[Settings ] Renamed manual_video settings section to video')


# Set when current.json could not be used and the app came up on the shipped
# template instead. Holds the rejected file's path and why, until a human
# decides what to do about it.
#
# Read it as `settings_init.rejected_current_json`, never via
# `from modules.settings_init import rejected_current_json`. Several modules
# import `settings` that second way, which copies the value at import time --
# for a dict that is harmless, for a flag that changes during startup it would
# freeze the answer to whatever it was before the settings even loaded.
rejected_current_json = None


def load_lvp_settings(logger, lvp_appdata):
    global settings, rejected_current_json

    # Reset per call: a second load (tests) must not inherit the first's verdict.
    rejected_current_json = None

    current_path = os.path.join(lvp_appdata, 'data', 'current.json')
    settings_path = os.path.join(lvp_appdata, 'data', 'settings.json')
    data_dir = os.path.join(lvp_appdata, 'data')

    if os.path.exists(current_path):
        try:
            load_settings(logger, current_path, lvp_appdata)
        except (json.JSONDecodeError, ValueError) as e:
            # current.json is unusable. Come up on the shipped template so the
            # user gets a working app and a chance to decide -- but do NOT
            # touch their file. It is the only copy of their configuration,
            # and the running app is now holding template values that would
            # overwrite it on the next save.
            logger.error(
                f'[Settings ] {current_path} could not be used ({e}); '
                'starting from the shipped defaults. The file has NOT been '
                'modified and no settings will be saved until this is resolved.'
            )
            settings = None
            if os.path.exists(settings_path):
                load_settings(logger, settings_path, lvp_appdata)
                rejected_current_json = (current_path, str(e))
            else:
                raise FileNotFoundError(
                    f'current.json corrupt and no settings.json fallback in {data_dir}'
                ) from e

        _migrate_image_mode_setting(logger)
        _migrate_video_settings(logger)

        # Merge missing keys from settings.json defaults into current.json.
        # current.json drifts from settings.json as new features add keys.
        # This ensures new keys are available without losing user values.
        if settings is not None and os.path.exists(settings_path):
            try:
                defaults = read_settings_json(settings_path, logger)
                added = _deep_merge_defaults(settings, defaults, logger=logger)
                if added:
                    logger.info(
                        f'[Settings ] Merged {len(added)} missing keys from settings.json: {added}'
                    )
            except Exception:
                logger.warning('[Settings ] Could not load settings.json for default merge')

    elif os.path.exists(settings_path):
        load_settings(logger, settings_path, lvp_appdata)
        _migrate_image_mode_setting(logger)
        _migrate_video_settings(logger)
    else:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Couldn't find 'data' directory at {data_dir}")
        else:
            raise FileNotFoundError(f'No settings files found in {data_dir}')


def retire_rejected_current_json():
    """Move the unusable current.json aside so a fresh one can take its place.

    Renamed, never deleted: it is the user's only copy of their
    configuration, and support can often read what they had out of it even
    when the app could not. Returns the new path.

    Called only after a human has chosen to start over -- the rename is the
    point of no return for that file's role, and nothing should reach it by
    timeout, by a dismissed dialog, or by any other default.
    """
    global rejected_current_json
    if rejected_current_json is None:
        return None
    path, _reason = rejected_current_json
    stamp = time.strftime('%Y%m%d-%H%M%S')
    retired = f'{path}.rejected-{stamp}'
    os.replace(path, retired)
    rejected_current_json = None
    return retired


def settings_are_provisional():
    """True while the app is running on defaults nobody has agreed to keep.

    Writing current.json in this state would replace a user's whole
    configuration with the template, so the writer refuses while it holds.
    """
    return rejected_current_json is not None


def targets_current_json(file):
    """Is this save aimed at the live user configuration?

    Matched on the resolved basename rather than the literal argument: the
    writer normalises its path afterwards (appends .json, absolutizes
    against the source root), and a caller outside this repo may hand it an
    absolute path to the same file. Comparing the string it was given would
    let those through.

    Lives here rather than beside the writer because which file holds the
    user's configuration is a fact about settings, not about the GUI -- any
    future writer needs the same answer.
    """
    if not isinstance(file, (str, os.PathLike)):
        return False
    name = os.fspath(file)
    if name[-5:].lower() != '.json':
        name += '.json'
    return os.path.basename(name).lower() == 'current.json'


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

        temp_settings = read_settings_json(filename)

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
        temp_settings = read_settings_json(filename)
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
        temp_settings = read_settings_json(filename)
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
        temp_settings = read_settings_json(filename)
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
        temp_settings = read_settings_json(filename)
    except Exception:
        return False

    return bool(temp_settings.get('fx2_debug_wire_enabled', False))
