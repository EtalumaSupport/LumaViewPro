# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import os
import json


settings = None

debug_setting = None

# Required top-level keys that must exist in a valid settings file.
# Missing keys cause hard-to-debug runtime errors downstream.
_REQUIRED_SETTINGS_KEYS = frozenset({
    'microscope',
    'live_folder',
    'frame',
})


def _validate_settings(settings: dict, filepath: str, logger) -> None:
    """Check that loaded settings contain all required keys.

    Logs warnings for missing keys but does not raise — allows the app
    to start with partial config rather than crashing on startup.
    """
    missing = _REQUIRED_SETTINGS_KEYS - settings.keys()
    if missing:
        raise ValueError(
            f'[Settings ] {filepath} missing required keys: {sorted(missing)}. '
            'App cannot start without these keys.'
        )

    # Type checks for critical nested structures
    if 'frame' in settings and not isinstance(settings['frame'], dict):
        logger.warning(f'[Settings ] {filepath}: "frame" should be a dict, got {type(settings["frame"]).__name__}')


def load_settings(logger, filename, lvp_appdata):

        global settings

        # load settings JSON file
        filepath = os.path.join(lvp_appdata, filename) if not os.path.isabs(filename) else filename
        try:
            with open(filepath, "r") as read_file:
                settings = json.load(read_file)
            _validate_settings(settings, filepath, logger)
        except json.JSONDecodeError:
            logger.exception(f'[LVP Main  ] Incompatible JSON file for Microscope Settings: {filepath}')
            settings = None
            raise
        except Exception:
            logger.exception(f'[LVP Main  ] Unable to open file {filepath}')
            raise

def load_lvp_settings(logger, lvp_appdata):
    global settings

    current_path = os.path.join(lvp_appdata, "data", "current.json")
    settings_path = os.path.join(lvp_appdata, "data", "settings.json")
    data_dir = os.path.join(lvp_appdata, "data")

    if os.path.exists(current_path):
        try:
            load_settings(logger, current_path, lvp_appdata)
        except (json.JSONDecodeError, ValueError):
            # current.json is corrupt — fall back to settings.json
            logger.warning(f'[Settings ] {current_path} is corrupt, falling back to settings.json')
            settings = None
            if os.path.exists(settings_path):
                load_settings(logger, settings_path, lvp_appdata)
            else:
                raise FileNotFoundError(f'current.json corrupt and no settings.json fallback in {data_dir}')

    elif os.path.exists(settings_path):
        load_settings(logger, settings_path, lvp_appdata)
    else:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Couldn't find 'data' directory at {data_dir}")
        else:
            raise FileNotFoundError(f'No settings files found in {data_dir}')

def load_debug_setting(directory):
    global debug_setting

    current_path = os.path.join(directory, "data", "current.json")
    settings_path = os.path.join(directory, "data", "settings.json")
    data_dir = os.path.join(directory, "data")

    try:
        if os.path.exists(current_path):
            filename = current_path
        elif os.path.exists(settings_path):
            filename = settings_path
        else:
            if not os.path.isdir(data_dir):
                raise FileNotFoundError(f"Couldn't find 'data' directory at {data_dir}")
            else:
                raise FileNotFoundError(f'No settings files found in {data_dir}')

        with open(filename, "r") as read_file:
            temp_settings = json.load(read_file)

        debug_setting = temp_settings.get("debug_mode", False)
        return debug_setting

    except Exception as e:
        raise e
