# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Application configuration read at the GUI's startup.

The instance lock's port is read from the settings files, before any
settings are prepared: a host takes the lock before it loads anything, so a
second launch exits first. Engineering mode is read from the prepared
settings.
"""

import logging
import pathlib

from modules.path_utils import get_source_root
from modules.settings_init import SettingsFileError, read_settings_json

logger = logging.getLogger('LVP.modules.app_config')


def _iter_settings_files(source_path: str | pathlib.Path | None = None):
    # Deliberately not _resolve_settings_path. The lock port is read at
    # startup before settings_init merges defaults, and this yields BOTH files so a key
    # absent from current.json falls through to settings.json (callers retry on
    # KeyError). The resolver returns only the first existing file -- no
    # fallthrough -- so swapping it in would drop keys that live only in
    # settings.json.
    #
    # The fallthrough earns its keep across upgrades: a key added in a later
    # release exists in the shipped settings.json but not in an installed
    # current.json, and these run before the merge that would supply it.
    data_dir = get_source_root(source_path) / 'data'
    yield data_dir / 'current.json'
    yield data_dir / 'settings.json'


def get_lvp_lock_port(source_path: str | pathlib.Path | None = None) -> int:
    """Read the LVP instance-lock port from settings, or return default."""
    DEFAULT_LVP_LOCK_PORT = 43101
    for settings_file in _iter_settings_files(source_path):
        try:
            data = read_settings_json(settings_file, logger)
        except FileNotFoundError:
            continue
        except SettingsFileError as e:
            # Try the next file: a key absent here may live in the other one.
            logger.warning(f'Failed to parse {settings_file}: {e}')
            continue

        try:
            return data['lvp_lock_port']
        except Exception:
            logger.warning('Failed to read lvp_lock_port from %s', settings_file, exc_info=True)

    return DEFAULT_LVP_LOCK_PORT


def load_mode(settings: dict) -> bool:
    """Whether the prepared settings start the app in engineering mode."""
    if settings['mode'] == 'engineering':
        logger.info('Enabling engineering mode')
        return True
    return False
