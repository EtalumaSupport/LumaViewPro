# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Application configuration loaders extracted from lumaviewpro.py.

These functions read settings from JSON files (current.json / settings.json)
and configure logging, engineering mode, and other app-level settings.
"""

import logging
import pathlib

import modules.autofocus_functions as autofocus_functions
from modules.path_utils import get_source_root
from modules.settings_init import SettingsFileError, read_settings_json

logger = logging.getLogger('LVP.modules.app_config')


def _iter_settings_files(source_path: str | pathlib.Path | None = None):
    # Deliberately not _resolve_settings_path. These readers run at startup
    # before settings_init merges defaults, and they yield BOTH files so a key
    # absent from current.json falls through to settings.json (callers retry on
    # KeyError). The resolver returns only the first existing file -- no
    # fallthrough -- so swapping it in would drop keys that live only in
    # settings.json. Revisit if these reads move onto the merged settings dict.
    #
    # The fallthrough earns its keep across upgrades: a key added in a later
    # release exists in the shipped settings.json but not in an installed
    # current.json, and these run before the merge that would supply it.
    data_dir = get_source_root(source_path) / 'data'
    yield data_dir / 'current.json'
    yield data_dir / 'settings.json'


def load_log_level(source_path: str | pathlib.Path | None = None) -> None:
    """Read log level from settings and apply to the root LVP logger."""
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
            log_level = logging.getLevelName(data['logging']['default']['level'])
            logger.setLevel(level=log_level)
            return
        except Exception:
            logger.warning('Failed to read log level from %s', settings_file, exc_info=True)


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


def load_autofocus_log_enable(source_path: str | pathlib.Path | None = None) -> None:
    """Enable autofocus score logging if configured in settings."""
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
            if data['logging']['autofocus']:
                autofocus_functions.enable_af_score_logging(enable=True)
            return
        except Exception:
            logger.warning(
                'Failed to read autofocus log setting from %s', settings_file, exc_info=True
            )


def load_mode(source_path: str | pathlib.Path | None = None) -> bool:
    """Read engineering mode flag from settings. Returns True if engineering mode."""
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
            mode = data['mode']
            if mode == 'engineering':
                logger.info('Enabling engineering mode')
                return True
        except Exception:
            logger.warning('Failed to read mode from %s', settings_file, exc_info=True)

    return False
