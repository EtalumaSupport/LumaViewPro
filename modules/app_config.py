# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Application configuration loaders extracted from lumaviewpro.py.

These functions read settings from JSON files (current.json / settings.json)
and configure logging, engineering mode, and other app-level settings.
"""

import json
import logging
import pathlib

import modules.autofocus_functions as autofocus_functions
from modules.path_utils import get_source_root

logger = logging.getLogger('LVP.modules.app_config')


def _iter_settings_files(source_path: str | pathlib.Path | None = None):
    data_dir = get_source_root(source_path) / "data"
    yield data_dir / "current.json"
    yield data_dir / "settings.json"


def load_log_level(source_path: str | pathlib.Path | None = None):
    """Read log level from settings and apply to the root LVP logger."""
    for settings_file in _iter_settings_files(source_path):
        if not settings_file.exists():
            continue

        with open(settings_file, 'r') as fp:
            try:
                data = json.load(fp)
            except json.JSONDecodeError as e:
                logger.warning(f'Failed to parse {settings_file}: {e}')
                continue

            try:
                log_level = logging.getLevelName(data['logging']['default']['level'])
                logger.setLevel(level=log_level)
                return
            except Exception:
                logger.warning("Failed to read log level from %s", settings_file, exc_info=True)


def get_lvp_lock_port(source_path: str | pathlib.Path | None = None) -> int:
    """Read the LVP instance-lock port from settings, or return default."""
    DEFAULT_LVP_LOCK_PORT = 43101
    for settings_file in _iter_settings_files(source_path):
        if not settings_file.exists():
            continue

        with open(settings_file, 'r') as fp:
            try:
                data = json.load(fp)
            except json.JSONDecodeError as e:
                logger.warning(f'Failed to parse {settings_file}: {e}')
                continue

            try:
                return data['lvp_lock_port']
            except Exception:
                logger.warning("Failed to read lvp_lock_port from %s", settings_file, exc_info=True)

    return DEFAULT_LVP_LOCK_PORT


def load_autofocus_log_enable(source_path: str | pathlib.Path | None = None):
    """Enable autofocus score logging if configured in settings."""
    for settings_file in _iter_settings_files(source_path):
        if not settings_file.exists():
            continue

        with open(settings_file, 'r') as fp:
            try:
                data = json.load(fp)
            except json.JSONDecodeError as e:
                logger.warning(f'Failed to parse {settings_file}: {e}')
                continue

            try:
                if data['logging']['autofocus']:
                    autofocus_functions.enable_af_score_logging(enable=True)
                return
            except Exception:
                logger.warning("Failed to read autofocus log setting from %s", settings_file, exc_info=True)


def load_mode(source_path: str | pathlib.Path | None = None) -> bool:
    """Read engineering mode flag from settings. Returns True if engineering mode."""
    for settings_file in _iter_settings_files(source_path):
        if not settings_file.exists():
            continue

        with open(settings_file, 'r') as fp:
            try:
                data = json.load(fp)
            except json.JSONDecodeError as e:
                logger.warning(f'Failed to parse {settings_file}: {e}')
                continue

            try:
                mode = data['mode']
                if mode == 'engineering':
                    logger.info("Enabling engineering mode")
                    return True
            except Exception:
                logger.warning("Failed to read mode from %s", settings_file, exc_info=True)

    return False
