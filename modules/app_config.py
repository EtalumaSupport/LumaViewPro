# Copyright Etaluma, Inc.
"""
Application configuration loaders extracted from lumaviewpro.py.

These functions read settings from JSON files (current.json / settings.json)
and configure logging, engineering mode, and other app-level settings.
"""

import json
import logging
import os

import modules.autofocus_functions as autofocus_functions

logger = logging.getLogger('LVP.modules.app_config')


def load_log_level():
    """Read log level from settings and apply to the root LVP logger."""
    for settings_file in ("./data/current.json", "./data/settings.json"):
        if not os.path.exists(settings_file):
            continue

        with open(settings_file, 'r') as fp:
            data = json.load(fp)

            try:
                log_level = logging.getLevelName(data['logging']['default']['level'])
                logger.setLevel(level=log_level)
                return
            except Exception:
                logger.warning("Failed to read log level from %s", settings_file, exc_info=True)


def get_lvp_lock_port() -> int:
    """Read the LVP instance-lock port from settings, or return default."""
    DEFAULT_LVP_LOCK_PORT = 43101
    for settings_file in ("./data/current.json", "./data/settings.json"):
        if not os.path.exists(settings_file):
            continue

        with open(settings_file, 'r') as fp:
            data = json.load(fp)

            try:
                return data['lvp_lock_port']
            except Exception:
                logger.warning("Failed to read lvp_lock_port from %s", settings_file, exc_info=True)

    return DEFAULT_LVP_LOCK_PORT


def load_autofocus_log_enable():
    """Enable autofocus score logging if configured in settings."""
    for settings_file in ("./data/current.json", "./data/settings.json"):
        if not os.path.exists(settings_file):
            continue

        with open(settings_file, 'r') as fp:
            data = json.load(fp)

            try:
                if data['logging']['autofocus']:
                    autofocus_functions.enable_af_score_logging(enable=True)
                return
            except Exception:
                logger.warning("Failed to read autofocus log setting from %s", settings_file, exc_info=True)


def load_mode() -> bool:
    """Read engineering mode flag from settings. Returns True if engineering mode."""
    for settings_file in ("./data/current.json", "./data/settings.json"):
        if not os.path.exists(settings_file):
            continue

        with open(settings_file, 'r') as fp:
            data = json.load(fp)

            try:
                mode = data['mode']
                if mode == 'engineering':
                    logger.info("Enabling engineering mode")
                    return True
            except Exception:
                logger.warning("Failed to read mode from %s", settings_file, exc_info=True)

    return False
