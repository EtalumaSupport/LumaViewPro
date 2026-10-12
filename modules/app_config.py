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
from modules.settings_init import read_before_preparation

logger = logging.getLogger('LVP.modules.app_config')


def get_lvp_lock_port(source_path: str | pathlib.Path | None = None) -> int:
    """The LVP instance-lock port, read before the settings are prepared."""
    port, _ = read_before_preparation(str(get_source_root(source_path)), 'lvp_lock_port')
    return port


def load_mode(settings: dict) -> bool:
    """Whether the prepared settings start the app in engineering mode."""
    if settings['mode'] == 'engineering':
        logger.info('Enabling engineering mode')
        return True
    return False
