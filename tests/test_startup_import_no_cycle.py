# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Startup import-cycle guard.

lvp_logger imports load_debug_setting from modules.settings_init at its module
top -- before lvp_logger.logger is defined further down the file. If
settings_init pulls in modules.image_mode at ITS module top, and image_mode
imports lvp_logger.logger at top, the chain resolves `from lvp_logger import
logger` against a half-initialized lvp_logger and raises ImportError, so the app
never launches.

The regular unit suite cannot catch this: conftest (and pytest's own startup)
import lvp_logger long before any test runs, so the module is already fully
initialized in-process and the cycle never re-triggers. A fresh subprocess
import is the only faithful reproduction of app startup, so these run the import
in a clean interpreter.
"""

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def _fresh_import(statement: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, '-c', statement],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )


def test_lvp_logger_imports_in_a_fresh_interpreter():
    # The exact app-startup path: lumaviewpro.py imports lvp_logger first.
    result = _fresh_import('import lvp_logger')
    assert result.returncode == 0, (
        f'fresh import of lvp_logger failed (startup import cycle?):\n{result.stderr}'
    )


def test_settings_init_and_image_mode_import_together():
    # The other end of the cycle: settings_init defers the image_mode import,
    # so loading both (in either order) must stay clean.
    result = _fresh_import('import modules.settings_init; import modules.image_mode')
    assert result.returncode == 0, (
        f'fresh import of settings_init + image_mode failed (import cycle?):\n{result.stderr}'
    )
