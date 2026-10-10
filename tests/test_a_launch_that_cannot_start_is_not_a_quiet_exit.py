# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A GUI launch that cannot load its settings stops by an exception, never a quiet exit.

The packaged build is windowed with no console. Its bootloader shows a dialog
with the message of any exception that escapes the application, and nothing
for a ``SystemExit``, which it takes for a deliberate stop. The settings load
used to log CRITICAL and call ``sys.exit(1)``, so a researcher whose settings
or installation files could not be used saw the app vanish. The refusal now
escapes: the crash hook logs it once, as CRASH, and the bootloader shows it.

The bootloader's dialog itself is PyInstaller's and is checked on Windows.
What is checked here is that the launch ends by an escaping exception, which
is what the dialog is shown for.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_a_settings_file_that_cannot_be_used_escapes_and_is_logged_once_as_a_crash(tmp_path):
    folder = tmp_path / 'launch'
    folder.mkdir()
    for item in ('lvp_logger.py', 'version.txt', 'modules', 'drivers', 'ui', 'lib', 'plugins'):
        if (REPO / item).exists():
            (folder / item).symlink_to(REPO / item)
    # Copied, not linked: a linked main script would run against the clone's data.
    shutil.copy(REPO / 'lumaviewpro.py', folder / 'lumaviewpro.py')
    shutil.copytree(REPO / 'data', folder / 'data')
    (folder / 'data' / 'current.json').unlink(missing_ok=True)
    (folder / 'data' / 'settings.json').write_text('{ not json')

    ended = subprocess.run(
        [sys.executable, 'lumaviewpro.py', '--simulate', '--no-engineering'],
        cwd=str(folder),
        capture_output=True,
        text=True,
        timeout=120,
    )

    log = (folder / 'logs' / 'LVP_Log' / 'lumaviewpro.log').read_text()
    assert ended.returncode != 0
    assert log.count('CRASH - Uncaught Exception') == 1, log[-3000:]
    assert 'InstallationFileError: settings.json in ' in log
    assert 'cannot continue' not in log
