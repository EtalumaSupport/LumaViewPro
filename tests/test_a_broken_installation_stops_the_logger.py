# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A shipped settings file the logger cannot read stops the launch and names it.

``lvp_logger`` reads ``debug_mode`` before the settings are prepared. When
the installation's own ``settings.json`` cannot answer, the import raises
``InstallationFileError`` naming the file, which the packaged build's
bootloader shows, and the main log records it once as a crash; reading the
flag as off would hide the broken installation behind a quieter log.

The suite's conftest stands a MagicMock in for ``lvp_logger``, so the real
module is imported in a child process, frozen with its executable in a
scratch folder so its data root is that folder.
"""

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_importing_the_logger_with_no_shipped_settings_raises_naming_the_file(tmp_path):
    exe = tmp_path / 'LumaViewPro.exe'
    code = f'import sys\nsys.frozen = True\nsys.executable = {str(exe)!r}\nimport lvp_logger\n'

    done = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=60, cwd=str(REPO)
    )

    assert done.returncode != 0, 'the import must not carry on with debug read as off'
    last = done.stderr.strip().splitlines()[-1]
    assert last.startswith('modules.exceptions.InstallationFileError: settings.json in '), last
    assert str(tmp_path / 'data') in last
    # The main log records it once, as the crash it is: no crash hook exists
    # yet at import, so the bundle would otherwise say nothing of the stop.
    log = (tmp_path / 'logs' / 'LVP_Log' / 'lumaviewpro.log').read_text()
    assert log.count('CRASH - Uncaught Exception') == 1, log
    assert 'InstallationFileError: settings.json in ' in log
