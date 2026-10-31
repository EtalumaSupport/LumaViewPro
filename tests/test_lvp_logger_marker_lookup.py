# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The startup banner's build fingerprint: the installed packages.

Where the process was launched from, and whether it is installed, is
``path_utils.app_runtime``'s answer, tested in ``tests/test_app_runtime.py``;
that the banner is logged once per process and reads the build's files
from the script root, ``tests/test_every_session_host_logs_its_environment_once.py``.
"""

from __future__ import annotations

import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
LVP_LOGGER_SRC = REPO / 'lvp_logger.py'


class TestInstalledPackageInventory:
    """The startup banner must record the full installed-package set so a
    support bundle is reproducible. install.bat / pip upgrades change
    behavior (e.g. the ffmpeg/x264 build bundled inside `av`) WITHOUT
    touching the LVP build identity, so the banner logs the freeze.

    Source-text assertions (this module mocks `lvp_logger` at import, so
    behavior is verified against the source, matching this file's pattern).
    """

    def _src(self):
        return LVP_LOGGER_SRC.read_text()

    def test_collector_uses_importlib_metadata_not_pip_subprocess(self):
        src = self._src()
        assert 'def _collect_installed_packages' in src, (
            'lvp_logger must define _collect_installed_packages for the banner.'
        )
        # Must enumerate via importlib.metadata (works in the frozen exe;
        # no pip subprocess), not by shelling out to `pip freeze`.
        assert 'importlib.metadata' in src
        assert 'pip freeze' not in src.replace('`pip freeze`', '')

    def test_banner_logs_key_deps_and_full_freeze(self):
        src = self._src()
        assert 'Key deps:' in src, 'banner must log a Key deps highlights line (av, numpy, ...).'
        assert 'Installed packages (' in src, 'banner must log the full installed-package freeze.'
        assert "'av'" in src, 'av must be in the highlighted key deps.'
