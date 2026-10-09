# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The startup banner's build fingerprint: the installed packages and where it reads version.txt.

Where the process was launched from, and whether it is installed, is
``path_utils.app_runtime``'s answer, tested in ``tests/test_app_runtime.py``.
"""

from __future__ import annotations

import pathlib
import re


REPO = pathlib.Path(__file__).resolve().parent.parent
LVP_LOGGER_SRC = REPO / 'lvp_logger.py'
LUMAVIEWPRO_SRC = REPO / 'lumaviewpro.py'


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


class TestBannerReadsBuildIdentityFromInstallDir:
    """The banner reads version.txt (Built / Branch / CommitGUID / BuildID)
    from the directory the executable runs from -- where version.txt ships --
    NOT the per-user data directory.

    Bug: the call passed source_path, which on an installed build points at
    the Documents appdata folder (only data/ + logs/ are copied there, never
    version.txt). The banner's open() then failed and the bare except left
    Built/Branch/CommitGUID reporting "unknown" -- even though the Runtime
    line correctly said "installed exe". The fix passes script_path (the
    install root); on a source/dev run the two paths coincide.
    """

    def test_banner_call_passes_install_dir_not_data_dir(self):
        """The single log_environment_banner call must pass script_path as
        its first argument. source_path is the data dir and has no
        version.txt on installed builds."""
        src = LUMAVIEWPRO_SRC.read_text()
        m = re.search(r'log_environment_banner\(\s*([A-Za-z_]\w*)', src)
        assert m, 'log_environment_banner call not found in lumaviewpro.py'
        first_arg = m.group(1)
        assert first_arg == 'script_path', (
            f'log_environment_banner must be called with script_path (the '
            f'install dir where version.txt ships), not {first_arg!r}. '
            f'source_path is the Documents data dir on installed builds and '
            f'has no version.txt, so Built/Branch/CommitGUID report "unknown".'
        )

    def test_banner_reads_version_txt_from_its_path_param(self):
        """The banner must read version.txt from its path parameter, so the
        install dir passed by the caller is the dir actually probed."""
        src = LVP_LOGGER_SRC.read_text()
        assert re.search(
            r"open\(\s*os\.path\.join\(\s*install_path\s*,\s*['\"]version\.txt['\"]",
            src,
        ), 'banner must read version.txt from its install_path parameter.'
