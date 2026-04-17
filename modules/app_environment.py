# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Application environment initialization — paths, version, platform detection."""

import logging
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass

_logger = logging.getLogger('LVP.app_environment')


@dataclass
class AppEnvironment:
    """Immutable snapshot of application environment determined at startup."""
    script_path: str
    source_path: str
    version: str
    build_timestamp: str
    windows_machine: bool
    num_cores: int
    lvp_installed: bool


def init_environment(main_file: str) -> AppEnvironment:
    """Determine paths, version, and platform. Called once at startup.

    Args:
        main_file: The ``__file__`` of the main script (lumaviewpro.py).

    Returns an AppEnvironment with all resolved values.
    """
    # Determine script location from the main entry point
    abspath = os.path.abspath(main_file)
    basename = os.path.basename(main_file)
    script_path = abspath[:-len(basename)]

    _logger.info(f"Script Location: {script_path}")

    windows_machine = os.name == "nt"

    # Read version and build timestamp via shared reader
    from modules.path_utils import read_version
    version, build_timestamp = read_version(pathlib.Path(script_path))

    # Get git commit hash for build identification (dev mode only)
    if not build_timestamp:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=script_path,
            )
            if result.returncode == 0:
                build_timestamp = result.stdout.strip()
        except Exception as e:
            _logger.debug(f'Failed to get git hash: {e}')

    # Check if running as installed application
    lvp_installed = False
    try:
        with open(os.path.join(script_path, "marker.lvpinstalled")) as f:
            lvp_installed = True
    except Exception:
        pass

    # Determine source_path (data directory)
    if windows_machine and lvp_installed:
        _logger.info("Machine-Type - WINDOWS")
        import userpaths
        documents_folder = userpaths.get_my_documents()
        # Use base version (without hash) for folder name
        # version is already path-safe (no timestamp, no parens)
        lvp_appdata = os.path.join(documents_folder, f"LumaViewPro {version}")

        if not os.path.exists(lvp_appdata):
            os.mkdir(lvp_appdata)

        source_path = lvp_appdata
        _logger.info(f"Data Location: {source_path}")

        if not os.path.exists(os.path.join(lvp_appdata, "data")):
            shutil.copytree(os.path.join(script_path, "data"), os.path.join(lvp_appdata, "data"))

        # Create logs directory if it doesn't exist. The source logs/ folder may not
        # exist in PyInstaller builds, so just create an empty directory structure.
        logs_dir = os.path.join(lvp_appdata, "logs", "LVP_Log")
        os.makedirs(logs_dir, exist_ok=True)

    elif windows_machine and not lvp_installed:
        _logger.info("Machine-Type - WINDOWS (not installed)")
        source_path = script_path
    else:
        _logger.info("Machine-Type - NON-WINDOWS")
        source_path = script_path

    num_cores = os.cpu_count()
    _logger.info(f"Num cores identified as {num_cores}")

    return AppEnvironment(
        script_path=script_path,
        source_path=source_path,
        version=version,
        build_timestamp=build_timestamp,
        windows_machine=windows_machine,
        num_cores=num_cores,
        lvp_installed=lvp_installed,
    )
