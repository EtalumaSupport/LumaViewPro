# Copyright Etaluma, Inc.
"""Application environment initialization — paths, version, platform detection."""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class AppEnvironment:
    """Immutable snapshot of application environment determined at startup."""
    script_path: str
    source_path: str
    version: str
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

    print(f"Script Location: {script_path}")

    windows_machine = os.name == "nt"

    # Read version
    version = ""
    try:
        with open(os.path.join(script_path, "version.txt")) as f:
            version = f.readlines()[0].strip()
    except Exception:
        pass

    # Get git commit hash for build identification
    build_hash = ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=script_path,
        )
        if result.returncode == 0:
            build_hash = result.stdout.strip()
    except Exception:
        pass

    if build_hash:
        version = f"{version} ({build_hash})"

    # Check if running as installed application
    try:
        with open(os.path.join(script_path, "marker.lvpinstalled")) as f:
            lvp_installed = True
    except Exception:
        lvp_installed = False

    # Determine source_path (data directory)
    if windows_machine and lvp_installed:
        print("Machine-Type - WINDOWS")
        import userpaths
        documents_folder = userpaths.get_my_documents()
        # Use base version (without hash) for folder name
        base_version = version.split(" (")[0] if " (" in version else version
        lvp_appdata = os.path.join(documents_folder, f"LumaViewPro {base_version}")

        if not os.path.exists(lvp_appdata):
            os.mkdir(lvp_appdata)

        source_path = lvp_appdata
        print(f"Data Location: {source_path}")

        if not os.path.exists(os.path.join(lvp_appdata, "data")):
            shutil.copytree(os.path.join(script_path, "data"), os.path.join(lvp_appdata, "data"))

        if not os.path.exists(os.path.join(lvp_appdata, "logs")):
            shutil.copytree(os.path.join(script_path, "logs"), os.path.join(lvp_appdata, "logs"))

    elif windows_machine and not lvp_installed:
        print("Machine-Type - WINDOWS (not installed)")
        source_path = script_path
    else:
        print("Machine-Type - NON-WINDOWS")
        source_path = script_path

    num_cores = os.cpu_count()
    print(f"Num cores identified as {num_cores}")

    return AppEnvironment(
        script_path=script_path,
        source_path=source_path,
        version=version,
        windows_machine=windows_machine,
        num_cores=num_cores,
        lvp_installed=lvp_installed,
    )
