# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The installation's own files, copied into a test's data folder.

A scope reads every file the installation ships from the folder it is
started on, before anything starts, and refuses to come up without them.
A test that builds a session on a folder of its own copies them here, so
a new installation file is added in one place rather than at every site.
"""

import pathlib
import shutil

SHIPPED_DATA = pathlib.Path(__file__).resolve().parents[1] / 'data'

INSTALLATION_FILES = (
    'labware.json',
    'objectives.json',
    'scopes.json',
    'motorconfig_defaults.json',
    'settings.json',
)


def copy_installation_files(data: pathlib.Path) -> None:
    """Copy the shipped installation files into ``data``, an existing data folder."""
    for name in INSTALLATION_FILES:
        shutil.copy(SHIPPED_DATA / name, pathlib.Path(data) / name)
