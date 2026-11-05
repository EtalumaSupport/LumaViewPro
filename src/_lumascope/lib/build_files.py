# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Where the build's own files are: the one answer every layer reads.

The build's files are the code, the data it ships (``version.txt``, the
camera timing records, the simulator's board INIs), the FX2 hex, the GUI's
kv file and icons, and the git checkout. Their folder is computed here, from
this module's own location with links followed, so a folder of links into a
checkout runs that checkout's files. Frozen, they are in the bundle's folder
(``sys._MEIPASS``), where the analysis collected the package by its import
name, without the checkout's ``src/``.

Whether the process is frozen is read here too, once: the build root needs
the answer, and this is the lowest layer that does.

The folder a run was launched from is a different question, with its own
answer (``_lumascope.modules.path_utils.launch_root``, links not followed); the working
folder answers neither. This module is in ``src/_lumascope/lib/`` so the drivers, which may
not import ``src/_lumascope/modules/``, read the same answer.
"""

import pathlib
import sys


def frozen() -> bool:
    """Whether this process is a PyInstaller build, installed or not.

    The one read of how the process was launched;
    ``_lumascope.modules.path_utils.app_runtime`` asks it.
    """
    return bool(getattr(sys, 'frozen', False))


def build_root() -> pathlib.Path:
    """The folder holding the build's own files, links followed."""
    if frozen():
        return pathlib.Path(sys._MEIPASS)
    return pathlib.Path(__file__).resolve().parents[3]


def build_file(relative: str) -> str:
    """The absolute path of one of the build's files, from its path under the build root.

    A ``str``, for the GUI: a Kivy ``StringProperty`` refuses a ``Path``, and
    a relative path handed to Kivy is looked up in the working folder first.
    """
    return str(build_root() / relative)
