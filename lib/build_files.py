# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Where the build's own files are: the one answer every layer reads.

The build's files are the code, the data it ships (``version.txt``, the
camera timing records, the simulator's board INIs), the FX2 hex, the GUI's
kv file and icons, and the git checkout. Their folder is computed here, from
this module's own location with links followed, so a folder of links into a
checkout runs that checkout's files. Frozen, the module sits in the bundle's
folder (``sys._MEIPASS``), which is the build root.

The folder a run was launched from is a different question, with its own
answer (``modules.path_utils.launch_root``, links not followed); the working
folder answers neither. This module is in ``lib/`` so the drivers, which may
not import ``modules/``, read the same answer.
"""

import pathlib


def build_root() -> pathlib.Path:
    """The folder holding the build's own files, links followed."""
    return pathlib.Path(__file__).resolve().parent.parent
