# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The shipped motor defaults, read and checked once, before any motor driver is built.

A motor driver computes each axis's limit from the travel and travel margin
these give it. A pair it cannot use, raised inside the driver, would be
taken by the registry for a board that did not come up, and the user told
the board was unusable rather than that the file was. So the pair is judged
here, where the file is read, and every reader of the file reads it here.
"""

from __future__ import annotations

import pathlib

from _lumascope.drivers.motorconfig import LIMITED_AXES, MARGIN, TRAVEL, unusable_travel
from _lumascope.modules.exceptions import InstallationFileError
from _lumascope.modules.path_utils import read_installation_file


def read_motor_defaults(path: str | pathlib.Path) -> dict:
    """The motor defaults in ``path``, every limited axis's travel usable.

    Raises:
        InstallationFileError: the file cannot be read as an installation
            file, or it gives X, Y or Z no travel, no travel margin, or a
            pair that leaves no travel (``unusable_travel``), naming the
            file and the key.
    """
    defaults = read_installation_file(path)
    for axis in LIMITED_AXES:
        for section in (TRAVEL, MARGIN):
            figures = defaults.get(section)
            if not isinstance(figures, dict) or axis not in figures:
                raise InstallationFileError(path, f'states no {section!r} for {axis}')
        problem = unusable_travel(defaults[TRAVEL][axis], defaults[MARGIN][axis])
        if problem is not None:
            raise InstallationFileError(
                path, f'gives {axis} no travel limit in {TRAVEL!r} / {MARGIN!r}: {problem}'
            )
    return defaults
