# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The shipped motor defaults, read and checked once, before any motor driver is built.

A motor driver computes each axis's limit from the travel and travel margin
these give it, converts every position by their microsteps per mm, and
scales every saved image by their optics. A figure it cannot use, raised
inside the driver, would be taken by the registry for a board that did not
come up, and the user told the board was unusable rather than that the file
was. So every figure is judged here, where the file is read, by the one
table the board's figures are judged by (`motorconfig.figure_problem`), and
every reader of the file reads it here.
"""

from __future__ import annotations

import pathlib

from _lumascope.drivers.motorconfig import (
    ANTIBACKLASH,
    JUDGED_SECTIONS,
    LIMITED_AXES,
    MARGIN,
    MICROSTEPS,
    TRAVEL,
    figure_problem,
    unusable_travel,
)
from _lumascope.modules.exceptions import InstallationFileError
from _lumascope.modules.path_utils import read_installation_file


# The figures a motor driver reads with no default under them: each limited
# axis's travel pair, every axis's conversion, and the Z antibacklash the
# focus moves take.
_REQUIRED = (
    (TRAVEL, LIMITED_AXES),
    (MARGIN, LIMITED_AXES),
    (MICROSTEPS, (*LIMITED_AXES, 'T')),
    (ANTIBACKLASH, ('Z',)),
)


def read_motor_defaults(path: str | pathlib.Path) -> dict:
    """The motor defaults in ``path``, every figure usable.

    Raises:
        InstallationFileError: the file cannot be read as an installation
            file; it states none of a figure a driver needs (``_REQUIRED``);
            it gives X, Y or Z a travel pair that leaves no travel
            (``unusable_travel``); or a section the table judges is not a
            mapping or holds a figure the table refuses
            (``figure_problem``). Each names the file and the key.
    """
    defaults = read_installation_file(path)
    for section, keys in _REQUIRED:
        for key in keys:
            figures = defaults.get(section)
            if not isinstance(figures, dict) or key not in figures:
                raise InstallationFileError(path, f'states no {section!r} for {key}')
    for axis in LIMITED_AXES:
        problem = unusable_travel(defaults[TRAVEL][axis], defaults[MARGIN][axis])
        if problem is not None:
            raise InstallationFileError(
                path, f'gives {axis} no travel limit in {TRAVEL!r} / {MARGIN!r}: {problem}'
            )
    for section in JUDGED_SECTIONS:
        if section not in defaults:
            continue
        figures = defaults[section]
        if not isinstance(figures, dict):
            raise InstallationFileError(path, f'{section!r} is not a mapping: {figures!r}')
        for key, figure in figures.items():
            problem = figure_problem(section, figure)
            if problem is not None:
                raise InstallationFileError(path, f'{section!r} {key}: {problem}')
    return defaults
