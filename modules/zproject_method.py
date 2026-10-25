# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""How a z-stack is projected to one image: the closed set ``zproject(method=)`` and ``ZProject.method`` take.

A leaf module, importing nothing of the project's, so the post-processing
API names the set without importing the projection backend and its image
library.
"""

import enum


class ZProjectMethod(enum.StrEnum):
    """How a z-stack's slices are reduced to one image, pixel by pixel.

    Each value is its name, the word a caller sends. A projection's file
    name and derivation record carry the name lower-cased
    (``_zproj_max``), so neither depends on these values.
    """

    Min = 'Min'
    Max = 'Max'
    Average = 'Average'
    Median = 'Median'
    Sum = 'Sum'
    StdDev = 'StdDev'
