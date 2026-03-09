# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import enum

@enum.unique
class ColorChannel(enum.Enum):
    Blue = 0
    Green = 1
    Red = 2
    BF = 3
    PC = 4
    DF = 5
    Lumi = 6
