# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import enum


@enum.unique
class PostFunction(enum.Enum):
    COMPOSITE = "Composite"
    STITCHED = "Stitched"
    ZPROJECT = "ZProject"
    VIDEO = "Video"
    HYPERSTACK = "Hyperstack"

    @classmethod
    def list_values(cls):
        return list(map(lambda c: c.value, cls))
