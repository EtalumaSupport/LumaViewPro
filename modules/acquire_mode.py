# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""What a layer captures: the closed set a layer's setting, a step and a protocol's Layer Settings name.

A leaf module, importing nothing of the project's, so the step-name and
size helpers in ``common_utils``, which the protocol imports, name the set
without importing the protocol.
"""

import enum


class AcquireMode(enum.StrEnum):
    """What a layer captures: an image or a video.

    A layer that captures nothing holds None, not a member. The values are
    the ones the settings and a protocol file store, so neither changes.
    """

    IMAGE = 'image'
    VIDEO = 'video'
