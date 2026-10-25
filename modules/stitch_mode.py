# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""How a stitch registers its tiles: the closed set ``stitch(mode=)`` and ``Stitch.mode`` take.

A leaf module, importing nothing of the project's, so the post-processing
API names the set without importing the stitcher, and the stitcher, its
core and the derivation record name it without importing the API.
"""

import enum


class StitchMode(enum.StrEnum):
    """How a stitch registers its tiles.

    ``QUALITY`` registers by bounded local NCC; ``FAST_PREVIEW`` by bounded
    FFT phase correlation. Both place the tiles from the recorded stage
    geometry and keep the source pixels. A fast preview's file name carries
    the ``FastPreview`` token; a quality stitch's carries none.
    """

    QUALITY = 'quality'
    FAST_PREVIEW = 'fast_preview'
