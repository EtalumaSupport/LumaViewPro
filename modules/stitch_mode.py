# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""How a stitch registers its tiles: the mode it is asked for, and the algorithm that made each mosaic.

A leaf module, importing nothing of the project's, so the post-processing
API names the sets without importing the stitcher, and the stitcher, its
core and the derivation record name them without importing the API.
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


class StitchAlgorithm(enum.StrEnum):
    """The algorithm that made a mosaic, from the stitch's fallback chain.

    A mode tries its registration first (``FAST_FFT_PHASE`` for a fast
    preview, ``QUALITY_LOCAL_NCC`` for quality) where the tiles overlap,
    then places them by their recorded stage positions
    (``STAGE_POSITION``), then by their rank in X and Y
    (``SIMPLE_POSITION``). The values are the ones a mosaic's file records.
    """

    FAST_FFT_PHASE = 'fast_fft_phase'
    QUALITY_LOCAL_NCC = 'quality_local_ncc'
    STAGE_POSITION = 'stage_position_stitcher'
    SIMPLE_POSITION = 'simple_position_stitcher'
