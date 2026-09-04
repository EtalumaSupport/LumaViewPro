# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The save encoding for a GUI-driven derived output, from the live image mode.

A stitch or a z-projection started from the GUI consults the one image-mode
store for its on-disk encoding, so a stitched fluorescence image honours the
user's false-color choice exactly as a freshly captured frame does.

This resolver reads the application context to find that mode, which is why
it lives here and not beside the image writers: the composite merge runs
inside the run engine on every run kind, headless included, and resolves its
encoding from the ruled constant instead (a merged composite is always 8-bit
RGB). Nothing the engine imports reaches this module, so the engine never
learns the mode through a process-wide store.
"""

import numpy as np

import modules.app_context as _app_ctx
import modules.image_mode as image_mode


def resolve_output_save_encoding(array: np.ndarray) -> str:
    """The save encoding for a derived output, resolved from the live image_mode.

    With no application context (headless or before the GUI has published
    one) there is no user image mode to consult, and the only meaningful
    encoding is the verbatim dtype-based one.

    Args:
        array: The pixels about to be written.

    Returns:
        The save-encoding token ``write_tiff`` consumes.
    """
    if _app_ctx.ctx is None:
        return image_mode.encoding_for_array(array)

    with _app_ctx.ctx.settings_lock:
        mode = image_mode.resolve_settings_image_mode(_app_ctx.ctx.settings)
    return image_mode.save_encoding_for_derived_output(array, mode)
