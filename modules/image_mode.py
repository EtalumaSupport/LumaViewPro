"""Image-mode: the single user choice that drives capture depth and save encoding.

One ``image_mode`` value replaces the former ``use_full_pixel_depth`` capture
toggle and ``false_color_16bit`` save toggle. It resolves to two derived facts
consumed across the capture, save, composite, and video paths:

  - ``capture_depth``: how many bits the camera acquires (8 or 12)
  - ``save_encoding``: how acquired pixels land on disk

Modes:
  ``8bit``                  -- 8-bit capture, 8-bit mono save
  ``12bit_scientific``      -- 12-bit capture, right-aligned 0..4095 + SignificantBits
  ``12bit_scaled``          -- 12-bit capture, MSB-aligned (x16) + SignificantBits
  ``12bit_false_color_rgb`` -- 12-bit capture, 3-channel RGB (per-layer color gated)

The 8-bit default preserves the shipped behavior: the retiring
``use_full_pixel_depth`` defaulted off, so an unmigrated install captures 8-bit.
"""

from __future__ import annotations

import numpy as np

from lvp_logger import logger
from modules.exceptions import ConfigError

IMAGE_MODE_8BIT = '8bit'
IMAGE_MODE_12BIT_SCIENTIFIC = '12bit_scientific'
IMAGE_MODE_12BIT_SCALED = '12bit_scaled'
IMAGE_MODE_12BIT_FALSE_COLOR_RGB = '12bit_false_color_rgb'

# Save-encoding tokens -- the derived on-disk shape, independent of the mode label.
SAVE_ENCODING_8BIT = '8bit'
SAVE_ENCODING_RIGHT_ALIGNED = 'right_aligned'
SAVE_ENCODING_MSB_ALIGNED = 'msb_aligned'
SAVE_ENCODING_RGB = 'rgb'

# Every save encoding a caller may legitimately pass to the save path. A value
# outside this set means a bad/typo'd encoding reached the writer, which would
# otherwise fall through to a plain mono write -- the save path validates
# against this set so that becomes a loud failure instead of a silent wrong file.
VALID_SAVE_ENCODINGS = frozenset(
    {
        SAVE_ENCODING_8BIT,
        SAVE_ENCODING_RIGHT_ALIGNED,
        SAVE_ENCODING_MSB_ALIGNED,
        SAVE_ENCODING_RGB,
    }
)

DEFAULT_IMAGE_MODE = IMAGE_MODE_8BIT

_MODE_TABLE: dict[str, dict] = {
    IMAGE_MODE_8BIT: {'capture_depth': 8, 'save_encoding': SAVE_ENCODING_8BIT},
    IMAGE_MODE_12BIT_SCIENTIFIC: {
        'capture_depth': 12,
        'save_encoding': SAVE_ENCODING_RIGHT_ALIGNED,
    },
    IMAGE_MODE_12BIT_SCALED: {'capture_depth': 12, 'save_encoding': SAVE_ENCODING_MSB_ALIGNED},
    IMAGE_MODE_12BIT_FALSE_COLOR_RGB: {'capture_depth': 12, 'save_encoding': SAVE_ENCODING_RGB},
}


def resolve_image_mode(mode: str) -> dict:
    """Resolve an image_mode value to its derived capture depth and save encoding.

    Args:
        mode: one of the IMAGE_MODE_* values.

    Returns:
        A fresh dict with 'capture_depth' (int) and 'save_encoding' (str).

    Raises:
        ConfigError: if mode is not a recognized IMAGE_MODE_* value.
    """
    try:
        return dict(_MODE_TABLE[mode])
    except KeyError:
        raise ConfigError(f'unknown image_mode: {mode!r}') from None


def depth_truncation_warning_active(feature_count: int, image_mode_value: str) -> bool:
    """Whether to warn that summing/binning range is discarded at save time.

    Summing frames or sum-binning accumulates signal beyond a single frame's
    range, but an 8-bit save downconverts that accumulated range back to 8 bits,
    so the extra range is lost on disk. True when such a feature is active
    (count > 1) AND the active mode saves 8-bit -- the cue to pick a 12-bit mode
    to keep the range.

    Args:
        feature_count: The sum or binning factor (1 or None means inactive).
        image_mode_value: The active image_mode (the SSOT value).

    Returns:
        True if the depth-loss warning should be shown.
    """
    if feature_count is None or feature_count <= 1:
        return False
    return resolve_image_mode(image_mode_value)['capture_depth'] == 8


def encoding_for_array(array: np.ndarray) -> str:
    """The save encoding for a derived data product that preserves its pixels.

    A projection / stitch / composite copies its inputs' values verbatim, so it
    saves them as-is rather than re-scaling: an 8-bit array stores 8bit, a
    right-aligned uint16 payload stores right_aligned (the depth tag, not a
    container-filling shift, carries the meaning). MSB-aligning here would alter
    the stored values relative to the right-aligned inputs the producer loaded.

    Args:
        array: The pixels about to be written.

    Returns:
        SAVE_ENCODING_8BIT for a uint8 array, SAVE_ENCODING_RIGHT_ALIGNED
        otherwise (a uint16 container holding a right-aligned payload).
    """
    if array.dtype == np.uint8:
        return SAVE_ENCODING_8BIT
    return SAVE_ENCODING_RIGHT_ALIGNED


def encoding_fills_container(save_encoding: str) -> bool:
    """Whether a save encoding left-justifies the payload to fill its container.

    The scaled (msb_aligned) and false-color (rgb) encodings both brighten a
    narrow payload to fill the 16-bit container so plain viewers render it
    bright; right_aligned and 8bit store the payload at its own width (the depth
    tag carries the scale). The false-color mode shares the scaled mode's
    brightening through this one predicate, so a false-color frame is brightened
    before it is colorized -- colorizing a still-narrow payload would store dark
    color that no plain viewer can show.
    """
    return save_encoding in (SAVE_ENCODING_MSB_ALIGNED, SAVE_ENCODING_RGB)


def save_encoding_for_derived_output(array: np.ndarray, image_mode_value: str) -> str:
    """The save encoding for a derived product, honoring the false-color mode.

    A projection / stitch / composite preserves its inputs' pixel values verbatim
    -- but the RGB false-color mode is a rendering choice that applies to derived
    fluorescence products too: under that mode a derived fluorescence image widens
    to 3-channel false color the same way a freshly captured frame does, so a
    stitched / projected fluorescence image renders in color in plain viewers
    instead of silently demoting to mono. The quantitative modes (scientific /
    scaled / 8bit) keep the verbatim dtype-based encoding -- only the RGB mode
    changes a derived product's on-disk shape.

    Args:
        array: The pixels about to be written.
        image_mode_value: The user's active image_mode (the SSOT value).

    Returns:
        SAVE_ENCODING_RGB when image_mode_value is the RGB mode, else the
        verbatim dtype-based encoding from encoding_for_array.
    """
    if resolve_image_mode(image_mode_value)['save_encoding'] == SAVE_ENCODING_RGB:
        return SAVE_ENCODING_RGB
    return encoding_for_array(array)


def migrate_legacy_settings(use_full_pixel_depth: bool, false_color_16bit: bool) -> str:
    """Map the two retiring settings keys onto a single image_mode value.

    8-bit capture has no 16-bit false color, so false_color_16bit is ignored
    when use_full_pixel_depth is False. No legacy combination maps to
    12bit_scaled -- that mode is new and reachable only by explicit selection.
    """
    if not use_full_pixel_depth:
        return IMAGE_MODE_8BIT
    if false_color_16bit:
        return IMAGE_MODE_12BIT_FALSE_COLOR_RGB
    return IMAGE_MODE_12BIT_SCIENTIFIC


# User-facing labels for the Image mode selector. The selector is the only
# place these strings appear; storage and the resolver use the enum values.
IMAGE_MODE_LABELS = {
    IMAGE_MODE_8BIT: '8-bit',
    IMAGE_MODE_12BIT_SCIENTIFIC: '12-bit (scientific)',
    IMAGE_MODE_12BIT_SCALED: '12-bit (scaled)',
    IMAGE_MODE_12BIT_FALSE_COLOR_RGB: '12-bit false color (RGB)',
}

LABEL_TO_IMAGE_MODE = {label: mode for mode, label in IMAGE_MODE_LABELS.items()}

# Pixel formats that mean the camera can deliver a 12-bit payload. A camera
# offering neither (the FX2/MT9P031 in the LS560/620/720 streams only the top
# 8 bits) can capture 8-bit only, so it must not be offered the 12-bit modes.
_TWELVE_BIT_PIXEL_FORMATS = ('Mono12', 'Mono12p')


def camera_supports_12bit(supported_pixel_formats) -> bool:
    """Whether a camera advertising these pixel formats can capture 12-bit."""
    formats = supported_pixel_formats or ()
    return any(fmt in formats for fmt in _TWELVE_BIT_PIXEL_FORMATS)


def available_modes(supported_pixel_formats) -> list:
    """The image_mode values selectable on a camera with these pixel formats.

    8-bit is always available; the three 12-bit modes require Mono12/Mono12p,
    so an 8-bit-only camera never offers an impossible 12-bit choice.
    """
    modes = [IMAGE_MODE_8BIT]
    if camera_supports_12bit(supported_pixel_formats):
        modes.extend(
            [
                IMAGE_MODE_12BIT_SCIENTIFIC,
                IMAGE_MODE_12BIT_SCALED,
                IMAGE_MODE_12BIT_FALSE_COLOR_RGB,
            ]
        )
    return modes


def available_mode_labels(supported_pixel_formats) -> list:
    """The user-facing labels for available_modes, in selector order."""
    return [IMAGE_MODE_LABELS[mode] for mode in available_modes(supported_pixel_formats)]


def resolve_settings_image_mode(settings) -> str:
    """The authoritative image_mode for a settings dict.

    Prefers an explicit ``image_mode`` key; falls back to deriving it from the
    legacy ``use_full_pixel_depth`` / ``false_color_16bit`` keys for installs
    saved before the consolidated key existed.
    """
    mode = settings.get('image_mode')
    if mode in _MODE_TABLE:
        return mode
    # A missing image_mode is a pre-consolidation install -- migrate it silently
    # from the legacy keys. A present-but-unrecognized value is a corrupt setting:
    # surface it, because the coercion below otherwise hides the data loss.
    if mode is not None:
        logger.warning(
            f'[ImageMode] Stored image_mode {mode!r} is not recognized; '
            'coercing to the legacy-derived default.'
        )
    return migrate_legacy_settings(
        settings.get('use_full_pixel_depth', False),
        settings.get('false_color_16bit', False),
    )


def migrate_settings_dict(settings: dict) -> bool:
    """Fold the two retiring keys into image_mode in-place, then drop them.

    Run once on load, before the settings.json default-merge: an install
    saved with the legacy keys (and no image_mode) keeps its capture/save
    choice instead of being reset to the merged-in default. Returns whether
    the dict changed, so the caller can log the one-time migration.

    Args:
        settings: the loaded settings dict, mutated in place.

    Returns:
        True if image_mode was set or a legacy key was removed.
    """
    had_legacy = 'use_full_pixel_depth' in settings or 'false_color_16bit' in settings
    needs_mode = settings.get('image_mode') not in _MODE_TABLE
    if not had_legacy and not needs_mode:
        return False
    if needs_mode:
        settings['image_mode'] = resolve_settings_image_mode(settings)
    settings.pop('use_full_pixel_depth', None)
    settings.pop('false_color_16bit', None)
    return True
