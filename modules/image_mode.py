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

IMAGE_MODE_8BIT = '8bit'
IMAGE_MODE_12BIT_SCIENTIFIC = '12bit_scientific'
IMAGE_MODE_12BIT_SCALED = '12bit_scaled'
IMAGE_MODE_12BIT_FALSE_COLOR_RGB = '12bit_false_color_rgb'

# Save-encoding tokens -- the derived on-disk shape, independent of the mode label.
SAVE_ENCODING_8BIT = '8bit'
SAVE_ENCODING_RIGHT_ALIGNED = 'right_aligned'
SAVE_ENCODING_MSB_ALIGNED = 'msb_aligned'
SAVE_ENCODING_RGB = 'rgb'

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
        ValueError: if mode is not a recognized IMAGE_MODE_* value.
    """
    try:
        return dict(_MODE_TABLE[mode])
    except KeyError:
        raise ValueError(f'unknown image_mode: {mode!r}') from None


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
