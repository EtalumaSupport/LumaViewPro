# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

BINNING_SIZE_MAP = {
    '1x1': 1,
    '2x2': 2,
    '4x4': 4,
}


def binning_size_str_to_int(text: str) -> int:
    return BINNING_SIZE_MAP.get(text, 1)


def binning_size_int_to_str(val: int):
    for k, v in BINNING_SIZE_MAP.items():
        if v == val:
            return k

    return '1x1'


def _align_down(value: int, alignment: int) -> int:
    """Floor value to the nearest multiple of alignment, never below it."""
    if alignment <= 1:
        return max(int(value), 1)
    return max((int(value) // alignment) * alignment, alignment)


def native_to_displayed(native: dict, binning: int, alignment: dict | None = None) -> dict:
    """Frame size shown and captured at a binning level: native / binning.

    ``native`` is the unbinned ROI -- the source of truth. Dividing by the
    binning factor and flooring to the sensor's pixel alignment is fully
    determined by ``native`` and ``binning`` alone, so cycling binning up and
    then back down always reproduces the same displayed size. The previous
    code iterated on the already-displayed (and already-floored) value, which
    lost pixels on the way down and never recovered them on the way back up.

    Args:
        native: Unbinned ROI as ``{'width': int, 'height': int}``.
        binning: Binning factor (1, 2, 4, ...).
        alignment: Camera pixel alignment ``{'width': int, 'height': int}``;
            defaults to 1x1 (no alignment constraint).

    Returns:
        Displayed/captured ROI as ``{'width': int, 'height': int}``.
    """
    align = alignment or {'width': 1, 'height': 1}
    return {
        'width': _align_down(native['width'] // binning, align['width']),
        'height': _align_down(native['height'] // binning, align['height']),
    }


def displayed_to_native(displayed: dict, binning: int, native_max: dict) -> dict:
    """Convert a user-entered displayed ROI back to the unbinned native ROI.

    The frame width/height fields are in displayed (post-binning) pixels, so
    the implied native ROI is ``displayed * binning``. It is capped at the
    sensor's physical native resolution so a large value entered at a high
    binning factor cannot imply an ROI bigger than the sensor.

    Args:
        displayed: User-entered ROI as ``{'width': int, 'height': int}``.
        binning: Binning factor the value was entered at.
        native_max: Sensor native resolution ``{'width': int, 'height': int}``.

    Returns:
        Native ROI as ``{'width': int, 'height': int}``.
    """
    return {
        'width': min(int(displayed['width']) * binning, native_max['width']),
        'height': min(int(displayed['height']) * binning, native_max['height']),
    }
