# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

from modules.exceptions import ConfigError


def binning_size_str_to_int(text: str) -> int:
    """Parse a binning label ``'NxN'`` into its factor N.

    A binning label is arithmetic, not a lookup: the offered set is the
    camera's own supported-factor list rendered as ``f'{s}x{s}'``, so a
    hardcoded table here is a second source of truth for which factors may
    exist -- and the narrower one, which silently answered 1 for every
    label it did not carry.

    Only a POSITIVE SQUARE label is a binning factor. The three shapes the
    old table was containing by accident all have live consequences:
    ``'0x0'`` divides the native ROI by zero, ``'-1x-1'`` persists a
    negative native ROI, and ``'2x4'`` is not a single factor at all.

    Raises:
        ConfigError: on anything that is not a positive square label, and
            NOTHING ELSE. Callers boundary on ConfigError alone, so a
            leaked ValueError or AttributeError -- from a bare ``int()`` on
            a placeholder, or a ``.split`` on a non-string ``"size": 4`` --
            would walk straight past them into the host's re-raise.
    """
    if not isinstance(text, str):
        raise ConfigError(f'binning size must be an NxN label, got {type(text).__name__}: {text!r}')
    halves = text.split('x')
    if len(halves) != 2:
        raise ConfigError(f'binning size is not an NxN label: {text!r}')
    try:
        width = int(halves[0])
        height = int(halves[1])
    except ValueError:
        raise ConfigError(f'binning size is not an NxN label: {text!r}') from None
    if width != height:
        raise ConfigError(f'binning size is not square: {text!r}')
    if width < 1:
        raise ConfigError(f'binning size must be at least 1x1: {text!r}')
    return width


def binning_size_int_to_str(val: int) -> str:
    """Format a binning factor as its ``'NxN'`` label.

    Does NOT raise. Its one production caller is fed
    ``imaging.get_binning_size()``, which contracts to >= 1, and it sits in
    a failure-recovery path where raising would be the hazard.
    """
    return f'{val}x{val}'


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
