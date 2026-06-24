# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Reference unpackers for the IDS packed monochrome wire formats.

The IMX676 body in the U3-34L0XCP-M delivers two packed formats:

  - Mono12g24IDS -- 12 bits/pixel, lsb-grouped into 24 bits (2 px / 3 B)
  - Mono10g40IDS -- 10 bits/pixel, lsb-grouped into 40 bits (4 px / 5 B)

These functions decode such a wire buffer into a right-aligned uint16 array
(value range 0..4095 for Mono12, 0..1023 for Mono10, sitting in the low bits of
the 16-bit container). They are NOT the production decode path: the driver lets
the SDK do the per-frame unpack via ids_peak_ipl ConvertTo/Convert, which is
faster and reuses its output buffer. These exist as a verification oracle -- the
bit layout below is derived from the EMVA GenICam PFNC "lsb-grouped" definition
(IDS publishes only the bit budget, no wire diagram), so the SDK's output must
be cross-checked against it on real hardware before it is trusted.

One detail is undocumented and pending hardware confirmation against a known
gray ramp: whether the decoded value is right- or left-aligned inside the
uint16. IDS states "6 or 4 bits unused" but not which end. This module
implements the RIGHT-aligned layout (full scale == (1 << bits) - 1). If the
bench shows the SDK left-aligns, the consuming code -- not this oracle -- adapts.

Endianness: every operation is on individual uint8 lanes with explicit shifts,
so the result is independent of host byte order. The buffer is consumed in wire
order (byte 0 first).
"""

from __future__ import annotations

import numpy as np

# format name -> (pixels_per_group, bytes_per_group, significant_bits)
IDS_PACKED_FORMATS = {
    'Mono12g24IDS': (2, 3, 12),
    'Mono10g40IDS': (4, 5, 10),
}


def _as_u8(packed) -> np.ndarray:
    """Return packed as a 1-D uint8 array; raise if it is not a raw byte buffer.

    The wire payload is raw bytes. A multi-byte memoryview or a non-uint8 array
    would be silently reinterpreted (wrong element count, or truncated values),
    decoding garbage instead of failing -- so reject it loudly.
    """
    if isinstance(packed, memoryview):
        if packed.itemsize != 1:
            raise ValueError(
                f'packed buffer must be a byte buffer, got memoryview with '
                f'itemsize {packed.itemsize}'
            )
        return np.frombuffer(packed, dtype=np.uint8)
    if isinstance(packed, (bytes, bytearray)):
        return np.frombuffer(packed, dtype=np.uint8)
    arr = np.asarray(packed)
    if arr.dtype != np.uint8:
        raise ValueError(f'packed buffer must be uint8 bytes, got dtype {arr.dtype}')
    return arr.reshape(-1)


def _grouped(
    packed, width: int, height: int, px_per_group: int, bytes_per_group: int
) -> np.ndarray:
    """Validate sizes and return the byte buffer reshaped to (n_groups, B).

    Width must be a whole number of pixel groups: these formats pack groups
    within a row and the IMX676 widths leave no intra-row padding, so a row
    starts on a group boundary and the buffer decodes as one flat group stream.
    Validating width (not just the pixel product) rejects a width that splits a
    group across a row boundary even when height makes the product divisible.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f'width and height must be positive (got {width}x{height})')
    if width % px_per_group:
        raise ValueError(
            f'width {width} is not a multiple of {px_per_group} px/group; '
            f'packed rows must contain whole pixel groups'
        )
    n_groups = (width * height) // px_per_group
    need = n_groups * bytes_per_group
    raw = _as_u8(packed)
    if raw.size < need:
        raise ValueError(
            f'buffer too short: need {need} bytes for {width}x{height}, got {raw.size}'
        )
    # Trailing bytes (transport padding) past the pixel payload are ignored.
    return raw[:need].reshape(n_groups, bytes_per_group)


def unpack_mono12g24(packed, width: int, height: int) -> np.ndarray:
    """Decode Mono12g24IDS (2 px / 3 B) to a right-aligned uint16 (h, w) array.

    Layout per 3-byte group (B0, B1, B2) holding pixels (P0, P1):
        P0 = (B0 << 4) | (B2 & 0x0F)   # 0..4095
        P1 = (B1 << 4) | (B2 >> 4)     # 0..4095
    """
    g = _grouped(packed, width, height, 2, 3)
    b0 = g[:, 0].astype(np.uint16)
    b1 = g[:, 1].astype(np.uint16)
    b2 = g[:, 2].astype(np.uint16)
    out = np.empty(width * height, dtype=np.uint16)
    out[0::2] = (b0 << 4) | (b2 & 0x0F)
    out[1::2] = (b1 << 4) | (b2 >> 4)
    return out.reshape(height, width)


def unpack_mono10g40(packed, width: int, height: int) -> np.ndarray:
    """Decode Mono10g40IDS (4 px / 5 B) to a right-aligned uint16 (h, w) array.

    Layout per 5-byte group (B0..B4) holding pixels (P0..P3); B4 carries the two
    low bits of each pixel, lsb of P0 in B4's lowest bits:
        Px = (Bx << 2) | ((B4 >> (2 * x)) & 0x03)   # 0..1023, x in 0..3
    """
    g = _grouped(packed, width, height, 4, 5)
    b4 = g[:, 4].astype(np.uint16)
    out = np.empty(width * height, dtype=np.uint16)
    for x in range(4):
        out[x::4] = (g[:, x].astype(np.uint16) << 2) | ((b4 >> (2 * x)) & 0x03)
    return out.reshape(height, width)


def unpack(format_name: str, packed, width: int, height: int) -> np.ndarray:
    """Dispatch to the unpacker for a packed IDS format name.

    Raises ValueError for a format this oracle does not decode (e.g. Mono8, which
    is not packed and needs no unpack).
    """
    if format_name == 'Mono12g24IDS':
        return unpack_mono12g24(packed, width, height)
    if format_name == 'Mono10g40IDS':
        return unpack_mono10g40(packed, width, height)
    raise ValueError(f'no packed-format unpacker for {format_name!r}')


def significant_bits_for(format_name: str) -> int | None:
    """Significant bits for a known packed IDS format, else None.

    The driver derives depth from the format name generically; this mirror lets
    the oracle's tests assert the full-scale value without the camera base class.
    """
    spec = IDS_PACKED_FORMATS.get(format_name)
    return spec[2] if spec else None
