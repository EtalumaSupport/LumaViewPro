# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Byte-exact tests for the IDS packed-format reference unpackers.

These pin the derived PFNC lsb-grouped bit layout that the SDK's ConvertTo
output is cross-checked against on the bench. Every expected value is built
from explicit byte sequences (wire order, byte 0 first), so they also pin
endianness: the unpack must not depend on host byte order.

The two packed formats:
  - Mono12g24IDS: 2 px / 3 B, decoded value range 0..4095
  - Mono10g40IDS: 4 px / 5 B, decoded value range 0..1023
Decoded values are RIGHT-aligned in the uint16 (full scale == (1<<bits)-1).
"""

from __future__ import annotations

import numpy as np
import pytest

from drivers.ids_unpack import (
    IDS_PACKED_FORMATS,
    significant_bits_for,
    unpack,
    unpack_mono10g40,
    unpack_mono12g24,
)


class TestMono12g24:
    def test_zero_buffer_is_all_zero(self):
        out = unpack_mono12g24(bytes(3), 2, 1)
        assert out.dtype == np.uint16
        assert out.shape == (1, 2)
        assert out.tolist() == [[0, 0]]

    def test_full_scale_both_pixels(self):
        # [0xFF, 0xFF, 0xFF] -> P0 = P1 = 4095 (12-bit full scale).
        out = unpack_mono12g24(bytes([0xFF, 0xFF, 0xFF]), 2, 1)
        assert out.tolist() == [[4095, 4095]]

    def test_known_distinct_values(self):
        # P0 = 0xABC (2748): hi8=0xAB -> B0, lo4=0xC -> B2 low nibble.
        # P1 = 0x123 (291):  hi8=0x12 -> B1, lo4=0x3 -> B2 high nibble.
        # B2 = (0x3 << 4) | 0xC = 0x3C.
        out = unpack_mono12g24(bytes([0xAB, 0x12, 0x3C]), 2, 1)
        assert out.tolist() == [[0xABC, 0x123]]

    def test_alternating_pattern(self):
        # 0x55,0xAA repeated; verify the decode is stable across groups.
        buf = bytes([0x55, 0xAA, 0x55] * 2)
        out = unpack_mono12g24(buf, 2, 2)
        p0 = (0x55 << 4) | (0x55 & 0x0F)
        p1 = (0xAA << 4) | (0x55 >> 4)
        assert out.tolist() == [[p0, p1], [p0, p1]]

    def test_shape_is_height_by_width(self):
        out = unpack_mono12g24(bytes(3 * 6), 4, 3)
        assert out.shape == (3, 4)

    def test_ramp_round_trips(self):
        # Build a known pixel ramp, pack it, unpack, and compare.
        w, h = 4, 2
        px = (np.arange(w * h, dtype=np.uint16) * 137) & 0x0FFF  # 0..4095
        packed = bytearray()
        for i in range(0, px.size, 2):
            p0, p1 = int(px[i]), int(px[i + 1])
            packed += bytes([(p0 >> 4) & 0xFF, (p1 >> 4) & 0xFF, ((p1 & 0x0F) << 4) | (p0 & 0x0F)])
        out = unpack_mono12g24(bytes(packed), w, h)
        assert out.reshape(-1).tolist() == px.tolist()

    def test_values_never_exceed_12_bit_range(self):
        rng = np.random.default_rng(0)
        buf = rng.integers(0, 256, size=3 * 100, dtype=np.uint8).tobytes()
        out = unpack_mono12g24(buf, 2, 100)
        assert int(out.max()) <= 4095
        assert int(out.min()) >= 0


class TestMono10g40:
    def test_zero_buffer_is_all_zero(self):
        out = unpack_mono10g40(bytes(5), 4, 1)
        assert out.dtype == np.uint16
        assert out.shape == (1, 4)
        assert out.tolist() == [[0, 0, 0, 0]]

    def test_full_scale_all_pixels(self):
        # [0xFF]*5 -> every pixel 1023 (10-bit full scale).
        out = unpack_mono10g40(bytes([0xFF] * 5), 4, 1)
        assert out.tolist() == [[1023, 1023, 1023, 1023]]

    def test_known_distinct_values(self):
        # P0=0x2A9, P1=0x155, P2=0x000, P3=0x3FF.
        #   B0=P0>>2=0xAA, B1=P1>>2=0x55, B2=0x00, B3=P3>>2=0xFF
        #   B4 = (P3&3)<<6 | (P2&3)<<4 | (P1&3)<<2 | (P0&3)
        #      = 3<<6 | 0 | 1<<2 | 1 = 0xC5
        out = unpack_mono10g40(bytes([0xAA, 0x55, 0x00, 0xFF, 0xC5]), 4, 1)
        assert out.tolist() == [[0x2A9, 0x155, 0x000, 0x3FF]]

    def test_lsb_byte_routes_to_correct_pixel(self):
        # Only B4 set, to bit pair for P2 (bits 4-5) = 0b10 -> P2 low bits = 2.
        out = unpack_mono10g40(bytes([0, 0, 0, 0, 0b00100000]), 4, 1)
        assert out.tolist() == [[0, 0, 2, 0]]

    def test_shape_is_height_by_width(self):
        out = unpack_mono10g40(bytes(5 * 6), 8, 3)
        assert out.shape == (3, 8)

    def test_ramp_round_trips(self):
        w, h = 8, 2
        px = (np.arange(w * h, dtype=np.uint16) * 53) & 0x03FF  # 0..1023
        packed = bytearray()
        for i in range(0, px.size, 4):
            quad = [int(px[i + k]) for k in range(4)]
            b4 = 0
            for k in range(4):
                b4 |= (quad[k] & 0x03) << (2 * k)
            packed += bytes(
                [
                    (quad[0] >> 2) & 0xFF,
                    (quad[1] >> 2) & 0xFF,
                    (quad[2] >> 2) & 0xFF,
                    (quad[3] >> 2) & 0xFF,
                    b4,
                ]
            )
        out = unpack_mono10g40(bytes(packed), w, h)
        assert out.reshape(-1).tolist() == px.tolist()

    def test_values_never_exceed_10_bit_range(self):
        rng = np.random.default_rng(1)
        buf = rng.integers(0, 256, size=5 * 100, dtype=np.uint8).tobytes()
        out = unpack_mono10g40(buf, 4, 100)
        assert int(out.max()) <= 1023
        assert int(out.min()) >= 0


class TestInputForms:
    def test_accepts_numpy_uint8_array(self):
        arr = np.array([0xFF, 0xFF, 0xFF], dtype=np.uint8)
        assert unpack_mono12g24(arr, 2, 1).tolist() == [[4095, 4095]]

    def test_accepts_bytearray(self):
        assert unpack_mono12g24(bytearray([0xFF, 0xFF, 0xFF]), 2, 1).tolist() == [[4095, 4095]]

    def test_ignores_trailing_padding(self):
        # One extra padding byte past the 3-byte payload is tolerated.
        out = unpack_mono12g24(bytes([0xFF, 0xFF, 0xFF, 0x00]), 2, 1)
        assert out.tolist() == [[4095, 4095]]


class TestErrors:
    def test_non_multiple_pixel_count_mono12(self):
        with pytest.raises(ValueError):
            unpack_mono12g24(bytes(3), 3, 1)  # 3 px not a multiple of 2

    def test_non_multiple_pixel_count_mono10(self):
        with pytest.raises(ValueError):
            unpack_mono10g40(bytes(5), 2, 1)  # 2 px not a multiple of 4

    def test_short_buffer_mono12(self):
        with pytest.raises(ValueError):
            unpack_mono12g24(bytes(2), 2, 1)  # need 3 bytes

    def test_short_buffer_mono10(self):
        with pytest.raises(ValueError):
            unpack_mono10g40(bytes(4), 4, 1)  # need 5 bytes

    def test_non_positive_dims(self):
        with pytest.raises(ValueError):
            unpack_mono12g24(bytes(3), 0, 1)

    def test_width_not_a_whole_group_even_when_product_divides(self):
        # 2x2 = 4 px is divisible by the 4-px Mono10 group, but width 2 splits a
        # group across the row boundary -> must still raise.
        with pytest.raises(ValueError):
            unpack_mono10g40(bytes(5 * 2), 2, 2)

    def test_rejects_non_uint8_array(self):
        # A uint16 array would be silently truncated to bytes -> reject loudly.
        with pytest.raises(ValueError):
            unpack_mono12g24(np.zeros(3, dtype=np.uint16), 2, 1)

    def test_rejects_multibyte_memoryview(self):
        with pytest.raises(ValueError):
            unpack_mono12g24(memoryview(np.zeros(3, dtype=np.uint16)), 2, 1)


class TestDispatchAndMetadata:
    def test_dispatch_mono12(self):
        out = unpack('Mono12g24IDS', bytes([0xFF, 0xFF, 0xFF]), 2, 1)
        assert out.tolist() == [[4095, 4095]]

    def test_dispatch_mono10(self):
        out = unpack('Mono10g40IDS', bytes([0xFF] * 5), 4, 1)
        assert out.tolist() == [[1023, 1023, 1023, 1023]]

    def test_dispatch_unknown_format_raises(self):
        with pytest.raises(ValueError):
            unpack('Mono8', bytes(4), 4, 1)

    def test_format_table(self):
        assert IDS_PACKED_FORMATS['Mono12g24IDS'] == (2, 3, 12)
        assert IDS_PACKED_FORMATS['Mono10g40IDS'] == (4, 5, 10)

    def test_significant_bits_for(self):
        assert significant_bits_for('Mono12g24IDS') == 12
        assert significant_bits_for('Mono10g40IDS') == 10
        assert significant_bits_for('Mono8') is None
