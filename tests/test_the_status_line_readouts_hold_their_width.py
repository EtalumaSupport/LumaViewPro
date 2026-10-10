# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The status line's readouts hold their width, so the line does not shift as they change.

The readouts -- the frame rates, the camera's MB/s, the cursor's pixel and
plate position -- were in the window title, which the operating system centres
and draws in its own font. They changed width as their values changed: a 9.8
became a 10.2, a pixel at 87 one at 1203, and in macOS's font a 1 is two thirds
the width of a 0, so the whole title slid sideways several times a second.

They are now on a status line above the live view, drawn in Roboto, whose
digits, figure space and minus sign are one width, and each number is padded
with figure spaces to its largest value. The line re-centres only when the
cursor's part appears or vanishes as the cursor crosses the image. What the
tests hold is each readout's width signature: every character in the same
slot is the same width class whatever the value.
"""

import pytest

from ui.shader import cursor_title, frame_rate_title
from ui.ui_helpers import FIGURE_SPACE, MINUS, fixed_number

_DIGIT_WIDE = set('0123456789') | {FIGURE_SPACE, MINUS}


def _signature(text: str) -> str:
    """Each character as its width class: D digit-wide, else itself."""
    return ''.join('D' if c in _DIGIT_WIDE else c for c in text)


def _one_width(texts) -> None:
    signatures = {_signature(t) for t in texts}
    assert len(signatures) == 1, f'the readout changes width: {sorted(texts)}'


class TestANumberHoldsItsWidth:
    def test_every_value_up_to_its_largest_is_one_width(self):
        values = [0, 0.04, 4.26, 9.96, 10.2, 99.9, 123.4, 999.9]
        _one_width([fixed_number(v, whole_digits=3, decimals=1) for v in values])

    def test_a_sign_takes_its_own_slot(self):
        values = [-999.99, -12.34, -0.5, 0, 0.5, 127.76]
        texts = [fixed_number(v, whole_digits=3, decimals=2, signed=True) for v in values]
        _one_width(texts)
        assert texts[1] == FIGURE_SPACE + MINUS + '12.34'
        assert texts[4] == FIGURE_SPACE * 3 + '0.50'

    def test_a_value_is_padded_to_its_largest(self):
        assert fixed_number(4.26, whole_digits=3, decimals=1) == FIGURE_SPACE * 2 + '4.3'


class TestTheStatusLineHoldsItsWidth:
    @pytest.mark.parametrize('engineering', [True, False])
    def test_the_frame_rates_hold_their_width(self, engineering):
        rates = [(0, 0), (4.26, 4.31), (9.9, 10.2), (42.4, 29.5), (99.4, 99.4)]
        _one_width([frame_rate_title(c, d, engineering=engineering) for c, d in rates])

    def test_the_cursor_part_holds_its_width_over_the_image(self):
        positions = [
            ((0, 0), (0.0, 0.0)),
            ((87, 5), (-0.5, 3.0)),
            ((1203, 1877), (-12.34, 127.76)),
            ((3839, 3839), (-999.99, 999.99)),
        ]
        _one_width([cursor_title(pixel, plate) for pixel, plate in positions])

    def test_the_cursor_part_is_gone_off_the_image(self):
        assert cursor_title(None, None) == ''

    def test_without_a_plate_position_only_the_pixel_shows(self):
        text = cursor_title((10, 10), None)
        assert 'Pixel' in text
        assert 'Plate' not in text
