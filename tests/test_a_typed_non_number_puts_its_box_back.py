# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every typed-number box parses through one helper and fails one way.

The kv float and int filters admit '', '.', '-' and '-.', none of them a
number. The Z, X, Y and acceleration boxes each parsed their own text and
chose their own failure -- silent with the text left on screen, or a
refusal record with the text left on screen. ``typed_number`` is the one
parse: a number comes back as one; anything else puts the box back and
comes back as None. The boxes' own behaviour is driven through their real
handlers in test_four_boxes_record_what_was_typed.py and
test_a_refused_action_is_still_recorded.py.
"""

import pytest

from ui.ui_helpers import typed_number


@pytest.mark.parametrize(
    ('text', 'cast', 'number'), [('2.5', float, 2.5), ('-3', float, -3.0), ('40', int, 40)]
)
def test_a_number_comes_back_and_the_box_is_left_alone(text, cast, number):
    put_back = []
    assert typed_number(text, cast, lambda: put_back.append(1)) == number
    assert put_back == []


@pytest.mark.parametrize('text', ['', '.', '-', '-.'])
@pytest.mark.parametrize('cast', [float, int])
def test_what_the_filters_admit_but_is_no_number_puts_the_box_back(text, cast):
    put_back = []
    assert typed_number(text, cast, lambda: put_back.append(1)) is None
    assert put_back == [1]
