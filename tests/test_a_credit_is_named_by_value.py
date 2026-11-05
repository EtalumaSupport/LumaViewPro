# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A frame's credit names who counted it by its value.

``Counter`` is a string enum, as every set the API answers is: a record
built with the plain word holds the member, and the record's plain and wire
form carry the value (``'preview'``), not the member's name.
"""

from modules.frame_validity import Counter, Credit
from modules.wire_encoding import plain


def test_a_credit_built_with_the_plain_word_holds_the_member_and_answers_its_value():
    credit = Credit(1, 'preview')

    assert type(credit.by) is Counter
    assert plain(credit) == {'seq': 1, 'by': 'preview'}
