# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Where an auto-gain arm landed is named from one set, ``AutoGainConvergence``.

It is a string enum, as every set the API answers is: an ``AutoGainLock``
built with the plain word holds the member, one built with any other word is
refused ``not_offered`` naming its field, and the record's plain form carries
the word a saved image's metadata carries.
"""

import pytest

from modules.exceptions import ArgumentRefusedError
from modules.lumascope_api.camera import AutoGainConvergence, AutoGainLock
from modules.wire_encoding import plain


def test_a_lock_built_with_the_plain_word_holds_the_member_and_answers_the_word():
    lock = AutoGainLock('MAXED', exposure_ms=50.0, gain_db=6.0)

    assert type(lock.state) is AutoGainConvergence
    assert plain(lock)['state'] == 'MAXED'


def test_a_lock_built_with_any_other_word_is_refused_naming_its_field():
    with pytest.raises(ArgumentRefusedError) as refused:
        AutoGainLock('SATURATED')
    assert (refused.value.reason, refused.value.argument) == ('not_offered', 'AutoGainLock.state')
