# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a capture waits out is named from one set, ``FrameValiditySource``.

A recorded change names its source as a member, so a reader keys by the
member or its plain string alike; a capture's exclusions are judged at the
door, taking a member's plain string and refusing any other word
``not_offered``; and a capture that fails for a change says that change in
a person's words, whichever source it was.
"""

import pytest

from modules.exceptions import ArgumentRefusedError, CaptureFailedError
from modules.frame_validity import FrameValiditySource
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield s
    s.shutdown()


def test_a_recorded_change_names_its_source_as_the_member(session):
    session.scope.illumination.led_on('Red', 10.0)

    changes = session.scope.imaging.recorded_changes()

    led = changes['led']
    assert type(led.source) is FrameValiditySource
    assert led.source == FrameValiditySource.LED
    assert all(type(source) is FrameValiditySource for source in changes)


def test_an_exclusion_takes_a_members_plain_string_and_refuses_any_other_word(session):
    imaging = session.scope.imaging
    session.scope.illumination.led_on('Red', 10.0)

    imaging.frames_until_valid(exclude_sources=('led',))
    with pytest.raises(ArgumentRefusedError) as refused:
        imaging.frames_until_valid(exclude_sources=('led', 'flash'))
    assert (refused.value.reason, refused.value.argument) == ('not_offered', 'exclude_sources[1]')
    assert refused.value.offered == tuple(FrameValiditySource)


def test_a_capture_failure_says_every_sources_change_in_words():
    unsaid = [
        source
        for source in FrameValiditySource
        if 'None' in str(CaptureFailedError('no_fresh_frame', source=source))
    ]
    assert unsaid == []
