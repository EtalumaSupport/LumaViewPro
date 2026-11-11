# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scope is composed once, by its session, and brought up once.

The session gives its scope the activity claim and the settings it acts
on, then brings it up; ``create`` ends the bring-up. Given again, the claim
or the settings would point the scope at a claim or a configuration its
session does not hold, and a bring-up write made after the bring-up would
be kept on its record and told as bring-up's. Each is refused, and the
scope keeps what its session gave it.
"""

from __future__ import annotations

import pytest

from _lumascope.modules.activity_claim import ActivityClaim
from _lumascope.modules.exceptions import AlreadyStartedError
from _lumascope.modules.scope_init_config import ScopeInitConfig
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session():
    s = ScopeSession.create(complete_settings(), simulate=True, no_engineering=True)
    yield s
    s.shutdown()


def test_a_second_activity_claim_is_refused(session):
    with pytest.raises(AlreadyStartedError) as refused:
        session.scope.set_activity_claim(ActivityClaim())

    assert refused.value.reason == 'scope_composed'


def test_a_second_settings_binding_is_refused_and_the_sessions_stay(session):
    def another_reader(path):
        return 'not the session'

    def another_writer(updates):
        raise AssertionError('the scope saved to settings its session does not hold')

    with pytest.raises(AlreadyStartedError) as refused:
        session.scope.bind_settings(another_reader, another_writer)

    assert refused.value.reason == 'scope_composed'
    assert session.scope.read_setting('objective_id') == session.settings['objective_id']


def test_after_bring_up_nothing_is_written_or_recorded_as_bring_ups(session):
    record = session.bring_up_record()
    written = []

    with pytest.raises(AlreadyStartedError) as write_refused:
        session.scope.write_camera_at_bring_up(lambda: written.append('write'))
    with pytest.raises(AlreadyStartedError) as configure_refused:
        session.scope.initialize(ScopeInitConfig.from_settings(session.settings))

    assert (write_refused.value.reason, configure_refused.value.reason) == (
        'bring_up_ended',
        'bring_up_ended',
    )
    assert written == []
    assert session.bring_up_record() == record
