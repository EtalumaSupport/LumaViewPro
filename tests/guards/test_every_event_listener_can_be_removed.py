# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every listener member that declares its event has a ``remove_`` beside it.

A client that adds a listener must be able to take it away: the REST
stream adds one per ``@api(event=...)`` member each time the app opens and
removes each, through ``remove_<event>_listener``, when it closes, so an
add with no remove leaves a listener behind on every lifespan. Walks the
session and every sub-object its marked reads lead to, as the stream does.
"""

from __future__ import annotations

import inspect

import pytest

from modules import api_surface
from modules.scope_session import ScopeSession
from rest.events import _owners
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path / 'live')), simulate=True)
    yield s
    s.shutdown()


def test_every_event_listener_member_has_its_remove(session):
    adds = []
    missing = []
    for owner in _owners(session):
        for name, member in inspect.getmembers(type(owner)):
            if api_surface.event_of(member) is None:
                continue
            adds.append(name)
            remove = name.replace('add_', 'remove_', 1)
            if api_surface.mark_of(getattr(type(owner), remove, None)) is None:
                missing.append(f'{type(owner).__name__}.{remove}')

    assert adds, 'the walk found no event listener member: it is not reading the surface'
    assert missing == [], f'event listener members with no marked remove: {missing}'
