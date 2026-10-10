# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A wire caller's path is a name under the live folder, and reaches nothing beside it.

The REST server passes every path a caller gives through one member,
``ScopeSession.live_folder_path``. A name that is a path of its own -- absolute,
on a drive, on a network share -- or that climbs out through ``..`` or a
link, would reach the rest of the machine; each is refused, on every host,
in the forms either operating system writes. A missing live folder is
refused and never created.
"""

from __future__ import annotations

import os
import shutil

import pytest

from modules.exceptions import LiveFolderPathRefusedError, Refusal
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


def _session(live_folder):
    return ScopeSession.create(complete_settings(live_folder=str(live_folder)), simulate=True)


@pytest.fixture
def live(tmp_path):
    folder = tmp_path / 'live'
    folder.mkdir()
    return folder


@pytest.fixture
def session(live):
    s = _session(live)
    yield s
    s.shutdown()


@pytest.mark.parametrize(
    ('name', 'under'),
    [
        ('ProtocolData/run1', ('ProtocolData', 'run1')),
        ('plate.tsv', ('plate.tsv',)),
        ('a/../b', ('b',)),
        ('.', ()),
    ],
)
def test_a_name_under_the_live_folder_is_answered_absolute(session, live, name, under):
    assert session.live_folder_path(name) == live.resolve().joinpath(*under)


@pytest.mark.parametrize(
    'name',
    [
        '',
        '/x',
        '/',
        'C:\\x',
        'C:x',
        'C:/x',
        '\\x',
        '\\\\server\\share\\x',
        '//server/share/x',
        '../x',
        'a/../../x',
        '..',
        'a\x00b',
    ],
)
def test_a_name_that_leaves_the_live_folder_is_refused(session, name):
    with pytest.raises(LiveFolderPathRefusedError) as refused:
        session.live_folder_path(name)

    assert refused.value.reason == 'outside_live_folder'
    assert refused.value.name == name
    assert isinstance(refused.value, Refusal)


@pytest.mark.skipif(not hasattr(os, 'symlink'), reason='needs symbolic links')
def test_a_link_out_of_the_live_folder_is_refused(session, live, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (live / 'way_out').symlink_to(outside, target_is_directory=True)

    with pytest.raises(LiveFolderPathRefusedError) as refused:
        session.live_folder_path('way_out/x')

    assert refused.value.reason == 'outside_live_folder'


@pytest.mark.skipif(not hasattr(os, 'symlink'), reason='needs symbolic links')
def test_a_live_folder_reached_through_a_link_answers_inside_it(tmp_path):
    real = tmp_path / 'real_live'
    real.mkdir()
    linked = tmp_path / 'linked_live'
    linked.symlink_to(real, target_is_directory=True)
    s = _session(linked)
    try:
        assert s.live_folder_path('ProtocolData/run1') == real.resolve() / 'ProtocolData' / 'run1'
        with pytest.raises(LiveFolderPathRefusedError):
            s.live_folder_path('../real_live_sibling')
    finally:
        s.shutdown()


def test_a_missing_live_folder_is_refused_and_not_created(session, live):
    shutil.rmtree(live)

    with pytest.raises(LiveFolderPathRefusedError) as refused:
        session.live_folder_path('ProtocolData')

    assert refused.value.reason == 'capture_location_unusable'
    assert not live.exists()
