# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A client lists the live folder by name, one level at a time.

A wire client names a file only by its name under the live folder, and
until now nothing told it which names exist: a run's folder, the protocol
saved at the top, the images a run wrote. ``session.live_folder_listing``
lists one level, each entry by the full name a path parameter, a download
and a further listing all take, ``/``-separated on every host. A name
outside the live folder is refused as ``live_folder_path`` refuses it, and
a name that is not a folder is refused rather than answered empty.
"""

from __future__ import annotations

import datetime
import os

import pytest

from modules.exceptions import LiveFolderPathRefusedError
from modules.scope_session import LiveFolderEntry, ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def live(tmp_path):
    folder = tmp_path / 'live'
    (folder / 'ProtocolData' / 'run1').mkdir(parents=True)
    (folder / 'ProtocolData' / 'run1' / 'BF_001.tiff').write_bytes(b'x' * 7)
    (folder / 'plate.tsv').write_text('steps', encoding='utf-8')
    return folder


@pytest.fixture
def session(live):
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    yield s
    s.shutdown()


def test_the_top_level_names_exactly_what_is_on_disk(session, live):
    listing = session.live_folder_listing()

    assert [(e.name, e.kind) for e in listing] == [
        ('ProtocolData', 'folder'),
        ('plate.tsv', 'file'),
    ]
    plate = listing[1]
    assert plate.size == len('steps')
    assert (
        plate.modified
        == datetime.datetime.fromtimestamp(os.stat(live / 'plate.tsv').st_mtime).astimezone()
    )
    assert plate.modified.tzinfo is not None
    assert all(isinstance(e, LiveFolderEntry) for e in listing)


def test_a_deeper_level_is_named_from_the_top_and_its_names_list_further(session):
    (run,) = session.live_folder_listing('ProtocolData')
    assert (run.name, run.kind) == ('ProtocolData/run1', 'folder')

    (image,) = session.live_folder_listing(run.name)
    assert (image.name, image.kind, image.size) == ('ProtocolData/run1/BF_001.tiff', 'file', 7)
    assert session.live_folder_path(image.name).read_bytes() == b'x' * 7


def test_a_live_folder_given_through_a_link_names_entries_the_same(tmp_path, live):
    link = tmp_path / 'link_to_live'
    link.symlink_to(live)
    s = ScopeSession.create(complete_settings(live_folder=str(link)), simulate=True)
    try:
        assert [e.name for e in s.live_folder_listing('ProtocolData')] == ['ProtocolData/run1']
    finally:
        s.shutdown()


@pytest.mark.parametrize('name', ['..', '../x', '/etc', ''])
def test_a_name_outside_the_live_folder_is_refused(session, name):
    with pytest.raises(LiveFolderPathRefusedError) as refused:
        session.live_folder_listing(name)
    assert refused.value.reason == 'outside_live_folder'


@pytest.mark.parametrize('name', ['plate.tsv', 'no_such_folder'])
def test_a_name_that_is_not_a_folder_is_refused(session, name):
    with pytest.raises(LiveFolderPathRefusedError) as refused:
        session.live_folder_listing(name)
    assert refused.value.reason == 'not_a_folder'
    assert refused.value.name == name
