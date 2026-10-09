# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A wire client downloads a file by the name the live folder gives it.

``GET /api/v1/files/<name>``, the name as path segments, as a listing
(``live_folder_listing``) and every returned path's ``name`` give it. It is
resolved through ``ScopeSession.live_folder_path``, so a name that leaves the
live folder is refused as a Python caller's is. The file comes with its type
from its extension, as an attachment under its own name, and a range of it
can be asked for. A name that is no file is ``not_found``.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.settings_fixtures import complete_settings


@pytest.fixture
def live(tmp_path):
    folder = tmp_path / 'live'
    (folder / 'ProtocolData' / 'run1').mkdir(parents=True)
    (folder / 'ProtocolData' / 'run1' / 'notes.txt').write_bytes(b'0123456789')
    return folder


@pytest.fixture
def session(live):
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    yield s
    s.shutdown()


@pytest.fixture
def client(session):
    with TestClient(build_app(session)) as client:
        yield client


def test_a_file_is_downloaded_by_the_name_its_listing_gives(client, session):
    (entry,) = [
        e for e in session.live_folder_listing('ProtocolData/run1') if e.name.endswith('.txt')
    ]

    answer = client.get(f'/api/v1/files/{entry.name}')

    assert answer.status_code == 200
    assert answer.content == b'0123456789'
    assert answer.headers['content-type'].startswith('text/plain')
    assert answer.headers['content-disposition'] == 'attachment; filename="notes.txt"'


def test_a_range_of_a_file_is_answered_on_its_own(client):
    answer = client.get('/api/v1/files/ProtocolData/run1/notes.txt', headers={'Range': 'bytes=2-4'})

    assert answer.status_code == 206
    assert answer.content == b'234'


@pytest.mark.skipif(not hasattr(os, 'symlink'), reason='needs symbolic links')
def test_a_name_outside_the_live_folder_is_refused_as_a_python_caller_is(client, live):
    (live.parent / 'secret.txt').write_text('no')
    # A link out of the live folder: a client's own URL handling folds a
    # `..` away before it is sent, a link it cannot.
    (live / 'escape').symlink_to(live.parent, target_is_directory=True)

    answer = client.get('/api/v1/files/escape/secret.txt')

    assert answer.status_code == 422
    assert answer.json()['reason'] == 'outside_live_folder'


@pytest.mark.parametrize('name', ['ProtocolData/run1/missing.txt', 'ProtocolData/run1'])
def test_a_name_that_is_no_file_is_not_found(client, name):
    answer = client.get(f'/api/v1/files/{name}')

    assert answer.status_code == 404
    assert answer.headers['content-type'] == 'application/problem+json'
    assert answer.json()['reason'] == 'not_found'
    assert answer.json()['detail'] == f'No file {name} is in the live folder.'
