# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A wire client puts a file into the live folder by its name there.

``PUT /api/v1/files/<name>`` writes the body as a new file through
``ScopeSession.put_live_folder_file``: the name goes through
``live_folder_path``, its folders are made, and a file already there is
refused, so an upload never replaces what a capture wrote. It is reached as
every write is -- on a call thread stamped as a REST client, in the control
session its header names -- so a client that does not hold a held session
is refused.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modules import path_utils
from modules.caller import Caller, current_caller
from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.settings_fixtures import complete_settings


@pytest.fixture
def live(tmp_path):
    folder = tmp_path / 'live'
    folder.mkdir()
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


def test_a_files_bytes_are_taken_whatever_type_the_client_names(client, live):
    """The upload takes bytes; the JSON rule is the JSON routes'. curl's own
    defaults name other types, and the route declares octet-stream."""
    for name, media in (('plate.bin', 'application/octet-stream'), ('plate.png', 'image/png')):
        answer = client.put(
            f'/api/v1/files/{name}', content=b'\x89PNG', headers={'Content-Type': media}
        )

        assert answer.status_code == 201, (media, answer.text)
        assert (live / name).read_bytes() == b'\x89PNG'


def test_a_file_is_written_under_its_name_with_its_folders_made(client, live):
    answer = client.put('/api/v1/files/Uploads/plate 1/layout.csv', content=b'A1,B1\n')

    assert answer.status_code == 201, answer.text
    assert answer.json()['name'] == 'Uploads/plate 1/layout.csv'
    assert (live / 'Uploads' / 'plate 1' / 'layout.csv').read_bytes() == b'A1,B1\n'


def test_a_file_already_there_is_refused_and_kept(client, live):
    (live / 'live_BF_000001.tiff').write_bytes(b'what the capture wrote')

    refused = client.put('/api/v1/files/live_BF_000001.tiff', content=b'an upload')

    assert refused.status_code == 409
    assert refused.json()['reason'] == 'file_exists'
    assert (live / 'live_BF_000001.tiff').read_bytes() == b'what the capture wrote'


def test_a_name_outside_the_live_folder_is_refused_and_nothing_is_written(client, live):
    refused = client.put('/api/v1/files/..%2Fescape.txt', content=b'x')

    assert refused.status_code == 422
    assert refused.json()['reason'] == 'outside_live_folder'
    assert not (live.parent / 'escape.txt').exists()


def test_the_write_runs_as_a_rest_client(client, monkeypatch):
    seen = []
    real = path_utils.write_new_file

    def watched(path, data):
        # Observes the caller where the write runs; the real writer writes.
        seen.append(current_caller())
        real(path, data)

    monkeypatch.setattr(path_utils, 'write_new_file', watched)

    assert client.put('/api/v1/files/notes.txt', content=b'x').status_code == 201
    assert seen == [Caller.REST]


def test_under_a_held_session_only_its_holder_writes(client, live):
    session_id = client.post('/api/v1/take_control', json={'name': 'plate reader'}).json()

    refused = client.put('/api/v1/files/stranger.txt', content=b'x')
    written = client.put(
        '/api/v1/files/holder.txt', content=b'x', headers={'Control-Session': session_id}
    )

    assert refused.status_code == 409
    assert refused.json()['reason'] == 'control_session_held'
    assert not (live / 'stranger.txt').exists()
    assert written.status_code == 201, written.text
    assert (live / 'holder.txt').read_bytes() == b'x'


def test_a_write_the_disk_refuses_is_a_fault_naming_the_file_and_leaves_nothing(client, live):
    (live / 'locked').mkdir()
    (live / 'locked').chmod(0o500)
    try:
        failed = client.put('/api/v1/files/locked/notes.txt', content=b'x')
    finally:
        (live / 'locked').chmod(0o700)

    assert failed.status_code == 500
    assert failed.json()['reason'] == 'live_folder_file_not_written', failed.json()
    assert 'locked/notes.txt' in failed.json()['detail']
    assert not (live / 'locked' / 'notes.txt').exists()
