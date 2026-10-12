# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A path the caller gave that cannot be used is refused as the request it is.

Every member that reads or writes a path the caller names answers the same
refusal, ``LiveFolderPathRefusedError``, with the reason that names what is
wrong with the path: ``not_found`` (nothing is there), ``not_a_file`` (a
folder where a file is expected) or ``not_a_folder`` (a file where a folder
is expected). A wire client gets the same reason as a 422. A fault
(``ProtocolNotLoadedError``) is left for a path that is usable and still
failed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modules.exceptions import LiveFolderPathRefusedError
from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def live(tmp_path):
    folder = tmp_path / 'live'
    folder.mkdir()
    (folder / 'a_file.txt').write_text('not an image, not a folder')
    (folder / 'a_folder').mkdir()
    return folder


@pytest.fixture
def session(live):
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    # A protocol is made on a homed scope, with steps for a layer that acquires.
    home_sim_scope(s.scope)
    s.set_layer_acquire('BF', 'image')
    yield s
    s.shutdown()


def _python_rows(session, live):
    """Each member and path, by the reason it is refused with."""
    missing = live / 'nothing_here'
    a_folder = live / 'a_folder'
    return [
        ('load_protocol missing', 'not_found', lambda: session.load_protocol(missing / 'p.tsv')),
        ('load_protocol a folder', 'not_a_file', lambda: session.load_protocol(a_folder)),
        ('open_protocol missing', 'not_found', lambda: session.open_protocol(missing / 'p.tsv')),
        ('open_protocol a folder', 'not_a_file', lambda: session.open_protocol(a_folder)),
        (
            'live_folder_listing missing',
            'not_found',
            lambda: session.live_folder_listing('nothing_here'),
        ),
        (
            'live_folder_listing a file',
            'not_a_folder',
            lambda: session.live_folder_listing('a_file.txt'),
        ),
    ]


def test_every_unusable_path_is_refused_by_what_is_wrong_with_it(session, live):
    wrong = []
    for case, reason, call in _python_rows(session, live):
        try:
            call()
        except LiveFolderPathRefusedError as refused:
            if refused.reason != reason:
                wrong.append(f'{case}: refused {refused.reason!r}, not {reason!r}')
        except Exception as raised:  # each offender is named in the one failure
            wrong.append(
                f'{case}: raised {type(raised).__name__} {getattr(raised, "reason", None)!r}'
            )
        else:
            wrong.append(f'{case}: was not refused')
    assert wrong == []


_WIRE_ROWS = [
    (
        'load_protocol missing',
        '/api/v1/load_protocol',
        {'file_path': 'nothing_here/p.tsv'},
        'not_found',
    ),
    ('load_protocol a folder', '/api/v1/load_protocol', {'file_path': 'a_folder'}, 'not_a_file'),
    (
        'live_folder_listing missing',
        '/api/v1/live_folder_listing',
        {'name': 'nothing_here'},
        'not_found',
    ),
]


def test_a_wire_client_is_refused_the_same_unusable_path_as_a_422(session):
    wrong = []
    with TestClient(build_app(session)) as client:
        for case, route, body, reason in _WIRE_ROWS:
            answer = client.post(route, json=body)
            problem = answer.json()
            got = (answer.status_code, problem.get('reason'), problem.get('title'))
            if got != (422, reason, 'Path Not Available'):
                wrong.append(f'{case}: answered {got}, not 422 {reason!r}')
    assert wrong == []
