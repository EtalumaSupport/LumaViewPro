# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The tiling grids an installation offers are the record's own data.

``TilingConfig`` crosses the wire as a record, and a record has no
address, so its two methods are in-process; a wire client holds its data.
Until now it had none: the grids were reachable only through the
methods. ``available`` and ``default`` are read from tiling.json once, at
construction, and the in-process methods answer from them.
"""

import json

from modules.exceptions import InstallationFileError
from modules.tiling_config import TilingConfig


def _write(tmp_path, document):
    path = tmp_path / 'tiling.json'
    path.write_text(json.dumps(document), encoding='utf-8')
    return path


def test_the_grids_and_the_default_are_fields(tmp_path):
    path = _write(
        tmp_path,
        {
            'metadata': {'default': '1x1'},
            'data': {'1x1': {'m': 1, 'n': 1}, '2x2': {'m': 2, 'n': 2}},
        },
    )

    grids = TilingConfig(path)

    assert grids.available == ('1x1', '2x2')
    assert grids.default == '1x1'
    assert grids.available_configs() == ['1x1', '2x2']
    assert grids.default_config() == '1x1'


_GRIDS = {'1x1': {'m': 1, 'n': 1}}

# Each a tiling.json no reader can use, with what the refusal names.
_BROKEN = {
    'not JSON': ('{"data": ', 'is not valid JSON'),
    'a list': ([], 'holds a list'),
    'no grids': ({'metadata': {'default': '1x1'}}, "no 'data' object"),
    'a grid with no n': (
        {'metadata': {'default': '1x1'}, 'data': {'1x1': {'m': 1}}},
        "grid '1x1' without a whole-number",
    ),
    'a grid that is no object': (
        {'metadata': {'default': '1x1'}, 'data': {'1x1': [1, 1]}},
        "grid '1x1' without a whole-number",
    ),
    'no default': ({'data': _GRIDS}, 'names no default grid'),
    'a default it does not offer': (
        {'metadata': {'default': '9x9'}, 'data': _GRIDS},
        'names no default grid',
    ),
}


def test_a_tiling_file_no_reader_can_use_is_refused_as_the_installation_s(tmp_path):
    """A grid left out of a broken file would be one the person cannot choose
    and a run could not lay out, so the file is refused whole where it is read."""
    wrong = {}
    for case, (document, words) in _BROKEN.items():
        path = tmp_path / case / 'tiling.json'
        path.parent.mkdir()
        path.write_text(document if isinstance(document, str) else json.dumps(document))
        try:
            TilingConfig(path)
        except InstallationFileError as refused:
            if refused.reason != 'installation_file_unusable' or words not in str(refused):
                wrong[case] = str(refused)
        else:
            wrong[case] = 'read'
    assert wrong == {}


def test_a_missing_tiling_file_answers_a_wire_client_as_the_installation_s(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from modules.scope_session import ScopeSession
    from rest.app import build_app
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(complete_settings(), simulate=True)
    try:
        # a stand-in by design: the shipped tiling.json is whole; this one is gone
        monkeypatch.setattr(
            session.scope.protocols, 'tiling_configs_path', lambda: tmp_path / 'tiling.json'
        )
        with TestClient(build_app(session)) as client:
            answer = client.post('/api/v1/scope/protocols/tiling_config')
    finally:
        session.shutdown()

    assert (answer.status_code, answer.json()['reason']) == (500, 'installation_file_unusable')
    assert 'tiling.json' in answer.json()['detail']
