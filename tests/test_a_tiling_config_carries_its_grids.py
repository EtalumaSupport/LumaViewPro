# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The tiling grids an installation offers are the record's own data.

``TilingConfig`` crosses the wire as a record, and a record has no
address, so its two methods are in-process; a wire client holds its data.
Until now it had none: the grids were reachable only through the
methods. ``available`` and ``default`` are read from tiling.json once, at
construction, and the in-process methods answer from them.
"""

import json

import pytest

from modules.exceptions import InstallationFileError
from modules.scope_session import ScopeSession
from modules.tiling_config import TilingConfig
from tests.installation_fixtures import copy_installation_files
from tests.settings_fixtures import complete_settings


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
    'a default that is no name': (
        {'metadata': {'default': ['1x1']}, 'data': _GRIDS},
        'names no default grid',
    ),
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


def _installation(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    copy_installation_files(data)
    return data


def test_a_tiling_file_no_reader_can_use_refuses_the_bring_up_naming_it(tmp_path):
    data = _installation(tmp_path)
    (data / 'tiling.json').write_text('{"data": ', encoding='utf-8')

    with pytest.raises(InstallationFileError) as refused:
        ScopeSession.create(complete_settings(), source_path=str(tmp_path), simulate=True)

    assert 'tiling.json' in str(refused.value)


def test_the_grids_a_session_offers_are_the_ones_its_scope_read_at_bring_up(tmp_path):
    """Read once with the other catalogues: an edit while the program runs
    is seen at the next start, so every caller in one session agrees."""
    data = _installation(tmp_path)
    session = ScopeSession.create(complete_settings(), source_path=str(tmp_path), simulate=True)
    try:
        offered = session.tiling_config().available
        _write(data, {'data': {'13x17': {'m': 13, 'n': 17}}, 'metadata': {'default': '13x17'}})

        assert session.tiling_config().available == offered
    finally:
        session.shutdown()

    assert '13x17' not in offered
