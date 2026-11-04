# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The tiling grids an installation offers: one set per session, handed out as a frozen record.

``TilingConfig`` is the record ``tiling_config()`` answers, in-process and
over the wire alike: ``available`` and ``default``, which no caller can
change. The grids themselves (``TilingGrids``) are read from tiling.json
once, by the scope at bring-up, and every protocol the session builds or
loads lays its tiles out on that one copy, so the grids a client is offered
and the grids ``apply_tiling`` accepts never differ.
"""

import dataclasses
import json

import pytest

from modules.exceptions import InstallationFileError
from modules.scope_session import ScopeSession
from modules.tiling_config import TilingConfig, TilingGrids
from tests.installation_fixtures import copy_installation_files
from tests.scope_fakes import home_sim_scope
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

    assert TilingGrids(path).config == TilingConfig(available=('1x1', '2x2'), default='1x1')


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
            TilingGrids(path)
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


def test_a_caller_cannot_change_the_grids_a_session_offers(tmp_path):
    _installation(tmp_path)
    session = ScopeSession.create(complete_settings(), source_path=str(tmp_path), simulate=True)
    try:
        offered = session.tiling_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            offered.available = ('9x9',)

        assert session.tiling_config() == offered
    finally:
        session.shutdown()


def test_a_grid_handed_out_is_a_copy(tmp_path):
    grids = TilingGrids(
        _write(tmp_path, {'metadata': {'default': '2x2'}, 'data': {'2x2': {'m': 2, 'n': 2}}})
    )

    grids.get_mxn_size('2x2')['m'] = 9

    assert grids.get_mxn_size('2x2') == {'m': 2, 'n': 2}


def test_a_protocol_built_after_an_edit_takes_the_grids_the_session_offers(tmp_path):
    """A tiling.json edited mid-session reaches no protocol: one session, one set of grids."""
    data = _installation(tmp_path)
    session = ScopeSession.create(complete_settings(), source_path=str(tmp_path), simulate=True)
    try:
        home_sim_scope(session.scope)
        session.settings['BF']['acquire'] = 'image'
        session.scope.motion.move_xy(x_um=20000, y_um=40000)
        _write(data, {'metadata': {'default': '1x1'}, 'data': {'1x1': {'m': 1, 'n': 1}}})
        protocol = session.create_empty_protocol()
        session.add_step(protocol, after_step_index=-1)

        session.apply_tiling(protocol, '2x2')

        assert '2x2' in session.tiling_config().available
        assert protocol.tiling() == '2x2'
    finally:
        session.shutdown()
