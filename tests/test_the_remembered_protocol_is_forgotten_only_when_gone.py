# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The remembered protocol is forgotten only when its file is gone or cannot be read.

The GUI opened the protocol the last start left behind and decided the
remembered path's fate itself, clearing it only when no file was there: a
file that was there but could not be read kept its path, and the next start
failed on it again. ``ScopeSession.open_remembered_protocol`` owns the rule
beside the path's one writer: a missing or unreadable file is forgotten, and
a refusal (a plate the catalogue lacks, a malformed file, glass the turret
cannot place) keeps the path, because the file is real and what is wrong can
be put right.
"""

import logging

import pytest

from modules.exceptions import ProtocolNotLoadedError
from modules.protocol import ProtocolFormatError
from tests.test_labware_name_resolution import _protocol_file
from tests.test_loading_a_protocol_puts_the_scope_on_its_plate import (  # noqa: F401 -- pytest fixture
    session,
)


def _remember(session, path) -> None:
    session.set_protocol_filepath(str(path))


def _remembered(session) -> str:
    return session.settings['protocol']['filepath']


def test_no_remembered_path_opens_nothing(session):
    _remember(session, '')

    assert session.open_remembered_protocol() is None
    assert _remembered(session) == ''


def test_a_remembered_file_that_is_gone_is_forgotten_at_info(session, tmp_path, caplog):
    _remember(session, tmp_path / 'deleted.tsv')

    with caplog.at_level(logging.INFO):
        assert session.open_remembered_protocol() is None

    assert _remembered(session) == ''
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_remembered_file_that_cannot_be_read_is_forgotten_and_raised(session, tmp_path):
    unreadable = tmp_path / 'folder.tsv'
    unreadable.mkdir()
    _remember(session, unreadable)

    with pytest.raises(ProtocolNotLoadedError):
        session.open_remembered_protocol()

    assert _remembered(session) == ''


@pytest.mark.parametrize(
    'contents',
    [None, 'not a protocol\n'],
    ids=['plate-not-in-catalogue', 'malformed'],
)
def test_a_refused_remembered_file_keeps_its_path(session, tmp_path, contents):
    path = _protocol_file(tmp_path, 'A plate no catalogue has')
    if contents is not None:
        path.write_text(contents)
    _remember(session, path)

    with pytest.raises(ProtocolFormatError):
        session.open_remembered_protocol()

    assert _remembered(session) == str(path)


def test_a_remembered_file_that_loads_is_opened_and_kept(session, tmp_path):
    path = _protocol_file(tmp_path, '6 well microplate')
    _remember(session, path)

    protocol = session.open_remembered_protocol()

    assert protocol is not None and protocol.labware() == '6 well microplate'
    assert _remembered(session) == str(path)
