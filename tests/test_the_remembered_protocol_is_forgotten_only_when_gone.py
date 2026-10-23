# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The remembered protocol is forgotten only when its file is gone or cannot be read.

The GUI opened the protocol the last start left behind and decided the
remembered path's fate itself, clearing it only when no file was there: a
file that was there but could not be read kept its path, and the next start
failed on it again. ``ScopeSession.open_remembered_protocol`` owns the rule
beside the path's one writer: a path with no file at it (nothing, or a
folder) is forgotten quietly, an unreadable file is forgotten and raised, and
a refusal (a plate the catalogue lacks, a malformed file, glass the turret
cannot place) keeps the path, because the file is real and what is wrong can
be put right.
"""

import logging

import pytest

from modules.exceptions import ProtocolNotLoadedError
import modules.protocol as protocol_module
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


@pytest.mark.parametrize('kind', ['deleted', 'a folder'])
def test_a_remembered_path_with_no_file_is_forgotten_at_info(session, tmp_path, caplog, kind):
    path = tmp_path / 'protocol.tsv'
    if kind == 'a folder':
        path.mkdir()
    _remember(session, path)

    with caplog.at_level(logging.INFO):
        assert session.open_remembered_protocol() is None

    assert _remembered(session) == ''
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_remembered_file_that_cannot_be_read_is_forgotten_and_raised(
    session, tmp_path, monkeypatch
):
    path = _protocol_file(tmp_path, '6 well microplate')
    _remember(session, path)

    # a stand-in by design: a disk that fails a read, which the simulator
    # cannot produce; the reader opens the file through its module's ``open``.
    def failing_open(*args, **kwargs):
        raise PermissionError(13, 'Permission denied')

    monkeypatch.setattr(protocol_module, 'open', failing_open, raising=False)

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

    assert protocol is not None and protocol.labware_id() == '6 well microplate'
    assert _remembered(session) == str(path)
