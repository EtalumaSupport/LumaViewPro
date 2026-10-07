# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A loaded protocol is remembered for the next start only once the Session has opened it.

The GUI's Load wrote the remembered path itself, first, and then drew the
panel; a failure partway through left the panel holding the new protocol
and path beside widgets from the old one. ``ScopeSession.open_protocol``
makes every write a Load makes -- the plate, the Layer Settings, then the
path -- so the panel only draws, and a refused file leaves the remembered
path where it was.
"""

import pytest

from modules.protocol import ProtocolFormatError
from tests.test_labware_name_resolution import _protocol_file
from tests.test_loading_a_protocol_puts_the_scope_on_its_plate import (  # noqa: F401 -- pytest fixture
    FILE_PLATE,
    session,
)

PREVIOUS = '/data/previous.tsv'


def _remembered(session) -> str:
    return session.settings['protocol']['filepath']


def test_an_opened_protocol_is_on_its_plate_and_remembered(session, tmp_path):
    session.set_protocol_filepath(PREVIOUS)
    path = _protocol_file(tmp_path, FILE_PLATE)

    protocol = session.open_protocol(path)

    assert protocol.labware() == FILE_PLATE
    assert session.settings['protocol']['labware'] == FILE_PLATE
    assert _remembered(session) == str(path)


def test_a_refused_file_leaves_the_remembered_path(session, tmp_path):
    session.set_protocol_filepath(PREVIOUS)

    with pytest.raises(ProtocolFormatError):
        session.open_protocol(_protocol_file(tmp_path, 'A plate no catalogue has'))

    assert _remembered(session) == PREVIOUS


def test_refused_layer_settings_leave_the_remembered_path(session, tmp_path, monkeypatch):
    session.set_protocol_filepath(PREVIOUS)
    path = _protocol_file(tmp_path, FILE_PLATE)

    def _refuse(protocol):
        raise ProtocolFormatError('a Layer Settings cell is not a number', file=path)

    monkeypatch.setattr(session, 'apply_layer_settings', _refuse)

    with pytest.raises(ProtocolFormatError):
        session.open_protocol(path)

    assert _remembered(session) == PREVIOUS
