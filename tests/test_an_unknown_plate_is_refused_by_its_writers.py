# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A plate the catalogue does not have is refused by every writer of the selection.

Every plate position is converted through the selected plate, so a
different plate's geometry would put every well in the wrong place while
the protocol reads as if it ran normally. A stored plate the catalogue
cannot resolve is replaced at bring-up by the shipped one
(test_an_unusable_stored_plate_is_replaced_at_bring_up.py); after it, the
settings writers refuse an unknown plate.
"""

import pytest

from modules.exceptions import SettingRefusedError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


def test_the_shipped_template_plate_brings_up(tmp_path):
    # The plate bring-up puts in place of an unusable stored one: it only
    # works if the template's own plate resolves.
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        assert session.scope.runtime_state.get_labware() is not None
    finally:
        session.shutdown()


class TestThePlateHasOneWriter:
    @pytest.fixture
    def session(self, tmp_path):
        s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        yield s
        s.shutdown()

    def test_the_settings_writer_refuses_the_plate_naming_its_member(self, session):
        stored = dict(session.settings['protocol'])
        with pytest.raises(SettingRefusedError, match='select_labware'):
            session.update_settings('protocol.labware', 'nonexistent')
        assert session.settings['protocol'] == stored

    def test_the_protocol_block_is_not_written_whole(self, session):
        stored = session.settings['protocol']
        with pytest.raises(SettingRefusedError, match='block'):
            session.update_settings('protocol', {**stored, 'labware': 'nonexistent'})
        assert session.settings['protocol'] is stored
