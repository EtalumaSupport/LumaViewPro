# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A path the caller gave that cannot be used is refused as the request it is.

Every member that reads or writes a path the caller names answers the same
refusal, ``LiveFolderPathRefusedError``, with the reason that names what is
wrong with the path: ``not_found`` (nothing is there), ``not_a_file`` (a
folder where a file is expected) or ``not_a_folder`` (a file where a folder
is expected). Each member's rows, in Python and over REST, are walked in
``tests/test_every_enrolled_member_answers_as_its_contract_says.py``; this
file holds what that walk does not: the cell-count reader, which is no
member, and a writer making a missing folder where it was sent.
"""

from __future__ import annotations

import pytest

from modules.exceptions import LiveFolderPathRefusedError
from modules.post_processing import read_cell_count_image
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def live(tmp_path):
    folder = tmp_path / 'live'
    folder.mkdir()
    (folder / 'a_file.txt').write_text('not an image, not a folder')
    (folder / 'a_folder').mkdir()
    (folder / 'a_folder.tsv').mkdir()
    return folder


@pytest.fixture
def session(live):
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    # A protocol is made on a homed scope, with steps for a layer that acquires.
    home_sim_scope(s.scope)
    s.set_layer_acquire('BF', 'image')
    yield s
    s.shutdown()


def test_the_cell_count_reader_refuses_a_missing_file(live):
    with pytest.raises(LiveFolderPathRefusedError) as refused:
        read_cell_count_image(live / 'nothing_here' / 'x.tif')

    assert refused.value.reason == 'not_found'


def test_a_protocol_saved_into_a_missing_folder_is_written_there(session, live):
    target = live / 'new' / 'deeper' / 'plate'

    written = session.save_protocol(session.new_protocol(), target)

    assert written == target.with_name('plate.tsv')
    assert session.load_protocol(written).labware() == session.new_protocol().labware()
