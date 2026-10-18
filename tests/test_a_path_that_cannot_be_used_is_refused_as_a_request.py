# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A path the caller gave that cannot be used is refused as the request it is.

Every member that reads or writes a path the caller names answers the same
refusal, ``LiveFolderPathRefusedError``, with the reason that names what is
wrong with the path: ``not_found`` (nothing is there), ``not_a_file`` (a
folder where a file is expected), ``not_a_folder`` (a file where a folder
is expected) or ``wrong_extension`` (a file to write whose name does not end
in its kind's extension). Each member's rows, in Python and over REST, are
walked in ``tests/test_every_enrolled_member_answers_as_its_contract_says.py``;
this file holds what that walk does not: the cell-count reader and writer,
which are no members, a writer making a missing folder where it was sent,
and every writer writing exactly the name it was given.

A writer once added its extension to a name without it, after a Save As
dialog had checked that bare name for a file to replace: typing ``plate1``
wrote over ``plate1.tsv`` with no prompt. The name given is now the name
written, so the name a dialog checked is the one that is replaced.
"""

from __future__ import annotations

import pytest

from modules.exceptions import LiveFolderPathRefusedError
from modules.post_processing import (
    default_cell_count_method,
    read_cell_count_image,
    save_cell_count_method,
)
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
    (folder / 'results.csv').write_text('count,mean_area\n3,1.5\n5,2.5\n')
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


def _writers(session):
    """Each writer of a named file, with the extension its kind takes."""
    protocol = session.new_protocol()
    results = f'{session.settings["live_folder"]}/results.csv'
    return {
        'PostProcessingAPI.graph': (
            '.png',
            lambda p: session.post_processing.graph(results, p, x='count', y='mean_area'),
        ),
        'ScopeSession.save_protocol': ('.tsv', lambda p: session.save_protocol(protocol, p)),
        'Protocol.to_file': ('.tsv', protocol.to_file),
        'save_cell_count_method': (
            '.json',
            lambda p: save_cell_count_method(default_cell_count_method(), p),
        ),
    }


def test_each_writer_makes_a_missing_folder_where_it_was_sent(session, tmp_path):
    missing = []
    for writer, (extension, write) in _writers(session).items():
        target = tmp_path / writer / 'new' / 'deeper' / f'plate{extension}'
        write(target)
        if not target.is_file():
            missing.append(f'{writer} wrote nothing at {target}')
    assert missing == []


def test_each_writer_writes_exactly_the_name_it_was_given(session, tmp_path):
    wrong = []
    for writer, (extension, write) in _writers(session).items():
        folder = tmp_path / writer
        folder.mkdir()
        for name in (f'plate1{extension}', f'plate2{extension.upper()}'):
            write(folder / name)
        written = sorted(p.name for p in folder.iterdir())
        if written != sorted([f'plate1{extension}', f'plate2{extension.upper()}']):
            wrong.append(f'{writer} wrote {written}')
    assert wrong == []


def test_a_name_without_its_kinds_extension_is_refused_and_nothing_is_written(session, tmp_path):
    wrong = []
    for writer, (extension, write) in _writers(session).items():
        folder = tmp_path / writer
        folder.mkdir()
        # The file a dialog that checked the bare name would not have asked about.
        for stem in ('plate1', '$plate1'):
            (folder / f'{stem}{extension}').write_text('kept')
        before = {p.name: p.read_text() for p in folder.iterdir()}
        for name in ('plate1', '$plate1', 'plate1.txt', 'exp_2026.10.14'):
            try:
                write(folder / name)
            except LiveFolderPathRefusedError as refused:
                if refused.reason != 'wrong_extension':
                    wrong.append(f'{writer}({name!r}) refused {refused.reason!r}')
            else:
                wrong.append(f'{writer}({name!r}) was not refused')
        after = {p.name: p.read_text() for p in folder.iterdir()}
        if after != before:
            wrong.append(f'{writer} changed the folder: {sorted(after)}')
    assert wrong == []
