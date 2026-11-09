# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol states what it does, and it does not stimulate.

``Protocol.from_file`` and ``ProtocolAsksForStimulationError``: a file this
release saves carries no stimulation, neither a step's ``Stim_Config`` nor a
Layer Settings ``Stim_Enabled``. A file an earlier release saved carries
both; it loads when every entry is off, the entries unread, and is refused,
naming the step or the channel and the remedy, when one asks for
stimulation, since a run would stimulate nothing and say so nowhere. A
value that is neither True nor False refuses the file as a cell. Post-
processing reads such a file's run, since it never runs it.
"""

from __future__ import annotations

import csv
import io
import json
import pathlib

import pytest

from _lumascope.modules.protocol import (
    Protocol,
    ProtocolAsksForStimulationError,
    ProtocolFormatError,
    StoredNames,
)
from tests.protocol_fixtures import protocol_from_file
from tests.test_an_old_protocol_runs_the_leds_it_was_made_with import _session
from tests.test_going_to_a_step_is_the_sessions_move import _protocol, _step

_PLATE = '96 well microplate'
_OFF = {
    'enabled': False,
    'illumination_ma': 100,
    'frequency': 1,
    'pulse_width': 10,
    'pulse_count': 1,
}


def _saved(session, path: pathlib.Path, colors=('Ex405', 'Ex488')) -> pathlib.Path:
    """A protocol of one step per colour, saved by this release with each colour acquiring."""
    objective = session.scope.runtime_state.get_current_objective_id()
    steps = [
        _step(60.0, 40.0, 5000.0, objective) | {'Name': f'A1_{color}', 'channel': color}
        for color in colors
    ]
    for color in colors:
        session.set_channel_acquire(color, 'image')
    return session.save_protocol(_protocol(_PLATE, *steps), path)


def _as_v11(saved: pathlib.Path, stim_cells: list[dict], layer_stim: dict | None = None) -> None:
    """Rewrite ``saved`` as version 11 wrote it.

    Each step takes a Stim_Config cell, in order, and each Layer Settings
    row a Stim_Enabled cell: ``layer_stim``'s for its channel, else False.
    A column the writer already wrote takes the cell in place, so the file
    is the same shape whichever release saved it.
    """
    rows = list(csv.reader(io.StringIO(saved.read_text(encoding='utf-8')), delimiter='\t'))
    steps_header = next(at for at, row in enumerate(rows) if row[:1] == ['Name'])
    layer_header = next(at for at, row in enumerate(rows) if row[:1] == ['Layer'])

    def with_cell(row: list, header: list, column: str, cell: str) -> list:
        if column in header:
            at = header.index(column)
            return [*row[:at], cell, *row[at + 1 :]]
        return [*row, cell]

    layer_columns, step_columns = rows[layer_header], rows[steps_header]
    out_rows = []
    for at, row in enumerate(rows):
        if row[:1] == ['Version']:
            row = ['Version', '11']
        elif at in (layer_header, steps_header):
            column = 'Stim_Enabled' if at == layer_header else 'Stim_Config'
            row = row if column in row else [*row, column]
        elif layer_header < at < steps_header and len(row) > 1:
            on = str((layer_stim or {}).get(row[0], False))
            row = with_cell(row, layer_columns, 'Stim_Enabled', on)
        elif at > steps_header and row:
            cell = json.dumps(stim_cells[at - steps_header - 1])
            row = with_cell(row, step_columns, 'Stim_Config', cell)
        out_rows.append(row)
    out = io.StringIO()
    csv.writer(out, delimiter='\t', lineterminator='\n').writerows(out_rows)
    saved.write_text(out.getvalue(), encoding='utf-8')


@pytest.fixture
def session(tmp_path, monkeypatch):
    session = _session(tmp_path / 'unit', monkeypatch, microscope='LS850')
    yield session
    session.shutdown()


def test_a_saved_protocol_states_no_stimulation(session, tmp_path):
    """The file has no stimulation column, in its steps or its Layer
    Settings, and the step record no stimulation field."""
    saved = _saved(session, tmp_path / 'new.tsv')
    protocol = session.load_protocol(str(saved))

    assert 'Stim' not in saved.read_text()
    assert 'Stim_Config' not in Protocol.CURRENT_COLUMNS
    assert all('stim' not in field for field in vars(protocol.step_records()[0]))


def test_an_old_file_whose_stimulation_is_all_off_loads_without_it(session, tmp_path):
    """Version 11's entries, every one off and one naming a channel this
    release lacks, state nothing: the file loads, its steps and Layer
    Settings as saved, and nothing of the stimulation is kept."""
    saved = _saved(session, tmp_path / 'old.tsv')
    off = {'Ex405': _OFF, 'Ex488': _OFF, 'Ex999': _OFF}
    _as_v11(saved, [off, off])

    protocol = session.load_protocol(str(saved))

    assert list(protocol.steps()['channel']) == ['Ex405', 'Ex488']
    assert 'Stim_Config' not in protocol.steps().columns
    assert set(protocol.channel_settings()) == {'Ex405', 'Ex488'}


def test_an_old_file_whose_step_asks_for_stimulation_is_refused_naming_it_and_the_remedy(
    session, tmp_path
):
    """Both steps ask, the first on Ex550: the load is refused, naming step
    1 as the file spells it, the channel, how many steps ask, and how to put
    the file right."""
    saved = _saved(session, tmp_path / 'old.tsv')
    on = {'Ex405': _OFF, 'Ex550': {**_OFF, 'enabled': True}}
    _as_v11(saved, [on, on])

    with pytest.raises(ProtocolAsksForStimulationError) as refused:
        session.load_protocol(str(saved))

    words = str(refused.value)
    assert refused.value.reason == 'stimulation_not_performed'
    assert 'step 1 (A1_Ex405)' in words and 'Ex550' in words and '2 steps ask' in words
    assert 'set "enabled" to false' in words and 'delete the column' in words


def test_an_old_layer_settings_row_with_stimulation_on_is_refused_naming_the_channel(
    session, tmp_path
):
    """Every step's entries off, Ex488's Layer Settings row on: refused,
    naming Ex488 and its remedy."""
    saved = _saved(session, tmp_path / 'old.tsv')
    _as_v11(saved, [{'Ex405': _OFF}, {'Ex405': _OFF}], layer_stim={'Ex488': True})

    with pytest.raises(ProtocolAsksForStimulationError) as refused:
        session.load_protocol(str(saved))

    words = str(refused.value)
    assert refused.value.reason == 'stimulation_not_performed'
    assert 'Ex488' in words and 'Stim_Enabled to False' in words


# parametrize-justified: two cells, each refused by its own step of the reader.
@pytest.mark.parametrize(
    ('entry', 'said'),
    [
        pytest.param({'Ex405': {**_OFF, 'enabled': 'on'}}, 'is not True or False', id='enabled'),
        pytest.param({'Ex405': 5}, 'is not a config', id='entry'),
    ],
)
def test_a_stimulation_entry_that_is_not_an_entry_refuses_the_file_as_a_cell(
    session, tmp_path, entry, said
):
    """An ``enabled`` of ``'on'`` is not taken for on or off, and an entry
    that is not a config is no entry: the file is refused as a cell,
    naming the step and the channel."""
    saved = _saved(session, tmp_path / 'old.tsv')
    _as_v11(saved, [entry, {'Ex405': _OFF}])

    with pytest.raises(ProtocolFormatError) as refused:
        session.load_protocol(str(saved))

    words = str(refused.value)
    assert refused.value.reason == 'protocol_invalid'
    assert 'step 1 (A1_Ex405)' in words and 'Ex405' in words and said in words


def test_post_processing_reads_an_old_run_record_that_asks_for_stimulation(session, tmp_path):
    """The reader that never runs the protocol keeps it: the record reads,
    its steps as saved, the stimulation dropped; read to run, it is refused."""
    saved = _saved(session, tmp_path / 'old.tsv')
    on = {'Ex405': {**_OFF, 'enabled': True}}
    _as_v11(saved, [on, on], layer_stim={'Ex405': True})

    kept = protocol_from_file(saved, unit=StoredNames.KEPT)
    with pytest.raises(ProtocolAsksForStimulationError):
        protocol_from_file(saved)

    assert list(kept.steps()['channel']) == ['Ex405', 'Ex488']
    assert 'Stim_Config' not in kept.steps().columns
