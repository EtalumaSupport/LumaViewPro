# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A table LumaViewPro wrote reads back exactly as it was written.

Every table writer is ``csv.writer``, which writes each cell as its text.
The readers parsed with ``pd.read_csv``, which guesses: the cells ``NA``,
``None``, ``nan``, ``null``, ``NaN`` and ``NULL`` became missing values and
then ``''``, a NUL ended its cell, and its quoting was not the writer's. A
step renamed ``NA`` -- a name the rename accepts on every client -- saved as
typed and reloaded as the step's machine name. The tables are now read by
``read_table``, ``csv.reader`` and nothing else, and each reader types its
own columns; a file no LumaViewPro writer produces is refused naming it.
"""

import pytest

from modules.common_utils import read_table
from modules.protocol import Protocol, ProtocolFormatError
from tests.test_protocol_roundtrip import TILING_CONFIGS, _build_protocol, _make_step

NA_WORDS = ['NA', 'None', 'nan', 'null', 'NaN', 'NULL', 'N/A', '#N/A', '']


@pytest.mark.parametrize('text', NA_WORDS)
def test_every_cell_is_the_text_written(text):
    header, rows = read_table(f'Name\tX\n{text}\t1\n', sep='\t')
    assert header == ['Name', 'X']
    assert rows == [[text, '1']]


@pytest.mark.parametrize(
    ('text', 'words'),
    [
        ('Name\tX\nA\x001\t1\n', 'NUL'),
        ('Name\tX\n"A1\t1\nB1\t2\n', 'not a well-formed table'),
        ('Name\tX\nA1\t1\t2\n', 'row 2 of its table has 3 cells where the header has 2'),
        ('\n\n', 'no header row'),
    ],
    ids=['nul', 'unbalanced-quote', 'extra-cell', 'empty'],
)
def test_a_table_no_writer_produces_is_refused(text, words):
    with pytest.raises(ValueError, match=words):
        read_table(text, sep='\t')


@pytest.mark.parametrize('name', ['NA', 'None', 'nan', 'null', 'NaN', 'NULL'])
def test_a_step_renamed_to_a_missing_value_word_reloads_with_that_name(tmp_path, name):
    protocol = _build_protocol([_make_step(well='A1')])
    protocol.modify_name(0, name)
    path = tmp_path / 'protocol.tsv'
    protocol.to_file(path)

    step = Protocol.from_file(path, TILING_CONFIGS).step(idx=0)

    assert (step['Label'], step['Name']) == (name, f'{name}_BF_Z0')


def _saved_with_well_cell(tmp_path, text):
    path = tmp_path / 'protocol.tsv'
    _build_protocol([_make_step(well='A1')]).to_file(path)
    path.write_text(path.read_text().replace('\tA1\t', f'\t{text}\t'))
    return path


@pytest.mark.parametrize('runnable', [True, False])
@pytest.mark.parametrize(
    ('cell', 'words'), [('A\x001', 'NUL'), ('"A1', 'not a well-formed table')], ids=['nul', 'quote']
)
def test_a_protocol_whose_step_cell_no_writer_produces_is_refused_naming_it(
    tmp_path, runnable, cell, words
):
    path = _saved_with_well_cell(tmp_path, cell)

    with pytest.raises(ProtocolFormatError, match=words) as refused:
        Protocol.from_file(path, TILING_CONFIGS, runnable=runnable)

    assert refused.value.file == path
