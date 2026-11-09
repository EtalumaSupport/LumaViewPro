# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol header row the csv module cannot read is refused naming the file.

The step table was read strictly and its ``csv.Error`` refused as
``ProtocolFormatError``, but the header block (the Version, Period,
Duration, Labware and Capture Root rows, the search for 'Steps' and the
Layer Settings block) was read by a bare ``csv.reader``: a cell over the
csv field limit escaped as a raw ``_csv.Error``, an untyped crash where
every host reports a protocol refusal, and a cell with text after its
closing quote was taken as written where the step table refuses it.
"""

import csv
import pathlib

import pytest

from _lumascope.modules.protocol import ProtocolFormatError
from tests.protocol_fixtures import protocol_from_file

REPO = pathlib.Path(__file__).resolve().parents[1]
SHIPPED_DEFAULT = REPO / 'data' / 'new_default_protocol.tsv'

OVER_THE_FIELD_LIMIT = 'x' * (csv.field_size_limit() + 1)


def _with_version_cell_over_the_limit(lines):
    lines[1] = f'Version\t{OVER_THE_FIELD_LIMIT}'


def _with_layer_settings_cell_over_the_limit(lines):
    capture_root = lines.index('Capture Root')
    lines[capture_root + 1 : capture_root + 1] = [
        'Layer Settings',
        'Layer\tGain',
        f'BF\t{OVER_THE_FIELD_LIMIT}',
    ]


def _with_text_after_a_closing_quote(lines):
    (labware,) = [i for i, line in enumerate(lines) if line.startswith('Labware\t')]
    lines[labware] = 'Labware\t"96 well microplate"x'


# parametrize-justified: three rows the header reader reaches at different
# places (a single-row read, the Layer Settings loop) or by a different
# csv rule (the field limit, strict quoting).
@pytest.mark.parametrize(
    'damage',
    [
        _with_version_cell_over_the_limit,
        _with_layer_settings_cell_over_the_limit,
        _with_text_after_a_closing_quote,
    ],
)
def test_an_unreadable_header_row_is_refused_naming_the_file(tmp_path, damage):
    lines = SHIPPED_DEFAULT.read_text(encoding='utf-8').splitlines()
    damage(lines)
    path = tmp_path / 'damaged.tsv'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    with pytest.raises(ProtocolFormatError) as refused:
        protocol_from_file(file_path=path)

    assert type(refused.value) is ProtocolFormatError
    assert refused.value.reason == 'protocol_invalid'
    assert refused.value.file == path
    assert 'its header block is not well-formed' in str(refused.value)
