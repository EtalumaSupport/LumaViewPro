# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol header row with no value cell is refused by name.

The reader took each header row's value as ``row[1]``, so a Version,
Period, Duration or Labware row cut down to its name (a hand edit, a
truncated save) raised IndexError out of the reader: a crash where the
reader exists to say what is wrong with the file.
"""

import pathlib

import pytest

from modules.protocol import Protocol, ProtocolFormatError
from tests.test_protocol_roundtrip import TILING_CONFIGS

REPO = pathlib.Path(__file__).resolve().parents[1]
SHIPPED_DEFAULT = REPO / 'data' / 'new_default_protocol.tsv'


def _load_with_row(tmp_path, name, replacement):
    lines = SHIPPED_DEFAULT.read_text().splitlines()
    (index,) = [i for i, line in enumerate(lines) if line.split('\t')[0] == name]
    lines[index] = replacement
    path = tmp_path / 'cut.tsv'
    path.write_text('\n'.join(lines) + '\n')
    return Protocol.from_file(file_path=path, tiling_configs_file_loc=TILING_CONFIGS)


def test_the_shipped_default_loads(tmp_path):
    assert _load_with_row(tmp_path, 'Version', 'Version\t5') is not None


@pytest.mark.parametrize('name', ['Version', 'Period', 'Duration', 'Labware'])
def test_a_row_with_no_value_cell_is_refused_naming_it(tmp_path, name):
    with pytest.raises(ProtocolFormatError, match=f"'{name}' row has no value"):
        _load_with_row(tmp_path, name, name)
