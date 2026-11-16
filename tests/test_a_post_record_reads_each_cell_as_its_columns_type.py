# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A post-processing record's Z-Slice is read as a whole number, or the record is refused.

The owner is ``ProtocolPostRecord.from_file``. Z-Slice was read through a
float and truncated: a hand-edited ``2.5`` loaded as slice 2, and a cell
that was no number raised a bare ``ValueError``. It is now read by the
protocol's whole-number reader, as the record's other number columns are,
so a cell it cannot hold refuses the record naming the image row and the
column; a blank is an output with no single slice, kept as -1.
"""

import datetime
import pathlib

import pytest

from _lumascope.modules import post_derivation
from _lumascope.modules.common_utils import PostFunction
from _lumascope.modules.protocol_post_record import ProtocolPostRecord

_STITCHED = post_derivation.Step(PostFunction.STITCHED, (('mode', 'quality'),)).text()


def _record_with_z_slice(folder: pathlib.Path, text: str) -> pathlib.Path:
    """A record of one output whose Z-Slice cell is ``text``, as a hand edit would leave it."""
    folder.mkdir(exist_ok=True)
    record_loc = folder / ProtocolPostRecord.DEFAULT_FILENAME
    record = ProtocolPostRecord(file_loc=record_loc)
    record.add_record(
        root_path=folder,
        file_path=pathlib.Path('first.tiff'),
        timestamp=datetime.datetime(2026, 11, 15, 9, 30, 0),
        name='first',
        label='',
        scan_count=0,
        x=1.0,
        y=2.0,
        z=3.0,
        z_slice=4,
        well='A1',
        color='BF',
        objective='10x',
        tile_group_id=-1,
        tile='',
        custom_step=False,
        channel_group_id=-1,
        derivation=_STITCHED,
    )
    record.complete()
    lines = record_loc.read_text(encoding='utf-8').splitlines()
    header_at = lines.index('Images') + 1
    column = lines[header_at].split('\t').index('Z-Slice')
    row = lines[header_at + 1].split('\t')
    assert row[column] == '4'
    row[column] = text
    lines[header_at + 1] = '\t'.join(row)
    record_loc.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return record_loc


@pytest.mark.parametrize('text', ['2.5', 'top'])
def test_a_z_slice_that_is_no_whole_number_refuses_the_record_naming_row_and_column(tmp_path, text):
    with pytest.raises(ValueError) as refused:
        ProtocolPostRecord.from_file(_record_with_z_slice(tmp_path, text))

    assert str(refused.value).startswith(f"image row 1 has Z-Slice '{text}', which is not")


def test_a_blank_z_slice_is_kept_as_minus_one_and_a_whole_one_as_written(tmp_path):
    blank = ProtocolPostRecord.from_file(_record_with_z_slice(tmp_path / 'blank', ''))
    whole = ProtocolPostRecord.from_file(_record_with_z_slice(tmp_path / 'whole', '7'))

    assert blank.records()['Z-Slice'].tolist() == [-1]
    assert whole.records()['Z-Slice'].tolist() == [7]
