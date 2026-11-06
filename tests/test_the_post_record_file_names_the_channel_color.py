# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The post-processing record's file names the channel column 'Color'; in
memory it is 'channel'.

What is saved stays as saved: a record written by this release keeps the
column every earlier release wrote, so an earlier release reads it, and the
record read back names the column as the protocol's steps do.
"""

import datetime
import pathlib

from _lumascope.modules import post_derivation
from _lumascope.modules.common_utils import PostFunction
from _lumascope.modules.protocol_post_record import ProtocolPostRecord
from tests.stock_channels import LED1

_STITCHED = post_derivation.Step(PostFunction.STITCHED, (('mode', 'quality'),)).text()


def test_a_post_record_writes_color_and_reads_back_channel(tmp_path):
    outfile = tmp_path / ProtocolPostRecord.DEFAULT_FILENAME
    record = ProtocolPostRecord(file_loc=outfile)
    record.add_record(
        root_path=tmp_path,
        file_path=pathlib.Path('stitched.tiff'),
        timestamp=datetime.datetime(2026, 11, 6, 9, 30, 0),
        name=f'A1_{LED1}_stitched',
        label='',
        scan_count=0,
        x=1.0,
        y=2.0,
        z=3.0,
        z_slice=-1,
        well='A1',
        color=LED1,
        objective='10x',
        tile_group_id=-1,
        tile='',
        custom_step=False,
        channel_group_id=-1,
        derivation=_STITCHED,
    )
    record.complete()

    lines = outfile.read_text().splitlines()
    header = lines[lines.index('Images') + 1].split('\t')
    assert header == list(ProtocolPostRecord.COLUMNS)
    assert 'Color' in header

    records = ProtocolPostRecord.from_file(outfile).records()
    assert 'channel' in records.columns and 'Color' not in records.columns
    assert list(records['channel']) == [LED1]
