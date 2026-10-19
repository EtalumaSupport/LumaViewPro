# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's records read back every timestamp they wrote, a whole second included.

Post-processing reads a run's frames through its record, and its outputs
through the post-processing record. A time that falls on a whole second is
a row like any other: each record is read back whole, and each row carries
the time it was written with.
"""

import datetime
import pathlib

from modules import post_derivation
from modules.common_utils import PostFunction
from modules.protocol_execution_record import ProtocolExecutionRecord
from modules.protocol_post_record import ProtocolPostRecord

# A stitched output's derivation, as a Quality stitch of raw tiles records it.
_STITCHED = post_derivation.Step(PostFunction.STITCHED, (('mode', 'quality'),)).text()

WRITTEN = (
    datetime.datetime(2026, 10, 18, 9, 30, 0),
    datetime.datetime(2026, 10, 18, 9, 30, 0, 250000),
)


def test_a_run_record_row_written_on_a_whole_second_reads_back_with_its_time(tmp_path):
    outfile = tmp_path / ProtocolExecutionRecord.DEFAULT_FILENAME
    record = ProtocolExecutionRecord(protocol_file_loc=tmp_path / 'protocol.tsv', outfile=outfile)
    for index, timestamp in enumerate(WRITTEN):
        record.add_step(
            capture_result_file_name=f'frame_{index}.tiff',
            step_name='S1',
            step_index=0,
            scan_count=index,
            timestamp=timestamp,
        )

    read = ProtocolExecutionRecord.from_file(outfile)

    assert [entry['Timestamp'] for entry in read.saved_entries()] == list(WRITTEN)


def test_a_post_record_row_written_on_a_whole_second_reads_back_with_its_time(tmp_path):
    outfile = tmp_path / ProtocolPostRecord.DEFAULT_FILENAME
    record = ProtocolPostRecord(file_loc=outfile)
    for index, timestamp in enumerate(WRITTEN):
        record.add_record(
            root_path=tmp_path,
            file_path=pathlib.Path(f'stitched_{index}.tiff'),
            timestamp=timestamp,
            name=f'A1_BF_stitched_{index}',
            label='',
            scan_count=index,
            x=1.0,
            y=2.0,
            z=3.0,
            z_slice=-1,
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

    read = ProtocolPostRecord.from_file(outfile)

    assert list(read.records()['Timestamp']) == list(WRITTEN)
