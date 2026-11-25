# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run record that lost rows says why each was lost, never "check the log".

``ProtocolExecutionRecord.complete`` raises ``RecordIncompleteError`` when
fewer rows were written than captures attempted. A row is lost two ways it
knows of: a write that raised, told in the system's words, and a row not
written once the save drive was declared unresponsive.
"""

import datetime

import pytest

from _lumascope.modules.exceptions import RecordIncompleteError, system_words
from _lumascope.modules.protocol_execution_record import ProtocolExecutionRecord


def _add(record, step_name):
    record.note_capture_attempt()
    record.add_step(
        capture_result_file_name='image.tiff',
        step_name=step_name,
        step_index=0,
        scan_count=0,
        timestamp=datetime.datetime(2026, 11, 24, 12, 0, 0),
    )


def test_a_row_whose_write_raised_is_told_in_the_systems_words(tmp_path):
    outfile = tmp_path / 'protocol_record.tsv'
    record = ProtocolExecutionRecord(protocol_file_loc=tmp_path / 'p.tsv', outfile=outfile)
    # The record's file becomes a folder: every later append is refused by the system.
    outfile.unlink()
    outfile.mkdir()
    try:
        with open(outfile, 'a', encoding='utf-8'):
            pass
    except OSError as refused:
        system = system_words(refused).rstrip('.')
    else:
        pytest.fail('a folder opened for append (precondition)')
    _add(record, 'C1')
    _add(record, 'C2')

    with pytest.raises(RecordIncompleteError) as short:
        record.complete()

    said = str(short.value)
    assert f'{system} (2: C1, C2)' in said, said
    assert 'log' not in said, said


def test_a_row_not_written_to_an_unresponsive_drive_says_so(tmp_path):
    record = ProtocolExecutionRecord(
        protocol_file_loc=tmp_path / 'p.tsv', outfile=tmp_path / 'protocol_record.tsv'
    )
    _add(record, 'C1')
    record.mark_target_unresponsive()
    _add(record, 'C2')

    with pytest.raises(RecordIncompleteError) as short:
        record.complete()

    said = str(short.value)
    assert '1 of 2 captures' in said, said
    assert 'the save drive stopped responding (C2)' in said, said
    assert 'log' not in said, said
