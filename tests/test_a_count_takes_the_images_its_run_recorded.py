# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A cell count of a run folder counts the images the run's record names.

The count listed the folder and counted every image in it, so a Quick
Enhance copy, or any file no capture or build wrote, was counted beside the
captures as if the run had taken it. In a run's folder it now counts only
what the run record names (a capture) or the post-processing record does (a
build's output), and names each image it left out, with why; in a folder
with no record it counts every image but a Quick Enhance copy, told by its
recipe beside it. It reads the records and writes none, and an unreadable
record, or a protocol that is gone, is refused rather than counted around.
Rows come in the order the images were taken, then by name.

The runs are the simulator's, saved through the session as a run saves them.
"""

from __future__ import annotations

import datetime
import shutil

import numpy as np
import pytest

from _lumascope.modules import image_utils
from _lumascope.modules.channel_record import release_color
from _lumascope.modules.exceptions import LeftOutReason, PostProcessingRefusedError
from _lumascope.modules.lumascope_api.frame_record import TimestampSource
from _lumascope.modules.protocol_post_record import ProtocolPostRecord
from tests.count_runs import (
    RUN_RECORD,
    copy_run,
    count,
    count_scope,  # noqa: F401 -- the module's runs, a fixture
    images_in,
    results_rows,
    session,  # noqa: F401 -- a fixture
)


def test_a_run_folder_counts_its_captures_and_names_what_it_left_out(
    count_scope, session, tmp_path
):
    folder = copy_run(count_scope, 'scan', tmp_path)
    captures = images_in(folder)
    (copy,) = session.post_processing.enhance(folder / captures[0]).created
    shutil.copy(folder / captures[0], folder / 'stray.tiff')

    result = count(session, folder)

    assert sorted(row['file'] for row in results_rows(folder)) == captures
    assert {(str(left.path), left.reason) for left in result.left_out} == {
        (copy.output_path.name, LeftOutReason.NOT_IN_RECORD),
        ('stray.tiff', LeftOutReason.NOT_IN_RECORD),
    }
    assert copy.output_path.name in result.message and 'stray.tiff' in result.message


def test_a_folder_with_no_run_record_counts_all_but_a_quick_enhance_copy(
    count_scope, session, tmp_path
):
    still = count_scope[1]['scan'] / images_in(count_scope[1]['scan'])[0]
    folder = tmp_path / 'stills'
    folder.mkdir()
    shutil.copy(still, folder / still.name)
    (copy,) = session.post_processing.enhance(folder / still.name).created

    result = count(session, folder)

    assert [row['file'] for row in results_rows(folder)] == [still.name]
    assert [(str(left.path), left.reason) for left in result.left_out] == [
        (copy.output_path.name, LeftOutReason.QUICK_ENHANCE_COPY)
    ]


def test_a_jpg_run_folder_is_counted(count_scope, session, tmp_path):
    folder = copy_run(count_scope, 'jpg', tmp_path)
    captures = images_in(folder)
    assert captures and all(name.lower().endswith('.jpg') for name in captures)

    result = count(session, folder)

    assert result.counted == len(captures) and result.left_out == ()


def test_a_run_folder_whose_record_cannot_be_read_is_refused(count_scope, session, tmp_path):
    folder = copy_run(count_scope, 'scan', tmp_path)
    (folder / RUN_RECORD).write_text('not a run record\n', encoding='utf-8')

    with pytest.raises(PostProcessingRefusedError) as refused:
        count(session, folder)

    assert refused.value.reason == 'protocol_data_unreadable'
    assert not (folder / 'results.csv').exists()


def test_a_run_folder_whose_protocol_is_gone_is_refused_saying_so(count_scope, session, tmp_path):
    folder = copy_run(count_scope, 'scan', tmp_path)
    (protocol,) = [p for p in folder.glob('*.tsv') if p.name != RUN_RECORD]
    protocol.unlink()

    with pytest.raises(PostProcessingRefusedError) as refused:
        count(session, folder)

    assert refused.value.reason == 'protocol_data_unreadable'
    assert 'is not in the run folder' in str(refused.value)


def test_a_run_folder_whose_post_record_cannot_be_read_is_refused_and_keeps_it(
    count_scope, session, tmp_path
):
    folder = copy_run(count_scope, 'scan', tmp_path)
    post_record = folder / ProtocolPostRecord.DEFAULT_FILENAME
    post_record.write_text('not a post-processing record\n', encoding='utf-8')

    with pytest.raises(PostProcessingRefusedError) as refused:
        count(session, folder)

    assert refused.value.reason == 'protocol_data_unreadable'
    assert post_record.read_text(encoding='utf-8') == 'not a post-processing record\n'


def test_a_count_of_a_run_folder_writes_only_its_results(count_scope, session, tmp_path):
    """The count reads the run's records and writes none: no post record is
    started and nothing is moved aside."""
    folder = copy_run(count_scope, 'scan', tmp_path)
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}

    count(session, folder)

    after = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    assert set(after) - set(before) == {'results.csv', 'results_method.json'}
    assert {name: after[name] for name in before} == before


def test_rows_are_in_the_order_their_images_were_taken(session, tmp_path):
    """Names that sort the opposite way from their times come in time order."""
    for name, minute in (('a.tiff', 30), ('b.tiff', 10), ('c.tiff', 20)):
        taken = datetime.datetime(2026, 11, 8, 9, minute, tzinfo=datetime.UTC)
        image_utils.write_tiff(
            data=np.full((32, 32), 40, dtype=np.uint8),
            file_loc=tmp_path / name,
            metadata={
                'objective': {},
                'pixel_size_um': None,
                'channel': 'BF',
                'datetime': taken,
                'timestamp_iso': taken.isoformat(timespec='microseconds'),
                'timestamp_source': TimestampSource.CAMERA_FRAME_START.value,
            },
            ome=False,
            color=release_color('BF'),
            significant_bits=8,
            save_encoding='right_aligned',
            white_bits=None,
        )

    count(session, tmp_path)

    assert [row['file'] for row in results_rows(tmp_path)] == ['b.tiff', 'c.tiff', 'a.tiff']


def test_a_projections_time_is_its_scans_from_the_run_record(count_scope, session, tmp_path):
    """A projection states no time of its own; its row's time is the one
    its record row holds, the time Create Video stamps it with."""
    folder = copy_run(count_scope, 'zstack', tmp_path)
    (projection,) = session.post_processing.zproject(folder).artifact_paths
    recorded = ProtocolPostRecord.read_rows(folder / ProtocolPostRecord.DEFAULT_FILENAME)
    (row,) = [row for row in recorded.to_dict('records') if folder / row['Filepath'] == projection]
    assert image_utils.stated_frame_time(image_utils.read_file_record(projection)) is None

    count(session, projection.parent)

    (counted,) = [r for r in results_rows(projection.parent) if r['file'] == projection.name]
    assert datetime.datetime.fromisoformat(counted['time']) == row['Timestamp']
