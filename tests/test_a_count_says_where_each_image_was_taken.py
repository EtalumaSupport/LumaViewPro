# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A cell count's row says where and when its image was taken.

``results.csv`` held only the file, its time and the measurements, so the
rows of a plate run (wells, channels, scans and tiles) could not be told
apart without parsing file names. Each row now carries ``well``,
``channel``, ``scan``, ``tile`` and ``z``. The well, channel and Z are the
image's own, never the step's plan, which a moved or inserted step leaves
stale or empty and autofocus moves from; an image that states no channel (a
JPG) takes its record's, and one that states no well or Z leaves it blank.
The scan (from 0, as the file names number it) and the tile are the run
record's, blank in a folder with no record. The header and every row are
one list of columns.
"""

from __future__ import annotations

import csv
import shutil

from _lumascope.modules import image_utils
from _lumascope.modules.post_processing import RESULTS_COLUMNS
from _lumascope.modules.protocol_execution_record import ProtocolExecutionRecord
from tests.count_runs import (
    RUN_RECORD,
    copy_run,
    count,
    count_scope,  # noqa: F401 -- the module's runs, a fixture
    images_in,
    results_rows,
    session,  # noqa: F401 -- a fixture
)


def _stated(path):
    """The well, channel and Z the image at *path* states, as the results write them."""
    facts = image_utils.read_file_record(path)
    plate = facts.get('plate') or {}
    z = facts.get('z_pos_um')
    return (plate.get('well_label') or '', facts.get('channel') or '', '' if z is None else str(z))


def test_each_row_says_the_well_channel_and_z_its_image_states_and_its_scan_and_tile(
    count_scope, session, tmp_path
):
    folder = copy_run(count_scope, 'plate', tmp_path)
    record = ProtocolExecutionRecord.from_file(file_path=folder / RUN_RECORD)

    count(session, folder)

    rows = results_rows(folder)
    assert sorted(row['file'] for row in rows) == images_in(folder)
    for row in rows:
        assert (row['well'], row['channel'], row['z']) == _stated(folder / row['file'])
        assert row['scan'] == str(record.get_data_from_filename(row['file'])['Scan Count'])
        assert f'_T{row["tile"]}_' in row['file'] or row['file'].endswith(f'_T{row["tile"]}')
    # Each tile states the well it was captured in, so tiles at a well's
    # edge name the wells they landed in, not the position's.
    assert len({row['well'] for row in rows}) >= 2
    assert {row['scan'] for row in rows} == {'0', '1'}
    assert len({row['tile'] for row in rows}) == 4
    assert len({row['channel'] for row in rows}) == 2


def test_a_step_whose_plan_moved_after_the_run_still_reads_where_its_image_was_taken(
    count_scope, session, tmp_path
):
    """The run's protocol copy is the steps' plan: a Z autofocus moved from,
    a well left empty on an inserted step or stale on a moved one. A plan
    edited after the run changes no row's well or Z."""
    folder = copy_run(count_scope, 'plate', tmp_path)
    (protocol,) = [p for p in folder.glob('*.tsv') if p.name != RUN_RECORD]
    lines = protocol.read_text(encoding='utf-8').splitlines(keepends=True)
    header_at = next(i for i, line in enumerate(lines) if line.startswith('Name\t'))
    columns = lines[header_at].rstrip('\n').split('\t')
    z_at, well_at = columns.index('Z'), columns.index('Well')
    for i in range(header_at + 1, len(lines)):
        cells = lines[i].rstrip('\n').split('\t')
        cells[z_at], cells[well_at] = '1234.5', 'H12'
        lines[i] = '\t'.join(cells) + '\n'
    protocol.write_text(''.join(lines), encoding='utf-8')

    count(session, folder)

    for row in results_rows(folder):
        assert (row['well'], row['channel'], row['z']) == _stated(folder / row['file'])
        assert row['well'] != 'H12' and row['z'] != '1234.5'


def test_a_jpg_runs_rows_take_the_records_channel_and_state_no_well_or_z(
    count_scope, session, tmp_path
):
    folder = copy_run(count_scope, 'jpg', tmp_path)

    count(session, folder)

    rows = results_rows(folder)
    assert rows
    for row in rows:
        assert row['channel'] and row['channel'] in row['file']
        assert row['well'] == '' and row['z'] == ''
        assert row['scan'] == '0'


def test_a_projections_row_states_no_z(count_scope, session, tmp_path):
    """A projection of slices at different Z states no Z of its own."""
    folder = copy_run(count_scope, 'zstack', tmp_path)
    (projection,) = session.post_processing.zproject(folder).artifact_paths

    count(session, projection.parent)

    (row,) = [r for r in results_rows(projection.parent) if r['file'] == projection.name]
    assert row['z'] == ''
    assert row['channel'] == 'BF'


def test_a_folder_with_no_record_states_no_scan_or_tile(count_scope, session, tmp_path):
    run = count_scope[1]['plate']
    folder = tmp_path / 'stills'
    folder.mkdir()
    for name in images_in(run)[:2]:
        shutil.copy(run / name, folder / name)

    count(session, folder)

    for row in results_rows(folder):
        assert row['scan'] == '' and row['tile'] == ''
        assert (row['well'], row['channel'], row['z']) == _stated(folder / row['file'])


def test_the_header_and_every_row_have_the_same_columns_in_order(count_scope, session, tmp_path):
    folder = copy_run(count_scope, 'scan', tmp_path)

    count(session, folder)

    with open(folder / 'results.csv', newline='', encoding='utf-8') as f:
        header, *rows = list(csv.reader(f))
    assert tuple(header) == RESULTS_COLUMNS
    assert header[:7] == ['file', 'well', 'channel', 'scan', 'tile', 'z', 'time']
    assert rows and all(len(row) == len(header) for row in rows)
