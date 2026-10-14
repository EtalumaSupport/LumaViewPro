# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each row of a run's record that names a file names it from the run folder.

Post-processing finds a run's frames through its record: a row's file name
is where the frame is, from the folder holding the record. With a folder per
channel a still is recorded under its channel's folder; a video saved as
frames is too, its frames folder written there and named from the run folder.
Driven end to end on the simulated scope.
"""

import csv
import pathlib

from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 60.0


def _video(name, index):
    step = _step(name, index, x=20.0, gain=1.0)
    step['Acquire'] = 'video'
    step['Video Config'] = {'duration': 1.0, 'fps': 5}
    return step


def _file_rows(run_parent: pathlib.Path) -> tuple[pathlib.Path, list[str]]:
    """The record's folder, and every row naming a file (a frame count past 0)."""
    (record,) = run_parent.rglob('protocol_record.tsv')
    with open(record, newline='') as fp:
        rows = list(csv.reader(fp, delimiter='\t'))[4:]
    return record.parent, [row[0] for row in rows if int(row[5]) > 0]


def test_with_a_folder_per_channel_every_file_row_names_where_its_file_is(tmp_path):
    settings = headless_settings(tmp_path)
    settings['separate_folder_per_channel'] = True
    settings['video_as_frames'] = True
    with open_composite_session(settings) as (_session, runner):
        pending = runner.start_single_scan(
            protocol=_protocol([_step('S1', 0, x=20.0, gain=1.0), _video('V1', 1)]),
            parent_dir=str(tmp_path / 'runs'),
        )
        assert pending.result(timeout=WAIT_S).status == 'completed'
        assert pending.files.result(timeout=WAIT_S) is not None

    run_dir, names = _file_rows(tmp_path / 'runs')

    assert len(names) == 2, names
    assert [name for name in names if not (run_dir / name).exists()] == []
