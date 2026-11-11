# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A post-processing record writes its folder's file only when a build records an output.

The owner is ``ProtocolPostRecord``: a build loads it before it knows
whether it will build anything, so loading it, and completing it with
nothing recorded, leave the folder's file as they found it: none where
there was none, an older version's bytes unchanged. The first output
recorded writes the file in the current version, every row once.
"""

import datetime
import pathlib

from _lumascope.modules import post_derivation
from _lumascope.modules.common_utils import PostFunction
from _lumascope.modules.protocol_post_record import ProtocolPostRecord

_STITCHED = post_derivation.Step(PostFunction.STITCHED, (('mode', 'quality'),)).text()


def _record_output(record: ProtocolPostRecord, root: pathlib.Path, name: str) -> None:
    record.add_record(
        root_path=root,
        file_path=pathlib.Path(f'{name}.tiff'),
        timestamp=datetime.datetime(2026, 11, 11, 9, 30, 0),
        name=name,
        label='',
        scan_count=0,
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


def _older_version_record(folder: pathlib.Path) -> pathlib.Path:
    """A record holding one output, as the version before the current one wrote it."""
    record_loc = folder / ProtocolPostRecord.DEFAULT_FILENAME
    record = ProtocolPostRecord(file_loc=record_loc)
    _record_output(record, folder, 'first')
    record.complete()
    lines = record_loc.read_text(encoding='utf-8').splitlines(keepends=True)
    lines[1] = f'Version\t{ProtocolPostRecord.CURRENT_VERSION - 1}\n'
    record_loc.write_text(''.join(lines), encoding='utf-8')
    return record_loc


def test_a_record_that_recorded_nothing_leaves_no_file(tmp_path):
    record_loc = tmp_path / ProtocolPostRecord.DEFAULT_FILENAME

    ProtocolPostRecord(file_loc=record_loc).complete()

    assert not record_loc.exists()


def test_an_older_record_loaded_and_given_nothing_is_left_as_it_was(tmp_path):
    record_loc = _older_version_record(tmp_path)
    before = record_loc.read_bytes()

    ProtocolPostRecord.from_file(record_loc).complete()

    assert record_loc.read_bytes() == before


def test_an_older_record_given_an_output_is_rewritten_current_with_every_row_once(tmp_path):
    record_loc = _older_version_record(tmp_path)

    record = ProtocolPostRecord.from_file(record_loc)
    _record_output(record, tmp_path, 'second')
    record.complete()

    assert record_loc.read_text(encoding='utf-8').splitlines()[1] == (
        f'Version\t{ProtocolPostRecord.CURRENT_VERSION}'
    )
    assert [str(p) for p in ProtocolPostRecord.read_rows(record_loc)['Filepath']] == [
        'first.tiff',
        'second.tiff',
    ]
