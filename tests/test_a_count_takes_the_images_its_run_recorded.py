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

import csv
import datetime
import pathlib
import shutil
from collections.abc import Callable

import numpy as np
import pytest

from _lumascope.modules import image_utils
from _lumascope.modules.cell_count_method import CellCountMethod
from _lumascope.modules.channel_record import release_color
from _lumascope.modules.lumascope_api.frame_record import TimestampSource
from _lumascope.modules.exceptions import LeftOutReason, PostProcessingRefusedError
from _lumascope.modules.image_mode import OutputFormat
from _lumascope.modules.protocol_post_record import ProtocolPostRecord
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0

_RECORD = 'protocol_record.tsv'


def _over_a_well(session) -> list[dict]:
    """Park over a plate position clear of the lower Z limit, and return it as a run's one position."""
    motion = session.scope.motion
    motion.move_xy(x_um=40000.0)
    motion.move_xy(y_um=30000.0)
    motion.move_z(1000.0)
    motion.wait_until_finished_moving(timeout_s=60)
    here = session.get_current_plate_position()
    return [{'x': here.x_mm, 'y': here.y_mm, 'z': here.z_um, 'name': 'A1'}]


def _saved(live: pathlib.Path, start: Callable[[], object]) -> pathlib.Path:
    """The folder of the run *start* launches, once it has completed and its files are written."""
    before = set(live.rglob(_RECORD))
    pending = start()
    outcome = pending.result(timeout=120)
    assert outcome.status == 'completed', outcome
    assert pending.files.result(timeout=60) is not None, "the run's files never finished"
    (record,) = set(live.rglob(_RECORD)) - before
    return record.parent


@pytest.fixture(scope='module')
def scope(tmp_path_factory):
    """A simulated session and the runs it saved: one BF and LED 0 scan as
    TIFF, the same as JPG, and a three-slice BF Z-stack."""
    live = tmp_path_factory.mktemp('runs')
    session = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    try:
        home_sim_scope(session.scope)
        runner = session.protocol_runner()
        session.set_channel_acquire('BF', 'image')
        session.set_channel_acquire(LED0, 'image')

        def scan() -> object:
            positions = _over_a_well(session)
            config = session.get_sequenced_capture_config(tiling='1x1', use_zstacking=False)
            config['positions'] = positions
            return session.scope.protocols.create_protocol(input_config=config)

        launch = {'parent_dir': str(live), 'enable_image_saving': True}
        runs = {
            'scan': _saved(
                live,
                lambda: runner.start_single_scan(protocol=scan(), sequence_name='scan', **launch),
            )
        }
        session.update_settings('image_output_format.sequenced', OutputFormat.JPG)
        runs['jpg'] = _saved(
            live,
            lambda: runner.start_single_scan(protocol=scan(), sequence_name='jpg', **launch),
        )
        session.update_settings('image_output_format.sequenced', OutputFormat.TIFF)
        session.update_settings('zstack.step_size', 10.0)
        session.update_settings('zstack.range', 20.0)
        _over_a_well(session)
        runs['zstack'] = _saved(live, lambda: runner.start_zstack(channel='BF', sequence_name='zs'))
        yield session, runs
    finally:
        session.shutdown()


@pytest.fixture
def session(scope):
    return scope[0]


def _copy(scope, run: str, tmp_path) -> pathlib.Path:
    """A copy of run folder *run*, for one test alone: a count writes into the folder it reads."""
    own = tmp_path / run
    shutil.copytree(scope[1][run], own)
    return own


def _captures(folder: pathlib.Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if image_utils.is_image(p.name))


def _rows(folder: pathlib.Path) -> list[dict]:
    with open(folder / 'results.csv', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _count(session, folder: pathlib.Path):
    return session.post_processing.count_cells(folder, method=CellCountMethod())


def test_a_run_folder_counts_its_captures_and_names_what_it_left_out(scope, session, tmp_path):
    folder = _copy(scope, 'scan', tmp_path)
    captures = _captures(folder)
    (copy,) = session.post_processing.enhance(folder / captures[0]).created
    shutil.copy(folder / captures[0], folder / 'stray.tiff')

    result = _count(session, folder)

    assert sorted(row['file'] for row in _rows(folder)) == captures
    assert {(str(left.path), left.reason) for left in result.left_out} == {
        (copy.output_path.name, LeftOutReason.NOT_IN_RECORD),
        ('stray.tiff', LeftOutReason.NOT_IN_RECORD),
    }
    assert copy.output_path.name in result.message and 'stray.tiff' in result.message


def test_a_folder_with_no_run_record_counts_all_but_a_quick_enhance_copy(scope, session, tmp_path):
    still = scope[1]['scan'] / _captures(scope[1]['scan'])[0]
    folder = tmp_path / 'stills'
    folder.mkdir()
    shutil.copy(still, folder / still.name)
    (copy,) = session.post_processing.enhance(folder / still.name).created

    result = _count(session, folder)

    assert [row['file'] for row in _rows(folder)] == [still.name]
    assert [(str(left.path), left.reason) for left in result.left_out] == [
        (copy.output_path.name, LeftOutReason.QUICK_ENHANCE_COPY)
    ]


def test_a_jpg_run_folder_is_counted(scope, session, tmp_path):
    folder = _copy(scope, 'jpg', tmp_path)
    captures = _captures(folder)
    assert captures and all(name.lower().endswith('.jpg') for name in captures)

    result = _count(session, folder)

    assert result.counted == len(captures) and result.left_out == ()


def test_a_run_folder_whose_record_cannot_be_read_is_refused(scope, session, tmp_path):
    folder = _copy(scope, 'scan', tmp_path)
    (folder / _RECORD).write_text('not a run record\n', encoding='utf-8')

    with pytest.raises(PostProcessingRefusedError) as refused:
        _count(session, folder)

    assert refused.value.reason == 'protocol_data_unreadable'
    assert not (folder / 'results.csv').exists()


def test_a_run_folder_whose_protocol_is_gone_is_refused_saying_so(scope, session, tmp_path):
    folder = _copy(scope, 'scan', tmp_path)
    (protocol,) = [p for p in folder.glob('*.tsv') if p.name != _RECORD]
    protocol.unlink()

    with pytest.raises(PostProcessingRefusedError) as refused:
        _count(session, folder)

    assert refused.value.reason == 'protocol_data_unreadable'
    assert 'is not in the run folder' in str(refused.value)


def test_a_run_folder_whose_post_record_cannot_be_read_is_refused_and_keeps_it(
    scope, session, tmp_path
):
    folder = _copy(scope, 'scan', tmp_path)
    post_record = folder / ProtocolPostRecord.DEFAULT_FILENAME
    post_record.write_text('not a post-processing record\n', encoding='utf-8')

    with pytest.raises(PostProcessingRefusedError) as refused:
        _count(session, folder)

    assert refused.value.reason == 'protocol_data_unreadable'
    assert post_record.read_text(encoding='utf-8') == 'not a post-processing record\n'


def test_a_count_of_a_run_folder_writes_only_its_results(scope, session, tmp_path):
    """The count reads the run's records and writes none: no post record is
    started and nothing is moved aside."""
    folder = _copy(scope, 'scan', tmp_path)
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}

    _count(session, folder)

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

    _count(session, tmp_path)

    assert [row['file'] for row in _rows(tmp_path)] == ['b.tiff', 'c.tiff', 'a.tiff']


def test_a_projections_time_is_its_scans_from_the_run_record(scope, session, tmp_path):
    """A projection states no time of its own; its row's time is the one
    its record row holds, the time Create Video stamps it with."""
    folder = _copy(scope, 'zstack', tmp_path)
    (projection,) = session.post_processing.zproject(folder).artifact_paths
    recorded = ProtocolPostRecord.read_rows(folder / ProtocolPostRecord.DEFAULT_FILENAME)
    (row,) = [row for row in recorded.to_dict('records') if folder / row['Filepath'] == projection]
    assert image_utils.stated_frame_time(image_utils.read_file_record(projection)) is None

    _count(session, projection.parent)

    (counted,) = [r for r in _rows(projection.parent) if r['file'] == projection.name]
    assert datetime.datetime.fromisoformat(counted['time']) == row['Timestamp']
