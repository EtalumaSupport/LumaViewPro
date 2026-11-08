# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Run folders the simulator saved, for the cell count's tests.

``count_scope`` is a session and the runs it saved, each as a run of this
release saves it: ``scan``, one BF and LED 0 scan at one position, as TIFF;
``jpg``, the same as JPG; ``zstack``, a three-slice BF Z-stack; and
``plate``, two scans of two wells, 2x2 tiled, in BF and LED 0. A test counts
a copy (``copy_run``), since a count writes into the folder it reads.
"""

from __future__ import annotations

import csv
import pathlib
import shutil
from collections.abc import Callable

import pytest

from _lumascope.modules import image_utils
from _lumascope.modules.cell_count_method import CellCountMethod
from _lumascope.modules.image_mode import OutputFormat
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0

RUN_RECORD = 'protocol_record.tsv'

# The 96-well plate's well pitch: the next well along a row.
_WELL_PITCH_MM = 9.0


def _over_a_well(session, wells: int = 1) -> list[dict]:
    """Park over a plate position clear of the lower Z limit, and return it
    and the next *wells* - 1 wells along its row as a run's positions."""
    motion = session.scope.motion
    motion.move_xy(x_um=40000.0)
    motion.move_xy(y_um=30000.0)
    motion.move_z(1000.0)
    motion.wait_until_finished_moving(timeout_s=60)
    here = session.get_current_plate_position()
    return [
        {'x': here.x_mm + i * _WELL_PITCH_MM, 'y': here.y_mm, 'z': here.z_um, 'name': f'P{i}'}
        for i in range(wells)
    ]


def _saved(live: pathlib.Path, start: Callable[[], object]) -> pathlib.Path:
    """The folder of the run *start* launches, once it has completed and its files are written."""
    before = set(live.rglob(RUN_RECORD))
    pending = start()
    outcome = pending.result(timeout=120)
    assert outcome.status == 'completed', outcome
    assert pending.files.result(timeout=60) is not None, "the run's files never finished"
    (record,) = set(live.rglob(RUN_RECORD)) - before
    return record.parent


@pytest.fixture(scope='module')
def count_scope(tmp_path_factory):
    live = tmp_path_factory.mktemp('runs')
    session = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    try:
        home_sim_scope(session.scope)
        runner = session.protocol_runner()
        session.set_channel_acquire('BF', 'image')
        session.set_channel_acquire(LED0, 'image')

        def protocol(tiling: str = '1x1', wells: int = 1) -> object:
            positions = _over_a_well(session, wells)
            config = session.get_sequenced_capture_config(tiling=tiling, use_zstacking=False)
            config['positions'] = positions
            return session.scope.protocols.create_protocol(input_config=config)

        launch = {'parent_dir': str(live), 'enable_image_saving': True}
        runs = {
            'scan': _saved(
                live,
                lambda: runner.start_single_scan(
                    protocol=protocol(), sequence_name='scan', **launch
                ),
            )
        }
        session.update_settings('image_output_format.sequenced', OutputFormat.JPG)
        runs['jpg'] = _saved(
            live,
            lambda: runner.start_single_scan(protocol=protocol(), sequence_name='jpg', **launch),
        )
        session.update_settings('image_output_format.sequenced', OutputFormat.TIFF)
        # Minutes and hours, as the protocol settings hold them: two scans a second apart.
        session.update_settings('protocol.period', 1 / 60)
        session.update_settings('protocol.duration', 2 / 3600)
        runs['plate'] = _saved(
            live,
            lambda: runner.start_protocol(
                protocol=protocol('2x2', wells=2), sequence_name='plate', **launch
            ),
        )
        session.update_settings('zstack.step_size', 10.0)
        session.update_settings('zstack.range', 20.0)
        _over_a_well(session)
        runs['zstack'] = _saved(live, lambda: runner.start_zstack(channel='BF', sequence_name='zs'))
        yield session, runs
    finally:
        session.shutdown()


@pytest.fixture
def session(count_scope):
    return count_scope[0]


def copy_run(count_scope, run: str, tmp_path) -> pathlib.Path:
    """A copy of run folder *run*, for one test alone."""
    own = tmp_path / run
    shutil.copytree(count_scope[1][run], own)
    return own


def images_in(folder: pathlib.Path) -> list[str]:
    """The images directly in *folder*, by name."""
    return sorted(p.name for p in folder.iterdir() if image_utils.is_image(p.name))


def results_rows(folder: pathlib.Path) -> list[dict]:
    """The rows of the ``results.csv`` a count wrote in *folder*."""
    with open(folder / 'results.csv', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def count(session, folder: pathlib.Path):
    """Count *folder* through the session, by the default method."""
    return session.post_processing.count_cells(folder, method=CellCountMethod())
