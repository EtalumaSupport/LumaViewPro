# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A post-processing build names the files it left out, and how to get their output.

A stitch takes no composites: a stitched composite is built by Composite
from the stitched channels, run after the stitch. A stitch over a folder
that already holds per-tile composites left them out in silence, because
the folder's rows were numbered twice over (raw, then post-processed, each
from 0) and the count of what the filter dropped looked rows up by number:
a composite numbered like a surviving tile was never counted. A build's
own earlier outputs are what it has already made, not left out.

Each test runs the real builds through ``session.post_processing`` on a run
folder written as a run of this release saves it.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from modules import image_utils
from modules.exceptions import (
    ExcludedInput,
    PostProcessingFailedError,
    PostProcessingOperation,
    PostProcessingRefusedError,
)
from modules.layer_record import release_color

_COLUMNS = [
    'Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Color', 'False_Color', 'Illumination',
    'Gain', 'Auto_Gain', 'Exposure', 'Sum', 'Objective', 'Well', 'Tile', 'Z-Slice',
    'Custom Step', 'Tile Group ID', 'Z-Stack Group ID', 'Acquire', 'Video Config',
    'Stim_Config', 'Auto_Named', 'Label',
]  # fmt: skip

_PIXEL_SIZE_UM = 0.5
_TILE_PX = 64
# 15% overlap: tiles one step apart share 10 columns (or rows).
_STEP_MM = (_TILE_PX - 10) * _PIXEL_SIZE_UM / 1000
_TILES = {'A1': (0, 0), 'A2': (1, 0), 'B1': (0, 1), 'B2': (1, 1)}
_CHANNELS = {'BF': (0, 'False', 'custom0000'), 'Blue': (1, 'True', 'custom0001')}


def _step(tile: str, color: str) -> list[str]:
    column, row = _TILES[tile]
    group, false_color, prefix = _CHANNELS[color]
    return [
        f'{prefix}_{color}_T{tile}', f'{10 + column * _STEP_MM:.4f}', f'{10 + row * _STEP_MM:.4f}',
        '5000.0', 'False', color, false_color, '5.0', '1.0', 'False', '2.0', '1', '20x Oly',
        '', tile, '-1', 'True', str(group), '-1', 'image',
        '"{""duration"": 5, ""fps"": 30}"', '"{}"', 'True', '',
    ]  # fmt: skip


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live')), simulate=True
    )
    yield session
    session.shutdown()


@pytest.fixture
def run_folder(tmp_path):
    """A 2x2 tiled BF + Blue run at 15% overlap, as a run of this release saves it."""
    folder = tmp_path / 'live' / 'ProtocolData' / '20261012_120000'
    folder.mkdir(parents=True)
    steps = [_step(tile, color) for tile in _TILES for color in _CHANNELS]
    (folder / 'run.tsv').write_text(
        'LumaViewPro Protocol\nVersion\t8\nPeriod\t20.0\nDuration\t48.0\n'
        'Labware\t96 well microplate\nCapture Root\t\n\nSteps\n'
        + '\t'.join(_COLUMNS)
        + '\n'
        + ''.join('\t'.join(step) + '\n' for step in steps)
    )
    rng = np.random.default_rng(12)
    scene = (rng.random((2 * _TILE_PX, 2 * _TILE_PX)) * 200).astype(np.uint8)
    record_rows = []
    for index, step in enumerate(steps):
        name, color, tile = step[0], step[5], step[14]
        column, row = _TILES[tile]
        y0, x0 = row * (_TILE_PX - 10), column * (_TILE_PX - 10)
        filename = f'{name}_0000.tiff'
        image_utils.write_tiff(
            data=scene[y0 : y0 + _TILE_PX, x0 : x0 + _TILE_PX].copy(),
            file_loc=folder / filename,
            metadata={
                'pixel_size_um': _PIXEL_SIZE_UM,
                'channel': color,
                'objective': '20x Oly',
                'exposure_time_ms': 2.0,
                'gain_db': 1.0,
                'illumination_ma': 5.0,
                'z_pos_um': 5000.0,
                'plate_pos_mm': {'x': float(step[1]), 'y': float(step[2])},
                'datetime': '2026:10:12 12:00:00',
                'camera_make': 'Test',
                'microscope': 'TestScope',
                'well_label': '',
                'significant_bits': 8,
            },
            ome=False,
            color=release_color(color),
            significant_bits=8,
            save_encoding='right_aligned',
            white_bits=None,
        )
        record_rows.append(
            f'{filename}\t{name}\t{index}\t0\t2026-10-12 12:00:{index:02d}.000001\t1\t0.0\n'
        )
    (folder / 'protocol_record.tsv').write_text(
        'LumaViewPro Protocol Execution Record\nVersion\t3\nProtocol File\trun.tsv\n'
        'Filename\tStep Name\tStep Index\tScan Count\tTimestamp\tFrame Count\tDuration (s)\n'
        + ''.join(record_rows)
    )
    return folder


def test_a_stitch_names_the_composites_it_left_out_and_the_route_to_them(session, run_folder):
    """Per-tile composites in the folder are left out of the stitch, named,
    with the route: Composite builds them from the stitched channels."""
    composited = session.post_processing.composite(run_folder)
    assert composited.new_count == 4

    stitched = session.post_processing.stitch(run_folder)

    assert stitched.new_count == 2
    assert stitched.excluded_inputs == (
        ExcludedInput(kind=PostProcessingOperation.COMPOSITE, count=4),
    )
    assert '4 composite file(s)' in stitched.message
    assert 'Composite builds them from the stitched channels' in stitched.message


def test_composite_after_stitch_builds_the_stitched_composite_and_leaves_nothing_out(
    session, run_folder
):
    """The stitched composite comes from the stitched channels; the per-tile
    composites are Composite's own earlier outputs, not left out."""
    session.post_processing.composite(run_folder)
    session.post_processing.stitch(run_folder)

    again = session.post_processing.composite(run_folder)

    assert again.new_count == 1
    assert [path.parent.name for path in again.artifact_paths] == ['Composite-Stitched']
    assert again.excluded_inputs == ()
    assert 'Excluded' not in again.message


def test_a_restitch_leaves_out_only_what_is_not_its_own(session, run_folder):
    """Stitched files, the stitched composite among them, are the stitch's
    own; the per-tile composites are still left out."""
    session.post_processing.composite(run_folder)
    session.post_processing.stitch(run_folder)
    session.post_processing.composite(run_folder)

    restitched = session.post_processing.stitch(run_folder)

    assert restitched.excluded_inputs == (
        ExcludedInput(kind=PostProcessingOperation.COMPOSITE, count=4),
    )


def test_a_folder_of_composites_alone_is_refused_with_the_route(session, run_folder):
    """Stitching the Composite subfolder makes nothing; the refusal says
    why and how a stitched composite is made."""
    session.post_processing.composite(run_folder)

    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.stitch(run_folder / 'Composite')

    assert refused.value.reason == 'excluded_inputs'
    assert '4 composite file(s)' in str(refused.value)
    assert 'Composite builds them from the stitched channels' in str(refused.value)


def test_a_build_with_a_failed_group_still_names_what_it_left_out(session, run_folder, monkeypatch):
    """A group that fails does not silence the files never tried."""
    from modules.protocol_post_processing_result import PostProcResult
    from modules.stitcher import Stitcher

    session.post_processing.composite(run_folder)
    # a stand-in by design: the simulator cannot yet make a stitch group fail.
    monkeypatch.setattr(
        Stitcher, '_group_algorithm', lambda self, **kwargs: PostProcResult.failed('no tiles')
    )

    with pytest.raises(PostProcessingFailedError) as failed:
        session.post_processing.stitch(run_folder)

    assert failed.value.excluded_inputs == (
        ExcludedInput(kind=PostProcessingOperation.COMPOSITE, count=4),
    )
    assert '4 composite file(s)' in str(failed.value)


@pytest.fixture(scope='module')
def _recorded_runs(tmp_path_factory):
    """Two real simulated scans with a recording saved as frames: one with BF
    and Green stills beside it, one of the recording alone. Run once; each
    test builds in its own copy."""
    from tests.test_a_late_write_records_its_frame import _protocol
    from tests.test_a_late_write_records_its_frame import _step as _run_step
    from tests.test_composite_run_e2e import headless_settings, open_composite_session

    root = tmp_path_factory.mktemp('recorded_runs')
    settings = headless_settings(root)
    settings['video_as_frames'] = True
    shapes = {
        'stills_and_recording': (('BF', 'image'), ('Green', 'image'), ('Blue', 'video')),
        'recording_alone': (('Blue', 'video'),),
    }
    folders = {}
    with open_composite_session(settings) as (_session, runner):
        for name, shape in shapes.items():
            steps = []
            for index, (color, acquire) in enumerate(shape):
                step = _run_step('P1', index, x=20.0, gain=1.0)
                step['Color'] = color
                step['Acquire'] = acquire
                steps.append(step)
            pending = runner.start_single_scan(
                protocol=_protocol(steps), parent_dir=str(root / name)
            )
            assert pending.result(timeout=60).status == 'completed'
            assert pending.files.result(timeout=60) is not None, "the run's files never finished"
            (record,) = (root / name).rglob('protocol_record.tsv')
            folders[name] = record.parent
    return folders


@pytest.fixture
def recorded(_recorded_runs, tmp_path):
    """A copy of each recorded run, for one test to build in."""
    import shutil

    return {
        name: pathlib.Path(shutil.copytree(folder, tmp_path / name))
        for name, folder in _recorded_runs.items()
    }


def test_a_recordings_frames_are_not_left_out_of_a_composite(session, recorded):
    """Only Video takes a recording's frames, and nobody looks for them in a
    composite: the composite beside them names nothing left out."""
    result = session.post_processing.composite(recorded['stills_and_recording'])

    assert result.message == 'Success.'
    assert result.excluded_inputs == ()


def test_a_folder_holding_only_a_recording_says_video_builds_it(session, recorded):
    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.composite(recorded['recording_alone'])

    assert refused.value.reason == 'no_data'
    assert 'this folder holds a recording, not stills; Video builds it' in str(refused.value)


def test_stills_beside_a_recording_are_not_called_a_recording_or_derived(session, recorded):
    """Two single stills and a recording: nothing to project, and the answer
    says so of the stills, not of the recording or of derived outputs."""
    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.zproject(recorded['stills_and_recording'])

    assert refused.value.reason == 'no_data'
    assert 'holds a recording' not in str(refused.value)
    assert 'derived outputs' not in str(refused.value)
    assert '2 single-image group(s) skipped' in str(refused.value)


def test_a_deleted_recording_is_not_offered_to_video(session, recorded):
    """A recording deleted since its run is one Video cannot build: the
    answer does not send the person there."""
    import shutil

    folder = recorded['recording_alone']
    (frames,) = (path for path in folder.iterdir() if path.is_dir())
    shutil.rmtree(frames)

    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.composite(folder)

    assert 'Video builds it' not in str(refused.value)
    assert 'derived outputs' not in str(refused.value)
