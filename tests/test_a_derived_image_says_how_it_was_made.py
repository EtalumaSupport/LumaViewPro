# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A derived image says how it was made, and no build mixes two ways of making it.

A folder can hold two derivations of the same images: two projections of
one stack (Max and Sum), two stitches of one scan (Quality and Fast
Preview), two composites made with different settings. Each is its own
output, named for how it was made, and a later build takes each on its own:
a stitch of a folder holding two projections makes a mosaic of each, never
one mosaic built from both. A rebuild with new settings makes the output
those settings describe; one with the same settings makes nothing new.

Each test runs the real builds through ``session.post_processing`` on a
run folder written as a run of this release saves it: a 2x2 tiled scan at
15% overlap, a 3-slice Z-stack at each tile, BF and the stock LED 0.
"""

from __future__ import annotations

import datetime
import shutil

import numpy as np
import pytest
import tifffile

from _lumascope.modules import image_utils
from _lumascope.modules.exceptions import (
    PostProcessingFailedError,
    PostProcessingOperation,
    PostProcessingRefusedError,
)
from _lumascope.modules.protocol_post_record import ProtocolPostRecord
from _lumascope.modules.channel_record import release_color, stored_key
from tests.stock_channels import LED0

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
_SLICES = (0, 1, 2)


def _channels(fluorescence: str) -> dict[str, tuple[int, str, str]]:
    """Each channel of the run: its channel group, false colour and step prefix."""
    return {'BF': (0, 'False', 'custom0000'), fluorescence: (1, 'True', 'custom0001')}


def _step(channels: dict, tile: str, color: str, z_slice: int) -> list[str]:
    column, row = _TILES[tile]
    group, false_color, prefix = channels[color]
    return [
        f'{prefix}_{color}_T{tile}_Z{z_slice}', f'{10 + column * _STEP_MM:.4f}',
        f'{10 + row * _STEP_MM:.4f}', f'{5000.0 + 5 * z_slice:.1f}', 'False', color,
        false_color, '5.0', '1.0', 'False', '2.0', '1', '20x Oly', '', tile, str(z_slice),
        'True', str(group), '0', 'image', '"{""duration"": 5, ""fps"": 30}"', '"{}"', 'True',
        prefix,
    ]  # fmt: skip


@pytest.fixture
def session(tmp_path):
    from _lumascope.modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live')), simulate=True
    )
    yield session
    session.shutdown()


@pytest.fixture
def run_folder(tmp_path):
    """A 2x2 tiled, 3-slice BF + LED 0 run, each slice a different scene."""
    return write_run_folder(tmp_path, LED0)


def write_run_folder(
    tmp_path,
    fluorescence: str,
    *,
    bits: int = 8,
    fluorescence_encoding: str = 'right_aligned',
    ome: bool = False,
    plated: bool = False,
):
    """The run folder of a 2x2 tiled, 3-slice run of BF and ``fluorescence``,
    each slice a different scene, as a release naming it so saves it.

    ``bits`` is the capture depth (8, or 12 in a 16-bit container), and
    ``fluorescence_encoding`` how the fluorescence frames were saved: 'rgb'
    is a false-colour capture, three channels at 12 bits. ``ome`` saves
    every frame OME-TIFF, as an OME-TIFF run does, else plain TIFF.
    ``plated`` records each frame's 96-well plate and its well, A1, as a
    well-plate run does; else no plate and no well.
    """
    plate = {'plate': {'name': '96 well microplate', 'rows': 8, 'columns': 12, 'well_label': 'A1'}}
    channels = _channels(fluorescence)
    folder = tmp_path / 'live' / 'ProtocolData' / '20261018_120000'
    folder.mkdir(parents=True)
    steps = [
        _step(channels, tile, color, z_slice)
        for tile in _TILES
        for color in channels
        for z_slice in _SLICES
    ]
    (folder / 'run.tsv').write_text(
        'LumaViewPro Protocol\nVersion\t8\nPeriod\t20.0\nDuration\t48.0\n'
        'Labware\t96 well microplate\nCapture Root\t\n\nSteps\n'
        + '\t'.join(_COLUMNS)
        + '\n'
        + ''.join('\t'.join(step) + '\n' for step in steps)
    )
    rng = np.random.default_rng(18)
    peak, dtype = (80, np.uint8) if bits == 8 else (3000, np.uint16)
    scenes = {z: (rng.random((2 * _TILE_PX, 2 * _TILE_PX)) * peak).astype(dtype) for z in _SLICES}
    record_rows = []
    for index, step in enumerate(steps):
        name, color, tile, z_slice = step[0], step[5], step[14], int(step[15])
        column, row = _TILES[tile]
        y0, x0 = row * (_TILE_PX - 10), column * (_TILE_PX - 10)
        filename = f'{name}_0000{".ome.tiff" if ome else ".tiff"}'
        image_utils.write_tiff(
            data=scenes[z_slice][y0 : y0 + _TILE_PX, x0 : x0 + _TILE_PX].copy(),
            file_loc=folder / filename,
            metadata={
                'pixel_size_um': _PIXEL_SIZE_UM,
                'channel': color,
                'objective': {'id': '20x Oly'},
                'exposure_time_ms': 2.0,
                'gain_db': 1.0,
                'illumination_ma': 5.0,
                'z_pos_um': float(step[3]),
                'plate_pos_mm': {'x': float(step[1]), 'y': float(step[2])},
                'datetime': datetime.datetime(2026, 10, 18, 12, 0, 0),
                'significant_bits': bits,
                **(plate if plated else {}),
            },
            ome=ome,
            color=release_color(stored_key(color)),
            significant_bits=bits,
            save_encoding='right_aligned' if color == 'BF' else fluorescence_encoding,
            white_bits=None,
        )
        record_rows.append(
            f'{filename}\t{name}\t{index}\t0\t2026-10-18 12:00:{index:02d}.000001\t1\t0.0\n'
        )
    (folder / 'protocol_record.tsv').write_text(
        'LumaViewPro Protocol Execution Record\nVersion\t3\nProtocol File\trun.tsv\n'
        'Filename\tStep Name\tStep Index\tScan Count\tTimestamp\tFrame Count\tDuration (s)\n'
        + ''.join(record_rows)
    )
    return folder


def _names(paths, folder: str) -> list[str]:
    return sorted(path.name for path in paths if path.parent.name == folder)


def test_a_stitch_of_two_projections_makes_a_mosaic_of_each(session, run_folder, tmp_path):
    """Max and Sum projections of one stack, stitched, are two mosaics a
    channel, each named for its projection and equal to stitching that
    projection alone -- never one mosaic built from both."""
    alone = {
        method: shutil.copytree(run_folder, tmp_path / f'alone_{method}')
        for method in ('Max', 'Sum')
    }
    session.post_processing.zproject(run_folder, method='Max')
    session.post_processing.zproject(run_folder, method='Sum')
    for method, folder in alone.items():
        session.post_processing.zproject(folder, method=method)
        session.post_processing.stitch(folder)

    stitched = session.post_processing.stitch(run_folder)

    mosaics = _names(stitched.artifact_paths, 'Stitched-ZProject')
    assert mosaics == [
        'custom0000_BF_0000_zproj_max_stitched.tiff',
        'custom0000_BF_0000_zproj_sum_stitched.tiff',
        f'custom0001_{LED0}_0000_zproj_max_stitched.tiff',
        f'custom0001_{LED0}_0000_zproj_sum_stitched.tiff',
    ]
    for method, folder in alone.items():
        name = f'custom0000_BF_0000_zproj_{method.lower()}_stitched.tiff'
        mosaic = tifffile.imread(run_folder / 'Stitched-ZProject' / name)
        mosaic_alone = tifffile.imread(folder / 'Stitched-ZProject' / name)
        assert mosaic.dtype == mosaic_alone.dtype
        assert np.array_equal(mosaic, mosaic_alone)


def test_a_projection_takes_the_slices_and_names_the_stitched_mosaics_with_the_remedy(
    session, run_folder
):
    """Z-Projection takes the raw slices, as the build chain does: Quality
    and Fast Preview mosaics of the slices are left out and named, with the
    route to a stitched projection, and each tile's projection is the sum of
    its own slices."""
    quality = session.post_processing.stitch(run_folder, mode='quality')
    fast = session.post_processing.stitch(run_folder, mode='fast_preview')
    mosaics = len(quality.artifact_paths) + len(fast.artifact_paths)

    projected = session.post_processing.zproject(run_folder, method='Sum')

    assert _names(projected.artifact_paths, 'Stitched-ZProject') == []
    assert len(_names(projected.artifact_paths, 'ZProject')) == len(_TILES) * 2
    assert [(left.kind, left.count) for left in projected.excluded_inputs] == [
        (PostProcessingOperation.STITCH, mosaics)
    ]
    assert 'Stitch then stitches the projections' in projected.message
    slices = [
        tifffile.imread(run_folder / f'custom0000_BF_TA1_Z{z}_0000.tiff').astype(np.int64)
        for z in _SLICES
    ]
    projection = tifffile.imread(
        run_folder / 'ZProject' / 'custom0000_BF_TA1_0000_zproj_sum.tiff'
    ).astype(np.int64)
    assert np.array_equal(projection, np.sum(slices, axis=0))


def test_a_folder_of_mosaics_alone_is_refused_for_what_it_holds(session, run_folder):
    """A folder holding only stitched mosaics of the slices holds nothing a
    projection takes: it is refused, naming the mosaics and the remedy."""
    session.post_processing.stitch(run_folder)

    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.zproject(run_folder / 'Stitched', method='Sum')

    assert refused.value.reason == 'excluded_inputs'
    assert 'Stitch then stitches the projections' in str(refused.value)


def test_a_stitched_projection_is_a_projection_already_made(session, run_folder):
    """Projected, stitched, then projected again: the stitched projections
    are the projection's own, already made; only the mosaics of the slices
    are left out."""
    session.post_processing.zproject(run_folder, method='Sum')
    stitched = session.post_processing.stitch(run_folder)
    slice_mosaics = _names(stitched.artifact_paths, 'Stitched')
    assert slice_mosaics and _names(stitched.artifact_paths, 'Stitched-ZProject')

    again = session.post_processing.zproject(run_folder, method='Sum')

    assert again.new_count == 0
    assert [(left.kind, left.count) for left in again.excluded_inputs] == [
        (PostProcessingOperation.STITCH, len(slice_mosaics))
    ]


def test_a_composite_rebuilt_with_a_new_threshold_is_built_and_one_with_the_same_is_not(
    session, run_folder
):
    """A rebuild with a new blend threshold makes the composites it
    describes, beside the earlier ones; a rebuild with the same settings
    makes nothing new."""
    first = session.post_processing.composite(run_folder)
    session.update_settings(f'{LED0}.composite_brightness_threshold', 80)

    rebuilt = session.post_processing.composite(run_folder)
    again = session.post_processing.composite(run_folder)

    assert rebuilt.new_count == first.new_count == 12
    assert not set(rebuilt.artifact_paths) & set(first.artifact_paths)
    assert again.new_count == 0


def _as_written_by_release_3(folder) -> None:
    """Rewrite ``folder``'s post-processing record as a release before
    derivations wrote it: version 3, each row's five kind columns in place
    of its derivation."""
    record = folder / 'protocol_post_record.tsv'
    head, _, table = record.read_text().partition('Images\n')
    lines = table.splitlines()
    columns = lines[0].split('\t')
    derivation_at = columns.index('Derivation')
    kinds = ['Composite', 'Stitched', 'ZProject', 'Video', 'Hyperstack']
    rows = []
    for line in lines[1:]:
        cells = line.split('\t')
        made = cells.pop(derivation_at)
        rows.append(cells + [str(f'{kind}(' in made) for kind in kinds])
    header = columns[:derivation_at] + columns[derivation_at + 1 :] + kinds
    record.write_text(
        head.replace(f'Version\t{ProtocolPostRecord.CURRENT_VERSION}', 'Version\t3')
        + 'Images\n'
        + '\n'.join('\t'.join(row) for row in [header, *rows])
        + '\n'
    )


def test_a_projection_an_earlier_release_made_is_stitched_and_named_for_no_method(
    session, run_folder
):
    """A folder an earlier release projected still stitches: its record
    names no method, so the mosaic carries the projection's bare token, and
    the record is upgraded to say how the mosaic was made."""
    session.post_processing.zproject(run_folder, method='Max')
    _as_written_by_release_3(run_folder)

    stitched = session.post_processing.stitch(run_folder)

    assert _names(stitched.artifact_paths, 'Stitched-ZProject') == [
        'custom0000_BF_0000_zproj_stitched.tiff',
        f'custom0001_{LED0}_0000_zproj_stitched.tiff',
    ]
    record = (run_folder / 'protocol_post_record.tsv').read_text()
    assert 'Version\t5' in record
    assert 'ZProject(?)>Stitched(mode=quality)' in record


def test_a_folder_saved_before_times_stated_their_offset_builds_and_keeps_its_times(
    session, run_folder
):
    """A run saved before every time stated its offset (a version 3 run
    record), whose outputs a version 4 post record lists, still builds a
    mosaic; the post record is upgraded in place, and every row's
    time reads back as it was written, with no offset given to it."""
    session.post_processing.zproject(run_folder, method='Max')
    record = run_folder / 'protocol_post_record.tsv'
    record.write_text(
        record.read_text().replace(f'Version\t{ProtocolPostRecord.CURRENT_VERSION}', 'Version\t4')
    )

    stitched = session.post_processing.stitch(run_folder)

    assert stitched.new_count > 0
    assert f'Version\t{ProtocolPostRecord.CURRENT_VERSION}' in record.read_text()
    times = list(ProtocolPostRecord.from_file(record).records()['Timestamp'])
    assert times
    assert all(time.utcoffset() is None for time in times)


def test_two_projections_an_earlier_release_made_are_not_stitched_into_one(session, run_folder):
    """Two projections an earlier release made, with no record of which
    method made which, fill each tile position twice: the stitch builds
    neither mosaic, names the files and says to build them again, and still
    stitches the slices."""
    session.post_processing.zproject(run_folder, method='Max')
    session.post_processing.zproject(run_folder, method='Sum')
    _as_written_by_release_3(run_folder)

    with pytest.raises(PostProcessingFailedError) as failed:
        session.post_processing.stitch(run_folder)

    message = str(failed.value)
    assert 'more than one image for one tile position' in message
    assert 'custom0000_BF_TA1_0000_zproj_max.tiff' in message
    assert 'custom0000_BF_TA1_0000_zproj_sum.tiff' in message
    assert 'build them again with this release' in message
    assert sorted(path.split('/')[-2] for path in failed.value.produced_paths) == ['Stitched'] * 6


def test_a_slice_the_run_recorded_twice_is_not_projected_into_one(session, run_folder):
    """Two raw images the run recorded for one slice fill it twice: that
    tile's projection is not built, and names the files; the other tiles'
    are."""
    twice = 'custom0000_BF_TA1_Z0_0000_again.tiff'
    shutil.copy(run_folder / 'custom0000_BF_TA1_Z0_0000.tiff', run_folder / twice)
    record = run_folder / 'protocol_record.tsv'
    first = next(
        line for line in record.read_text().splitlines() if line.startswith('custom0000_BF_TA1_Z0')
    )
    record.write_text(record.read_text() + twice + first[first.index('\t') :] + '\n')

    with pytest.raises(PostProcessingFailedError) as failed:
        session.post_processing.zproject(run_folder, method='Sum')

    message = str(failed.value)
    assert 'more than one image for one slice' in message
    assert twice in message
    assert len(failed.value.produced_paths) == len(_TILES) * 2 - 1


def test_steps_inserted_without_a_tile_name_are_stitched_by_their_position(session, run_folder):
    """Inserted steps carry no tile name and no tile group; the stitch takes
    each by its stage position, so four of them at four positions are one
    mosaic a channel and slice."""
    run = run_folder / 'run.tsv'
    head, _, steps = run.read_text().partition('Steps\n')
    lines = steps.splitlines()
    columns = lines[0].split('\t')
    tile, group = columns.index('Tile'), columns.index('Tile Group ID')
    rows = []
    for line in lines[1:]:
        cells = line.split('\t')
        cells[tile], cells[group] = '', '-1'
        rows.append('\t'.join(cells))
    run.write_text(head + 'Steps\n' + '\n'.join([lines[0], *rows]) + '\n')

    stitched = session.post_processing.stitch(run_folder)

    assert stitched.new_count == 6
