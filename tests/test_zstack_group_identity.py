# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A z-stack group identifies the slices at ONE XY position.

Bug history
-----------
Applying tiling to an already-z-stacked protocol copied the parent step's
``Z-Stack Group ID`` onto every tile, so four distinct XY positions
claimed to be one z-stack group. The LED-hold decision reads exactly that
column -- ``same_zstack_group`` holds illumination across a step boundary
unconditionally, because within a real stack the stage barely moves -- so
the forged id held the LED lit while the stage travelled from tile to
tile, exposing sample regions that were not being imaged. No user setting
avoided it: the Advanced Settings opt-in is the OTHER arm of that ``or``.

The two construction paths disagreed about what the column meant.
``Protocol.from_config`` allocates a fresh id inside its tile loop, so a
New-Protocol build was already correct; only the Apply-Z-Stacking then
Apply-Tiling button order produced the forged id.

These tests pin the INVARIANT -- one group, one XY -- rather than either
button order.
"""

from __future__ import annotations

import pytest

from _lumascope.modules.protocol import ProtocolFormatError, StoredNames
from tests.protocol_fixtures import protocol_from_file
from _lumascope.modules.coord_transformations import AxisLimits, StageOffset
from _lumascope.modules.labware_loader import WellPlateLoader
from _lumascope.modules.objectives_loader import ObjectiveLoader
from tests.test_step_label_ssot import (
    _ZSTACK,
    _WIDE_Z,
    _build_protocol,
    _labeled_step,
)


_WIDE_XY = {
    'X': AxisLimits(min_um=-1_000_000.0, max_um=1_000_000.0),
    'Y': AxisLimits(min_um=-1_000_000.0, max_um=1_000_000.0),
}


def _tile_2x2(proto, capabilities):
    return proto.apply_tiling(
        tiling='2x2',
        frame_dimensions={'width': 1900, 'height': 1900},
        binning_size=1,
        axis_limits=_WIDE_XY,
        labware=WellPlateLoader().get_plate('6 well microplate'),
        stage_offset=StageOffset(x_um=0.0, y_um=0.0),
        overlap_percent=0.0,
        capabilities=capabilities,
        objective_helper=ObjectiveLoader(),
    )


def _groups_to_xy(steps) -> dict[int, set[tuple[float, float]]]:
    groups: dict[int, set[tuple[float, float]]] = {}
    for _, step in steps.iterrows():
        group = int(step['Z-Stack Group ID'])
        if group == -1:
            continue
        groups.setdefault(group, set()).add((step['X'], step['Y']))
    return groups


def test_tiling_a_zstack_gives_each_tile_its_own_group(scale_capabilities):
    """One group must never span more than one XY position."""
    proto = _build_protocol([_labeled_step(x=60.0, y=40.0, z=5000.0)])
    proto.apply_zstacking(zstack_params=_ZSTACK, axis_limits=_WIDE_Z)
    assert len(proto.steps()) == 6

    _tile_2x2(proto, scale_capabilities)

    steps = proto.steps()
    assert len(steps) == 24, '6 slices x 4 tiles'

    groups = _groups_to_xy(steps)
    assert len(groups) == 4, f'one group per tile, got {sorted(groups)}'
    for group, positions in groups.items():
        assert len(positions) == 1, (
            f'z-stack group {group} spans {len(positions)} XY positions: '
            f'{sorted(positions)} -- the LED-hold decision would hold '
            f'illumination across the move between them'
        )


def test_tiling_a_zstack_keeps_every_slice_of_a_tile_together(scale_capabilities):
    """The grouping must still bind the slices AT one position, each slice
    keeping its offset from its reference slice, so every tile is placed as
    the stack was built."""
    proto = _build_protocol([_labeled_step(x=60.0, y=40.0, z=5000.0)])
    proto.apply_zstacking(zstack_params=_ZSTACK, axis_limits=_WIDE_Z)
    built = sorted(proto.steps()['Z-Stack Offset'])
    _tile_2x2(proto, scale_capabilities)

    steps = proto.steps()
    counts = steps.groupby('Z-Stack Group ID').size()
    assert set(counts) == {6}, f'each tile keeps its 6 slices, got {dict(counts)}'
    offsets = {
        group_id: sorted(group['Z-Stack Offset'])
        for group_id, group in steps.groupby('Z-Stack Group ID')
    }
    assert all(tile == built for tile in offsets.values()), (built, offsets)


def test_tiling_an_ungrouped_step_allocates_no_group(scale_capabilities):
    """A step that is not part of a stack stays ungrouped through tiling."""
    proto = _build_protocol([_labeled_step(x=60.0, y=40.0)])
    _tile_2x2(proto, scale_capabilities)

    steps = proto.steps()
    assert len(steps) == 4
    assert set(steps['Z-Stack Group ID']) == {-1}


# A file saved by a build before the fix above still holds one group over
# every tile; a runnable read gives each tile its own group, so each is lit
# and focused at its own place.


def _saved(proto, path, *, rewrite=lambda row, cells: None, drop=(), version=None):
    """``proto`` saved to ``path``, each step row's cells passed through ``rewrite``.

    ``rewrite`` gets each row as a column->index map and its cell list, and
    changes the cells in place; ``drop`` removes columns, ``version``
    restamps the file, as an older build wrote it.
    """
    proto.to_file(path)
    lines = path.read_text().splitlines()
    header = lines.index('Steps') + 1
    columns = lines[header].split('\t')
    kept = [i for i, name in enumerate(columns) if name not in drop]
    at = {name: kept.index(i) for i, name in enumerate(columns) if i in kept}
    out = []
    for i, line in enumerate(lines):
        if version is not None and line.startswith('Version'):
            line = f'Version\t{version}'
        elif i >= header and line:
            cells = [cell for j, cell in enumerate(line.split('\t')) if j in kept]
            if i > header:
                rewrite(at, cells)
            line = '\t'.join(cells)
        out.append(line)
    path.write_text('\n'.join(out) + '\n')
    return path


def _one_group_over_every_tile(at, cells):
    if cells[at['Z-Stack Group ID']] != '-1':
        cells[at['Z-Stack Group ID']] = '0'


def _stacked_and_tiled(scale_capabilities):
    proto = _build_protocol([_labeled_step(x=60.0, y=40.0, z=5000.0)])
    proto.apply_zstacking(zstack_params=_ZSTACK, axis_limits=_WIDE_Z)
    _tile_2x2(proto, scale_capabilities)
    return proto


def test_a_file_holding_one_group_over_every_tile_reads_as_one_group_per_tile(
    scale_capabilities, tmp_path, caplog
):
    path = _saved(
        _stacked_and_tiled(scale_capabilities),
        tmp_path / 'forged.tsv',
        rewrite=_one_group_over_every_tile,
    )

    with caplog.at_level('WARNING'):
        steps = protocol_from_file(path).steps()

    groups = _groups_to_xy(steps)
    assert len(groups) == 4, f'one group per tile, got {sorted(groups)}'
    assert all(len(positions) == 1 for positions in groups.values()), groups
    assert set(steps.groupby('Z-Stack Group ID').size()) == {6}
    assert [r for r in caplog.records if 'forged.tsv' in r.getMessage()], 'the split is warned of'


def test_a_file_from_before_offsets_gives_each_tile_offsets_from_its_own_slices(
    scale_capabilities, tmp_path
):
    proto = _stacked_and_tiled(scale_capabilities)
    built = sorted(proto.steps()['Z-Stack Offset'][proto.steps()['Tile'] == 'A1'])
    raised_tile = 'B2'

    def forged_with_one_tile_raised(at, cells):
        _one_group_over_every_tile(at, cells)
        if cells[at['Tile']] == raised_tile:
            cells[at['Z']] = str(float(cells[at['Z']]) + 30.0)

    path = _saved(
        proto,
        tmp_path / 'before_offsets.tsv',
        rewrite=forged_with_one_tile_raised,
        drop=('Z-Stack Offset',),
        version=9,
    )

    steps = protocol_from_file(path).steps()

    for tile, slices in steps.groupby('Tile'):
        assert sorted(slices['Z-Stack Offset']) == built, tile


def test_a_stack_at_one_tile_reads_as_one_group_whatever_its_slices_x_and_y(tmp_path):
    proto = _build_protocol([_labeled_step(x=60.0, y=40.0, z=5000.0)])
    proto.apply_zstacking(zstack_params=_ZSTACK, axis_limits=_WIDE_Z)
    group = int(proto.steps()['Z-Stack Group ID'].iloc[0])

    def one_slice_nudged(at, cells):
        if cells[at['Z-Slice']] == '0':
            cells[at['X']] = '60.5'

    def no_stage(at, cells):
        cells[at['X']] = ''
        cells[at['Y']] = ''

    for name, rewrite in (('nudged', one_slice_nudged), ('no_stage', no_stage)):
        path = _saved(proto, tmp_path / f'{name}.tsv', rewrite=rewrite)
        assert set(protocol_from_file(path).steps()['Z-Stack Group ID']) == {group}, name


def test_a_blank_group_cell_in_a_file_holding_one_group_over_every_tile_is_refused(
    scale_capabilities, tmp_path
):
    def forged_with_one_blank(at, cells):
        _one_group_over_every_tile(at, cells)
        if cells[at['Tile']] == 'A2' and cells[at['Z-Slice']] == '0':
            cells[at['Z-Stack Group ID']] = ''

    path = _saved(
        _stacked_and_tiled(scale_capabilities),
        tmp_path / 'blank.tsv',
        rewrite=forged_with_one_blank,
    )

    with pytest.raises(ProtocolFormatError) as refused:
        protocol_from_file(path)

    assert 'Z-Stack Group ID' in str(refused.value)


def test_a_read_that_never_runs_keeps_the_groups_the_file_holds(scale_capabilities, tmp_path):
    path = _saved(
        _stacked_and_tiled(scale_capabilities),
        tmp_path / 'forged.tsv',
        rewrite=_one_group_over_every_tile,
    )

    steps = protocol_from_file(path, unit=StoredNames.KEPT).steps()

    assert set(steps['Z-Stack Group ID']) == {0}
